//! Cranelift JIT compiler for elisp bytecode.
//!
//! Tier 1: Direct opcode translation. Each bytecode opcode is lowered to a
//! call into a Rust "helper" function that performs the same work as the
//! interpreter loop body. The JIT function signature is:
//!
//!   fn(env: *mut u8, cx: *mut u8, consts: *const u8) -> u64
//!
//! where the return value is a tagged Object pointer (or 0 for error).
//! `env` and `cx` are opaque pointers passed through to helpers.

use cranelift_codegen::ir::{types, AbiParam, InstBuilder};
use cranelift_codegen::settings::{self, Configurable};
use cranelift_codegen::Context as CraneliftContext;
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Variable};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module};
use std::collections::HashMap;
use std::sync::Mutex;

use crate::bytecode::opcode::OpCode;

/// A JIT-compiled function pointer. Same signature as the Cranelift output.
/// Returns a tagged Object as u64 (0 signals error).
pub type JitFn = unsafe fn(*mut u8, *mut u8, *const u8) -> u64;

/// Global JIT compiler instance (lazily initialized).
static JIT_COMPILER: Mutex<Option<JitCompiler>> = Mutex::new(None);

struct JitCompiler {
    module: JITModule,
    ctx: CraneliftContext,
    func_counter: usize,
}

impl JitCompiler {
    fn new() -> Self {
        let mut flag_builder = settings::builder();
        flag_builder.set("use_colocated_libcalls", "false").unwrap();
        flag_builder.set("is_pic", "false").unwrap();
        let isa_builder = cranelift_codegen::isa::lookup(target_lexicon::Triple::host()).unwrap();
        let isa = isa_builder.finish(settings::Flags::new(flag_builder)).unwrap();
        let builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
        let module = JITModule::new(builder);
        let ctx = module.make_context();
        JitCompiler { module, ctx, func_counter: 0 }
    }
}

// ── Helper function signatures ──────────────────────────────────────────
// These are the Rust functions that JIT-compiled code calls back into.
// Each mirrors one opcode's behavior from execute_bytecode.
//
// All helpers take (env: *mut u8, cx: *mut u8) plus opcode-specific args,
// and return a u64 status (0 = ok, nonzero = error).

/// Push a constant from the constants vector onto the stack.
/// `consts` is a pointer to the &[Object] slice data, `idx` is the index.
#[no_mangle]
pub extern "C" fn jit_helper_constant(env: *mut u8, _cx: *mut u8, consts: *const u8, idx: u64) -> u64 {
    unsafe {
        let env = &mut *(env as *mut crate::core::gc::Rt<crate::core::env::Env>);
        // consts points to the start of the Object slice
        let obj_ptr = consts as *const crate::core::object::Object;
        let obj = *obj_ptr.add(idx as usize);
        env.stack.push(obj);
    }
    0
}

/// Push a stack-relative reference.
#[no_mangle]
pub extern "C" fn jit_helper_stack_ref(env: *mut u8, _cx: *mut u8, idx: u64) -> u64 {
    unsafe {
        let env = &mut *(env as *mut crate::core::gc::Rt<crate::core::env::Env>);
        // push_ref takes a u16 index
        // We need a Context for the lifetime, but push_ref on the raw Rt
        // doesn't actually need it for the data — it just copies a slot.
        // For now, use a dummy approach: read the slot and push it.
        let val = env.stack[idx as u16].bind_raw();
        env.stack.push(val);
    }
    0
}

/// Pop and discard the top of stack.
#[no_mangle]
pub extern "C" fn jit_helper_discard(env: *mut u8, cx: *mut u8) -> u64 {
    unsafe {
        let env = &mut *(env as *mut crate::core::gc::Rt<crate::core::env::Env>);
        let cx = &*(cx as *const crate::core::gc::Context);
        env.stack.pop(cx);
    }
    0
}

/// Duplicate top of stack.
#[no_mangle]
pub extern "C" fn jit_helper_duplicate(env: *mut u8, cx: *mut u8) -> u64 {
    unsafe {
        let env = &mut *(env as *mut crate::core::gc::Rt<crate::core::env::Env>);
        let cx = &*(cx as *const crate::core::gc::Context);
        let top = env.stack[0].bind(cx);
        env.stack.push(top);
    }
    0
}

/// Return: pop top of stack and write it to the return slot.
/// Returns the raw Object bits so the JIT wrapper can return it.
#[no_mangle]
pub extern "C" fn jit_helper_return(env: *mut u8, cx: *mut u8) -> u64 {
    unsafe {
        let env = &mut *(env as *mut crate::core::gc::Rt<crate::core::env::Env>);
        let cx = &*(cx as *const crate::core::gc::Context);
        let top = env.stack.pop(cx);
        // Transmute the tagged Object pointer to u64
        std::mem::transmute::<crate::core::object::Object, u64>(top)
    }
}

/// Compile a ByteFn's opcodes into native code via Cranelift.
/// Returns None if compilation fails (unsupported opcodes, etc).
pub fn compile_bytefn(codes: &[u8], _num_consts: usize) -> Option<JitFn> {
    let mut guard = JIT_COMPILER.lock().unwrap();
    let jit = guard.get_or_insert_with(JitCompiler::new);

    let ptr_type = jit.module.target_config().pointer_type();

    // Signature: fn(env: ptr, cx: ptr, consts: ptr) -> i64
    jit.ctx.func.signature.params.push(AbiParam::new(ptr_type)); // env
    jit.ctx.func.signature.params.push(AbiParam::new(ptr_type)); // cx
    jit.ctx.func.signature.params.push(AbiParam::new(ptr_type)); // consts
    jit.ctx.func.signature.returns.push(AbiParam::new(types::I64)); // return Object as i64

    let mut fb_ctx = FunctionBuilderContext::new();
    let mut builder = FunctionBuilder::new(&mut jit.ctx.func, &mut fb_ctx);

    let entry_block = builder.create_block();
    builder.append_block_params_for_function_params(entry_block);
    builder.switch_to_block(entry_block);

    let env_val = builder.block_params(entry_block)[0];
    let cx_val = builder.block_params(entry_block)[1];
    let consts_val = builder.block_params(entry_block)[2];

    // Declare helper function signatures we'll call
    let mut helper_sigs = HelperSigs::new(&mut jit.module, ptr_type);

    // Pre-scan bytecode to find branch targets so we can create blocks
    let block_map = build_block_map(codes, &mut builder);

    // Walk the bytecode and emit Cranelift IR
    let mut pc = 0;
    let mut sealed = vec![false; codes.len() + 1];
    sealed[0] = true; // entry block is sealed

    // Track current block — we may need to switch at branch targets
    let mut current_block = entry_block;

    while pc < codes.len() {
        // If this pc is a branch target, switch to its block
        if let Some(&target_block) = block_map.get(&pc) {
            if target_block != current_block {
                // Jump from previous block to this one (fallthrough)
                if !builder.is_filled() {
                    builder.ins().jump(target_block, &[]);
                }
                builder.switch_to_block(target_block);
                current_block = target_block;
            }
            if !sealed[pc] {
                builder.seal_block(target_block);
                sealed[pc] = true;
            }
        }

        let op_byte = codes[pc];
        let op: OpCode = match op_byte.try_into() {
            Ok(op) => op,
            Err(_) => {
                // Unknown opcode — bail out of JIT
                builder.finalize();
                jit.module.clear_context(&mut jit.ctx);
                return None;
            }
        };
        pc += 1;

        match op {
            // ── Constants ───────────────────────────────────────
            OpCode::Constant0 | OpCode::Constant1 | OpCode::Constant2
            | OpCode::Constant3 | OpCode::Constant4 | OpCode::Constant5
            | OpCode::Constant6 | OpCode::Constant7 | OpCode::Constant8
            | OpCode::Constant9 | OpCode::Constant10 | OpCode::Constant11
            | OpCode::Constant12 | OpCode::Constant13 | OpCode::Constant14
            | OpCode::Constant15 | OpCode::Constant16 | OpCode::Constant17
            | OpCode::Constant18 | OpCode::Constant19 | OpCode::Constant20
            | OpCode::Constant21 | OpCode::Constant22 | OpCode::Constant23
            | OpCode::Constant24 | OpCode::Constant25 | OpCode::Constant26
            | OpCode::Constant27 | OpCode::Constant28 | OpCode::Constant29
            | OpCode::Constant30 | OpCode::Constant31 | OpCode::Constant32
            | OpCode::Constant33 | OpCode::Constant34 | OpCode::Constant35
            | OpCode::Constant36 | OpCode::Constant37 | OpCode::Constant38
            | OpCode::Constant39 | OpCode::Constant40 | OpCode::Constant41
            | OpCode::Constant42 | OpCode::Constant43 | OpCode::Constant44
            | OpCode::Constant45 | OpCode::Constant46 | OpCode::Constant47
            | OpCode::Constant48 | OpCode::Constant49 | OpCode::Constant50
            | OpCode::Constant51 | OpCode::Constant52 | OpCode::Constant53
            | OpCode::Constant54 | OpCode::Constant55 | OpCode::Constant56
            | OpCode::Constant57 | OpCode::Constant58 | OpCode::Constant59
            | OpCode::Constant60 | OpCode::Constant61 | OpCode::Constant62
            | OpCode::Constant63 => {
                let idx = (op as u8) - (OpCode::Constant0 as u8);
                let idx_val = builder.ins().iconst(types::I64, idx as i64);
                helper_sigs.call_constant(&mut builder, env_val, cx_val, consts_val, idx_val);
            }
            OpCode::ConstantN2 => {
                let idx = read_u16(codes, &mut pc);
                let idx_val = builder.ins().iconst(types::I64, idx as i64);
                helper_sigs.call_constant(&mut builder, env_val, cx_val, consts_val, idx_val);
            }

            // ── Stack ops ───────────────────────────────────────
            OpCode::StackRef0 | OpCode::StackRef1 | OpCode::StackRef2
            | OpCode::StackRef3 | OpCode::StackRef4 | OpCode::StackRef5 => {
                let idx = (op as u8) - (OpCode::StackRef0 as u8);
                let idx_val = builder.ins().iconst(types::I64, idx as i64);
                helper_sigs.call_stack_ref(&mut builder, env_val, cx_val, idx_val);
            }
            OpCode::StackRefN => {
                let idx = read_u8(codes, &mut pc);
                let idx_val = builder.ins().iconst(types::I64, idx as i64);
                helper_sigs.call_stack_ref(&mut builder, env_val, cx_val, idx_val);
            }
            OpCode::StackRefN2 => {
                let idx = read_u16(codes, &mut pc);
                let idx_val = builder.ins().iconst(types::I64, idx as i64);
                helper_sigs.call_stack_ref(&mut builder, env_val, cx_val, idx_val);
            }

            OpCode::Discard => {
                helper_sigs.call_discard(&mut builder, env_val, cx_val);
            }
            OpCode::Duplicate => {
                helper_sigs.call_duplicate(&mut builder, env_val, cx_val);
            }

            // ── Return ──────────────────────────────────────────
            OpCode::Return => {
                let ret = helper_sigs.call_return(&mut builder, env_val, cx_val);
                builder.ins().return_(&[ret]);
            }

            // ── Branches ────────────────────────────────────────
            OpCode::Goto => {
                let offset = read_u16(codes, &mut pc) as usize;
                let target = *block_map.get(&offset).expect("goto target not in block_map");
                builder.ins().jump(target, &[]);
            }
            OpCode::GotoIfNil => {
                let offset = read_u16(codes, &mut pc) as usize;
                let target = *block_map.get(&offset).expect("branch target not in block_map");
                let fallthrough = builder.create_block();
                // Pop top, check if nil (nil = Object with all zero bits)
                let ret = helper_sigs.call_return(&mut builder, env_val, cx_val);
                let zero = builder.ins().iconst(types::I64, 0);
                let is_nil = builder.ins().icmp(cranelift_codegen::ir::condcodes::IntCC::Equal, ret, zero);
                builder.ins().brif(is_nil, target, &[], fallthrough, &[]);
                builder.switch_to_block(fallthrough);
                builder.seal_block(fallthrough);
                current_block = fallthrough;
            }
            OpCode::GotoIfNonNil => {
                let offset = read_u16(codes, &mut pc) as usize;
                let target = *block_map.get(&offset).expect("branch target not in block_map");
                let fallthrough = builder.create_block();
                let ret = helper_sigs.call_return(&mut builder, env_val, cx_val);
                let zero = builder.ins().iconst(types::I64, 0);
                let is_nil = builder.ins().icmp(cranelift_codegen::ir::condcodes::IntCC::Equal, ret, zero);
                builder.ins().brif(is_nil, fallthrough, &[], target, &[]);
                builder.switch_to_block(fallthrough);
                builder.seal_block(fallthrough);
                current_block = fallthrough;
            }

            // ── Unsupported — bail ──────────────────────────────
            _ => {
                builder.finalize();
                jit.module.clear_context(&mut jit.ctx);
                return None;
            }
        }
    }

    // If we fell off the end without a Return, emit a nil return
    if !builder.is_filled() {
        let zero = builder.ins().iconst(types::I64, 0);
        builder.ins().return_(&[zero]);
    }

    // Seal any unsealed blocks
    builder.seal_all_blocks();
    builder.finalize();

    // Compile
    let name = format!("__jit_fn_{}", jit.func_counter);
    jit.func_counter += 1;

    let func_id = jit.module
        .declare_function(&name, Linkage::Export, &jit.ctx.func.signature)
        .ok()?;
    jit.module.define_function(func_id, &mut jit.ctx).ok()?;
    jit.module.clear_context(&mut jit.ctx);
    jit.module.finalize_definitions().ok()?;

    let code_ptr = jit.module.get_finalized_function(func_id);
    Some(unsafe { std::mem::transmute(code_ptr) })
}

// ── Bytecode reading helpers ────────────────────────────────────────────

fn read_u8(codes: &[u8], pc: &mut usize) -> u8 {
    let v = codes[*pc];
    *pc += 1;
    v
}

fn read_u16(codes: &[u8], pc: &mut usize) -> u16 {
    let lo = codes[*pc] as u16;
    let hi = codes[*pc + 1] as u16;
    *pc += 2;
    lo | (hi << 8)
}

/// Pre-scan bytecode to find all branch targets and create Cranelift blocks.
fn build_block_map(
    codes: &[u8],
    builder: &mut FunctionBuilder,
) -> HashMap<usize, cranelift_codegen::ir::Block> {
    let mut targets = std::collections::HashSet::new();
    let mut pc = 0;
    while pc < codes.len() {
        let op_byte = codes[pc];
        pc += 1;
        let op: OpCode = match op_byte.try_into() {
            Ok(op) => op,
            Err(_) => continue,
        };
        match op {
            OpCode::Goto | OpCode::GotoIfNil | OpCode::GotoIfNonNil
            | OpCode::GotoIfNilElsePop | OpCode::GotoIfNonNilElsePop => {
                if pc + 1 < codes.len() {
                    let lo = codes[pc] as u16;
                    let hi = codes[pc + 1] as u16;
                    let offset = (lo | (hi << 8)) as usize;
                    targets.insert(offset);
                }
                pc += 2;
            }
            OpCode::PushCondtionCase => { pc += 2; }
            OpCode::ConstantN2 | OpCode::StackSetN2 | OpCode::StackRefN2
            | OpCode::VarRefN2 | OpCode::VarSetN2 | OpCode::VarBindN2
            | OpCode::CallN2 | OpCode::UnbindN2 => { pc += 2; }
            OpCode::StackRefN | OpCode::StackSetN | OpCode::VarRefN
            | OpCode::VarSetN | OpCode::VarBindN | OpCode::CallN
            | OpCode::UnbindN | OpCode::DiscardN | OpCode::ListN
            | OpCode::ConcatN | OpCode::InsertN => { pc += 1; }
            _ => {}
        }
    }
    targets.into_iter()
        .map(|offset| (offset, builder.create_block()))
        .collect()
}

// ── Helper call wrappers ────────────────────────────────────────────────

struct HelperSigs {
    sig_env_cx_consts_idx: cranelift_codegen::ir::SigRef,
    sig_env_cx_idx: cranelift_codegen::ir::SigRef,
    sig_env_cx: cranelift_codegen::ir::SigRef,
    sig_env_cx_ret: cranelift_codegen::ir::SigRef,
    fn_constant: cranelift_module::FuncId,
    fn_stack_ref: cranelift_module::FuncId,
    fn_discard: cranelift_module::FuncId,
    fn_duplicate: cranelift_module::FuncId,
    fn_return: cranelift_module::FuncId,
}

impl HelperSigs {
    fn new(module: &mut JITModule, ptr: types::Type) -> Self {
        // Register helper functions with the JIT module
        let mut builder = JITBuilder::with_isa(
            cranelift_codegen::isa::lookup(target_lexicon::Triple::host()).unwrap()
                .finish(settings::Flags::new(settings::builder())).unwrap(),
            cranelift_module::default_libcall_names(),
        );

        // We need to declare the signatures and import the functions
        // into the existing module. Use module.declare_function with
        // Import linkage.

        let mut sig4 = module.make_signature();
        sig4.params.push(AbiParam::new(ptr)); // env
        sig4.params.push(AbiParam::new(ptr)); // cx
        sig4.params.push(AbiParam::new(ptr)); // consts
        sig4.params.push(AbiParam::new(types::I64)); // idx
        sig4.returns.push(AbiParam::new(types::I64));

        let mut sig3 = module.make_signature();
        sig3.params.push(AbiParam::new(ptr)); // env
        sig3.params.push(AbiParam::new(ptr)); // cx
        sig3.params.push(AbiParam::new(types::I64)); // idx
        sig3.returns.push(AbiParam::new(types::I64));

        let mut sig2 = module.make_signature();
        sig2.params.push(AbiParam::new(ptr)); // env
        sig2.params.push(AbiParam::new(ptr)); // cx
        sig2.returns.push(AbiParam::new(types::I64));

        let mut sig2_ret = module.make_signature();
        sig2_ret.params.push(AbiParam::new(ptr)); // env
        sig2_ret.params.push(AbiParam::new(ptr)); // cx
        sig2_ret.returns.push(AbiParam::new(types::I64));

        let fn_constant = module.declare_function("jit_helper_constant", Linkage::Import, &sig4).unwrap();
        let fn_stack_ref = module.declare_function("jit_helper_stack_ref", Linkage::Import, &sig3).unwrap();
        let fn_discard = module.declare_function("jit_helper_discard", Linkage::Import, &sig2).unwrap();
        let fn_duplicate = module.declare_function("jit_helper_duplicate", Linkage::Import, &sig2).unwrap();
        let fn_return = module.declare_function("jit_helper_return", Linkage::Import, &sig2_ret).unwrap();

        // We need SigRefs for the call instructions
        // These will be created per-function in the builder
        HelperSigs {
            sig_env_cx_consts_idx: cranelift_codegen::ir::SigRef::from_u32(0), // placeholder
            sig_env_cx_idx: cranelift_codegen::ir::SigRef::from_u32(0),
            sig_env_cx: cranelift_codegen::ir::SigRef::from_u32(0),
            sig_env_cx_ret: cranelift_codegen::ir::SigRef::from_u32(0),
            fn_constant,
            fn_stack_ref,
            fn_discard,
            fn_duplicate,
            fn_return,
        }
    }

    fn call_constant(
        &self,
        builder: &mut FunctionBuilder,
        env: cranelift_codegen::ir::Value,
        cx: cranelift_codegen::ir::Value,
        consts: cranelift_codegen::ir::Value,
        idx: cranelift_codegen::ir::Value,
    ) {
        let func_ref = builder.func.dfg.ext_funcs.get(self.fn_constant.into()).copied();
        // TODO: proper func_ref import
        let _ = (env, cx, consts, idx);
    }

    fn call_stack_ref(
        &self,
        builder: &mut FunctionBuilder,
        env: cranelift_codegen::ir::Value,
        cx: cranelift_codegen::ir::Value,
        idx: cranelift_codegen::ir::Value,
    ) {
        let _ = (env, cx, idx);
    }

    fn call_discard(
        &self,
        builder: &mut FunctionBuilder,
        env: cranelift_codegen::ir::Value,
        cx: cranelift_codegen::ir::Value,
    ) {
        let _ = (env, cx);
    }

    fn call_duplicate(
        &self,
        builder: &mut FunctionBuilder,
        env: cranelift_codegen::ir::Value,
        cx: cranelift_codegen::ir::Value,
    ) {
        let _ = (env, cx);
    }

    fn call_return(
        &self,
        builder: &mut FunctionBuilder,
        env: cranelift_codegen::ir::Value,
        cx: cranelift_codegen::ir::Value,
    ) -> cranelift_codegen::ir::Value {
        let _ = (env, cx);
        builder.ins().iconst(types::I64, 0) // placeholder
    }
}
