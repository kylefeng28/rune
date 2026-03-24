//! Cranelift JIT compiler for elisp bytecode.
//!
//! Tier 1: Direct opcode translation. Each bytecode opcode is lowered to a
//! call into a Rust helper function that performs the same work as the
//! interpreter. The JIT function signature is:
//!
//!   fn(env: *mut u8, cx: *mut u8, consts: *const u8) -> u64
//!
//! where the return is a tagged Object (or 0/nil).

use cranelift_codegen::ir::{types, AbiParam, InstBuilder, condcodes::IntCC};
use cranelift_codegen::settings::{self, Configurable};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module};
use std::collections::HashMap;

use crate::bytecode::opcode::OpCode;

/// A JIT-compiled function pointer.
pub type JitFn = unsafe fn(*mut u8, *mut u8, *const u8) -> u64;

// ── Helper functions called from JIT code ───────────────────────────────
// Each is extern "C" so Cranelift can call them directly.

#[unsafe(no_mangle)]
pub extern "C" fn jit_constant(env: *mut u8, _cx: *mut u8, consts: *const u8, idx: u64) -> u64 {
    unsafe {
        let env = &mut *(env as *mut crate::core::gc::Rt<crate::core::env::Env>);
        let obj = *(consts as *const crate::core::object::Object).add(idx as usize);
        env.stack.push(obj);
    }
    0
}

#[unsafe(no_mangle)]
pub extern "C" fn jit_stack_ref(env: *mut u8, cx: *mut u8, idx: u64) -> u64 {
    unsafe {
        let env = &mut *(env as *mut crate::core::gc::Rt<crate::core::env::Env>);
        let cx = &*(cx as *const crate::core::gc::Context);
        env.stack.push_ref(idx as u16, cx);
    }
    0
}

#[unsafe(no_mangle)]
pub extern "C" fn jit_stack_set(env: *mut u8, _cx: *mut u8, idx: u64) -> u64 {
    unsafe {
        let env = &mut *(env as *mut crate::core::gc::Rt<crate::core::env::Env>);
        env.stack.set_ref(idx as u16);
    }
    0
}

#[unsafe(no_mangle)]
pub extern "C" fn jit_discard(env: *mut u8, cx: *mut u8) -> u64 {
    unsafe {
        let env = &mut *(env as *mut crate::core::gc::Rt<crate::core::env::Env>);
        let cx = &*(cx as *const crate::core::gc::Context);
        env.stack.pop(cx);
    }
    0
}

#[unsafe(no_mangle)]
pub extern "C" fn jit_duplicate(env: *mut u8, cx: *mut u8) -> u64 {
    unsafe {
        let env = &mut *(env as *mut crate::core::gc::Rt<crate::core::env::Env>);
        let cx = &*(cx as *const crate::core::gc::Context);
        let top = env.stack[0].bind(cx);
        env.stack.push(top);
    }
    0
}

/// Pop top of stack and return its raw bits.
#[unsafe(no_mangle)]
pub extern "C" fn jit_pop_return(env: *mut u8, cx: *mut u8) -> u64 {
    unsafe {
        let env = &mut *(env as *mut crate::core::gc::Rt<crate::core::env::Env>);
        let cx = &*(cx as *const crate::core::gc::Context);
        let top = env.stack.pop(cx);
        std::mem::transmute(top)
    }
}

/// Pop top of stack, return its raw bits (for branch testing), then push it back.
#[unsafe(no_mangle)]
pub extern "C" fn jit_peek_pop(env: *mut u8, cx: *mut u8) -> u64 {
    unsafe {
        let env = &mut *(env as *mut crate::core::gc::Rt<crate::core::env::Env>);
        let cx = &*(cx as *const crate::core::gc::Context);
        let top = env.stack.pop(cx);
        std::mem::transmute(top)
    }
}

// ── AOT compilation ─────────────────────────────────────────────────────

/// Eagerly compile all ByteFn objects reachable from the global symbol table.
pub fn aot_compile_all(cx: &crate::core::gc::Context) {
    use crate::core::env::INTERNED_SYMBOLS;
    use crate::core::object::FunctionType;

    eprintln!("Starting AOT compilation...");
    let mut compiled = 0usize;
    let mut failed = 0usize;

    let map = INTERNED_SYMBOLS.lock().unwrap();
    for (name, sym) in map.iter() {
        if let Some(func) = sym.follow_indirect(cx) {
            if let FunctionType::ByteFn(bf) = func.untag() {
                if bf.jit_code.get().is_some() {
                    continue;
                }
                eprintln!("  JIT compiling {name}");
                match compile(bf.codes()) {
                    Some(jit_fn) => {
                        bf.jit_code.set(Some(jit_fn));
                        compiled += 1;
                    }
                    None => {
                        failed += 1;
                    }
                }
            }
        }
    }
    eprintln!("AOT compilation complete: {compiled} compiled, {failed} unsupported");
}

// ── Compiler ────────────────────────────────────────────────────────────

/// Compile a ByteFn's opcodes into native code. Returns None if any
/// unsupported opcode is encountered.
pub fn compile(codes: &[u8]) -> Option<JitFn> {
    let mut flag_builder = settings::builder();
    flag_builder.set("use_colocated_libcalls", "false").unwrap();
    flag_builder.set("is_pic", "false").unwrap();
    let isa = cranelift_codegen::isa::lookup(target_lexicon::Triple::host())
        .unwrap()
        .finish(settings::Flags::new(flag_builder))
        .unwrap();

    let mut builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());

    // Register our helper symbols so Cranelift can resolve them
    builder.symbol("jit_constant", jit_constant as *const u8);
    builder.symbol("jit_stack_ref", jit_stack_ref as *const u8);
    builder.symbol("jit_stack_set", jit_stack_set as *const u8);
    builder.symbol("jit_discard", jit_discard as *const u8);
    builder.symbol("jit_duplicate", jit_duplicate as *const u8);
    builder.symbol("jit_pop_return", jit_pop_return as *const u8);
    builder.symbol("jit_peek_pop", jit_peek_pop as *const u8);

    let mut module = JITModule::new(builder);
    let ptr_ty = module.target_config().pointer_type();
    let mut ctx = module.make_context();

    // fn(env, cx, consts) -> i64
    ctx.func.signature.params.push(AbiParam::new(ptr_ty));
    ctx.func.signature.params.push(AbiParam::new(ptr_ty));
    ctx.func.signature.params.push(AbiParam::new(ptr_ty));
    ctx.func.signature.returns.push(AbiParam::new(types::I64));

    // Declare imported helper functions
    let helpers = declare_helpers(&mut module, ptr_ty);

    let mut fb_ctx = FunctionBuilderContext::new();
    let mut b = FunctionBuilder::new(&mut ctx.func, &mut fb_ctx);

    // Import helper FuncIds into this function
    let h = helpers.import_into(&mut module, &mut b);

    let entry = b.create_block();
    b.append_block_params_for_function_params(entry);
    b.switch_to_block(entry);

    let env = b.block_params(entry)[0];
    let cx_v = b.block_params(entry)[1];
    let consts = b.block_params(entry)[2];

    // Pre-scan for branch targets
    let block_map = scan_branch_targets(codes, &mut b);

    let mut pc = 0usize;
    let mut current = entry;
    let mut block_filled = false;

    while pc < codes.len() {
        // Switch to target block if this PC is a branch destination
        if let Some(&blk) = block_map.get(&pc) {
            if blk != current {
                if !block_filled {
                    b.ins().jump(blk, &[]);
                }
                b.switch_to_block(blk);
                current = blk;
                block_filled = false;
            }
        }

        let op: OpCode = match codes[pc].try_into() {
            Ok(op) => op,
            Err(_) => { return None; }
        };
        pc += 1;

        match op {
            // ── Constants ───────────────────────────────────
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
                let idx_v = b.ins().iconst(types::I64, idx as i64);
                b.ins().call(h.constant, &[env, cx_v, consts, idx_v]);
            }
            OpCode::ConstantN2 => {
                let idx = read_u16(codes, &mut pc);
                let idx_v = b.ins().iconst(types::I64, idx as i64);
                b.ins().call(h.constant, &[env, cx_v, consts, idx_v]);
            }

            // ── Stack ops ───────────────────────────────────
            OpCode::StackRef0 | OpCode::StackRef1 | OpCode::StackRef2
            | OpCode::StackRef3 | OpCode::StackRef4 | OpCode::StackRef5 => {
                let idx = (op as u8) - (OpCode::StackRef0 as u8);
                let v = b.ins().iconst(types::I64, idx as i64);
                b.ins().call(h.stack_ref, &[env, cx_v, v]);
            }
            OpCode::StackRefN => {
                let idx = read_u8(codes, &mut pc);
                let v = b.ins().iconst(types::I64, idx as i64);
                b.ins().call(h.stack_ref, &[env, cx_v, v]);
            }
            OpCode::StackRefN2 => {
                let idx = read_u16(codes, &mut pc);
                let v = b.ins().iconst(types::I64, idx as i64);
                b.ins().call(h.stack_ref, &[env, cx_v, v]);
            }
            OpCode::StackSetN => {
                let idx = read_u8(codes, &mut pc);
                let v = b.ins().iconst(types::I64, idx as i64);
                b.ins().call(h.stack_set, &[env, cx_v, v]);
            }
            OpCode::StackSetN2 => {
                let idx = read_u16(codes, &mut pc);
                let v = b.ins().iconst(types::I64, idx as i64);
                b.ins().call(h.stack_set, &[env, cx_v, v]);
            }
            OpCode::Discard => {
                b.ins().call(h.discard, &[env, cx_v]);
            }
            OpCode::Duplicate => {
                b.ins().call(h.duplicate, &[env, cx_v]);
            }

            // ── Return ──────────────────────────────────────
            OpCode::Return => {
                let inst = b.ins().call(h.pop_return, &[env, cx_v]);
                let ret = b.inst_results(inst)[0];
                b.ins().return_(&[ret]);
                block_filled = true;
            }

            // ── Branches ────────────────────────────────────
            OpCode::Goto => {
                let offset = read_u16(codes, &mut pc) as usize;
                let target = get_or_create_block(&block_map, offset, &mut b);
                b.ins().jump(target, &[]);
                let dead = b.create_block();
                b.switch_to_block(dead);
                current = dead;
                block_filled = false;
            }
            OpCode::GotoIfNil => {
                let offset = read_u16(codes, &mut pc) as usize;
                let target = get_or_create_block(&block_map, offset, &mut b);
                let fall = b.create_block();
                let inst = b.ins().call(h.peek_pop, &[env, cx_v]);
                let val = b.inst_results(inst)[0];
                let zero = b.ins().iconst(types::I64, 0);
                let is_nil = b.ins().icmp(IntCC::Equal, val, zero);
                b.ins().brif(is_nil, target, &[], fall, &[]);
                b.switch_to_block(fall);
                b.seal_block(fall);
                current = fall;
                block_filled = false;
            }
            OpCode::GotoIfNonNil => {
                let offset = read_u16(codes, &mut pc) as usize;
                let target = get_or_create_block(&block_map, offset, &mut b);
                let fall = b.create_block();
                let inst = b.ins().call(h.peek_pop, &[env, cx_v]);
                let val = b.inst_results(inst)[0];
                let zero = b.ins().iconst(types::I64, 0);
                let is_nil = b.ins().icmp(IntCC::Equal, val, zero);
                b.ins().brif(is_nil, fall, &[], target, &[]);
                b.switch_to_block(fall);
                b.seal_block(fall);
                current = fall;
                block_filled = false;
            }

            // ── Anything else: bail ─────────────────────────
            _ => {
                eprintln!("    unsupported opcode: {op:?} (0x{:02x})", op as u8);
                if !block_filled {
                    let zero = b.ins().iconst(types::I64, 0);
                    b.ins().return_(&[zero]);
                }
                b.seal_all_blocks();
                b.finalize();
                module.clear_context(&mut ctx);
                return None;
            }
        }
    }

    // Implicit nil return if we fall off the end
    if !block_filled {
        let zero = b.ins().iconst(types::I64, 0);
        b.ins().return_(&[zero]);
    }

    b.seal_all_blocks();
    b.finalize();

    let func_id = module
        .declare_function("__jit_entry", Linkage::Export, &ctx.func.signature)
        .ok()?;
    module.define_function(func_id, &mut ctx).ok()?;
    module.clear_context(&mut ctx);
    module.finalize_definitions().ok()?;

    let ptr = module.get_finalized_function(func_id);
    // SAFETY: we just compiled this with the correct signature
    Some(unsafe { std::mem::transmute(ptr) })
}

// ── Bytecode reading ────────────────────────────────────────────────────

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

// ── Branch target scanning ──────────────────────────────────────────────

fn scan_branch_targets(
    codes: &[u8],
    builder: &mut FunctionBuilder,
) -> HashMap<usize, cranelift_codegen::ir::Block> {
    let mut targets = std::collections::HashSet::new();
    let mut pc = 0;
    while pc < codes.len() {
        let op: OpCode = match codes[pc].try_into() {
            Ok(op) => op,
            Err(_) => { pc += 1; continue; }
        };
        pc += 1;
        match op {
            OpCode::Goto | OpCode::GotoIfNil | OpCode::GotoIfNonNil
            | OpCode::GotoIfNilElsePop | OpCode::GotoIfNonNilElsePop => {
                if pc + 1 < codes.len() {
                    let offset = (codes[pc] as u16 | ((codes[pc + 1] as u16) << 8)) as usize;
                    targets.insert(offset);
                }
                pc += 2;
            }
            OpCode::PushCondtionCase | OpCode::ConstantN2 | OpCode::StackSetN2
            | OpCode::StackRefN2 | OpCode::VarRefN2 | OpCode::VarSetN2
            | OpCode::VarBindN2 | OpCode::CallN2 | OpCode::UnbindN2 => { pc += 2; }
            OpCode::StackRefN | OpCode::StackSetN | OpCode::VarRefN
            | OpCode::VarSetN | OpCode::VarBindN | OpCode::CallN
            | OpCode::UnbindN | OpCode::DiscardN | OpCode::ListN
            | OpCode::ConcatN | OpCode::InsertN => { pc += 1; }
            _ => {}
        }
    }
    targets.into_iter().map(|off| (off, builder.create_block())).collect()
}

fn get_or_create_block(
    map: &HashMap<usize, cranelift_codegen::ir::Block>,
    offset: usize,
    _builder: &mut FunctionBuilder,
) -> cranelift_codegen::ir::Block {
    *map.get(&offset).expect("branch target not found in block map")
}

// ── Helper function declarations ────────────────────────────────────────

struct HelperFuncIds {
    constant: cranelift_module::FuncId,
    stack_ref: cranelift_module::FuncId,
    stack_set: cranelift_module::FuncId,
    discard: cranelift_module::FuncId,
    duplicate: cranelift_module::FuncId,
    pop_return: cranelift_module::FuncId,
    peek_pop: cranelift_module::FuncId,
}

struct HelperFuncRefs {
    constant: cranelift_codegen::ir::FuncRef,
    stack_ref: cranelift_codegen::ir::FuncRef,
    stack_set: cranelift_codegen::ir::FuncRef,
    discard: cranelift_codegen::ir::FuncRef,
    duplicate: cranelift_codegen::ir::FuncRef,
    pop_return: cranelift_codegen::ir::FuncRef,
    peek_pop: cranelift_codegen::ir::FuncRef,
}

fn declare_helpers(module: &mut JITModule, ptr_ty: types::Type) -> HelperFuncIds {
    // (env, cx, consts, idx) -> i64
    let mut sig4 = module.make_signature();
    sig4.params.extend_from_slice(&[
        AbiParam::new(ptr_ty), AbiParam::new(ptr_ty),
        AbiParam::new(ptr_ty), AbiParam::new(types::I64),
    ]);
    sig4.returns.push(AbiParam::new(types::I64));

    // (env, cx, idx) -> i64
    let mut sig3 = module.make_signature();
    sig3.params.extend_from_slice(&[
        AbiParam::new(ptr_ty), AbiParam::new(ptr_ty), AbiParam::new(types::I64),
    ]);
    sig3.returns.push(AbiParam::new(types::I64));

    // (env, cx) -> i64
    let mut sig2 = module.make_signature();
    sig2.params.extend_from_slice(&[AbiParam::new(ptr_ty), AbiParam::new(ptr_ty)]);
    sig2.returns.push(AbiParam::new(types::I64));

    HelperFuncIds {
        constant: module.declare_function("jit_constant", Linkage::Import, &sig4).unwrap(),
        stack_ref: module.declare_function("jit_stack_ref", Linkage::Import, &sig3).unwrap(),
        stack_set: module.declare_function("jit_stack_set", Linkage::Import, &sig3).unwrap(),
        discard: module.declare_function("jit_discard", Linkage::Import, &sig2).unwrap(),
        duplicate: module.declare_function("jit_duplicate", Linkage::Import, &sig2).unwrap(),
        pop_return: module.declare_function("jit_pop_return", Linkage::Import, &sig2).unwrap(),
        peek_pop: module.declare_function("jit_peek_pop", Linkage::Import, &sig2).unwrap(),
    }
}

impl HelperFuncIds {
    fn import_into(self, module: &mut JITModule, builder: &mut FunctionBuilder) -> HelperFuncRefs {
        HelperFuncRefs {
            constant: module.declare_func_in_func(self.constant, builder.func),
            stack_ref: module.declare_func_in_func(self.stack_ref, builder.func),
            stack_set: module.declare_func_in_func(self.stack_set, builder.func),
            discard: module.declare_func_in_func(self.discard, builder.func),
            duplicate: module.declare_func_in_func(self.duplicate, builder.func),
            pop_return: module.declare_func_in_func(self.pop_return, builder.func),
            peek_pop: module.declare_func_in_func(self.peek_pop, builder.func),
        }
    }
}
