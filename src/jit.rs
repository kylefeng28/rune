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
// A single dispatch helper handles all opcodes. The JIT emits a call to
// jit_exec_op(env, cx, consts, opcode, arg) for each instruction.
// Returns: 0 = ok (continue), 1 = return (value in top-of-stack),
//          2 = error.

#[unsafe(no_mangle)]
pub extern "C" fn jit_exec_op(
    env: *mut u8,
    cx: *mut u8,
    consts: *const u8,
    opcode: u64,
    arg: u64,
) -> u64 {
    use crate::bytecode::opcode::OpCode as op;
    use crate::core::object::{ObjectType, NIL};
    use crate::{alloc, arith, data, fns};

    unsafe {
        let env = &mut *(env as *mut crate::core::gc::Rt<crate::core::env::Env>);
        let cx = &mut *(cx as *mut crate::core::gc::Context);

        let opcode: op = match (opcode as u8).try_into() {
            Ok(o) => o,
            Err(_) => return 2,
        };

        let result: Result<(), anyhow::Error> = (|| {
            match opcode {
                // ── Stack ───────────────────────────────────
                op::StackRef0 | op::StackRef1 | op::StackRef2
                | op::StackRef3 | op::StackRef4 | op::StackRef5 => {
                    env.stack.push_ref(arg as u16, cx);
                }
                op::StackRefN | op::StackRefN2 => {
                    env.stack.push_ref(arg as u16, cx);
                }
                op::StackSetN | op::StackSetN2 => {
                    env.stack.set_ref(arg as u16);
                }
                op::Discard => { env.stack.pop(cx); }
                op::Duplicate => {
                    let top = env.stack[0].bind(cx);
                    env.stack.push(top);
                }
                op::DiscardN => {
                    let keep_tos = (arg & 0x80) != 0;
                    let count = (arg & 0x7F) as usize;
                    let cur_len = env.stack.len();
                    if keep_tos {
                        let top = env.stack.top().bind(cx);
                        env.stack.truncate(cur_len - count);
                        env.stack.top().set(top);
                    } else {
                        env.stack.truncate(cur_len - count);
                    }
                }

                // ── Constants ───────────────────────────────
                op::Constant0 | op::Constant1 | op::Constant2
                | op::Constant3 | op::Constant4 | op::Constant5
                | op::Constant6 | op::Constant7 | op::Constant8
                | op::Constant9 | op::Constant10 | op::Constant11
                | op::Constant12 | op::Constant13 | op::Constant14
                | op::Constant15 | op::Constant16 | op::Constant17
                | op::Constant18 | op::Constant19 | op::Constant20
                | op::Constant21 | op::Constant22 | op::Constant23
                | op::Constant24 | op::Constant25 | op::Constant26
                | op::Constant27 | op::Constant28 | op::Constant29
                | op::Constant30 | op::Constant31 | op::Constant32
                | op::Constant33 | op::Constant34 | op::Constant35
                | op::Constant36 | op::Constant37 | op::Constant38
                | op::Constant39 | op::Constant40 | op::Constant41
                | op::Constant42 | op::Constant43 | op::Constant44
                | op::Constant44 | op::Constant45 | op::Constant46
                | op::Constant47 | op::Constant48 | op::Constant49
                | op::Constant50 | op::Constant51 | op::Constant52
                | op::Constant53 | op::Constant54 | op::Constant55
                | op::Constant56 | op::Constant57 | op::Constant58
                | op::Constant59 | op::Constant60 | op::Constant61
                | op::Constant62 | op::Constant63
                | op::ConstantN2 => {
                    let obj = *(consts as *const crate::core::object::Object).add(arg as usize);
                    env.stack.push(obj);
                }

                // ── VarRef ──────────────────────────────────
                op::VarRef0 | op::VarRef1 | op::VarRef2
                | op::VarRef3 | op::VarRef4 | op::VarRef5
                | op::VarRefN | op::VarRefN2 => {
                    let sym_obj = *(consts as *const crate::core::object::Object).add(arg as usize);
                    if let ObjectType::Symbol(sym) = sym_obj.untag() {
                        let var = env.vars.get(sym)
                            .ok_or_else(|| anyhow::anyhow!("Void Variable: {sym}"))?;
                        let var = var.bind(cx);
                        env.stack.push(var);
                    } else {
                        anyhow::bail!("Varref was not a symbol");
                    }
                }

                // ── VarSet ──────────────────────────────────
                op::VarSet0 | op::VarSet1 | op::VarSet2
                | op::VarSet3 | op::VarSet4 | op::VarSet5
                | op::VarSetN | op::VarSetN2 => {
                    let sym_obj = *(consts as *const crate::core::object::Object).add(arg as usize);
                    let symbol: crate::core::object::Symbol = sym_obj.try_into()?;
                    let value = env.stack.pop(cx);
                    data::set(symbol, value, env)?;
                }

                // ── VarBind ─────────────────────────────────
                op::VarBind0 | op::VarBind1 | op::VarBind2
                | op::VarBind3 | op::VarBind4 | op::VarBind5
                | op::VarBindN | op::VarBindN2 => {
                    let value = env.stack.pop(cx);
                    let sym_obj = *(consts as *const crate::core::object::Object).add(arg as usize);
                    let ObjectType::Symbol(sym) = sym_obj.untag() else {
                        anyhow::bail!("Varbind was not a symbol");
                    };
                    env.varbind(sym, value, cx);
                }

                // ── Unbind ──────────────────────────────────
                op::Unbind0 | op::Unbind1 | op::Unbind2
                | op::Unbind3 | op::Unbind4 | op::Unbind5
                | op::UnbindN | op::UnbindN2 => {
                    env.unbind(arg as u16, cx);
                }

                // ── Call ────────────────────────────────────
                op::Call0 | op::Call1 | op::Call2
                | op::Call3 | op::Call4 | op::Call5
                | op::CallN | op::CallN2 => {
                    let arg_cnt = arg as usize;
                    let func: crate::core::object::Function = env.stack[arg_cnt].bind(cx).try_into()?;
                    let mut frame = crate::core::env::stack::CallFrame::new_with_args(env, arg_cnt);
                    root!(func, cx);
                    let result = func.call(&mut frame, None, cx)?;
                    drop(frame);
                    env.stack.top().set(result);
                    cx.garbage_collect(false);
                }

                // ── Type predicates ─────────────────────────
                op::Symbolp => { let t = env.stack.top(); t.set(data::symbolp(t.bind(cx))); }
                op::Consp => { let t = env.stack.top(); t.set(data::consp(t.bind(cx))); }
                op::Stringp => { let t = env.stack.top(); t.set(data::stringp(t.bind(cx))); }
                op::Listp => { let t = env.stack.top(); t.set(data::listp(t.bind(cx))); }
                op::Numberp => { let t = env.stack.top(); t.set(data::numberp(t.bind(cx))); }
                op::Integerp => { let t = env.stack.top(); t.set(data::integerp(t.bind(cx))); }
                op::Not => { let t = env.stack.top(); t.set(data::null(t.bind(cx))); }

                // ── Cons ops ────────────────────────────────
                op::Car => { let t = env.stack.top(); t.set(data::car(t.bind_as(cx)?)); }
                op::Cdr => { let t = env.stack.top(); t.set(data::cdr(t.bind_as(cx)?)); }
                op::CarSafe => { let t = env.stack.top(); t.set(data::car_safe(t.bind(cx))); }
                op::CdrSafe => { let t = env.stack.top(); t.set(data::cdr_safe(t.bind(cx))); }
                op::Cons => {
                    let cdr = env.stack.pop(cx);
                    let car = env.stack.top();
                    car.set(data::cons(car.bind(cx), cdr, cx));
                }
                op::Setcar => {
                    let newcar = env.stack.pop(cx);
                    let t = env.stack.top();
                    t.set(data::setcar(t.bind_as(cx)?, newcar)?);
                }
                op::Setcdr => {
                    let newcdr = env.stack.pop(cx);
                    let t = env.stack.top();
                    t.set(data::setcdr(t.bind_as(cx)?, newcdr)?);
                }

                // ── List ops ────────────────────────────────
                op::List1 => { let t = env.stack.top(); t.set(alloc::list(&[t.bind(cx)], cx)); }
                op::List2 => {
                    let a2 = env.stack.pop(cx);
                    let t = env.stack.top();
                    t.set(alloc::list(&[t.bind(cx), a2], cx));
                }
                op::List3 => {
                    let a3 = env.stack.pop(cx);
                    let a2 = env.stack.pop(cx);
                    let t = env.stack.top();
                    t.set(alloc::list(&[t.bind(cx), a2, a3], cx));
                }
                op::List4 => {
                    let a4 = env.stack.pop(cx);
                    let a3 = env.stack.pop(cx);
                    let a2 = env.stack.pop(cx);
                    let t = env.stack.top();
                    t.set(alloc::list(&[t.bind(cx), a2, a3, a4], cx));
                }
                op::ListN => {
                    let size = arg as usize;
                    let slice = crate::core::gc::Rt::bind_slice(&env.stack[..size], cx);
                    let list = alloc::list(slice, cx);
                    let len = env.stack.len();
                    env.stack.truncate(len - (size - 1));
                    env.stack.top().set(list);
                }
                op::Nconc => {
                    let list2 = env.stack.pop(cx);
                    let t = env.stack.top();
                    t.set(fns::nconc(&[t.bind_as(cx)?, list2.try_into()?])?);
                }
                op::Nreverse => {
                    let t = env.stack.top();
                    t.set(fns::nreverse(t.bind_as(cx)?)?);
                }

                // ── Sequence ops ────────────────────────────
                op::Length => {
                    let t = env.stack.top();
                    t.set(fns::length(t.bind(cx))? as i64);
                }
                op::Nth => {
                    let list = env.stack.pop(cx);
                    let t = env.stack.top();
                    t.set(fns::nth(t.bind_as(cx)?, list.try_into()?)?);
                }
                op::Nthcdr => {
                    let list = env.stack.pop(cx);
                    let t = env.stack.top();
                    t.set(fns::nthcdr(t.bind_as(cx)?, list.try_into()?)?.as_obj_copy());
                }
                op::Elt => {
                    let n = env.stack.pop(cx);
                    let t = env.stack.top();
                    t.set(fns::elt(t.bind(cx), n.try_into()?, cx)?);
                }
                op::Aref => {
                    let idx = env.stack.pop(cx);
                    let t = env.stack.top();
                    t.set(data::aref(t.bind(cx), idx.try_into()?, cx)?);
                }
                op::Aset => {
                    let newlet = env.stack.pop(cx);
                    let idx = env.stack.pop(cx);
                    let t = env.stack.top();
                    t.set(data::aset(t.bind(cx), idx.try_into()?, newlet)?);
                }
                op::Member => {
                    let list = env.stack.pop(cx);
                    let t = env.stack.top();
                    t.set(fns::member(t.bind(cx), list.try_into()?)?);
                }
                op::Assq => {
                    let alist = env.stack.pop(cx);
                    let t = env.stack.top();
                    t.set(fns::assq(t.bind(cx), alist.try_into()?)?);
                }
                op::Memq => {
                    let list = env.stack.pop(cx);
                    let t = env.stack.top();
                    t.set(fns::memq(t.bind(cx), list.try_into()?)?);
                }
                op::Eq => {
                    let v1 = env.stack.pop(cx);
                    let t = env.stack.top();
                    t.set(fns::eq(t.bind(cx), v1));
                }
                op::Equal => {
                    let rhs = env.stack.pop(cx);
                    let t = env.stack.top();
                    t.set(fns::equal(t.bind(cx), rhs));
                }

                // ── Symbol ops ──────────────────────────────
                op::SymbolValue => {
                    let top = env.stack.top().bind_as(cx)?;
                    let value = data::symbol_value(top, env, cx).unwrap_or_default();
                    env.stack.top().set(value);
                }
                op::SymbolFunction => {
                    let t = env.stack.top();
                    t.set(data::symbol_function(t.bind_as(cx)?, cx));
                }
                op::Set => {
                    let newlet = env.stack.pop(cx);
                    let top = env.stack.top().bind_as(cx)?;
                    let value = data::set(top, newlet, env)?;
                    env.stack.top().set(value);
                }
                op::Fset => {
                    let def = env.stack.pop(cx);
                    let t = env.stack.top();
                    t.set::<crate::core::object::Object>(data::fset(t.bind_as(cx)?, def)?.into());
                }
                op::Get => {
                    let prop = env.stack.pop(cx).try_into()?;
                    let top = env.stack.top().bind_as(cx)?;
                    let value = data::get(top, prop, env, cx);
                    env.stack.top().set(value);
                }

                // ── Arithmetic ──────────────────────────────
                op::Add1 => {
                    let t = env.stack.top();
                    t.set(cx.add(arith::add_one(t.bind_as(cx)?)));
                }
                op::Sub1 => {
                    let t = env.stack.top();
                    t.set(cx.add(arith::sub_one(t.bind_as(cx)?)));
                }
                op::Plus => {
                    let a1 = env.stack.pop(cx);
                    let t = env.stack.top();
                    let args = &[t.bind_as(cx)?, a1.try_into()?];
                    t.set(cx.add(arith::add(args)));
                }
                op::Multiply => {
                    let a1 = env.stack.pop(cx);
                    let t = env.stack.top();
                    let args = &[t.bind_as(cx)?, a1.try_into()?];
                    t.set(cx.add(arith::mul(args)));
                }
                op::Negate => {
                    let t = env.stack.top();
                    t.set(cx.add(arith::sub(t.bind_as(cx)?, &[])));
                }
                op::Max => {
                    let a1 = env.stack.pop(cx);
                    let t = env.stack.top();
                    t.set(cx.add(arith::max(t.bind_as(cx)?, &[a1.try_into()?])));
                }
                op::Min => {
                    let a1 = env.stack.pop(cx);
                    let t = env.stack.top();
                    t.set(cx.add(arith::min(t.bind_as(cx)?, &[a1.try_into()?])));
                }

                // ── Comparisons ─────────────────────────────
                op::EqlSign => {
                    let rhs = env.stack.pop(cx);
                    let t = env.stack.top();
                    t.set::<crate::core::object::Object>(arith::num_eq(t.bind_as(cx)?, &[rhs.try_into()?]).into());
                }
                op::GreaterThan => {
                    let v1 = env.stack.pop(cx);
                    let t = env.stack.top();
                    t.set(arith::greater_than(t.bind_as(cx)?, &[v1.try_into()?]));
                }
                op::LessThan => {
                    let v1 = env.stack.pop(cx);
                    let t = env.stack.top();
                    t.set(arith::less_than(t.bind_as(cx)?, &[v1.try_into()?]));
                }
                op::LessThanOrEqual => {
                    let v1 = env.stack.pop(cx);
                    let t = env.stack.top();
                    t.set(arith::less_than_or_eq(t.bind_as(cx)?, &[v1.try_into()?]));
                }
                op::GreaterThanOrEqual => {
                    let v1 = &[env.stack.pop(cx).try_into()?];
                    let t = env.stack.top();
                    t.set(arith::greater_than_or_eq(t.bind_as(cx)?, v1));
                }

                // ── Condition/handler ───────────────────────
                op::PopHandler | op::PushCondtionCase | op::PushCatch
                | op::SaveExcursion | op::SaveRestriction | op::UnwindProtect
                | op::Switch => {
                    // These require VM-level state (handler stack, etc.)
                    // that the tier-1 JIT doesn't model. Bail.
                    return Err(anyhow::anyhow!("unsupported VM opcode: {opcode:?}"));
                }

                // ── Branches/Return handled by Cranelift directly ──
                op::Goto | op::GotoIfNil | op::GotoIfNonNil
                | op::GotoIfNilElsePop | op::GotoIfNonNilElsePop
                | op::Return => {
                    // Should never reach here — these are emitted as
                    // Cranelift control flow, not helper calls.
                    unreachable!("branch/return should not be dispatched to helper");
                }

                // ── Remaining ops: todo stubs ───────────────
                _ => {
                    return Err(anyhow::anyhow!("unimplemented opcode: {opcode:?}"));
                }
            }
            Ok(())
        })();

        match result {
            Ok(()) => 0,  // continue
            Err(_) => 2,  // error
        }
    }
}

/// Pop top of stack and return its raw bits (for Return opcode).
#[unsafe(no_mangle)]
pub extern "C" fn jit_pop_return(env: *mut u8, cx: *mut u8) -> u64 {
    unsafe {
        let env = &mut *(env as *mut crate::core::gc::Rt<crate::core::env::Env>);
        let cx = &*(cx as *const crate::core::gc::Context);
        let top = env.stack.pop(cx);
        std::mem::transmute(top)
    }
}

/// Pop and return raw bits (for branch condition testing).
#[unsafe(no_mangle)]
pub extern "C" fn jit_peek_pop(env: *mut u8, cx: *mut u8) -> u64 {
    unsafe {
        let env = &mut *(env as *mut crate::core::gc::Rt<crate::core::env::Env>);
        let cx = &*(cx as *const crate::core::gc::Context);
        let top = env.stack.pop(cx);
        std::mem::transmute(top)
    }
}

/// For GotoIfNilElsePop: peek top, return raw bits. Don't pop.
#[unsafe(no_mangle)]
pub extern "C" fn jit_peek_top(env: *mut u8, cx: *mut u8) -> u64 {
    unsafe {
        let env = &mut *(env as *mut crate::core::gc::Rt<crate::core::env::Env>);
        let cx = &*(cx as *const crate::core::gc::Context);
        let top = env.stack[0].bind(cx);
        std::mem::transmute::<crate::core::object::Object, u64>(top)
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

/// Scan all ByteFns and report every unique opcode used (for development).
pub fn scan_all_opcodes(cx: &crate::core::gc::Context) {
    use crate::core::env::INTERNED_SYMBOLS;
    use crate::core::object::FunctionType;
    use crate::bytecode::opcode::OpCode;
    use std::collections::BTreeSet;

    let mut all_ops = BTreeSet::new();
    let map = INTERNED_SYMBOLS.lock().unwrap();
    for (_name, sym) in map.iter() {
        if let Some(func) = sym.follow_indirect(cx) {
            if let FunctionType::ByteFn(bf) = func.untag() {
                for &byte in bf.codes() {
                    if let Ok(op) = OpCode::try_from(byte) {
                        all_ops.insert(op as u8);
                    }
                }
            }
        }
    }
    eprintln!("All opcodes used across ByteFns:");
    for byte in &all_ops {
        if let Ok(op) = OpCode::try_from(*byte) {
            eprintln!("  {op:?} (0x{byte:02x})");
        }
    }
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
    builder.symbol("jit_exec_op", jit_exec_op as *const u8);
    builder.symbol("jit_pop_return", jit_pop_return as *const u8);
    builder.symbol("jit_peek_pop", jit_peek_pop as *const u8);
    builder.symbol("jit_peek_top", jit_peek_top as *const u8);

    let mut module = JITModule::new(builder);
    let ptr_ty = module.target_config().pointer_type();
    let mut ctx = module.make_context();

    // fn(env, cx, consts) -> i64
    ctx.func.signature.params.push(AbiParam::new(ptr_ty));
    ctx.func.signature.params.push(AbiParam::new(ptr_ty));
    ctx.func.signature.params.push(AbiParam::new(ptr_ty));
    ctx.func.signature.returns.push(AbiParam::new(types::I64));

    // Declare helper signatures
    // jit_exec_op(env, cx, consts, opcode, arg) -> i64
    let mut sig_exec = module.make_signature();
    sig_exec.params.extend_from_slice(&[
        AbiParam::new(ptr_ty), AbiParam::new(ptr_ty), AbiParam::new(ptr_ty),
        AbiParam::new(types::I64), AbiParam::new(types::I64),
    ]);
    sig_exec.returns.push(AbiParam::new(types::I64));

    // jit_pop_return/jit_peek_pop/jit_peek_top(env, cx) -> i64
    let mut sig2 = module.make_signature();
    sig2.params.extend_from_slice(&[AbiParam::new(ptr_ty), AbiParam::new(ptr_ty)]);
    sig2.returns.push(AbiParam::new(types::I64));

    let fn_exec = module.declare_function("jit_exec_op", Linkage::Import, &sig_exec).unwrap();
    let fn_ret = module.declare_function("jit_pop_return", Linkage::Import, &sig2).unwrap();
    let fn_pop = module.declare_function("jit_peek_pop", Linkage::Import, &sig2).unwrap();
    let fn_peek = module.declare_function("jit_peek_top", Linkage::Import, &sig2).unwrap();

    let mut fb_ctx = FunctionBuilderContext::new();
    let mut b = FunctionBuilder::new(&mut ctx.func, &mut fb_ctx);

    let fr_exec = module.declare_func_in_func(fn_exec, b.func);
    let fr_ret = module.declare_func_in_func(fn_ret, b.func);
    let fr_pop = module.declare_func_in_func(fn_pop, b.func);
    let fr_peek = module.declare_func_in_func(fn_peek, b.func);

    let entry = b.create_block();
    b.append_block_params_for_function_params(entry);
    b.switch_to_block(entry);

    let env = b.block_params(entry)[0];
    let cx_v = b.block_params(entry)[1];
    let consts = b.block_params(entry)[2];

    let block_map = scan_branch_targets(codes, &mut b);

    let mut pc = 0usize;
    let mut current = entry;
    let mut block_filled = false;

    while pc < codes.len() {
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

        let op_byte = codes[pc];
        let op: OpCode = match op_byte.try_into() {
            Ok(op) => op,
            Err(_) => return None,
        };
        pc += 1;

        // Compute the immediate arg for this opcode
        let arg: u64 = match op {
            // Opcodes with inline index in the opcode byte
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
            | OpCode::Constant63 => (op_byte - OpCode::Constant0 as u8) as u64,

            OpCode::StackRef0 | OpCode::StackRef1 | OpCode::StackRef2
            | OpCode::StackRef3 | OpCode::StackRef4 | OpCode::StackRef5 =>
                (op_byte - OpCode::StackRef0 as u8) as u64,

            OpCode::VarRef0 | OpCode::VarRef1 | OpCode::VarRef2
            | OpCode::VarRef3 | OpCode::VarRef4 | OpCode::VarRef5 =>
                (op_byte - OpCode::VarRef0 as u8) as u64,

            OpCode::VarSet0 | OpCode::VarSet1 | OpCode::VarSet2
            | OpCode::VarSet3 | OpCode::VarSet4 | OpCode::VarSet5 =>
                (op_byte - OpCode::VarSet0 as u8) as u64,

            OpCode::VarBind0 | OpCode::VarBind1 | OpCode::VarBind2
            | OpCode::VarBind3 | OpCode::VarBind4 | OpCode::VarBind5 =>
                (op_byte - OpCode::VarBind0 as u8) as u64,

            OpCode::Call0 | OpCode::Call1 | OpCode::Call2
            | OpCode::Call3 | OpCode::Call4 | OpCode::Call5 =>
                (op_byte - OpCode::Call0 as u8) as u64,

            OpCode::Unbind0 | OpCode::Unbind1 | OpCode::Unbind2
            | OpCode::Unbind3 | OpCode::Unbind4 | OpCode::Unbind5 =>
                (op_byte - OpCode::Unbind0 as u8) as u64,

            // 1-byte arg opcodes
            OpCode::StackRefN | OpCode::StackSetN | OpCode::VarRefN
            | OpCode::VarSetN | OpCode::VarBindN | OpCode::CallN
            | OpCode::UnbindN | OpCode::DiscardN | OpCode::ListN => {
                read_u8(codes, &mut pc) as u64
            }

            // 2-byte arg opcodes
            OpCode::StackRefN2 | OpCode::StackSetN2 | OpCode::VarRefN2
            | OpCode::VarSetN2 | OpCode::VarBindN2 | OpCode::CallN2
            | OpCode::UnbindN2 | OpCode::ConstantN2 => {
                read_u16(codes, &mut pc) as u64
            }

            // Branch opcodes read their own arg below
            OpCode::Goto | OpCode::GotoIfNil | OpCode::GotoIfNonNil
            | OpCode::GotoIfNilElsePop | OpCode::GotoIfNonNilElsePop => 0,

            // Return and no-arg opcodes
            _ => 0,
        };

        match op {
            // ── Return: pop and return value ────────────────
            OpCode::Return => {
                let inst = b.ins().call(fr_ret, &[env, cx_v]);
                let ret = b.inst_results(inst)[0];
                b.ins().return_(&[ret]);
                block_filled = true;
            }

            // ── Goto: unconditional jump ────────────────────
            OpCode::Goto => {
                let offset = read_u16(codes, &mut pc) as usize;
                let target = *block_map.get(&offset).expect("goto target");
                b.ins().jump(target, &[]);
                let dead = b.create_block();
                b.switch_to_block(dead);
                current = dead;
                block_filled = false;
            }

            // ── GotoIfNil: pop, branch if nil ───────────────
            OpCode::GotoIfNil => {
                let offset = read_u16(codes, &mut pc) as usize;
                let target = *block_map.get(&offset).expect("branch target");
                let fall = b.create_block();
                let inst = b.ins().call(fr_pop, &[env, cx_v]);
                let val = b.inst_results(inst)[0];
                let zero = b.ins().iconst(types::I64, 0);
                let is_nil = b.ins().icmp(IntCC::Equal, val, zero);
                b.ins().brif(is_nil, target, &[], fall, &[]);
                b.switch_to_block(fall);
                b.seal_block(fall);
                current = fall;
                block_filled = false;
            }

            // ── GotoIfNonNil: pop, branch if non-nil ────────
            OpCode::GotoIfNonNil => {
                let offset = read_u16(codes, &mut pc) as usize;
                let target = *block_map.get(&offset).expect("branch target");
                let fall = b.create_block();
                let inst = b.ins().call(fr_pop, &[env, cx_v]);
                let val = b.inst_results(inst)[0];
                let zero = b.ins().iconst(types::I64, 0);
                let is_nil = b.ins().icmp(IntCC::Equal, val, zero);
                b.ins().brif(is_nil, fall, &[], target, &[]);
                b.switch_to_block(fall);
                b.seal_block(fall);
                current = fall;
                block_filled = false;
            }

            // ── GotoIfNilElsePop: peek, branch if nil, else pop ─
            OpCode::GotoIfNilElsePop => {
                let offset = read_u16(codes, &mut pc) as usize;
                let target = *block_map.get(&offset).expect("branch target");
                let fall = b.create_block();
                let inst = b.ins().call(fr_peek, &[env, cx_v]);
                let val = b.inst_results(inst)[0];
                let zero = b.ins().iconst(types::I64, 0);
                let is_nil = b.ins().icmp(IntCC::Equal, val, zero);
                b.ins().brif(is_nil, target, &[], fall, &[]);
                b.switch_to_block(fall);
                b.seal_block(fall);
                // Pop in the fallthrough (non-nil) case
                b.ins().call(fr_pop, &[env, cx_v]);
                current = fall;
                block_filled = false;
            }

            // ── GotoIfNonNilElsePop: peek, branch if non-nil, else pop
            OpCode::GotoIfNonNilElsePop => {
                let offset = read_u16(codes, &mut pc) as usize;
                let target = *block_map.get(&offset).expect("branch target");
                let fall = b.create_block();
                let inst = b.ins().call(fr_peek, &[env, cx_v]);
                let val = b.inst_results(inst)[0];
                let zero = b.ins().iconst(types::I64, 0);
                let is_nil = b.ins().icmp(IntCC::Equal, val, zero);
                b.ins().brif(is_nil, fall, &[], target, &[]);
                b.switch_to_block(fall);
                b.seal_block(fall);
                // Pop in the fallthrough (nil) case
                b.ins().call(fr_pop, &[env, cx_v]);
                current = fall;
                block_filled = false;
            }

            // ── VM-level opcodes we can't support in tier 1 ─
            OpCode::PushCondtionCase | OpCode::PushCatch
            | OpCode::SaveExcursion | OpCode::SaveRestriction
            | OpCode::UnwindProtect | OpCode::Switch => {
                // These need the handler stack / VM state
                eprintln!("    unsupported VM opcode: {op:?}");
                if !block_filled {
                    let zero = b.ins().iconst(types::I64, 0);
                    b.ins().return_(&[zero]);
                }
                b.seal_all_blocks();
                b.finalize();
                module.clear_context(&mut ctx);
                return None;
            }

            // ── Everything else: dispatch to jit_exec_op ────
            _ => {
                let op_v = b.ins().iconst(types::I64, op_byte as i64);
                let arg_v = b.ins().iconst(types::I64, arg as i64);
                let inst = b.ins().call(fr_exec, &[env, cx_v, consts, op_v, arg_v]);
                let status = b.inst_results(inst)[0];
                // Check for error (status == 2)
                let two = b.ins().iconst(types::I64, 2);
                let is_err = b.ins().icmp(IntCC::Equal, status, two);
                let ok_block = b.create_block();
                let err_block = b.create_block();
                b.ins().brif(is_err, err_block, &[], ok_block, &[]);
                // Error path: return 0 (nil / error sentinel)
                b.switch_to_block(err_block);
                b.seal_block(err_block);
                let zero = b.ins().iconst(types::I64, 0);
                b.ins().return_(&[zero]);
                // Continue
                b.switch_to_block(ok_block);
                b.seal_block(ok_block);
                current = ok_block;
                block_filled = false;
            }
        }
    }

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

// (old helper infrastructure removed — now using unified jit_exec_op dispatch)
