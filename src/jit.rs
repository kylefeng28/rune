use cranelift_codegen::ir::{
    types, AbiParam, InstBuilder, GlobalValue,
    condcodes::IntCC,
};
use cranelift_codegen::settings;
use cranelift_codegen::settings::Configurable;
use cranelift_codegen::{Context as CraneliftContext};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Variable};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{DataDescription, DataId, FuncId, Linkage, Module};

use crate::core::{
    gc::{Context as GcContext},
    object::{FunctionType, Gc, Object},
};

use crate::eval::{ErrorType, EvalError, EvalResult, add_trace};

#[repr(C)]
struct StrResult {
    ptr: *const u8,
    len: i64,
}

pub struct JIT {
    module: JITModule,
    ctx: CraneliftContext,
}

impl JIT {
    pub fn new() -> Self {
        let mut flag_builder = settings::builder();
        flag_builder.set("use_colocated_libcalls", "false").unwrap();
        flag_builder.set("is_pic", "false").unwrap();

        // let isa_builder = cranelift_codegen::isa::lookup_by_name("aarch64-apple-darwin").unwrap();
        use target_lexicon::Triple;
        let isa_builder = cranelift_codegen::isa::lookup(Triple::host()).unwrap();

        let isa = isa_builder.finish(settings::Flags::new(flag_builder)).unwrap();

        let mut builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());

        let mut module = JITModule::new(builder);
        let ptr_type = module.target_config().pointer_type();

        let ctx = module.make_context();

        JIT { module, ctx }
    }

    /// Compile and run a program, returning the value of the last expression
    pub fn compile_and_run(&mut self, program: &IrProgram) -> i64 {
        self.translate(program);

        let main_id = self
            .module
            .declare_function("__chapeau_main", Linkage::Export, &self.ctx.func.signature)
            .unwrap();

        self.module.define_function(main_id, &mut self.ctx).unwrap();
        self.module.clear_context(&mut self.ctx);
        self.module.finalize_definitions().unwrap();

        let code_ptr = self.module.get_finalized_function(main_id);
        let func: fn() -> i64 = unsafe { std::mem::transmute(code_ptr) };
        func()
    }

    fn translate(&mut self, program: &IrProgram) {
        // main takes no args, returns i64
        self.ctx.func.signature.returns.push(AbiParam::new(types::I64));

        let mut fb_ctx = FunctionBuilderContext::new();
        let mut builder = FunctionBuilder::new(&mut self.ctx.func, &mut fb_ctx);

        // Allocate Cranelift variables for each IR register
        let mut reg_vars: Vec<Variable> = Vec::new();
        for (i, info) in program.registers.iter().enumerate() {
            let var = builder.declare_var(types::I64);
            reg_vars.push(var);
        }

        // Create entry block for Cranelift
        let entry_block = builder.create_block();
        builder.switch_to_block(entry_block);
        builder.seal_block(entry_block);

        let mut emitter = Emitter {
            builder,
            reg_vars,
        };

        emitter.emit_instructions(&program.instructions);

        // Return the last expression value
        let ret_val = emitter.builder.use_var(emitter.reg_vars[program.ret_reg as usize]);
        emitter.builder.ins().return_(&[ret_val]);

        emitter.builder.finalize();
    }
}

struct Emitter<'a> {
    builder: FunctionBuilder<'a>,
    /// Cranelift variable for each IR register
    reg_vars: Vec<Variable>,
}

impl<'a> Emitter<'a> {
    fn emit_instructions(&mut self, instructions: &[Instruction]) {
        for inst in instructions {
            self.emit_instruction(inst);
        }
    }

    fn emit_instruction(&mut self, inst: &Instruction) {
        match inst {
            Instruction::LoadInt(rd, val) => {
                let v = self.builder.ins().iconst(types::I64, *val);
                self.builder.def_var(self.reg_vars[*rd as usize], v);
            },
            Instruction::Mov(rd, rs) => {
                let val = self.builder.use_var(self.reg_vars[*rs as usize]);
                self.builder.def_var(self.reg_vars[*rd as usize], val);
            }
            Instruction::BinOp(rd, op, rs1, rs2) => {
                let l = self.builder.use_var(self.reg_vars[*rs1 as usize]);
                let r = self.builder.use_var(self.reg_vars[*rs2 as usize]);
                let result = match op {
                    BinOp::Add => self.builder.ins().iadd(l, r),
                    BinOp::Sub => self.builder.ins().isub(l, r),
                    BinOp::Mul => self.builder.ins().imul(l, r),
                    BinOp::Div => self.builder.ins().sdiv(l, r),
                    BinOp::Eq => {
                        let cmp = self.builder.ins().icmp(IntCC::Equal, l, r);
                        self.builder.ins().uextend(types::I64, cmp)
                    }
                    BinOp::Neq => {
                        let cmp = self.builder.ins().icmp(IntCC::NotEqual, l, r);
                        self.builder.ins().uextend(types::I64, cmp)
                    }
                    BinOp::Lt => {
                        let cmp = self.builder.ins().icmp(IntCC::SignedLessThan, l, r);
                        self.builder.ins().uextend(types::I64, cmp)
                    }
                    BinOp::Gt => {
                        let cmp = self.builder.ins().icmp(IntCC::SignedGreaterThan, l, r);
                        self.builder.ins().uextend(types::I64, cmp)
                    }
                    BinOp::Lte => {
                        let cmp = self.builder.ins().icmp(IntCC::SignedLessThanOrEqual, l, r);
                        self.builder.ins().uextend(types::I64, cmp)
                    }
                    BinOp::Gte => {
                        let cmp = self.builder.ins().icmp(IntCC::SignedGreaterThanOrEqual, l, r);
                        self.builder.ins().uextend(types::I64, cmp)
                    }
                    _ => panic!("unsupported binary op: {op:?}"),
                };
                self.builder.def_var(self.reg_vars[*rd as usize], result);
            }
            Instruction::UnOp(rd, op, rs) => {
                let val = self.builder.use_var(self.reg_vars[*rs as usize]);
                let result = match op {
                    UnaryOp::Neg => self.builder.ins().ineg(val),
                    UnaryOp::Not => {
                        let zero = self.builder.ins().iconst(types::I64, 0);
                        let cmp = self.builder.ins().icmp(IntCC::Equal, val, zero);
                        self.builder.ins().uextend(types::I64, cmp)
                    }
                    _ => panic!("unsupported unary op: {op:?}"),
                };
                self.builder.def_var(self.reg_vars[*rd as usize], result);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn jit_run(input: &str) -> i64 {
        let tokens = crate::lexer::lex(input).unwrap();
        let program = crate::parser::parse_program(&mut tokens.peekable());
        let ir_program = crate::ir::lower(&program);
        let mut jit = JIT::new();
        jit.compile_and_run(&ir_program)
    }

    #[test]
    fn test_arithmetic() {
        assert_eq!(jit_run("1 + 2"), 3);
        assert_eq!(jit_run("10 * 5"), 50);
        assert_eq!(jit_run("(100 + 2) / 6"), 17);
    }

    #[test]
    fn test_unary() {
        assert_eq!(jit_run("-1"), -1);
        assert_eq!(jit_run("-1 + 2"), 1);
    }

    #[test]
    fn test_comparisons() {
        assert_eq!(jit_run("1 + 2 == 3"), 1);
        assert_eq!(jit_run("5 > 3"), 1);
        assert_eq!(jit_run("3 > 5"), 0);
        assert_eq!(jit_run("3 < 5"), 1);
        assert_eq!(jit_run("5 <= 5"), 1);
        assert_eq!(jit_run("5 >= 6"), 0);
        assert_eq!(jit_run("1 != 2"), 1);
    }
}
