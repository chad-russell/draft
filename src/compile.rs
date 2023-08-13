use std::collections::{HashMap, HashSet};

use cranelift_codegen::entity::EntityRef;
use cranelift_codegen::ir::condcodes::IntCC;
use cranelift_codegen::ir::immediates::Imm64;
use cranelift_codegen::ir::{FuncRef, GlobalValue, GlobalValueData, StackSlot};
// use cranelift_codegen::ir::SourceLoc;
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Variable};

use cranelift_codegen::ir::{
    stackslot::StackSize, types, AbiParam, Block, InstBuilder, MemFlags, Signature, StackSlotData,
    StackSlotKind, Type as CraneliftType, Value as CraneliftValue,
};
use cranelift_codegen::settings::{self, Configurable};
use cranelift_codegen::Context as CodegenContext;
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{default_libcall_names, DataDescription, FuncId, Init, Linkage, Module};
use tracing::instrument;

use crate::{
    ArrayLen, AsCastStyle, CompileError, Context, DraftResult, IdVec, Node, NodeId, Op,
    StaticMemberResolution, Type,
};

#[derive(Clone, Copy, Debug)]
pub enum BackendValue {
    Func(FuncId),             // A function id, can use this to get to a function pointer
    Register(CraneliftValue), // A value type which can fit in a register
    Aggregate(StackSlot), // A value type which cannot fit into a register, in this case held in a stack slot
    AggregatePointer(CraneliftValue), // A value type which cannot fit into a register, in this case held in a pointer
}

#[derive(Clone, Copy, Debug)]
pub enum BackendReference {
    StackSlot(StackSlot),    // The stack slot directly holds the value
    Pointer(CraneliftValue), // The value is behind a pointer. For example, a struct field
}

#[derive(Clone, Copy, Debug)]
pub enum Value {
    Value(BackendValue),
    Reference(BackendReference),
}

impl Context {
    #[instrument(skip_all)]
    pub fn make_module() -> JITModule {
        let mut flags_builder = settings::builder();
        flags_builder.set("is_pic", "false").unwrap();
        flags_builder.set("enable_verifier", "false").unwrap();
        flags_builder.set("enable_alias_analysis", "false").unwrap();

        flags_builder.set("opt_level", "none").unwrap();
        // flags_builder.set("opt_level", "speed").unwrap();
        // flags_builder.set("opt_level", "speed_and_size").unwrap();

        let isa_builder = cranelift_native::builder().unwrap_or_else(|msg| {
            panic!("host machine is not supported: {}", msg);
        });
        let isa = isa_builder
            .finish(settings::Flags::new(flags_builder))
            .unwrap();

        let mut jit_builder = JITBuilder::with_isa(isa, default_libcall_names());

        jit_builder.hotswap(false);

        jit_builder.symbol("print_i64", print_i64 as *const u8);
        jit_builder.symbol("print_f64", print_f64 as *const u8);
        jit_builder.symbol("print_enum_tag", print_enum_tag as *const u8);
        jit_builder.symbol("put_char", put_char as *const u8);
        jit_builder.symbol("print_str", print_str as *const u8);

        JITModule::new(jit_builder)
    }

    #[instrument(skip_all)]
    pub fn predeclare_string_constants(&mut self) -> DraftResult<()> {
        let data_id = self
            .module
            .declare_data("string_constants", Linkage::Local, false, false)
            .map_err(|err| CompileError::Message(err.to_string()))?;
        self.string_literal_data_id = Some(data_id);

        let mut bytes = Vec::new();
        for id in self.string_literals.iter() {
            let Node::StringLiteral(sym) = self.nodes[id] else {
                unreachable!()
            };
            let s = self.string_interner.resolve(sym.0).unwrap();
            self.string_literal_offsets.insert(*id, bytes.len());
            bytes.extend(s.as_bytes());
        }

        let mut desc = DataDescription::new();
        desc.init = Init::Bytes {
            contents: bytes.into_boxed_slice(),
        };

        self.module.define_data(data_id, &desc).unwrap();

        Ok(())
    }

    #[instrument(skip_all)]
    pub fn predeclare_functions(&mut self) -> DraftResult<()> {
        for t in self.topo.clone() {
            if !self.completes.contains(&t) {
                continue;
            }

            self.predeclare_function(t)?;
        }

        Ok(())
    }

    #[instrument(skip_all)]
    pub fn define_functions(&mut self) -> DraftResult<()> {
        let mut codegen_ctx = self.module.make_context();
        let mut func_ctx = FunctionBuilderContext::new();

        let mut compile_ctx = ToplevelCompileContext {
            ctx: self,
            codegen_ctx: &mut codegen_ctx,
            func_ctx: &mut func_ctx,
        };

        for t in compile_ctx.ctx.topo.clone() {
            compile_ctx.compile_toplevel_id(t)?;
        }

        self.module
            .finalize_definitions()
            .expect("Failed to finalize definitions");

        Ok(())
    }

    #[instrument(skip_all)]
    pub fn predeclare_function(&mut self, id: NodeId) -> DraftResult<()> {
        if let Node::FnDefinition {
            name,
            params,
            return_ty,
            ..
        } = self.nodes[id].clone()
        {
            let mut sig = self.module.make_signature();

            for &param in params.borrow().iter() {
                sig.params
                    .push(AbiParam::new(self.get_cranelift_type(param)));
            }

            let return_size = return_ty.map(|rt| self.id_type_size(rt)).unwrap_or(0);

            if return_size > 0 {
                sig.returns
                    .push(AbiParam::new(self.get_cranelift_type(return_ty.unwrap())));
            }

            let func_name = self.mangled_func_name(name, id);

            let func = self
                .module
                .declare_function(&func_name, Linkage::Local, &sig)
                .unwrap();

            self.values
                .insert(id, Value::Value(BackendValue::Func(func)));
        }

        Ok(())
    }

    #[instrument(skip_all)]
    fn mangled_func_name(&self, name: Option<NodeId>, anonymous_id: NodeId) -> String {
        name.map(|n| {
            let sym = self.nodes[n].as_symbol().unwrap();
            if self.polymorph_copies.contains(&anonymous_id) {
                format!(
                    "{}_polycopy{}",
                    self.string_interner.resolve(sym.0).unwrap(),
                    anonymous_id.0
                )
            } else {
                self.string_interner.resolve(sym.0).unwrap().to_string()
            }
        })
        .unwrap_or_else(|| format!("anonymous__{}", anonymous_id.0))
    }

    #[instrument(skip_all)]
    pub fn call_fn(&mut self, fn_name: &str) -> DraftResult<()> {
        let fn_name_interned = self.string_interner.get_or_intern(fn_name);

        let node_id = self
            .top_level
            .iter()
            .filter(|tl| match self.nodes[**tl] {
                Node::FnDefinition {
                    name: Some(name), ..
                } => self.get_symbol(name).0 == fn_name_interned,
                _ => false,
            })
            .next()
            .cloned();

        if let (Some(id), true) = (node_id, self.args.run) {
            let code = self.module.get_finalized_function(self.get_func_id(id));
            let func = unsafe { std::mem::transmute::<_, fn() -> i64>(code) };
            func();
        }

        Ok(())
    }

    #[instrument(skip_all)]
    fn get_cranelift_type(&self, param: NodeId) -> CraneliftType {
        match self.types[&param] {
            Type::I8 | Type::U8 | Type::Bool => types::I8,
            Type::I16 | Type::U16 => types::I16,
            Type::I32 | Type::U32 => types::I32,
            Type::I64 | Type::U64 => types::I64,
            Type::F32 => types::F32,
            Type::F64 => types::F64,
            Type::Pointer(_)
            | Type::Func { .. }
            | Type::Struct { .. }
            | Type::Enum { .. }
            | Type::Array(_, _)
            | Type::String => self.get_pointer_type(),
            Type::Infer(_) => panic!(),
            _ => todo!("{:?}", &self.types[&param]),
        }
    }

    #[instrument(skip_all)]
    fn get_pointer_type(&self) -> CraneliftType {
        self.module.isa().pointer_type()
    }

    #[instrument(skip_all)]
    fn enum_tag_size(&self) -> u32 {
        std::mem::size_of::<u16>() as u32
    }

    #[instrument(skip_all)]
    fn id_type_size(&self, param: NodeId) -> StackSize {
        self.type_size(self.types[&param].clone())
    }

    #[instrument(skip_all)]
    fn type_size(&self, ty: Type) -> StackSize {
        match ty {
            Type::I8 | Type::U8 | Type::Bool => 1,
            Type::I16 | Type::U16 => 2,
            Type::I32 | Type::U32 | Type::F32 => 4,
            Type::I64 | Type::U64 | Type::F64 => 8,
            Type::Func { .. } => 8,
            Type::Pointer(_) => self.get_pointer_type().bytes(),
            Type::Array(ty, ArrayLen::Some(len)) => self.id_type_size(ty) * len as StackSize,
            Type::Struct { params, .. } => {
                // todo(chad): c struct packing rules if annotated
                params
                    .clone()
                    .borrow()
                    .iter()
                    .map(|f| self.id_type_size(*f))
                    .sum()
            }
            Type::Enum { params, .. } => {
                params
                    .clone()
                    .borrow()
                    .iter()
                    .map(|f| self.id_type_size(*f))
                    .max()
                    .unwrap_or_default()
                    + self.enum_tag_size()
            }
            Type::Array(_, ArrayLen::None) | Type::String => 16,
            Type::EnumNoneType | Type::Empty => 0,
            _ => todo!("get_type_size for {:?}", ty),
        }
    }

    #[instrument(skip_all)]
    pub fn get_func_id(&self, id: NodeId) -> FuncId {
        match self.values[&id] {
            Value::Value(BackendValue::Func(f)) => f,
            _ => panic!("Not a function"),
        }
    }

    #[instrument(skip_all)]
    pub fn get_func_signature(&self, func: NodeId, param_ids: &Vec<NodeId>) -> Signature {
        let func_ty = self.get_type(func);
        let return_ty = match func_ty {
            Type::Func {
                return_ty: Some(return_ty),
                ..
            } => Some(return_ty),
            _ => None,
        };

        let return_size = return_ty.map(|rt| self.id_type_size(rt)).unwrap_or(0);

        let mut sig = self.module.make_signature();
        for param in param_ids.iter() {
            sig.params
                .push(AbiParam::new(self.get_cranelift_type(*param)));
        }

        if return_size > 0 {
            sig.returns
                .push(AbiParam::new(self.get_cranelift_type(return_ty.unwrap())));
        }

        sig
    }

    #[instrument(skip_all)]
    pub fn id_is_aggregate_type(&self, id: NodeId) -> bool {
        self.is_aggregate_type(self.get_type(id))
    }

    #[instrument(skip_all)]
    pub fn id_base_is_aggregate_type(&self, id: NodeId) -> bool {
        let ty = self.get_type(id);
        if self.is_aggregate_type(ty.clone()) {
            return true;
        }
        if let Type::Pointer(base) = ty {
            return self.id_base_is_aggregate_type(base);
        }
        return false;
    }

    #[instrument(skip_all)]
    pub fn is_aggregate_type(&self, ty: Type) -> bool {
        matches!(
            ty,
            Type::Struct { .. } | Type::Enum { .. } | Type::Array(_, _) | Type::String
        )
    }
}

pub fn print_i64(n: i64) {
    println!("{}", n);
}

pub fn print_f64(n: f64) {
    println!("{}", n);
}

#[repr(C)]
pub struct Enum {
    pub tag: u16,
    pub value: i64,
}

pub fn print_enum_tag(n: *const Enum) {
    println!("{}", unsafe { (*n).tag });
}

pub fn put_char(n: u8) {
    print!("{}", n as char);
}

pub fn print_str(s: *const u8, len: i64) {
    print!("{}", unsafe {
        std::str::from_utf8_unchecked(std::slice::from_raw_parts(s, len as usize))
    });
}

struct ToplevelCompileContext<'a> {
    pub ctx: &'a mut Context,
    pub codegen_ctx: &'a mut CodegenContext,
    pub func_ctx: &'a mut FunctionBuilderContext,
}

struct FunctionCompileContext<'a> {
    pub ctx: &'a mut Context,
    pub builder: FunctionBuilder<'a>,
    pub exited_blocks: HashSet<Block>, // which blocks have been terminated
    pub current_block: Block,          // where we are currently emitting instructions
    pub resolve_addr: Option<Value>, // the address of the stack slot where we store the value of any Node::Resolve we encounter
    pub resolve_jump_target: Option<Block>, // the jump target for any Node::Resolve we encounter
    pub declared_func_ids: HashMap<FuncId, FuncRef>,
    pub global_str_ptr: Option<GlobalValue>,
}

impl<'a> FunctionCompileContext<'a> {
    #[instrument(skip_all)]
    pub fn compile_id(&mut self, id: NodeId) -> DraftResult<()> {
        // idempotency
        match self.ctx.values.get(&id) {
            None => {}
            _ => return Ok(()),
        };

        // if id.0 < 10 {
        //     self.builder.set_srcloc(SourceLoc::new(id.0 as u32));
        // } else {
        //     self.builder.set_srcloc(SourceLoc::default());
        // }

        match self.ctx.nodes[id].clone() {
            Node::Symbol(sym) => {
                let resolved = self.ctx.scope_get(sym, id);

                match resolved {
                    Some(res) => {
                        self.compile_id(res)?;

                        self.ctx
                            .values
                            .get(&res)
                            .cloned()
                            .map(|v| self.ctx.values.insert(id, v));
                        Ok(())
                    }
                    _ => todo!(),
                }
            }
            Node::IntLiteral(n, _) => {
                match self.ctx.types[&id] {
                    Type::I64 | Type::U64 => {
                        let value = self.builder.ins().iconst(types::I64, n);
                        self.ctx
                            .values
                            .insert(id, Value::Value(BackendValue::Register(value)));
                    }
                    Type::I32 | Type::U32 => {
                        let value = self.builder.ins().iconst(types::I32, n);
                        self.ctx
                            .values
                            .insert(id, Value::Value(BackendValue::Register(value)));
                    }
                    _ => todo!(),
                };

                Ok(())
            }
            Node::FloatLiteral(n, _) => {
                match self.ctx.types[&id] {
                    Type::F64 => {
                        let value = self.builder.ins().f64const(n);
                        self.ctx
                            .values
                            .insert(id, Value::Value(BackendValue::Register(value)));
                    }
                    Type::F32 => {
                        let value = self.builder.ins().f32const(n as f32);
                        self.ctx
                            .values
                            .insert(id, Value::Value(BackendValue::Register(value)));
                    }
                    _ => todo!(),
                };

                Ok(())
            }
            Node::BoolLiteral(b) => {
                let value = self.builder.ins().iconst(types::I8, if b { 1 } else { 0 });
                self.ctx
                    .values
                    .insert(id, Value::Value(BackendValue::Register(value)));
                Ok(())
            }
            Node::Return(rv) => {
                self.exit_current_block();

                if let Some(rv) = rv {
                    self.compile_id(rv)?;
                    let value = self.id_value(rv);
                    self.builder.ins().return_(&[value]);
                } else {
                    self.builder.ins().return_(&[]);
                }

                Ok(())
            }
            Node::Let { expr, .. } => {
                let size: u32 = self.ctx.id_type_size(id);
                let slot = self.builder.create_sized_stack_slot(StackSlotData {
                    kind: StackSlotKind::ExplicitSlot,
                    size,
                });

                let value = Value::Reference(BackendReference::StackSlot(slot));

                if let Some(expr) = expr {
                    self.compile_id(expr)?;
                    self.store_copy(expr, value);

                    if self.ctx.id_is_aggregate_type(expr) {
                        self.ctx
                            .values
                            .insert(id, Value::Value(BackendValue::Aggregate(slot)));
                    } else {
                        self.ctx
                            .values
                            .insert(id, Value::Reference(BackendReference::StackSlot(slot)));
                    }
                } else {
                    self.ctx
                        .values
                        .insert(id, Value::Reference(BackendReference::StackSlot(slot)));
                }

                Ok(())
            }
            Node::Assign { name, expr, .. } => {
                match self.ctx.nodes[name] {
                    Node::Deref(_value) => {
                        self.ctx.in_assign_lhs = true;
                        self.compile_id(name)?;
                        self.ctx.in_assign_lhs = false;
                    }
                    _ => {
                        self.compile_id(name)?;
                    }
                }

                self.compile_id(expr)?;

                let addr = self.ctx.values[&name];
                self.store_copy(expr, addr);

                Ok(())
            }
            Node::FnDeclParam { index, .. } => {
                let params = self.builder.block_params(self.current_block);
                let param_value = params[index as usize];

                if self.ctx.id_is_aggregate_type(id) {
                    self.ctx.values.insert(
                        id,
                        Value::Value(BackendValue::AggregatePointer(param_value)),
                    );
                } else if self.ctx.id_base_is_aggregate_type(id) {
                    self.ctx
                        .values
                        .insert(id, Value::Value(BackendValue::Register(param_value)));
                } else {
                    self.ctx
                        .values
                        .insert(id, Value::Value(BackendValue::Register(param_value)));
                }

                Ok(())
            }
            Node::ValueParam { value, .. } => {
                self.compile_id(value)?;
                self.ctx.values.insert(id, self.ctx.values[&value]);
                Ok(())
            }
            Node::BinOp { op, lhs, rhs } if op == Op::And => {
                let variable = Variable::new(id.0);
                self.builder.declare_var(variable, types::I8);

                let test_rhs_block = self.builder.create_block();
                let cont_block = self.builder.create_block();

                self.compile_id(lhs)?;
                let lhs_rvalue = self.id_value(lhs);
                self.builder.def_var(variable, lhs_rvalue);
                self.builder
                    .ins()
                    .brif(lhs_rvalue, test_rhs_block, &[], cont_block, &[]);

                self.switch_to_block(test_rhs_block);
                self.compile_id(rhs)?;
                let rhs_rvalue = self.id_value(rhs);
                self.builder.def_var(variable, rhs_rvalue);
                self.builder.ins().jump(cont_block, &[]);

                self.switch_to_block(cont_block);

                let variable_value = self.builder.use_var(variable);
                self.ctx
                    .values
                    .insert(id, Value::Value(BackendValue::Register(variable_value)));

                Ok(())
            }
            Node::BinOp { op, lhs, rhs } if op == Op::Or => {
                let variable = Variable::new(id.0);
                self.builder.declare_var(variable, types::I8);

                let test_rhs_block = self.builder.create_block();
                let cont_block = self.builder.create_block();

                self.compile_id(lhs)?;
                let lhs_rvalue = self.id_value(lhs);
                self.builder.def_var(variable, lhs_rvalue);
                self.builder
                    .ins()
                    .brif(lhs_rvalue, cont_block, &[], test_rhs_block, &[]);

                self.switch_to_block(test_rhs_block);
                self.compile_id(rhs)?;
                let rhs_rvalue = self.id_value(rhs);
                self.builder.def_var(variable, rhs_rvalue);
                self.builder.ins().jump(cont_block, &[]);

                self.switch_to_block(cont_block);

                let variable_value = self.builder.use_var(variable);
                self.ctx
                    .values
                    .insert(id, Value::Value(BackendValue::Register(variable_value)));

                Ok(())
            }
            Node::BinOp { op, lhs, rhs } => {
                self.compile_id(lhs)?;
                self.compile_id(rhs)?;

                let lhs_value = self.get_register_value(lhs);
                let rhs_value = self.get_register_value(rhs);

                let ty = self.ctx.types[&lhs].clone();

                match ty {
                    Type::I8
                    | Type::I16
                    | Type::I32
                    | Type::I64
                    | Type::U8
                    | Type::U16
                    | Type::U32
                    | Type::U64 => {
                        let is_signed = matches!(ty, Type::I8 | Type::I16 | Type::I32 | Type::I64);

                        let value = match op {
                            Op::Add => self.builder.ins().iadd(lhs_value, rhs_value),
                            Op::Sub => self.builder.ins().isub(lhs_value, rhs_value),
                            Op::Mul => self.builder.ins().imul(lhs_value, rhs_value),
                            Op::Div => self.builder.ins().sdiv(lhs_value, rhs_value),
                            Op::EqEq => self.builder.ins().icmp(IntCC::Equal, lhs_value, rhs_value),
                            Op::Neq => {
                                self.builder
                                    .ins()
                                    .icmp(IntCC::NotEqual, lhs_value, rhs_value)
                            }
                            Op::Gt => self.builder.ins().icmp(
                                if is_signed {
                                    IntCC::SignedGreaterThan
                                } else {
                                    IntCC::UnsignedGreaterThan
                                },
                                lhs_value,
                                rhs_value,
                            ),
                            Op::Lt => self.builder.ins().icmp(
                                if is_signed {
                                    IntCC::SignedLessThan
                                } else {
                                    IntCC::UnsignedLessThan
                                },
                                lhs_value,
                                rhs_value,
                            ),
                            Op::GtEq => self.builder.ins().icmp(
                                if is_signed {
                                    IntCC::SignedGreaterThanOrEqual
                                } else {
                                    IntCC::UnsignedGreaterThanOrEqual
                                },
                                lhs_value,
                                rhs_value,
                            ),
                            Op::LtEq => self.builder.ins().icmp(
                                if is_signed {
                                    IntCC::SignedLessThanOrEqual
                                } else {
                                    IntCC::UnsignedLessThanOrEqual
                                },
                                lhs_value,
                                rhs_value,
                            ),
                            Op::And | Op::Or => unreachable!(),
                        };
                        self.ctx
                            .values
                            .insert(id, Value::Value(BackendValue::Register(value)));
                    }
                    a => todo!("BinOp for {:?}", a),
                }

                Ok(())
            }
            Node::Call { func, params, .. } | Node::ThreadingCall { func, params, .. } => {
                self.compile_id_for_call(id, func, params)
            }
            Node::StructLiteral { params, .. } => {
                let size = self.ctx.id_type_size(id);
                let slot = self.builder.create_sized_stack_slot(StackSlotData {
                    kind: StackSlotKind::ExplicitSlot,
                    size,
                });

                let value = Value::Value(BackendValue::Aggregate(slot));

                self.ctx.values.insert(id, value);

                let mut offset = 0;
                for field in params.borrow().iter() {
                    self.compile_id(*field)?;
                    self.store_with_offset(
                        *field,
                        Value::Reference(BackendReference::StackSlot(slot)),
                        offset,
                    );
                    offset += self.ctx.id_type_size(*field);
                }

                Ok(())
            }
            Node::MemberAccess { value, member } => {
                self.compile_id(value)?;

                let Node::Symbol(_) = &self.ctx.nodes[member] else {
                    todo!("Member access for {:?}", &self.ctx.nodes[member])
                };

                let offset = self.get_member_offset(value, member)?;

                let mut pointiness = 0;
                let mut ty = self.ctx.get_type(value);
                while let Type::Pointer(inner) = ty {
                    pointiness += 1;
                    ty = self.ctx.get_type(inner);
                }

                let value = self.ctx.values[&value];
                let mut value = self.as_cranelift_value(value);

                while pointiness > 0 {
                    value = self.load(types::I64, value, 0);
                    pointiness -= 1;
                }

                if offset != 0 {
                    value = self.builder.ins().iadd_imm(value, offset as i64);
                }

                if self.ctx.id_is_aggregate_type(id) {
                    self.ctx
                        .values
                        .insert(id, Value::Value(BackendValue::AggregatePointer(value)));
                } else {
                    self.ctx
                        .values
                        .insert(id, Value::Reference(BackendReference::Pointer(value)));
                }

                Ok(())
            }
            Node::AddressOf(a) => {
                self.compile_id(a)?;

                let value = self.ctx.values.get(&a).unwrap();

                match value {
                    Value::Value(v) => match v {
                        BackendValue::Func(_) | BackendValue::Register(_) => {
                            let slot = self.builder.create_sized_stack_slot(StackSlotData {
                                kind: StackSlotKind::ExplicitSlot,
                                size: self.ctx.id_type_size(a),
                            });

                            let value = self.as_cranelift_value(*value);
                            self.builder.ins().stack_store(value, slot, 0);

                            let slot_addr =
                                self.builder
                                    .ins()
                                    .stack_addr(self.ctx.get_pointer_type(), slot, 0);

                            self.ctx
                                .values
                                .insert(id, Value::Value(BackendValue::Register(slot_addr)));
                        }
                        BackendValue::Aggregate(slot) => {
                            let slot_addr = self.builder.ins().stack_addr(
                                self.ctx.get_pointer_type(),
                                *slot,
                                0,
                            );

                            self.ctx
                                .values
                                .insert(id, Value::Value(BackendValue::Register(slot_addr)));
                        }
                        BackendValue::AggregatePointer(ptr) => {
                            self.ctx
                                .values
                                .insert(id, Value::Value(BackendValue::Register(*ptr)));
                        }
                    },
                    Value::Reference(reference) => match reference {
                        BackendReference::StackSlot(slot) => {
                            let slot_addr = self.builder.ins().stack_addr(
                                self.ctx.get_pointer_type(),
                                *slot,
                                0,
                            );

                            self.ctx
                                .values
                                .insert(id, Value::Value(BackendValue::Register(slot_addr)));
                        }
                        BackendReference::Pointer(ptr) => {
                            self.ctx
                                .values
                                .insert(id, Value::Value(BackendValue::Register(*ptr)));
                        }
                    },
                }
                // self.ctx.values.insert(id, *cranelift_value);

                Ok(())
            }
            Node::Deref(value) => {
                self.compile_id(value)?;

                if self.ctx.in_assign_lhs {
                    let value = self.get_register_value(value);

                    self.ctx
                        .values
                        .insert(id, Value::Reference(BackendReference::Pointer(value)));
                } else {
                    if self.ctx.id_is_aggregate_type(id) {
                        // If we're dereferencing to an aggregate type, then we are going to need a stack slot and will become an Aggregate value ourselves
                        let slot = self.builder.create_sized_stack_slot(StackSlotData {
                            kind: StackSlotKind::ExplicitSlot,
                            size: self.ctx.id_type_size(id),
                        });

                        let size = self.ctx.id_type_size(id);

                        let value = self.get_register_value(value);
                        let slot_addr =
                            self.builder
                                .ins()
                                .stack_addr(self.ctx.get_pointer_type(), slot, 0);

                        self.emit_small_memory_copy(slot_addr, value, size as _);

                        self.ctx
                            .values
                            .insert(id, Value::Value(BackendValue::Aggregate(slot)));
                    } else {
                        let rvalue = self.get_register_value(value);
                        let ty = self.ctx.get_cranelift_type(value);
                        let value = self.load(ty, rvalue, 0);
                        self.ctx
                            .values
                            .insert(id, Value::Value(BackendValue::Register(value)));
                    }
                }

                Ok(())
            }
            Node::Extern {
                name,
                params,
                return_ty,
            } => {
                let mut sig = self.ctx.module.make_signature();

                let return_size = return_ty.map(|rt| self.ctx.id_type_size(rt)).unwrap_or(0);
                if return_size > 0 {
                    sig.returns.push(AbiParam::new(
                        self.ctx.get_cranelift_type(return_ty.unwrap()),
                    ));
                }

                for param in params.borrow().iter() {
                    sig.params
                        .push(AbiParam::new(self.ctx.get_cranelift_type(*param)));
                }

                let name = self.ctx.get_symbol(name);
                let name = self
                    .ctx
                    .string_interner
                    .resolve(name.0)
                    .unwrap()
                    .to_string();

                let func_id = self
                    .ctx
                    .module
                    .declare_function(&name, Linkage::Import, &sig)
                    .unwrap();

                self.ctx
                    .values
                    .insert(id, Value::Value(BackendValue::Func(func_id)));

                Ok(())
            }
            Node::If {
                cond,
                then_block,
                else_block,
            } => {
                self.compile_id(cond)?;
                let cond_value = self.id_value(cond);

                // Make a stack slot for the result of the if statement
                let slot = self.builder.create_sized_stack_slot(StackSlotData {
                    kind: StackSlotKind::ExplicitSlot,
                    size: self.ctx.id_type_size(id),
                });

                let value = if self.ctx.id_is_aggregate_type(id) {
                    Value::Value(BackendValue::Aggregate(slot))
                } else {
                    Value::Reference(BackendReference::StackSlot(slot))
                };

                let saved_resolve_addr = self.resolve_addr;
                self.resolve_addr = Some(value);

                let then_ebb = self.builder.create_block();
                let else_ebb = self.builder.create_block();
                let merge_ebb = self.builder.create_block();

                let saved_resolve_jump_target = self.resolve_jump_target;
                self.resolve_jump_target = Some(merge_ebb);

                self.builder
                    .ins()
                    .brif(cond_value, then_ebb, &[], else_ebb, &[]);
                self.switch_to_block(then_ebb);
                self.compile_id(then_block)?;
                if !self.exited_blocks.contains(&self.current_block) {
                    self.builder.ins().jump(merge_ebb, &[]);
                }

                self.switch_to_block(else_ebb);
                match else_block {
                    crate::NodeElse::Block(b) => {
                        self.compile_id(b)?;
                        if !self.exited_blocks.contains(&self.current_block) {
                            self.builder.ins().jump(merge_ebb, &[]);
                        }
                    }
                    crate::NodeElse::If(else_if) => {
                        self.compile_id(else_if)?;
                        self.store_copy(else_if, self.resolve_addr.unwrap());
                        self.builder.ins().jump(merge_ebb, &[]);
                    }
                    crate::NodeElse::None => {
                        self.builder.ins().jump(merge_ebb, &[]);
                    }
                }

                self.builder.switch_to_block(merge_ebb);

                self.resolve_jump_target = saved_resolve_jump_target;
                self.resolve_addr = saved_resolve_addr;

                Ok(())
            }
            Node::For {
                label,
                iterable,
                block,
            } => {
                self.compile_id(iterable)?;
                let array_value = self.id_value(iterable);
                let array_length_value = self.load(
                    types::I64,
                    array_value,
                    self.ctx.get_pointer_type().bytes() as _,
                );
                let array_pointer_value = self.load(self.ctx.get_pointer_type(), array_value, 0);
                let array_elem_type_size = self.ctx.id_type_size(label);

                // Make a stack slot for the label
                let label_slot = self.builder.create_sized_stack_slot(StackSlotData {
                    kind: StackSlotKind::ExplicitSlot,
                    size: self.ctx.id_type_size(label),
                });
                let label_slot_addr = self.builder.ins().stack_addr(
                    self.ctx.module.isa().pointer_type(),
                    label_slot,
                    0,
                );
                self.ctx
                    .values
                    .insert(label, Value::Value(BackendValue::Register(label_slot_addr)));

                // Index starts at 0
                let index = Variable::new(id.0);
                self.builder.declare_var(index, types::I64);
                let const_zero = self.builder.ins().iconst(types::I64, 0);
                self.builder.def_var(index, const_zero);

                let cond_ebb = self.builder.create_block();
                let block_ebb = self.builder.create_block();
                let merge_ebb = self.builder.create_block();

                self.builder.ins().jump(cond_ebb, &[]);
                self.switch_to_block(cond_ebb);

                // if iterator >= array_length, jump to merge
                let used_index = self.builder.use_var(index);
                let cond = self.builder.ins().icmp(
                    IntCC::SignedGreaterThanOrEqual,
                    used_index,
                    array_length_value,
                );
                self.builder
                    .ins()
                    .brif(cond, merge_ebb, &[], block_ebb, &[]);

                self.switch_to_block(block_ebb);

                // Store the current value in the label slot
                let used_index = self.builder.use_var(index);
                let array_element_offset = self
                    .builder
                    .ins()
                    .imul_imm(used_index, array_elem_type_size as i64);
                let element_ptr = self
                    .builder
                    .ins()
                    .iadd(array_pointer_value, array_element_offset);

                self.emit_small_memory_copy(
                    label_slot_addr,
                    element_ptr,
                    array_elem_type_size as _,
                );

                self.compile_id(block)?;

                // increment iterator
                let used_index = self.builder.use_var(index);
                let incremented = self.builder.ins().iadd_imm(used_index, 1);
                self.builder.def_var(index, incremented);

                if !self.exited_blocks.contains(&self.current_block) {
                    self.builder.ins().jump(cond_ebb, &[]);
                }

                self.builder.switch_to_block(merge_ebb);

                Ok(())
            }
            Node::While { cond, block } => {
                let cond_block = self.builder.create_block();
                let while_block = self.builder.create_block();
                let merge_block = self.builder.create_block();

                self.builder.ins().jump(cond_block, &[]);
                self.switch_to_block(cond_block);
                self.compile_id(cond)?;
                let cond_value = self.id_value(cond);
                self.builder
                    .ins()
                    .brif(cond_value, while_block, &[], merge_block, &[]);

                self.switch_to_block(while_block);
                self.compile_id(block)?;
                if !self.exited_blocks.contains(&self.current_block) {
                    self.builder.ins().jump(cond_block, &[]);
                }

                self.switch_to_block(merge_block);

                Ok(())
            }
            Node::Block {
                stmts,
                is_standalone,
                ..
            } => {
                if is_standalone {
                    // Make a stack slot for the result of the block statement
                    let slot = self.builder.create_sized_stack_slot(StackSlotData {
                        kind: StackSlotKind::ExplicitSlot,
                        size: self.ctx.id_type_size(id),
                    });

                    let value = if self.ctx.id_is_aggregate_type(id) {
                        Value::Value(BackendValue::Aggregate(slot))
                    } else {
                        Value::Reference(BackendReference::StackSlot(slot))
                    };
                    self.ctx.values.insert(id, value);

                    let saved_resolve_addr = self.resolve_addr;
                    self.resolve_addr = Some(value);

                    let merge_ebb = self.builder.create_block();

                    let saved_resolve_jump_target = self.resolve_jump_target;
                    self.resolve_jump_target = Some(merge_ebb);
                    self.compile_ids(stmts)?;
                    if !self.exited_blocks.contains(&self.current_block) {
                        self.builder.ins().jump(merge_ebb, &[]);
                    }

                    self.switch_to_block(merge_ebb);

                    self.resolve_addr = saved_resolve_addr;
                    self.resolve_jump_target = saved_resolve_jump_target;
                } else {
                    self.compile_ids(stmts)?;
                }

                Ok(())
            }
            Node::Resolve(r) => {
                if let Some(r) = r {
                    self.compile_id(r)?;
                    self.store_copy(r, self.resolve_addr.unwrap());
                }

                self.builder
                    .ins()
                    .jump(self.resolve_jump_target.unwrap(), &[]);

                self.exit_current_block();

                Ok(())
            }
            Node::ArrayLiteral { members, ty } => {
                let member_size = self.ctx.id_type_size(ty);

                // create stack slot big enough for all of the members
                let member_storage_slot = self.builder.create_sized_stack_slot(StackSlotData {
                    kind: StackSlotKind::ExplicitSlot,
                    size: member_size * members.borrow().len() as u32,
                });

                let mut offset = 0;
                for &member in members.borrow().iter() {
                    self.compile_id(member)?;
                    let member_value = self.as_cranelift_value(self.ctx.values[&member]);
                    self.builder.ins().stack_store(
                        member_value,
                        member_storage_slot,
                        offset as i32,
                    );
                    offset += member_size;
                }

                self.ctx.values.insert(
                    id,
                    Value::Value(BackendValue::Aggregate(member_storage_slot)),
                );

                Ok(())
            }
            Node::ArrayAccess { array, index } => {
                self.compile_id(array)?;
                self.compile_id(index)?;

                let array_ty = self.ctx.get_type(array);

                let member_size = self.ctx.id_type_size(id);
                let array_value = self.id_value(array);
                let index_value = self.id_value(index);

                let array_ptr_value = if let Type::Array(_, ArrayLen::Some(_)) = array_ty {
                    array_value
                } else {
                    self.load(types::I64, array_value, 0)
                };

                let index_value = self.builder.ins().imul_imm(index_value, member_size as i64);
                let element_ptr = self.builder.ins().iadd(array_ptr_value, index_value);

                if self.ctx.id_is_aggregate_type(id) {
                    self.ctx.values.insert(
                        id,
                        Value::Value(BackendValue::AggregatePointer(element_ptr)),
                    );
                } else {
                    self.ctx
                        .values
                        .insert(id, Value::Value(BackendValue::Register(element_ptr)));
                }

                Ok(())
            }
            Node::StringLiteral(sym) => {
                // todo(chad): store this length somewhere the first time we look it up, so we don't have to look it up again just to get the length
                let sym_str = self.ctx.string_interner.resolve(sym.0).unwrap();

                let slot = self.builder.create_sized_stack_slot(StackSlotData {
                    kind: StackSlotKind::ExplicitSlot,
                    size: self.ctx.id_type_size(id),
                });

                let gsp = self.global_str_ptr.get_or_insert_with(|| {
                    self.ctx.module.declare_data_in_func(
                        self.ctx.string_literal_data_id.unwrap(),
                        self.builder.func,
                    )
                });

                let offset = *self.ctx.string_literal_offsets.get(&id).unwrap();
                let offset_str_ptr = if offset == 0 {
                    *gsp
                } else {
                    self.builder.create_global_value(GlobalValueData::IAddImm {
                        base: *gsp,
                        offset: Imm64::new(offset as _),
                        global_type: types::I64,
                    })
                };
                let offset_str_ptr = self.builder.ins().global_value(types::I64, offset_str_ptr);

                let len = self.builder.ins().iconst(types::I64, sym_str.len() as i64);
                self.builder.ins().stack_store(offset_str_ptr, slot, 0);
                self.builder.ins().stack_store(len, slot, 8);

                self.ctx
                    .values
                    .insert(id, Value::Value(BackendValue::Aggregate(slot)));

                Ok(())
            }
            Node::Cast { value, .. } => {
                self.compile_id(value)?;
                self.ctx.values.insert(id, self.ctx.values[&value]);
                Ok(())
            }
            Node::SizeOf(ty) => {
                let ty_size = self.ctx.id_type_size(ty);
                let value = self.builder.ins().iconst(types::I64, ty_size as i64);

                self.ctx
                    .values
                    .insert(id, Value::Value(BackendValue::Register(value)));

                Ok(())
            }
            Node::PolySpecialize {
                copied: Some(copied),
                ..
            } => {
                self.compile_id(copied)?;
                self.ctx
                    .values
                    .get(&copied)
                    .cloned()
                    .map(|v| self.ctx.values.insert(id, v));

                Ok(())
            }
            Node::StaticMemberAccess { value, .. } => {
                let value_ty = self.ctx.get_type(value);
                let Type::Array(_, ArrayLen::Some(len)) = value_ty else {
                    unreachable!()
                };

                let value = self.builder.ins().iconst(types::I64, len as i64);

                self.ctx
                    .values
                    .insert(id, Value::Value(BackendValue::Register(value)));

                Ok(())
            }
            Node::AsCast { value, style, .. } => match style {
                AsCastStyle::StaticToDynamicArray => {
                    self.compile_id(value)?;
                    let cranelift_value = self.id_value(value);

                    // Make a stack slot for the dynamic array
                    let slot = self.builder.create_sized_stack_slot(StackSlotData {
                        kind: StackSlotKind::ExplicitSlot,
                        size: self.ctx.id_type_size(id),
                    });

                    // Store the data pointer at offset 0
                    self.builder.ins().stack_store(cranelift_value, slot, 0);

                    // Store the length at offset 8
                    let Type::Array(_, ArrayLen::Some(array_len)) = self.ctx.get_type(value) else {
                        unreachable!()
                    };
                    let array_len = self.builder.ins().iconst(types::I64, array_len as i64);
                    self.builder.ins().stack_store(array_len, slot, 8);

                    self.ctx
                        .values
                        .insert(id, Value::Value(BackendValue::Aggregate(slot)));

                    Ok(())
                }
                AsCastStyle::StructToDynamicArray => {
                    self.compile_id(value)?;
                    self.ctx.values.insert(id, self.ctx.values[&value]);
                    Ok(())
                }
                _ => unreachable!(),
            },
            Node::FnDefinition { .. }
            | Node::StructDefinition { .. }
            | Node::EnumDefinition { .. } => Ok(()), // This should have already been handled by the toplevel context
            _ => todo!("compile_id for {:?}", &self.ctx.nodes[id]),
        }
    }

    #[instrument(skip_all)]
    fn compile_ids(&mut self, ids: IdVec) -> DraftResult<()> {
        for id in ids.borrow().iter() {
            self.compile_id(*id)?;
        }

        Ok(())
    }

    #[instrument(skip_all)]
    fn compile_id_for_call(&mut self, id: NodeId, func: NodeId, params: IdVec) -> DraftResult<()> {
        self.compile_id(func)?;

        let param_ids = params.clone();
        let mut param_values = Vec::new();
        for &param in param_ids.borrow().iter() {
            self.compile_id(param)?;
            param_values.push(self.get_register_value(param));
        }

        // direct call?
        let call_inst =
            if let Some(Value::Value(BackendValue::Func(func_id))) = self.ctx.values.get(&func) {
                let func_ref = *self.declared_func_ids.entry(*func_id).or_insert_with(|| {
                    self.ctx
                        .module
                        .declare_func_in_func(*func_id, self.builder.func)
                });
                self.builder.ins().call(func_ref, &param_values)
            } else {
                let sig_ref = self
                    .ctx
                    .get_func_signature(func, param_ids.borrow().as_ref());
                let sig = self.builder.import_signature(sig_ref);

                let callee = self.id_value(func);

                self.builder.ins().call_indirect(sig, callee, &param_values)
            };

        if self
            .ctx
            .types
            .get(&id)
            .map(|t| !matches!(t, Type::Infer(_)))
            .unwrap_or_default()
        {
            let value = self.builder.func.dfg.first_result(call_inst);

            if self.ctx.id_is_aggregate_type(id) {
                self.ctx
                    .values
                    .insert(id, Value::Value(BackendValue::AggregatePointer(value)));
            } else {
                self.ctx
                    .values
                    .insert(id, Value::Value(BackendValue::Register(value)));
            }
        }

        Ok(())
    }

    #[instrument(skip_all)]
    fn switch_to_block(&mut self, block: Block) {
        self.builder.switch_to_block(block);
        self.current_block = block;
    }

    #[instrument(skip_all)]
    fn exit_current_block(&mut self) {
        self.exited_blocks.insert(self.current_block);
    }

    #[instrument(skip_all)]
    fn load(&mut self, ty: types::Type, value: CraneliftValue, offset: i32) -> CraneliftValue {
        self.builder.ins().load(ty, MemFlags::new(), value, offset)
    }

    #[instrument(skip_all)]
    fn get_member_offset(&mut self, value: NodeId, member: NodeId) -> DraftResult<u32> {
        let member_name = self.ctx.nodes[member].as_symbol().unwrap();

        let mut ty = self.ctx.get_type(value);
        while let Type::Pointer(inner) = ty {
            ty = self.ctx.get_type(inner);
        }

        if let Type::Array(_, _) | Type::String = ty {
            let data_sym = self.ctx.string_interner.get_or_intern("data");
            let len_sym = self.ctx.string_interner.get_or_intern("len");
            if member_name.0 == data_sym {
                return Ok(0);
            } else if member_name.0 == len_sym {
                return Ok(self.ctx.get_pointer_type().bytes());
            } else {
                panic!()
            }
        }

        let Type::Struct { params, .. } = ty else {
            panic!("Not a struct: {:?}", ty);
        };

        let fields = params.clone();

        let mut offset = 0;
        for field in fields.borrow().iter() {
            let field_name = match &self.ctx.nodes[field] {
                Node::StructDeclParam { name, .. } => *name,
                Node::ValueParam {
                    name: Some(name), ..
                } => *name,
                _ => panic!("Not a param: {:?}", &self.ctx.nodes[field]),
            };
            let field_name = self.ctx.nodes[field_name].as_symbol().unwrap();

            if field_name == member_name {
                return Ok(offset);
            }

            offset += self.ctx.id_type_size(*field);
        }

        return Err(CompileError::Generic(
            "Member not found".to_string(),
            self.ctx.ranges[value],
        ));
    }

    #[instrument(skip_all)]
    fn as_cranelift_value(&mut self, value: Value) -> CraneliftValue {
        match value {
            Value::Value(BackendValue::Register(val) | BackendValue::AggregatePointer(val)) => val,
            Value::Value(BackendValue::Func(func_id)) => {
                let func_ref = *self.declared_func_ids.entry(func_id).or_insert_with(|| {
                    self.ctx
                        .module
                        .declare_func_in_func(func_id, self.builder.func)
                });

                self.builder
                    .ins()
                    .func_addr(self.ctx.get_pointer_type(), func_ref)
            }
            Value::Value(BackendValue::Aggregate(slot)) => {
                self.builder
                    .ins()
                    .stack_addr(self.ctx.get_pointer_type(), slot, 0)
            }
            Value::Reference(reference) => match reference {
                BackendReference::StackSlot(slot) => {
                    self.builder
                        .ins()
                        .stack_addr(self.ctx.get_pointer_type(), slot, 0)
                }
                BackendReference::Pointer(ptr) => ptr,
            },
        }
    }

    #[instrument(skip_all)]
    fn store_with_offset(&mut self, id: NodeId, dest: Value, offset: u32) {
        let size = self.ctx.id_type_size(id) as u64;

        let source = self.ctx.values[&id];

        match (source, dest) {
            (Value::Value(value), Value::Reference(reference)) => match (value, reference) {
                (
                    BackendValue::Func(_) | BackendValue::Register(_),
                    BackendReference::StackSlot(ss),
                ) => {
                    let source_value = self.as_cranelift_value(source);
                    self.builder
                        .ins()
                        .stack_store(source_value, ss, offset as i32);
                }
                (BackendValue::Aggregate(source_slot), BackendReference::StackSlot(dest_slot)) => {
                    let source_ptr = self.builder.ins().stack_addr(
                        self.ctx.module.isa().pointer_type(),
                        source_slot,
                        offset as i32,
                    );
                    let dest_ptr = self.builder.ins().stack_addr(
                        self.ctx.module.isa().pointer_type(),
                        dest_slot,
                        offset as i32,
                    );
                    self.emit_small_memory_copy(dest_ptr, source_ptr, size);
                }
                (
                    BackendValue::AggregatePointer(source_ptr),
                    BackendReference::StackSlot(dest_slot),
                ) => {
                    let dest_ptr = self.builder.ins().stack_addr(
                        self.ctx.module.isa().pointer_type(),
                        dest_slot,
                        offset as i32,
                    );
                    self.emit_small_memory_copy(dest_ptr, source_ptr, size);
                }
                (
                    BackendValue::Func(_) | BackendValue::Register(_),
                    BackendReference::Pointer(dest_ptr),
                ) => {
                    let source_value = self.as_cranelift_value(source);
                    self.builder.ins().store(
                        MemFlags::new(),
                        source_value,
                        dest_ptr,
                        offset as i32,
                    );
                }
                (BackendValue::Aggregate(source_slot), BackendReference::Pointer(dest_ptr)) => {
                    let source_ptr = self.builder.ins().stack_addr(
                        self.ctx.module.isa().pointer_type(),
                        source_slot,
                        offset as i32,
                    );
                    self.emit_small_memory_copy(dest_ptr, source_ptr, size);
                }
                (BackendValue::AggregatePointer(src_ptr), BackendReference::Pointer(dest_ptr)) => {
                    self.emit_small_memory_copy(dest_ptr, src_ptr, size);
                }
            },
            (Value::Reference(source_reference), Value::Reference(dest_reference)) => {
                let source_ptr = match source_reference {
                    BackendReference::Pointer(p) => p,
                    BackendReference::StackSlot(slot) => self.builder.ins().stack_addr(
                        self.ctx.module.isa().pointer_type(),
                        slot,
                        offset as i32,
                    ),
                };

                let dest_ptr = match dest_reference {
                    BackendReference::Pointer(p) => p,
                    BackendReference::StackSlot(slot) => self.builder.ins().stack_addr(
                        self.ctx.module.isa().pointer_type(),
                        slot,
                        offset as i32,
                    ),
                };

                self.emit_small_memory_copy(dest_ptr, source_ptr, size);
            }
            (_, Value::Value(_)) => panic!("Cannot store into a value"),
        }
    }

    #[instrument(skip_all)]
    fn store_copy(&mut self, id: NodeId, dest: Value) {
        self.store_with_offset(id, dest, 0);
    }

    #[instrument(skip_all)]
    fn emit_small_memory_copy(
        &mut self,
        dest_value: CraneliftValue,
        source_value: CraneliftValue,
        size: u64,
    ) {
        self.builder.emit_small_memory_copy(
            self.ctx.module.isa().frontend_config(),
            dest_value,
            source_value,
            size as _,
            1,
            1,
            true, // non-overlapping
            MemFlags::new(),
        );
    }

    #[instrument(skip_all)]
    fn id_value(&mut self, id: NodeId) -> CraneliftValue {
        self.as_cranelift_value(self.ctx.values[&id])
    }

    #[instrument(skip_all)]
    fn get_register_value(&mut self, id: NodeId) -> CraneliftValue {
        let value = self.ctx.values[&id];
        match value {
            Value::Value(v) => match v {
                BackendValue::Func(_) => self.as_cranelift_value(value),
                BackendValue::Aggregate(ss) => {
                    self.builder
                        .ins()
                        .stack_addr(self.ctx.get_pointer_type(), ss, 0)
                }
                BackendValue::AggregatePointer(r) | BackendValue::Register(r) => r,
            },
            Value::Reference(reference) => {
                let ty = match self.ctx.get_type(id) {
                    Type::Array(_, _) | Type::Struct { .. } | Type::String | Type::Enum { .. } => {
                        todo!()
                    }
                    Type::Pointer(_) => self.ctx.get_pointer_type(),
                    Type::F32 => types::F32,
                    Type::F64 => types::F64,
                    Type::I8 | Type::Bool => types::I8,
                    Type::I16 => types::I16,
                    Type::I32 => types::I32,
                    Type::I64 => types::I64,
                    ty => todo!("Unexpected type for register value: {:?}", ty),
                };
                match reference {
                    BackendReference::StackSlot(ss) => self.builder.ins().stack_load(ty, ss, 0),
                    BackendReference::Pointer(p) => {
                        self.builder.ins().load(ty, MemFlags::new(), p, 0)
                    }
                }
            }
        }
    }
}

impl<'a> ToplevelCompileContext<'a> {
    pub fn compile_toplevel_id(&mut self, id: NodeId) -> DraftResult<()> {
        // idempotency
        match self.ctx.values.get(&id) {
            None | Some(Value::Value(BackendValue::Func(_))) => {}
            _ => return Ok(()),
        };

        match self.ctx.nodes[id].clone() {
            Node::Symbol(_) => todo!(),
            Node::FnDefinition {
                name,
                stmts,
                params,
                return_ty,
                ..
            } => {
                let name_str = self.ctx.mangled_func_name(name, id);

                let mut sig = self.ctx.module.make_signature();

                for &param in params.borrow().iter() {
                    sig.params
                        .push(AbiParam::new(self.ctx.get_cranelift_type(param)));
                }

                if let Some(return_ty) = return_ty {
                    sig.returns
                        .push(AbiParam::new(self.ctx.get_cranelift_type(return_ty)));
                }

                let func_id = self
                    .ctx
                    .module
                    .declare_function(&name_str, Linkage::Export, &sig)
                    .unwrap();

                let mut builder = FunctionBuilder::new(&mut self.codegen_ctx.func, self.func_ctx);
                builder.func.signature = sig;
                builder.func.collect_debug_info();

                self.ctx
                    .values
                    .insert(id, Value::Value(BackendValue::Func(func_id)));

                let ebb = builder.create_block();
                builder.append_block_params_for_function_params(ebb);
                builder.switch_to_block(ebb);

                let mut builder_ctx = FunctionCompileContext {
                    ctx: self.ctx,
                    builder,
                    exited_blocks: Default::default(),
                    current_block: ebb,
                    resolve_addr: None,
                    resolve_jump_target: None,
                    declared_func_ids: Default::default(),
                    global_str_ptr: None,
                };

                for &stmt in stmts.borrow().iter() {
                    builder_ctx.compile_id(stmt)?;
                }

                builder_ctx.builder.seal_all_blocks();
                builder_ctx.builder.finalize();

                if self.ctx.args.dump_ir {
                    println!("{}", self.codegen_ctx.func.display());
                }

                self.ctx
                    .module
                    .define_function(func_id, self.codegen_ctx)
                    .unwrap();

                self.ctx.module.clear_context(self.codegen_ctx);

                Ok(())
            }
            Node::StaticMemberAccess { resolved, .. } => {
                let resolved = resolved.unwrap();

                match resolved {
                    StaticMemberResolution::EnumConstructor { base, index } => {
                        // We need to generate a function which takes one or zero parameters, and outputs an enum of this type
                        let Type::Enum { params, .. } = self.ctx.types[&base].clone() else {
                            todo!("{:?}", self.ctx.nodes[base])
                        };
                        let param = params.borrow()[index as usize];
                        let Node::EnumDeclParam { name, ty } = self.ctx.nodes[param] else {
                            todo!()
                        };

                        let name_str = format!(
                            "__enum_constructor_{}_{}__{}",
                            self.ctx
                                .string_interner
                                .resolve(self.ctx.get_symbol(name).0)
                                .unwrap(),
                            index,
                            id.0
                        );

                        let mut sig = self.ctx.module.make_signature();

                        if let Some(ty) = ty {
                            sig.params
                                .push(AbiParam::new(self.ctx.get_cranelift_type(ty)));
                        }

                        sig.returns
                            .push(AbiParam::new(self.ctx.get_cranelift_type(base)));

                        let func_id = self
                            .ctx
                            .module
                            .declare_function(&name_str, Linkage::Export, &sig)
                            .unwrap();

                        let mut builder =
                            FunctionBuilder::new(&mut self.codegen_ctx.func, self.func_ctx);
                        builder.func.signature = sig;

                        self.ctx
                            .values
                            .insert(id, Value::Value(BackendValue::Func(func_id)));

                        let ebb = builder.create_block();
                        builder.append_block_params_for_function_params(ebb);
                        builder.switch_to_block(ebb);

                        let enum_type_size = self.ctx.id_type_size(base);

                        // Allocate a slot as big as the enum
                        let slot = builder.create_sized_stack_slot(StackSlotData {
                            kind: StackSlotKind::ExplicitSlot,
                            size: enum_type_size,
                        });
                        let slot_addr =
                            builder
                                .ins()
                                .stack_addr(self.ctx.module.isa().pointer_type(), slot, 0);

                        // Assign the enum tag to the first 2 bytes
                        let tag = builder.ins().iconst(types::I16, index as i64);
                        builder.ins().store(MemFlags::new(), tag, slot_addr, 0);

                        if ty.is_some() {
                            // Assign the value to the rest of however many bytes it needs
                            // builder.ins().store(MemFlags::new(), ???, slot_addr, 2);
                            let params = builder.block_params(ebb);
                            let param_value = params[0];

                            if self.ctx.id_is_aggregate_type(param) {
                                let size = self.ctx.id_type_size(param);

                                let source_value = param_value;

                                let dest_value = slot_addr;
                                let dest_value = builder.ins().iadd_imm(dest_value, 2);

                                builder.emit_small_memory_copy(
                                    self.ctx.module.isa().frontend_config(),
                                    dest_value,
                                    source_value,
                                    size as _,
                                    1,
                                    1,
                                    true, // non-overlapping
                                    MemFlags::new(),
                                );
                            } else {
                                builder
                                    .ins()
                                    .store(MemFlags::new(), param_value, slot_addr, 2);
                            }
                        }

                        builder.ins().return_(&[slot_addr]);

                        builder.seal_all_blocks();
                        builder.finalize();

                        if self.ctx.args.dump_ir {
                            println!("{}", self.codegen_ctx.func.display());
                        }

                        self.ctx
                            .module
                            .define_function(func_id, self.codegen_ctx)
                            .unwrap();

                        self.ctx.module.clear_context(self.codegen_ctx);

                        Ok(())
                    }
                    _ => todo!(),
                }
            }
            _ => todo!(),
        }
    }
}
