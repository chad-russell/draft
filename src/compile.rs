use std::collections::HashSet;

use cranelift_codegen::ir::condcodes::IntCC;
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};

use cranelift_codegen::ir::{
    stackslot::StackSize, types, AbiParam, Block, InstBuilder, MemFlags, Signature, StackSlotData,
    StackSlotKind, Type as CraneliftType, Value as CraneliftValue,
};
use cranelift_codegen::settings::{self, Configurable};
use cranelift_codegen::Context as CodegenContext;
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{default_libcall_names, FuncId, Linkage, Module};

use crate::{CompileError, Context, IdVec, Node, NodeId, Op, StaticMemberResolution, Type};

#[derive(Clone, Copy, Debug)]
pub enum Value {
    Func(FuncId),
    Value(CraneliftValue),
}

impl Context {
    pub fn make_module() -> JITModule {
        let mut flags_builder = settings::builder();
        flags_builder.set("is_pic", "false").unwrap();
        // flags_builder.set("enable_verifier", "false").unwrap();
        flags_builder.set("opt_level", "none").unwrap();
        flags_builder.set("enable_probestack", "false").unwrap();

        let isa_builder = cranelift_native::builder().unwrap_or_else(|msg| {
            panic!("host machine is not supported: {}", msg);
        });
        let isa = isa_builder
            .finish(settings::Flags::new(flags_builder))
            .unwrap();

        let mut jit_builder = JITBuilder::with_isa(isa, default_libcall_names());

        // no hot swapping for now
        jit_builder.hotswap(false);

        // jit_builder.symbol("__panic", panic_helper as *const u8);
        // jit_builder.symbol("__dbg_poke", dbg_poke as *const u8);
        // jit_builder.symbol("__dbg_repr_internal", dbg_repr_internal as *const u8);
        // jit_builder.symbol("print_i8", print_i8 as *const u8);
        // jit_builder.symbol("print_i16", print_i16 as *const u8);
        // jit_builder.symbol("print_i32", print_i32 as *const u8);
        jit_builder.symbol("print_i64", print_i64 as *const u8);
        // jit_builder.symbol("print_u8", print_u8 as *const u8);
        // jit_builder.symbol("print_u16", print_u16 as *const u8);
        // jit_builder.symbol("print_u32", print_u32 as *const u8);
        // jit_builder.symbol("print_u64", print_u64 as *const u8);
        // jit_builder.symbol("print_f32", print_f32 as *const u8);
        jit_builder.symbol("print_f64", print_f64 as *const u8);
        // jit_builder.symbol("print_string", print_string as *const u8);
        // jit_builder.symbol("alloc", libc::malloc as *const u8);
        // jit_builder.symbol("realloc", libc::realloc as *const u8);
        // jit_builder.symbol("debug_data", &semantic as *const _ as *const u8);
        jit_builder.symbol("print_enum_tag", print_enum_tag as *const u8);

        JITModule::new(jit_builder)
    }

    pub fn predeclare_functions(&mut self) -> Result<(), CompileError> {
        let mut codegen_ctx = self.module.make_context();
        let mut func_ctx = FunctionBuilderContext::new();

        for &t in self.topo.clone().iter() {
            if !self.completes.contains(&t) {
                continue;
            }

            self.predeclare_function(t)?;
        }

        let mut compile_ctx = ToplevelCompileContext {
            ctx: self,
            codegen_ctx: &mut codegen_ctx,
            func_ctx: &mut func_ctx,
        };

        for &t in compile_ctx.ctx.topo.clone().iter() {
            compile_ctx.compile_toplevel_id(t)?;
        }

        self.module
            .finalize_definitions()
            .expect("Failed to finalize definitions");

        Ok(())
    }

    pub fn predeclare_function(&mut self, id: NodeId) -> Result<(), CompileError> {
        if let Node::FnDefinition {
            name,
            params,
            return_ty,
            ..
        } = self.nodes[id]
        {
            let mut sig = self.module.make_signature();

            for &param in self.id_vecs[params].borrow().iter() {
                sig.params
                    .push(AbiParam::new(self.get_cranelift_type(param)));
            }

            let return_size = return_ty.map(|rt| self.get_type_size(rt)).unwrap_or(0);

            if return_size > 0 {
                sig.returns
                    .push(AbiParam::new(self.get_cranelift_type(return_ty.unwrap())));
            }

            let func_name = self.func_name(name, id);

            let func = self
                .module
                .declare_function(&func_name, Linkage::Local, &sig)
                .unwrap();

            self.values.insert(id, Value::Func(func));
        }

        Ok(())
    }

    fn func_name(&self, name: Option<NodeId>, anonymous_id: NodeId) -> String {
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

    pub fn call_fn(&mut self, fn_name: &str) -> Result<(), CompileError> {
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

        if let Some(id) = node_id {
            let code = self.module.get_finalized_function(self.get_func_id(id));
            let func = unsafe { std::mem::transmute::<_, fn() -> i64>(code) };
            func();
        }

        Ok(())
    }

    fn get_cranelift_type(&self, param: NodeId) -> CraneliftType {
        match self.types[&param] {
            Type::I8 | Type::Bool => types::I8,
            Type::I16 | Type::U16 => types::I16,
            Type::I32 | Type::U32 => types::I32,
            Type::I64 | Type::U64 => types::I64,
            Type::F32 => types::F32,
            Type::F64 => types::F64,
            Type::Pointer(_) | Type::Func { .. } | Type::Struct { .. } | Type::Enum { .. } => {
                self.get_pointer_type()
            }
            Type::Infer(_) => todo!(),
            _ => todo!("{:?}", &self.types[&param]),
        }
    }

    fn get_pointer_type(&self) -> CraneliftType {
        self.module.isa().pointer_type()
    }

    fn enum_tag_size(&self) -> u32 {
        std::mem::size_of::<u16>() as u32
    }

    fn get_type_size(&self, param: NodeId) -> StackSize {
        match self.types[&param] {
            Type::I8 | Type::U8 | Type::Bool => 1,
            Type::I16 | Type::U16 => 2,
            Type::I32 | Type::U32 | Type::F32 => 4,
            Type::I64 | Type::U64 | Type::F64 => 8,
            Type::Func { .. } => 8,
            Type::Pointer(_) => 8,
            Type::Struct { params, .. } => {
                // todo(chad): c struct packing rules if annotated
                self.id_vecs[params]
                    .clone()
                    .borrow()
                    .iter()
                    .map(|f| self.get_type_size(*f))
                    .sum()
            }
            Type::Enum { params, .. } => {
                self.id_vecs[params]
                    .clone()
                    .borrow()
                    .iter()
                    .map(|f| self.get_type_size(*f))
                    .max()
                    .unwrap_or_default()
                    + self.enum_tag_size()
            }
            Type::Array(_) => 16,
            Type::EnumNoneType | Type::Empty => 0,
            _ => todo!("get_type_size for {:?}", self.types[&param]),
        }
    }

    pub fn get_func_id(&self, id: NodeId) -> FuncId {
        match self.values[&id] {
            Value::Func(f) => f,
            _ => panic!("Not a function"),
        }
    }

    pub fn get_func_signature(&self, func: NodeId, param_ids: &Vec<NodeId>) -> Signature {
        let func_ty = self.get_type(func);
        let return_ty = match func_ty {
            Type::Func {
                return_ty: Some(return_ty),
                ..
            } => Some(return_ty),
            _ => None,
        };

        let return_size = return_ty.map(|rt| self.get_type_size(rt)).unwrap_or(0);

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

    pub fn should_pass_by_ref(&self, id: NodeId) -> bool {
        matches!(
            self.get_type(id),
            Type::Struct { .. } | Type::Enum { .. } | Type::Array(_)
        )
    }

    pub fn should_pass_base_by_ref(&self, id: NodeId) -> bool {
        let mut base_ty = self.get_type(id);
        while let Type::Pointer(inner) = base_ty {
            base_ty = self.get_type(inner);
        }

        matches!(base_ty, Type::Struct { .. } | Type::Enum { .. })
    }
}

pub fn print_i64(n: u64) {
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
}

impl<'a> FunctionCompileContext<'a> {
    pub fn compile_id(&mut self, id: NodeId) -> Result<(), CompileError> {
        // idempotency
        match self.ctx.values.get(&id) {
            None => {}
            _ => return Ok(()),
        };

        match self.ctx.nodes[id] {
            Node::Symbol(sym) => {
                let resolved = self.ctx.scope_get(sym, id);

                match resolved {
                    Some(res) => {
                        if self.ctx.addressable_nodes.contains(&res) {
                            self.ctx.addressable_nodes.insert(id);
                        }

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
                        self.ctx.values.insert(id, Value::Value(value));
                    }
                    Type::I32 | Type::U32 => {
                        let value = self.builder.ins().iconst(types::I32, n);
                        self.ctx.values.insert(id, Value::Value(value));
                    }
                    _ => todo!(),
                };

                self.store_in_slot_if_addressable(id);

                Ok(())
            }
            Node::FloatLiteral(n, _) => {
                match self.ctx.types[&id] {
                    Type::F64 => {
                        let value = self.builder.ins().f64const(n);
                        self.ctx.values.insert(id, Value::Value(value));
                    }
                    Type::F32 => {
                        let value = self.builder.ins().f32const(n as f32);
                        self.ctx.values.insert(id, Value::Value(value));
                    }
                    _ => todo!(),
                };

                self.store_in_slot_if_addressable(id);

                Ok(())
            }
            Node::BoolLiteral(b) => {
                let value = self.builder.ins().iconst(types::I8, if b { 1 } else { 0 });
                self.ctx.values.insert(id, Value::Value(value));
                self.store_in_slot_if_addressable(id);
                Ok(())
            }
            Node::Return(rv) => {
                self.exit_current_block();

                if let Some(rv) = rv {
                    self.compile_id(rv)?;
                    let value = self.rvalue(rv);
                    self.builder.ins().return_(&[value]);
                } else {
                    self.builder.ins().return_(&[]);
                }

                Ok(())
            }
            Node::Let { expr, .. } => {
                let size: u32 = self.ctx.get_type_size(id);
                let slot = self.builder.create_sized_stack_slot(StackSlotData {
                    kind: StackSlotKind::ExplicitSlot,
                    size,
                });

                let slot_addr =
                    self.builder
                        .ins()
                        .stack_addr(self.ctx.module.isa().pointer_type(), slot, 0);
                let value = Value::Value(slot_addr);

                if let Some(expr) = expr {
                    self.compile_id(expr)?;
                    self.store(expr, value);
                }

                self.ctx.values.insert(id, value);

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

                let addr = self.ctx.values[&name];
                self.compile_id(expr)?;
                self.store(expr, addr);

                Ok(())
            }
            Node::FnDeclParam { index, .. } => {
                let params = self.builder.block_params(self.current_block);
                let param_value = params[index as usize];

                let pass_base_by_ref = self.ctx.should_pass_base_by_ref(id);

                if self.ctx.addressable_nodes.contains(&id) || pass_base_by_ref {
                    // we need our own storage
                    let size = self.ctx.get_type_size(id);
                    let slot = self.builder.create_sized_stack_slot(StackSlotData {
                        kind: StackSlotKind::ExplicitSlot,
                        size,
                    });

                    let slot_addr = self.builder.ins().stack_addr(
                        self.ctx.module.isa().pointer_type(),
                        slot,
                        0,
                    );
                    let value = Value::Value(slot_addr);
                    self.ctx.values.insert(id, value);

                    let size = self.ctx.get_type_size(id);

                    let source_value = param_value;
                    let dest_value = slot_addr;

                    let is_ptr = matches!(self.ctx.get_type(id), Type::Pointer(_));
                    if pass_base_by_ref && is_ptr {
                        self.builder
                            .ins()
                            .store(MemFlags::new(), source_value, dest_value, 0);
                    } else {
                        self.emit_small_memory_copy(dest_value, source_value, size as _);
                    }
                } else {
                    self.ctx.values.insert(id, Value::Value(param_value));
                }

                Ok(())
            }
            Node::ValueParam { value, .. } => {
                self.compile_id(value)?;
                self.ctx.values.insert(id, self.ctx.values[&value]);
                Ok(())
            }
            Node::BinOp { op, lhs, rhs } => {
                self.compile_id(lhs)?;
                self.compile_id(rhs)?;

                let lhs_value = self.rvalue(lhs);
                let rhs_value = self.rvalue(rhs);

                match self.ctx.types[&lhs] {
                    Type::I8
                    | Type::I16
                    | Type::I32
                    | Type::I64
                    | Type::U8
                    | Type::U16
                    | Type::U32
                    | Type::U64 => {
                        let value = match op {
                            Op::Add => self.builder.ins().iadd(lhs_value, rhs_value),
                            Op::Sub => self.builder.ins().isub(lhs_value, rhs_value),
                            Op::Mul => self.builder.ins().imul(lhs_value, rhs_value),
                            Op::Div => self.builder.ins().sdiv(lhs_value, rhs_value),
                            Op::EqEq => self.builder.ins().icmp(IntCC::Equal, lhs_value, rhs_value),
                        };
                        self.ctx.values.insert(id, Value::Value(value));
                    }
                    a => todo!("BinOp for {:?}", a),
                }

                Ok(())
            }
            Node::Call { func, params, .. } => {
                self.compile_id(func)?;

                let param_ids = self.ctx.id_vecs[params].clone();
                let mut param_values = Vec::new();
                for &param in param_ids.borrow().iter() {
                    self.compile_id(param)?;
                    param_values.push(self.rvalue(param));
                }

                // direct call?
                let call_inst = if let Some(Value::Func(func_id)) = self.ctx.values.get(&func) {
                    let func_ref = self
                        .ctx
                        .module
                        .declare_func_in_func(*func_id, self.builder.func);
                    self.builder.ins().call(func_ref, &param_values)
                } else {
                    let sig_ref = self
                        .ctx
                        .get_func_signature(func, param_ids.borrow().as_ref());
                    let sig = self.builder.import_signature(sig_ref);

                    self.as_cranelift_value(self.ctx.values[&func]);
                    let callee = self.rvalue(func);

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
                    self.ctx.values.insert(id, Value::Value(value));
                }

                Ok(())
            }
            Node::StructLiteral { params, .. } => {
                let size = self.ctx.get_type_size(id);
                let slot = self.builder.create_sized_stack_slot(StackSlotData {
                    kind: StackSlotKind::ExplicitSlot,
                    size,
                });
                let addr =
                    self.builder
                        .ins()
                        .stack_addr(self.ctx.module.isa().pointer_type(), slot, 0);

                self.ctx.values.insert(id, Value::Value(addr));

                let mut offset = 0;
                for field in self.ctx.id_vecs[params].clone().borrow().iter() {
                    self.compile_id(*field)?;
                    self.store_with_offset(*field, Value::Value(addr), offset);
                    offset += self.ctx.get_type_size(*field);
                }

                Ok(())
            }
            Node::MemberAccess { value, member } => {
                self.compile_id(value)?;

                let (offset, mut pointiness) = self.get_member_offset(value, member)?;

                let mut value = self.as_cranelift_value(self.ctx.values[&value]);
                while pointiness > 0 {
                    value = self.load(types::I64, value, 0);
                    pointiness -= 1;
                }

                if offset != 0 {
                    value = self.builder.ins().iadd_imm(value, offset as i64);
                }

                self.ctx.values.insert(id, Value::Value(value));

                Ok(())
            }
            Node::AddressOf(value) => {
                self.compile_id(value)?;

                let cranelift_value = self.ctx.values.get(&value).unwrap();
                match cranelift_value {
                    Value::Value(_) | Value::Func(_) => {
                        self.ctx.values.insert(id, *cranelift_value)
                    }
                };

                self.store_in_slot_if_addressable(id);

                Ok(())
            }
            Node::Deref(value) => {
                self.compile_id(value)?;

                let cranelift_value = if self.ctx.in_assign_lhs {
                    self.ctx.in_assign_lhs = false;

                    match self.ctx.values.get(&value).unwrap() {
                        Value::Value(value) => *value,
                        cv => todo!("Deref for {:?}", cv),
                    }
                } else {
                    self.rvalue(value)
                };

                let ty = &self.ctx.get_type(id);
                let ty = match ty {
                    Type::Struct { .. } => self.ctx.module.isa().pointer_type(),
                    _ => self.ctx.get_cranelift_type(id),
                };

                let loaded = if self.ctx.addressable_nodes.contains(&value) {
                    self.load(types::I64, cranelift_value, 0)
                } else {
                    self.load(ty, cranelift_value, 0)
                };

                self.ctx.values.insert(id, Value::Value(loaded));

                // If we are meant to be addressable, then we need to store our actual value into something which has an address
                // If however we are in the lhs of an assignment, don't store ourselves in a new slot,
                // as this would overwrite the value we are trying to assign to
                self.store_in_slot_if_addressable(id);

                Ok(())
            }
            Node::Extern {
                name,
                params,
                return_ty,
            } => {
                let mut sig = self.ctx.module.make_signature();

                let return_size = return_ty.map(|rt| self.ctx.get_type_size(rt)).unwrap_or(0);
                if return_size > 0 {
                    sig.returns.push(AbiParam::new(
                        self.ctx.get_cranelift_type(return_ty.unwrap()),
                    ));
                }

                for param in self.ctx.id_vecs[params].borrow().iter() {
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

                self.ctx.values.insert(id, Value::Func(func_id));

                Ok(())
            }
            Node::If {
                cond,
                then_block,
                else_block,
            } => {
                self.compile_id(cond)?;
                let cond_value = self.rvalue(cond);

                // Make a stack slot for the result of the if statement
                let slot = self.builder.create_sized_stack_slot(StackSlotData {
                    kind: StackSlotKind::ExplicitSlot,
                    size: self.ctx.get_type_size(id),
                });
                let slot_addr =
                    self.builder
                        .ins()
                        .stack_addr(self.ctx.module.isa().pointer_type(), slot, 0);
                self.ctx.values.insert(id, Value::Value(slot_addr));

                let saved_resolve_addr = self.resolve_addr;
                self.resolve_addr = Some(Value::Value(slot_addr));

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
                        self.store(else_if, self.resolve_addr.unwrap());
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
            Node::Block {
                stmts,
                is_standalone,
                ..
            } => {
                if is_standalone {
                    // Make a stack slot for the result of the if statement
                    let slot = self.builder.create_sized_stack_slot(StackSlotData {
                        kind: StackSlotKind::ExplicitSlot,
                        size: self.ctx.get_type_size(id),
                    });
                    let slot_addr = self.builder.ins().stack_addr(
                        self.ctx.module.isa().pointer_type(),
                        slot,
                        0,
                    );
                    self.ctx.values.insert(id, Value::Value(slot_addr));

                    let saved_resolve_addr = self.resolve_addr;
                    self.resolve_addr = Some(Value::Value(slot_addr));

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
                    self.store(r, self.resolve_addr.unwrap());
                }

                self.builder
                    .ins()
                    .jump(self.resolve_jump_target.unwrap(), &[]);

                self.exit_current_block();

                Ok(())
            }
            Node::ArrayLiteral { members, ty } => {
                let member_size = self.ctx.get_type_size(ty);

                let members = self.ctx.id_vecs[members].clone();

                // create stack slot big enough for all of the members
                let size = self.ctx.get_type_size(ty);
                let member_storage_slot = self.builder.create_sized_stack_slot(StackSlotData {
                    kind: StackSlotKind::ExplicitSlot,
                    size: size * members.borrow().len() as u32,
                });
                let member_storage_slot_addr = self.builder.ins().stack_addr(
                    self.ctx.module.isa().pointer_type(),
                    member_storage_slot,
                    0,
                );

                let mut offset = 0;
                for &member in members.borrow().iter() {
                    self.compile_id(member)?;
                    self.store_with_offset(member, Value::Value(member_storage_slot_addr), offset);
                    offset += member_size;
                }

                let slot = self.builder.create_sized_stack_slot(StackSlotData {
                    kind: StackSlotKind::ExplicitSlot,
                    size: self.ctx.get_type_size(id),
                });
                let slot_addr =
                    self.builder
                        .ins()
                        .stack_addr(self.ctx.module.isa().pointer_type(), slot, 0);

                let len = self
                    .builder
                    .ins()
                    .iconst(types::I64, members.borrow().len() as i64);
                self.builder
                    .ins()
                    .store(MemFlags::new(), member_storage_slot_addr, slot_addr, 0);
                self.builder.ins().store(MemFlags::new(), len, slot_addr, 8);

                self.ctx.values.insert(id, Value::Value(slot_addr));

                Ok(())
            }
            Node::ArrayAccess { array, index } => {
                self.compile_id(array)?;
                self.compile_id(index)?;

                let member_size = self.ctx.get_type_size(id);

                let array_value = self.rvalue(array);
                let array_ptr_value = self.load(types::I64, array_value, 0);
                let index_value = self.rvalue(index);
                let index_value = self.builder.ins().imul_imm(index_value, member_size as i64);
                let array_ptr_value = self.builder.ins().iadd(array_ptr_value, index_value);

                self.ctx.values.insert(id, Value::Value(array_ptr_value));

                Ok(())
            }
            Node::FnDefinition { .. }
            | Node::StructDefinition { .. }
            | Node::EnumDefinition { .. }
            | Node::StaticMemberAccess { .. } => Ok(()), // This should have already been handled by the toplevel context
            _ => todo!("compile_id for {:?}", &self.ctx.nodes[id]),
        }
    }

    fn compile_ids(&mut self, ids: IdVec) -> Result<(), CompileError> {
        for id in self.ctx.id_vecs[ids].clone().borrow().iter() {
            self.compile_id(*id)?;
        }

        Ok(())
    }

    fn switch_to_block(&mut self, block: Block) {
        self.builder.switch_to_block(block);
        self.current_block = block;
    }

    fn exit_current_block(&mut self) {
        self.exited_blocks.insert(self.current_block);
    }

    fn store_in_slot_if_addressable(&mut self, id: NodeId) {
        if !self.ctx.addressable_nodes.contains(&id) {
            return;
        }

        let size = self.ctx.get_pointer_type().bytes() as u32;
        let slot = self.builder.create_sized_stack_slot(StackSlotData {
            kind: StackSlotKind::ExplicitSlot,
            size,
        });
        let slot_addr =
            self.builder
                .ins()
                .stack_addr(self.ctx.module.isa().pointer_type(), slot, 0);
        let value = Value::Value(slot_addr);
        self.store_value(id, value, None);
        self.ctx.values.insert(id, value);
    }

    fn load(&mut self, ty: types::Type, value: CraneliftValue, offset: i32) -> CraneliftValue {
        self.builder.ins().load(ty, MemFlags::new(), value, offset)
    }

    fn get_member_offset(
        &mut self,
        value: NodeId,
        member: NodeId,
    ) -> Result<(u32, u32), CompileError> {
        let member_name = self.ctx.nodes[member].as_symbol().unwrap();

        let mut ty = self.ctx.get_type(value);
        let mut pointiness = 0;
        while let Type::Pointer(inner) = ty {
            ty = self.ctx.get_type(inner);
            pointiness += 1;
        }

        let Type::Struct { params, .. } = ty else {
            panic!("Not a struct");
        };

        let fields = self.ctx.id_vecs[params].clone();

        let mut offset = 0;
        for field in fields.borrow().iter() {
            let field_name = match &self.ctx.nodes[*field] {
                Node::StructDeclParam { name, .. } => *name,
                Node::ValueParam {
                    name: Some(name), ..
                } => *name,
                _ => panic!("Not a param: {:?}", &self.ctx.nodes[*field]),
            };
            let field_name = self.ctx.nodes[field_name].as_symbol().unwrap();

            if field_name == member_name {
                return Ok((offset, pointiness));
            }

            offset += self.ctx.get_type_size(*field);
        }

        return Err(CompileError::Generic(
            "Member not found".to_string(),
            self.ctx.ranges[value],
        ));
    }

    fn as_cranelift_value(&mut self, value: Value) -> CraneliftValue {
        match value {
            Value::Value(val) => val,
            Value::Func(func_id) => {
                let func_ref = self
                    .ctx
                    .module
                    .declare_func_in_func(func_id, self.builder.func);

                self.builder
                    .ins()
                    .func_addr(self.ctx.get_pointer_type(), func_ref)
            }
        }
    }

    fn rvalue(&mut self, id: NodeId) -> CraneliftValue {
        let value = self.as_cranelift_value(self.ctx.values[&id]);

        if self.ctx.addressable_nodes.contains(&id) && !self.ctx.should_pass_by_ref(id) {
            let ty = self.ctx.get_cranelift_type(id);
            self.builder.ins().load(ty, MemFlags::new(), value, 0)
        } else {
            value
        }
    }

    fn store(&mut self, id: NodeId, dest: Value) {
        if self.ctx.addressable_nodes.contains(&id) {
            self.store_copy(id, dest);
        } else {
            self.store_value(id, dest, None);
        }
    }

    fn store_with_offset(&mut self, id: NodeId, dest: Value, offset: u32) {
        if self.ctx.addressable_nodes.contains(&id) {
            let dest = if offset == 0 {
                dest
            } else {
                let Value::Value(dest) = dest else { panic!("not a value") };
                let dest = self.builder.ins().iadd_imm(dest, offset as i64);
                Value::Value(dest)
            };

            self.store_copy(id, dest);
        } else {
            self.store_value(id, dest, Some(offset as _));
        }
    }

    fn store_copy(&mut self, id: NodeId, dest: Value) {
        let size = self.ctx.get_type_size(id);

        let source_value = self.as_cranelift_value(self.ctx.values[&id]);
        let dest_value = self.as_cranelift_value(dest);

        self.emit_small_memory_copy(dest_value, source_value, size as _);
    }

    fn store_value(&mut self, id: NodeId, dest: Value, offset: Option<i32>) {
        let source_value = match self.ctx.values[&id] {
            Value::Value(value) => value,
            Value::Func(func_id) => {
                let func_ref = self
                    .ctx
                    .module
                    .declare_func_in_func(func_id, self.builder.func);
                self.builder
                    .ins()
                    .func_addr(self.ctx.get_cranelift_type(id), func_ref)
            }
        };

        match dest {
            Value::Value(value) => {
                self.builder.ins().store(
                    MemFlags::new(),
                    source_value,
                    value,
                    offset.unwrap_or_default(),
                );
            }
            _ => todo!("store_value dest for {:?}", dest),
        }
    }

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
}

impl<'a> ToplevelCompileContext<'a> {
    pub fn compile_toplevel_id(&mut self, id: NodeId) -> Result<(), CompileError> {
        // idempotency
        match self.ctx.values.get(&id) {
            None | Some(Value::Func(_)) => {}
            _ => return Ok(()),
        };

        match self.ctx.nodes[id] {
            Node::Symbol(_) => todo!(),
            Node::FnDefinition {
                name,
                stmts,
                params,
                return_ty,
                ..
            } => {
                let name_str = self.ctx.func_name(name, id);

                let mut sig = self.ctx.module.make_signature();

                for &param in self.ctx.id_vecs[params].borrow().iter() {
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

                self.ctx.values.insert(id, Value::Func(func_id));

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
                };

                for &stmt in builder_ctx.ctx.id_vecs[stmts].clone().borrow().iter() {
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
                        let Type::Enum { params, .. } = self.ctx.types[&base] else { todo!("{:?}", self.ctx.nodes[base]) };
                        let params = self.ctx.id_vecs[params].clone();
                        let param = params.borrow()[index as usize];
                        let Node::EnumDeclParam { name, ty } = self.ctx.nodes[param] else { todo!() };

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

                        self.ctx.values.insert(id, Value::Func(func_id));

                        let ebb = builder.create_block();
                        builder.append_block_params_for_function_params(ebb);
                        builder.switch_to_block(ebb);

                        let enum_type_size = self.ctx.get_type_size(base);

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

                            if self.ctx.should_pass_by_ref(param) {
                                let size = self.ctx.get_type_size(param);

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
