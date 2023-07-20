use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};

use cranelift_codegen::ir::{
    stackslot::StackSize, types, AbiParam, Block, InstBuilder, MemFlags, Signature, StackSlotData,
    StackSlotKind, Type as CraneliftType, Value as CraneliftValue,
};
use cranelift_codegen::settings::{self, Configurable};
use cranelift_codegen::Context as CodegenContext;
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{default_libcall_names, FuncId, Linkage, Module};

use crate::{CompileError, Context, Node, NodeId, Op, Type};

#[derive(Clone, Copy, Debug)]
pub enum Value {
    Unassigned,
    None,
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
        if let Node::Func {
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
                Node::Func {
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
            Type::I8 => types::I8,
            Type::I16 => types::I16,
            Type::I32 => types::I32,
            Type::I64 => types::I64,
            Type::F32 => types::F32,
            Type::F64 => types::F64,
            Type::Pointer(_) | Type::Func { .. } | Type::Struct { .. } => self.get_pointer_type(),
            Type::Infer(_) => todo!(),
            _ => todo!("{:?}", &self.types[&param]),
        }
    }

    fn get_pointer_type(&self) -> CraneliftType {
        self.module.isa().pointer_type()
    }

    fn get_type_size(&self, param: NodeId) -> StackSize {
        match self.types[&param] {
            Type::I8 => 1,
            Type::I16 => 2,
            Type::I32 => 4,
            Type::I64 => 8,
            Type::F32 => 4,
            Type::F64 => 8,
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
}

pub fn print_i64(n: i64) {
    print!("{}\n", n);
}

pub fn print_f64(n: f64) {
    print!("{}\n", n);
}

struct ToplevelCompileContext<'a> {
    pub ctx: &'a mut Context,
    pub codegen_ctx: &'a mut CodegenContext,
    pub func_ctx: &'a mut FunctionBuilderContext,
}

struct FunctionCompileContext<'a> {
    pub ctx: &'a mut Context,
    pub builder: FunctionBuilder<'a>,
    pub current_block: Block,
    pub in_assign_lhs: bool,
}

impl<'a> FunctionCompileContext<'a> {
    pub fn compile_id(&mut self, id: NodeId) -> Result<(), CompileError> {
        // idempotency
        match self.ctx.values.get(&id) {
            None | Some(Value::Unassigned) => {}
            _ => return Ok(()),
        };

        match self.ctx.nodes[id] {
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
                    Type::I64 => {
                        let value = self.builder.ins().iconst(types::I64, n);
                        self.ctx.values.insert(id, Value::Value(value));
                    }
                    Type::I32 => {
                        let value = self.builder.ins().iconst(types::I32, n);
                        self.ctx.values.insert(id, Value::Value(value));
                    }
                    _ => todo!(),
                };

                if self.ctx.addressable_nodes.contains(&id) {
                    // todo(chad): is there any benefit to creating all of these up front?
                    let size = self.ctx.module.isa().pointer_bytes() as u32;
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
                    self.store_value(id, value, None);
                    self.ctx.values.insert(id, value);
                }

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

                if self.ctx.addressable_nodes.contains(&id) {
                    // todo(chad): is there any benefit to creating all of these up front?
                    let size = self.ctx.module.isa().pointer_bytes() as u32;
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
                    self.store_value(id, value, None);
                    self.ctx.values.insert(id, value);
                }

                Ok(())
            }
            Node::Type(_) => todo!(),
            Node::Return(rv) => {
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
                        self.in_assign_lhs = true;
                        self.compile_id(name)?;
                        self.in_assign_lhs = false;

                        let addr = self.ctx.values[&name];
                        self.compile_id(expr)?;
                        self.store(expr, addr);
                    }
                    _ => {
                        self.compile_id(name)?;
                        let addr = self.ctx.values[&name];
                        self.compile_id(expr)?;
                        self.store(expr, addr);
                    }
                }

                Ok(())
            }
            Node::FuncDeclParam { index, .. } => {
                // we need our own storage
                let size = self.ctx.get_type_size(id);
                let slot = self.builder.create_sized_stack_slot(StackSlotData {
                    kind: StackSlotKind::ExplicitSlot,
                    size,
                });

                let slot_addr =
                    self.builder
                        .ins()
                        .stack_addr(self.ctx.module.isa().pointer_type(), slot, 0);
                let value = Value::Value(slot_addr);
                self.ctx.values.insert(id, value);

                let params = self.builder.block_params(self.current_block);
                let param_value = params[index as usize];

                if self.ctx.addressable_nodes.contains(&id) {
                    let size = self.ctx.get_type_size(id);

                    let source_value = param_value;
                    let dest_value = slot_addr;

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
                } else {
                    self.builder
                        .ins()
                        .store(MemFlags::new(), param_value, slot_addr, 0);
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

                let value = match op {
                    Op::Add => self.builder.ins().iadd(lhs_value, rhs_value),
                    Op::Sub => self.builder.ins().isub(lhs_value, rhs_value),
                    Op::Mul => self.builder.ins().imul(lhs_value, rhs_value),
                    Op::Div => self.builder.ins().sdiv(lhs_value, rhs_value),
                };

                self.ctx.values.insert(id, Value::Value(value));

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

                self.ctx.addressable_nodes.insert(id);

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
                let value = self.builder.ins().iadd_imm(value, offset as i64);

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
                    _ => todo!("{:?}", cranelift_value),
                };

                // if we are meant to be addressable, then we need to store our actual value into something which has an address
                if self.ctx.addressable_nodes.contains(&id) {
                    // todo(chad): is there any benefit to creating all of these up front?
                    let size = self.ctx.module.isa().pointer_bytes() as u32;
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
                    self.store_value(id, value, None);
                    self.ctx.values.insert(id, value);
                }

                Ok(())
            }
            Node::Deref(value) => {
                self.compile_id(value)?;

                let cranelift_value = if self.in_assign_lhs {
                    let cranelift_value = self.ctx.values.get(&value).unwrap();
                    match cranelift_value {
                        Value::Value(value) => *value,
                        _ => todo!("Deref for {:?}", cranelift_value),
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
                if self.ctx.addressable_nodes.contains(&id) {
                    // todo(chad): is there any benefit to creating all of these up front?
                    let size = self.ctx.module.isa().pointer_bytes() as u32;
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
                    self.store_value(id, value, None);
                    self.ctx.values.insert(id, value);
                }

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
            Node::Func { .. } | Node::StructDefinition { .. } | Node::EnumDefinition { .. } => {
                Ok(())
            } // This should have already been handled by the toplevel context
            _ => todo!("compile_id for {:?}", &self.ctx.nodes[id]),
        }
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
            _ => panic!("not a cranelift value: {:?}", value),
        }
    }

    fn rvalue(&mut self, id: NodeId) -> CraneliftValue {
        let value = self.as_cranelift_value(self.ctx.values[&id]);

        let should_pass_by_ref = matches!(self.ctx.get_type(id), Type::Struct { .. });

        if self.ctx.addressable_nodes.contains(&id) && !should_pass_by_ref {
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
            a => todo!("store_value source for {:?}", a),
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
}

impl<'a> ToplevelCompileContext<'a> {
    pub fn compile_toplevel_id(&mut self, id: NodeId) -> Result<(), CompileError> {
        // idempotency
        match self.ctx.values.get(&id) {
            None | Some(Value::Unassigned | Value::Func(_)) => {}
            _ => return Ok(()),
        };

        match self.ctx.nodes[id] {
            Node::Symbol(_) => todo!(),
            Node::Func {
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
                    current_block: ebb,
                    in_assign_lhs: false,
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
            _ => todo!(),
        }
    }
}
