use std::collections::{HashMap, HashSet};

use cranelift_codegen::entity::EntityRef;
use cranelift_codegen::ir::condcodes::IntCC;
use cranelift_codegen::ir::immediates::Imm64;
use cranelift_codegen::ir::{Block as CraneliftBlock, JumpTableData};
use cranelift_codegen::ir::{FuncRef, GlobalValue, GlobalValueData, StackSlot};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Variable};

use cranelift_codegen::ir::{
    stackslot::StackSize, types, AbiParam, InstBuilder, MemFlags, Signature, StackSlotData,
    StackSlotKind, Type as CraneliftType, Value as CraneliftValue,
};
use cranelift_codegen::settings::{self, Configurable};
use cranelift_codegen::Context as CodegenContext;
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{
    default_libcall_names, DataDescription, DataId, FuncId, Init, Linkage, Module,
};

use crate::{
    ArrayLen, AsCastStyle, CompileError, Context, DraftResult, EmptyDraftResult, IdVec, IfCond,
    MatchCaseTag, Node, NodeId, Op, StaticMemberResolution, Sym, Type,
};

#[derive(Clone, Copy, Debug)]
pub enum Value {
    Func(FuncId),              // A function id, can use this to get to a function pointer
    Register(CraneliftValue),  // A value type which can fit in a register
    StackSlot(StackSlot), // A value type which may or may not fit into a register, held in a stack slot
    Reference(CraneliftValue), // The value is behind a pointer. For example, a struct field, array element, aggregate value, pointer to a global, etc.
}

pub struct GlobalBuilder {
    data_id: DataId,
    bytes: Vec<u8>,
    desc: DataDescription,
}

impl GlobalBuilder {
    pub fn write_u64(&mut self, n: u64) {
        self.bytes.extend(n.to_ne_bytes());
    }

    pub fn extend_bytes_to_length(&mut self, len: usize) {
        while self.bytes.len() < len {
            self.bytes.push(0);
        }
    }
}

impl Context {
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

    pub fn predeclare_string_constants(&mut self) -> EmptyDraftResult {
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

    pub fn predeclare_functions(&mut self) -> EmptyDraftResult {
        for t in self.topo.clone() {
            if !self.completes.contains(&t) {
                continue;
            }

            self.predeclare_function(t)?;
        }

        Ok(())
    }

    pub fn insert_type_infos_into_global_data(&mut self) -> EmptyDraftResult {
        for node_id in 0..self.nodes.len() {
            let Node::TypeInfo(id) = self.nodes[NodeId(node_id)] else {
                continue;
            };
            self.insert_type_info_into_global_data(id)?
        }

        Ok(())
    }

    pub fn insert_type_info_into_global_data(&mut self, id: NodeId) -> EmptyDraftResult {
        let ty = self.types[&id].clone();

        let mut gb = self.make_global_builder()?;

        match ty {
            Type::I64 => {
                // Write the tag, skip the value as it's never used anyway, and this is read-only memory
                gb.write_u64(5);
                gb.extend_bytes_to_length(self.type_info_size());
                self.gb_finalize(gb, Some(id));
            }
            Type::I32 => {
                // Write the tag, skip the value as it's never used anyway, and this is read-only memory
                gb.write_u64(4);
                gb.extend_bytes_to_length(self.type_info_size());
                self.gb_finalize(gb, Some(id));
            }
            Type::Pointer(pid) => {
                // Write the tag
                gb.write_u64(16);
                gb.extend_bytes_to_length(self.type_info_size());

                // Write the value. This is a pointer to the type info of the pointed-to type
                self.gb_write_type_info_pointer(&mut gb, pid)?;

                self.gb_finalize(gb, Some(id));
            }
            Type::Struct { params, decl, .. } => {
                // Write the tag
                gb.write_u64(14);

                // Write the name
                {
                    let name = decl.map(|d| match self.nodes[d] {
                        Node::StructDefinition { name, .. } => {
                            self.nodes[name].as_symbol().unwrap()
                        }
                        Node::EnumDefinition { name, .. } => self.nodes[name].as_symbol().unwrap(),
                        Node::Symbol(sym) => sym,
                        _ => unreachable!(),
                    });

                    match name {
                        Some(sym) => {
                            // Write the tag (Some)
                            gb.write_u64(0);

                            // Write the string value
                            {
                                // Write the pointer
                                self.gb_write_string_data_pointer(&mut gb, sym)?;

                                // Write the length
                                let sym_str = self.string_interner.resolve(sym.0).unwrap();
                                gb.write_u64(sym_str.len() as u64);
                            }
                        }
                        None => {
                            // Write the tag (None)
                            gb.write_u64(1);

                            // Pad with bytes that would represent the string value
                            gb.write_u64(0);
                            gb.write_u64(0);
                        }
                    }
                }

                // Write the params array
                {
                    // Build the data
                    let params_data_id = {
                        let mut params_gb = self.make_global_builder()?;

                        for param in params.borrow().iter() {
                            // TypeInfoParam
                            // Name (string)
                            {
                                let (Node::StructDeclParam { name, .. }
                                | Node::ValueParam {
                                    name: Some(name), ..
                                }) = self.nodes[*param].clone()
                                else {
                                    todo!("{:?}", self.nodes[*param].clone())
                                };
                                let name = self.nodes[name].as_symbol().unwrap();

                                // Write the pointer
                                self.gb_write_string_data_pointer(&mut params_gb, name)?;

                                // Write the length
                                let name_str = self.string_interner.resolve(name.0).unwrap();
                                params_gb.write_u64(name_str.len() as u64);
                            }

                            // Type Info Pointer
                            {
                                self.insert_type_info_into_global_data(*param)?;

                                let (param_data_id, _) = self.global_values[&param];
                                self.gb_write_global_pointer(&mut params_gb, param_data_id)?;
                            }
                        }

                        self.gb_finalize(params_gb, None)
                    };

                    // Write the data pointer
                    self.gb_write_global_pointer(&mut gb, params_data_id)?;

                    // Write the length
                    gb.write_u64(params.borrow().len() as u64);
                }

                gb.extend_bytes_to_length(self.type_info_size());

                self.gb_finalize(gb, Some(id));
            }
            a => todo!("insert_type_infos_into_global_data for {:?}", a),
        }

        Ok(())
    }

    #[inline]
    pub fn type_info_size(&self) -> usize {
        self.id_type_size(self.type_info_decl.unwrap()) as usize
    }

    pub fn get_string_data_id(&mut self, sym: Sym, name: &str) -> DraftResult<DataId> {
        if let Some(data_id) = self.symbol_data_ids.get(&sym) {
            return Ok(*data_id);
        }

        let sym_str = self.string_interner.resolve(sym.0).unwrap();

        let string_data_id = self
            .module
            .declare_data(&name, Linkage::Local, false, false)
            .map_err(|err| CompileError::Message(err.to_string()))?;
        let mut string_desc = DataDescription::new();
        string_desc.init = Init::Bytes {
            contents: sym_str.as_bytes().to_vec().into_boxed_slice(),
        };
        self.module
            .define_data(string_data_id, &string_desc)
            .unwrap();

        self.symbol_data_ids.insert(sym, string_data_id);

        Ok(string_data_id)
    }

    pub fn define_functions(&mut self) -> EmptyDraftResult {
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

    pub fn predeclare_function(&mut self, id: NodeId) -> EmptyDraftResult {
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
                    .push(AbiParam::new(self.id_cranelift_type(param)));
            }

            let return_size = return_ty.map(|rt| self.id_type_size(rt)).unwrap_or(0);

            if return_size > 0 {
                sig.returns
                    .push(AbiParam::new(self.id_cranelift_type(return_ty.unwrap())));
            }

            let func_name = self.mangled_func_name(name, id);

            let func = self
                .module
                .declare_function(&func_name, Linkage::Local, &sig)
                .unwrap();

            self.values.insert(id, Value::Func(func));
        }

        Ok(())
    }

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

    pub fn call_fn(&mut self, fn_name: &str) -> EmptyDraftResult {
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

    fn id_cranelift_type(&self, param: NodeId) -> CraneliftType {
        self.cranelift_type(self.types[&param].clone())
    }

    fn cranelift_type(&self, ty: Type) -> CraneliftType {
        match ty {
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
            a => todo!("{:?}", a),
        }
    }

    fn get_pointer_type(&self) -> CraneliftType {
        self.module.isa().pointer_type()
    }

    #[inline]
    fn enum_tag_size(&self) -> u32 {
        // todo(chad): this is horrible, but until alignment is sorted out it's the only valid size that works for everything
        std::mem::size_of::<u64>() as u32
    }

    pub fn id_type_size(&self, id: NodeId) -> StackSize {
        self.type_size(self.types[&id].clone())
    }

    pub fn type_size(&self, ty: Type) -> StackSize {
        match ty {
            Type::I8 | Type::U8 | Type::Bool => 1,
            Type::I16 | Type::U16 => 2,
            Type::I32 | Type::U32 | Type::F32 => 4,
            Type::I64 | Type::U64 | Type::F64 => 8,
            Type::Func { .. } => 8,
            Type::Pointer(_) => self.get_pointer_type().bytes(),
            Type::Array(ty, ArrayLen::Some(len)) => self.id_type_size(ty) * len as StackSize,
            Type::Struct { params, .. } => {
                // todo(chad): alignment
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

        let return_size = return_ty.map(|rt| self.id_type_size(rt)).unwrap_or(0);

        let mut sig = self.module.make_signature();
        for param in param_ids.iter() {
            sig.params
                .push(AbiParam::new(self.id_cranelift_type(*param)));
        }

        if return_size > 0 {
            sig.returns
                .push(AbiParam::new(self.id_cranelift_type(return_ty.unwrap())));
        }

        sig
    }

    pub fn id_is_aggregate_type(&self, id: NodeId) -> bool {
        self.is_aggregate_type(self.get_type(id))
    }

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

    pub fn is_aggregate_type(&self, ty: Type) -> bool {
        matches!(
            ty,
            Type::Struct { .. } | Type::Enum { .. } | Type::Array(_, _) | Type::String
        )
    }
}

impl Context {
    fn make_global_builder(&mut self) -> DraftResult<GlobalBuilder> {
        let data_id = self
            .module
            .declare_data(
                &format!("g{}", self.global_builder_id),
                Linkage::Local,
                false,
                false,
            )
            .map_err(|err| CompileError::Message(err.to_string()))?;

        self.global_builder_id += 1;

        let bytes = Vec::new();

        let desc = DataDescription::new();

        Ok(GlobalBuilder {
            data_id,
            bytes,
            desc,
        })
    }

    fn gb_write_string_data_pointer(
        &mut self,
        gb: &mut GlobalBuilder,
        sym: Sym,
    ) -> EmptyDraftResult {
        let string_data_id = self.get_string_data_id(sym, &format!("sym__{:?}", sym.0))?;
        let string_global = self
            .module
            .declare_data_in_data(string_data_id, &mut gb.desc);
        gb.desc
            .write_data_addr(gb.bytes.len() as u32, string_global, 0);
        gb.write_u64(0); // extend bytes to make room for the pointer

        Ok(())
    }

    fn gb_write_global_pointer(&mut self, gb: &mut GlobalBuilder, id: DataId) -> EmptyDraftResult {
        let global = self.module.declare_data_in_data(id, &mut gb.desc);
        gb.desc.write_data_addr(gb.bytes.len() as u32, global, 0);
        gb.write_u64(0); // extend bytes to make room for the pointer

        Ok(())
    }

    fn gb_write_type_info_pointer(
        &mut self,
        gb: &mut GlobalBuilder,
        id: NodeId,
    ) -> EmptyDraftResult {
        self.insert_type_info_into_global_data(id)?;
        let (pid_data_id, _) = self.global_values[&id];
        let pglobal = self.module.declare_data_in_data(pid_data_id, &mut gb.desc);

        // Write the data address of pglobal into the type info, after the tag
        gb.desc.write_data_addr(gb.bytes.len() as u32, pglobal, 0);
        gb.bytes.extend(0u64.to_ne_bytes()); // extend bytes to make room for the pointer

        Ok(())
    }

    pub fn gb_finalize(&mut self, mut gb: GlobalBuilder, id: Option<NodeId>) -> DataId {
        gb.desc.init = Init::Bytes {
            contents: gb.bytes.into_boxed_slice(),
        };

        self.module.define_data(gb.data_id, &gb.desc).unwrap();

        if let Some(id) = id {
            self.global_values.insert(id, (gb.data_id, None));
        }

        gb.data_id
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
    pub tag: u64,
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

#[derive(Clone, Debug)]
enum UnrolledDot {
    Symbol(NodeId),
    ArrayIndex(NodeId),
    Call(IdVec),
}

struct ToplevelCompileContext<'a> {
    pub ctx: &'a mut Context,
    pub codegen_ctx: &'a mut CodegenContext,
    pub func_ctx: &'a mut FunctionBuilderContext,
}

#[derive(Clone, Copy)]
enum Block {
    Block(CraneliftBlock),
    Continue(CraneliftBlock),
    Break(CraneliftBlock),
}

impl Into<CraneliftBlock> for Block {
    fn into(self) -> CraneliftBlock {
        match self {
            Block::Block(b) => b,
            Block::Continue(b) => b,
            Block::Break(b) => b,
        }
    }
}

struct FunctionCompileContext<'a> {
    pub ctx: &'a mut Context,
    pub builder: FunctionBuilder<'a>,
    pub exited_blocks: HashSet<CraneliftBlock>, // which blocks have been terminated
    pub break_addr: Option<Value>, // the stack slot where we store the value of any Node::Break we encounter
    pub declared_func_ids: HashMap<FuncId, FuncRef>,
    pub global_str_ptr: Option<GlobalValue>,
    pub blocks: Vec<(Option<Sym>, Block)>,
    pub current_block: CraneliftBlock,
    pub current_fn_block: CraneliftBlock,
}

impl<'a> FunctionCompileContext<'a> {
    pub fn compile_id(&mut self, id: NodeId) -> EmptyDraftResult {
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
                    _ => todo!("{:?}", self.ctx.ranges[id]),
                }
            }
            Node::IntLiteral(n, _) => {
                match self.ctx.types[&id] {
                    Type::I64 | Type::U64 => {
                        let value = self.builder.ins().iconst(types::I64, n);
                        self.ctx.values.insert(id, Value::Register(value));
                    }
                    Type::I32 | Type::U32 => {
                        let value = self.builder.ins().iconst(types::I32, n);
                        self.ctx.values.insert(id, Value::Register(value));
                    }
                    _ => todo!(),
                };

                Ok(())
            }
            Node::FloatLiteral(n, _) => {
                match self.ctx.types[&id] {
                    Type::F64 => {
                        let value = self.builder.ins().f64const(n);
                        self.ctx.values.insert(id, Value::Register(value));
                    }
                    Type::F32 => {
                        let value = self.builder.ins().f32const(n as f32);
                        self.ctx.values.insert(id, Value::Register(value));
                    }
                    _ => todo!(),
                };

                Ok(())
            }
            Node::BoolLiteral(b) => {
                let value = self.builder.ins().iconst(types::I8, if b { 1 } else { 0 });
                self.ctx.values.insert(id, Value::Register(value));
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
                if let Some(expr) = expr {
                    self.compile_id(expr)?;

                    if let Node::StructLiteral { .. }
                    | Node::ArrayLiteral { .. }
                    | Node::StringLiteral { .. } = self.ctx.nodes[expr]
                    {
                        // If we're assigning from an aggregate literal, then nothing else can access it so we can just cheat and use the slot it already made.
                        self.ctx.values.insert(id, self.ctx.values[&expr]);
                    } else {
                        let size: u32 = self.ctx.id_type_size(id);
                        let slot = self.builder.create_sized_stack_slot(StackSlotData {
                            kind: StackSlotKind::ExplicitSlot,
                            size,
                        });
                        self.store_copy(expr, Value::StackSlot(slot));
                        self.ctx.values.insert(id, Value::StackSlot(slot));
                    }
                } else {
                    let size: u32 = self.ctx.id_type_size(id);
                    let slot = self.builder.create_sized_stack_slot(StackSlotData {
                        kind: StackSlotKind::ExplicitSlot,
                        size,
                    });
                    self.ctx.values.insert(id, Value::StackSlot(slot));
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
                let params = self.builder.block_params(self.current_fn_block);
                let param_value = params[index as usize];

                if self.ctx.id_is_aggregate_type(id) {
                    self.ctx.values.insert(id, Value::Reference(param_value));
                } else {
                    self.ctx.values.insert(id, Value::Register(param_value));
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

                let test_rhs_block = self.create_block(None);
                let cont_block = self.create_block(None);

                self.compile_id(lhs)?;
                let lhs_rvalue = self.id_value(lhs);
                self.builder.def_var(variable, lhs_rvalue);
                self.builder.ins().brif(
                    lhs_rvalue,
                    test_rhs_block.into(),
                    &[],
                    cont_block.into(),
                    &[],
                );

                self.switch_to_block(test_rhs_block);
                self.compile_id(rhs)?;
                let rhs_rvalue = self.id_value(rhs);
                self.builder.def_var(variable, rhs_rvalue);
                self.builder.ins().jump(cont_block.into(), &[]);

                self.switch_to_block(cont_block);

                let variable_value = self.builder.use_var(variable);
                self.ctx.values.insert(id, Value::Register(variable_value));

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
                self.ctx.values.insert(id, Value::Register(variable_value));

                Ok(())
            }
            Node::BinOp { op, lhs, rhs } => {
                self.compile_id(lhs)?;
                self.compile_id(rhs)?;

                let lhs_value = self.id_value(lhs);
                let rhs_value = self.id_value(rhs);

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
                        self.ctx.values.insert(id, Value::Register(value));
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

                let value = Value::StackSlot(slot);

                self.ctx.values.insert(id, value);

                let mut offset = 0;
                for field in params.borrow().iter() {
                    self.compile_id(*field)?;
                    self.store_with_offset(*field, Value::StackSlot(slot), offset);
                    offset += self.ctx.id_type_size(*field);
                }

                Ok(())
            }
            Node::MemberAccess { value, member } => {
                let mut unrolled = Vec::new();
                self.unroll_member_access(member, &mut unrolled)?;

                // a dot b
                let mut a_val;
                let mut a_ty;
                {
                    let a = value;
                    self.compile_id(a)?;
                    a_val = self.id_value(a);
                    a_ty = self.ctx.get_type(a);
                }

                let mut cur = 0;
                while cur < unrolled.len() {
                    // We start out with an id_value already. But each additional time through the loop,
                    // we need to manually create it. We know `a_val` is a Value::Reference(_), so we can just follow
                    // the same rules for Reference from id_value
                    if cur > 0 && !self.ctx.is_aggregate_type(a_ty.clone()) {
                        let a_ty = self.ctx.cranelift_type(a_ty.clone());
                        a_val = self.load(a_ty, a_val, 0);
                    }

                    let b = unrolled[cur].clone();

                    let mut pointiness = 0;
                    while let Type::Pointer(inner) = a_ty {
                        pointiness += 1;
                        a_ty = self.ctx.get_type(inner);
                    }

                    // Dereference the pointer until we get to the actual value
                    // Stopping at 1 because structs are always reference types anyway, so member access on a pointer to a struct
                    // is effectively the same as member access on a struct
                    while pointiness > 1 {
                        a_val = self.load(types::I64, a_val, 0);
                        pointiness -= 1;
                    }

                    match b {
                        UnrolledDot::Symbol(b) => {
                            let (offset, ty) = self.get_member_offset(a_ty, b)?;
                            a_ty = ty;
                            if offset != 0 {
                                a_val = self.builder.ins().iadd_imm(a_val, offset as i64);
                            }
                        }
                        UnrolledDot::ArrayIndex(idx) => {
                            self.compile_id(idx)?;

                            let Type::Array(array_ty, _) = a_ty.clone() else {
                                todo!()
                            };
                            let array_ty = self.ctx.get_type(array_ty);

                            let ty_size = self.ctx.type_size(array_ty.clone());
                            let ty_size = self.builder.ins().iconst(types::I64, ty_size as i64);

                            let array_ptr_val = if let Type::Array(_, ArrayLen::Some(_)) = array_ty
                            {
                                a_val
                            } else {
                                self.load(types::I64, a_val, 0)
                            };

                            let idx_value = self.id_value(idx);
                            let idx_value = self.builder.ins().imul(idx_value, ty_size);
                            a_val = self.builder.ins().iadd(array_ptr_val, idx_value);
                            a_ty = array_ty;
                        }
                        UnrolledDot::Call(params) => {
                            for p in params.borrow().iter() {
                                self.compile_id(*p)?;
                            }

                            let param_ids = params.clone();
                            let mut param_values = Vec::new();
                            for &param in param_ids.borrow().iter() {
                                self.compile_id(param)?;
                                param_values.push(self.id_value(param));
                            }

                            // todo(chad): direct call?
                            let sig_ref = {
                                let func_ty = a_ty.clone();
                                let return_ty = match func_ty {
                                    Type::Func {
                                        return_ty: Some(return_ty),
                                        ..
                                    } => Some(return_ty),
                                    _ => None,
                                };

                                let return_size =
                                    return_ty.map(|rt| self.ctx.id_type_size(rt)).unwrap_or(0);

                                let mut sig = self.ctx.module.make_signature();
                                for param in param_ids.borrow().iter() {
                                    sig.params
                                        .push(AbiParam::new(self.ctx.id_cranelift_type(*param)));
                                }

                                if return_size > 0 {
                                    sig.returns.push(AbiParam::new(
                                        self.ctx.id_cranelift_type(return_ty.unwrap()),
                                    ));
                                }

                                sig
                            };
                            let sig = self.builder.import_signature(sig_ref);

                            let call_inst =
                                self.builder.ins().call_indirect(sig, a_val, &param_values);

                            if self
                                .ctx
                                .types
                                .get(&id)
                                .map(|t| !matches!(t, Type::Infer(_)))
                                .unwrap_or_default()
                            {
                                a_val = self.builder.func.dfg.first_result(call_inst);
                            } else {
                                todo!()
                            }

                            let Type::Func { return_ty, .. } = a_ty.clone() else {
                                todo!()
                            };

                            a_ty = self.ctx.get_type(return_ty.unwrap());
                        }
                    }

                    self.ctx.values.insert(id, Value::Reference(a_val));

                    cur += 1;
                }

                Ok(())
            }
            Node::AddressOf(a) => {
                self.compile_id(a)?;

                let value = self.ctx.values.get(&a).unwrap();

                match value {
                    Value::Func(_) | Value::Register(_) => {
                        let slot = self.builder.create_sized_stack_slot(StackSlotData {
                            kind: StackSlotKind::ExplicitSlot,
                            size: self.ctx.id_type_size(a),
                        });

                        let value = self.id_value(a);
                        self.builder.ins().stack_store(value, slot, 0);

                        let slot_addr =
                            self.builder
                                .ins()
                                .stack_addr(self.ctx.get_pointer_type(), slot, 0);

                        self.ctx.values.insert(id, Value::Register(slot_addr));
                    }
                    Value::StackSlot(slot) => {
                        let slot_addr =
                            self.builder
                                .ins()
                                .stack_addr(self.ctx.get_pointer_type(), *slot, 0);

                        self.ctx.values.insert(id, Value::Register(slot_addr));
                    }
                    Value::Reference(ptr) => {
                        self.ctx.values.insert(id, Value::Register(*ptr));
                    }
                }
                // self.ctx.values.insert(id, *cranelift_value);

                Ok(())
            }
            Node::Deref(value) => {
                self.compile_id(value)?;

                if self.ctx.in_assign_lhs {
                    let value = self.id_value(value);

                    self.ctx.values.insert(id, Value::Reference(value));
                } else {
                    if self.ctx.id_is_aggregate_type(id) {
                        // If we're dereferencing to an aggregate type, then we are going to need a stack slot and will become an Aggregate value ourselves
                        let slot = self.builder.create_sized_stack_slot(StackSlotData {
                            kind: StackSlotKind::ExplicitSlot,
                            size: self.ctx.id_type_size(id),
                        });

                        let size = self.ctx.id_type_size(id);

                        let value = self.id_value(value);
                        let slot_addr =
                            self.builder
                                .ins()
                                .stack_addr(self.ctx.get_pointer_type(), slot, 0);

                        self.emit_small_memory_copy(slot_addr, value, size as _);

                        self.ctx.values.insert(id, Value::StackSlot(slot));
                    } else {
                        let rvalue = self.id_value(value);
                        let ty = self.ctx.id_cranelift_type(value);
                        let value = self.load(ty, rvalue, 0);
                        self.ctx.values.insert(id, Value::Register(value));
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
                        self.ctx.id_cranelift_type(return_ty.unwrap()),
                    ));
                }

                for param in params.borrow().iter() {
                    sig.params
                        .push(AbiParam::new(self.ctx.id_cranelift_type(*param)));
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
                then_label,
                else_block,
                else_label,
            } => {
                let cond_value = match cond {
                    IfCond::Expr(expr) => {
                        self.compile_id(expr)?;
                        self.id_value(expr)
                    }
                    IfCond::Let { tag, alias, expr } => {
                        self.compile_id(expr)?;

                        let tag_sym = self.ctx.get_symbol(tag);

                        let expr_ty = self.ctx.get_type(expr);
                        let Type::Enum { params, .. } = expr_ty else {
                            unreachable!()
                        };
                        let mut param_index = -1;
                        for (idx, &param) in params.borrow().iter().enumerate() {
                            if let Node::EnumDeclParam { name, .. } = self.ctx.nodes[param].clone()
                            {
                                let name_sym = self.ctx.get_symbol(name);
                                if name_sym == tag_sym {
                                    param_index = idx as i64;
                                    break;
                                }
                            } else {
                                todo!()
                            };
                        }

                        // Regardless, assign the value to the alias in case the branch is taken.
                        // If the branch isn't taken it doesn't matter anyway
                        if let Some(alias) = alias {
                            let tag_offset = self.ctx.enum_tag_size();
                            let enum_value_ptr = self.id_value(expr);
                            let enum_value_ptr = self
                                .builder
                                .ins()
                                .iadd_imm(enum_value_ptr, tag_offset as i64);
                            let enum_value_ptr = Value::Reference(enum_value_ptr);
                            self.ctx.values.insert(alias, enum_value_ptr);
                        }

                        // Get the tag value of expr
                        let expr_value = self.id_value(expr);
                        let tag_value = self.load(types::I64, expr_value, 0);

                        // Compare against param_index
                        self.builder
                            .ins()
                            .icmp_imm(IntCC::Equal, tag_value, param_index)
                    }
                };

                let saved_break_addr = self.break_addr;

                // Make a stack slot for the result of the if statement
                let slot_size = self.ctx.id_type_size(id);
                if slot_size > 0 {
                    let slot = self.builder.create_sized_stack_slot(StackSlotData {
                        kind: StackSlotKind::ExplicitSlot,
                        size: slot_size,
                    });

                    let value = Value::StackSlot(slot);
                    self.ctx.values.insert(id, value);

                    self.break_addr = Some(value);
                }

                let then_ebb = self.builder.create_block();
                let else_ebb = self.builder.create_block();
                let merge_ebb = self.builder.create_block();

                self.blocks.push((then_label, Block::Break(merge_ebb)));
                self.blocks.push((else_label, Block::Break(merge_ebb)));

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
                        if let Some(break_addr) = self.break_addr {
                            self.store_copy(else_if, break_addr);
                        }
                        self.builder.ins().jump(merge_ebb, &[]);
                    }
                    crate::NodeElse::None => {
                        self.builder.ins().jump(merge_ebb, &[]);
                    }
                }

                self.builder.switch_to_block(merge_ebb);

                self.blocks.pop();
                self.blocks.pop();

                self.break_addr = saved_break_addr;

                Ok(())
            }
            Node::For {
                label,
                iterable,
                block,
                block_label,
            } => {
                self.compile_id(iterable)?;

                let (array_pointer_value, array_length_value) = match self.ctx.get_type(iterable) {
                    Type::Array(_, ArrayLen::None) => {
                        let array_value = self.id_value(iterable);
                        let array_pointer_value = self.load(types::I64, array_value, 0);
                        let array_length_value = self.load(
                            types::I64,
                            array_value,
                            self.ctx.get_pointer_type().bytes() as _,
                        );

                        (array_pointer_value, array_length_value)
                    }
                    Type::Array(_, ArrayLen::Some(len)) => {
                        let array_pointer_value = self.id_value(iterable);
                        let array_length_value = self.builder.ins().iconst(types::I64, len as i64);

                        (array_pointer_value, array_length_value)
                    }
                    ty => panic!("For loop over non-iterable type {:?}", ty),
                };

                let array_elem_type_size = self.ctx.id_type_size(label);

                // Make a stack slot for the label
                let label_slot = self.builder.create_sized_stack_slot(StackSlotData {
                    kind: StackSlotKind::ExplicitSlot,
                    size: array_elem_type_size,
                });
                self.ctx.values.insert(label, Value::StackSlot(label_slot));

                // Index starts at 0
                let index = Variable::new(id.0);
                self.builder.declare_var(index, types::I64);
                let const_zero = self.builder.ins().iconst(types::I64, 0);
                self.builder.def_var(index, const_zero);

                let cond_ebb = self.builder.create_block();
                let block_ebb = self.builder.create_block();
                let merge_ebb = self.builder.create_block();
                let incr_ebb = self.builder.create_block();

                self.blocks.push((block_label, Block::Continue(incr_ebb)));
                self.blocks.push((block_label, Block::Break(merge_ebb)));

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

                let label_slot_addr =
                    self.builder
                        .ins()
                        .stack_addr(self.ctx.get_pointer_type(), label_slot, 0);
                self.emit_small_memory_copy(
                    label_slot_addr,
                    element_ptr,
                    array_elem_type_size as _,
                );

                self.compile_id(block)?;

                // increment iterator
                if !self.exited_blocks.contains(&self.current_block) {
                    self.builder.ins().jump(incr_ebb, &[]);
                    self.switch_to_block(incr_ebb);
                }

                let used_index = self.builder.use_var(index);
                let incremented = self.builder.ins().iadd_imm(used_index, 1);
                self.builder.def_var(index, incremented);

                if !self.exited_blocks.contains(&self.current_block) {
                    self.builder.ins().jump(cond_ebb, &[]);
                }

                self.builder.switch_to_block(merge_ebb);

                self.blocks.pop();
                self.blocks.pop();

                Ok(())
            }
            Node::While {
                cond,
                block,
                block_label,
            } => {
                let cond_block = self.builder.create_block();
                let while_block = self.builder.create_block();
                let merge_block = self.builder.create_block();

                self.blocks.push((block_label, Block::Continue(cond_block)));
                self.blocks.push((block_label, Block::Break(merge_block)));

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

                self.blocks.pop();
                self.blocks.pop();

                Ok(())
            }
            Node::Block {
                label,
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

                    let value = Value::StackSlot(slot);
                    self.ctx.values.insert(id, value);

                    let saved_break_addr = self.break_addr;
                    self.break_addr = Some(value);

                    let merge_ebb = self.builder.create_block();

                    self.blocks.push((label, Block::Break(merge_ebb)));

                    self.compile_ids(stmts)?;
                    if !self.exited_blocks.contains(&self.current_block) {
                        self.builder.ins().jump(merge_ebb, &[]);
                    }

                    self.switch_to_block(merge_ebb);

                    self.break_addr = saved_break_addr;
                    self.blocks.pop();
                } else {
                    self.compile_ids(stmts)?;
                }

                Ok(())
            }
            Node::Break(r, name) => {
                if let Some(r) = r {
                    self.compile_id(r)?;
                    self.store_copy(r, self.break_addr.unwrap());
                }

                let break_block = self.find_break_block(name).unwrap();

                self.builder.ins().jump(break_block, &[]);

                self.exit_current_block();

                Ok(())
            }
            Node::Continue(sym) => {
                let continue_block = self.find_continue_block(sym).unwrap();

                self.builder.ins().jump(continue_block, &[]);

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

                let member_is_aggregate_type = self.ctx.id_is_aggregate_type(ty);

                let mut offset = 0;
                for &member in members.borrow().iter() {
                    self.compile_id(member)?;
                    let member_value = self.id_value(member);

                    if member_is_aggregate_type {
                        let member_storage_slot_addr = self.builder.ins().stack_addr(
                            self.ctx.get_pointer_type(),
                            member_storage_slot,
                            offset as i32,
                        );
                        self.emit_small_memory_copy(
                            member_storage_slot_addr,
                            member_value,
                            member_size as _,
                        );
                    } else {
                        self.builder.ins().stack_store(
                            member_value,
                            member_storage_slot,
                            offset as i32,
                        );
                    }

                    offset += member_size;
                }

                self.ctx
                    .values
                    .insert(id, Value::StackSlot(member_storage_slot));

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

                self.ctx.values.insert(id, Value::Reference(element_ptr));

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

                self.ctx.values.insert(id, Value::StackSlot(slot));

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

                self.ctx.values.insert(id, Value::Register(value));

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

                self.ctx.values.insert(id, Value::Register(value));

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

                    self.ctx.values.insert(id, Value::StackSlot(slot));

                    Ok(())
                }
                AsCastStyle::StructToDynamicArray => {
                    self.compile_id(value)?;
                    self.ctx.values.insert(id, self.ctx.values[&value]);
                    Ok(())
                }
                _ => unreachable!(),
            },
            Node::TypeInfo(tid) => {
                let (data_id, _gv) = self.ctx.global_values[&tid];

                let gv = self
                    .ctx
                    .module
                    .declare_data_in_func(data_id, self.builder.func);

                let ptr = self
                    .builder
                    .ins()
                    .global_value(self.ctx.get_pointer_type(), gv);

                self.ctx.values.insert(id, Value::Reference(ptr));

                Ok(())
            }
            Node::Match { value, cases } => {
                self.compile_id(value)?;

                let saved_break_addr = self.break_addr;

                // Make a stack slot for the result of the block
                let slot_size = self.ctx.id_type_size(id);
                if slot_size > 0 {
                    let slot = self.builder.create_sized_stack_slot(StackSlotData {
                        kind: StackSlotKind::ExplicitSlot,
                        size: slot_size,
                    });

                    let value = Value::StackSlot(slot);
                    self.ctx.values.insert(id, value);

                    self.break_addr = Some(value);
                }

                for case in cases.borrow().iter() {
                    let Node::MatchCase {
                        tag,
                        alias,
                        block,
                        block_label,
                    } = self.ctx.nodes[*case]
                    else {
                        unreachable!()
                    };

                    let value_ty = self.ctx.get_type(value);
                    let Type::Enum { params, .. } = value_ty else {
                        unreachable!()
                    };

                    let then_ebb = self.builder.create_block();
                    let merge_ebb = self.builder.create_block();

                    match tag {
                        MatchCaseTag::CatchAll => {
                            self.blocks.push((block_label, Block::Break(merge_ebb)));

                            self.builder.ins().jump(then_ebb, &[]);

                            self.builder.switch_to_block(then_ebb);
                            self.compile_id(block)?;
                        }
                        MatchCaseTag::Node(tag) => {
                            let mut param_index = -1;
                            for (idx, &param) in params.borrow().iter().enumerate() {
                                if let Node::EnumDeclParam { name, .. } =
                                    self.ctx.nodes[param].clone()
                                {
                                    let name_sym = self.ctx.get_symbol(name);

                                    let tag_sym = match self.ctx.nodes[tag] {
                                        Node::Symbol(sym) => sym,
                                        Node::EnumDeclParam { name, .. } => {
                                            self.ctx.get_symbol(name)
                                        }
                                        _ => todo!(),
                                    };

                                    if name_sym == tag_sym {
                                        param_index = idx as i64;
                                        break;
                                    }
                                } else {
                                    todo!()
                                };
                            }

                            // Assign the value to the alias in case the branch is taken.
                            if let Some(alias) = alias {
                                let tag_offset = self.ctx.enum_tag_size();
                                let enum_value_ptr = self.id_value(value);
                                let enum_value_ptr = self
                                    .builder
                                    .ins()
                                    .iadd_imm(enum_value_ptr, tag_offset as i64);
                                let enum_value_ptr = Value::Reference(enum_value_ptr);
                                self.ctx.values.insert(alias, enum_value_ptr);
                            }

                            // Get the tag value of expr
                            let expr_value = self.id_value(value);
                            let tag_value = self.load(types::I64, expr_value, 0);

                            // Compare against param_index
                            let cond_value =
                                self.builder
                                    .ins()
                                    .icmp_imm(IntCC::Equal, tag_value, param_index);

                            self.blocks.push((block_label, Block::Break(merge_ebb)));

                            self.builder
                                .ins()
                                .brif(cond_value, then_ebb, &[], merge_ebb, &[]);

                            self.builder.switch_to_block(then_ebb);
                            self.compile_id(block)?;
                        }
                    }

                    if !self.exited_blocks.contains(&self.current_block) {
                        self.builder.ins().jump(merge_ebb, &[]);
                    }

                    self.builder.switch_to_block(merge_ebb);

                    self.blocks.pop();
                    self.blocks.pop();
                }

                self.break_addr = saved_break_addr;

                Ok(())
            }
            Node::MatchCase { .. } => {
                unreachable!("Should have already been handled when compiling 'Match'")
            }
            Node::FnDefinition { .. }
            | Node::StructDefinition { .. }
            | Node::EnumDefinition { .. } => Ok(()), // This should have already been handled by the toplevel context
            _ => todo!("compile_id for {:?}", &self.ctx.nodes[id]),
        }
    }

    fn unroll_member_access(
        &self,
        id: NodeId,
        unrolled: &mut Vec<UnrolledDot>,
    ) -> EmptyDraftResult {
        match self.ctx.nodes[id].clone() {
            Node::Symbol(_) => unrolled.push(UnrolledDot::Symbol(id)),
            Node::StructDeclParam { name, .. } => unrolled.push(UnrolledDot::Symbol(name)),
            Node::EnumDeclParam { name, .. } => unrolled.push(UnrolledDot::Symbol(name)),
            Node::MemberAccess {
                value: inner_value,
                member: inner_member,
            } => {
                self.unroll_member_access(inner_value, unrolled)?;
                self.unroll_member_access(inner_member, unrolled)?;
            }
            Node::ArrayAccess { array, index } => {
                self.unroll_member_access(array, unrolled)?;
                unrolled.push(UnrolledDot::ArrayIndex(index))
            }
            Node::Call { func, params } => {
                self.unroll_member_access(func, unrolled)?;
                unrolled.push(UnrolledDot::Call(params));
            }
            a => todo!("Unroll member access for value node {:?}", a),
        }

        Ok(())
    }

    fn compile_ids(&mut self, ids: IdVec) -> EmptyDraftResult {
        for id in ids.borrow().iter() {
            self.compile_id(*id)?;
        }

        Ok(())
    }

    fn compile_id_for_call(&mut self, id: NodeId, func: NodeId, params: IdVec) -> EmptyDraftResult {
        self.compile_id(func)?;

        let param_ids = params.clone();
        let mut param_values = Vec::new();
        for &param in param_ids.borrow().iter() {
            self.compile_id(param)?;
            param_values.push(self.id_value(param));
        }

        // direct call?
        let call_inst = if let Some(Value::Func(func_id)) = self.ctx.values.get(&func) {
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
                self.ctx.values.insert(id, Value::Reference(value));
            } else {
                self.ctx.values.insert(id, Value::Register(value));
            }
        }

        Ok(())
    }

    fn switch_to_block(&mut self, block: impl Into<CraneliftBlock>) {
        let block = block.into();

        self.builder.switch_to_block(block);
        self.current_block = block;
    }

    fn exit_current_block(&mut self) {
        self.exited_blocks.insert(self.current_block);
    }

    fn load(&mut self, ty: types::Type, value: CraneliftValue, offset: i32) -> CraneliftValue {
        self.builder.ins().load(ty, MemFlags::new(), value, offset)
    }

    fn get_member_offset(
        &mut self,
        mut value_ty: Type,
        member: NodeId,
    ) -> DraftResult<(u32, Type)> {
        let member_name = self.ctx.get_symbol(member);

        while let Type::Pointer(inner) = value_ty {
            value_ty = self.ctx.get_type(inner);
        }

        while let Type::Func {
            return_ty: Some(return_ty),
            ..
        } = value_ty
        {
            value_ty = self.ctx.get_type(return_ty);
        }

        if let Type::Array(_, _) | Type::String = value_ty {
            let data_sym = self.ctx.string_interner.get_or_intern("data");
            let len_sym = self.ctx.string_interner.get_or_intern("len");
            if member_name.0 == data_sym {
                return Ok((0, Type::Infer(None)));
            } else if member_name.0 == len_sym {
                return Ok((self.ctx.get_pointer_type().bytes(), Type::Infer(None)));
            } else {
                panic!()
            }
        }

        if let Type::Enum { params, .. } = value_ty {
            let offset = self.ctx.enum_tag_size();

            for param in params.borrow().iter() {
                let param_name = match &self.ctx.nodes[param] {
                    Node::EnumDeclParam { name, .. } => *name,
                    _ => panic!("Not a param: {:?}", &self.ctx.nodes[param]),
                };
                let param_name = self.ctx.nodes[param_name].as_symbol().unwrap();

                if param_name == member_name {
                    return Ok((offset, self.ctx.get_type(*param)));
                }
            }

            let member_name_str = self.ctx.get_symbol_str(member);

            return Err(CompileError::Generic(
                format!("Member not found: {:?}", member_name_str),
                self.ctx.ranges[member],
            ));
        }

        let Type::Struct { params, .. } = value_ty else {
            panic!("Not a struct: {value_ty:?}")
        };

        let mut offset = 0;
        for param in params.borrow().iter() {
            let field_name = match &self.ctx.nodes[param] {
                Node::StructDeclParam { name, .. } => *name,
                Node::ValueParam {
                    name: Some(name), ..
                } => *name,
                _ => panic!("Not a param: {:?}", &self.ctx.nodes[param]),
            };
            let field_name = self.ctx.nodes[field_name].as_symbol().unwrap();

            if field_name == member_name {
                return Ok((offset, self.ctx.get_type(*param)));
            }

            offset += self.ctx.id_type_size(*param);
        }

        let member_name_str = self.ctx.get_symbol_str(member);

        return Err(CompileError::Generic(
            format!("Member not found: {:?}", member_name_str),
            self.ctx.ranges[member],
        ));
    }

    fn as_cranelift_value(&mut self, value: Value) -> CraneliftValue {
        match value {
            Value::Register(val) | Value::Reference(val) => val,
            Value::Func(func_id) => {
                let func_ref = *self.declared_func_ids.entry(func_id).or_insert_with(|| {
                    self.ctx
                        .module
                        .declare_func_in_func(func_id, self.builder.func)
                });

                self.builder
                    .ins()
                    .func_addr(self.ctx.get_pointer_type(), func_ref)
            }
            Value::StackSlot(slot) => {
                self.builder
                    .ins()
                    .stack_addr(self.ctx.get_pointer_type(), slot, 0)
            }
        }
    }

    fn store_with_offset(&mut self, id: NodeId, dest: Value, offset: u32) {
        let size = self.ctx.id_type_size(id) as u64;

        let Some(source) = self.ctx.values.get(&id) else {
            return;
        };

        match (source, dest) {
            (_, Value::Func(_)) => {
                panic!("Attempt to store into a function pointer");
            }
            (_, Value::Register(_)) => {
                panic!("Attempt to store into a register");
            }
            (a @ (Value::Func(_) | Value::Register(_)), Value::StackSlot(dest_ss)) => {
                let source_value = self.as_cranelift_value(*a);
                self.builder
                    .ins()
                    .stack_store(source_value, dest_ss, offset as i32);
            }
            (a @ (Value::StackSlot(_) | Value::Reference(_)), Value::StackSlot(dest_ss)) => {
                let source_ptr = self.as_cranelift_value(*a);
                let dest_ptr = self.builder.ins().stack_addr(
                    self.ctx.module.isa().pointer_type(),
                    dest_ss,
                    offset as i32,
                );

                self.emit_small_memory_copy(dest_ptr, source_ptr, size);
            }
            (a @ (Value::Func(_) | Value::Register(_)), Value::Reference(dest_ptr)) => {
                let source_value = self.as_cranelift_value(*a);
                self.builder
                    .ins()
                    .store(MemFlags::new(), source_value, dest_ptr, offset as i32);
            }
            (a @ (Value::StackSlot(_) | Value::Reference(_)), Value::Reference(dest_ptr)) => {
                let source_ptr = self.as_cranelift_value(*a);
                let dest_ptr = self.builder.ins().iadd_imm(dest_ptr, offset as i64);
                self.emit_small_memory_copy(dest_ptr, source_ptr, size);
            }
        }
    }

    fn store_copy(&mut self, id: NodeId, dest: Value) {
        self.store_with_offset(id, dest, 0);
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

    fn id_value(&mut self, id: NodeId) -> CraneliftValue {
        let value = self.ctx.values[&id];
        let value_ty = self.ctx.get_type(id);

        self.id_value_from_parts(value, value_ty)
    }

    fn id_value_from_parts(&mut self, value: Value, ty: Type) -> CraneliftValue {
        match value {
            Value::Func(func_id) => {
                let func_ref = *self.declared_func_ids.entry(func_id).or_insert_with(|| {
                    self.ctx
                        .module
                        .declare_func_in_func(func_id, self.builder.func)
                });

                self.builder
                    .ins()
                    .func_addr(self.ctx.get_pointer_type(), func_ref)
            }
            Value::Register(val) => val,
            Value::Reference(r) => {
                // aggregate types don't fit into a register. Anything else can be eagerly "promoted"
                if self.ctx.is_aggregate_type(ty.clone()) {
                    r
                } else {
                    self.builder
                        .ins()
                        .load(self.ctx.cranelift_type(ty), MemFlags::new(), r, 0)
                }
            }
            Value::StackSlot(slot) => {
                // aggregate types don't fit into a register. Anything else can be eagerly "promoted"
                if self.ctx.is_aggregate_type(ty.clone()) {
                    self.builder
                        .ins()
                        .stack_addr(self.ctx.get_pointer_type(), slot, 0)
                } else {
                    self.builder
                        .ins()
                        .stack_load(self.ctx.cranelift_type(ty), slot, 0)
                }
            }
        }
    }

    #[inline]
    fn create_block(&mut self, name: Option<Sym>) -> Block {
        let block = self.builder.create_block();
        let block = Block::Block(block);
        self.blocks.push((name, block));
        block
    }

    fn find_break_block(&self, name: Option<Sym>) -> Option<CraneliftBlock> {
        match name {
            Some(name) => {
                for (bname, b) in self.blocks.iter().rev() {
                    if let Block::Break(b) = b {
                        if let Some(bname) = bname {
                            if *bname == name {
                                return Some(*b);
                            }
                        }
                    }
                }

                return None;
            }
            None => {
                for (_, b) in self.blocks.iter().rev() {
                    if let Block::Break(b) = b {
                        return Some(*b);
                    }
                }
            }
        }

        return None;
    }

    fn find_continue_block(&self, name: Option<Sym>) -> Option<CraneliftBlock> {
        match name {
            Some(name) => {
                for (bname, b) in self.blocks.iter().rev() {
                    if let Block::Continue(b) = b {
                        if let Some(bname) = bname {
                            if *bname == name {
                                return Some(*b);
                            }
                        }
                    }
                }

                return None;
            }
            None => {
                for (_, b) in self.blocks.iter().rev() {
                    if let Block::Continue(b) = b {
                        return Some(*b);
                    }
                }
            }
        }

        return None;
    }
}

impl<'a> ToplevelCompileContext<'a> {
    pub fn compile_toplevel_id(&mut self, id: NodeId) -> EmptyDraftResult {
        // idempotency
        match self.ctx.values.get(&id) {
            None | Some(Value::Func(_)) => {}
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
                        .push(AbiParam::new(self.ctx.id_cranelift_type(param)));
                }

                if let Some(return_ty) = return_ty {
                    sig.returns
                        .push(AbiParam::new(self.ctx.id_cranelift_type(return_ty)));
                }

                let func_id = self
                    .ctx
                    .module
                    .declare_function(&name_str, Linkage::Export, &sig)
                    .unwrap();

                let mut builder = FunctionBuilder::new(&mut self.codegen_ctx.func, self.func_ctx);
                builder.func.signature = sig;
                builder.func.collect_debug_info();

                self.ctx.values.insert(id, Value::Func(func_id));

                let ebb = builder.create_block();
                builder.append_block_params_for_function_params(ebb);
                builder.switch_to_block(ebb);

                let mut builder_ctx = FunctionCompileContext {
                    ctx: self.ctx,
                    builder,
                    exited_blocks: Default::default(),
                    break_addr: None,
                    declared_func_ids: Default::default(),
                    global_str_ptr: None,
                    blocks: Default::default(),
                    current_block: ebb,
                    current_fn_block: ebb,
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
                        let Node::EnumDeclParam { name, ty, .. } = self.ctx.nodes[param] else {
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
                                .push(AbiParam::new(self.ctx.id_cranelift_type(ty)));
                        }

                        sig.returns
                            .push(AbiParam::new(self.ctx.id_cranelift_type(base)));

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

                        // Assign the enum tag
                        let tag = builder.ins().iconst(types::I64, index as i64);
                        builder.ins().store(MemFlags::new(), tag, slot_addr, 0);

                        if ty.is_some() {
                            // Assign the value to the rest of however many bytes it needs
                            let params = builder.block_params(ebb);
                            let param_value = params[0];

                            if self.ctx.id_is_aggregate_type(param) {
                                let size = self.ctx.id_type_size(param);

                                let source_value = param_value;

                                let dest_value = slot_addr;
                                let dest_value = builder
                                    .ins()
                                    .iadd_imm(dest_value, self.ctx.enum_tag_size() as i64);

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
                                builder.ins().store(
                                    MemFlags::new(),
                                    param_value,
                                    slot_addr,
                                    self.ctx.enum_tag_size() as i32,
                                );
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
