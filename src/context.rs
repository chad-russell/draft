use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

use cranelift_codegen::ir::GlobalValue;
use cranelift_jit::JITModule;
use cranelift_module::{DataId, Module};
use cranelift_object::ObjectModule;
use string_interner::StringInterner;

use crate::{
    AddressableMatch, Args, CompileError, EmptyDraftResult, Location, Node, NodeId, Range,
    RopeySource, Scope, ScopeId, Scopes, Source, SourceInfo, StaticStrSource, Sym, Type, TypeMatch,
    Value,
};

pub enum ObjectOrJITModule {
    Object(ObjectModule),
    JIT(JITModule),
    Consumed, // Producing object code consumes the module
}

impl ObjectOrJITModule {
    pub fn as_module(&self) -> &dyn Module {
        match self {
            ObjectOrJITModule::Object(module) => module,
            ObjectOrJITModule::JIT(module) => module,
            _ => panic!("Module was already consumed"),
        }
    }

    pub fn as_module_mut(&mut self) -> &mut dyn Module {
        match self {
            ObjectOrJITModule::Object(module) => module,
            ObjectOrJITModule::JIT(module) => module,
            _ => panic!("Module was already consumed"),
        }
    }
}

#[derive(Clone)]
pub struct DenseStorage<T>(Vec<T>);

impl<T> Default for DenseStorage<T> {
    fn default() -> Self {
        Self(Vec::new())
    }
}

impl<T> DenseStorage<T> {
    pub fn clear(&mut self) {
        self.0.clear();
    }

    pub fn push(&mut self, val: T) -> NodeId {
        let id = NodeId(self.0.len());
        self.0.push(val);
        id
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn iter(&self) -> core::slice::Iter<'_, T> {
        self.0.iter()
    }
}

impl<T, I> std::ops::Index<I> for DenseStorage<T>
where
    I: Into<NodeId>,
{
    type Output = T;

    fn index(&self, index: I) -> &Self::Output {
        &self.0[index.into().0]
    }
}

impl<T, I> std::ops::IndexMut<I> for DenseStorage<T>
where
    I: Into<NodeId>,
{
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        &mut self.0[index.into().0]
    }
}

pub type SecondaryMap<V> = HashMap<NodeId, V>;
pub type SecondaryScopeMap<V> = HashMap<ScopeId, V>;
pub type SecondarySet = HashSet<NodeId>;

pub struct Context {
    pub args: Args,

    pub string_interner: StringInterner,
    pub file_sources: HashMap<PathBuf, &'static str>,
    pub file_modules: HashMap<PathBuf, NodeId>,
    pub line_offsets: HashMap<&'static str, Vec<usize>>, // file -> (line -> char_offset)

    pub nodes: DenseStorage<Node>,
    pub ranges: DenseStorage<Range>,
    pub node_scopes: DenseStorage<ScopeId>,
    pub polymorph_target: bool,
    pub string_literals: Vec<NodeId>,
    pub string_literal_offsets: SecondaryMap<usize>, // Offset into the global data segment for the start of the string literal with

    pub hardstop: usize,
    pub max_hardstop: usize,

    // todo(chad): switch to MaybeUninit at some point
    pub string_literal_data_id: Option<DataId>,
    pub ty_empty: Option<NodeId>,
    pub ty_u8: Option<NodeId>, // For easily creating string pointers, which must be *u8. We need to have a node ID to point to

    // stack of returns - pushed when entering parsing a function, popped when exiting
    pub returns: Vec<Vec<NodeId>>,

    // stack of breaks - pushed when entering parsing a break-able block, popped when exiting
    pub breaks: Vec<Vec<NodeId>>,

    // stack of break labels - pushed when doing semantic analysis on a break-able block with a label, popped when exiting
    pub break_labels: Vec<Option<Sym>>,

    // stack of continues - pushed when entering parsing a continue-able block, popped when exiting
    pub continues: Vec<Vec<NodeId>>,

    // stack of continue labels - pushed when doing semantic analysis on a continue-able block with a label, popped when exiting
    pub continue_labels: Vec<Option<Sym>>,

    // associates a scope with all defer blocks which need to be executed when the scope is exited
    pub defers: SecondaryScopeMap<Vec<NodeId>>,

    pub scopes: Scopes,
    pub top_scope: ScopeId,

    pub errors: Vec<CompileError>,

    pub top_level: Vec<NodeId>,
    pub funcs: Vec<NodeId>,
    pub imports: Vec<NodeId>,

    pub types: SecondaryMap<Type>,
    pub type_matches: Vec<TypeMatch>,
    pub type_array_reverse_map: SecondaryMap<usize>,
    pub topo: Vec<NodeId>,
    pub deferreds: Vec<NodeId>,
    pub addressable_matches: Vec<AddressableMatch>,
    pub addressable_array_reverse_map: SecondaryMap<usize>,
    pub in_assign_lhs: bool,

    pub module: ObjectOrJITModule,
    pub values: SecondaryMap<Value>,
    pub global_values: SecondaryMap<(DataId, Option<GlobalValue>)>,
    pub global_builder_id: u32,
    pub symbol_data_ids: HashMap<Sym, DataId>,
    pub polymorph_sources: SecondarySet,

    pub polymorph_copies: SecondarySet,
    pub completes: SecondarySet,
    pub transparent_deferreds: SecondarySet,
    pub circular_dependency_nodes: SecondarySet,
    pub circular_concrete_types: SecondarySet,
    pub impls: SecondarySet,

    pub type_info_decl: Option<NodeId>,
}

unsafe impl Send for Context {}

impl Context {
    pub fn new(args: Args) -> Self {
        let is_jit = args.jit;

        let mut ctx = Self {
            args,

            string_interner: StringInterner::new(),
            file_sources: Default::default(),
            file_modules: Default::default(),
            line_offsets: Default::default(),

            nodes: Default::default(),
            ranges: Default::default(),
            node_scopes: Default::default(),
            polymorph_target: false,
            polymorph_sources: Default::default(),
            polymorph_copies: Default::default(),
            string_literals: Default::default(),
            string_literal_offsets: Default::default(),
            string_literal_data_id: None,
            ty_empty: None,
            ty_u8: None,

            hardstop: 0,
            max_hardstop: 8,

            returns: Default::default(),
            breaks: Default::default(),
            break_labels: Default::default(),
            continues: Default::default(),
            continue_labels: Default::default(),
            defers: Default::default(),

            scopes: Scopes::new(vec![Scope::new_top()]),
            top_scope: ScopeId(0),

            errors: Default::default(),

            top_level: Default::default(),
            funcs: Default::default(),
            imports: Default::default(),

            types: Default::default(),
            type_matches: Default::default(),
            type_array_reverse_map: Default::default(),
            completes: Default::default(),
            transparent_deferreds: Default::default(),
            topo: Default::default(),
            circular_dependency_nodes: Default::default(),
            circular_concrete_types: Default::default(),
            impls: Default::default(),
            deferreds: Default::default(),
            addressable_matches: Default::default(),
            addressable_array_reverse_map: Default::default(),
            in_assign_lhs: false,

            module: Self::make_module(is_jit),
            values: Default::default(),
            global_values: Default::default(),
            global_builder_id: 0,
            symbol_data_ids: Default::default(),

            type_info_decl: None,
        };

        ctx.setup_default_types();

        ctx
    }

    pub fn reset(&mut self) {
        self.string_interner = StringInterner::new();
        self.file_sources.clear();
        self.file_modules.clear();

        self.nodes.clear();
        self.ranges.clear();
        self.node_scopes.clear();
        self.polymorph_target = false;
        self.string_literals.clear();
        self.string_literal_offsets.clear();
        self.string_literal_data_id = None;
        self.ty_empty = None;
        self.polymorph_sources.clear();
        self.polymorph_copies.clear();
        self.returns.clear();
        self.breaks.clear();
        self.continues.clear();

        self.scopes.clear();
        self.scopes.push(Scope::new_top());
        self.top_scope = ScopeId(0);

        self.errors.clear();

        self.top_level.clear();
        self.funcs.clear();
        self.imports.clear();

        self.types.clear();
        self.type_matches.clear();
        self.type_array_reverse_map.clear();
        self.completes.clear();
        self.transparent_deferreds.clear();
        self.topo.clear();
        self.circular_dependency_nodes.clear();
        self.circular_concrete_types.clear();
        self.impls.clear();
        self.deferreds.clear();
        self.in_assign_lhs = false;

        match self.module {
            ObjectOrJITModule::Object(_) => {
                self.module = Self::make_module(false);
            }
            ObjectOrJITModule::JIT(_) => {
                self.module = Self::make_module(true);
            }
            _ => {}
        }

        self.values.clear();
        self.global_values.clear();
        self.global_builder_id = 0;
        self.symbol_data_ids.clear();

        self.setup_default_types();
    }

    fn setup_default_types(&mut self) {
        self.ty_empty = Some(self.push_node(
            Range::new(Location::default(), Location::default(), ""),
            Node::Type(Type::Empty),
        ));
        self.types.insert(self.ty_empty.unwrap(), Type::Empty);

        self.ty_u8 = Some(self.push_node(
            Range::new(Location::default(), Location::default(), ""),
            Node::Type(Type::U8),
        ));
        self.types.insert(self.ty_u8.unwrap(), Type::U8);
    }

    pub fn make_source_info_from_file(&mut self, path: PathBuf) -> SourceInfo<StaticStrSource> {
        let source_str = std::fs::read_to_string(&path).unwrap();
        let source_str: &'static str = Box::leak(source_str.into_boxed_str());

        self.file_sources.insert(path.clone(), source_str);

        let source = StaticStrSource::from_static_str(source_str);
        let chars_left = source.char_count();
        let path: &'static str = Box::leak(path.into_boxed_path().to_str().unwrap().into());

        let mut char_offset = 0;
        let mut char_offset_map = vec![0];
        for c in source_str.chars() {
            char_offset += 1;
            if c == '\n' {
                char_offset_map.push(char_offset);
            }
        }
        self.line_offsets.insert(path, char_offset_map);

        SourceInfo {
            path,
            source,
            chars_left,
            loc: Default::default(),
            top: Default::default(),
            second: Default::default(),
        }
    }

    pub fn make_ropey_source_info_from_file(&mut self, file_name: &str) -> SourceInfo<RopeySource> {
        let path = PathBuf::from(file_name);
        let source_str = std::fs::read_to_string(&path).unwrap();
        let source_str: &'static str = Box::leak(source_str.into_boxed_str());

        self.file_sources.insert(path.clone(), source_str);

        let source = RopeySource::from_str(source_str);
        let chars_left = source.char_count();
        let path: &'static str = Box::leak(path.into_boxed_path().to_str().unwrap().into());

        let mut char_offset = 0;
        let mut char_offset_map = Vec::new();
        for c in source_str.chars() {
            char_offset += 1;
            if c == '\n' {
                char_offset_map.push(char_offset);
            }
        }
        self.line_offsets.insert(path, char_offset_map);

        SourceInfo {
            path,
            source,
            chars_left,
            loc: Default::default(),
            top: Default::default(),
            second: Default::default(),
        }
    }

    pub fn make_source_info_from_range(&mut self, range: Range) -> SourceInfo<StaticStrSource> {
        let source_path = PathBuf::from(range.source_path);
        let source = self.file_sources.get(&source_path).unwrap();

        let start_char_offset = self.char_offset_for_location(range.source_path, range.start);
        let end_char_offset = self.char_offset_for_location(range.source_path, range.end);
        let source = &source[start_char_offset..end_char_offset];

        let source = StaticStrSource::from_static_str(source);
        let chars_left = source.char_count();
        let source_path = Box::leak(source_path.into_boxed_path().to_str().unwrap().into());

        SourceInfo {
            path: source_path,
            source,
            chars_left,
            loc: range.start,
            top: Default::default(),
            second: Default::default(),
        }
    }

    pub fn defer_on(&mut self, ids: &[NodeId], error: CompileError) {
        self.deferreds.extend(ids.iter());
        if self.hardstop == self.max_hardstop - 1 {
            self.errors.push(error);
        }
    }

    pub fn prepare(&mut self) -> EmptyDraftResult {
        // self.assign_type(self.type_info_decl.unwrap());
        // self.unify_types();
        // dbg!(self.id_type_size(self.type_info_decl.unwrap()));

        // self.types
        //     .insert(self.type_info_decl.unwrap(), Type::TypeInfo);
        // self.circular_dependency_nodes
        //     .insert(self.type_info_decl.unwrap());
        // self.completes.insert(self.type_info_decl.unwrap());

        for id in self.top_level.clone() {
            self.assign_type(id);
            self.unify_types();
            self.circular_dependency_nodes.clear();
        }

        self.hardstop = 0;
        while self.hardstop < self.max_hardstop && !self.deferreds.is_empty() {
            for node in std::mem::take(&mut self.deferreds) {
                self.assign_type(node);
                self.unify_types();
                self.circular_dependency_nodes.clear();
            }

            self.hardstop += 1;
        }

        for id in 0..self.nodes.len() {
            let id = NodeId(id);

            match self.nodes[id] {
                Node::Cast { ty, value } => {
                    // todo(chad): find a more robust way of doing this.
                    // Basically looping through all nodes isn't great because we will also loop through
                    // nodes in polymorph sources. So for now if the node hasn't been typechecked, then we just skip it.

                    if !self.types.contains_key(&ty) || !self.types.contains_key(&value) {
                        continue;
                    }

                    let ty = self.types[&ty].clone();
                    let value_ty = self.types[&value].clone();

                    let Type::Pointer(_) = ty else {
                        self.errors.push(CompileError::Node(
                            format!("Can only cast to pointer types (casting to {:?})", ty),
                            id,
                        ));
                        break;
                    };

                    let Type::Pointer(_) = value_ty else {
                        self.errors.push(CompileError::Node(
                            format!("Can only cast from pointer types (found {:?})", value_ty),
                            id,
                        ));
                        break;
                    };
                }
                _ => {}
            }
        }

        if self.args.print_type_matches {
            println!(
                "type matches: {:#?}",
                self.type_matches
                    .iter()
                    .map(|tm| {
                        tm.ids
                            .iter()
                            .map(|id| (self.nodes[id].ty(), self.ranges[*id]))
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            );
        }

        // Check if any types are still unassigned
        for (id, ty) in self.types.iter() {
            if let Type::Infer(_) = *ty {
                self.errors
                    .push(CompileError::Node("Type not assigned".to_string(), *id));
            }
            if *ty == Type::IntLiteral {
                self.errors.push(CompileError::Node(
                    "Int literal could not be sized".to_string(),
                    *id,
                ));
            }
            if *ty == Type::FloatLiteral {
                self.errors.push(CompileError::Node(
                    "Float literal could not be sized".to_string(),
                    *id,
                ));
            }
        }

        if !self.errors.is_empty() {
            return Err(self.errors[0].clone());
        }

        self.predeclare_string_constants()?;
        self.predeclare_functions()?;
        // self.insert_type_infos_into_global_data()?;
        self.define_functions()?;

        Ok(())
    }

    pub fn get_symbol(&self, sym_id: NodeId) -> Sym {
        self.nodes[sym_id].as_symbol().unwrap()
    }

    pub fn get_symbol_str(&self, sym_id: NodeId) -> &str {
        let sym = self.get_symbol(sym_id);
        self.string_interner.resolve(sym.0).unwrap()
    }

    pub fn report_error(&self, err: CompileError) {
        match err {
            CompileError::Message(msg) => {
                println!("{}", msg);
            }
            CompileError::Generic(msg, range) => {
                println!("{:?}: {}", range, msg);
            }
            CompileError::Node(msg, id) => {
                println!("{:?}: {}", self.ranges[id], msg);
            }
            CompileError::Node2(msg, id1, id2) => {
                println!("{:?}, {:?}: {}", self.ranges[id1], self.ranges[id2], msg);
            }
        }
    }

    pub fn char_offset_for_location(&self, source_path: &str, start: Location) -> usize {
        self.line_offsets[source_path][start.line - 1] + start.col
    }

    pub fn char_span(&self, range: Range) -> usize {
        return self.char_offset_for_location(range.source_path, range.end)
            - self.char_offset_for_location(range.source_path, range.start);
    }

    pub fn module_mut(&mut self) -> &mut dyn Module {
        self.module.as_module_mut()
    }

    pub fn module(&self) -> &dyn Module {
        self.module.as_module()
    }
}
