use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use string_interner::StringInterner;

use cranelift_jit::JITModule;

use crate::{
    AddressableMatch, Args, CompileError, IdVec, Node, NodeId, Range, Scope, ScopeId, Sym, Type,
    TypeMatch, UnificationData, Value,
};

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

#[derive(Default)]
pub struct IdVecs(Vec<Rc<RefCell<Vec<NodeId>>>>);

impl IdVecs {
    pub fn clear(&mut self) {
        self.0.clear();
    }

    pub fn push(&mut self, vec: Rc<RefCell<Vec<NodeId>>>) -> IdVec {
        let id = IdVec(self.0.len());
        self.0.push(vec);
        id
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }
}

impl<T> std::ops::Index<T> for IdVecs
where
    T: Into<IdVec>,
{
    type Output = Rc<RefCell<Vec<NodeId>>>;

    fn index(&self, index: T) -> &Self::Output {
        &self.0[index.into().0]
    }
}

impl From<&IdVec> for IdVec {
    fn from(id_vec: &IdVec) -> Self {
        *id_vec
    }
}

pub type SecondaryMap<V> = HashMap<NodeId, V>;
pub type SecondarySet = HashSet<NodeId>;

pub struct Context {
    pub args: Args,

    pub string_interner: StringInterner,

    pub nodes: DenseStorage<Node>,
    pub ranges: DenseStorage<Range>,
    pub id_vecs: IdVecs,
    pub node_scopes: Vec<ScopeId>,
    pub polymorph_target: bool,

    // stack of returns - pushed when entering parsing a function, popped when exiting
    pub returns: Vec<Vec<NodeId>>,

    // stack of resolves - pushed when entering parsing a resolve block, popped when exiting
    pub resolves: Vec<Vec<NodeId>>,

    pub scopes: Vec<Scope>,
    pub function_scopes: Vec<ScopeId>,
    pub top_scope: ScopeId,

    pub errors: Vec<CompileError>,

    pub top_level: Vec<NodeId>,
    pub funcs: Vec<NodeId>,

    pub types: SecondaryMap<Type>,
    pub type_matches: Vec<TypeMatch>,
    pub type_array_reverse_map: SecondaryMap<usize>,
    pub topo: Vec<NodeId>,
    pub unification_data: UnificationData,
    pub deferreds: Vec<NodeId>,
    pub addressable_matches: Vec<AddressableMatch>,
    pub addressable_array_reverse_map: SecondaryMap<usize>,
    pub in_assign_lhs: bool,

    pub module: JITModule,
    pub values: SecondaryMap<Value>,

    pub addressable_nodes: SecondarySet,
    pub polymorph_sources: SecondarySet,
    pub polymorph_copies: SecondarySet,
    pub completes: SecondarySet,
    pub circular_dependency_nodes: SecondarySet,
    pub circular_concrete_types: SecondarySet,
}

unsafe impl Send for Context {}

impl Context {
    pub fn new(args: Args) -> Self {
        Self {
            args,
            string_interner: StringInterner::new(),

            nodes: Default::default(),
            ranges: Default::default(),
            id_vecs: Default::default(),
            node_scopes: Default::default(),
            addressable_nodes: Default::default(),
            polymorph_target: false,
            polymorph_sources: Default::default(),
            polymorph_copies: Default::default(),

            returns: Default::default(),
            resolves: Default::default(),

            scopes: vec![Scope::new_top()],
            function_scopes: Default::default(),
            top_scope: ScopeId(0),

            errors: Default::default(),

            top_level: Default::default(),
            funcs: Default::default(),

            types: Default::default(),
            type_matches: Default::default(),
            type_array_reverse_map: Default::default(),
            completes: Default::default(),
            topo: Default::default(),
            circular_dependency_nodes: Default::default(),
            circular_concrete_types: Default::default(),
            unification_data: Default::default(),
            deferreds: Default::default(),
            addressable_matches: Default::default(),
            addressable_array_reverse_map: Default::default(),
            in_assign_lhs: false,

            module: Self::make_module(),
            values: Default::default(),
        }
    }

    pub fn reset(&mut self) {
        self.string_interner = StringInterner::new();

        self.nodes.clear();
        self.ranges.clear();
        self.id_vecs.clear();
        self.node_scopes.clear();
        self.addressable_nodes.clear();
        self.polymorph_target = false;
        self.polymorph_sources.clear();
        self.polymorph_copies.clear();
        self.returns.clear();
        self.resolves.clear();

        self.scopes.clear();
        self.scopes.push(Scope::new_top());
        self.function_scopes.clear();
        self.top_scope = ScopeId(0);

        self.errors.clear();

        self.top_level.clear();
        self.funcs.clear();

        self.types.clear();
        self.type_matches.clear();
        self.type_array_reverse_map.clear();
        self.completes.clear();
        self.topo.clear();
        self.circular_dependency_nodes.clear();
        self.circular_concrete_types.clear();
        self.unification_data.reset();
        self.deferreds.clear();
        self.in_assign_lhs = false;

        self.module = Self::make_module();
        self.values.clear();
    }

    pub fn prepare(&mut self) -> Result<(), CompileError> {
        for id in self.top_level.clone() {
            self.assign_type(id);
            self.unify_types();
            self.circular_dependency_nodes.clear();
        }

        let mut hardstop = 0;
        while hardstop < 3 && !self.deferreds.is_empty() {
            for node in std::mem::take(&mut self.deferreds) {
                self.assign_type(node);
                self.unify_types();
            }

            hardstop += 1;
        }

        for id in self.funcs.clone() {
            match &self.nodes[id] {
                Node::Func { params, .. } => {
                    // don't directly codegen a polymorph, wait until it's copied first
                    if self.polymorph_sources.contains(&id) {
                        continue;
                    }

                    for &param in self.id_vecs[params].clone().borrow().iter() {
                        // All structs passed as function args are passed by address (for now...)
                        if let Some(Type::Struct { .. }) = &self.types.get(&param) {
                            self.addressable_nodes.insert(param);
                            self.match_addressable(param, param); // todo(chad): @hack?
                        }
                    }
                }
                _ => (),
            }
        }

        if self.args.print_type_matches {
            // println!(
            //     "type matches: {:#?}",
            //     self.type_matches
            //         .iter()
            //         .map(|tm| {
            //             tm.ids
            //                 .iter()
            //                 .map(|id| (self.nodes[*id].ty(), self.ranges[*id]))
            //                 .collect::<Vec<_>>()
            //         })
            //         .collect::<Vec<_>>()
            // );

            println!(
                "addressability matches: {:#?}",
                self.addressable_matches
                    .iter()
                    .map(|tm| {
                        tm.ids
                            .iter()
                            .map(|id| {
                                (
                                    self.nodes[*id].ty(),
                                    self.ranges[*id],
                                    self.addressable_nodes.contains(id),
                                )
                            })
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
                    "Int literal not assigned".to_string(),
                    *id,
                ));
            }
            if *ty == Type::FloatLiteral {
                self.errors.push(CompileError::Node(
                    "Float literal not assigned".to_string(),
                    *id,
                ));
            }
        }

        if self.errors.is_empty() {
            self.predeclare_functions()
        } else {
            Err(self.errors[0].clone())
        }
    }

    pub fn get_symbol(&self, sym_id: NodeId) -> Sym {
        self.nodes[sym_id].as_symbol().unwrap()
    }

    pub fn report_error(&self, err: CompileError) {
        match err {
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
}
