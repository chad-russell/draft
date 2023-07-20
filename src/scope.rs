use std::collections::BTreeMap;

use crate::{CompileError, Context, NodeId, Sym};

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Hash, PartialOrd, Ord)]
pub struct ScopeId(pub usize);

#[derive(Debug, Clone, Copy)]
pub struct PushedScope(pub ScopeId, pub bool);

#[derive(Default, Debug)]
pub struct Scope {
    pub parent: Option<ScopeId>,
    pub entries: BTreeMap<Sym, NodeId>,
}

impl Scope {
    pub fn new(parent: ScopeId) -> Self {
        Self {
            parent: Some(parent),
            entries: Default::default(),
        }
    }

    pub fn new_top() -> Self {
        Self {
            parent: None,
            entries: Default::default(),
        }
    }
}

impl Context {
    pub fn push_scope(&mut self, is_function_scope: bool) -> PushedScope {
        self.scopes.push(Scope::new(self.top_scope));
        let pushed = PushedScope(self.top_scope, is_function_scope);
        self.top_scope = ScopeId(self.scopes.len() - 1);

        if is_function_scope {
            self.function_scopes.push(self.top_scope);
        }

        pushed
    }

    pub fn pop_scope(&mut self, pushed: PushedScope) {
        self.top_scope = pushed.0;
        if pushed.1 {
            self.function_scopes.pop();
        }
    }

    pub fn scope_insert(&mut self, sym: Sym, id: NodeId) {
        if self.scopes[self.top_scope.0].entries.contains_key(&sym) {
            self.errors.push(CompileError::Node(
                format!(
                    "Duplicate symbol definition '{}'",
                    self.string_interner.resolve(sym.0).unwrap()
                ),
                id,
            ));
        }
        self.scopes[self.top_scope.0].entries.insert(sym, id);
    }

    pub fn scope_get_with_scope_id(&self, sym: Sym, scope: ScopeId) -> Option<NodeId> {
        match self.scopes[scope.0].entries.get(&sym).copied() {
            Some(id) => Some(id),
            None => match self.scopes[scope.0].parent {
                Some(parent) => self.scope_get_with_scope_id(sym, parent),
                None => None,
            },
        }
    }

    pub fn scope_get(&self, sym: Sym, node: NodeId) -> Option<NodeId> {
        let scope_id = self.node_scopes[node.0];
        self.scope_get_with_scope_id(sym, scope_id)
    }
}
