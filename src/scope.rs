use std::collections::BTreeMap;

use crate::{CompileError, Context, NodeId, Sym};

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Hash, PartialOrd, Ord)]
pub struct ScopeId(pub usize);

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

pub struct Scopes(Vec<Scope>);

impl Scopes {
    pub fn new(v: Vec<Scope>) -> Self {
        Self(v)
    }

    pub fn push(&mut self, scope: Scope) {
        self.0.push(scope);
    }

    pub fn get(&self, scope_id: ScopeId) -> Option<&Scope> {
        self.0.get(scope_id.0)
    }

    pub fn clear(&mut self) {
        self.0.clear();
    }
}

impl Default for Scopes {
    fn default() -> Self {
        Self(Vec::new())
    }
}

impl<I> std::ops::Index<I> for Scopes
where
    I: Into<ScopeId>,
{
    type Output = Scope;

    fn index(&self, index: I) -> &Self::Output {
        &self.0[index.into().0]
    }
}

impl<I> std::ops::IndexMut<I> for Scopes
where
    I: Into<ScopeId>,
{
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        &mut self.0[index.into().0]
    }
}

impl Context {
    pub fn push_scope(&mut self) -> ScopeId {
        self.scopes.0.push(Scope::new(self.top_scope));
        let pushed = self.top_scope;
        self.top_scope = ScopeId(self.scopes.0.len() - 1);

        pushed
    }

    pub fn pop_scope(&mut self, pushed: ScopeId) {
        self.top_scope = pushed;
    }

    pub fn scope_insert(&mut self, sym: Sym, id: NodeId) {
        self.scope_insert_into_scope_id(sym, id, self.top_scope);
    }

    pub fn scope_insert_into_scope_id(&mut self, sym: Sym, id: NodeId, scope_id: ScopeId) {
        if self.scopes[scope_id].entries.contains_key(&sym) {
            self.errors.push(CompileError::Node(
                format!(
                    "Duplicate symbol definition '{}'",
                    self.string_interner.resolve(sym.0).unwrap()
                ),
                id,
            ));
        }
        self.scopes[scope_id].entries.insert(sym, id);
    }

    pub fn scope_get_with_scope_id(&self, sym: Sym, scope: ScopeId) -> Option<NodeId> {
        match self.scopes[scope].entries.get(&sym).copied() {
            Some(id) => Some(id),
            None => match self.scopes[scope].parent {
                Some(parent) => self.scope_get_with_scope_id(sym, parent),
                None => None,
            },
        }
    }

    pub fn scope_get(&self, sym: Sym, node: NodeId) -> Option<NodeId> {
        let scope_id = self.node_scopes[node];
        self.scope_get_with_scope_id(sym, scope_id)
    }
}
