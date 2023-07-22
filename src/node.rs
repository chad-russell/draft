use crate::{NumericSpecification, Op, ScopeId, Sym, Type};

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Hash, PartialOrd, Ord)]
pub struct NodeId(pub usize);

impl From<&NodeId> for NodeId {
    fn from(id: &NodeId) -> Self {
        *id
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Hash, PartialOrd, Ord)]
pub struct IdVec(pub usize);

#[derive(Debug, Clone, Copy)]
pub enum StaticMemberResolution {
    Node(NodeId),
    EnumConstructor { base: NodeId, index: u16 },
}

#[derive(Debug, Clone, Copy)]
pub enum NodeElse {
    Block(NodeId),
    If(NodeId),
    None,
}

#[derive(Debug, Clone, Copy)]
pub enum Node {
    Symbol(Sym),
    IntLiteral(i64, NumericSpecification),
    FloatLiteral(f64, NumericSpecification),
    BoolLiteral(bool),
    Type(Type),
    Return(Option<NodeId>),
    Resolve(Option<NodeId>),
    Let {
        name: NodeId,
        ty: Option<NodeId>,
        expr: Option<NodeId>,
    },
    Assign {
        name: NodeId,
        expr: NodeId,
        is_store: bool,
    },
    Func {
        name: Option<NodeId>,
        scope: ScopeId,
        params: IdVec,
        return_ty: Option<NodeId>,
        stmts: IdVec,
        returns: IdVec,
    },
    Block {
        scope: ScopeId,
        stmts: IdVec,
        resolves: IdVec,
    },
    Extern {
        name: NodeId,
        params: IdVec,
        return_ty: Option<NodeId>,
    },
    StructDeclParam {
        name: NodeId,
        ty: Option<NodeId>,
        default: Option<NodeId>,
        index: u16,
    },
    EnumDeclParam {
        name: NodeId,
        ty: Option<NodeId>,
    },
    FuncDeclParam {
        name: NodeId,
        ty: Option<NodeId>,
        default: Option<NodeId>,
        index: u16,
    },
    ValueParam {
        name: Option<NodeId>,
        value: NodeId,
        index: u16,
    },
    BinOp {
        op: Op,
        lhs: NodeId,
        rhs: NodeId,
    },
    Call {
        func: NodeId,
        params: IdVec,
    },
    StructDefinition {
        name: NodeId,
        params: IdVec,
    },
    StructLiteral {
        name: Option<NodeId>,
        params: IdVec,
    },
    EnumDefinition {
        name: NodeId,
        params: IdVec,
    },
    MemberAccess {
        value: NodeId,
        member: NodeId,
    },
    StaticMemberAccess {
        value: NodeId,
        member: NodeId,
        resolved: Option<StaticMemberResolution>,
    },
    AddressOf(NodeId),
    Deref(NodeId),
    If {
        cond: NodeId,
        then_block: NodeId,
        else_block: NodeElse,
    },
}

impl Node {
    pub fn as_symbol(&self) -> Option<Sym> {
        match self {
            Node::Symbol(sym) => Some(*sym),
            _ => None,
        }
    }

    pub fn ty(&self) -> String {
        match self {
            Node::Symbol(_) => "Symbol".to_string(),
            Node::IntLiteral(_, _) => "IntLiteral".to_string(),
            Node::FloatLiteral(_, _) => "FloatLiteral".to_string(),
            Node::BoolLiteral(_) => "BoolLiteral".to_string(),
            Node::Type(_) => "Type".to_string(),
            Node::Return(_) => "Return".to_string(),
            Node::Let { .. } => "Let".to_string(),
            Node::Assign { .. } => "Set".to_string(),
            Node::Func { .. } => "Func".to_string(),
            Node::Block { .. } => "Block".to_string(),
            Node::Resolve(_) => "Block".to_string(),
            Node::Extern { .. } => "Extern".to_string(),
            Node::StructDeclParam { .. } => "StructDeclParam".to_string(),
            Node::EnumDeclParam { .. } => "EnumDeclParam".to_string(),
            Node::FuncDeclParam { .. } => "FuncDeclParam".to_string(),
            Node::ValueParam { .. } => "ValueParam".to_string(),
            Node::BinOp { .. } => "BinOp".to_string(),
            Node::Call { .. } => "Call".to_string(),
            Node::StructDefinition { .. } => "StructDefinition".to_string(),
            Node::EnumDefinition { .. } => "EnumDefinition".to_string(),
            Node::StructLiteral { .. } => "StructLiteral".to_string(),
            Node::MemberAccess { .. } => "MemberAccess".to_string(),
            Node::StaticMemberAccess { .. } => "StaticMemberAccess".to_string(),
            Node::AddressOf(_) => "AddressOf".to_string(),
            Node::Deref(_) => "Deref".to_string(),
            Node::If { .. } => "If".to_string(),
        }
    }
}
