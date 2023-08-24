use std::{cell::RefCell, rc::Rc};

use crate::{NumericSpecification, Op, ScopeId, Sym, Type};

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Hash, PartialOrd, Ord)]
pub struct NodeId(pub usize);

impl From<&NodeId> for NodeId {
    fn from(id: &NodeId) -> Self {
        *id
    }
}

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
pub enum AsCastStyle {
    None,
    StaticToDynamicArray,
    StructToDynamicArray,
}

pub type IdVec = Rc<RefCell<Vec<NodeId>>>;

#[derive(Debug, Clone)]
pub enum Node {
    Symbol(Sym),
    PolySpecialize {
        sym: Sym,
        overrides: IdVec,
        copied: Option<NodeId>,
    },
    PolySpecializeOverride {
        sym: NodeId,
        ty: NodeId,
    },
    IntLiteral(i64, NumericSpecification),
    FloatLiteral(f64, NumericSpecification),
    StringLiteral(Sym),
    BoolLiteral(bool),
    Type(Type),
    Return(Option<NodeId>),
    Break(Option<NodeId>, Option<Sym>),
    Continue(Option<Sym>),
    Let {
        name: NodeId,
        ty: Option<NodeId>,
        expr: Option<NodeId>,
        transparent: bool,
    },
    Assign {
        name: NodeId,
        expr: NodeId,
        is_store: bool,
    },
    FnDefinition {
        name: Option<NodeId>,
        scope: ScopeId,
        params: IdVec,
        return_ty: Option<NodeId>,
        stmts: IdVec,
        returns: IdVec,
        transparent: bool,
    },
    Block {
        label: Option<Sym>,
        stmts: IdVec,
        breaks: IdVec,
        is_standalone: bool,
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
        transparent: bool,
    },
    EnumDeclParam {
        name: NodeId,
        ty: Option<NodeId>,
    },
    FnDeclParam {
        name: NodeId,
        ty: Option<NodeId>,
        default: Option<NodeId>,
        index: u16,
        transparent: bool,
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
    ThreadingCall {
        func: NodeId,
        params: IdVec,
    },
    ThreadingParamTarget,
    ArrayAccess {
        array: NodeId,
        index: NodeId,
    },
    StructDeclaration {
        name: Option<NodeId>,
        params: IdVec,
        scope: ScopeId,
    },
    StructLiteral {
        name: Option<NodeId>,
        params: IdVec,
    },
    EnumDefinition {
        name: NodeId,
        params: IdVec,
    },
    ArrayLiteral {
        members: IdVec,
        ty: NodeId,
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
        then_label: Option<Sym>,
        else_block: NodeElse,
        else_label: Option<Sym>,
    },
    For {
        label: NodeId,
        iterable: NodeId,
        block: NodeId,
        block_label: Option<Sym>,
    },
    While {
        cond: NodeId,
        block: NodeId,
        block_label: Option<Sym>,
    },
    Cast {
        ty: NodeId,
        value: NodeId,
    },
    SizeOf(NodeId),
    AsCast {
        value: NodeId,
        ty: NodeId,
        style: AsCastStyle,
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
            Node::PolySpecialize { .. } => "PolySpecialize".to_string(),
            Node::PolySpecializeOverride { .. } => "PolySpecializeOverride".to_string(),
            Node::IntLiteral(_, _) => "IntLiteral".to_string(),
            Node::FloatLiteral(_, _) => "FloatLiteral".to_string(),
            Node::BoolLiteral(_) => "BoolLiteral".to_string(),
            Node::StringLiteral(_) => "StringLiteral".to_string(),
            Node::Type(_) => "Type".to_string(),
            Node::Return(_) => "Return".to_string(),
            Node::Let { .. } => "Let".to_string(),
            Node::Assign { .. } => "Set".to_string(),
            Node::FnDefinition { .. } => "Func".to_string(),
            Node::Block { .. } => "Block".to_string(),
            Node::Break(_, _) => "Break".to_string(),
            Node::Continue(_) => "Continue".to_string(),
            Node::Extern { .. } => "Extern".to_string(),
            Node::StructDeclParam { .. } => "StructDeclParam".to_string(),
            Node::EnumDeclParam { .. } => "EnumDeclParam".to_string(),
            Node::FnDeclParam { .. } => "FuncDeclParam".to_string(),
            Node::ValueParam { .. } => "ValueParam".to_string(),
            Node::BinOp { .. } => "BinOp".to_string(),
            Node::Call { .. } => "Call".to_string(),
            Node::ThreadingCall { .. } => "ThreadingCall".to_string(),
            Node::ThreadingParamTarget => "ThreadingParamTarget".to_string(),
            Node::StructDeclaration { .. } => "StructDefinition".to_string(),
            Node::EnumDefinition { .. } => "EnumDefinition".to_string(),
            Node::StructLiteral { .. } => "StructLiteral".to_string(),
            Node::MemberAccess { .. } => "MemberAccess".to_string(),
            Node::StaticMemberAccess { .. } => "StaticMemberAccess".to_string(),
            Node::AddressOf(_) => "AddressOf".to_string(),
            Node::Deref(_) => "Deref".to_string(),
            Node::If { .. } => "If".to_string(),
            Node::For { .. } => "For".to_string(),
            Node::While { .. } => "While".to_string(),
            Node::ArrayLiteral { .. } => "ArrayLiteral".to_string(),
            Node::ArrayAccess { .. } => "ArrayAccess".to_string(),
            Node::Cast { .. } => "Cast".to_string(),
            Node::SizeOf(_) => "SizeOf".to_string(),
            Node::AsCast { .. } => "AsCast".to_string(),
        }
    }
}
