use crate::{NodeId, Range};

// todo(chad): replace this with something that implements Copy
#[derive(Clone, Debug)]
pub enum CompileError {
    Message(String),
    Generic(String, Range),
    Node(String, NodeId),
    Node2(String, NodeId, NodeId),
}

pub type DraftResult<T> = Result<T, CompileError>;
