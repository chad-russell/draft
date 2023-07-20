use crate::{NodeId, Range};

// todo(chad): replace this with something that implements Copy
#[derive(Clone, Debug)]
pub enum CompileError {
    Generic(String, Range),
    Node(String, NodeId),
    Node2(String, NodeId, NodeId),
}
