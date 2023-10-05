use std::{cell::RefCell, rc::Rc};

use tracing::instrument;

use crate::{
    AsCastStyle, CompileError, Context, DraftResult, EmptyDraftResult, IdVec, IfCond, Node,
    NodeElse, NodeId, NumericSpecification, Op, ParseTarget, ScopeId, StaticMemberResolution, Sym,
};

#[derive(Debug, Clone, PartialEq)]
pub enum Type {
    Empty,
    Infer(Option<Sym>),
    IntLiteral,
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    FloatLiteral,
    F32,
    F64,
    Bool,
    String,
    Func {
        input_tys: IdVec,
        return_ty: Option<NodeId>,
    },
    Struct {
        decl: Option<NodeId>,
        scope: Option<ScopeId>,
        params: IdVec,
    },
    Enum {
        decl: Option<NodeId>,
        scope: Option<ScopeId>,
        params: IdVec,
    },
    EnumNoneType, // Type given to enum members with no storage. We need a distinct type to differentiate it from being unassigned
    Pointer(NodeId),
    Array(NodeId, ArrayLen),
}

impl Type {
    #[instrument(name = "Type::is_basic", skip_all)]
    pub fn is_basic(&self) -> bool {
        matches!(
            &self,
            Type::IntLiteral
                | Type::I8
                | Type::I16
                | Type::I32
                | Type::I64
                | Type::U8
                | Type::U16
                | Type::U32
                | Type::U64
                | Type::FloatLiteral
                | Type::F32
                | Type::F64
        )
    }

    #[instrument(name = "Type::is_int", skip_all)]
    pub fn is_int(&self) -> bool {
        matches!(
            &self,
            Type::IntLiteral
                | Type::I8
                | Type::I16
                | Type::I32
                | Type::I64
                | Type::U8
                | Type::U16
                | Type::U32
                | Type::U64
        )
    }

    #[instrument(name = "Type::is_float", skip_all)]
    pub fn is_float(&self) -> bool {
        matches!(&self, Type::FloatLiteral | Type::F32 | Type::F64)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ArrayLen {
    Some(usize),
    Infer,
    None,
}

#[derive(Debug)]
pub struct TypeMatch {
    pub changed: bool,
    pub unified: Type,
    pub ids: Vec<NodeId>,
}

#[derive(Debug)]
pub struct AddressableMatch {
    pub changed: bool,
    pub unified: bool,
    pub ids: Vec<NodeId>,
}

impl Context {
    #[instrument(skip_all)]
    pub fn unify_types(&mut self) {
        for uid in 0..self.type_matches.len() {
            if !self.type_matches[uid].changed {
                continue;
            }
            self.type_matches[uid].changed = false;

            let most_specific_ty = self.type_matches[uid].unified.clone();
            if matches!(most_specific_ty, Type::Infer(_)) {
                continue;
            }

            for &id in self.type_matches[uid].ids.iter() {
                if let (
                    Type::Struct {
                        params: sparams, ..
                    }
                    | Type::Enum {
                        params: sparams, ..
                    },
                    Type::Struct { params, .. } | Type::Enum { params, .. },
                ) = (most_specific_ty.clone(), self.get_type(id))
                {
                    for (sparam, param) in sparams.borrow().iter().zip(params.borrow().iter()) {
                        let Node::StructDeclParam {
                            transparent: stransparent,
                            ..
                        } = self.nodes[*sparam].clone()
                        else {
                            continue;
                        };
                        let Node::StructDeclParam { transparent, .. } = self.nodes[*param].clone()
                        else {
                            continue;
                        };
                        if stransparent && !transparent {
                            self.errors.push(CompileError::Node2(
                                "Could not match types: struct fields differ in transparency"
                                    .to_string(),
                                *sparam,
                                *param,
                            ));
                        }
                    }
                }
                self.types.insert(id, most_specific_ty.clone());
            }
        }
    }

    #[instrument(skip_all)]
    pub fn unify(&mut self, first: Type, second: Type, err_ids: (NodeId, NodeId)) -> Type {
        match (first.clone(), second.clone()) {
            (a, b) if a == b => a,

            // Types/Inferreds get coerced to anything
            (Type::Infer(_), other) | (other, Type::Infer(_)) => other,

            // Check int/float literals match
            (Type::IntLiteral, bt) | (bt, Type::IntLiteral) if bt.is_int() => bt,
            (Type::FloatLiteral, bt) | (bt, Type::FloatLiteral) if bt.is_float() => bt,

            (
                Type::Struct {
                    decl: d1,
                    scope: s1,
                    ..
                }
                | Type::Enum {
                    decl: d1,
                    scope: s1,
                    ..
                },
                Type::Struct {
                    decl: d2,
                    scope: s2,
                    ..
                }
                | Type::Enum {
                    decl: d2,
                    scope: s2,
                    ..
                },
            ) => {
                // prefer structs that relate to a declaration, to structs that are inferred
                match (d1, d2) {
                    (Some(_), None) => first,
                    (None, Some(_)) => second,
                    // prefer structs that have a scope to those that don't
                    _ => match (s1, s2) {
                        (Some(_), None) => first,
                        (None, Some(_)) => second,
                        _ => first, // if both are named or both are unnamed, and both either have a scope or don't, then it doesn't matter which is chosen
                    },
                }
            }

            // Coerce array with an inferred length to an array with a known length
            (Type::Array(_, ArrayLen::Some(_)), Type::Array(_, ArrayLen::Infer)) => first,
            (Type::Array(_, ArrayLen::Infer), Type::Array(_, ArrayLen::Some(_))) => second,

            // For aggregate types, the type matcher should have already detected a mismatch
            // so it doesn't really matter which is chosen
            (Type::Func { .. }, Type::Func { .. })
            | (Type::Pointer(_), Type::Pointer(_))
            | (Type::Array(_, _), Type::Array(_, _)) => first,

            // Anything else
            _ => {
                self.errors.push(CompileError::Node2(
                    format!("Type mismatch: {:?} and {:?}", first, second),
                    err_ids.0,
                    err_ids.1,
                ));
                Type::Infer(None)
            }
        }
    }

    #[instrument(skip_all)]
    pub fn rearrange_params(
        &mut self,
        given: &[NodeId],
        given_scope: Option<ScopeId>,
        decl: &[NodeId],
        err_id: NodeId,
    ) -> DraftResult<Vec<NodeId>> {
        let given_len = given.len();

        if given.len() > decl.len() {
            return Err(CompileError::Node(
                format!(
                    "Too many parameters given: expected {}, got {}",
                    decl.len(),
                    given.len()
                ),
                err_id,
            ));
        }

        let mut rearranged_given = Vec::new();

        // While there are free params, push them into rearranged_given
        if given_len > 0 {
            let mut given_idx = 0;
            while given_idx < given_len
                && matches!(
                    self.nodes[given[given_idx]],
                    Node::ValueParam { name: None, .. }
                )
            {
                let id_to_push = given[given_idx];

                let Node::ValueParam {
                    value: existing_value,
                    index: existing_index,
                    ..
                } = self.nodes[id_to_push]
                else {
                    panic!()
                };

                // Add the name
                let name = decl[given_idx];
                match &self.nodes[name] {
                    Node::StructDeclParam { name, .. }
                    | Node::FnDeclParam { name, .. }
                    | Node::EnumDeclParam { name, .. }
                    | Node::ValueParam {
                        name: Some(name), ..
                    } => {
                        self.nodes[id_to_push] = Node::ValueParam {
                            name: Some(name.clone()),
                            value: existing_value,
                            index: existing_index,
                        };
                    }
                    _ => {
                        self.nodes[id_to_push] = Node::ValueParam {
                            name: None,
                            value: existing_value,
                            index: existing_index,
                        }
                    }
                }

                rearranged_given.push(id_to_push);
                given_idx += 1;
            }

            let mut cgiven_idx = given_idx;
            while cgiven_idx < given_len {
                if matches!(
                    self.nodes[given[cgiven_idx]],
                    Node::ValueParam { name: None, .. }
                ) {
                    return Err(CompileError::Node(
                        "Cannot have unnamed params after named params".to_string(),
                        err_id,
                    ));
                }
                cgiven_idx += 1;
            }
        }

        let starting_rearranged_len = rearranged_given.len();

        for &d in decl.iter().skip(starting_rearranged_len) {
            let mut found = false;

            let decl_name = match &self.nodes[d] {
                Node::FnDeclParam { name, .. }
                | Node::StructDeclParam { name, .. }
                | Node::EnumDeclParam { name, .. }
                | Node::ValueParam {
                    name: Some(name), ..
                } => *name,
                a => panic!(
                    "Expected FnDeclParam, StructDeclParam, or EnumDeclParam, got {:?}",
                    a.ty()
                ),
            };
            let decl_name_sym = self.get_symbol(decl_name);

            for &g in given.iter().skip(starting_rearranged_len) {
                let given_name = match &self.nodes[g] {
                    Node::ValueParam {
                        name: Some(name), ..
                    }
                    | Node::StructDeclParam { name, .. } => *name,
                    a => {
                        return Err(CompileError::Node(
                            format!("Expected ValueParam, got {:?}", a.ty()),
                            g,
                        ));
                    }
                };
                let given_name_sym = self.get_symbol(given_name);

                if given_name_sym == decl_name_sym {
                    rearranged_given.push(g);
                    found = true;
                    break;
                }
            }

            match (found, given_scope) {
                (false, Some(given_scope)) => {
                    // Look for the param in the implicit scope
                    let implicit_scope = &self.scopes[given_scope].implicit_entries;
                    if let Some(imp) = implicit_scope.get(&decl_name_sym) {
                        rearranged_given.push(*imp);
                        found = true;
                    }
                }
                _ => {}
            };

            if !found {
                match self.nodes[d] {
                    Node::FnDeclParam {
                        default: Some(def), ..
                    } => rearranged_given.push(def),
                    Node::EnumDeclParam { ty: None, .. } => {
                        // this is okay, as the enum *should* be constructed without a parameter. This is the name-only case
                    }
                    _ => {
                        return Err(CompileError::Node(
                            format!(
                                "Could not find parameter '{}'",
                                self.string_interner.resolve(decl_name_sym.0).unwrap()
                            ),
                            err_id,
                        ));
                    }
                }
            }
        }

        Ok(rearranged_given)
    }

    #[instrument(skip_all)]
    pub fn match_types(&mut self, ty1: NodeId, ty2: NodeId) {
        self.handle_match_types(ty1, ty2);
        self.merge_type_matches(ty1, ty2);
    }

    #[instrument(skip_all)]
    pub fn handle_match_types(&mut self, ty1: NodeId, ty2: NodeId) {
        if ty1 == ty2 {
            return;
        }

        // println!(
        //     "matching {:?} ({:?}) with {:?} ({:?})",
        //     self.get_type(ty1),
        //     self.nodes[ty1].ty(),
        //     self.get_type(ty2),
        //     self.nodes[ty2].ty()
        // );

        match (self.get_type(ty1), self.get_type(ty2)) {
            (Type::Pointer(pt1), Type::Pointer(pt2)) => {
                self.match_types(pt1, pt2);
            }
            (
                Type::Func {
                    return_ty: return_ty1,
                    input_tys: input_tys1,
                    ..
                },
                Type::Func {
                    return_ty: return_ty2,
                    input_tys: input_tys2,
                    ..
                },
            ) => {
                match (return_ty1, return_ty2) {
                    (Some(return_ty1), Some(return_ty2)) => {
                        self.match_types(return_ty1, return_ty2);
                    }
                    (None, None) => {}
                    _ => {
                        self.errors.push(CompileError::Node2(
                            "Could not match function return types".to_string(),
                            ty1,
                            ty2,
                        ));
                    }
                }

                if input_tys1.borrow().len() != input_tys2.borrow().len() {
                    self.errors.push(CompileError::Node2(
                        "Could not match types: input types differ in length".to_string(),
                        ty1,
                        ty2,
                    ));
                }

                for (it1, it2) in input_tys1.borrow().iter().zip(input_tys2.borrow().iter()) {
                    self.match_types(*it1, *it2);
                }
            }
            (
                Type::Struct {
                    decl: d1,
                    params: f1,
                    ..
                },
                Type::Struct {
                    decl: d2,
                    params: f2,
                    ..
                },
            )
            | (
                Type::Enum {
                    decl: d1,
                    params: f1,
                    ..
                },
                Type::Enum {
                    decl: d2,
                    params: f2,
                    ..
                },
            ) => {
                match (d1, d2) {
                    (Some(n1), Some(n2)) => {
                        // todo(chad): @hack? Compare the range of the declaration site, since if these are polymorphs
                        // they could be different node ids. Maybe polymorph deduplication will eventually solve this
                        let n1r = self.ranges[n1];
                        let n2r = self.ranges[n2];

                        if n1r != n2r {
                            self.errors.push(CompileError::Node2(
                            format!("Could not match types: declaration sites differ ({:?} vs {:?})", self.nodes[&n1].clone(), self.nodes[&n2].clone()),
                            n1,
                            n2,
                        ));
                        }

                        // Same declaration site means same number of parameters, so we can just match them up
                        for (f1, f2) in f1.clone().borrow().iter().zip(f2.clone().borrow().iter()) {
                            self.match_types(*f1, *f2);
                        }
                    }
                    (None, Some(n)) => {
                        if let Err(err) = self.match_params_to_named_struct(f1, n, ty1) {
                            self.errors.push(err);
                        }
                    }
                    (Some(n), None) => {
                        if let Err(err) = self.match_params_to_named_struct(f2, n, ty2) {
                            self.errors.push(err);
                        }
                    }
                    (None, None) => {
                        if f1.borrow().len() == 0 && f2.borrow().len() == 0 {
                            // Nothing to do, two empty things can match each other
                        } else {
                            let f1_is_decl = if f1.borrow().len() > 0 {
                                matches!(
                                    self.nodes[f1.borrow()[0]],
                                    Node::StructDeclParam { .. }
                                        | Node::FnDeclParam { .. }
                                        | Node::EnumDeclParam { .. }
                                )
                            } else {
                                !matches!(
                                    self.nodes[f2.borrow()[0]],
                                    Node::StructDeclParam { .. }
                                        | Node::FnDeclParam { .. }
                                        | Node::EnumDeclParam { .. }
                                )
                            };

                            if f1_is_decl {
                                let rearranged = self.rearrange_params(
                                    &f2.borrow(),
                                    Some(self.node_scopes[ty2]),
                                    &f1.borrow(),
                                    ty2,
                                );

                                match rearranged {
                                    Ok(rearranged) => {
                                        *f2.borrow_mut() = rearranged;
                                    }
                                    Err(err) => self.errors.push(err),
                                }
                            } else {
                                let rearranged = self.rearrange_params(
                                    &f1.borrow(),
                                    Some(self.node_scopes[ty1]),
                                    &f2.borrow(),
                                    ty2,
                                );

                                match rearranged {
                                    Ok(rearranged) => {
                                        *f1.borrow_mut() = rearranged;
                                    }
                                    Err(err) => self.errors.push(err),
                                }
                            }

                            if f1.borrow().len() != f2.borrow().len() {
                                self.errors.push(CompileError::Node2(
                                    "Could not match types: struct fields differ in length"
                                        .to_string(),
                                    ty1,
                                    ty2,
                                ));
                            }

                            for (f1, f2) in f1.borrow().iter().zip(f2.borrow().iter()) {
                                self.match_types(*f1, *f2);
                            }
                        }
                    }
                }
            }
            (Type::Array(n1, l1), Type::Array(n2, l2)) => {
                if let (ArrayLen::Some(_), ArrayLen::None) | (ArrayLen::None, ArrayLen::Some(_)) =
                    (l1, l2)
                {
                    self.errors.push(CompileError::Node2(
                        "Could not match types: static vs dynamic length arrays".to_string(),
                        ty1,
                        ty2,
                    ));
                }

                if let (ArrayLen::Some(l1), ArrayLen::Some(l2)) = (l1, l2) {
                    if l1 != l2 {
                        self.errors.push(CompileError::Node2(
                            "Could not match types: array lengths differ".to_string(),
                            ty1,
                            ty2,
                        ));
                    }
                }

                self.match_types(n1, n2);
            }
            (Type::IntLiteral, bt) | (bt, Type::IntLiteral) if bt.is_basic() => {
                if !self.check_int_literal_type(bt.clone()) {
                    self.errors.push(CompileError::Node2(
                        format!("Type mismatch - int literal with {:?}", bt),
                        ty1,
                        ty2,
                    ));
                }
            }
            (Type::FloatLiteral, bt) | (bt, Type::FloatLiteral) if bt.is_basic() => {
                if !self.check_float_literal_type(bt.clone()) {
                    self.errors.push(CompileError::Node2(
                        format!("Type mismatch - float literal with {:?}", bt),
                        ty1,
                        ty2,
                    ));
                }
            }
            (bt1, bt2) if bt1 == bt2 => (),
            (Type::Infer(_), _) | (_, Type::Infer(_)) => (),
            (_, _) => {
                self.errors.push(CompileError::Node2(
                    format!(
                        "Could not match types: {} ({:?}) was {:?}, {} ({:?}) was {:?}",
                        self.nodes[ty1].ty(),
                        self.ranges[ty1],
                        self.types[&ty1],
                        self.nodes[ty2].ty(),
                        self.ranges[ty2],
                        self.types[&ty2]
                    ),
                    ty1,
                    ty2,
                ));
            }
        }
    }

    #[instrument(skip_all)]
    pub fn check_int_literal_type(&self, bt: Type) -> bool {
        matches!(
            bt,
            Type::IntLiteral
                | Type::I8
                | Type::I16
                | Type::I32
                | Type::I64
                | Type::U8
                | Type::U16
                | Type::U32
                | Type::U64
        )
    }

    #[instrument(skip_all)]
    pub fn check_float_literal_type(&self, bt: Type) -> bool {
        matches!(bt, Type::FloatLiteral | Type::F32 | Type::F64)
    }

    #[instrument(skip_all)]
    pub fn get_type(&self, id: NodeId) -> Type {
        return self.types.get(&id).cloned().unwrap_or(Type::Infer(None));
    }

    #[instrument(skip_all)]
    pub fn is_fully_concrete(&mut self, id: NodeId) -> bool {
        self.is_fully_concrete_ty(self.get_type(id))
    }

    #[instrument(skip_all)]
    pub fn is_fully_concrete_ty(&mut self, ty: Type) -> bool {
        if let Type::Pointer(pt) = ty {
            if self.circular_concrete_types.contains(&pt) {
                return true;
            }

            self.circular_concrete_types.insert(pt);
            return self.is_fully_concrete(pt);
        }

        if let Type::Struct {
            decl: Some(_),
            params,
            ..
        }
        | Type::Enum {
            decl: Some(_),
            params,
            ..
        } = ty
        {
            for &field in params.clone().borrow().iter() {
                if !self.is_fully_concrete(field) {
                    return false;
                }
            }
            return true;
        }

        matches!(
            ty,
            Type::I8
                | Type::I16
                | Type::I32
                | Type::I64
                | Type::U8
                | Type::U16
                | Type::U32
                | Type::U64
                | Type::F32
                | Type::F64
        )
    }

    #[instrument(skip_all)]
    pub fn find_type_array_index(&self, id: NodeId) -> Option<usize> {
        self.type_array_reverse_map.get(&id).cloned()
    }

    #[instrument(skip_all)]
    pub fn find_addressable_array_index(&self, id: NodeId) -> Option<usize> {
        self.addressable_array_reverse_map.get(&id).cloned()
    }

    #[instrument(skip_all)]
    pub fn merge_type_matches(&mut self, ty1: NodeId, ty2: NodeId) {
        match (self.get_type(ty1), self.get_type(ty2)) {
            (Type::Infer(_), Type::Infer(_)) => (),
            (Type::Infer(_), ty) => {
                self.types.insert(ty1, ty);
            }
            (ty, Type::Infer(_)) => {
                self.types.insert(ty2, ty);
            }
            _ => (),
        }

        let id1 = self.find_type_array_index(ty1);
        let id2 = self.find_type_array_index(ty2);

        match (id1, id2) {
            (None, None) => {
                if self.is_fully_concrete(ty1) {
                    self.types.insert(ty2, self.get_type(ty1));
                } else if self.is_fully_concrete(ty2) {
                    self.types.insert(ty1, self.get_type(ty2));
                } else {
                    let unified = self.unify(self.get_type(ty1), self.get_type(ty2), (ty1, ty2));
                    self.type_matches.push(TypeMatch {
                        changed: true,
                        unified,
                        ids: vec![ty1, ty2],
                    });

                    self.type_array_reverse_map
                        .insert(ty1, self.type_matches.len() - 1);
                    self.type_array_reverse_map
                        .insert(ty2, self.type_matches.len() - 1);
                }
            }
            (Some(id), None) => {
                self.type_matches[id].ids.push(ty2);
                self.type_array_reverse_map.insert(ty2, id);
                self.type_matches[id].changed = true;
                self.type_matches[id].unified = self.unify(
                    self.type_matches[id].unified.clone(),
                    self.get_type(ty2),
                    (ty1, ty2),
                );

                if self.is_fully_concrete(ty1) {
                    for t in self.type_matches[id].ids.clone() {
                        self.types.insert(t, self.get_type(ty1));
                        self.type_array_reverse_map.remove(&t);
                    }
                    self.type_matches.swap_remove(id);
                    if self.type_matches.len() > id {
                        for t in self.type_matches[id].ids.clone() {
                            self.type_array_reverse_map.insert(t, id);
                        }
                    }
                } else if self.is_fully_concrete(ty2) {
                    for t in self.type_matches[id].ids.clone() {
                        self.types.insert(t, self.get_type(ty2));
                        self.type_array_reverse_map.remove(&t);
                    }
                    self.type_matches.swap_remove(id);
                    if self.type_matches.len() > id {
                        for t in self.type_matches[id].ids.clone() {
                            self.type_array_reverse_map.insert(t, id);
                        }
                    }
                }
            }
            (None, Some(id)) => {
                self.type_matches[id].ids.push(ty1);
                self.type_array_reverse_map.insert(ty1, id);
                self.type_matches[id].changed = true;
                self.type_matches[id].unified = self.unify(
                    self.type_matches[id].unified.clone(),
                    self.get_type(ty1),
                    (ty1, ty2),
                );

                if self.is_fully_concrete(ty1) {
                    for t in self.type_matches[id].ids.clone() {
                        self.types.insert(t, self.get_type(ty1));
                        self.type_array_reverse_map.remove(&t);
                    }
                    self.type_matches.swap_remove(id);
                    if self.type_matches.len() > id {
                        for t in self.type_matches[id].ids.clone() {
                            self.type_array_reverse_map.insert(t, id);
                        }
                    }
                } else if self.is_fully_concrete(ty2) {
                    for t in self.type_matches[id].ids.clone() {
                        self.types.insert(t, self.get_type(ty2));
                        self.type_array_reverse_map.remove(&t);
                    }
                    self.type_matches.swap_remove(id);
                    if self.type_matches.len() > id {
                        for t in self.type_matches[id].ids.clone() {
                            self.type_array_reverse_map.insert(t, id);
                        }
                    }
                }
            }
            (Some(id1), Some(id2)) if id1 != id2 => {
                let lower = id1.min(id2);
                let upper = id1.max(id2);

                let unified = self.unify(
                    self.type_matches[lower].unified.clone(),
                    self.type_matches[upper].unified.clone(),
                    (ty1, ty2),
                );

                let (lower_matches, upper_matches) = self.type_matches.split_at_mut(lower + 1);
                lower_matches[lower]
                    .ids
                    .extend(upper_matches[upper - lower - 1].ids.iter());
                for t in self.type_matches[lower].ids.clone() {
                    self.type_array_reverse_map.insert(t, lower);
                }

                self.type_matches[lower].changed = true;
                self.type_matches[lower].unified = unified;

                self.type_matches.swap_remove(upper);

                if self.type_matches.len() > upper {
                    for t in self.type_matches[upper].ids.clone() {
                        self.type_array_reverse_map.insert(t, upper);
                    }
                }

                if self.is_fully_concrete(ty1) {
                    for t in self.type_matches[lower].ids.clone() {
                        self.types.insert(t, self.get_type(ty1));
                    }
                } else if self.is_fully_concrete(ty2) {
                    for t in self.type_matches[lower].ids.clone() {
                        self.types.insert(t, self.get_type(ty2));
                    }
                }
            }
            (_, _) => (),
        }
    }

    #[instrument(skip_all)]
    pub fn assign_type(&mut self, id: NodeId) {
        if self.completes.contains(&id) {
            return;
        }

        if self.polymorph_sources.contains(&id) {
            return;
        }

        if self.circular_dependency_nodes.contains(&id) {
            return;
        }
        self.circular_dependency_nodes.insert(id);

        if self.assign_type_inner(id) {
            self.completes.insert(id);
        }

        // self.unify_types();
    }

    #[instrument(skip_all)]
    pub fn assign_type_inner(&mut self, id: NodeId) -> bool {
        match self.nodes[id].clone() {
            Node::FnDefinition {
                params,
                return_ty,
                stmts,
                returns,
                transparent,
                ..
            } => {
                self.returns.push(Vec::new());

                if let Some(value) =
                    self.assign_type_fn(id, return_ty, params, stmts, returns, transparent)
                {
                    self.returns.pop();
                    return value;
                }

                self.returns.pop();
            }
            Node::Extern {
                params, return_ty, ..
            } => {
                self.assign_type_extern(return_ty, params, id);
            }
            Node::Type(ty) => {
                self.assign_type_type(id, ty);
            }
            Node::Return(ret_id) => {
                self.assign_type_return(ret_id, id);
            }
            Node::IntLiteral(_, spec) => self.assign_type_int_literal(spec, id),
            Node::BoolLiteral(_) => {
                self.assign_type_bool(id);
            }
            Node::StringLiteral(_) => {
                self.assign_type_string_literal(id);
            }
            Node::Symbol(sym) => {
                return self.assign_type_symbol(sym, id);
            }
            Node::FnDeclParam {
                name,
                ty,
                default,
                transparent,
                ..
            } => {
                if let Some(value) =
                    self.assign_type_fn_decl_param(id, name, ty, default, transparent)
                {
                    return value;
                }
            }
            Node::StructDeclParam {
                name,
                ty,
                default,
                transparent,
                ..
            } => {
                if let Some(value) =
                    self.assign_type_struct_decl_param(id, name, ty, default, transparent)
                {
                    return value;
                }
            }
            Node::EnumDeclParam {
                name,
                ty,
                transparent,
                ..
            } => {
                if let Some(value) = self.assign_type_enum_decl_param(id, name, ty, transparent) {
                    return value;
                }
            }
            Node::ValueParam { name, value, .. } => {
                self.assign_type_value_param(value, id, name);
            }
            Node::Let {
                name,
                ty,
                expr,
                transparent,
            } => {
                if let Some(value) = self.assign_type_let(id, name, ty, expr, transparent) {
                    return value;
                }
            }
            Node::FloatLiteral(_, spec) => {
                if let Some(value) = self.assign_type_float_literal(spec, id) {
                    return value;
                }
            }
            Node::Assign { name, expr, .. } => {
                self.in_assign_lhs = true;
                self.assign_type(name);
                self.in_assign_lhs = true;

                self.assign_type(expr);

                self.match_types(id, expr);
                self.match_types(name, expr);
            }
            Node::BinOp { op, lhs, rhs } => {
                self.assign_type(lhs);
                self.assign_type(rhs);

                match op {
                    Op::And | Op::Or => {
                        self.types.insert(id, Type::Bool);
                        self.match_types(id, lhs);
                        self.match_types(lhs, rhs);
                    }
                    Op::Add | Op::Sub | Op::Mul | Op::Div => {
                        self.match_types(id, lhs);
                        self.match_types(lhs, rhs);
                    }
                    Op::EqEq | Op::Neq | Op::Gt | Op::Lt | Op::GtEq | Op::LtEq => {
                        self.types.insert(id, Type::Bool);
                        self.match_types(lhs, rhs);
                    }
                }
            }
            Node::Call { func, params, .. } | Node::ThreadingCall { func, params } => {
                return self.assign_type_inner_call(id, func, params);
            }
            Node::StructDefinition {
                name,
                params,
                scope,
            } => {
                // don't directly codegen a polymorph, wait until it's copied first
                if self.polymorph_sources.contains(&id) {
                    return true;
                }

                for &param in params.borrow().iter() {
                    self.assign_type(param);
                }

                self.types.insert(
                    id,
                    Type::Struct {
                        params: params.clone(),
                        decl: Some(id),
                        scope: Some(scope),
                    },
                );

                self.match_types(id, name);

                let mut should_defer = false;
                for param in params.borrow().iter() {
                    if !self.types.contains_key(param) {
                        should_defer = true;
                        self.deferreds.push(*param);
                    }
                }

                if should_defer {
                    self.deferreds.push(id);
                    return false;
                }
            }
            Node::EnumDefinition {
                name,
                params,
                scope,
            } => {
                // don't directly codegen a polymorph, wait until it's copied first
                if self.polymorph_sources.contains(&id) {
                    return true;
                };
                for &param in params.borrow().iter() {
                    if let Node::EnumDeclParam { ty: None, .. } = self.nodes[param] {
                        self.types.insert(param, Type::EnumNoneType);
                    } else {
                        self.assign_type(param);
                    }
                }

                self.types.insert(
                    id,
                    Type::Enum {
                        decl: Some(id),
                        params: params.clone(),
                        scope: Some(scope),
                    },
                );

                self.match_types(id, name);

                let mut should_defer = false;
                for param in params.borrow().iter() {
                    if !self.types.contains_key(param) {
                        should_defer = true;
                        self.deferreds.push(*param);
                    }
                }

                if should_defer {
                    self.deferreds.push(id);
                    return false;
                }
            }
            Node::StructLiteral { name, params } => {
                if let Some(name) = name {
                    self.assign_type(name);

                    // If this is a polymorph, copy it first
                    let name = if self.polymorph_sources.contains(&name) {
                        let &key = self.polymorph_sources.get(&name).unwrap();
                        match self.copy_polymorph(key, ParseTarget::StructDeclaration) {
                            Ok(copied) => {
                                self.assign_type(copied);
                                self.nodes[id] = Node::StructLiteral {
                                    name: Some(copied),
                                    params: params.clone(),
                                };

                                copied
                            }
                            Err(err) => {
                                self.errors.push(err);
                                return true;
                            }
                        }
                    } else {
                        name
                    };

                    self.match_types(name, id);

                    if let Err(err) = self.match_params_to_named_struct(params, name, id) {
                        self.errors.push(err);
                        return true;
                    }
                } else {
                    for &field in params.clone().borrow().iter() {
                        self.assign_type(field);
                    }
                    self.types.insert(
                        id,
                        Type::Struct {
                            params,
                            decl: None,
                            scope: None,
                        },
                    );
                }
            }
            Node::MemberAccess { value, member, .. } => {
                self.assign_type(value);

                let mut value_ty = self.get_type(value);

                while let Type::Pointer(ty) = value_ty {
                    value_ty = self.get_type(ty);
                }

                match value_ty {
                    Type::Struct {
                        scope: Some(scope), ..
                    }
                    | Type::Enum {
                        scope: Some(scope), ..
                    } => {
                        match &self.nodes[member] {
                            Node::Symbol(member_name_sym) => {
                                match self.scopes[scope].entries.get(member_name_sym) {
                                    Some(&found) => {
                                        self.match_types(id, found);

                                        // If we found something good, replace the member access with it.
                                        // This is particularly useful since we could be replacing with an entire transparency tree
                                        self.nodes[member] = self.nodes[found].clone();
                                    }
                                    None => {
                                        self.defer_on(&[id], CompileError::Node(
                                            format!("Could not find member {} in scope {}, which only contains {:?}", self.string_interner.resolve(member_name_sym.0).unwrap(), scope.0, self.debug_scope(scope)),
                                            member,
                                        ));
                                        return false;
                                    }
                                }
                            }
                            Node::StructDeclParam { .. }
                            | Node::EnumDeclParam { .. }
                            | Node::MemberAccess { .. } => {
                                self.match_types(id, member);
                            }
                            _ => {
                                self.deferreds.push(member);
                                self.deferreds.push(id);

                                return false;
                            }
                        }
                    }
                    Type::Struct {
                        decl: None, params, ..
                    }
                    | Type::Enum {
                        decl: None, params, ..
                    } => {
                        let member_name_sym = self.get_symbol(member);

                        let field_ids = params.clone();
                        let mut found = false;
                        for &field in field_ids.borrow().iter() {
                            let field_name = match &self.nodes[field] {
                                Node::ValueParam {
                                    name: Some(name), ..
                                }
                                | Node::StructDeclParam { name, .. } => *name,
                                Node::EnumDeclParam { name, .. } => *name,
                                a => {
                                    self.defer_on(&[id], CompileError::Node(
                                        format!(
                                            "Cannot perform member access as this field's name (at {:?}) could not be found. Node type: {:?}",
                                            &self.ranges[field],
                                            a.ty()
                                        ),
                                        id,
                                    ));
                                    return false;
                                }
                            };
                            let field_name_sym = self.get_symbol(field_name);

                            if field_name_sym == member_name_sym {
                                self.match_types(id, field);
                                found = true;
                                break;
                            }
                        }

                        if !found {
                            self.errors.push(CompileError::Node(
                                "Could not find member in fields".to_string(),
                                member,
                            ));
                        }
                    }
                    Type::Array(array_ty, len) => {
                        let member_name_sym = self.get_symbol(member);

                        let name = self.string_interner.resolve(member_name_sym.0).unwrap();
                        match name {
                            "len" => {
                                if let ArrayLen::Some(_) | ArrayLen::Infer = len {
                                    self.errors.push(CompileError::Node(
                                        "'len' is not a property on a static length array. Try ::len to access the static length property"
                                            .to_string(),
                                        member,
                                    ));
                                } else {
                                    self.types.insert(id, Type::I64);
                                }
                            }
                            "data" => {
                                if let ArrayLen::Some(_) | ArrayLen::Infer = len {
                                    self.errors.push(CompileError::Node(
                                        "'data' is not a property on a static length array. Perhaps you meant to simply take the address of the static array?"
                                            .to_string(),
                                        member,
                                    ));
                                } else {
                                    self.types.insert(id, Type::Pointer(array_ty));
                                }
                            }
                            _ => {
                                self.errors.push(CompileError::Node(
                                    format!(
                                        "Array has no member {:?}: options are 'len' or 'data'",
                                        name
                                    ),
                                    member,
                                ));
                            }
                        }
                    }
                    Type::String => {
                        let member_name_sym = self.get_symbol(member);

                        let name = self.string_interner.resolve(member_name_sym.0).unwrap();
                        match name {
                            "len" => {
                                self.types.insert(id, Type::I64);
                            }
                            "data" => {
                                self.types.insert(id, Type::Infer(None));
                            }
                            _ => {
                                self.errors.push(CompileError::Node(
                                    format!(
                                        "Array has no member {:?}: options are 'len' or 'data'",
                                        name
                                    ),
                                    member,
                                ));
                            }
                        }
                    }
                    Type::Infer(_) => {
                        self.deferreds.push(id);
                        return false;
                    }
                    _ => {
                        self.defer_on(
                            &[value, id],
                            CompileError::Node(
                                format!("Member access on a non-struct (type {:?})", value_ty),
                                id,
                            ),
                        );
                        return false;
                    }
                }
            }
            Node::StaticMemberAccess { value, member, .. } => {
                self.assign_type(value);

                // If this is a polymorph, copy it first
                let value = if self.polymorph_sources.contains(&value) {
                    let &key = self.polymorph_sources.get(&value).unwrap();
                    match self.copy_polymorph(key, ParseTarget::EnumDefinition) {
                        Ok(copied) => {
                            self.assign_type(copied);
                            self.nodes[id] = Node::StaticMemberAccess {
                                value: copied,
                                member,
                                resolved: None,
                            };

                            copied
                        }
                        Err(err) => {
                            self.errors.push(err);
                            return true;
                        }
                    }
                } else {
                    value
                };

                let mut value_ty = self.get_type(value);

                while let Type::Pointer(ty) = value_ty {
                    value_ty = self.get_type(ty);
                }

                match value_ty {
                    Type::Enum { params, .. } => {
                        return self
                            .assign_type_static_member_access_enum(params, member, value, id);
                    }
                    Type::Array(_, _) => {
                        let member_name = self.get_symbol_str(member);
                        if member_name == "len" {
                            self.types.insert(id, Type::I64);
                        } else {
                            self.errors.push(CompileError::Node(
                                format!("Array has no static member {:?}. The only static member is 'len'", member_name),
                                member,
                            ));
                        }
                    }
                    Type::Infer(_) => {
                        self.deferreds.push(id);
                        return false;
                    }
                    _ => {
                        self.errors.push(CompileError::Node(
                            format!("Static member access on a non-enum (type {:?})", value_ty),
                            id,
                        ));
                    }
                }
            }
            Node::AddressOf(value) => {
                self.assign_type(value);
                self.types.insert(id, Type::Pointer(value));
            }
            Node::Deref(value) => {
                // In the case where we're compiling something like `*a = 5`, we need to know that `a` is addressable.
                // However, only a needs to be addressable. So for example in a more complicate case
                // like `**a = 12`, only the `*a` needs to be made addressable, not necessarily `a` itself
                if self.in_assign_lhs {
                    self.in_assign_lhs = false;
                }

                self.assign_type(value);

                match self.get_type(value) {
                    Type::Pointer(ty) => {
                        self.match_types(id, ty);
                    }
                    Type::Infer(_) => {
                        self.deferreds.push(id);
                        return false;
                    }
                    _ => {
                        self.errors.push(CompileError::Node(
                            "Dereference on non-pointer".to_string(),
                            id,
                        ));
                    }
                }
            }
            Node::If {
                cond,
                then_block,
                else_block,
                then_label,
                else_label,
            } => {
                match cond {
                    IfCond::Expr(cond) => {
                        self.assign_type(cond);
                    }
                    IfCond::Let { tag, alias, expr } => {
                        self.assign_type(expr);

                        let tag_sym = self.get_symbol(tag);

                        // Look up the tag in expr's type
                        let expr_ty = self.get_type(expr);
                        if let Type::Infer(_) = expr_ty {
                            self.deferreds.push(id);
                            return false;
                        }

                        let Type::Enum { params, .. } = expr_ty else {
                            self.errors.push(CompileError::Node(
                                format!("Cannot use let if on non-enum type {:?}", expr_ty),
                                expr,
                            ));
                            return true;
                        };
                        let mut found = false;
                        for &param in params.borrow().iter() {
                            if let Node::EnumDeclParam { name, ty, .. } = self.nodes[param].clone()
                            {
                                let name_sym = self.get_symbol(name);
                                if name_sym == tag_sym {
                                    found = true;
                                    if let Some(alias) = alias {
                                        self.match_types(alias, param);

                                        // If an alias was included, make sure the decl param actually has data attached to it
                                        if ty.is_none() {
                                            self.errors.push(CompileError::Node(
                                                format!(
                                                    "Cannot use alias on enum decl param {} as it has no data",
                                                    self.string_interner.resolve(name_sym.0).unwrap(),
                                                ),
                                                alias,
                                            ));
                                        }
                                    }
                                    break;
                                }
                            } else {
                                todo!()
                            };
                        }

                        if !found {
                            self.errors.push(CompileError::Node(
                                format!(
                                    "Could not find tag {} in enum",
                                    self.string_interner.resolve(tag_sym.0).unwrap(),
                                ),
                                tag,
                            ));
                        }
                    }
                }

                self.push_break(then_label);
                self.returns.push(Vec::new());

                self.assign_type(then_block);
                self.match_types(id, then_block);

                self.pop_break();
                self.returns.last_mut().unwrap().clear();

                match else_block {
                    NodeElse::Block(else_block) => {
                        self.push_break(else_label);
                        self.assign_type(else_block);
                        self.match_types(id, else_block);
                    }
                    NodeElse::If(else_if) => {
                        self.assign_type(else_if);
                        self.match_types(id, else_if);
                    }
                    NodeElse::None => {
                        // The two arms of the if must have the same type. So if there's no else arm, then enforce that it is an empty type
                        self.types.insert(id, Type::Empty);
                    }
                }

                self.pop_break();
                self.returns.pop();
            }
            Node::Match { value, cases } => {
                self.assign_type(value);

                for c in cases.borrow().iter() {
                    let Node::MatchCase {
                        block, block_label, ..
                    } = self.nodes[*c].clone()
                    else {
                        unreachable!()
                    };

                    self.push_break(block_label);

                    self.assign_type(block);

                    self.pop_break();
                    self.returns.last_mut().unwrap().clear();

                    self.match_types(id, block);
                }

                if let Type::Infer(_) = self.get_type(value) {
                    self.deferreds.push(id);
                    return false;
                }

                let Type::Enum { params, .. } = self.get_type(value) else {
                    self.errors.push(CompileError::Node(
                        "Cannot match on non-enum type".to_string(),
                        value,
                    ));
                    return true;
                };

                for p in params.borrow().iter() {
                    let tag_to_search_for = match &self.nodes[*p] {
                        Node::EnumDeclParam { name, .. } => *name,
                        _ => todo!(),
                    };
                    let tag_to_search_for_sym = self.get_symbol(tag_to_search_for);

                    let mut found = false;
                    for c in cases.borrow().iter() {
                        let Node::MatchCase { tag, .. } = self.nodes[*c].clone() else {
                            unreachable!()
                        };

                        let tag_sym = match self.nodes[tag] {
                            Node::Symbol(sym) => sym,
                            Node::EnumDeclParam { name, .. } => self.get_symbol(name),
                            _ => todo!(),
                        };

                        if tag_sym == tag_to_search_for_sym {
                            found = true;
                            break;
                        }
                    }

                    if !found {
                        self.errors.push(CompileError::Node(
                            format!(
                                "Could not find tag {} in match cases",
                                self.string_interner
                                    .resolve(tag_to_search_for_sym.0)
                                    .unwrap(),
                            ),
                            value,
                        ));
                    }
                }

                // Check nothing was specified twice, by comparing the lengths of the params and cases
                if params.borrow().len() != cases.borrow().len() {
                    self.errors.push(CompileError::Node(
                        "Match cases must specify all enum params exactly once".to_string(),
                        value,
                    ));
                }
            }
            Node::MatchCase {
                tag: _,
                alias: _,
                block: _,
                block_label: _,
            } => {
                // Should have already been handled in the parent match node
                unreachable!()
            }
            Node::Break(r, label) => {
                self.ensure_break_label_exists(label, id);

                match self.breaks.last_mut() {
                    Some(r) => r.push(id),
                    _ => {
                        self.errors.push(CompileError::Node(
                            "Break stmt outside of a breakable block!".to_string(),
                            id,
                        ));
                        return true;
                    }
                }

                match r {
                    Some(r) => {
                        self.assign_type(r);
                        self.match_types(id, r);
                    }
                    None => {
                        self.types.insert(id, Type::Empty);
                    }
                };
            }
            Node::Continue(label) => {
                self.ensure_continue_label_exists(label, id);
                self.types.insert(id, Type::Empty);
            }
            Node::Block {
                stmts,
                breaks,
                label,
                is_standalone,
                ..
            } => {
                if is_standalone {
                    self.push_break(label);
                }

                for &stmt in stmts.clone().borrow().iter() {
                    self.ensure_not_already_exited_block(stmt);
                    self.assign_type(stmt);
                }

                let breaks = breaks.clone();

                if breaks.borrow().is_empty() {
                    self.types.insert(id, Type::Empty);
                }

                for &br in breaks.borrow().iter() {
                    self.assign_type(br);
                    self.match_types(id, br);
                }

                if is_standalone {
                    self.pop_break();
                }
            }
            Node::ArrayLiteral { members, ty } => {
                for &member in members.clone().borrow().iter() {
                    self.assign_type(member);
                    self.match_types(member, ty);
                    self.check_not_unspecified_polymorph(member);
                }

                self.types
                    .insert(id, Type::Array(ty, ArrayLen::Some(members.borrow().len())));
            }
            Node::ArrayAccess { array, index } => {
                self.assign_type(array);
                self.assign_type(index);

                let array_ty = self.get_type(array);

                match array_ty {
                    Type::Array(ty, _) => {
                        // todo(chad): @Hack: there needs to be a way of doing `self.match_types(id, Type::I64);`
                        // for now just hardcode to i64 for testing purposes
                        self.types.insert(index, Type::I64);
                        self.match_types(id, ty);
                    }
                    Type::Infer(_) => {
                        self.deferreds.push(id);
                        return false;
                    }
                    _ => {
                        self.errors.push(CompileError::Node(
                            "Array access on non-array".to_string(),
                            id,
                        ));
                    }
                }

                let index_ty = self.get_type(index);
                match index_ty {
                    Type::I64 => (),
                    Type::Infer(_) => {
                        self.deferreds.push(id);
                        return false;
                    }
                    _ => {
                        self.errors.push(CompileError::Node(
                            "Array access with non-integer index".to_string(),
                            id,
                        ));
                    }
                }
            }
            Node::PolySpecialize { sym, overrides, .. } => {
                let resolved = self.scope_get(sym, id).unwrap();

                if self.polymorph_sources.get(&resolved).is_none() {
                    self.errors.push(CompileError::Node(
                        "Cannot specialize non-polymorphic type".to_string(),
                        id,
                    ));
                    return true;
                }

                let copied = match self.copy_polymorph_if_needed(resolved) {
                    Ok(copied) => copied,
                    Err(err) => {
                        self.errors.push(err);
                        return true;
                    }
                };
                self.assign_type(copied);
                self.match_types(id, copied);

                self.nodes[id] = Node::PolySpecialize {
                    sym,
                    overrides: overrides.clone(),
                    copied: Some(copied),
                };
                for o in overrides.borrow().iter() {
                    self.assign_type(*o);
                    let Node::PolySpecializeOverride { sym, ty } = self.nodes[o] else {
                        panic!()
                    };
                    let sym = self.get_symbol(sym);

                    let (Node::StructDefinition {
                        scope: scope_id, ..
                    }
                    | Node::EnumDefinition {
                        scope: scope_id, ..
                    }) = self.nodes[copied]
                    else {
                        panic!(
                            "Expected StructDeclaration or EnumDefinition, found {}",
                            self.nodes[copied].ty()
                        )
                    };

                    let resolved = self.scope_get_with_scope_id(sym, scope_id);

                    match resolved {
                        Some(resolved) => {
                            self.assign_type(resolved);
                            self.match_types(ty, resolved);
                        }
                        None => {
                            let sym_str = self.string_interner.resolve(sym.0).unwrap();
                            self.errors.push(CompileError::Node(
                                format!("Symbol {} not found", sym_str),
                                *o,
                            ));
                        }
                    }
                }
            }
            Node::PolySpecializeOverride { ty, .. } => {
                self.assign_type(ty);
                self.match_types(id, ty);
            }
            Node::Cast { ty, value } => {
                self.assign_type(ty);
                self.match_types(id, ty);

                self.assign_type(value);
            }
            Node::SizeOf(ty) => {
                self.assign_type(ty);
                self.types.insert(id, Type::I64);
            }
            Node::TypeInfo(ty) => {
                self.assign_type(ty);
                self.match_types(id, self.type_info_decl.unwrap());
            }
            Node::For {
                iterable,
                label,
                block,
                block_label,
                ..
            } => {
                self.assign_type(iterable);

                match self.get_type(iterable) {
                    Type::Infer(_) => {
                        self.deferreds.push(id);
                        return false;
                    }
                    Type::Array(array_element_ty, _) => {
                        self.match_types(label, array_element_ty);
                    }
                    Type::String => {
                        self.types.insert(label, Type::U8);
                    }
                    _ => todo!(),
                }

                self.push_break(block_label);
                self.push_continue(block_label);
                self.returns.push(Vec::new());

                self.assign_type(block);

                self.pop_break();
                self.pop_continue();
                self.returns.pop();
            }
            Node::While {
                cond,
                block,
                block_label,
                ..
            } => {
                self.assign_type(cond);

                match self.get_type(cond) {
                    Type::Infer(_) => {
                        self.deferreds.push(id);
                        return false;
                    }
                    Type::Bool => {}
                    _ => todo!(),
                }

                self.push_break(block_label);
                self.push_continue(block_label);
                self.returns.push(Vec::new());

                self.assign_type(block);

                self.pop_break();
                self.pop_continue();
                self.returns.pop();
            }
            Node::ThreadingParamTarget => todo!(),
            Node::AsCast { value, ty, .. } => {
                self.assign_type(value);
                self.assign_type(ty);

                let mut fully_done = false;

                match (self.get_type(value), self.get_type(ty)) {
                    (Type::Infer(_), _) | (_, Type::Infer(_)) => {}
                    (Type::Array(_, ArrayLen::Infer), _) => {}
                    (Type::Array(ty1, ArrayLen::Some(_)), Type::Array(ty2, ArrayLen::None)) => {
                        self.match_types(ty1, ty2);
                        self.nodes[id] = Node::AsCast {
                            value,
                            ty,
                            style: AsCastStyle::StaticToDynamicArray,
                        };

                        fully_done = true;
                    }
                    (Type::Struct { params, .. }, Type::Array(ty, ArrayLen::None)) => {
                        // Make sure params is laid out exactly like an array
                        let param_ids = params.clone();
                        if param_ids.borrow().len() != 2 {
                            self.errors.push(CompileError::Node(
                                "Struct must have exactly two parameters".to_string(),
                                id,
                            ));
                            return true;
                        }

                        // Match the first param to the data pointer
                        {
                            let p0 = param_ids.borrow()[0];

                            self.assign_type(p0);
                            if let Type::Infer(_) = self.get_type(p0) {
                                self.deferreds.push(p0);
                                self.deferreds.push(id);
                                return false;
                            }

                            let Node::ValueParam {
                                name: Some(name),
                                value,
                                ..
                            } = self.nodes[p0]
                            else {
                                self.errors.push(CompileError::Node(
                                    "Expected a value param called 'data' here".to_string(),
                                    p0,
                                ));
                                return true;
                            };

                            // todo(chad): directly compare symbols, it's faster
                            let name = self.get_symbol_str(name);
                            if name != "data" {
                                self.errors.push(CompileError::Node(
                                    "Expected a value param called 'data' here".to_string(),
                                    p0,
                                ));
                                return true;
                            }

                            self.assign_type(value);
                            if let Type::Infer(_) = self.get_type(value) {
                                self.deferreds.push(value);
                                self.deferreds.push(id);
                                return false;
                            }

                            let Type::Pointer(array_ty) = self.get_type(value) else {
                                self.errors.push(CompileError::Node(
                                    "Expected a pointer here".to_string(),
                                    value,
                                ));
                                return true;
                            };

                            self.match_types(array_ty, ty);
                        }

                        // Match the second param to the len field
                        {
                            let p1 = param_ids.borrow()[1];

                            self.assign_type(p1);
                            if let Type::Infer(_) = self.get_type(p1) {
                                self.deferreds.push(p1);
                                self.deferreds.push(id);
                                return false;
                            }

                            let Node::ValueParam {
                                name: Some(name),
                                value,
                                ..
                            } = self.nodes[p1]
                            else {
                                self.errors.push(CompileError::Node(
                                    "Expected a value param called 'len' here".to_string(),
                                    p1,
                                ));
                                return true;
                            };

                            // todo(chad): directly compare symbols, it's faster
                            let name = self.get_symbol_str(name);
                            if name != "len" {
                                self.errors.push(CompileError::Node(
                                    "Expected a value param called 'len' here".to_string(),
                                    p1,
                                ));
                                return true;
                            }

                            self.assign_type(value);
                            if let Type::Infer(_) = self.get_type(value) {
                                self.deferreds.push(value);
                                self.deferreds.push(id);
                                return false;
                            }

                            let Type::I64 = self.get_type(value) else {
                                self.errors.push(CompileError::Node(
                                    "Expected an i64 here".to_string(),
                                    value,
                                ));
                                return true;
                            };
                        }

                        self.nodes[id] = Node::AsCast {
                            value,
                            ty,
                            style: AsCastStyle::StructToDynamicArray,
                        };

                        fully_done = true;
                    }
                    (ty1, ty2) => {
                        self.errors.push(CompileError::Node2(
                            format!("Invalid cast from {:?} to {:?}", ty1, ty2),
                            value,
                            ty,
                        ));
                    }
                }

                self.match_types(id, ty);

                if !fully_done {
                    self.deferreds.push(id);
                    return false;
                }
            }
        }

        return true;
    }

    #[instrument(skip_all)]
    fn push_break(&mut self, label: Option<Sym>) {
        self.breaks.push(Vec::new());
        self.break_labels.push(label);
    }

    #[instrument(skip_all)]
    fn pop_break(&mut self) {
        self.breaks.pop();
        self.break_labels.pop();
    }

    #[instrument(skip_all)]
    fn push_continue(&mut self, label: Option<Sym>) {
        self.continues.push(Vec::new());
        self.continue_labels.push(label);
    }

    #[instrument(skip_all)]
    fn pop_continue(&mut self) {
        self.continues.pop();
        self.continue_labels.pop();
    }

    #[instrument(skip_all)]
    fn ensure_continue_label_exists(&mut self, label: Option<Sym>, id: NodeId) {
        if label.is_none() {
            return;
        }

        if !self.continue_labels.contains(&label) {
            self.errors.push(CompileError::Node(
                format!(
                    "Continue label {} does not exist",
                    self.string_interner.resolve(label.unwrap().0).unwrap()
                ),
                id,
            ));
        }
    }

    #[instrument(skip_all)]
    fn ensure_break_label_exists(&mut self, label: Option<Sym>, id: NodeId) {
        if label.is_none() {
            return;
        }

        if !self.break_labels.contains(&label) {
            self.errors.push(CompileError::Node(
                format!(
                    "Break label {} does not exist",
                    self.string_interner.resolve(label.unwrap().0).unwrap()
                ),
                id,
            ));
        }
    }

    #[instrument(skip_all)]
    fn ensure_not_already_exited_block(&mut self, id: NodeId) {
        let retlen = self.returns.last().map(|r| r.len()).unwrap_or(0);
        let brlen = self.breaks.last().map(|r| r.len()).unwrap_or(0);
        let contlen = self.continues.last().map(|r| r.len()).unwrap_or(0);

        if retlen > 0 || brlen > 0 || contlen > 0 {
            self.errors.push(CompileError::Node(
                "Already exited this block".to_string(),
                id,
            ));
        }
    }

    #[instrument(skip_all)]
    fn assign_type_static_member_access_enum(
        &mut self,
        params: IdVec,
        member: NodeId,
        value: NodeId,
        id: NodeId,
    ) -> bool {
        let field_ids = params.clone();
        let mut found = false;

        for (index, &field) in field_ids.borrow().iter().enumerate() {
            self.assign_type(field);

            let (field_name, is_none_type) = match &self.nodes[field] {
                Node::EnumDeclParam { name, ty, .. } => (*name, ty.is_none()),
                _ => {
                    self.errors.push(CompileError::Node(
                                        format!(
                                            "Cannot perform member access as this field's name ({:?}) could not be found",
                                            &self.ranges[field]
                                        ),
                                        id,
                                    ));
                    return true;
                }
            };

            let field_name_sym = self.get_symbol(field_name);
            let member_name_sym = self.get_symbol(member);

            if field_name_sym == member_name_sym {
                // If the field is a name-only enum param, then there are no input types
                let input_tys = if is_none_type { vec![] } else { vec![field] };
                let input_tys = self.push_id_vec(input_tys);

                self.types.insert(
                    id,
                    Type::Func {
                        input_tys,
                        return_ty: Some(value),
                    },
                );

                self.nodes[id] = Node::StaticMemberAccess {
                    value,
                    member,
                    resolved: Some(StaticMemberResolution::EnumConstructor {
                        base: value,
                        index: index as _,
                    }),
                };

                if !self.topo.contains(&id) {
                    self.topo.push(id);
                }

                found = true;
                break;
            }
        }

        if !found {
            self.errors.push(CompileError::Node(
                "Could not find member".to_string(),
                member,
            ));
        }

        return true;
    }

    #[instrument(skip_all)]
    fn assign_type_float_literal(
        &mut self,
        spec: NumericSpecification,
        id: NodeId,
    ) -> Option<bool> {
        match spec {
            NumericSpecification::None => {
                self.types.insert(id, Type::FloatLiteral);
            }
            NumericSpecification::F32 => {
                self.types.insert(id, Type::F32);
            }
            NumericSpecification::F64 => {
                self.types.insert(id, Type::F64);
            }
            _ => {
                self.errors.push(CompileError::Node(
                    "Under-specified float literal".to_string(),
                    id,
                ));
                return Some(true);
            }
        }
        None
    }

    #[instrument(skip_all)]
    fn assign_type_let(
        &mut self,
        id: NodeId,
        name: NodeId,
        ty: Option<NodeId>,
        expr: Option<NodeId>,
        transparent: bool,
    ) -> Option<bool> {
        if let Some(expr) = expr {
            self.assign_type(expr);
            self.check_not_unspecified_polymorph(expr);
            self.match_types(id, expr);
        }

        if let Some(ty) = ty {
            self.assign_type(ty);
            self.check_not_unspecified_polymorph(ty);
            self.match_types(id, ty);
        }

        if transparent {
            if !self.types.contains_key(&id) {
                self.deferreds.push(id);
                self.transparent_deferreds.insert(id);
                return Some(false);
            }

            self.transparent_deferreds.remove(&id);

            if !self.propagate_transparency(id, name, self.node_scopes[id]) {
                return Some(false);
            }
        }

        None
    }

    #[instrument(skip_all)]
    fn debug_scope(&self, scope: ScopeId) -> Vec<String> {
        self.scopes[scope]
            .entries
            .keys()
            .map(|k| (self.string_interner.resolve(k.0).unwrap().to_string()))
            .collect::<Vec<_>>()
    }

    #[instrument(skip_all)]
    fn assign_type_value_param(&mut self, value: NodeId, id: NodeId, name: Option<NodeId>) {
        self.assign_type(value);
        self.check_not_unspecified_polymorph(value);
        self.match_types(id, value);
        if let Some(name) = name {
            self.match_types(name, value);
        }
    }

    #[instrument(skip_all)]
    fn assign_type_enum_decl_param(
        &mut self,
        id: NodeId,
        name: NodeId,
        ty: Option<NodeId>,
        transparent: bool,
    ) -> Option<bool> {
        if let Some(ty) = ty {
            self.assign_type(ty);
        }

        if let Some(ty) = ty {
            self.check_not_unspecified_polymorph(ty);
            self.match_types(id, ty);
        }

        if transparent {
            if !self.types.contains_key(&id) {
                self.deferreds.push(id);
                self.transparent_deferreds.insert(id);
                return Some(false);
            }

            self.transparent_deferreds.remove(&id);

            if !self.propagate_transparency(id, name, self.node_scopes[id]) {
                return Some(false);
            }
        }

        None
    }

    #[instrument(skip_all)]
    fn assign_type_struct_decl_param(
        &mut self,
        id: NodeId,
        name: NodeId,
        ty: Option<NodeId>,
        default: Option<NodeId>,
        transparent: bool,
    ) -> Option<bool> {
        if let Some(ty) = ty {
            self.assign_type(ty);
        }

        if let Some(default) = default {
            self.assign_type(default);
            self.match_types(id, default);
        }

        if let Some(ty) = ty {
            self.check_not_unspecified_polymorph(ty);
            self.match_types(id, ty);
        }

        if transparent {
            if !self.types.contains_key(&id) {
                self.deferreds.push(id);
                self.transparent_deferreds.insert(id);
                return Some(false);
            }

            self.transparent_deferreds.remove(&id);

            if !self.propagate_transparency(id, name, self.node_scopes[id]) {
                return Some(false);
            }
        }

        None
    }

    #[instrument(skip_all)]
    fn assign_type_fn_decl_param(
        &mut self,
        id: NodeId,
        name: NodeId,
        ty: Option<NodeId>,
        default: Option<NodeId>,
        transparent: bool,
    ) -> Option<bool> {
        if let Some(ty) = ty {
            self.assign_type(ty);

            if self.polymorph_sources.contains(&ty) {
                self.errors.push(CompileError::Node(
                    format!(
                        "Generic arguments needed: {} is not a concrete type",
                        self.nodes[ty].ty()
                    ),
                    ty,
                ));
                return Some(true);
            }
        }
        if let Some(default) = default {
            self.assign_type(default);
            self.match_types(id, default);
        }

        if let Some(ty) = ty {
            self.match_types(id, ty);
        }

        if transparent {
            if !self.types.contains_key(&id) {
                self.deferreds.push(id);
                self.transparent_deferreds.insert(id);
                return Some(false);
            }

            self.transparent_deferreds.remove(&id);

            if !self.propagate_transparency(id, name, self.node_scopes[id]) {
                return Some(false);
            }
        }

        None
    }

    #[instrument(skip_all)]
    fn assign_type_symbol(&mut self, sym: Sym, id: NodeId) -> bool {
        let resolved = self.scope_get(sym, id);

        match resolved {
            Some(resolved) => {
                self.assign_type(resolved);
                self.match_types(id, resolved);

                if self.polymorph_sources.contains(&resolved) {
                    self.polymorph_sources.insert(id);
                }

                return true;
            }
            None => {
                self.defer_on(
                    &[id],
                    CompileError::Node(format!("Symbol not found: {}", id.0), id),
                );
                return false;
            }
        }
    }

    #[instrument(skip_all)]
    fn assign_type_string_literal(&mut self, id: NodeId) {
        self.types.insert(id, Type::String);
    }

    #[instrument(skip_all)]
    fn assign_type_bool(&mut self, id: NodeId) {
        self.types.insert(id, Type::Bool);
    }

    #[instrument(skip_all)]
    fn assign_type_int_literal(&mut self, spec: NumericSpecification, id: NodeId) {
        match spec {
            NumericSpecification::None => {
                self.types.insert(id, Type::IntLiteral);
            }
            NumericSpecification::I8 => {
                self.types.insert(id, Type::I8);
            }
            NumericSpecification::I16 => {
                self.types.insert(id, Type::I16);
            }
            NumericSpecification::I32 => {
                self.types.insert(id, Type::I32);
            }
            NumericSpecification::I64 => {
                self.types.insert(id, Type::I64);
            }
            NumericSpecification::U8 => {
                self.types.insert(id, Type::U8);
            }
            NumericSpecification::U16 => {
                self.types.insert(id, Type::U16);
            }
            NumericSpecification::U32 => {
                self.types.insert(id, Type::U32);
            }
            NumericSpecification::U64 => {
                self.types.insert(id, Type::U64);
            }
            NumericSpecification::F32 => {
                self.types.insert(id, Type::F32);
            }
            NumericSpecification::F64 => {
                self.types.insert(id, Type::F64);
            }
        }
    }

    #[instrument(skip_all)]
    fn assign_type_return(&mut self, ret_id: Option<NodeId>, id: NodeId) {
        match self.returns.last_mut() {
            Some(r) => r.push(id),
            _ => {
                self.errors.push(CompileError::Node(
                    "Return outside of a block!".to_string(),
                    id,
                ));
                return;
            }
        }

        if let Some(ret_id) = ret_id {
            self.assign_type(ret_id);
            self.match_types(id, ret_id);
        }
    }

    #[instrument(skip_all)]
    fn assign_type_type(&mut self, id: NodeId, ty: Type) {
        self.types.insert(id, ty.clone());

        match ty {
            Type::Func {
                return_ty,
                input_tys,
                ..
            } => {
                if let Some(return_ty) = return_ty {
                    self.assign_type(return_ty);
                }

                for &input_ty in input_tys.borrow().iter() {
                    self.assign_type(input_ty);
                }
            }
            Type::Struct { params, decl, .. } | Type::Enum { params, decl, .. } => {
                if let Some(decl) = decl {
                    self.assign_type(decl);
                }

                for &field in params.borrow().iter() {
                    self.assign_type(field);
                }
            }
            Type::Pointer(ty) => {
                self.assign_type(ty);
            }
            Type::Array(ty, _) => {
                self.assign_type(ty);
            }
            Type::Empty
            | Type::IntLiteral
            | Type::I8
            | Type::I16
            | Type::I32
            | Type::I64
            | Type::U8
            | Type::U16
            | Type::U32
            | Type::U64
            | Type::FloatLiteral
            | Type::F32
            | Type::F64
            | Type::Bool
            | Type::String
            | Type::EnumNoneType
            | Type::Infer(_) => {}
        }
    }

    #[instrument(skip_all)]
    fn assign_type_extern(&mut self, return_ty: Option<NodeId>, params: IdVec, id: NodeId) {
        if let Some(return_ty) = return_ty {
            self.assign_type(return_ty);
        }

        for &param in params.borrow().iter() {
            self.assign_type(param);
        }

        self.types.insert(
            id,
            Type::Func {
                return_ty,
                input_tys: params,
            },
        );
    }

    #[instrument(skip_all)]
    fn assign_type_fn(
        &mut self,
        id: NodeId,
        return_ty: Option<NodeId>,
        params: IdVec,
        stmts: IdVec,
        returns: IdVec,
        transparent: bool,
    ) -> Option<bool> {
        // don't directly codegen a polymorph, wait until it's copied first
        if self.polymorph_sources.contains(&id) {
            return Some(true);
        }
        if let Some(return_ty) = return_ty {
            self.assign_type(return_ty);
        }
        for &param in params.borrow().iter() {
            self.assign_type(param);
        }
        self.types.insert(
            id,
            Type::Func {
                return_ty,
                input_tys: params,
            },
        );
        for &stmt in stmts.borrow().iter() {
            self.ensure_not_already_exited_block(stmt);
            self.assign_type(stmt);
        }
        for &ret_id in returns.borrow().iter() {
            let ret_id = match self.nodes[ret_id].clone() {
                Node::Return(Some(id)) => Ok(id),
                Node::Return(None) if return_ty.is_some() => Err(CompileError::Node(
                    "Empty return not allowed".to_string(),
                    ret_id,
                )),
                Node::Return(None) => Ok(ret_id),
                a => panic!("Expected return, got {:?}", a),
            };

            match (ret_id, return_ty) {
                (Err(err), _) => self.errors.push(err),
                (Ok(ret_id), Some(return_ty)) => {
                    self.match_types(return_ty, ret_id);
                    self.check_not_unspecified_polymorph(return_ty);
                }
                (Ok(ret_id), None) => {
                    if let Node::Return(Some(_)) = self.nodes[ret_id] {
                        self.errors.push(CompileError::Node(
                            "Return type not specified".to_string(),
                            ret_id,
                        ));
                    }
                }
            }
        }

        if let (Some(return_ty), true) = (return_ty, transparent) {
            if !self.types.contains_key(&return_ty) {
                self.deferreds.push(id);
                return Some(false);
            }

            if !self.propagate_transparency(id, id, self.node_scopes[id]) {
                return Some(false);
            }
        }

        if !self.topo.contains(&id) {
            self.topo.push(id);
        }

        None
    }

    #[instrument(skip_all)]
    fn assign_type_inner_call(&mut self, id: NodeId, mut func: NodeId, params: IdVec) -> bool {
        self.assign_type(func);

        // If func is a polymorph, copy it first
        if self.polymorph_sources.contains(&func) {
            let &key = self.polymorph_sources.get(&func).unwrap();
            match self.copy_polymorph(key, ParseTarget::FnDefinition) {
                Ok(func_id) => {
                    func = func_id;
                    self.assign_type(func);
                    self.nodes[id] = Node::Call {
                        func,
                        params: params.clone(),
                    };
                }
                Err(err) => {
                    self.errors.push(err);
                    return true;
                }
            }
        }

        let param_ids = params.clone();
        for &param in param_ids.borrow().iter() {
            self.assign_type(param);
        }

        match self.get_type(func) {
            Type::Func {
                input_tys,
                return_ty,
            } => {
                let given = param_ids;
                let decl = input_tys.clone();

                let rearranged_given = match self.rearrange_params(
                    &given.borrow(),
                    Some(self.node_scopes[id]),
                    &decl.borrow(),
                    id,
                ) {
                    Ok(r) => r,
                    Err(err) => {
                        self.errors.push(err);
                        return true;
                    }
                };

                *params.borrow_mut() = rearranged_given.clone();

                let input_ty_ids = input_tys.clone();

                if input_ty_ids.borrow().len() != rearranged_given.len() {
                    self.errors.push(CompileError::Node(
                        "Incorrect number of parameters".to_string(),
                        id,
                    ));
                    return true;
                } else {
                    for (param, input_ty) in
                        rearranged_given.iter().zip(input_ty_ids.borrow().iter())
                    {
                        self.match_types(*param, *input_ty);
                    }

                    if let Some(return_ty) = return_ty {
                        self.match_types(id, return_ty);
                    }
                }
            }
            Type::Infer(_) => {
                self.defer_on(
                    &[id],
                    CompileError::Node(format!("Could not infer type of for called function"), id),
                );
                return false;
            }
            ty => {
                self.errors
                    .push(CompileError::Node(format!("Not a function: {:?}", ty), id));
            }
        }

        return true;
    }

    #[instrument(skip_all)]
    fn match_params_to_named_struct(
        &mut self,
        params: IdVec,
        name: NodeId,
        struct_literal_id: NodeId,
    ) -> EmptyDraftResult {
        let Some(Type::Struct { params: decl, .. }) = self.types.get(&name) else {
            return Err(CompileError::Node(
                format!("Not a struct: {:?}", self.types.get(&name)),
                struct_literal_id,
            ));
        };

        let given = params.clone();
        let decl = decl.clone();

        let rearranged = match self.rearrange_params(
            &given.borrow(),
            Some(self.node_scopes[struct_literal_id]),
            &decl.borrow(),
            struct_literal_id,
        ) {
            Ok(r) => Some(r),
            Err(err) => return Err(err),
        };

        if let Some(rearranged) = rearranged {
            *params.borrow_mut() = rearranged.clone();

            if rearranged.len() != decl.borrow().len() {
                self.errors.push(CompileError::Node(
                    "Incorrect number of parameters".to_string(),
                    struct_literal_id,
                ));
            } else {
                for (field, decl_field) in rearranged.iter().zip(decl.borrow().iter()) {
                    self.assign_type(*field);
                    self.match_types(*field, *decl_field);
                }
            }
        }

        Ok(())
    }

    #[instrument(skip_all)]
    pub fn copy_polymorph_if_needed(&mut self, ty: NodeId) -> DraftResult<NodeId> {
        if let Some(&ty) = self.polymorph_sources.get(&ty) {
            let parse_target = match self.nodes[ty].clone() {
                Node::StructDefinition { .. } => ParseTarget::StructDeclaration,
                Node::EnumDefinition { .. } => ParseTarget::EnumDefinition,
                Node::FnDefinition { .. } => ParseTarget::FnDefinition,
                a => panic!("Expected struct, enum or fn definition: got {:?}", a),
            };

            self.copy_polymorph(ty, parse_target)
        } else {
            Ok(ty)
        }
    }

    #[instrument(skip_all)]
    pub fn check_not_unspecified_polymorph(&mut self, value: NodeId) {
        let mut base_value = value;
        let mut ty = self.get_type(value);
        while let Type::Pointer(pt) = ty {
            ty = self.get_type(pt);
            base_value = pt;
        }

        if self.polymorph_sources.contains(&value) || self.polymorph_sources.contains(&base_value) {
            self.errors.push(CompileError::Node(
                format!(
                    "Generic arguments needed: {} is not a concrete type",
                    self.nodes[value].ty()
                ),
                value,
            ));
        }
    }

    #[instrument(skip_all)]
    fn propagate_transparency(
        &mut self,
        type_id: NodeId,
        mut name: NodeId,
        scope: ScopeId,
    ) -> bool {
        let mut ty = self.get_type(type_id);

        let mut should_deep_copy = false;

        while let Type::Func { return_ty, .. } = ty {
            should_deep_copy = true;

            if let Some(return_ty) = return_ty {
                ty = self.get_type(return_ty);
                let params = self.push_id_vec(vec![]);
                name = self.push_node(self.ranges[name], Node::Call { func: name, params });
                self.assign_type(name);
            } else {
                break;
            }
        }

        while let Type::Pointer(base_ty) = ty {
            ty = self.types[&base_ty].clone();
        }

        let (Type::Struct {
            scope: Some(struct_scope),
            params,
            ..
        }
        | Type::Enum {
            scope: Some(struct_scope),
            params,
            ..
        }) = ty
        else {
            self.errors.push(CompileError::Node(
                format!("Type not eligible for transparency: {:?}", ty),
                name,
            ));
            return true;
        };

        for param in params.borrow().iter() {
            if self.transparent_deferreds.contains(param) {
                self.deferreds.push(*param);
                self.deferreds.push(type_id);
                return false;
            }
        }

        // todo(chad): @clone
        for (entry_name, entry) in self.scopes[struct_scope].entries.clone() {
            let name = if should_deep_copy {
                self.deep_copy_node(name)
            } else {
                name
            };

            let transparent_member_access = self.push_node(
                self.ranges[entry],
                Node::MemberAccess {
                    value: name,
                    member: entry,
                },
            );
            self.assign_type(transparent_member_access);

            self.scope_insert_into_scope_id(entry_name, transparent_member_access, scope);
        }

        return true;
    }

    #[instrument(skip_all)]
    pub fn deep_copy_node(&mut self, id: NodeId) -> NodeId {
        let range = self.ranges[id];

        match self.nodes[id].clone() {
            Node::Symbol(sym) => self.push_node(range, Node::Symbol(sym)),
            Node::PolySpecialize {
                sym: _,
                overrides: _,
                copied: _,
            } => todo!(),
            Node::PolySpecializeOverride { sym: _, ty: _ } => todo!(),
            Node::IntLiteral(_, _) => todo!(),
            Node::FloatLiteral(_, _) => todo!(),
            Node::StringLiteral(_) => id,
            Node::BoolLiteral(_) => todo!(),
            Node::Type(_) => todo!(),
            Node::Return(r) => {
                let r = r.map(|r| self.deep_copy_node(r));
                self.push_node(range, Node::Return(r))
            }
            Node::Break(_, _) => todo!(),
            Node::Continue(_) => todo!(),
            Node::Let {
                name: _,
                ty: _,
                expr: _,
                transparent: _,
            } => todo!(),
            Node::Assign {
                name: _,
                expr: _,
                is_store: _,
            } => todo!(),
            Node::FnDefinition {
                name,
                scope,
                params,
                return_ty,
                stmts,
                returns,
                transparent,
            } => {
                let name = name.map(|name| self.deep_copy_node(name));
                let params = self.deep_copy_id_vec(params);
                let return_ty = return_ty.map(|ty| self.deep_copy_node(ty));
                let stmts = self.deep_copy_id_vec(stmts);

                self.push_node(
                    range,
                    Node::FnDefinition {
                        name,
                        scope,
                        params,
                        return_ty,
                        stmts,
                        returns,
                        transparent,
                    },
                )
            }
            Node::Block {
                label: _,
                stmts: _,
                breaks: _,
                is_standalone: _,
            } => todo!(),
            Node::Extern {
                name: _,
                params: _,
                return_ty: _,
            } => todo!(),
            Node::StructDeclParam {
                name: _,
                ty: _,
                default: _,
                index: _,
                transparent: _,
            } => todo!(),
            Node::EnumDeclParam {
                name: _,
                ty: _,
                transparent: _,
            } => todo!(),
            Node::FnDeclParam {
                name: _,
                ty: _,
                default: _,
                index: _,
                transparent: _,
            } => todo!(),
            Node::ValueParam { name, value, index } => {
                let name = name.map(|name| self.deep_copy_node(name));
                let value = self.deep_copy_node(value);

                self.push_node(range, Node::ValueParam { name, value, index })
            }
            Node::BinOp {
                op: _,
                lhs: _,
                rhs: _,
            } => todo!(),
            Node::Call { func, params } => {
                // let func = self.deep_copy_node(func);
                let params = self.deep_copy_id_vec(params);
                self.push_node(range, Node::Call { func, params })
            }
            Node::ThreadingCall { func: _, params: _ } => todo!(),
            Node::ThreadingParamTarget => todo!(),
            Node::ArrayAccess { array: _, index: _ } => todo!(),
            Node::StructDefinition {
                name: _,
                params: _,
                scope: _,
            } => todo!(),
            Node::StructLiteral { name: _, params: _ } => todo!(),
            Node::EnumDefinition {
                scope: _,
                name: _,
                params: _,
            } => todo!(),
            Node::ArrayLiteral { members: _, ty: _ } => todo!(),
            Node::MemberAccess {
                value: _,
                member: _,
            } => todo!(),
            Node::StaticMemberAccess {
                value: _,
                member: _,
                resolved: _,
            } => todo!(),
            Node::AddressOf(_) => todo!(),
            Node::Deref(_) => todo!(),
            Node::If {
                cond: _,
                then_block: _,
                then_label: _,
                else_block: _,
                else_label: _,
            } => todo!(),
            Node::For {
                label: _,
                iterable: _,
                block: _,
                block_label: _,
            } => todo!(),
            Node::While {
                cond: _,
                block: _,
                block_label: _,
            } => todo!(),
            Node::Match { value: _, cases: _ } => todo!(),
            Node::MatchCase {
                tag: _,
                alias: _,
                block: _,
                block_label: _,
            } => todo!(),
            Node::Cast { ty: _, value: _ } => todo!(),
            Node::SizeOf(_) => todo!(),
            Node::TypeInfo(_) => todo!(),
            Node::AsCast {
                value: _,
                ty: _,
                style: _,
            } => todo!(),
        }
    }

    #[instrument(skip_all)]
    pub fn deep_copy_id_vec(&mut self, v: IdVec) -> IdVec {
        let v = v
            .borrow_mut()
            .iter()
            .map(|&id| self.deep_copy_node(id))
            .collect::<Vec<_>>();

        Rc::new(RefCell::new(v))
    }
}
