use tracing::instrument;

use crate::{
    CompileError, Context, DraftResult, IdVec, Node, NodeElse, NodeId, NumericSpecification, Op,
    ParseTarget, StaticMemberResolution, Sym,
};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Type {
    Empty,
    SelfPointer,
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
        name: Option<NodeId>,
        params: IdVec,
    },
    Enum {
        name: Option<NodeId>,
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

#[derive(Default, Debug)]
pub struct UnificationData {
    pub future_matches: Vec<(NodeId, NodeId)>,
}

impl UnificationData {
    pub fn reset(&mut self) {
        self.future_matches.clear();
    }
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
        self.unification_data.reset();

        for uid in 0..self.type_matches.len() {
            if !self.type_matches[uid].changed {
                continue;
            }
            self.type_matches[uid].changed = false;

            let most_specific_ty = self.type_matches[uid].unified;
            if matches!(most_specific_ty, Type::Infer(_)) {
                continue;
            }

            for &id in self.type_matches[uid].ids.iter() {
                self.types.insert(id, most_specific_ty);
            }
        }

        while !self.unification_data.future_matches.is_empty() {
            let (id1, id2) = self.unification_data.future_matches.pop().unwrap();
            self.match_types(id1, id2);
        }
    }

    #[instrument(skip_all)]
    pub fn unify(&mut self, first: Type, second: Type, err_ids: (NodeId, NodeId)) -> Type {
        match (first, second) {
            (a, b) if a == b => a,

            // Types/Inferreds get coerced to anything
            (Type::Infer(_), other) | (other, Type::Infer(_)) => other,

            // Check int/float literals match
            (Type::IntLiteral, bt) | (bt, Type::IntLiteral) if bt.is_int() => bt,
            (Type::FloatLiteral, bt) | (bt, Type::FloatLiteral) if bt.is_float() => bt,

            (Type::Struct { name: n1, .. }, Type::Struct { name: n2, .. }) => {
                // prefer named to unnamed structs
                match (n1, n2) {
                    (Some(_), None) => first,
                    (None, Some(_)) => second,
                    _ => first, // if both are named or both are unnamed, it doesn't matter which is chosen
                }
            }

            // Coerce array with an inferred length to an array with a known length
            (Type::Array(_, ArrayLen::Some(_)), Type::Array(_, ArrayLen::Infer)) => first,
            (Type::Array(_, ArrayLen::Infer), Type::Array(_, ArrayLen::Some(_))) => second,

            // For aggregate types, the type matcher should have already detected a mismatch
            // so it doesn't really matter which is chosen
            (Type::Func { .. }, Type::Func { .. })
            | (Type::Enum { .. }, Type::Enum { .. })
            | (Type::Pointer(_) | Type::SelfPointer, Type::Pointer(_) | Type::SelfPointer)
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
                } = self.nodes[id_to_push] else { panic!() };

                // Add the name
                let name = decl[given_idx];
                match self.nodes[name] {
                    Node::StructDeclParam { name, .. }
                    | Node::FnDeclParam { name, .. }
                    | Node::EnumDeclParam { name, .. } => {
                        self.nodes[id_to_push] = Node::ValueParam {
                            name: Some(name),
                            value: existing_value,
                            index: existing_index,
                        };
                    }
                    _ => panic!(),
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
                Node::FnDeclParam { name, .. } | Node::StructDeclParam { name, .. } => *name,
                _ => unreachable!(),
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
    pub fn match_addressable(&mut self, n1: NodeId, n2: NodeId) {
        let id1 = self.find_addressable_array_index(n1);
        let id2 = self.find_addressable_array_index(n2);

        match (id1, id2) {
            (None, None) => {
                if self.addressable_nodes.contains(&n1) {
                    self.addressable_nodes.insert(n2);
                    return;
                }

                if self.addressable_nodes.contains(&n2) {
                    self.addressable_nodes.insert(n1);
                    return;
                }

                let unified =
                    self.addressable_nodes.contains(&n1) || self.addressable_nodes.contains(&n2);
                self.addressable_matches.push(AddressableMatch {
                    changed: true,
                    unified,
                    ids: vec![n1, n2],
                });

                self.addressable_array_reverse_map
                    .insert(n1, self.addressable_matches.len() - 1);
                self.addressable_array_reverse_map
                    .insert(n2, self.addressable_matches.len() - 1);
            }
            (Some(id), None) => {
                self.addressable_matches[id].ids.push(n2);
                self.addressable_array_reverse_map.insert(n2, id);
                self.addressable_matches[id].changed = true;
                self.addressable_matches[id].unified =
                    self.addressable_matches[id].unified || self.addressable_nodes.contains(&n2);

                if self.addressable_nodes.contains(&n1) {
                    for t in self.addressable_matches[id].ids.clone() {
                        self.addressable_nodes.insert(t);
                        self.addressable_array_reverse_map.remove(&t);
                    }
                    self.addressable_matches.swap_remove(id);
                    if self.addressable_matches.len() > id {
                        for t in self.addressable_matches[id].ids.clone() {
                            self.addressable_array_reverse_map.insert(t, id);
                        }
                    }
                } else if self.addressable_nodes.contains(&n2) {
                    for t in self.addressable_matches[id].ids.clone() {
                        self.addressable_nodes.insert(t);
                        self.addressable_array_reverse_map.remove(&t);
                    }
                    self.addressable_matches.swap_remove(id);
                    if self.addressable_matches.len() > id {
                        for t in self.addressable_matches[id].ids.clone() {
                            self.addressable_array_reverse_map.insert(t, id);
                        }
                    }
                }
            }
            (None, Some(id)) => {
                self.addressable_matches[id].ids.push(n1);
                self.addressable_array_reverse_map.insert(n1, id);
                self.addressable_matches[id].changed = true;
                self.addressable_matches[id].unified =
                    self.addressable_matches[id].unified || self.addressable_nodes.contains(&n1);

                if self.addressable_nodes.contains(&n2) {
                    for t in self.addressable_matches[id].ids.clone() {
                        self.addressable_nodes.insert(t);
                        self.addressable_array_reverse_map.remove(&t);
                    }
                    self.addressable_matches.swap_remove(id);
                    if self.addressable_matches.len() > id {
                        for t in self.addressable_matches[id].ids.clone() {
                            self.addressable_array_reverse_map.insert(t, id);
                        }
                    }
                } else if self.addressable_nodes.contains(&n1) {
                    for t in self.addressable_matches[id].ids.clone() {
                        self.addressable_nodes.insert(t);
                        self.addressable_array_reverse_map.remove(&t);
                    }
                    self.addressable_matches.swap_remove(id);
                    if self.addressable_matches.len() > id {
                        for t in self.addressable_matches[id].ids.clone() {
                            self.addressable_array_reverse_map.insert(t, id);
                        }
                    }
                }
            }
            (Some(id1), Some(id2)) if id1 != id2 => {
                let lower = id1.min(id2);
                let upper = id1.max(id2);

                let unified = self.addressable_matches[lower].unified
                    || self.addressable_matches[upper].unified;

                let (lower_matches, upper_matches) =
                    self.addressable_matches.split_at_mut(lower + 1);
                lower_matches[lower]
                    .ids
                    .extend(upper_matches[upper - lower - 1].ids.iter());
                for t in self.addressable_matches[lower].ids.clone() {
                    self.addressable_array_reverse_map.insert(t, lower);
                }

                self.addressable_matches[lower].changed = true;
                self.addressable_matches[lower].unified = unified;

                self.addressable_matches.swap_remove(upper);

                if self.addressable_matches.len() > upper {
                    for t in self.addressable_matches[upper].ids.clone() {
                        self.addressable_array_reverse_map.insert(t, upper);
                    }
                }

                if self.addressable_nodes.contains(&n1) {
                    for t in self.addressable_matches[lower].ids.clone() {
                        self.addressable_nodes.insert(t);
                    }
                } else if self.addressable_nodes.contains(&n2) {
                    for t in self.addressable_matches[lower].ids.clone() {
                        self.addressable_nodes.insert(t);
                    }
                }
            }
            (_, _) => (),
        }
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
            (Type::Pointer(_), Type::SelfPointer) | (Type::SelfPointer, Type::Pointer(_)) => {
                // This is always valid, because literally the only way to get a SelfPointer is by construction from the compiler.
                // So we already know it's valid
            }
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

                let input_tys1 = self.id_vecs[input_tys1].clone();
                let input_tys2 = self.id_vecs[input_tys2].clone();

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
                    name: n1,
                    params: f1,
                },
                Type::Struct {
                    name: n2,
                    params: f2,
                },
            )
            | (
                Type::Enum {
                    name: n1,
                    params: f1,
                },
                Type::Enum {
                    name: n2,
                    params: f2,
                },
            ) => match (n1, n2) {
                (Some(n1), Some(n2)) => {
                    let n1d = self.scope_get(self.get_symbol(n1), n1);
                    let n2d = self.scope_get(self.get_symbol(n2), n2);

                    if n1d != n2d {
                        self.errors.push(CompileError::Node2(
                            "Could not match types: declaration sites differ".to_string(),
                            n1,
                            n2,
                        ));
                    }

                    // Same declaration site means same number of parameters, so we can just match them up
                    for (f1, f2) in self.id_vecs[f1]
                        .clone()
                        .borrow()
                        .iter()
                        .zip(self.id_vecs[f2].clone().borrow().iter())
                    {
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
                    let f1 = self.id_vecs[f1].clone();
                    let f2 = self.id_vecs[f2].clone();

                    if f1.borrow().len() != f2.borrow().len() {
                        self.errors.push(CompileError::Node2(
                            "Could not match types: struct fields differ in length".to_string(),
                            ty1,
                            ty2,
                        ));
                    }

                    for (f1, f2) in f1.borrow().iter().zip(f2.borrow().iter()) {
                        self.match_types(*f1, *f2);
                    }
                }
            },
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
                if !self.check_int_literal_type(bt) {
                    self.errors.push(CompileError::Node2(
                        format!("Type mismatch - int literal with {:?}", bt),
                        ty1,
                        ty2,
                    ));
                }
            }
            (Type::FloatLiteral, bt) | (bt, Type::FloatLiteral) if bt.is_basic() => {
                if !self.check_float_literal_type(bt) {
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
            name: Some(_),
            params,
        } = ty
        {
            for &field in self.id_vecs[params].clone().borrow().iter() {
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
                    self.type_matches[id].unified,
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
                    self.type_matches[id].unified,
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
                    self.type_matches[lower].unified,
                    self.type_matches[upper].unified,
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

        if self.polymorph_sources.contains_key(&id) {
            return;
        }

        if self.circular_dependency_nodes.contains(&id) {
            return;
        }
        self.circular_dependency_nodes.insert(id);

        if self.assign_type_inner(id) {
            self.completes.insert(id);
        }
    }

    #[instrument(skip_all)]
    pub fn assign_type_inner(&mut self, id: NodeId) -> bool {
        match self.nodes[id] {
            Node::FnDefinition {
                params,
                return_ty,
                stmts,
                returns,
                ..
            } => {
                if let Some(value) = self.assign_type_fn(id, return_ty, params, stmts, returns) {
                    return value;
                }
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
                self.assign_type_symbol(sym, id);
            }
            Node::FnDeclParam { ty, default, .. } => {
                if let Some(value) = self.assign_type_fn_decl_param(ty, id, default) {
                    return value;
                }
            }
            Node::StructDeclParam { ty, default, .. } => {
                self.assign_type_struct_decl_param(ty, default, id);
            }
            Node::EnumDeclParam { ty, .. } => {
                self.assign_type_enum_decl_param(ty, id);
            }
            Node::ValueParam { name, value, .. } => {
                self.assign_type_value_param(value, id, name);
            }
            Node::Let { ty, expr, .. } => {
                if let Some(value) = self.assign_type_let(expr, id, ty) {
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
                        self.match_types(lhs, rhs);
                        self.match_types(id, lhs);
                    }
                    Op::EqEq | Op::Neq | Op::Gt | Op::Lt | Op::GtEq | Op::LtEq => {
                        self.match_types(lhs, rhs);
                        self.types.insert(id, Type::Bool);
                    }
                }
            }
            Node::Call { func, params, .. } | Node::ThreadingCall { func, params } => {
                return self.assign_type_inner_call(id, func, params);
            }
            Node::StructDefinition { name, params, .. } => {
                // don't directly codegen a polymorph, wait until it's copied first
                if self.polymorph_sources.contains_key(&id) {
                    return true;
                }

                let param_ids = self.id_vecs[params].clone();
                for &param in param_ids.borrow().iter() {
                    self.assign_type(param);
                }

                self.types.insert(id, Type::Struct { name, params });

                if let Some(name) = name {
                    self.match_types(id, name);
                }
            }
            Node::EnumDefinition { name, params } => {
                // don't directly codegen a polymorph, wait until it's copied first
                if self.polymorph_sources.contains_key(&id) {
                    return true;
                }

                let param_ids = self.id_vecs[params].clone();
                for &param in param_ids.borrow().iter() {
                    if let Node::EnumDeclParam { ty: None, .. } = self.nodes[param] {
                        self.types.insert(param, Type::EnumNoneType);
                    } else {
                        self.assign_type(param);
                    }
                }

                self.types.insert(
                    id,
                    Type::Enum {
                        name: Some(name),
                        params,
                    },
                );

                self.match_types(id, name);
            }
            Node::StructLiteral { name, params, .. } => {
                if let Some(name) = name {
                    self.assign_type(name);

                    // If this is a polymorph, copy it first
                    let name = if self.polymorph_sources.contains_key(&name) {
                        let &key = self.polymorph_sources.get(&name).unwrap();
                        match self.polymorph_copy(key, ParseTarget::StructDefinition) {
                            Ok(copied) => {
                                self.assign_type(copied);
                                self.nodes[id] = Node::StructLiteral {
                                    name: Some(copied),
                                    params,
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
                    for &field in self.id_vecs[params].clone().borrow().iter() {
                        self.assign_type(field);
                    }
                    self.types.insert(id, Type::Struct { name: None, params });
                }
            }
            Node::MemberAccess { value, member } => {
                self.assign_type(value);

                let member_name_sym = self.get_symbol(member);

                let mut value_ty = self.get_type(value);

                while let Type::Pointer(ty) = value_ty {
                    value_ty = self.get_type(ty);
                }

                match value_ty {
                    Type::Struct { params, .. } => {
                        let field_ids = self.id_vecs[params].clone();
                        let mut found = false;

                        for &field in field_ids.borrow().iter() {
                            let field_name = match &self.nodes[field] {
                                Node::ValueParam {
                                    name: Some(name), ..
                                }
                                | Node::StructDeclParam { name, .. } => *name,
                                a => {
                                    self.errors.push(CompileError::Node(
                                        format!(
                                            "Cannot perform member access as this field's name (at {:?}) could not be found. Node type: {:?}",
                                            &self.ranges[field],
                                            a.ty()
                                        ),
                                        id,
                                    ));
                                    return true;
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
                                "Could not find member".to_string(),
                                member,
                            ));
                        }
                    }
                    Type::Array(array_ty, len) => {
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
                                self.types.insert(id, Type::Pointer(array_ty));
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
                        self.errors.push(CompileError::Node(
                            format!("Member access on a non-struct (type {:?})", value_ty),
                            id,
                        ));
                    }
                }
            }
            Node::StaticMemberAccess { value, member, .. } => {
                self.assign_type(value);

                // If this is a polymorph, copy it first
                let value = if self.polymorph_sources.contains_key(&value) {
                    let &key = self.polymorph_sources.get(&value).unwrap();
                    match self.polymorph_copy(key, ParseTarget::EnumDefinition) {
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
                    self.addressable_nodes.insert(value);
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
            } => {
                self.assign_type(cond);

                self.assign_type(then_block);
                self.match_types(id, then_block);

                match else_block {
                    NodeElse::Block(else_block) => {
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

                self.addressable_nodes.insert(id);
            }
            Node::Resolve(r) => match r {
                Some(r) => {
                    self.assign_type(r);
                    self.match_types(id, r);
                }
                None => {
                    self.types.insert(id, Type::Empty);
                }
            },
            Node::Block {
                stmts, resolves, ..
            } => {
                for &stmt in self.id_vecs[stmts].clone().borrow().iter() {
                    self.assign_type(stmt);
                }

                let resolves = self.id_vecs[resolves].clone();

                if resolves.borrow().is_empty() {
                    self.types.insert(id, Type::Empty);
                }

                for &resolve in resolves.borrow().iter() {
                    self.assign_type(resolve);
                    self.match_types(id, resolve);
                }
            }
            Node::ArrayLiteral { members, ty } => {
                for &member in self.id_vecs[members].clone().borrow().iter() {
                    self.assign_type(member);
                    self.match_types(member, ty);
                    self.check_not_unspecified_polymorph(member);
                }

                self.types.insert(
                    id,
                    Type::Array(ty, ArrayLen::Some(self.id_vecs[members].borrow().len())),
                );
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
                    overrides,
                    copied: Some(copied),
                };

                let overrides = self.id_vecs[overrides].clone();
                for o in overrides.borrow().iter() {
                    self.assign_type(*o);
                    let Node::PolySpecializeOverride { sym, ty } = self.nodes[o] else { panic!() };
                    let sym = self.get_symbol(sym);

                    let Node::StructDefinition {
                        scope: scope_id, ..
                    } = self.nodes[copied] else { panic!() };

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
                self.match_addressable(id, value);

                self.assign_type(value);
            }
            Node::SizeOf(ty) => {
                self.assign_type(ty);
                self.types.insert(id, Type::I64);
            }
            Node::For {
                iterable,
                label,
                block,
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

                self.assign_type(block);
            }
            Node::While { cond, block } => {
                self.assign_type(cond);

                match self.get_type(cond) {
                    Type::Infer(_) => {
                        self.deferreds.push(id);
                        return false;
                    }
                    Type::Bool => {}
                    _ => todo!(),
                }

                self.assign_type(block);
            }
            Node::ThreadingParamTarget => todo!(),
            Node::AsCast { value, ty, .. } => {
                self.assign_type(value);
                self.assign_type(ty);

                match (self.get_type(value), self.get_type(ty)) {
                    (Type::Infer(_), _) | (_, Type::Infer(_)) => {
                        self.deferreds.push(id);
                        return false;
                    }
                    _ => todo!(),
                }

                self.match_addressable(id, value);

                // self.match_types(value, ty);

                self.match_types(id, ty);
            }
        }

        return true;
    }

    fn assign_type_static_member_access_enum(
        &mut self,
        params: IdVec,
        member: NodeId,
        value: NodeId,
        id: NodeId,
    ) -> bool {
        let field_ids = self.id_vecs[params].clone();
        let mut found = false;

        for (index, &field) in field_ids.borrow().iter().enumerate() {
            self.assign_type(field);

            let (field_name, is_none_type) = match &self.nodes[field] {
                Node::EnumDeclParam { name, ty } => (*name, ty.is_none()),
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
                    value: value,
                    member: member,
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
        expr: Option<NodeId>,
        id: NodeId,
        ty: Option<NodeId>,
    ) -> Option<bool> {
        if let Some(expr) = expr {
            self.assign_type(expr);
            self.match_types(id, expr);
            self.check_not_unspecified_polymorph(expr);
        }

        if let Some(ty) = ty {
            if self.polymorph_sources.contains_key(&ty) {
                self.errors.push(CompileError::Node(
                    format!(
                        "Generic arguments needed: {} is not a concrete type",
                        self.nodes[ty].ty()
                    ),
                    ty,
                ));
                return Some(true);
            }

            self.assign_type(ty);
            self.match_types(id, ty);
        }
        None
    }

    #[instrument(skip_all)]
    fn assign_type_value_param(&mut self, value: NodeId, id: NodeId, name: Option<NodeId>) {
        self.assign_type(value);
        self.check_not_unspecified_polymorph(value);
        self.match_types(id, value);
        if let Some(name) = name {
            self.match_types(name, value);
        }

        self.match_addressable(id, value);
    }

    #[instrument(skip_all)]
    fn assign_type_enum_decl_param(&mut self, ty: Option<NodeId>, id: NodeId) {
        if let Some(ty) = ty {
            self.assign_type(ty);
        }

        if let Some(ty) = ty {
            self.match_types(id, ty);
        }
    }

    #[instrument(skip_all)]
    fn assign_type_struct_decl_param(
        &mut self,
        ty: Option<NodeId>,
        default: Option<NodeId>,
        id: NodeId,
    ) {
        if let Some(ty) = ty {
            self.assign_type(ty);
        }

        if let Some(default) = default {
            self.assign_type(default);
            self.match_types(id, default);
        }

        if let Some(ty) = ty {
            self.match_types(id, ty);
        }
    }

    #[instrument(skip_all)]
    fn assign_type_fn_decl_param(
        &mut self,
        ty: Option<NodeId>,
        id: NodeId,
        default: Option<NodeId>,
    ) -> Option<bool> {
        if let Some(ty) = ty {
            self.assign_type(ty);

            if self.polymorph_sources.contains_key(&ty) {
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
        if self.id_is_aggregate_type(id) {
            self.addressable_nodes.insert(id);
        }
        if let Some(default) = default {
            self.assign_type(default);
            self.match_types(id, default);
        }

        if let Some(ty) = ty {
            self.match_types(id, ty);
        }
        None
    }

    #[instrument(skip_all)]
    fn assign_type_symbol(&mut self, sym: Sym, id: NodeId) {
        let resolved = self.scope_get(sym, id);

        match resolved {
            Some(resolved) => {
                self.assign_type(resolved);
                self.match_types(id, resolved);
                self.match_addressable(id, resolved);

                if let Some(&resolved) = self.polymorph_sources.get(&resolved) {
                    self.polymorph_sources.insert(id, resolved);
                }
            }
            None => {
                self.errors
                    .push(CompileError::Node("Symbol not found".to_string(), id));
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
        if let Some(ret_id) = ret_id {
            self.assign_type(ret_id);
            self.match_types(id, ret_id);
        }
    }

    #[instrument(skip_all)]
    fn assign_type_type(&mut self, id: NodeId, ty: Type) {
        self.types.insert(id, ty);

        match ty {
            Type::Func {
                return_ty,
                input_tys,
                ..
            } => {
                if let Some(return_ty) = return_ty {
                    self.assign_type(return_ty);
                }

                for &input_ty in self.id_vecs[input_tys].clone().borrow().iter() {
                    self.assign_type(input_ty);
                }
            }
            Type::Struct { name, params } => {
                if let Some(name) = name {
                    self.assign_type(name);
                }

                for &field in self.id_vecs[params].clone().borrow().iter() {
                    self.assign_type(field);
                }
            }
            Type::Enum { name, params } => {
                if let Some(name) = name {
                    self.assign_type(name);
                }

                for &field in self.id_vecs[params].clone().borrow().iter() {
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
            | Type::SelfPointer
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

        for &param in self.id_vecs[params].clone().borrow().iter() {
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
    ) -> Option<bool> {
        // don't directly codegen a polymorph, wait until it's copied first
        if self.polymorph_sources.contains_key(&id) {
            return Some(true);
        }
        if let Some(return_ty) = return_ty {
            self.assign_type(return_ty);
        }
        for &param in self.id_vecs[params].clone().borrow().iter() {
            self.assign_type(param);
        }
        self.types.insert(
            id,
            Type::Func {
                return_ty,
                input_tys: params,
            },
        );
        for &stmt in self.id_vecs[stmts].clone().borrow().iter() {
            self.assign_type(stmt);
        }
        for &ret_id in self.id_vecs[returns].clone().borrow().iter() {
            let ret_id = match self.nodes[ret_id] {
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

        if !self.topo.contains(&id) {
            self.topo.push(id);
        }

        None
    }

    #[instrument(skip_all)]
    fn assign_type_inner_call(&mut self, id: NodeId, mut func: NodeId, params: IdVec) -> bool {
        self.assign_type(func);

        // If func is a polymorph, copy it first
        if self.polymorph_sources.contains_key(&func) {
            let &key = self.polymorph_sources.get(&func).unwrap();
            match self.polymorph_copy(key, ParseTarget::FnDefinition) {
                Ok(func_id) => {
                    func = func_id;
                    self.assign_type(func);
                    self.nodes[id] = Node::Call { func, params };
                }
                Err(err) => {
                    self.errors.push(err);
                    return true;
                }
            }
        }

        let param_ids = self.id_vecs[params].clone();
        for &param in param_ids.borrow().iter() {
            self.assign_type(param);
        }

        match self.get_type(func) {
            Type::Func {
                input_tys,
                return_ty,
            } => {
                let given = param_ids;
                let decl = self.id_vecs[input_tys].clone();

                let rearranged_given =
                    match self.rearrange_params(&given.borrow(), &decl.borrow(), id) {
                        Ok(r) => r,
                        Err(err) => {
                            self.errors.push(err);
                            return true;
                        }
                    };

                *self.id_vecs[params].borrow_mut() = rearranged_given.clone();

                let input_ty_ids = self.id_vecs[input_tys].clone();

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

                        if self.should_pass_id_by_ref(return_ty) {
                            self.addressable_nodes.insert(id);
                        }
                    }
                }
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
    ) -> DraftResult<()> {
        let Some(Type::Struct { params: decl, .. }) = self.types.get(&name) else { return Ok(()); };

        let given = self.id_vecs[params].clone();
        let decl = self.id_vecs[*decl].clone();

        let rearranged =
            match self.rearrange_params(&given.borrow(), &decl.borrow(), struct_literal_id) {
                Ok(r) => Some(r),
                Err(err) => return Err(err),
            };

        if let Some(rearranged) = rearranged {
            *self.id_vecs[params].borrow_mut() = rearranged.clone();

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
            let parse_target = match self.nodes[ty] {
                Node::StructDefinition { .. } => ParseTarget::StructDefinition,
                Node::EnumDefinition { .. } => ParseTarget::EnumDefinition,
                Node::FnDefinition { .. } => ParseTarget::FnDefinition,
                a => panic!("Expected struct, enum or fn definition: got {:?}", a),
            };

            self.polymorph_copy(ty, parse_target)
        } else {
            Ok(ty)
        }
    }

    #[instrument(skip_all)]
    pub fn check_not_unspecified_polymorph(&mut self, value: NodeId) {
        if self.polymorph_sources.get(&value).is_some() {
            self.errors.push(CompileError::Node(
                format!(
                    "Generic arguments needed: {} is not a concrete type",
                    self.nodes[value].ty()
                ),
                value,
            ));
        }
    }
}
