use crate::{
    CompileError, Context, IdVec, Node, NodeId, NumericSpecification, Op, StaticMemberResolution,
    Sym,
};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Type {
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
}

impl Type {
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

    pub fn is_float(&self) -> bool {
        matches!(&self, Type::FloatLiteral | Type::F32 | Type::F64)
    }
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

    pub fn unify(&mut self, first: Type, second: Type, err_ids: (NodeId, NodeId)) -> Type {
        match (first, second) {
            (a, b) if a == b => a,

            // Types/Inferreds get coerced to anything
            (Type::Infer(_), other) | (other, Type::Infer(_)) => other,

            // Check int/float literals match
            (Type::IntLiteral, bt) | (bt, Type::IntLiteral) if bt.is_int() => bt,
            (Type::FloatLiteral, bt) | (bt, Type::FloatLiteral) if bt.is_float() => bt,

            // For aggregate types, the type matcher should have already detected a mismatch
            // so it doesn't really matter which is chosen
            (Type::Func { .. }, Type::Func { .. }) => first,
            (Type::Struct { .. }, Type::Struct { .. }) => first,
            (Type::Pointer(_), Type::Pointer(_)) => first,

            // Anything else
            _ => {
                self.errors.push(CompileError::Node2(
                    "Type mismatch".to_string(),
                    err_ids.0,
                    err_ids.1,
                ));
                Type::Infer(None)
            }
        }
    }

    pub fn rearrange_params(
        &self,
        given: &[NodeId],
        decl: &[NodeId],
        err_id: NodeId,
    ) -> Result<Vec<NodeId>, CompileError> {
        let given_len = given.len();

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
                rearranged_given.push(given[given_idx]);
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

            for &g in given.iter().skip(starting_rearranged_len) {
                let decl_name = match &self.nodes[d] {
                    Node::FuncDeclParam { name, .. } | Node::StructDeclParam { name, .. } => *name,
                    _ => unreachable!(),
                };
                let decl_name_sym = self.get_symbol(decl_name);

                let given_name = match &self.nodes[g] {
                    Node::ValueParam {
                        name: Some(name), ..
                    } => *name,
                    _ => unreachable!(),
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
                    Node::FuncDeclParam {
                        default: Some(def), ..
                    } => rearranged_given.push(def),
                    Node::EnumDeclParam { ty: None, .. } => {
                        // this is okay, as the enum *should* be constructed without a parameter. This is the name-only case
                    }
                    _ => {
                        return Err(CompileError::Node(
                            "Could not find parameter".to_string(),
                            err_id,
                        ));
                    }
                }
            }
        }

        Ok(rearranged_given)
    }

    pub fn match_types(&mut self, ty1: NodeId, ty2: NodeId) {
        self.handle_match_types(ty1, ty2);
        self.merge_type_matches(ty1, ty2); // todo(chad): can we just do this once at the end? Would it be faster?
    }

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

    pub fn handle_match_types(&mut self, ty1: NodeId, ty2: NodeId) {
        if ty1 == ty2 {
            return;
        }

        // println!(
        //     "matching {} ({:?}) with {} ({:?})",
        //     self.nodes[ty1].ty(),
        //     self.ranges[ty1],
        //     self.nodes[ty2].ty(),
        //     self.ranges[ty2]
        // );

        // println!(
        //     "matching {:?} ({:?}) with {:?} ({:?})",
        //     self.types.get(&ty1),
        //     self.ranges[ty1],
        //     self.types.get(&ty2),
        //     self.ranges[ty2]
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
            ) => match (n1, n2) {
                (Some(n1), Some(n2)) => {
                    let n1d = self.scope_get(self.get_symbol(n1), n1);
                    let n2d = self.scope_get(self.get_symbol(n2), n2);

                    if n1d != n2d {
                        self.errors.push(CompileError::Node2(
                            "Could not match types: struct declarations differ".to_string(),
                            n1,
                            n2,
                        ));
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
            (bt1, bt2) if bt1 == bt2 => (),
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

    pub fn check_float_literal_type(&self, bt: Type) -> bool {
        matches!(bt, Type::FloatLiteral | Type::F32 | Type::F64)
    }

    pub fn get_type(&self, id: NodeId) -> Type {
        return self.types.get(&id).cloned().unwrap_or(Type::Infer(None));
    }

    pub fn is_fully_concrete(&self, id: NodeId) -> bool {
        self.is_fully_concrete_ty(self.get_type(id))
    }

    pub fn is_fully_concrete_ty(&self, ty: Type) -> bool {
        if let Type::Pointer(pt) = ty {
            return self.is_fully_concrete(pt);
        }

        if let Type::Struct {
            name: Some(_),
            params,
        } = ty
        {
            for &field in self.id_vecs[params].borrow().iter() {
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

    pub fn find_type_array_index(&self, id: NodeId) -> Option<usize> {
        self.type_array_reverse_map.get(&id).cloned()
    }

    pub fn find_addressable_array_index(&self, id: NodeId) -> Option<usize> {
        self.addressable_array_reverse_map.get(&id).cloned()
    }

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

    pub fn assign_type(&mut self, id: NodeId) {
        if self.completes.contains(&id) {
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

    pub fn assign_type_inner(&mut self, id: NodeId) -> bool {
        match self.nodes[id] {
            Node::Func {
                params,
                return_ty,
                stmts,
                returns,
                ..
            } => {
                // don't directly codegen a polymorph, wait until it's copied first
                if self.polymorph_sources.contains(&id) {
                    return true;
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
                        _ => unreachable!(),
                    };

                    match (ret_id, return_ty) {
                        (Err(err), _) => self.errors.push(err),
                        (Ok(ret_id), Some(return_ty)) => {
                            self.match_types(return_ty, ret_id);
                        }
                        (Ok(ret_id), None) => {
                            self.errors.push(CompileError::Node(
                                "Return type not specified".to_string(),
                                ret_id,
                            ));
                        }
                    }
                }

                if !self.topo.contains(&id) {
                    self.topo.push(id);
                }
            }
            Node::Extern {
                params, return_ty, ..
            } => {
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
            Node::Type(ty) => {
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
                    | Type::Bool
                    | Type::EnumNoneType
                    | Type::Infer(_) => {}
                }
            }
            Node::Return(ret_id) => {
                if let Some(ret_id) = ret_id {
                    self.assign_type(ret_id);
                    self.match_types(id, ret_id);
                }
            }
            Node::IntLiteral(_, spec) => match spec {
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
            },
            Node::BoolLiteral(_) => {
                self.types.insert(id, Type::Bool);
            }
            Node::Symbol(sym) => {
                let resolved = self.scope_get(sym, id);

                match resolved {
                    Some(resolved) => {
                        self.assign_type(resolved);
                        self.match_types(id, resolved);
                        self.match_addressable(id, resolved);

                        if self.polymorph_sources.contains(&resolved) {
                            self.polymorph_sources.insert(id);
                        }
                    }
                    None => {
                        self.errors
                            .push(CompileError::Node("Symbol not found".to_string(), id));
                    }
                }
            }
            Node::StructDeclParam { ty, default, .. } | Node::FuncDeclParam { ty, default, .. } => {
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
            Node::EnumDeclParam { ty, .. } => {
                if let Some(ty) = ty {
                    self.assign_type(ty);
                }

                if let Some(ty) = ty {
                    self.match_types(id, ty);
                }
            }
            Node::ValueParam { name, value, .. } => {
                self.assign_type(value);
                self.match_types(id, value);
                if let Some(name) = name {
                    self.match_types(name, value);
                }
                self.match_addressable(id, value);
            }
            Node::Let { name: _, ty, expr } => {
                if let Some(expr) = expr {
                    self.assign_type(expr);
                    self.match_types(id, expr);
                }

                if let Some(ty) = ty {
                    self.assign_type(ty);
                    self.match_types(id, ty);
                }
            }
            Node::FloatLiteral(_, spec) => match spec {
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
                    return true;
                }
            },
            Node::Assign { name, expr, .. } => {
                self.assign_type(name);
                self.assign_type(expr);
                self.match_types(id, expr);
                self.match_types(name, expr);
            }
            Node::BinOp { op, lhs, rhs } => {
                self.assign_type(lhs);
                self.assign_type(rhs);

                match op {
                    Op::Add | Op::Sub | Op::Mul | Op::Div => {
                        self.match_types(lhs, rhs);
                        self.match_types(id, lhs);
                    }
                }
            }
            Node::Call {
                mut func, params, ..
            } => {
                self.assign_type(func);

                // If func is a polymorph, copy it first
                if self.polymorph_sources.contains(&func) {
                    match self.polymorph_copy(func) {
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

                                if self.should_pass_by_ref(return_ty) {
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
            }
            Node::StructDefinition { name, params } => {
                let param_ids = self.id_vecs[params].clone();
                for &param in param_ids.borrow().iter() {
                    self.assign_type(param);
                }

                self.types.insert(
                    id,
                    Type::Struct {
                        name: Some(name),
                        params,
                    },
                );

                self.match_types(id, name);
            }
            Node::EnumDefinition { name, params } => {
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

                let mut value_ty = self.get_type(value);

                while let Type::Pointer(ty) = value_ty {
                    value_ty = self.get_type(ty);
                }

                match value_ty {
                    Type::Enum { params, .. } => {
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
            Node::AddressOf(value) => {
                self.assign_type(value);
                self.types.insert(id, Type::Pointer(value));
            }
            Node::Deref(value) => {
                self.assign_type(value);
                match self.get_type(value) {
                    Type::Pointer(ty) => {
                        self.match_types(id, ty);
                    }
                    _ => {
                        self.errors.push(CompileError::Node(
                            "Dereference on non-pointer".to_string(),
                            id,
                        ));
                    }
                }
            }
        }

        return true;
    }

    fn match_params_to_named_struct(
        &mut self,
        params: IdVec,
        name: NodeId,
        struct_literal_id: NodeId,
    ) -> Result<(), CompileError> {
        let Some(Type::Struct { params: decl, .. }) = self.types.get(&name) else { return Ok(()); };

        let given = self.id_vecs[params].clone();
        let decl = self.id_vecs[*decl].clone();

        let rearranged = match self.rearrange_params(&given.borrow(), &decl.borrow(), name) {
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
}
