use std::{
    collections::{BTreeMap, HashMap, HashSet},
    path::PathBuf,
};

use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use string_interner::{symbol::SymbolU32, StringInterner};

use cranelift_codegen::ir::{
    stackslot::StackSize, types, AbiParam, InstBuilder, MemFlags, StackSlotData, StackSlotKind,
    Type as CraneliftType, Value as CraneliftValue,
};
use cranelift_codegen::settings::{self, Configurable};
use cranelift_codegen::Context as CodegenContext;
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{default_libcall_names, FuncId, Linkage, Module};

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash, PartialOrd, Ord)]
pub struct Sym(SymbolU32);

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Hash, PartialOrd, Ord)]
pub struct NodeId(usize);

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Hash, PartialOrd, Ord)]
pub struct ScopeId(usize);

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Hash, PartialOrd, Ord)]
pub struct IdVec(usize);

#[derive(Debug, Clone, Copy)]
pub struct PushedScope(pub ScopeId, pub bool);

#[derive(Clone, Copy, PartialEq)]
pub struct Location {
    pub line: usize,
    pub col: usize,
    pub char_offset: usize,
}

impl Default for Location {
    fn default() -> Self {
        Self {
            line: 1,
            col: 1,
            char_offset: 0,
        }
    }
}

impl std::fmt::Debug for Location {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{}", self.line, self.col)
    }
}

#[derive(Clone, Copy, Default, PartialEq)]
pub struct Range {
    pub start: Location,
    pub end: Location,
    pub source_path: &'static str,
}

impl Range {
    pub fn new(start: Location, end: Location, source_path: &'static str) -> Self {
        Self {
            start,
            end,
            source_path,
        }
    }

    pub fn spanning(start: Range, end: Range, source_path: &'static str) -> Self {
        Self {
            start: start.start,
            end: end.end,
            source_path,
        }
    }
}

impl std::fmt::Debug for Range {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}-{:?}", self.start, self.end)
    }
}

#[derive(Default, Debug)]
pub struct Scope {
    pub parent: Option<ScopeId>,
    pub entries: BTreeMap<Sym, NodeId>,
}

impl Scope {
    fn new(parent: ScopeId) -> Self {
        Self {
            parent: Some(parent),
            entries: Default::default(),
        }
    }

    fn new_top() -> Self {
        Self {
            parent: None,
            entries: Default::default(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NumericSpecification {
    None,
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    F32,
    F64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Token {
    LParen,
    RParen,
    LCurly,
    RCurly,
    Semicolon,
    Colon,
    Comma,
    Underscore,
    Eq,
    Fn,
    Let,
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    F32,
    F64,
    Symbol(Sym),
    IntegerLiteral(i64, NumericSpecification),
    FloatLiteral(f64, NumericSpecification),
    Plus,
    Dash,
    Star,
    Slash,
    Return,
    Eof,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Lexeme {
    pub tok: Token,
    pub range: Range,
}

impl Lexeme {
    pub fn new(tok: Token, range: Range) -> Self {
        Self { tok, range }
    }
}

impl Default for Lexeme {
    fn default() -> Self {
        Lexeme::new(Token::Eof, Default::default())
    }
}

fn is_special(c: char) -> bool {
    c == ' '
        || c == '\t'
        || c == '\n'
        || c == '\r'
        || c == '{'
        || c == '}'
        || c == '('
        || c == ')'
        || c == '['
        || c == ']'
        || c == '<'
        || c == '>'
        || c == '+'
        || c == '-'
        || c == '*'
        || c == '/'
        || c == '-'
        || c == '.'
        || c == ':'
        || c == '\''
        || c == '"'
        || c == '`'
        || c == '!'
        || c == '|'
        || c == ','
        || c == ';'
}

#[derive(Debug)]
pub struct SourceInfo<'a> {
    pub source_path: &'static str,
    pub original_source: &'static str,
    pub byte_offset: usize,

    pub loc: Location,
    pub top: Lexeme,
    pub second: Lexeme,

    pub string_interner: &'a mut StringInterner,
}

impl<'a> SourceInfo<'a> {
    fn new(file_name: &str, string_interner: &'a mut StringInterner) -> Self {
        let source_path = PathBuf::from(file_name);
        let original_source = std::fs::read_to_string(&source_path).unwrap();
        let source_path = Box::leak(source_path.into_boxed_path().to_str().unwrap().into());
        Self {
            source_path,
            byte_offset: 0,
            original_source: Box::leak(original_source.into_boxed_str()),
            loc: Default::default(),
            top: Default::default(),
            second: Default::default(),
            string_interner,
        }
    }

    pub fn make_range(&self, start: Location, end: Location) -> Range {
        Range::new(start, end, self.source_path)
    }

    fn expect(&mut self, tok: &Token) -> Result<(), CompileError> {
        match &self.top.tok {
            t if t == tok => {
                self.pop();
                Ok(())
            }
            _ => {
                let msg = Box::leak(Box::new(
                    format!("Expected {:?}, found {:?}", tok, self.top.tok).to_string(),
                ));
                Err(CompileError::Generic(msg, self.top.range))
            }
        }
    }

    fn expect_range(&mut self, start: Location, token: Token) -> Result<Range, CompileError> {
        let range = self.make_range(start, self.top.range.end);

        if self.top.tok == token {
            self.pop();
            Ok(range)
        } else {
            let msg = Box::leak(Box::new(
                format!("Expected {:?}, found {:?}", token, self.top.tok).to_string(),
            ));
            Err(CompileError::Generic(msg, range))
        }
    }

    fn prefix(&mut self, pat: &str, tok: Token) -> bool {
        if self.source().len() >= pat.len() && self.source().starts_with(pat) {
            let start = self.loc;
            self.eat(pat.len());
            self.second = Lexeme::new(tok, self.make_range(start, self.loc));
            true
        } else {
            false
        }
    }

    fn prefix_keyword(&mut self, pat: &str, tok: Token) -> bool {
        if self.source().len() > pat.len()
            && self.source().starts_with(pat)
            && is_special(
                self.source()
                    .chars()
                    .skip(pat.len())
                    .take(1)
                    .next()
                    .unwrap(),
            )
        {
            let start = self.loc;
            self.eat(pat.len());
            self.second = Lexeme::new(tok, self.make_range(start, self.loc));
            true
        } else {
            false
        }
    }

    fn pop(&mut self) {
        self.eat_spaces();

        let start = self.loc;
        self.top = self.second;

        if self.prefix_keyword("fn", Token::Fn) {
            return;
        }
        if self.prefix_keyword("let", Token::Let) {
            return;
        }
        if self.prefix_keyword("return", Token::Return) {
            return;
        }
        if self.prefix_keyword("i8", Token::I8) {
            return;
        }
        if self.prefix_keyword("i16", Token::I16) {
            return;
        }
        if self.prefix_keyword("i32", Token::I32) {
            return;
        }
        if self.prefix_keyword("i64", Token::I64) {
            return;
        }
        if self.prefix_keyword("u8", Token::U8) {
            return;
        }
        if self.prefix_keyword("u16", Token::U16) {
            return;
        }
        if self.prefix_keyword("u32", Token::U32) {
            return;
        }
        if self.prefix_keyword("u64", Token::U64) {
            return;
        }
        if self.prefix_keyword("f32", Token::F32) {
            return;
        }
        if self.prefix_keyword("f64", Token::F64) {
            return;
        }
        if self.prefix("(", Token::LParen) {
            return;
        }
        if self.prefix(")", Token::RParen) {
            return;
        }
        if self.prefix("{", Token::LCurly) {
            return;
        }
        if self.prefix("}", Token::RCurly) {
            return;
        }
        if self.prefix(",", Token::Comma) {
            return;
        }
        if self.prefix(";", Token::Semicolon) {
            return;
        }
        if self.prefix(":", Token::Colon) {
            return;
        }
        if self.prefix("=", Token::Eq) {
            return;
        }
        if self.prefix("_", Token::Underscore) {
            return;
        }
        if self.prefix("+", Token::Plus) {
            return;
        }
        if self.prefix("-", Token::Dash) {
            return;
        }
        if self.prefix("*", Token::Star) {
            return;
        }
        if self.prefix("/", Token::Slash) {
            return;
        }

        let new_second = match self.source().chars().next() {
            Some(c) if c.is_digit(10) => {
                let index = match self.source().chars().position(|c| !c.is_digit(10)) {
                    Some(index) => index,
                    None => self.source().len(),
                };

                let has_decimal = match self.source().get(index..index + 1) {
                    Some(c) => c == ".",
                    _ => false,
                };

                let digit = self.source()[..index]
                    .parse::<i64>()
                    .expect("Failed to parse numeric literal");

                self.eat(index);

                if has_decimal {
                    self.eat(1);

                    let decimal_index = match self.source().chars().position(|c| !c.is_digit(10)) {
                        Some(index) => index,
                        None => self.source().len(),
                    };

                    let decimal_digit = self.source()[..decimal_index]
                        .parse::<i64>()
                        .expect("Failed to parse numeric literal");

                    self.eat(decimal_index);

                    let digit: f64 = format!("{}.{}", digit, decimal_digit).parse().unwrap();

                    let mut spec = NumericSpecification::None;
                    if self.source().starts_with("f32") {
                        spec = NumericSpecification::F32;
                        self.eat(3);
                    } else if self.source().starts_with("f64") {
                        spec = NumericSpecification::F64;
                        self.eat(3);
                    }

                    let end = self.loc;
                    Lexeme::new(
                        Token::FloatLiteral(digit, spec),
                        self.make_range(start, end),
                    )
                } else {
                    let mut spec = NumericSpecification::None;

                    if self.source().starts_with("i8") {
                        spec = NumericSpecification::I8;
                        self.eat(2);
                    } else if self.source().starts_with("i16") {
                        spec = NumericSpecification::I16;
                        self.eat(3);
                    } else if self.source().starts_with("i32") {
                        spec = NumericSpecification::I32;
                        self.eat(3);
                    } else if self.source().starts_with("i64") {
                        spec = NumericSpecification::I64;
                        self.eat(3);
                    } else if self.source().starts_with("u8") {
                        spec = NumericSpecification::U8;
                        self.eat(2);
                    } else if self.source().starts_with("u16") {
                        spec = NumericSpecification::U16;
                        self.eat(3);
                    } else if self.source().starts_with("u32") {
                        spec = NumericSpecification::U32;
                        self.eat(3);
                    } else if self.source().starts_with("u64") {
                        spec = NumericSpecification::U64;
                        self.eat(3);
                    } else if self.source().starts_with("f32") {
                        spec = NumericSpecification::F32;
                        self.eat(3);
                    } else if self.source().starts_with("f64") {
                        spec = NumericSpecification::F64;
                        self.eat(3);
                    }

                    let end = self.loc;
                    Lexeme::new(
                        Token::IntegerLiteral(digit, spec),
                        self.make_range(start, end),
                    )
                }
            }
            Some(_) => {
                let index = match self.source().chars().position(is_special) {
                    Some(index) => index,
                    None => self.source().len(),
                };

                if index == 0 {
                    Lexeme::new(Token::Eof, Default::default())
                } else {
                    let sym = self.string_interner.get_or_intern(&self.source()[..index]);
                    self.eat(index);

                    let end = self.loc;

                    Lexeme::new(Token::Symbol(Sym(sym)), self.make_range(start, end))
                }
            }
            None => Lexeme::new(Token::Eof, Default::default()),
        };

        self.second = new_second;
    }

    fn source(&self) -> &'static str {
        &self.original_source[self.byte_offset..]
    }

    fn eat_bytes(&mut self, bytes: usize) {
        self.byte_offset += bytes;
        if self.byte_offset > self.original_source.len() {
            self.byte_offset = self.original_source.len();
        }
    }

    fn eat(&mut self, chars: usize) {
        self.eat_bytes(chars);
        self.loc.col += chars;
        self.loc.char_offset += chars;
    }

    fn newline(&mut self) {
        self.eat_bytes(1);
        self.loc.line += 1;
        self.loc.col = 1;
        self.loc.char_offset += 1;
    }

    fn eat_rest_of_line(&mut self) {
        let chars = self
            .source()
            .chars()
            .position(|c| c == '\n')
            .unwrap_or_else(|| self.source().len());
        self.eat(chars);

        if !self.source().is_empty() {
            self.newline();
        }
    }

    fn eat_spaces(&mut self) {
        loop {
            let mut br = true;
            while let Some(' ') = self.source().chars().next() {
                br = false;
                self.eat(1);
            }
            while let Some('\r') | Some('\n') = self.source().chars().next() {
                br = false;
                self.newline();
            }
            while let Some('#') = self.source().chars().next() {
                br = false;
                self.eat_rest_of_line();
            }

            if br {
                break;
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Type {
    Unassigned,
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
    Func {
        return_ty: Option<NodeId>,
        input_tys: IdVec,
    },
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

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Op {
    Add,
    Sub,
    Mul,
    Div,
}

impl Op {
    fn precedence(self) -> u8 {
        match self {
            Op::Add | Op::Sub => 1,
            Op::Mul | Op::Div => 2,
        }
    }
}

impl From<Token> for Op {
    fn from(value: Token) -> Self {
        match value {
            Token::Plus => Op::Add,
            Token::Dash => Op::Sub,
            Token::Star => Op::Mul,
            Token::Slash => Op::Div,
            _ => unreachable!(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Node {
    Symbol(Sym),
    IntLiteral(i64, NumericSpecification),
    FloatLiteral(f64, NumericSpecification),
    TypeLiteral(Type),
    Return(Option<NodeId>),
    Let {
        name: NodeId,
        ty: Option<NodeId>,
        expr: Option<NodeId>,
    },
    Set {
        name: NodeId,
        expr: NodeId,
        is_store: bool,
    },
    Func {
        name: NodeId,
        scope: ScopeId,
        params: IdVec,
        return_ty: Option<NodeId>,
        stmts: IdVec,
        returns: IdVec,
    },
    DeclParam {
        name: NodeId,
        ty: Option<NodeId>,
        default: Option<NodeId>,
        index: u16,
    },
    BinOp {
        op: Op,
        lhs: NodeId,
        rhs: NodeId,
    },
}

impl Node {
    pub fn as_symbol(&self) -> Option<Sym> {
        match self {
            Node::Symbol(sym) => Some(*sym),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum CompileError {
    Generic(&'static str, Range),
    Node(&'static str, NodeId),
    Node2(&'static str, NodeId, NodeId),
}

#[derive(Debug)]
pub struct TypeMatch {
    changed: bool,
    unified: Type,
    ids: Vec<NodeId>,
}

#[derive(Default, Debug)]
pub struct UnificationData {
    future_matches: Vec<(NodeId, NodeId)>,
}

impl UnificationData {
    pub fn clear(&mut self) {
        self.future_matches.clear();
    }
}

#[derive(Clone, Copy, Debug)]
pub enum Value {
    Unassigned,
    None,
    FuncId(FuncId),
    Value(CraneliftValue),
}

impl Value {
    fn as_func_id(&self) -> FuncId {
        match self {
            Value::FuncId(id) => *id,
            _ => panic!("not a func id"),
        }
    }

    fn as_cranelift_value(&self) -> CraneliftValue {
        match self {
            Value::Value(val) => *val,
            _ => panic!("not a cranelift value"),
        }
    }
}

pub struct Context {
    pub nodes: Vec<Node>,
    pub ranges: Vec<Range>,
    pub id_vecs: Vec<Vec<NodeId>>,
    pub node_scopes: Vec<ScopeId>,
    pub addressable_nodes: HashSet<NodeId>,

    pub scopes: Vec<Scope>,
    pub function_scopes: Vec<ScopeId>,
    pub top_scope: ScopeId,

    pub errors: Vec<CompileError>,

    pub top_level: Vec<NodeId>,

    pub types: HashMap<NodeId, Type>,
    pub type_matches: Vec<TypeMatch>,
    pub type_array_reverse_map: HashMap<NodeId, usize>,
    pub completes: HashSet<NodeId>,
    pub circular_dependency_nodes: HashSet<NodeId>,
    pub unification_data: UnificationData,

    pub module: JITModule,
    pub values: HashMap<NodeId, Value>,
    pub func_ids: HashMap<Sym, FuncId>,
    pub func_ids_by_name: HashMap<Sym, FuncId>,
}

impl Context {
    pub fn new() -> Self {
        let module = {
            let mut flags_builder = settings::builder();
            flags_builder.set("is_pic", "false").unwrap();
            // flags_builder.set("enable_verifier", "false").unwrap();
            flags_builder.set("opt_level", "none").unwrap();
            flags_builder.set("enable_probestack", "false").unwrap();

            let isa_builder = cranelift_native::builder().unwrap_or_else(|msg| {
                panic!("host machine is not supported: {}", msg);
            });
            let isa = isa_builder
                .finish(settings::Flags::new(flags_builder))
                .unwrap();

            let mut jit_builder = JITBuilder::with_isa(isa, default_libcall_names());

            // no hot swapping for now
            jit_builder.hotswap(false);

            // jit_builder.symbol("__panic", panic_helper as *const u8);
            // jit_builder.symbol("__dbg_poke", dbg_poke as *const u8);
            // jit_builder.symbol("__dbg_repr_internal", dbg_repr_internal as *const u8);
            // jit_builder.symbol("print_i8", print_i8 as *const u8);
            // jit_builder.symbol("print_i16", print_i16 as *const u8);
            // jit_builder.symbol("print_i32", print_i32 as *const u8);
            // jit_builder.symbol("print_i64", print_i64 as *const u8);
            // jit_builder.symbol("print_u8", print_u8 as *const u8);
            // jit_builder.symbol("print_u16", print_u16 as *const u8);
            // jit_builder.symbol("print_u32", print_u32 as *const u8);
            // jit_builder.symbol("print_u64", print_u64 as *const u8);
            // jit_builder.symbol("print_f32", print_f32 as *const u8);
            // jit_builder.symbol("print_f64", print_f64 as *const u8);
            // jit_builder.symbol("print_string", print_string as *const u8);
            // jit_builder.symbol("alloc", libc::malloc as *const u8);
            // jit_builder.symbol("realloc", libc::realloc as *const u8);
            // jit_builder.symbol("debug_data", &semantic as *const _ as *const u8);

            JITModule::new(jit_builder)
        };

        Self {
            nodes: Default::default(),
            ranges: Default::default(),
            id_vecs: Default::default(),
            node_scopes: Default::default(),
            addressable_nodes: Default::default(),

            scopes: vec![Scope::new_top()],
            function_scopes: Default::default(),
            top_scope: ScopeId(0),

            errors: Default::default(),

            top_level: Default::default(),

            types: Default::default(),
            type_matches: Default::default(),
            type_array_reverse_map: Default::default(),
            circular_dependency_nodes: Default::default(),
            completes: Default::default(),
            unification_data: Default::default(),

            module,
            values: Default::default(),
            func_ids: Default::default(),
            func_ids_by_name: Default::default(),
        }
    }

    pub fn parse(
        &mut self,
        file_name: &str,
        string_interner: &mut StringInterner,
    ) -> Result<(), CompileError> {
        let mut parser = Parser::new(self, file_name, string_interner);
        parser.parse()
    }

    pub fn push_node(&mut self, range: Range, node: Node) -> NodeId {
        self.nodes.push(node);
        self.ranges.push(range);
        self.node_scopes.push(self.top_scope);

        NodeId(self.nodes.len() - 1)
    }

    pub fn push_id_vec(&mut self, vec: Vec<NodeId>) -> IdVec {
        self.id_vecs.push(vec);
        IdVec(self.id_vecs.len() - 1)
    }

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
        self.scopes[self.top_scope.0].entries.insert(sym, id);
    }

    pub fn debug_tokens(
        &mut self,
        file_name: &str,
        string_interner: &mut StringInterner,
    ) -> Result<(), CompileError> {
        let mut source_info = SourceInfo::new(file_name, string_interner);

        source_info.pop();
        source_info.pop();

        let mut a = 0;
        while source_info.top.tok != Token::Eof && a < 25 {
            println!("{:?}", source_info.top);
            source_info.pop();
            a += 1;
        }

        Ok(())
    }

    fn get_symbol(&self, sym_id: NodeId) -> Sym {
        self.nodes[sym_id.0].as_symbol().unwrap()
    }
}

impl Context {
    pub fn perform_semantic_analysis(&mut self) {
        // todo(chad): split into two types
        for node in self.top_level.clone() {
            self.assign_type(node);
            self.unify_types();
            self.circular_dependency_nodes.clear();
        }
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

    pub fn assign_type(&mut self, id: NodeId) {
        if self.completes.contains(&id) {
            return;
        }

        if self.circular_dependency_nodes.contains(&id) {
            return;
        }
        self.circular_dependency_nodes.insert(id);

        match self.nodes[id.0] {
            Node::Func {
                params,
                return_ty,
                stmts,
                returns,
                ..
            } => {
                if let Some(return_ty) = return_ty {
                    self.assign_type(return_ty);
                }

                for param in self.id_vecs[params.0].clone() {
                    self.assign_type(param);
                }

                self.types.insert(
                    id,
                    Type::Func {
                        return_ty,
                        input_tys: params,
                    },
                );

                for stmt in self.id_vecs[stmts.0].clone() {
                    self.assign_type(stmt);
                }

                for ret_id in self.id_vecs[returns.0].clone() {
                    let ret_id = match self.nodes[ret_id.0] {
                        Node::Return(Some(id)) => id,
                        Node::Return(None) => continue,
                        _ => unreachable!(),
                    };

                    if let Some(return_ty) = return_ty {
                        self.match_types(return_ty, ret_id);
                    } else {
                        self.errors
                            .push(CompileError::Node("Return type not specified", id));
                    }
                }
            }
            Node::TypeLiteral(ty) => {
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

                        for input_ty in self.id_vecs[input_tys.0].clone() {
                            self.assign_type(input_ty);
                        }
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
                    | Type::Unassigned => {}
                }

                self.types.insert(id, ty);
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
            Node::Symbol(sym) => {
                let resolved = self.scope_get(sym, id);

                match resolved {
                    Some(resolved) => {
                        self.assign_type(resolved);
                        self.match_types(id, resolved);

                        if self.addressable_nodes.contains(&resolved) {
                            self.addressable_nodes.insert(id);
                        }
                    }
                    None => {
                        self.errors.push(CompileError::Node("Symbol not found", id));
                    }
                }
            }
            Node::DeclParam {
                name: _,  // Sym
                ty,       // Id
                default,  // Id
                index: _, // u16
            } => {
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
                    self.errors
                        .push(CompileError::Node("Under-specified float literal", id));
                    return;
                }
            },
            Node::Set { name, expr, .. } => {
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
        }

        self.completes.insert(id);
    }

    fn match_types(&mut self, ty1: NodeId, ty2: NodeId) {
        self.handle_match_types(ty1, ty2);
        self.merge_type_matches(ty1, ty2); // todo(chad): can we just do this once at the end? Would it be faster?
    }

    fn handle_match_types(&mut self, ty1: NodeId, ty2: NodeId) {
        if ty1 == ty2 {
            return;
        }

        match (self.get_type(ty1), self.get_type(ty2)) {
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
                        self.errors
                            .push(CompileError::Node2("Could not match types", ty1, ty2));
                    }
                }

                let input_tys1 = self.id_vecs[input_tys1.0].clone();
                let input_tys2 = self.id_vecs[input_tys2.0].clone();

                if input_tys1.len() != input_tys2.len() {
                    self.errors.push(CompileError::Node2(
                        "Could not match types: input types differ in length",
                        ty1,
                        ty2,
                    ));
                }

                for (it1, it2) in input_tys1.iter().zip(input_tys2.iter()) {
                    self.match_types(*it1, *it2);
                }
            }
            (bt1, bt2) if bt1 == bt2 => (),
            (Type::IntLiteral, bt) if bt.is_basic() => {
                if !self.check_int_literal_type(bt) {
                    self.errors
                        .push(CompileError::Node("Expected integer literal", ty2));
                }
            }
            (bt, Type::IntLiteral) if bt.is_basic() => {
                if !self.check_int_literal_type(bt) {
                    self.errors
                        .push(CompileError::Node("Expected integer literal", ty1));
                }
            }
            (Type::FloatLiteral, bt) if bt.is_basic() => {
                if !self.check_float_literal_type(bt) {
                    self.errors
                        .push(CompileError::Node("Expected float literal", ty2));
                }
            }
            (bt, Type::FloatLiteral) if bt.is_basic() => {
                if !self.check_float_literal_type(bt) {
                    self.errors
                        .push(CompileError::Node("Expected float literal", ty1));
                }
            }
            (Type::Unassigned, _) => (),
            (_, Type::Unassigned) => (),
            (_, _) => {
                self.errors
                    .push(CompileError::Node2("Could not match types", ty1, ty2));
            }
        }
    }

    fn check_int_literal_type(&self, bt: Type) -> bool {
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

    fn check_float_literal_type(&self, bt: Type) -> bool {
        matches!(bt, Type::FloatLiteral | Type::F32 | Type::F64)
    }

    fn get_type(&self, id: NodeId) -> Type {
        return self.types.get(&id).cloned().unwrap_or(Type::Unassigned);
    }

    fn is_fully_concrete(&self, id: NodeId) -> bool {
        self.is_fully_concrete_ty(self.get_type(id))
    }

    fn is_fully_concrete_ty(&self, ty: Type) -> bool {
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

    fn find_type_array_index(&self, id: NodeId) -> Option<usize> {
        self.type_array_reverse_map.get(&id).cloned()
    }

    fn merge_type_matches(&mut self, ty1: NodeId, ty2: NodeId) {
        match (self.get_type(ty1), self.get_type(ty2)) {
            (Type::Unassigned, Type::Unassigned) => (),
            (Type::Unassigned, ty) => {
                self.types.insert(ty1, ty);
            }
            (ty, Type::Unassigned) => {
                self.types.insert(ty2, ty);
            }
            (_, _) => (),
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

    fn unify(&mut self, first: Type, second: Type, err_ids: (NodeId, NodeId)) -> Type {
        match (first, second) {
            (a, b) if a == b => a,

            // Types/Unassigneds get coerced to anything
            (Type::Unassigned, other) | (other, Type::Unassigned) => other,

            // Check int/float literals match
            (Type::IntLiteral, bt) | (bt, Type::IntLiteral) if bt.is_int() => bt,
            (Type::FloatLiteral, bt) | (bt, Type::FloatLiteral) if bt.is_float() => bt,

            // Literally doesn't matter
            (Type::Func { .. }, Type::Func { .. }) => first,

            // Anything else
            _ => {
                self.errors
                    .push(CompileError::Node2("Type mismatch", err_ids.0, err_ids.1));
                Type::Unassigned
            }
        }
    }

    pub fn unify_types(&mut self) {
        self.unification_data.clear();

        // todo(chad): @Performance
        for uid in 0..self.type_matches.len() {
            if !self.type_matches[uid].changed {
                continue;
            }
            self.type_matches[uid].changed = false;

            let most_specific_ty = self.type_matches[uid].unified;
            if matches!(most_specific_ty, Type::Unassigned) {
                continue;
            }

            // println!("setting all to {:?}", most_specific_ty);

            match most_specific_ty {
                _ => {
                    for &id in self.type_matches[uid].ids.iter() {
                        self.types.insert(id, most_specific_ty);
                    }
                }
            }
        }

        while !self.unification_data.future_matches.is_empty() {
            let (id1, id2) = self.unification_data.future_matches.pop().unwrap();
            self.match_types(id1, id2);
        }
    }
}

impl Context {
    pub fn compile_fn(
        &mut self,
        fn_name: &str,
        string_interner: &mut StringInterner,
        codegen_ctx: &mut CodegenContext,
        func_ctx: &mut FunctionBuilderContext,
    ) -> Result<(), CompileError> {
        let fn_name_interned = string_interner.get_or_intern(fn_name);

        let node_id = self
            .top_level
            .iter()
            .filter(|tl| match self.nodes[tl.0] {
                Node::Func { name, .. } => self.get_symbol(name).0 == fn_name_interned,
                _ => false,
            })
            .next()
            .cloned();

        if let Some(id) = node_id {
            self.compile_toplevel_id(id, string_interner, codegen_ctx, func_ctx)?;

            self.module
                .finalize_definitions()
                .expect("Failed to finalize definitions");

            let code = self
                .module
                .get_finalized_function(self.values[&id].as_func_id());

            let func = unsafe { std::mem::transmute::<_, fn() -> i64>(code) };
            dbg!(func());
        }

        Ok(())
    }

    pub fn compile_id(
        &mut self,
        id: NodeId,
        string_interner: &mut StringInterner,
        builder: &mut FunctionBuilder,
    ) -> Result<(), CompileError> {
        // idempotency
        match self.values.get(&id) {
            None | Some(Value::Unassigned) => {}
            _ => return Ok(()),
        };

        match self.nodes[id.0] {
            Node::Symbol(sym) => {
                let resolved = self.scope_get(sym, id);
                match resolved {
                    Some(res) => {
                        self.compile_id(res, string_interner, builder)?;
                        let value = self.values.get(&res);
                        if let Some(value) = value {
                            self.values.insert(id, *value);
                        }
                        Ok(())
                    }
                    _ => todo!(),
                }
            }
            Node::IntLiteral(n, _) => match self.types[&id] {
                Type::I64 => {
                    let value = builder.ins().iconst(types::I64, n);
                    self.values.insert(id, Value::Value(value));
                    Ok(())
                }
                _ => todo!(),
            },
            Node::FloatLiteral(_, _) => todo!(),
            Node::TypeLiteral(_) => todo!(),
            Node::Return(rv) => {
                if let Some(rv) = rv {
                    self.compile_id(rv, string_interner, builder)?;
                    let value = self.rvalue(rv, builder);
                    builder.ins().return_(&[value]);
                } else {
                    builder.ins().return_(&[]);
                }

                Ok(())
            }
            Node::Let { expr, .. } => {
                let size: u32 = self.get_type_size(id);
                let slot = builder.create_sized_stack_slot(StackSlotData {
                    kind: StackSlotKind::ExplicitSlot,
                    size,
                });

                let slot_addr = builder
                    .ins()
                    .stack_addr(self.module.isa().pointer_type(), slot, 0);
                let value = Value::Value(slot_addr);

                if let Some(expr) = expr {
                    self.compile_id(expr, string_interner, builder)?;
                    self.store(builder, expr, &value);
                }

                self.values.insert(id, value);

                Ok(())
            }
            Node::Set { .. } => todo!(),
            Node::DeclParam { .. } => todo!(),
            Node::BinOp { op, lhs, rhs } => {
                self.compile_id(lhs, string_interner, builder)?;
                self.compile_id(rhs, string_interner, builder)?;

                let lhs_value = self.rvalue(lhs, builder);
                let rhs_value = self.rvalue(rhs, builder);

                let value = match op {
                    Op::Add => builder.ins().iadd(lhs_value, rhs_value),
                    Op::Sub => builder.ins().isub(lhs_value, rhs_value),
                    Op::Mul => builder.ins().imul(lhs_value, rhs_value),
                    Op::Div => builder.ins().sdiv(lhs_value, rhs_value),
                };

                self.values.insert(id, Value::Value(value));

                Ok(())
            }
            _ => todo!(),
        }
    }

    pub fn compile_toplevel_id(
        &mut self,
        id: NodeId,
        string_interner: &mut StringInterner,
        codegen_ctx: &mut CodegenContext,
        func_ctx: &mut FunctionBuilderContext,
    ) -> Result<(), CompileError> {
        // idempotency
        match self.values.get(&id) {
            None | Some(Value::Unassigned) => {}
            _ => return Ok(()),
        };

        match self.nodes[id.0] {
            Node::Symbol(_) => todo!(),
            Node::Func {
                name,
                stmts,
                params,
                ..
            } => {
                let name_sym = self.get_symbol(name);
                let name_str = string_interner.resolve(name_sym.0).unwrap();

                let mut sig = self.module.make_signature();

                for param in self.id_vecs[params.0].clone() {
                    sig.params
                        .push(AbiParam::new(self.get_cranelift_type(param)));
                }

                sig.returns.push(AbiParam::new(types::I64));

                let decl = self
                    .module
                    .declare_function(name_str, Linkage::Export, &sig)
                    .unwrap();

                self.values.insert(id, Value::FuncId(decl));

                let mut builder = FunctionBuilder::new(&mut codegen_ctx.func, func_ctx);
                builder.func.signature = sig;

                let ebb = builder.create_block();
                builder.append_block_params_for_function_params(ebb);
                builder.switch_to_block(ebb);

                for stmt in self.id_vecs[stmts.0].clone() {
                    self.compile_id(stmt, string_interner, &mut builder)?;
                }

                builder.seal_all_blocks();
                builder.finalize();

                println!("{}", codegen_ctx.func.display());

                self.module.define_function(decl, codegen_ctx).unwrap();

                self.module.clear_context(codegen_ctx);

                Ok(())
            }
            _ => todo!(),
        }
    }

    fn store(&mut self, builder: &mut FunctionBuilder, id: NodeId, dest: &Value) {
        if self.addressable_nodes.contains(&id) {
            self.store_copy(builder, id, dest);
        } else {
            self.store_value(builder, id, dest, None);
        }
    }

    fn store_copy(&mut self, builder: &mut FunctionBuilder, id: NodeId, dest: &Value) {
        let size = self.get_type_size(id);

        let source_value = self.values[&id].as_cranelift_value();
        let dest_value = dest.as_cranelift_value();

        builder.emit_small_memory_copy(
            self.module.isa().frontend_config(),
            dest_value,
            source_value,
            size as _,
            1,
            1,
            true, // non-overlapping
            MemFlags::new(),
        );
    }

    fn store_value(
        &mut self,
        builder: &mut FunctionBuilder,
        id: NodeId,
        dest: &Value,
        offset: Option<i32>,
    ) {
        let source_value = match self.values[&id] {
            Value::Value(value) => value,
            _ => todo!("store_value source for {:?}", id),
        };

        match dest {
            Value::Value(value) => {
                builder.ins().store(
                    MemFlags::new(),
                    source_value,
                    *value,
                    offset.unwrap_or_default(),
                );
            }
            _ => todo!("store_value dest for {:?}", dest),
        }
    }

    fn rvalue(&self, id: NodeId, builder: &mut FunctionBuilder) -> CraneliftValue {
        let value = self.values[&id].as_cranelift_value();

        if self.addressable_nodes.contains(&id) {
            let ty = self.get_cranelift_type(id);
            builder.ins().load(ty, MemFlags::new(), value, 0)
        } else {
            value
        }
    }

    fn get_cranelift_type(&self, param: NodeId) -> CraneliftType {
        match self.types[&param] {
            Type::I8 => types::I8,
            Type::I16 => types::I16,
            Type::I32 => types::I32,
            Type::I64 => types::I64,
            Type::F32 => types::F32,
            Type::F64 => types::F64,
            Type::Func { .. } => types::I64,
            Type::Unassigned => todo!(),
            _ => todo!(),
        }
    }

    fn get_type_size(&self, param: NodeId) -> StackSize {
        match self.types[&param] {
            Type::I8 => 1,
            Type::I16 => 2,
            Type::I32 => 4,
            Type::I64 => 8,
            Type::F32 => 4,
            Type::F64 | Type::Func { .. } => 8,
            Type::Unassigned => todo!(),
            _ => todo!(),
        }
    }
}

struct Parser<'a> {
    source_info: SourceInfo<'a>,
    ctx: &'a mut Context,

    // stack of returns - pushed when entering parsing a function, popped when exiting
    returns: Vec<Vec<NodeId>>,
}

impl<'a> Parser<'a> {
    pub fn new(
        context: &'a mut Context,
        file_name: &str,
        string_interner: &'a mut StringInterner,
    ) -> Self {
        Self {
            source_info: SourceInfo::new(file_name, string_interner),
            ctx: context,
            returns: Default::default(),
        }
    }

    pub fn parse(&mut self) -> Result<(), CompileError> {
        self.source_info.pop();
        self.source_info.pop();

        while self.source_info.top.tok != Token::Eof {
            let tl = self.parse_top_level()?;
            self.ctx.top_level.push(tl);
        }

        Ok(())
    }

    pub fn parse_top_level(&mut self) -> Result<NodeId, CompileError> {
        let tl = match self.source_info.top.tok {
            Token::Fn => Ok(self.parse_fn()?),
            _ => {
                let msg = Box::leak(Box::new(format!(
                    "expected 'fn', found '{:?}'",
                    self.source_info.top.tok
                )));

                Err(CompileError::Generic(msg, self.source_info.top.range))
            }
        }?;

        Ok(tl)
    }

    fn parse_symbol(&mut self) -> Result<NodeId, CompileError> {
        let range = self.source_info.top.range;
        match self.source_info.top.tok {
            Token::Symbol(sym) => {
                self.source_info.pop();
                Ok(self.ctx.push_node(range, Node::Symbol(sym)))
            }
            _ => Err(CompileError::Generic("expected symbol", range)),
        }
    }

    fn parse_decl_params(&mut self) -> Result<IdVec, CompileError> {
        let mut params = Vec::new();

        let mut index = 0;
        while self.source_info.top.tok != Token::RParen && self.source_info.top.tok != Token::RCurly
        {
            let input_start = self.source_info.top.range.start;

            let name = self.parse_symbol()?;
            let name_sym = self.ctx.nodes[name.0].as_symbol().unwrap();

            let (ty, default) = if self.source_info.top.tok == Token::Colon {
                self.source_info.pop(); // `:`
                let ty = self.parse_type()?;

                let mut default = None;
                if self.source_info.top.tok == Token::Eq {
                    self.source_info.pop(); // `=`
                    default = Some(self.parse_expression()?);
                }

                (Some(ty), default)
            } else if self.source_info.top.tok == Token::Eq {
                todo!("parse default parameters")
            } else {
                todo!()
            };

            let range_end = match (ty, default) {
                (_, Some(default)) => self.ctx.ranges[default.0].end,
                (Some(ty), _) => self.ctx.ranges[ty.0].end,
                _ => self.ctx.ranges[name.0].end,
            };

            let range = self.source_info.make_range(input_start, range_end);

            // put the param in scope
            let param = self.ctx.push_node(
                range,
                Node::DeclParam {
                    name,
                    ty,
                    default,
                    index,
                },
            );
            self.ctx.scope_insert(name_sym, param);
            params.push(param);

            if self.source_info.top.tok == Token::Comma {
                self.source_info.pop(); // `,`
            }

            index += 1;
        }

        Ok(self.ctx.push_id_vec(params))
    }

    pub fn parse_fn(&mut self) -> Result<NodeId, CompileError> {
        let start = self.source_info.top.range.start;

        self.source_info.pop();

        let name = self.parse_symbol()?;
        let name_sym = self.ctx.nodes[name.0].as_symbol().unwrap();

        // open a new scope
        let pushed_scope = self.ctx.push_scope(true);

        self.source_info.expect(&Token::LParen)?;
        let params = self.parse_decl_params()?;
        self.source_info.expect(&Token::RParen)?;

        let return_ty = if self.source_info.top.tok != Token::LCurly {
            Some(self.parse_type()?)
        } else {
            None
        };

        self.source_info.expect(&Token::LCurly)?;

        self.returns.push(Vec::new());

        let mut stmts = Vec::new();
        while self.source_info.top.tok != Token::RCurly {
            let stmt = self.parse_fn_stmt()?;
            stmts.push(stmt);
        }
        let stmts = self.ctx.push_id_vec(stmts);

        let range = self.source_info.expect_range(start, Token::RCurly)?;

        let returns = self.returns.pop().unwrap();
        let returns = self.ctx.push_id_vec(returns);
        let func = self.ctx.push_node(
            range,
            Node::Func {
                name,
                scope: self.ctx.top_scope,
                params,
                return_ty,
                stmts,
                returns,
            },
        );

        // pop the top scope
        self.ctx.pop_scope(pushed_scope);

        self.ctx.scope_insert(name_sym, func);
        // self.top_level_map.insert(name, func);

        // self.funcs.push(func);

        Ok(func)
    }

    pub fn parse_expression(&mut self) -> Result<NodeId, CompileError> {
        let mut operators = Vec::<Op>::new();
        let mut output = Vec::new();

        let (mut parsing_op, mut parsing_expr) = (false, true);

        loop {
            let _debug_tok = self.source_info.top.tok;

            match self.source_info.top.tok {
                Token::IntegerLiteral(_, _)
                | Token::FloatLiteral(_, _)
                | Token::LCurly
                | Token::LParen
                | Token::Symbol(_)
                | Token::Fn
                | Token::I8
                | Token::I16
                | Token::I32
                | Token::I64
                | Token::U8
                | Token::U16
                | Token::U32
                | Token::U64
                | Token::F32
                | Token::F64 => {
                    if !parsing_expr {
                        break;
                    }

                    let id = self.parse_expression_piece()?;
                    output.push(Shunting::Id(id))
                }
                Token::Plus => {
                    if !parsing_op {
                        break;
                    }

                    while !operators.is_empty()
                        && operators.last().unwrap().precedence()
                            >= Op::from(self.source_info.top.tok).precedence()
                    {
                        output.push(Shunting::Op(operators.pop().unwrap()));
                    }
                    operators.push(Op::Add);

                    self.source_info.pop(); // `+`
                }
                Token::Dash => {
                    if !parsing_op {
                        break;
                    }

                    self.source_info.pop(); // `-`
                    output.push(Shunting::Op(Op::Sub));
                }
                Token::Star => {
                    if !parsing_op {
                        break;
                    }

                    self.source_info.pop(); // `*`
                    output.push(Shunting::Op(Op::Mul));
                }
                Token::Slash => {
                    if !parsing_op {
                        break;
                    }

                    self.source_info.pop(); // `/`
                    output.push(Shunting::Op(Op::Div));
                }
                _ => break,
            }

            std::mem::swap(&mut parsing_op, &mut parsing_expr);
        }

        while !operators.is_empty() {
            output.push(Shunting::Op(operators.pop().unwrap()));
        }

        if output.len() == 1 {
            return Ok(match output[0] {
                Shunting::Id(id) => id,
                _ => unreachable!(),
            });
        }

        let output_id = output[0].as_id().unwrap();
        self.shunting_unroll(&mut output, output_id)
    }

    pub fn parse_expression_piece(&mut self) -> Result<NodeId, CompileError> {
        let value = match self.source_info.top.tok {
            Token::IntegerLiteral(_, _) | Token::FloatLiteral(_, _) => self.parse_numeric_literal(),
            Token::Fn => self.parse_fn(),
            _ => self.parse_lvalue(),
        }?;

        Ok(value)
    }

    fn parse_lvalue(&mut self) -> Result<NodeId, CompileError> {
        if self.source_info.top.tok == Token::LParen {
            self.source_info.pop(); // `(`
            let expr = self.parse_expression()?;
            self.source_info.expect(&Token::RParen)?;

            Ok(expr)
        } else if self.source_info.top.tok == Token::I8 {
            let range = self.source_info.top.range;
            self.source_info.pop();
            Ok(self.ctx.push_node(range, Node::TypeLiteral(Type::I8)))
        } else if self.source_info.top.tok == Token::I16 {
            let range = self.source_info.top.range;
            self.source_info.pop();
            Ok(self.ctx.push_node(range, Node::TypeLiteral(Type::I16)))
        } else if self.source_info.top.tok == Token::I32 {
            let range = self.source_info.top.range;
            self.source_info.pop();
            Ok(self.ctx.push_node(range, Node::TypeLiteral(Type::I32)))
        } else if self.source_info.top.tok == Token::I64 {
            let range = self.source_info.top.range;
            self.source_info.pop();
            Ok(self.ctx.push_node(range, Node::TypeLiteral(Type::I64)))
        } else if self.source_info.top.tok == Token::U8 {
            let range = self.source_info.top.range;
            self.source_info.pop();
            Ok(self.ctx.push_node(range, Node::TypeLiteral(Type::U8)))
        } else if self.source_info.top.tok == Token::U16 {
            let range = self.source_info.top.range;
            self.source_info.pop();
            Ok(self.ctx.push_node(range, Node::TypeLiteral(Type::U16)))
        } else if self.source_info.top.tok == Token::U32 {
            let range = self.source_info.top.range;
            self.source_info.pop();
            Ok(self.ctx.push_node(range, Node::TypeLiteral(Type::U32)))
        } else if self.source_info.top.tok == Token::U64 {
            let range = self.source_info.top.range;
            self.source_info.pop();
            Ok(self.ctx.push_node(range, Node::TypeLiteral(Type::U64)))
        } else if self.source_info.top.tok == Token::F32 {
            let range = self.source_info.top.range;
            self.source_info.pop();
            Ok(self.ctx.push_node(range, Node::TypeLiteral(Type::F32)))
        } else if self.source_info.top.tok == Token::F64 {
            let range = self.source_info.top.range;
            self.source_info.pop();
            Ok(self.ctx.push_node(range, Node::TypeLiteral(Type::F64)))
        } else if let Token::Symbol(_) = self.source_info.top.tok {
            Ok(self.parse_symbol()?)
        } else {
            Err(CompileError::Generic(
                "Could not parse lvalue",
                self.source_info.top.range,
            ))
        }
    }

    fn parse_numeric_literal(&mut self) -> Result<NodeId, CompileError> {
        match self.source_info.top.tok {
            Token::IntegerLiteral(i, s) => {
                let range = self.source_info.top.range;
                self.source_info.pop();
                Ok(self.ctx.push_node(range, Node::IntLiteral(i, s)))
            }
            Token::FloatLiteral(f, s) => {
                let range = self.source_info.top.range;
                self.source_info.pop();
                Ok(self.ctx.push_node(range, Node::FloatLiteral(f, s)))
            }
            _ => Err(CompileError::Generic(
                "Expected numeric literal",
                self.source_info.top.range,
            )),
        }
    }

    pub fn parse_let(&mut self, start: Location) -> Result<NodeId, CompileError> {
        self.source_info.pop(); // `let`
        let name = self.parse_symbol()?;
        let name_sym = self.ctx.nodes[name.0].as_symbol().unwrap();

        let ty = if self.source_info.top.tok == Token::Colon {
            self.source_info.pop(); // `:`
            Some(self.parse_type()?)
        } else {
            None
        };

        self.source_info.expect(&Token::Eq)?;

        let expr = match self.source_info.top.tok {
            Token::Semicolon => None,
            _ => Some(self.parse_expression()?),
        };
        let range = self.source_info.expect_range(start, Token::Semicolon)?;
        let let_id = self.ctx.push_node(range, Node::Let { name, ty, expr });

        self.ctx.addressable_nodes.insert(let_id);
        self.ctx.scope_insert(name_sym, let_id);

        Ok(let_id)
    }

    pub fn parse_fn_stmt(&mut self) -> Result<NodeId, CompileError> {
        let start = self.source_info.top.range.start;

        let r = match self.source_info.top.tok {
            Token::Return => {
                self.source_info.pop(); // `return`

                let expr = if self.source_info.top.tok != Token::Semicolon {
                    Some(self.parse_expression()?)
                } else {
                    None
                };

                let range = self.source_info.expect_range(start, Token::Semicolon)?;

                let ret_id = self.ctx.push_node(range, Node::Return(expr));
                self.returns.last_mut().unwrap().push(ret_id);
                Ok(ret_id)
            }
            Token::Let => self.parse_let(start),
            _ => {
                let lvalue = self.parse_expression()?;

                match self.source_info.top.tok {
                    // Assignment?
                    Token::Eq => {
                        // parsing something like "foo = expr;";
                        self.source_info.expect(&Token::Eq)?;
                        let expr = self.parse_expression()?;
                        let range = self.source_info.expect_range(start, Token::Semicolon)?;

                        Ok(self.ctx.push_node(
                            range,
                            Node::Set {
                                name: lvalue,
                                expr,
                                is_store: false,
                            },
                        ))
                    }
                    _ => {
                        self.ctx.ranges[lvalue.0] =
                            self.source_info.expect_range(start, Token::Semicolon)?;
                        Ok(lvalue)
                    }
                }
            }
        };

        r
    }

    fn parse_type(&mut self) -> Result<NodeId, CompileError> {
        match self.source_info.top.tok {
            Token::I8 => {
                let range = self.source_info.top.range;
                self.source_info.pop();
                Ok(self.ctx.push_node(range, Node::TypeLiteral(Type::I8)))
            }
            Token::I16 => {
                let range = self.source_info.top.range;
                self.source_info.pop();
                Ok(self.ctx.push_node(range, Node::TypeLiteral(Type::I16)))
            }
            Token::I32 => {
                let range = self.source_info.top.range;
                self.source_info.pop();
                Ok(self.ctx.push_node(range, Node::TypeLiteral(Type::I32)))
            }
            Token::I64 => {
                let range = self.source_info.top.range;
                self.source_info.pop();
                Ok(self.ctx.push_node(range, Node::TypeLiteral(Type::I64)))
            }
            Token::U8 => {
                let range = self.source_info.top.range;
                self.source_info.pop();
                Ok(self.ctx.push_node(range, Node::TypeLiteral(Type::U8)))
            }
            Token::U16 => {
                let range = self.source_info.top.range;
                self.source_info.pop();
                Ok(self.ctx.push_node(range, Node::TypeLiteral(Type::U16)))
            }
            Token::U32 => {
                let range = self.source_info.top.range;
                self.source_info.pop();
                Ok(self.ctx.push_node(range, Node::TypeLiteral(Type::U32)))
            }
            Token::U64 => {
                let range = self.source_info.top.range;
                self.source_info.pop();
                Ok(self.ctx.push_node(range, Node::TypeLiteral(Type::U64)))
            }
            Token::F32 => {
                let range = self.source_info.top.range;
                self.source_info.pop();
                Ok(self.ctx.push_node(range, Node::TypeLiteral(Type::F32)))
            }
            Token::F64 => {
                let range = self.source_info.top.range;
                self.source_info.pop();
                Ok(self.ctx.push_node(range, Node::TypeLiteral(Type::F64)))
            }
            Token::Underscore => {
                let range = self.source_info.top.range;
                self.source_info.pop();
                Ok(self
                    .ctx
                    .push_node(range, Node::TypeLiteral(Type::Unassigned)))
            }
            _ => Err(CompileError::Generic(
                "Expected type",
                self.source_info.top.range,
            )),
        }
    }

    fn shunting_unroll(
        &mut self,
        output: &mut Vec<Shunting>,
        err_node_id: NodeId,
    ) -> Result<NodeId, CompileError> {
        if output.is_empty() {
            return Err(CompileError::Node("Unfinished expression", err_node_id));
        }

        match output.last().unwrap().clone() {
            Shunting::Id(id) => {
                let id = id;
                output.pop();
                Ok(id)
            }
            Shunting::Op(op) => {
                output.pop();

                let rhs = self.shunting_unroll(output, err_node_id)?;
                let lhs = self.shunting_unroll(output, rhs)?;

                let range = self.make_range_spanning(lhs, rhs);

                let value = self.ctx.push_node(range, Node::BinOp { op, lhs, rhs });
                // self.local_insert(value);
                Ok(value)
            }
        }
    }

    pub fn make_range_spanning(&self, start: NodeId, end: NodeId) -> Range {
        let start = self.ctx.ranges[start.0].start;
        let end = self.ctx.ranges[end.0].end;

        if start.char_offset > end.char_offset {
            Range::new(end, start, self.source_info.source_path)
        } else {
            Range::new(start, end, self.source_info.source_path)
        }
    }
}

#[derive(Copy, Clone, Debug)]
enum Shunting {
    Op(Op),
    Id(NodeId),
}

impl Shunting {
    fn as_id(&self) -> Option<NodeId> {
        match self {
            Shunting::Id(id) => Some(*id),
            _ => None,
        }
    }
}

fn main() -> Result<(), CompileError> {
    let mut string_interner = StringInterner::new();

    let mut context = Context::new();

    // context.debug_tokens("foo.sm", &mut string_interner)?;

    context.parse("foo.sm", &mut string_interner)?;

    context.perform_semantic_analysis();

    if !context.errors.is_empty() {
        dbg!(&context.errors);
        return Ok(());
    }

    let mut codegen_ctx = context.module.make_context();
    let mut func_ctx = FunctionBuilderContext::new();
    context.compile_fn(
        "main",
        &mut string_interner,
        &mut codegen_ctx,
        &mut func_ctx,
    )?;

    Ok(())
}
