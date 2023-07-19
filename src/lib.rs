use std::{
    cell::RefCell,
    collections::{BTreeMap, HashMap, HashSet},
    path::PathBuf,
    rc::Rc,
};

use clap::Parser as ClapParser;

use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use string_interner::{symbol::SymbolU32, StringInterner};

use cranelift_codegen::ir::{
    stackslot::StackSize, types, AbiParam, Block, InstBuilder, MemFlags, Signature, StackSlotData,
    StackSlotKind, Type as CraneliftType, Value as CraneliftValue,
};
use cranelift_codegen::settings::{self, Configurable};
use cranelift_codegen::Context as CodegenContext;
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{default_libcall_names, FuncId, Linkage, Module};

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash, PartialOrd, Ord)]
pub struct Sym(pub SymbolU32);

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Hash, PartialOrd, Ord)]
pub struct NodeId(pub usize);

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Hash, PartialOrd, Ord)]
pub struct ScopeId(pub usize);

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Hash, PartialOrd, Ord)]
pub struct IdVec(pub usize);

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

    pub fn spanning(start: Range, end: Range) -> Self {
        assert!(start.source_path == end.source_path);

        let (start, end) = if start.start.line < end.start.line {
            (start, end)
        } else if start.start.line == end.start.line {
            if start.start.col < end.start.col {
                (start, end)
            } else {
                (end, start)
            }
        } else {
            (end, start)
        };

        Self {
            start: start.start,
            end: end.end,
            source_path: start.source_path,
        }
    }

    pub fn contains(&self, line: usize, col: usize) -> bool {
        if self.start.line < line && self.end.line > line {
            true
        } else if self.start.line == line && self.end.line == line {
            self.start.col <= col && self.end.col >= col
        } else if self.start.line == line {
            self.start.col <= col
        } else if self.end.line == line {
            self.end.col >= col
        } else {
            false
        }
    }

    pub fn char_span(&self) -> usize {
        self.end.char_offset - self.start.char_offset
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
    Dot,
    Underscore,
    UnderscoreSymbol(Sym),
    Eq,
    Fn,
    Extern,
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
    Struct,
    AddressOf,
    Bang,
    BangSymbol(Sym),
    UnderscoreLCurly,
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

fn breaks_symbol(c: char) -> bool {
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

pub trait Source: std::fmt::Debug + Sized {
    fn char_count(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.char_count() == 0
    }

    fn pop_chars(&mut self, chars: usize);

    fn next_char(&self) -> Option<char>;

    fn char_at(&self, index: usize) -> Option<char>;

    fn slice(&self, r: std::ops::Range<usize>) -> &str;

    fn starts_with(&self, pat: &str) -> bool;

    fn position_of<P: FnMut(char) -> bool>(&self, pred: P) -> Option<usize>;

    fn reset(&mut self);
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StrSource<'a> {
    pub original_source: &'a str,
    pub source: &'a str,
}

impl<'a> StrSource<'a> {
    pub fn from_str(source: &'a str) -> Self {
        Self {
            original_source: source,
            source,
        }
    }
}

impl<'a> Source for StrSource<'a> {
    fn char_count(&self) -> usize {
        self.source.chars().count()
    }

    fn pop_chars(&mut self, chars: usize) {
        self.source = &self.source[chars..];
    }

    fn next_char(&self) -> Option<char> {
        self.source.chars().next()
    }

    fn starts_with(&self, pat: &str) -> bool {
        self.source.starts_with(pat)
    }

    fn slice(&self, r: std::ops::Range<usize>) -> &'a str {
        &self.source[r]
    }

    fn char_at(&self, index: usize) -> Option<char> {
        self.source.chars().skip(index).take(1).next()
    }

    fn position_of<P: FnMut(char) -> bool>(&self, pred: P) -> Option<usize> {
        self.source.chars().position(pred)
    }

    fn reset(&mut self) {
        self.source = self.original_source;
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RopeySource {
    pub rope: ropey::Rope,
    pub char_index: usize,
    pub source_len: usize,
}

impl RopeySource {
    pub fn from_str(source: &str) -> Self {
        RopeySource {
            rope: source.into(),
            char_index: 0,
            source_len: source.chars().count(),
        }
    }
}

impl Source for RopeySource {
    fn char_count(&self) -> usize {
        self.source_len - self.char_index
    }

    fn pop_chars(&mut self, chars: usize) {
        self.char_index += chars;
    }

    fn next_char(&self) -> Option<char> {
        self.rope.get_char(self.char_index)
    }

    fn starts_with(&self, pat: &str) -> bool {
        self.rope
            .byte_slice(self.char_index..self.char_index + pat.len())
            .as_str()
            .unwrap()
            .starts_with(pat)
    }

    fn slice(&self, r: std::ops::Range<usize>) -> &str {
        self.rope
            .slice(r.start + self.char_index..r.end + self.char_index)
            .as_str()
            .unwrap()
    }

    fn char_at(&self, index: usize) -> Option<char> {
        self.rope.get_char(self.char_index + index)
    }

    fn position_of<P: FnMut(char) -> bool>(&self, pred: P) -> Option<usize> {
        self.rope.slice(self.char_index..).chars().position(pred)
    }

    fn reset(&mut self) {
        self.char_index = 0;
        self.source_len = self.rope.len_chars();
    }
}

#[derive(Debug)]
pub struct SourceInfo<W: Source> {
    pub source_path: &'static str,
    pub source: W,
    pub chars_left: usize,

    pub loc: Location,
    pub top: Lexeme,
    pub second: Lexeme,
}

impl<W: Source> SourceInfo<W> {
    pub fn from_file(file_name: &str) -> SourceInfo<StrSource> {
        let source_path = PathBuf::from(file_name);
        let source = std::fs::read_to_string(&source_path).unwrap();
        let source: &'static str = Box::leak(source.into_boxed_str());
        let source = StrSource::from_str(source);
        let chars_left = source.char_count();
        let source_path = Box::leak(source_path.into_boxed_path().to_str().unwrap().into());

        SourceInfo {
            source_path,
            source,
            chars_left,
            loc: Default::default(),
            top: Default::default(),
            second: Default::default(),
        }
    }

    pub fn ropey_from_file(file_name: &str) -> SourceInfo<RopeySource> {
        let source_path = PathBuf::from(file_name);
        let source = std::fs::read_to_string(&source_path).unwrap();
        let source: &'static str = Box::leak(source.into_boxed_str());
        let source = RopeySource::from_str(source);
        let chars_left = source.char_count();
        let source_path = Box::leak(source_path.into_boxed_path().to_str().unwrap().into());

        SourceInfo {
            source_path,
            source,
            chars_left,
            loc: Default::default(),
            top: Default::default(),
            second: Default::default(),
        }
    }

    pub fn from_str(source: &str) -> SourceInfo<StrSource> {
        let source = StrSource::from_str(source);
        let chars_left = source.char_count();

        SourceInfo {
            source_path: "<none>",
            source,
            chars_left,
            loc: Default::default(),
            top: Default::default(),
            second: Default::default(),
        }
    }

    pub fn from_source(source: W) -> Self {
        let chars_left = source.char_count();

        Self {
            source_path: "<none>",
            source,
            chars_left,
            loc: Default::default(),
            top: Default::default(),
            second: Default::default(),
        }
    }

    fn from_range(range: Range) -> SourceInfo<StrSource<'static>> {
        let source_path = PathBuf::from(range.source_path);
        let source = std::fs::read_to_string(&source_path).unwrap();
        let source = source[range.start.char_offset..range.end.char_offset].to_string();
        let source: &'static str = Box::leak(source.into_boxed_str());
        let source = StrSource::from_str(source);
        let chars_left = source.char_count();
        let source_path = Box::leak(source_path.into_boxed_path().to_str().unwrap().into());

        SourceInfo {
            source_path,
            source,
            chars_left,
            loc: range.start,
            top: Default::default(),
            second: Default::default(),
        }
    }

    pub fn reset(&mut self) {
        self.source.reset();
        self.chars_left = self.source.char_count();
        self.loc = Default::default();
        self.top = Default::default();
        self.second = Default::default();
    }

    pub fn make_range(&self, start: Location, end: Location) -> Range {
        Range::new(start, end, self.source_path)
    }

    fn prefix(&mut self, pat: &str, tok: Token) -> bool {
        if self.source.char_count() >= pat.len() && self.source.starts_with(pat) {
            let start = self.loc;
            self.eat(pat.len());
            self.second = Lexeme::new(tok, self.make_range(start, self.loc));
            true
        } else {
            false
        }
    }

    fn prefix_keyword(&mut self, pat: &str, tok: Token) -> bool {
        if self.source.char_count() > pat.len()
            && self.source.starts_with(pat)
            && breaks_symbol(self.source.char_at(pat.len()).unwrap())
        {
            let start = self.loc;
            self.eat(pat.len());
            self.second = Lexeme::new(tok, self.make_range(start, self.loc));
            true
        } else {
            false
        }
    }

    fn eat_chars(&mut self, chars: usize) {
        let chars = chars.min(self.chars_left);
        self.source.pop_chars(chars);
        self.chars_left -= chars;
    }

    fn eat(&mut self, chars: usize) {
        self.eat_chars(chars);
        self.loc.col += chars;
        self.loc.char_offset += chars;
    }

    fn newline(&mut self) {
        self.eat_chars(1);
        self.loc.line += 1;
        self.loc.col = 1;
        self.loc.char_offset += 1;
    }

    fn eat_rest_of_line(&mut self) {
        let chars = self
            .source
            .position_of(|c| c == '\n')
            .unwrap_or_else(|| self.source.char_count());
        self.eat(chars);

        if !self.source.is_empty() {
            self.newline();
        }
    }

    fn eat_spaces(&mut self) {
        loop {
            let mut br = true;
            while let Some(' ') = self.source.next_char() {
                br = false;
                self.eat(1);
            }
            while let Some('\r') | Some('\n') = self.source.next_char() {
                br = false;
                self.newline();
            }
            while let Some('#') = self.source.next_char() {
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
    Func {
        return_ty: Option<NodeId>,
        input_tys: IdVec,
    },
    Struct {
        name: Option<NodeId>,
        fields: IdVec,
    },
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
    Type(Type),
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
        name: Option<NodeId>,
        scope: ScopeId,
        params: IdVec,
        return_ty: Option<NodeId>,
        stmts: IdVec,
        returns: IdVec,
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
        rearranged_params: Option<IdVec>,
    },
    StructDefinition {
        name: NodeId,
        fields: IdVec,
    },
    StructLiteral {
        name: Option<NodeId>,
        fields: IdVec,
        rearranged_fields: Option<IdVec>,
    },
    MemberAccess {
        value: NodeId,
        member: NodeId,
    },
    AddressOf(NodeId),
    Deref(NodeId),
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
            Node::Type(_) => "Type".to_string(),
            Node::Return(_) => "Return".to_string(),
            Node::Let { .. } => "Let".to_string(),
            Node::Set { .. } => "Set".to_string(),
            Node::Func { .. } => "Func".to_string(),
            Node::Extern { .. } => "Extern".to_string(),
            Node::StructDeclParam { .. } => "StructDeclParam".to_string(),
            Node::FuncDeclParam { .. } => "FuncDeclParam".to_string(),
            Node::ValueParam { .. } => "ValueParam".to_string(),
            Node::BinOp { .. } => "BinOp".to_string(),
            Node::Call { .. } => "Call".to_string(),
            Node::StructDefinition { .. } => "StructDefinition".to_string(),
            Node::StructLiteral { .. } => "StructLiteral".to_string(),
            Node::MemberAccess { .. } => "MemberAccess".to_string(),
            Node::AddressOf(_) => "AddressOf".to_string(),
            Node::Deref(_) => "Deref".to_string(),
        }
    }
}

// todo(chad): replace this with something that implements Copy
#[derive(Clone, Debug)]
pub enum CompileError {
    Generic(String, Range),
    Node(String, NodeId),
    Node2(String, NodeId, NodeId),
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
    pub fn reset(&mut self) {
        self.future_matches.clear();
    }
}

#[derive(Clone, Copy, Debug)]
pub enum Value {
    Unassigned,
    None,
    Func(FuncId),
    Value(CraneliftValue),
}

#[derive(ClapParser, Debug, Default)]
#[command(version, about)]
pub struct Args {
    #[arg(long)]
    pub dump_ir: bool,

    #[arg(long)]
    pub dump_tokens: bool,

    #[arg(long)]
    pub print_type_matches: bool,
}

pub struct Context {
    pub args: Args,

    pub string_interner: StringInterner,

    pub nodes: Vec<Node>,
    pub ranges: Vec<Range>,
    pub id_vecs: Vec<Rc<RefCell<Vec<NodeId>>>>,
    pub node_scopes: Vec<ScopeId>,
    pub addressable_nodes: HashSet<NodeId>,
    pub polymorph_target: bool,
    pub polymorph_sources: HashSet<NodeId>,
    pub polymorph_copies: HashSet<NodeId>,

    pub scopes: Vec<Scope>,
    pub function_scopes: Vec<ScopeId>,
    pub top_scope: ScopeId,

    pub errors: Vec<CompileError>,

    pub top_level: Vec<NodeId>,

    pub types: HashMap<NodeId, Type>,
    pub type_matches: Vec<TypeMatch>,
    pub type_array_reverse_map: HashMap<NodeId, usize>,
    pub completes: HashSet<NodeId>,
    pub topo: Vec<NodeId>,
    pub circular_dependency_nodes: HashSet<NodeId>,
    pub unification_data: UnificationData,
    pub deferreds: Vec<NodeId>,

    pub module: JITModule,
    pub values: HashMap<NodeId, Value>,
}

unsafe impl Send for Context {}

impl Context {
    pub fn make_module() -> JITModule {
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
        jit_builder.symbol("print_i64", print_i64 as *const u8);
        // jit_builder.symbol("print_u8", print_u8 as *const u8);
        // jit_builder.symbol("print_u16", print_u16 as *const u8);
        // jit_builder.symbol("print_u32", print_u32 as *const u8);
        // jit_builder.symbol("print_u64", print_u64 as *const u8);
        // jit_builder.symbol("print_f32", print_f32 as *const u8);
        jit_builder.symbol("print_f64", print_f64 as *const u8);
        // jit_builder.symbol("print_string", print_string as *const u8);
        // jit_builder.symbol("alloc", libc::malloc as *const u8);
        // jit_builder.symbol("realloc", libc::realloc as *const u8);
        // jit_builder.symbol("debug_data", &semantic as *const _ as *const u8);

        JITModule::new(jit_builder)
    }

    pub fn new(args: Args) -> Self {
        Self {
            args,
            string_interner: StringInterner::new(),

            nodes: Default::default(),
            ranges: Default::default(),
            id_vecs: Default::default(),
            node_scopes: Default::default(),
            addressable_nodes: Default::default(),
            polymorph_target: false,
            polymorph_sources: Default::default(),
            polymorph_copies: Default::default(),

            scopes: vec![Scope::new_top()],
            function_scopes: Default::default(),
            top_scope: ScopeId(0),

            errors: Default::default(),

            top_level: Default::default(),

            types: Default::default(),
            type_matches: Default::default(),
            type_array_reverse_map: Default::default(),
            completes: Default::default(),
            topo: Default::default(),
            circular_dependency_nodes: Default::default(),
            unification_data: Default::default(),
            deferreds: Default::default(),

            module: Self::make_module(),
            values: Default::default(),
        }
    }

    pub fn reset(&mut self) {
        self.string_interner = StringInterner::new();

        self.nodes.clear();
        self.ranges.clear();
        self.id_vecs.clear();
        self.node_scopes.clear();
        self.addressable_nodes.clear();
        self.polymorph_target = false;
        self.polymorph_sources.clear();
        self.polymorph_copies.clear();

        self.scopes.clear();
        self.scopes.push(Scope::new_top());
        self.function_scopes.clear();
        self.top_scope = ScopeId(0);

        self.errors.clear();

        self.top_level.clear();

        self.types.clear();
        self.type_matches.clear();
        self.type_array_reverse_map.clear();
        self.completes.clear();
        self.topo.clear();
        self.circular_dependency_nodes.clear();
        self.unification_data.reset();
        self.deferreds.clear();

        self.module = Self::make_module();
        self.values.clear();
    }

    pub fn parse_file(&mut self, file_name: &str) -> Result<(), CompileError> {
        let mut source = SourceInfo::<StrSource>::from_file(file_name);
        let mut parser = Parser::from_source(self, &mut source);

        parser.parse()
    }

    pub fn ropey_parse_file(&mut self, file_name: &str) -> Result<(), CompileError> {
        let mut source = SourceInfo::<RopeySource>::ropey_from_file(file_name);
        let mut parser = Parser::from_source(self, &mut source);
        parser.parse()
    }

    pub fn parse_str(&mut self, source: &'static str) -> Result<(), CompileError> {
        let mut source = SourceInfo::<StrSource>::from_str(source);
        let mut parser = Parser::from_source(self, &mut source);
        parser.parse()
    }

    pub fn parse_source<W: Source>(
        &mut self,
        source: &mut SourceInfo<W>,
    ) -> Result<(), CompileError> {
        let mut parser = Parser::<W>::from_source(self, source);
        parser.parse()
    }

    pub fn push_node(&mut self, range: Range, node: Node) -> NodeId {
        self.nodes.push(node);
        self.ranges.push(range);
        self.node_scopes.push(self.top_scope);

        NodeId(self.nodes.len() - 1)
    }

    pub fn push_id_vec(&mut self, vec: Vec<NodeId>) -> IdVec {
        self.id_vecs.push(Rc::new(RefCell::new(vec)));
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

    pub fn debug_tokens<W: Source>(
        &mut self,
        source: &mut SourceInfo<W>,
    ) -> Result<(), CompileError> {
        let mut parser = Parser::from_source(self, source);

        parser.pop();
        parser.pop();

        let mut a = 0;
        while parser.source_info.top.tok != Token::Eof && a < 250_000 {
            println!("{:?}", parser.source_info.top);
            parser.pop();
            a += 1;
        }

        Ok(())
    }

    pub fn get_symbol(&self, sym_id: NodeId) -> Sym {
        self.nodes[sym_id.0].as_symbol().unwrap()
    }

    pub fn report_error(&self, err: CompileError) {
        match err {
            CompileError::Generic(msg, range) => {
                println!("{:?}: {}", range, msg);
            }
            CompileError::Node(msg, id) => {
                println!("{:?}: {}", self.ranges[id.0], msg);
            }
            CompileError::Node2(msg, id1, id2) => {
                println!(
                    "{:?}, {:?}: {}",
                    self.ranges[id1.0], self.ranges[id2.0], msg
                );
            }
        }
    }

    fn match_fields_to_named_struct(
        &mut self,
        fields: IdVec,
        name: NodeId,
        struct_literal_id: NodeId,
    ) -> Result<(), CompileError> {
        let Some(Type::Struct { fields: decl, .. }) = self.types.get(&name) else { return Ok(()); };

        let given = self.id_vecs[fields.0].clone();
        let decl = self.id_vecs[decl.0].clone();

        let rearranged = match self.rearrange_params(&given.borrow(), &decl.borrow(), name) {
            Ok(r) => Some(r),
            Err(err) => return Err(err),
        };

        if let Some(rearranged) = rearranged {
            self.nodes[struct_literal_id.0] = Node::StructLiteral {
                name: Some(name),
                fields,
                rearranged_fields: Some(self.push_id_vec(rearranged.clone())),
            };

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

impl Context {
    pub fn prepare(&mut self) -> Result<(), CompileError> {
        for node in self.top_level.clone() {
            self.assign_type(node);
            self.unify_types();
            self.circular_dependency_nodes.clear();
        }

        let mut hardstop = 0;
        while hardstop < 100 && !self.deferreds.is_empty() {
            for node in std::mem::take(&mut self.deferreds) {
                self.assign_type(node);
                self.unify_types();
            }

            hardstop += 1;
        }

        if self.args.print_type_matches {
            println!(
                "type matches: {:?}",
                self.type_matches
                    .iter()
                    .map(|tm| {
                        tm.ids
                            .iter()
                            .map(|id| (self.nodes[id.0].ty(), self.ranges[id.0]))
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            );
            //     println!(
            //         "matching {} ({:?}) with {} ({:?})",
            //         self.nodes[ty1.0].ty(),
            //         self.ranges[ty1.0],
            //         self.nodes[ty2.0].ty(),
            //         self.ranges[ty2.0]
            //     );
        }

        // Check if any types are still unassigned
        for (id, ty) in self.types.iter() {
            if let Type::Infer(_) = *ty {
                self.errors
                    .push(CompileError::Node("Type not assigned".to_string(), *id));
            }
            if *ty == Type::IntLiteral {
                self.errors.push(CompileError::Node(
                    "Int literal not assigned".to_string(),
                    *id,
                ));
            }
            if *ty == Type::FloatLiteral {
                self.errors.push(CompileError::Node(
                    "Float literal not assigned".to_string(),
                    *id,
                ));
            }
        }

        if self.errors.is_empty() {
            self.predeclare_functions()
        } else {
            Err(self.errors[0].clone())
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

        if self.assign_type_inner(id) {
            self.completes.insert(id);
        }
    }

    pub fn assign_type_inner(&mut self, id: NodeId) -> bool {
        match self.nodes[id.0] {
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

                for &param in self.id_vecs[params.0].clone().borrow().iter() {
                    self.assign_type(param);

                    // All structs passed as function args are passed by address (for now...)
                    if let Some(Type::Struct { .. }) = &self.types.get(&param) {
                        self.addressable_nodes.insert(param);
                    }
                }

                self.types.insert(
                    id,
                    Type::Func {
                        return_ty,
                        input_tys: params,
                    },
                );

                for &stmt in self.id_vecs[stmts.0].clone().borrow().iter() {
                    self.assign_type(stmt);
                }

                for &ret_id in self.id_vecs[returns.0].clone().borrow().iter() {
                    let ret_id = match self.nodes[ret_id.0] {
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

                for &param in self.id_vecs[params.0].clone().borrow().iter() {
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

                        for &input_ty in self.id_vecs[input_tys.0].clone().borrow().iter() {
                            self.assign_type(input_ty);
                        }
                    }
                    Type::Struct { name, fields } => {
                        if let Some(name) = name {
                            self.assign_type(name);
                        }

                        for &field in self.id_vecs[fields.0].clone().borrow().iter() {
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
            Node::Symbol(sym) => {
                let resolved = self.scope_get(sym, id);

                match resolved {
                    Some(resolved) => {
                        self.assign_type(resolved);
                        self.match_types(id, resolved);

                        if self.addressable_nodes.contains(&resolved) {
                            self.addressable_nodes.insert(id);
                        }

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
            Node::ValueParam { name, value, .. } => {
                self.assign_type(value);
                self.match_types(id, value);
                if let Some(name) = name {
                    self.match_types(name, value);
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
                    self.errors.push(CompileError::Node(
                        "Under-specified float literal".to_string(),
                        id,
                    ));
                    return true;
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
            Node::Call {
                mut func, params, ..
            } => {
                self.assign_type(func);

                // If func is a polymorph, copy it first
                if self.polymorph_sources.contains(&func) {
                    match self.polymorph_copy(func) {
                        Ok(id) => {
                            func = id;
                            self.assign_type(func);
                        }
                        Err(err) => {
                            self.errors.push(err);
                            return true;
                        }
                    }
                }

                let param_ids = self.id_vecs[params.0].clone();
                for &param in param_ids.borrow().iter() {
                    self.assign_type(param);
                }

                if let Type::Func {
                    return_ty,
                    input_tys,
                } = self.get_type(func)
                {
                    let given = param_ids;
                    let decl = self.id_vecs[input_tys.0].clone();

                    let rearranged_given =
                        match self.rearrange_params(&given.borrow(), &decl.borrow(), id) {
                            Ok(r) => r,
                            Err(err) => {
                                self.errors.push(err);
                                return true;
                            }
                        };

                    self.nodes[id.0] = Node::Call {
                        func,
                        params,
                        rearranged_params: Some(self.push_id_vec(rearranged_given.clone())),
                    };

                    let input_ty_ids = self.id_vecs[input_tys.0].clone();

                    if input_ty_ids.borrow().len() != rearranged_given.len() {
                        self.errors.push(CompileError::Node(
                            "Incorrect number of parameters".to_string(),
                            id,
                        ));
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
                } else {
                    self.errors
                        .push(CompileError::Node("Not a function".to_string(), id));
                }
            }
            Node::StructDefinition { name, fields } => {
                let field_ids = self.id_vecs[fields.0].clone();
                for &field in field_ids.borrow().iter() {
                    self.assign_type(field);
                }

                self.types.insert(
                    id,
                    Type::Struct {
                        name: Some(name),
                        fields,
                    },
                );

                self.match_types(id, name);
            }
            Node::StructLiteral { name, fields, .. } => {
                if let Some(name) = name {
                    self.assign_type(name);
                    self.match_types(name, id);
                    self.match_fields_to_named_struct(fields, name, id);
                } else {
                    for &field in self.id_vecs[fields.0].clone().borrow().iter() {
                        self.assign_type(field);
                    }
                    self.types.insert(id, Type::Struct { name: None, fields });
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
                    Type::Struct { fields, .. } => {
                        let field_ids = self.id_vecs[fields.0].clone();
                        let mut found = false;

                        for &field in field_ids.borrow().iter() {
                            let field_name = match &self.nodes[field.0] {
                                Node::ValueParam {
                                    name: Some(name), ..
                                }
                                | Node::StructDeclParam { name, .. } => *name,
                                _ => unreachable!(
                                    "Struct field {:?} is not a ValueParam",
                                    &self.nodes[field.0]
                                ),
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

    fn polymorph_copy(&mut self, id: NodeId) -> Result<NodeId, CompileError> {
        let id = match self.nodes[id.0] {
            Node::Symbol(sym) => self.scope_get(sym, id).ok_or_else(|| {
                CompileError::Node("Undeclared symbol when copying polymorph".to_string(), id)
            }),
            _ => Ok(id),
        }?;

        // Re-parse the region of the source code that contains the id
        // todo(chad): @performance
        let range = self.ranges[id.0];
        let mut source = SourceInfo::<StrSource>::from_range(range);
        let mut parser = Parser::from_source(self, &mut source);
        parser.is_polymorph_copying = true;
        parser.pop();
        parser.pop();
        let copied = parser.parse_fn(false)?;
        self.polymorph_sources.remove(&copied);
        self.polymorph_copies.insert(copied);
        Ok(copied)
    }

    fn match_types(&mut self, ty1: NodeId, ty2: NodeId) {
        self.handle_match_types(ty1, ty2);
        self.merge_type_matches(ty1, ty2); // todo(chad): can we just do this once at the end? Would it be faster?
    }

    fn handle_match_types(&mut self, ty1: NodeId, ty2: NodeId) {
        if ty1 == ty2 {
            return;
        }

        // println!(
        //     "matching {} ({:?}) with {} ({:?})",
        //     self.nodes[ty1.0].ty(),
        //     self.ranges[ty1.0],
        //     self.nodes[ty2.0].ty(),
        //     self.ranges[ty2.0]
        // );

        // println!(
        //     "matching {:?} ({:?}) with {:?} ({:?})",
        //     self.types.get(&ty1),
        //     self.ranges[ty1.0],
        //     self.types.get(&ty2),
        //     self.ranges[ty2.0]
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

                let input_tys1 = self.id_vecs[input_tys1.0].clone();
                let input_tys2 = self.id_vecs[input_tys2.0].clone();

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
                    fields: f1,
                },
                Type::Struct {
                    name: n2,
                    fields: f2,
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
                    if let Err(err) = self.match_fields_to_named_struct(f1, n, ty1) {
                        self.errors.push(err);
                    }
                }
                (Some(n), None) => {
                    if let Err(err) = self.match_fields_to_named_struct(f2, n, ty2) {
                        self.errors.push(err);
                    }
                }
                (None, None) => {
                    let f1 = self.id_vecs[f1.0].clone();
                    let f2 = self.id_vecs[f2.0].clone();

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
                        "Could not match types: {} ({:?}), {} ({:?})",
                        self.nodes[ty1.0].ty(),
                        self.ranges[ty1.0],
                        self.nodes[ty2.0].ty(),
                        self.ranges[ty2.0]
                    ),
                    ty1,
                    ty2,
                ));
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
        return self.types.get(&id).cloned().unwrap_or(Type::Infer(None));
    }

    fn is_fully_concrete(&self, id: NodeId) -> bool {
        self.is_fully_concrete_ty(self.get_type(id))
    }

    fn is_fully_concrete_ty(&self, ty: Type) -> bool {
        if let Type::Pointer(pt) = ty {
            return self.is_fully_concrete(pt);
        }

        if let Type::Struct {
            name: Some(_),
            fields,
        } = ty
        {
            for &field in self.id_vecs[fields.0].borrow().iter() {
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

    fn find_type_array_index(&self, id: NodeId) -> Option<usize> {
        self.type_array_reverse_map.get(&id).cloned()
    }

    fn merge_type_matches(&mut self, ty1: NodeId, ty2: NodeId) {
        match (self.get_type(ty1), self.get_type(ty2)) {
            (Type::Infer(_), Type::Infer(_)) => (),
            (Type::Infer(_), ty) => {
                self.types.insert(ty1, ty);
            }
            (ty, Type::Infer(_)) => {
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

    fn rearrange_params(
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
                    self.nodes[given[given_idx].0],
                    Node::ValueParam { name: None, .. }
                )
            {
                rearranged_given.push(given[given_idx]);
                given_idx += 1;
            }

            let mut cgiven_idx = given_idx;
            while cgiven_idx < given_len {
                if matches!(
                    self.nodes[given[cgiven_idx].0],
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
                let decl_name = match &self.nodes[d.0] {
                    Node::FuncDeclParam { name, .. } | Node::StructDeclParam { name, .. } => *name,
                    _ => unreachable!(),
                };
                let decl_name_sym = self.get_symbol(decl_name);

                let given_name = match &self.nodes[g.0] {
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
                if let Node::FuncDeclParam {
                    default: Some(def), ..
                } = self.nodes[d.0]
                {
                    rearranged_given.push(def);
                } else {
                    return Err(CompileError::Node(
                        "Could not find parameter".to_string(),
                        err_id,
                    ));
                }
            }
        }

        Ok(rearranged_given)
    }
}

impl Context {
    pub fn predeclare_functions(&mut self) -> Result<(), CompileError> {
        let mut codegen_ctx = self.module.make_context();
        let mut func_ctx = FunctionBuilderContext::new();

        for &t in self.topo.clone().iter() {
            if !self.completes.contains(&t) {
                continue;
            }

            self.predeclare_function(t)?;
        }

        let mut compile_ctx = ToplevelCompileContext {
            ctx: self,
            codegen_ctx: &mut codegen_ctx,
            func_ctx: &mut func_ctx,
        };

        for &t in compile_ctx.ctx.topo.clone().iter() {
            compile_ctx.compile_toplevel_id(t)?;
        }

        self.module
            .finalize_definitions()
            .expect("Failed to finalize definitions");

        Ok(())
    }

    pub fn predeclare_function(&mut self, id: NodeId) -> Result<(), CompileError> {
        if let Node::Func {
            name,
            params,
            return_ty,
            ..
        } = self.nodes[id.0]
        {
            let mut sig = self.module.make_signature();

            for &param in self.id_vecs[params.0].borrow().iter() {
                sig.params
                    .push(AbiParam::new(self.get_cranelift_type(param)));
            }

            let return_size = return_ty.map(|rt| self.get_type_size(rt)).unwrap_or(0);

            if return_size > 0 {
                sig.returns
                    .push(AbiParam::new(self.get_cranelift_type(return_ty.unwrap())));
            }

            let func_name = self.func_name(name, id);

            let func = self
                .module
                .declare_function(&func_name, Linkage::Local, &sig)
                .unwrap();

            self.values.insert(id, Value::Func(func));
        }

        Ok(())
    }

    fn func_name(&self, name: Option<NodeId>, anonymous_id: NodeId) -> String {
        name.map(|n| {
            let sym = self.nodes[n.0].as_symbol().unwrap();
            if self.polymorph_copies.contains(&anonymous_id) {
                format!(
                    "{}_polycopy{}",
                    self.string_interner.resolve(sym.0).unwrap(),
                    anonymous_id.0
                )
            } else {
                self.string_interner.resolve(sym.0).unwrap().to_string()
            }
        })
        .unwrap_or_else(|| format!("anonymous__{}", anonymous_id.0))
    }

    pub fn call_fn(&mut self, fn_name: &str) -> Result<(), CompileError> {
        let fn_name_interned = self.string_interner.get_or_intern(fn_name);

        let node_id = self
            .top_level
            .iter()
            .filter(|tl| match self.nodes[tl.0] {
                Node::Func {
                    name: Some(name), ..
                } => self.get_symbol(name).0 == fn_name_interned,
                _ => false,
            })
            .next()
            .cloned();

        if let Some(id) = node_id {
            let code = self.module.get_finalized_function(self.get_func_id(id));
            let func = unsafe { std::mem::transmute::<_, fn() -> i64>(code) };
            func();
        }

        Ok(())
    }

    fn get_cranelift_type(&self, param: NodeId) -> CraneliftType {
        match self.types[&param] {
            Type::I8 => types::I8,
            Type::I16 => types::I16,
            Type::I32 => types::I32,
            Type::I64 => types::I64,
            Type::F32 => types::F32,
            Type::F64 => types::F64,
            Type::Pointer(_) | Type::Func { .. } | Type::Struct { .. } => self.get_pointer_type(),
            Type::Infer(_) => todo!(),
            _ => todo!("{:?}", &self.types[&param]),
        }
    }

    fn get_pointer_type(&self) -> CraneliftType {
        self.module.isa().pointer_type()
    }

    fn get_type_size(&self, param: NodeId) -> StackSize {
        match self.types[&param] {
            Type::I8 => 1,
            Type::I16 => 2,
            Type::I32 => 4,
            Type::I64 => 8,
            Type::F32 => 4,
            Type::F64 => 8,
            Type::Func { .. } => 8,
            Type::Pointer(_) => 8,
            Type::Struct { fields, .. } => {
                // todo(chad): c struct packing rules if annotated
                self.id_vecs[fields.0]
                    .clone()
                    .borrow()
                    .iter()
                    .map(|f| self.get_type_size(*f))
                    .sum()
            }
            Type::Infer(_) => todo!(),
            _ => todo!("get_type_size for {:?}", self.types[&param]),
        }
    }

    pub fn get_func_id(&self, id: NodeId) -> FuncId {
        match self.values[&id] {
            Value::Func(f) => f,
            _ => panic!("Not a function"),
        }
    }

    pub fn get_func_signature(&self, func: NodeId, param_ids: &Vec<NodeId>) -> Signature {
        let func_ty = self.get_type(func);
        let return_ty = match func_ty {
            Type::Func {
                return_ty: Some(return_ty),
                ..
            } => Some(return_ty),
            _ => None,
        };

        let return_size = return_ty.map(|rt| self.get_type_size(rt)).unwrap_or(0);

        let mut sig = self.module.make_signature();
        for param in param_ids.iter() {
            sig.params
                .push(AbiParam::new(self.get_cranelift_type(*param)));
        }

        if return_size > 0 {
            sig.returns
                .push(AbiParam::new(self.get_cranelift_type(return_ty.unwrap())));
        }

        sig
    }
}

pub struct Parser<'a, W: Source> {
    pub source_info: &'a mut SourceInfo<W>,
    pub ctx: &'a mut Context,
    pub is_polymorph_copying: bool,

    // stack of returns - pushed when entering parsing a function, popped when exiting
    // todo(chad): parsers should be cheap to create. Consider moving this to the context?
    pub returns: Vec<Vec<NodeId>>,
}

impl<'a, W: Source> Parser<'a, W> {
    pub fn from_source(context: &'a mut Context, source_info: &'a mut SourceInfo<W>) -> Self {
        Self {
            source_info,
            ctx: context,
            is_polymorph_copying: false,
            returns: Default::default(),
        }
    }

    fn pop(&mut self) {
        self.source_info.eat_spaces();

        let start = self.source_info.loc;
        self.source_info.top = self.source_info.second;

        let (is_underscore, is_bang) = if self.source_info.source.char_count() > 1 {
            (
                self.source_info.source.starts_with("_")
                    && !breaks_symbol(self.source_info.source.char_at(1).unwrap()),
                self.source_info.source.starts_with("!")
                    && !breaks_symbol(self.source_info.source.char_at(1).unwrap()),
            )
        } else {
            (false, false)
        };

        if is_underscore || is_bang {
            let start = self.source_info.loc;
            self.source_info.eat(1);

            let index = match self.source_info.source.position_of(breaks_symbol) {
                Some(index) => index,
                None => self.source_info.source.char_count(),
            };

            if index == 0 {
                self.source_info.second = Lexeme::new(Token::Eof, Default::default());
                return;
            } else {
                let sym = self
                    .ctx
                    .string_interner
                    .get_or_intern(&self.source_info.source.slice(0..index));
                self.source_info.eat(index);

                let end = self.source_info.loc;

                let symtok = if is_underscore {
                    Token::UnderscoreSymbol(Sym(sym))
                } else if is_bang {
                    Token::BangSymbol(Sym(sym))
                } else {
                    unreachable!();
                };

                self.source_info.second =
                    Lexeme::new(symtok, self.source_info.make_range(start, end));

                return;
            }
        }

        if self.source_info.prefix_keyword("fn", Token::Fn) {
            return;
        }
        if self.source_info.prefix_keyword("extern", Token::Extern) {
            return;
        }
        if self.source_info.prefix_keyword("let", Token::Let) {
            return;
        }
        if self.source_info.prefix_keyword("return", Token::Return) {
            return;
        }
        if self.source_info.prefix_keyword("struct", Token::Struct) {
            return;
        }
        if self.source_info.prefix_keyword("i8", Token::I8) {
            return;
        }
        if self.source_info.prefix_keyword("i16", Token::I16) {
            return;
        }
        if self.source_info.prefix_keyword("i32", Token::I32) {
            return;
        }
        if self.source_info.prefix_keyword("i64", Token::I64) {
            return;
        }
        if self.source_info.prefix_keyword("u8", Token::U8) {
            return;
        }
        if self.source_info.prefix_keyword("u16", Token::U16) {
            return;
        }
        if self.source_info.prefix_keyword("u32", Token::U32) {
            return;
        }
        if self.source_info.prefix_keyword("u64", Token::U64) {
            return;
        }
        if self.source_info.prefix_keyword("f32", Token::F32) {
            return;
        }
        if self.source_info.prefix_keyword("f64", Token::F64) {
            return;
        }
        if self.source_info.prefix("(", Token::LParen) {
            return;
        }
        if self.source_info.prefix(")", Token::RParen) {
            return;
        }
        if self.source_info.prefix("{", Token::LCurly) {
            return;
        }
        if self.source_info.prefix("}", Token::RCurly) {
            return;
        }
        if self.source_info.prefix(",", Token::Comma) {
            return;
        }
        if self.source_info.prefix(".", Token::Dot) {
            return;
        }
        if self.source_info.prefix(";", Token::Semicolon) {
            return;
        }
        if self.source_info.prefix(":", Token::Colon) {
            return;
        }
        if self.source_info.prefix("=", Token::Eq) {
            return;
        }
        if self.source_info.prefix("_{", Token::UnderscoreLCurly) {
            return;
        }
        if self.source_info.prefix("_", Token::Underscore) {
            return;
        }
        if self.source_info.prefix("+", Token::Plus) {
            return;
        }
        if self.source_info.prefix("-", Token::Dash) {
            return;
        }
        if self.source_info.prefix("*", Token::Star) {
            return;
        }
        if self.source_info.prefix("&", Token::AddressOf) {
            return;
        }
        if self.source_info.prefix("/", Token::Slash) {
            return;
        }
        if self.source_info.prefix("!", Token::Bang) {
            return;
        }

        let new_second = match self.source_info.source.next_char() {
            Some(c) if c.is_digit(10) => {
                let index = match self.source_info.source.position_of(|c| !c.is_digit(10)) {
                    Some(index) => index,
                    None => self.source_info.source.char_count(),
                };

                let has_decimal = match self.source_info.source.char_at(index) {
                    Some(c) => c == '.',
                    _ => false,
                };

                let digit = self
                    .source_info
                    .source
                    .slice(0..index)
                    .parse::<i64>()
                    .expect("Failed to parse numeric literal");

                self.source_info.eat(index);

                if has_decimal {
                    self.source_info.eat(1);

                    let decimal_index =
                        match self.source_info.source.position_of(|c| !c.is_digit(10)) {
                            Some(index) => index,
                            None => self.source_info.source.char_count(),
                        };

                    let decimal_digit = self
                        .source_info
                        .source
                        .slice(0..decimal_index)
                        .parse::<i64>()
                        .expect("Failed to parse numeric literal");

                    self.source_info.eat(decimal_index);

                    let digit: f64 = format!("{}.{}", digit, decimal_digit).parse().unwrap();

                    let mut spec = NumericSpecification::None;
                    if self.source_info.source.starts_with("f32") {
                        spec = NumericSpecification::F32;
                        self.source_info.eat(3);
                    } else if self.source_info.source.starts_with("f64") {
                        spec = NumericSpecification::F64;
                        self.source_info.eat(3);
                    }

                    let end = self.source_info.loc;
                    Lexeme::new(
                        Token::FloatLiteral(digit, spec),
                        self.source_info.make_range(start, end),
                    )
                } else {
                    let mut spec = NumericSpecification::None;

                    if self.source_info.source.starts_with("i8") {
                        spec = NumericSpecification::I8;
                        self.source_info.eat(2);
                    } else if self.source_info.source.starts_with("i16") {
                        spec = NumericSpecification::I16;
                        self.source_info.eat(3);
                    } else if self.source_info.source.starts_with("i32") {
                        spec = NumericSpecification::I32;
                        self.source_info.eat(3);
                    } else if self.source_info.source.starts_with("i64") {
                        spec = NumericSpecification::I64;
                        self.source_info.eat(3);
                    } else if self.source_info.source.starts_with("u8") {
                        spec = NumericSpecification::U8;
                        self.source_info.eat(2);
                    } else if self.source_info.source.starts_with("u16") {
                        spec = NumericSpecification::U16;
                        self.source_info.eat(3);
                    } else if self.source_info.source.starts_with("u32") {
                        spec = NumericSpecification::U32;
                        self.source_info.eat(3);
                    } else if self.source_info.source.starts_with("u64") {
                        spec = NumericSpecification::U64;
                        self.source_info.eat(3);
                    } else if self.source_info.source.starts_with("f32") {
                        spec = NumericSpecification::F32;
                        self.source_info.eat(3);
                    } else if self.source_info.source.starts_with("f64") {
                        spec = NumericSpecification::F64;
                        self.source_info.eat(3);
                    }

                    let end = self.source_info.loc;
                    Lexeme::new(
                        Token::IntegerLiteral(digit, spec),
                        self.source_info.make_range(start, end),
                    )
                }
            }
            Some(_) => {
                let index = match self.source_info.source.position_of(breaks_symbol) {
                    Some(index) => index,
                    None => self.source_info.source.char_count(),
                };

                if index == 0 {
                    Lexeme::new(Token::Eof, Default::default())
                } else {
                    let sym = self
                        .ctx
                        .string_interner
                        .get_or_intern(&self.source_info.source.slice(0..index));
                    self.source_info.eat(index);

                    let end = self.source_info.loc;

                    Lexeme::new(
                        Token::Symbol(Sym(sym)),
                        self.source_info.make_range(start, end),
                    )
                }
            }
            None => Lexeme::new(Token::Eof, Default::default()),
        };

        self.source_info.second = new_second;
    }

    fn expect(&mut self, tok: Token) -> Result<(), CompileError> {
        match self.source_info.top.tok {
            t if t == tok => {
                self.pop();
                Ok(())
            }
            _ => {
                let msg =
                    format!("Expected {:?}, found {:?}", tok, self.source_info.top.tok).to_string();
                Err(CompileError::Generic(msg, self.source_info.top.range))
            }
        }
    }

    fn expect_range(&mut self, start: Location, token: Token) -> Result<Range, CompileError> {
        let range = self
            .source_info
            .make_range(start, self.source_info.top.range.end);

        if self.source_info.top.tok == token {
            self.pop();
            Ok(range)
        } else {
            let msg =
                format!("Expected {:?}, found {:?}", token, self.source_info.top.tok).to_string();
            Err(CompileError::Generic(msg, range))
        }
    }

    pub fn parse(&mut self) -> Result<(), CompileError> {
        self.pop();
        self.pop();

        while self.source_info.top.tok != Token::Eof {
            let tl = self.parse_top_level()?;
            self.ctx.top_level.push(tl);
        }

        Ok(())
    }

    pub fn parse_top_level(&mut self) -> Result<NodeId, CompileError> {
        let tl = match self.source_info.top.tok {
            Token::Fn => Ok(self.parse_fn(false)?),
            Token::Extern => Ok(self.parse_extern()?),
            Token::Struct => Ok(self.parse_struct_definition()?),
            _ => {
                let msg = format!("expected 'fn', found '{:?}'", self.source_info.top.tok);

                Err(CompileError::Generic(msg, self.source_info.top.range))
            }
        }?;

        Ok(tl)
    }

    fn parse_symbol(&mut self) -> Result<NodeId, CompileError> {
        let range = self.source_info.top.range;
        match self.source_info.top.tok {
            Token::Symbol(sym) => {
                self.pop();
                Ok(self.ctx.push_node(range, Node::Symbol(sym)))
            }
            _ => Err(CompileError::Generic("expected symbol".to_string(), range)),
        }
    }

    pub fn parse_value_params(&mut self) -> Result<IdVec, CompileError> {
        let mut params = Vec::new();

        while self.source_info.top.tok != Token::RParen && self.source_info.top.tok != Token::RCurly
        {
            if let Token::Colon = self.source_info.second.tok {
                let name = self.parse_symbol()?;
                self.expect(Token::Colon)?;
                let value = self.parse_expression()?;
                params.push(self.ctx.push_node(
                    self.make_range_spanning(name, value),
                    Node::ValueParam {
                        name: Some(name),
                        value,
                        index: params.len() as u16,
                    },
                ));
            } else {
                let value = self.parse_expression()?;
                params.push(self.ctx.push_node(
                    self.ctx.ranges[value.0],
                    Node::ValueParam {
                        name: None,
                        value,
                        index: params.len() as u16,
                    },
                ));
            }

            if self.source_info.top.tok == Token::Comma {
                self.pop(); // `,`
            } else {
                break;
            }
        }

        Ok(self.ctx.push_id_vec(params))
    }

    fn parse_decl_params(&mut self, is_fn: bool) -> Result<IdVec, CompileError> {
        let mut params = Vec::new();

        while self.source_info.top.tok != Token::RParen && self.source_info.top.tok != Token::RCurly
        {
            let input_start = self.source_info.top.range.start;

            let name = self.parse_symbol()?;
            let name_sym = self.ctx.nodes[name.0].as_symbol().unwrap();

            let (ty, default) = if self.source_info.top.tok == Token::Colon {
                self.pop(); // `:`
                let ty = self.parse_type()?;

                let mut default = None;
                if self.source_info.top.tok == Token::Eq {
                    self.pop(); // `=`
                    default = Some(self.parse_expression()?);
                }

                (Some(ty), default)
            } else if self.source_info.top.tok == Token::Eq {
                todo!("parse default parameters")
            } else {
                (None, None)
            };

            let range_end = match (ty, default) {
                (_, Some(default)) => self.ctx.ranges[default.0].end,
                (Some(ty), _) => self.ctx.ranges[ty.0].end,
                _ => self.ctx.ranges[name.0].end,
            };

            let range = self.source_info.make_range(input_start, range_end);

            // put the param in scope
            let node = if is_fn {
                Node::FuncDeclParam {
                    name,
                    ty,
                    default,
                    index: params.len() as u16,
                }
            } else {
                Node::StructDeclParam {
                    name,
                    ty,
                    default,
                    index: params.len() as u16,
                }
            };

            let param = self.ctx.push_node(range, node);
            self.ctx.scope_insert(name_sym, param);
            self.ctx.addressable_nodes.insert(param);
            params.push(param);

            if self.source_info.top.tok == Token::Comma {
                self.pop(); // `,`
            }
        }

        Ok(self.ctx.push_id_vec(params))
    }

    pub fn parse_struct_definition(&mut self) -> Result<NodeId, CompileError> {
        let start = self.source_info.top.range.start;

        self.pop(); // `struct`

        let name = self.parse_symbol()?;
        let name_sym = self.ctx.nodes[name.0].as_symbol().unwrap();

        let pushed_scope = self.ctx.push_scope(false);

        self.expect(Token::LCurly)?;
        let fields = self.parse_decl_params(false)?;
        let range = self.expect_range(start, Token::RCurly)?;

        self.ctx.pop_scope(pushed_scope);

        let struct_node = self
            .ctx
            .push_node(range, Node::StructDefinition { name, fields });
        self.ctx.scope_insert(name_sym, struct_node);

        Ok(struct_node)
    }

    pub fn parse_fn(&mut self, anonymous: bool) -> Result<NodeId, CompileError> {
        let old_polymorph_target = self.ctx.polymorph_target;
        self.ctx.polymorph_target = false;

        let start = self.source_info.top.range.start;

        self.pop(); // `fn`

        let name = if anonymous {
            None
        } else {
            Some(self.parse_symbol()?)
        };
        let name_sym = name.map(|n| self.ctx.nodes[n.0].as_symbol().unwrap());

        // open a new scope
        let pushed_scope = self.ctx.push_scope(true);

        self.expect(Token::LParen)?;
        let params = self.parse_decl_params(true)?;
        self.expect(Token::RParen)?;

        let return_ty = if self.source_info.top.tok != Token::LCurly {
            Some(self.parse_type()?)
        } else {
            None
        };

        self.expect(Token::LCurly)?;

        self.returns.push(Vec::new());

        let mut stmts = Vec::new();
        while self.source_info.top.tok != Token::RCurly {
            let stmt = self.parse_fn_stmt()?;
            stmts.push(stmt);
        }
        let stmts = self.ctx.push_id_vec(stmts);

        let range = self.expect_range(start, Token::RCurly)?;

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

        if !self.is_polymorph_copying {
            if let Some(name_sym) = name_sym {
                self.ctx.scope_insert(name_sym, func);
            }
        }

        if self.ctx.polymorph_target {
            self.ctx.polymorph_sources.insert(func);
        }

        self.ctx.polymorph_target = old_polymorph_target;

        Ok(func)
    }

    pub fn parse_extern(&mut self) -> Result<NodeId, CompileError> {
        let start = self.source_info.top.range.start;

        self.pop(); // `extern`

        let name = self.parse_symbol()?;
        let name_sym = self.ctx.nodes[name.0].as_symbol().unwrap();

        let pushed_scope = self.ctx.push_scope(false);

        self.expect(Token::LParen)?;
        let params = self.parse_decl_params(true)?;
        self.expect(Token::RParen)?;

        self.ctx.pop_scope(pushed_scope);

        let return_ty = if self.source_info.top.tok != Token::Semicolon {
            Some(self.parse_type()?)
        } else {
            None
        };

        let range = self.expect_range(start, Token::Semicolon)?;

        let id = self.ctx.push_node(
            range,
            Node::Extern {
                name,
                params,
                return_ty,
            },
        );

        self.ctx.scope_insert(name_sym, id);

        Ok(id)
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
                | Token::UnderscoreLCurly
                | Token::AddressOf
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

                    self.pop(); // `+`
                }
                Token::Dash => {
                    if !parsing_op {
                        break;
                    }

                    while !operators.is_empty()
                        && operators.last().unwrap().precedence()
                            >= Op::from(self.source_info.top.tok).precedence()
                    {
                        output.push(Shunting::Op(operators.pop().unwrap()));
                    }
                    operators.push(Op::Sub);

                    self.pop(); // `-`
                }
                Token::Star => {
                    if parsing_op {
                        while !operators.is_empty()
                            && operators.last().unwrap().precedence()
                                >= Op::from(self.source_info.top.tok).precedence()
                        {
                            output.push(Shunting::Op(operators.pop().unwrap()));
                        }
                        operators.push(Op::Mul);

                        self.pop(); // `*`
                    } else {
                        let start = self.source_info.top.range.start;

                        self.pop(); // `*`

                        let id = self.parse_expression_piece()?;
                        let id = self.ctx.push_node(
                            Range::new(
                                start,
                                self.ctx.ranges[id.0].end,
                                self.source_info.source_path,
                            ),
                            Node::Deref(id),
                        );

                        self.ctx.addressable_nodes.insert(id);

                        output.push(Shunting::Id(id))
                    }
                }
                Token::Slash => {
                    if !parsing_op {
                        break;
                    }

                    while !operators.is_empty()
                        && operators.last().unwrap().precedence()
                            >= Op::from(self.source_info.top.tok).precedence()
                    {
                        output.push(Shunting::Op(operators.pop().unwrap()));
                    }
                    operators.push(Op::Div);

                    self.pop(); // `/`
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

        if output.is_empty() {
            return Err(CompileError::Generic(
                "Could not parse expression".to_string(),
                self.source_info.top.range,
            ));
        }

        let output_id = output[0].as_id();
        if let Some(output_id) = output_id {
            self.shunting_unroll(&mut output, output_id)
        } else {
            Err(CompileError::Generic(
                "Could not parse expression".to_string(),
                self.source_info.top.range,
            ))
        }
    }

    pub fn parse_expression_piece(&mut self) -> Result<NodeId, CompileError> {
        let start = self.source_info.top.range.start;

        let mut value = match self.source_info.top.tok {
            Token::IntegerLiteral(_, _) | Token::FloatLiteral(_, _) => self.parse_numeric_literal(),
            Token::Fn => self.parse_fn(true),
            Token::AddressOf => {
                self.pop(); // `&`
                let expr = self.parse_expression_piece()?;
                let end = self.ctx.ranges[expr.0].end;
                let addrof = self.ctx.push_node(
                    Range::new(start, end, self.source_info.source_path),
                    Node::AddressOf(expr),
                );

                self.ctx.addressable_nodes.insert(addrof);

                Ok(addrof)
            }
            _ => self.parse_lvalue(),
        }?;

        // function call?
        while self.source_info.top.tok == Token::LParen {
            self.pop(); // `(`
            let params = self.parse_value_params()?;
            let end = self.expect_range(start, Token::RParen)?.end;
            value = self.ctx.push_node(
                Range::new(start, end, self.source_info.source_path),
                Node::Call {
                    func: value,
                    params,
                    rearranged_params: None,
                },
            );
        }

        // member access?
        while self.source_info.top.tok == Token::Dot {
            self.pop(); // `.`
            let member = self.parse_symbol()?;
            let end = self.ctx.ranges[member.0].end;
            value = self.ctx.push_node(
                Range::new(start, end, self.source_info.source_path),
                Node::MemberAccess { value, member },
            );
            self.ctx.addressable_nodes.insert(value);
        }

        Ok(value)
    }

    fn parse_lvalue(&mut self) -> Result<NodeId, CompileError> {
        if self.source_info.top.tok == Token::LParen {
            self.pop(); // `(`
            let expr = self.parse_expression()?;
            self.expect(Token::RParen)?;

            Ok(expr)
        } else if self.source_info.top.tok == Token::I8 {
            let range = self.source_info.top.range;
            self.pop();
            Ok(self.ctx.push_node(range, Node::Type(Type::I8)))
        } else if self.source_info.top.tok == Token::I16 {
            let range = self.source_info.top.range;
            self.pop();
            Ok(self.ctx.push_node(range, Node::Type(Type::I16)))
        } else if self.source_info.top.tok == Token::I32 {
            let range = self.source_info.top.range;
            self.pop();
            Ok(self.ctx.push_node(range, Node::Type(Type::I32)))
        } else if self.source_info.top.tok == Token::I64 {
            let range = self.source_info.top.range;
            self.pop();
            Ok(self.ctx.push_node(range, Node::Type(Type::I64)))
        } else if self.source_info.top.tok == Token::U8 {
            let range = self.source_info.top.range;
            self.pop();
            Ok(self.ctx.push_node(range, Node::Type(Type::U8)))
        } else if self.source_info.top.tok == Token::U16 {
            let range = self.source_info.top.range;
            self.pop();
            Ok(self.ctx.push_node(range, Node::Type(Type::U16)))
        } else if self.source_info.top.tok == Token::U32 {
            let range = self.source_info.top.range;
            self.pop();
            Ok(self.ctx.push_node(range, Node::Type(Type::U32)))
        } else if self.source_info.top.tok == Token::U64 {
            let range = self.source_info.top.range;
            self.pop();
            Ok(self.ctx.push_node(range, Node::Type(Type::U64)))
        } else if self.source_info.top.tok == Token::F32 {
            let range = self.source_info.top.range;
            self.pop();
            Ok(self.ctx.push_node(range, Node::Type(Type::F32)))
        } else if self.source_info.top.tok == Token::F64 {
            let range = self.source_info.top.range;
            self.pop();
            Ok(self.ctx.push_node(range, Node::Type(Type::F64)))
        } else if let Token::UnderscoreLCurly = self.source_info.top.tok {
            Ok(self.parse_struct_literal()?)
        } else if let Token::Symbol(_) = self.source_info.top.tok {
            if self.source_info.second.tok == Token::LCurly {
                Ok(self.parse_struct_literal()?)
            } else {
                Ok(self.parse_symbol()?)
            }
        } else {
            Err(CompileError::Generic(
                "Could not parse lvalue".to_string(),
                self.source_info.top.range,
            ))
        }
    }

    fn parse_struct_literal(&mut self) -> Result<NodeId, CompileError> {
        let start = self.source_info.top.range.start;

        let name = if self.source_info.top.tok == Token::UnderscoreLCurly {
            self.pop(); // `_{`
            None
        } else {
            let sym = self.parse_symbol()?;
            self.expect(Token::LCurly)?;
            Some(sym)
        };

        let fields = self.parse_value_params()?;
        let range = self.expect_range(start, Token::RCurly)?;

        let struct_node = self.ctx.push_node(
            range,
            Node::StructLiteral {
                name,
                fields,
                rearranged_fields: None,
            },
        );

        Ok(struct_node)
    }

    fn parse_numeric_literal(&mut self) -> Result<NodeId, CompileError> {
        match self.source_info.top.tok {
            Token::IntegerLiteral(i, s) => {
                let range = self.source_info.top.range;
                self.pop();
                Ok(self.ctx.push_node(range, Node::IntLiteral(i, s)))
            }
            Token::FloatLiteral(f, s) => {
                let range = self.source_info.top.range;
                self.pop();
                Ok(self.ctx.push_node(range, Node::FloatLiteral(f, s)))
            }
            _ => Err(CompileError::Generic(
                "Expected numeric literal".to_string(),
                self.source_info.top.range,
            )),
        }
    }

    pub fn parse_let(&mut self, start: Location) -> Result<NodeId, CompileError> {
        self.pop(); // `let`
        let name = self.parse_symbol()?;
        let name_sym = self.ctx.nodes[name.0].as_symbol().unwrap();

        let ty = if self.source_info.top.tok == Token::Colon {
            self.pop(); // `:`
            Some(self.parse_type()?)
        } else {
            None
        };

        let expr = match self.source_info.top.tok {
            Token::Semicolon => None,
            Token::Eq => {
                self.pop(); // `=`
                Some(self.parse_expression()?)
            }
            _ => {
                return Err(CompileError::Generic(
                    "Expected `;` or `=`".to_string(),
                    self.source_info.top.range,
                ));
            }
        };
        let range = self.expect_range(start, Token::Semicolon)?;
        let let_id = self.ctx.push_node(range, Node::Let { name, ty, expr });

        self.ctx.addressable_nodes.insert(let_id);
        self.ctx.scope_insert(name_sym, let_id);

        Ok(let_id)
    }

    pub fn parse_fn_stmt(&mut self) -> Result<NodeId, CompileError> {
        let start = self.source_info.top.range.start;

        let r = match self.source_info.top.tok {
            Token::Return => {
                self.pop(); // `return`

                let expr = if self.source_info.top.tok != Token::Semicolon {
                    Some(self.parse_expression()?)
                } else {
                    None
                };

                let range = self.expect_range(start, Token::Semicolon)?;

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
                        self.expect(Token::Eq)?;
                        let expr = self.parse_expression()?;
                        let range = self.expect_range(start, Token::Semicolon)?;

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
                        self.ctx.ranges[lvalue.0] = self.expect_range(start, Token::Semicolon)?;
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
                self.pop();
                Ok(self.ctx.push_node(range, Node::Type(Type::I8)))
            }
            Token::I16 => {
                let range = self.source_info.top.range;
                self.pop();
                Ok(self.ctx.push_node(range, Node::Type(Type::I16)))
            }
            Token::I32 => {
                let range = self.source_info.top.range;
                self.pop();
                Ok(self.ctx.push_node(range, Node::Type(Type::I32)))
            }
            Token::I64 => {
                let range = self.source_info.top.range;
                self.pop();
                Ok(self.ctx.push_node(range, Node::Type(Type::I64)))
            }
            Token::U8 => {
                let range = self.source_info.top.range;
                self.pop();
                Ok(self.ctx.push_node(range, Node::Type(Type::U8)))
            }
            Token::U16 => {
                let range = self.source_info.top.range;
                self.pop();
                Ok(self.ctx.push_node(range, Node::Type(Type::U16)))
            }
            Token::U32 => {
                let range = self.source_info.top.range;
                self.pop();
                Ok(self.ctx.push_node(range, Node::Type(Type::U32)))
            }
            Token::U64 => {
                let range = self.source_info.top.range;
                self.pop();
                Ok(self.ctx.push_node(range, Node::Type(Type::U64)))
            }
            Token::F32 => {
                let range = self.source_info.top.range;
                self.pop();
                Ok(self.ctx.push_node(range, Node::Type(Type::F32)))
            }
            Token::F64 => {
                let range = self.source_info.top.range;
                self.pop();
                Ok(self.ctx.push_node(range, Node::Type(Type::F64)))
            }
            Token::Fn => {
                let range = self.source_info.top.range;
                self.pop(); // `fn`

                self.expect(Token::LParen)?;
                let params = self.parse_decl_params(true)?;
                self.expect(Token::RParen)?;

                let return_ty = if !matches!(
                    self.source_info.top.tok,
                    Token::RCurly | Token::Comma | Token::Semicolon | Token::RParen
                ) {
                    Some(self.parse_type()?)
                } else {
                    None
                };

                Ok(self.ctx.push_node(
                    range,
                    Node::Type(Type::Func {
                        return_ty,
                        input_tys: params,
                    }),
                ))
            }
            Token::Struct => {
                let range = self.source_info.top.range;
                self.pop(); // `struct`

                self.expect(Token::LCurly)?;

                let pushed_scope = self.ctx.push_scope(false);

                let fields = self.parse_decl_params(false)?;
                let range = self.expect_range(range.start, Token::RCurly)?;

                self.ctx.pop_scope(pushed_scope);

                Ok(self
                    .ctx
                    .push_node(range, Node::Type(Type::Struct { name: None, fields })))
            }
            Token::Star => {
                let mut range = self.source_info.top.range;
                self.pop(); // `*`
                let ty = self.parse_type()?;
                range.end = self.ctx.ranges[ty.0].end;
                Ok(self.ctx.push_node(range, Node::Type(Type::Pointer(ty))))
            }
            Token::Underscore => {
                let range = self.source_info.top.range;
                self.pop(); // `_`
                Ok(self.ctx.push_node(range, Node::Type(Type::Infer(None))))
            }
            Token::UnderscoreSymbol(sym) => {
                let range = self.source_info.top.range;
                self.pop(); // `_T`
                let id = self
                    .ctx
                    .push_node(range, Node::Type(Type::Infer(Some(sym))));
                self.ctx.scope_insert(sym, id);
                Ok(id)
            }
            Token::Bang => {
                let range = self.source_info.top.range;
                self.pop(); // `!`
                self.ctx.polymorph_target = true;
                Ok(self.ctx.push_node(range, Node::Type(Type::Infer(None))))
            }
            Token::BangSymbol(sym) => {
                let range = self.source_info.top.range;
                self.pop(); // `!T`
                self.ctx.polymorph_target = true;
                let id = self
                    .ctx
                    .push_node(range, Node::Type(Type::Infer(Some(sym))));
                self.ctx.scope_insert(sym, id);
                Ok(id)
            }
            Token::Symbol(sym) => {
                let range = self.source_info.top.range;
                self.pop();
                Ok(self.ctx.push_node(range, Node::Symbol(sym)))
            }
            _ => Err(CompileError::Generic(
                "Expected type".to_string(),
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
            return Err(CompileError::Node(
                "Unfinished expression".to_string(),
                err_node_id,
            ));
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
        let start = self.ctx.ranges[start.0];
        let end = self.ctx.ranges[end.0];
        Range::spanning(start, end)
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

struct ToplevelCompileContext<'a> {
    pub ctx: &'a mut Context,
    pub codegen_ctx: &'a mut CodegenContext,
    pub func_ctx: &'a mut FunctionBuilderContext,
}

struct FunctionCompileContext<'a> {
    pub ctx: &'a mut Context,
    pub builder: FunctionBuilder<'a>,
    pub current_block: Block,
}

impl<'a> FunctionCompileContext<'a> {
    pub fn compile_id(&mut self, id: NodeId) -> Result<(), CompileError> {
        // idempotency
        match self.ctx.values.get(&id) {
            None | Some(Value::Unassigned) => {}
            _ => return Ok(()),
        };

        match self.ctx.nodes[id.0] {
            Node::Symbol(sym) => {
                let resolved = self.ctx.scope_get(sym, id);
                match resolved {
                    Some(res) => {
                        self.compile_id(res)?;
                        self.ctx
                            .values
                            .get(&res)
                            .cloned()
                            .map(|v| self.ctx.values.insert(id, v));
                        Ok(())
                    }
                    _ => todo!(),
                }
            }
            Node::IntLiteral(n, _) => match self.ctx.types[&id] {
                Type::I64 => {
                    let value = self.builder.ins().iconst(types::I64, n);
                    self.ctx.values.insert(id, Value::Value(value));
                    Ok(())
                }
                Type::I32 => {
                    let value = self.builder.ins().iconst(types::I32, n);
                    self.ctx.values.insert(id, Value::Value(value));
                    Ok(())
                }
                _ => todo!(),
            },
            Node::FloatLiteral(n, _) => match self.ctx.types[&id] {
                Type::F64 => {
                    let value = self.builder.ins().f64const(n);
                    self.ctx.values.insert(id, Value::Value(value));
                    Ok(())
                }
                Type::F32 => {
                    let value = self.builder.ins().f32const(n as f32);
                    self.ctx.values.insert(id, Value::Value(value));
                    Ok(())
                }
                _ => todo!(),
            },
            Node::Type(_) => todo!(),
            Node::Return(rv) => {
                if let Some(rv) = rv {
                    self.compile_id(rv)?;
                    let value = self.rvalue(rv);
                    self.builder.ins().return_(&[value]);
                } else {
                    self.builder.ins().return_(&[]);
                }

                Ok(())
            }
            Node::Let { expr, .. } => {
                let size: u32 = self.ctx.get_type_size(id);
                let slot = self.builder.create_sized_stack_slot(StackSlotData {
                    kind: StackSlotKind::ExplicitSlot,
                    size,
                });

                let slot_addr =
                    self.builder
                        .ins()
                        .stack_addr(self.ctx.module.isa().pointer_type(), slot, 0);
                let value = Value::Value(slot_addr);

                if let Some(expr) = expr {
                    self.compile_id(expr)?;
                    self.store(expr, value);
                }

                self.ctx.values.insert(id, value);

                Ok(())
            }
            Node::Set { name, expr, .. } => {
                self.compile_id(name)?;

                let addr = self.ctx.values[&name];

                self.compile_id(expr)?;
                self.store(expr, addr);

                Ok(())
            }
            Node::StructDeclParam { index, .. } => {
                // we need our own storage
                let size = self.ctx.get_type_size(id);
                let slot = self.builder.create_sized_stack_slot(StackSlotData {
                    kind: StackSlotKind::ExplicitSlot,
                    size,
                });

                let slot_addr =
                    self.builder
                        .ins()
                        .stack_addr(self.ctx.module.isa().pointer_type(), slot, 0);
                let value = Value::Value(slot_addr);
                self.ctx.values.insert(id, value);

                let params = self.builder.block_params(self.current_block);
                let param_value = params[index as usize];

                self.builder
                    .ins()
                    .store(MemFlags::new(), param_value, slot_addr, 0);

                Ok(())
            }
            Node::FuncDeclParam { index, .. } => {
                // we need our own storage
                let size = self.ctx.get_type_size(id);
                let slot = self.builder.create_sized_stack_slot(StackSlotData {
                    kind: StackSlotKind::ExplicitSlot,
                    size,
                });

                let slot_addr =
                    self.builder
                        .ins()
                        .stack_addr(self.ctx.module.isa().pointer_type(), slot, 0);
                let value = Value::Value(slot_addr);
                self.ctx.values.insert(id, value);

                let params = self.builder.block_params(self.current_block);
                let param_value = params[index as usize];

                if self.ctx.addressable_nodes.contains(&id) {
                    let size = self.ctx.get_type_size(id);

                    let source_value = param_value;
                    let dest_value = slot_addr;

                    self.builder.emit_small_memory_copy(
                        self.ctx.module.isa().frontend_config(),
                        dest_value,
                        source_value,
                        size as _,
                        1,
                        1,
                        true, // non-overlapping
                        MemFlags::new(),
                    );
                } else {
                    self.builder
                        .ins()
                        .store(MemFlags::new(), param_value, slot_addr, 0);
                }

                Ok(())
            }
            Node::ValueParam { value, .. } => {
                self.compile_id(value)?;
                self.ctx.values.insert(id, self.ctx.values[&value]);
                if self.ctx.addressable_nodes.contains(&value) {
                    self.ctx.addressable_nodes.insert(id);
                }
                Ok(())
            }
            Node::BinOp { op, lhs, rhs } => {
                self.compile_id(lhs)?;
                self.compile_id(rhs)?;

                let lhs_value = self.rvalue(lhs);
                let rhs_value = self.rvalue(rhs);

                let value = match op {
                    Op::Add => self.builder.ins().iadd(lhs_value, rhs_value),
                    Op::Sub => self.builder.ins().isub(lhs_value, rhs_value),
                    Op::Mul => self.builder.ins().imul(lhs_value, rhs_value),
                    Op::Div => self.builder.ins().sdiv(lhs_value, rhs_value),
                };

                self.ctx.values.insert(id, Value::Value(value));

                Ok(())
            }
            Node::Call {
                func,
                rearranged_params,
                ..
            } => {
                let params = rearranged_params.unwrap();

                self.compile_id(func)?;

                let param_ids = self.ctx.id_vecs[params.0].clone();
                let mut param_values = Vec::new();
                for &param in param_ids.borrow().iter() {
                    self.compile_id(param)?;
                    param_values.push(self.rvalue(param));
                }

                // direct call?
                let call_inst = if let Some(Value::Func(func_id)) = self.ctx.values.get(&func) {
                    let func_ref = self
                        .ctx
                        .module
                        .declare_func_in_func(*func_id, self.builder.func);
                    self.builder.ins().call(func_ref, &param_values)
                } else {
                    let sig_ref = self
                        .ctx
                        .get_func_signature(func, param_ids.borrow().as_ref());
                    let sig = self.builder.import_signature(sig_ref);

                    self.as_cranelift_value(self.ctx.values[&func]);
                    let callee = self.rvalue(func);

                    self.builder.ins().call_indirect(sig, callee, &param_values)
                };

                if self
                    .ctx
                    .types
                    .get(&id)
                    .map(|t| !matches!(t, Type::Infer(_)))
                    .unwrap_or_default()
                {
                    let value = self.builder.func.dfg.first_result(call_inst);
                    self.ctx.values.insert(id, Value::Value(value));
                }

                Ok(())
            }
            Node::StructLiteral {
                fields,
                rearranged_fields,
                ..
            } => {
                let fields = rearranged_fields.unwrap_or(fields);

                let size = self.ctx.get_type_size(id);
                let slot = self.builder.create_sized_stack_slot(StackSlotData {
                    kind: StackSlotKind::ExplicitSlot,
                    size,
                });
                let addr =
                    self.builder
                        .ins()
                        .stack_addr(self.ctx.module.isa().pointer_type(), slot, 0);

                self.ctx.values.insert(id, Value::Value(addr));

                let mut offset = 0;
                for field in self.ctx.id_vecs[fields.0].clone().borrow().iter() {
                    self.compile_id(*field)?;
                    self.store_with_offset(*field, Value::Value(addr), offset);
                    offset += self.ctx.get_type_size(*field);
                }

                self.ctx.addressable_nodes.insert(id);

                Ok(())
            }
            Node::MemberAccess { value, member } => {
                self.compile_id(value)?;

                let (offset, mut pointiness) = self.get_member_offset(value, member)?;

                let mut value = self.as_cranelift_value(self.ctx.values[&value]);
                while pointiness > 0 {
                    value = self.load(types::I64, value, 0);
                    pointiness -= 1;
                }
                let value = self.builder.ins().iadd_imm(value, offset as i64);

                self.ctx.values.insert(id, Value::Value(value));

                Ok(())
            }
            Node::AddressOf(value) => {
                self.compile_id(value)?;

                let cranelift_value = self.ctx.values.get(&value).unwrap();
                match cranelift_value {
                    Value::Value(_) | Value::Func(_) => {
                        self.ctx.values.insert(id, *cranelift_value)
                    }
                    _ => todo!("{:?}", cranelift_value),
                };

                // if we are meant to be addressable, then we need to store our actual value into something which has an address
                if self.ctx.addressable_nodes.contains(&id) {
                    // todo(chad): is there any benefit to creating all of these up front?
                    let size = self.ctx.module.isa().pointer_bytes() as u32;
                    let slot = self.builder.create_sized_stack_slot(StackSlotData {
                        kind: StackSlotKind::ExplicitSlot,
                        size,
                    });
                    let slot_addr = self.builder.ins().stack_addr(
                        self.ctx.module.isa().pointer_type(),
                        slot,
                        0,
                    );
                    let value = Value::Value(slot_addr);
                    self.store_value(id, value, None);
                    self.ctx.values.insert(id, value);
                }

                Ok(())
            }
            Node::Deref(value) => {
                self.compile_id(value)?;

                let cranelift_value = self.ctx.values.get(&value).unwrap();
                let cranelift_value = match cranelift_value {
                    Value::Value(value) => value,
                    _ => todo!("load for {:?}", cranelift_value),
                };

                let ty = &self.ctx.get_type(id);
                let ty = match ty {
                    Type::Struct { .. } => self.ctx.module.isa().pointer_type(),
                    _ => self.ctx.get_cranelift_type(id),
                };

                let loaded = if self.ctx.addressable_nodes.contains(&value) {
                    self.load(types::I64, *cranelift_value, 0)
                } else {
                    self.load(ty, *cranelift_value, 0)
                };

                self.ctx.values.insert(id, Value::Value(loaded));

                Ok(())
            }
            Node::Extern {
                name,
                params,
                return_ty,
            } => {
                let mut sig = self.ctx.module.make_signature();

                let return_size = return_ty.map(|rt| self.ctx.get_type_size(rt)).unwrap_or(0);
                if return_size > 0 {
                    sig.returns.push(AbiParam::new(
                        self.ctx.get_cranelift_type(return_ty.unwrap()),
                    ));
                }

                for param in self.ctx.id_vecs[params.0].borrow().iter() {
                    sig.params
                        .push(AbiParam::new(self.ctx.get_cranelift_type(*param)));
                }

                let name = self.ctx.get_symbol(name);
                let name = self
                    .ctx
                    .string_interner
                    .resolve(name.0)
                    .unwrap()
                    .to_string();

                let func_id = self
                    .ctx
                    .module
                    .declare_function(&name, Linkage::Import, &sig)
                    .unwrap();

                self.ctx.values.insert(id, Value::Func(func_id));

                Ok(())
            }
            Node::Func { .. } => Ok(()), // This should have already been handled by the toplevel context
            _ => todo!("{:?}", &self.ctx.nodes[id.0]),
        }
    }

    fn load(&mut self, ty: types::Type, value: CraneliftValue, offset: i32) -> CraneliftValue {
        self.builder.ins().load(ty, MemFlags::new(), value, offset)
    }

    fn get_member_offset(
        &mut self,
        value: NodeId,
        member: NodeId,
    ) -> Result<(u32, u32), CompileError> {
        let member_name = self.ctx.nodes[member.0].as_symbol().unwrap();

        let mut ty = self.ctx.get_type(value);
        let mut pointiness = 0;
        while let Type::Pointer(inner) = ty {
            ty = self.ctx.get_type(inner);
            pointiness += 1;
        }

        let Type::Struct { fields, .. } = ty else {
            panic!("Not a struct");
        };

        let fields = self.ctx.id_vecs[fields.0].clone();

        let mut offset = 0;
        for field in fields.borrow().iter() {
            let field_name = match &self.ctx.nodes[field.0] {
                Node::StructDeclParam { name, .. } => *name,
                _ => panic!("Not a param"),
            };
            let field_name = self.ctx.nodes[field_name.0].as_symbol().unwrap();

            if field_name == member_name {
                return Ok((offset, pointiness));
            }

            offset += self.ctx.get_type_size(*field);
        }

        return Err(CompileError::Generic(
            "Member not found".to_string(),
            self.ctx.ranges[value.0],
        ));
    }

    fn as_cranelift_value(&mut self, value: Value) -> CraneliftValue {
        match value {
            Value::Value(val) => val,
            Value::Func(func_id) => {
                let func_ref = self
                    .ctx
                    .module
                    .declare_func_in_func(func_id, self.builder.func);

                self.builder
                    .ins()
                    .func_addr(self.ctx.get_pointer_type(), func_ref)
            }
            _ => panic!("not a cranelift value: {:?}", value),
        }
    }

    fn rvalue(&mut self, id: NodeId) -> CraneliftValue {
        let value = self.as_cranelift_value(self.ctx.values[&id]);

        let should_pass_by_ref = matches!(self.ctx.get_type(id), Type::Struct { .. });

        if self.ctx.addressable_nodes.contains(&id) && !should_pass_by_ref {
            let ty = self.ctx.get_cranelift_type(id);
            self.builder.ins().load(ty, MemFlags::new(), value, 0)
        } else {
            value
        }
    }

    fn store(&mut self, id: NodeId, dest: Value) {
        if self.ctx.addressable_nodes.contains(&id) {
            self.store_copy(id, dest);
        } else {
            self.store_value(id, dest, None);
        }
    }

    fn store_with_offset(&mut self, id: NodeId, dest: Value, offset: u32) {
        if self.ctx.addressable_nodes.contains(&id) {
            let dest = if offset == 0 {
                dest
            } else {
                let Value::Value(dest) = dest else { panic!("not a value") };
                let dest = self.builder.ins().iadd_imm(dest, offset as i64);
                Value::Value(dest)
            };

            self.store_copy(id, dest);
        } else {
            self.store_value(id, dest, Some(offset as _));
        }
    }

    fn store_copy(&mut self, id: NodeId, dest: Value) {
        let size = self.ctx.get_type_size(id);

        let source_value = self.as_cranelift_value(self.ctx.values[&id]);
        let dest_value = self.as_cranelift_value(dest);

        self.builder.emit_small_memory_copy(
            self.ctx.module.isa().frontend_config(),
            dest_value,
            source_value,
            size as _,
            1,
            1,
            true, // non-overlapping
            MemFlags::new(),
        );
    }

    fn store_value(&mut self, id: NodeId, dest: Value, offset: Option<i32>) {
        let source_value = match self.ctx.values[&id] {
            Value::Value(value) => value,
            Value::Func(func_id) => {
                let func_ref = self
                    .ctx
                    .module
                    .declare_func_in_func(func_id, self.builder.func);
                self.builder
                    .ins()
                    .func_addr(self.ctx.get_cranelift_type(id), func_ref)
            }
            a => todo!("store_value source for {:?}", a),
        };

        match dest {
            Value::Value(value) => {
                self.builder.ins().store(
                    MemFlags::new(),
                    source_value,
                    value,
                    offset.unwrap_or_default(),
                );
            }
            _ => todo!("store_value dest for {:?}", dest),
        }
    }
}

impl<'a> ToplevelCompileContext<'a> {
    pub fn compile_toplevel_id(&mut self, id: NodeId) -> Result<(), CompileError> {
        // idempotency
        match self.ctx.values.get(&id) {
            None | Some(Value::Unassigned | Value::Func(_)) => {}
            _ => return Ok(()),
        };

        match self.ctx.nodes[id.0] {
            Node::Symbol(_) => todo!(),
            Node::Func {
                name,
                stmts,
                params,
                return_ty,
                ..
            } => {
                let name_str = self.ctx.func_name(name, id);

                let mut sig = self.ctx.module.make_signature();

                for &param in self.ctx.id_vecs[params.0].borrow().iter() {
                    sig.params
                        .push(AbiParam::new(self.ctx.get_cranelift_type(param)));
                }

                if let Some(return_ty) = return_ty {
                    sig.returns
                        .push(AbiParam::new(self.ctx.get_cranelift_type(return_ty)));
                }

                let func_id = self
                    .ctx
                    .module
                    .declare_function(&name_str, Linkage::Export, &sig)
                    .unwrap();

                let mut builder = FunctionBuilder::new(&mut self.codegen_ctx.func, self.func_ctx);
                builder.func.signature = sig;

                self.ctx.values.insert(id, Value::Func(func_id));

                let ebb = builder.create_block();
                builder.append_block_params_for_function_params(ebb);
                builder.switch_to_block(ebb);

                let mut builder_ctx = FunctionCompileContext {
                    ctx: self.ctx,
                    builder,
                    current_block: ebb,
                };

                for &stmt in builder_ctx.ctx.id_vecs[stmts.0].clone().borrow().iter() {
                    builder_ctx.compile_id(stmt)?;
                }

                builder_ctx.builder.seal_all_blocks();
                builder_ctx.builder.finalize();

                if self.ctx.args.dump_ir {
                    println!("{}", self.codegen_ctx.func.display());
                }

                self.ctx
                    .module
                    .define_function(func_id, self.codegen_ctx)
                    .unwrap();

                self.ctx.module.clear_context(self.codegen_ctx);

                Ok(())
            }
            _ => todo!(),
        }
    }
}

pub fn print_i64(n: i64) {
    print!("{}\n", n);
}

pub fn print_f64(n: f64) {
    print!("{}\n", n);
}
