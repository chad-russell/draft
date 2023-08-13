use std::{cell::RefCell, rc::Rc};

use string_interner::symbol::SymbolU32;
use tracing::instrument;

use crate::{
    ArrayLen, AsCastStyle, CompileError, Context, DraftResult, IdVec, Node, NodeElse, NodeId,
    Source, SourceInfo, StaticStrSource, Type,
};

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash, PartialOrd, Ord)]
pub struct Sym(pub SymbolU32);

#[derive(Clone, Copy, PartialEq)]
pub struct Location {
    pub line: usize,
    pub col: usize,
}

impl Default for Location {
    fn default() -> Self {
        Self { line: 1, col: 1 }
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
    #[instrument(name = "Range::new", skip_all)]
    pub fn new(start: Location, end: Location, source_path: &'static str) -> Self {
        Self {
            start,
            end,
            source_path,
        }
    }

    #[instrument(name = "Range::spanning", skip_all)]
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

    #[instrument(name = "Range::contains", skip_all)]
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
}

impl std::fmt::Debug for Range {
    #[instrument(name = "Range::fmt", skip_all)]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // write!(f, "{:?}-{:?}", self.start, self.end)
        write!(
            f,
            "{}:{}:{}",
            self.source_path, self.start.line, self.start.col
        )
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
    LSquare,
    RSquare,
    Semicolon,
    DoubleColon,
    Colon,
    Comma,
    Dot,
    Arrow,
    Underscore,
    UnderscoreSymbol(Sym),
    EqEq,
    Neq,
    Gt,
    Lt,
    GtEq,
    LtEq,
    Eq,
    Thread,
    Atmark,
    Fn,
    Impl,
    Extern,
    Let,
    If,
    Else,
    For,
    While,
    In,
    As,
    Bool,
    I8,
    I16,
    I32,
    I64,
    String,
    U8,
    U16,
    U32,
    U64,
    F32,
    F64,
    True,
    False,
    Symbol(Sym),
    StringLiteral(Sym),
    IntegerLiteral(i64, NumericSpecification),
    FloatLiteral(f64, NumericSpecification),
    Plus,
    Dash,
    Star,
    Slash,
    And,
    Or,
    Return,
    Resolve,
    Struct,
    Enum,
    AddressOf,
    Bang,
    UnderscoreLCurly,
    Cast,
    SizeOf,
    Transparent,
    Hash,
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

#[instrument(skip_all)]
pub fn breaks_symbol(c: char) -> bool {
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

#[derive(PartialEq)]
pub enum DeclParamParseType {
    Fn,
    Struct,
    Enum,
}

pub struct Parser<'a, W: Source> {
    pub source_info: &'a mut SourceInfo<W>,
    pub ctx: &'a mut Context,
    pub is_polymorph_copying: bool,
    pub in_struct_decl: bool,
    pub in_enum_decl: bool,
    pub in_fn_params_decl: bool,
}

impl<'a, W: Source> Parser<'a, W> {
    #[instrument(skip_all)]
    pub fn from_source(context: &'a mut Context, source_info: &'a mut SourceInfo<W>) -> Self {
        Self {
            source_info,
            ctx: context,
            is_polymorph_copying: false,
            in_struct_decl: false,
            in_enum_decl: false,
            in_fn_params_decl: false,
        }
    }

    #[instrument(skip_all)]
    pub fn pop(&mut self) {
        self.source_info.eat_spaces();

        let start = self.source_info.loc;
        self.source_info.top = self.source_info.second;

        let mut str_lit = String::new();
        let str_lit_index = if self.source_info.source.char_count() > 1
            && self.source_info.source.starts_with("\"")
        {
            let mut escaped = false;
            let mut index = 1;
            while index < self.source_info.source.char_count() {
                let c = self.source_info.source.char_at(index).unwrap();
                if c == '"' && !escaped {
                    break;
                } else if c == '\\' {
                    escaped = true;
                } else {
                    if escaped {
                        match c {
                            'n' => str_lit.push('\n'),
                            '0' => str_lit.push('\0'),
                            _ => str_lit.push(c),
                        }
                    } else {
                        str_lit.push(c);
                    }
                    escaped = false;
                }
                index += 1;
            }

            index += 1; // skip the closing quote

            Some(index)
        } else {
            None
        };

        if let Some(str_lit_index) = str_lit_index {
            let str_lit_sym = Sym(self.ctx.string_interner.get_or_intern(str_lit));

            self.source_info.eat(str_lit_index);

            let end = self.source_info.loc;
            self.source_info.second = Lexeme::new(
                Token::StringLiteral(str_lit_sym),
                self.source_info.make_range(start, end),
            );

            return;
        }

        let is_underscore = if self.source_info.source.char_count() > 1 {
            self.source_info.source.starts_with("_")
                && !breaks_symbol(self.source_info.source.char_at(1).unwrap())
        } else {
            false
        };

        if is_underscore {
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
        if self.source_info.prefix_keyword("impl", Token::Impl) {
            return;
        }
        if self.source_info.prefix_keyword("extern", Token::Extern) {
            return;
        }
        if self.source_info.prefix_keyword("let", Token::Let) {
            return;
        }
        if self.source_info.prefix_keyword("if", Token::If) {
            return;
        }
        if self.source_info.prefix_keyword("else", Token::Else) {
            return;
        }
        if self.source_info.prefix_keyword("for", Token::For) {
            return;
        }
        if self.source_info.prefix_keyword("while", Token::While) {
            return;
        }
        if self.source_info.prefix_keyword("in", Token::In) {
            return;
        }
        if self.source_info.prefix_keyword("as", Token::As) {
            return;
        }
        if self.source_info.prefix_keyword("and", Token::And) {
            return;
        }
        if self.source_info.prefix_keyword("or", Token::Or) {
            return;
        }
        if self.source_info.prefix_keyword("return", Token::Return) {
            return;
        }
        if self.source_info.prefix_keyword("resolve", Token::Resolve) {
            return;
        }
        if self.source_info.prefix_keyword("struct", Token::Struct) {
            return;
        }
        if self.source_info.prefix_keyword("enum", Token::Enum) {
            return;
        }
        if self.source_info.prefix_keyword("bool", Token::Bool) {
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
        if self.source_info.prefix_keyword("string", Token::String) {
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
        if self.source_info.prefix_keyword("true", Token::True) {
            return;
        }
        if self.source_info.prefix_keyword("false", Token::False) {
            return;
        }
        if self.source_info.prefix_keyword("#cast", Token::Cast) {
            return;
        }
        if self.source_info.prefix_keyword("#size_of", Token::SizeOf) {
            return;
        }
        if self
            .source_info
            .prefix_keyword("#transparent", Token::Transparent)
        {
            return;
        }
        if self.source_info.prefix("#", Token::Hash) {
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
        if self.source_info.prefix("[", Token::LSquare) {
            return;
        }
        if self.source_info.prefix("]", Token::RSquare) {
            return;
        }
        if self.source_info.prefix(",", Token::Comma) {
            return;
        }
        if self.source_info.prefix(".", Token::Dot) {
            return;
        }
        if self.source_info.prefix("->", Token::Arrow) {
            return;
        }
        if self.source_info.prefix(";", Token::Semicolon) {
            return;
        }
        if self.source_info.prefix("::", Token::DoubleColon) {
            return;
        }
        if self.source_info.prefix(":", Token::Colon) {
            return;
        }
        if self.source_info.prefix("==", Token::EqEq) {
            return;
        }
        if self.source_info.prefix(">=", Token::GtEq) {
            return;
        }
        if self.source_info.prefix(">", Token::Gt) {
            return;
        }
        if self.source_info.prefix("<=", Token::LtEq) {
            return;
        }
        if self.source_info.prefix("<", Token::Lt) {
            return;
        }
        if self.source_info.prefix("!=", Token::Neq) {
            return;
        }
        if self.source_info.prefix("=", Token::Eq) {
            return;
        }
        if self.source_info.prefix("@", Token::Atmark) {
            return;
        }
        if self.source_info.prefix("|>", Token::Thread) {
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

    #[instrument(skip_all)]
    fn expect(&mut self, tok: Token) -> DraftResult<()> {
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

    #[instrument(skip_all)]
    fn expect_range(&mut self, start: Location, token: Token) -> DraftResult<Range> {
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

    #[instrument(skip_all)]
    pub fn parse(&mut self) -> DraftResult<()> {
        self.pop();
        self.pop();

        while self.source_info.top.tok != Token::Eof {
            let tl = self.parse_top_level()?;
            self.ctx.top_level.push(tl);
        }

        Ok(())
    }

    #[instrument(skip_all)]
    pub fn parse_top_level(&mut self) -> DraftResult<NodeId> {
        let tl = match self.source_info.top.tok {
            Token::Fn => self.parse_fn(false),
            Token::Extern => self.parse_extern(),
            Token::Struct => self.parse_struct_definition(),
            Token::Enum => self.parse_enum_definition(),
            _ => {
                let msg = format!("expected 'fn', found '{:?}'", self.source_info.top.tok);

                Err(CompileError::Generic(msg, self.source_info.top.range))
            }
        }?;

        Ok(tl)
    }

    #[instrument(skip_all)]
    fn parse_symbol(&mut self) -> DraftResult<NodeId> {
        let range = self.source_info.top.range;
        match self.source_info.top.tok {
            Token::Symbol(sym) => {
                self.pop();
                Ok(self.ctx.push_node(range, Node::Symbol(sym)))
            }
            a => Err(CompileError::Generic(
                format!("Expected symbol, got {:?}", a),
                range,
            )),
        }
    }

    #[instrument(skip_all)]
    pub fn parse_poly_specialize(&mut self, sym: Sym) -> DraftResult<NodeId> {
        let mut range = self.source_info.top.range;

        self.pop();
        if self.in_struct_decl || self.in_enum_decl || self.in_fn_params_decl {
            self.ctx.polymorph_target = true;
        }

        range.end = self.source_info.top.range.end;
        self.pop(); // `!`

        let mut overrides = Vec::new();

        if self.source_info.top.tok == Token::LParen {
            self.pop(); // `(`

            while self.source_info.top.tok != Token::RParen {
                let sym = self.parse_symbol()?;
                self.expect(Token::Colon)?;

                let ty = self.parse_type()?;

                overrides.push(self.ctx.push_node(
                    self.make_range_spanning(sym, ty),
                    Node::PolySpecializeOverride { sym, ty },
                ));

                if self.source_info.top.tok == Token::Comma {
                    self.pop(); // `,`
                } else {
                    break;
                }
            }

            range.end = self.source_info.top.range.end;
            self.expect(Token::RParen)?; // `)`
        }

        let overrides = self.ctx.push_id_vec(overrides);

        Ok(self.ctx.push_node(
            range,
            Node::PolySpecialize {
                sym,
                overrides,
                copied: None,
            },
        ))
    }

    #[instrument(skip_all)]
    pub fn parse_value_params(&mut self, is_threading_func_call: bool) -> DraftResult<IdVec> {
        let mut params = Vec::new();

        while self.source_info.top.tok != Token::RParen && self.source_info.top.tok != Token::RCurly
        {
            if let (Token::Symbol(_), Token::Colon) =
                (self.source_info.top.tok, self.source_info.second.tok)
            {
                let name = self.parse_symbol()?;
                self.expect(Token::Colon)?;
                let value = self.parse_expression(true)?;
                params.push(self.ctx.push_node(
                    self.make_range_spanning(name, value),
                    Node::ValueParam {
                        name: Some(name),
                        value,
                        index: params.len() as u16,
                    },
                ));
            } else if is_threading_func_call && self.source_info.top.tok == Token::Atmark {
                let range = self.source_info.top.range;
                self.pop(); // `@`
                params.push(self.ctx.push_node(range, Node::ThreadingParamTarget));
            } else {
                let value = self.parse_expression(true)?;
                params.push(self.ctx.push_node(
                    self.ctx.ranges[value],
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

    #[instrument(skip_all)]
    fn parse_decl_params(&mut self, parse_type: DeclParamParseType) -> DraftResult<IdVec> {
        let mut params = Vec::new();

        while self.source_info.top.tok != Token::RParen && self.source_info.top.tok != Token::RCurly
        {
            let input_start = self.source_info.top.range.start;

            // transparent?
            let transparent = if self.source_info.top.tok == Token::Transparent {
                self.pop(); // `#transparent`
                true
            } else {
                false
            };

            let name = self.parse_symbol()?;
            let name_sym = self.ctx.nodes[name].as_symbol().unwrap();

            let (ty, default) = if self.source_info.top.tok == Token::Colon {
                self.pop(); // `:`
                let ty = self.parse_type()?;

                let mut default = None;
                if self.source_info.top.tok == Token::Eq {
                    self.pop(); // `=`
                    default = Some(self.parse_expression(true)?);
                }

                (Some(ty), default)
            } else if self.source_info.top.tok == Token::Eq {
                self.pop(); // `=`
                let default = self.parse_expression(true)?;

                (None, Some(default))
            } else {
                (None, None)
            };

            let range_end = match (ty, default) {
                (_, Some(default)) => self.ctx.ranges[default].end,
                (Some(ty), _) => self.ctx.ranges[ty].end,
                _ => self.ctx.ranges[name].end,
            };

            let range = self.source_info.make_range(input_start, range_end);

            // put the param in scope
            let node = match parse_type {
                DeclParamParseType::Fn => Node::FnDeclParam {
                    name,
                    ty,
                    default,
                    index: params.len() as u16,
                    transparent,
                },
                DeclParamParseType::Struct => Node::StructDeclParam {
                    name,
                    ty,
                    default,
                    index: params.len() as u16,
                    transparent,
                },
                DeclParamParseType::Enum => {
                    if default.is_some() {
                        return Err(CompileError::Generic(
                            "enum parameters cannot have default values".to_string(),
                            range,
                        ));
                    }
                    Node::EnumDeclParam { name, ty }
                }
            };

            let param = self.ctx.push_node(range, node);
            self.ctx.scope_insert(name_sym, param);

            params.push(param);

            if self.source_info.top.tok != Token::RCurly
                && self.source_info.top.tok != Token::RParen
            {
                self.expect(Token::Comma)?;
            }
        }

        Ok(self.ctx.push_id_vec(params))
    }

    #[instrument(skip_all)]
    pub fn parse_struct_definition(&mut self) -> DraftResult<NodeId> {
        let start = self.source_info.top.range.start;

        self.pop(); // `struct`

        self.in_struct_decl = true;

        let old_polymorph_target = self.ctx.polymorph_target;
        self.ctx.polymorph_target = false;

        let name = self.parse_symbol()?;
        let name_sym = self.ctx.nodes[name].as_symbol().unwrap();

        let pushed_scope = self.ctx.push_scope();
        let struct_scope = self.ctx.top_scope;

        self.expect(Token::LCurly)?;
        let fields = self.parse_decl_params(DeclParamParseType::Struct)?;
        let range = self.expect_range(start, Token::RCurly)?;

        self.ctx.pop_scope(pushed_scope);

        let struct_node = self.ctx.push_node(
            range,
            Node::StructDefinition {
                name: Some(name),
                params: fields,
                scope: struct_scope,
            },
        );

        if !self.is_polymorph_copying {
            self.ctx.scope_insert(name_sym, struct_node);

            if self.ctx.polymorph_target {
                self.ctx.polymorph_sources.insert(struct_node, struct_node);
            }
        }

        self.ctx.polymorph_target = old_polymorph_target;
        self.in_struct_decl = false;

        Ok(struct_node)
    }

    #[instrument(skip_all)]
    pub fn parse_enum_definition(&mut self) -> DraftResult<NodeId> {
        let start = self.source_info.top.range.start;

        self.pop(); // `enum`

        self.in_enum_decl = true;

        let old_polymorph_target = self.ctx.polymorph_target;
        self.ctx.polymorph_target = false;

        let name = self.parse_symbol()?;
        let name_sym = self.ctx.nodes[name].as_symbol().unwrap();

        let pushed_scope = self.ctx.push_scope();

        self.expect(Token::LCurly)?;
        let fields = self.parse_decl_params(DeclParamParseType::Enum)?;
        let range = self.expect_range(start, Token::RCurly)?;

        self.ctx.pop_scope(pushed_scope);

        let enum_node = self.ctx.push_node(
            range,
            Node::EnumDefinition {
                name,
                params: fields,
            },
        );

        if !self.is_polymorph_copying {
            self.ctx.scope_insert(name_sym, enum_node);
        }

        if self.ctx.polymorph_target {
            self.ctx.polymorph_sources.insert(enum_node, enum_node);
        }

        self.ctx.polymorph_target = old_polymorph_target;
        self.in_enum_decl = false;

        Ok(enum_node)
    }

    #[instrument(skip_all)]
    pub fn parse_fn(&mut self, anonymous: bool) -> DraftResult<NodeId> {
        let old_polymorph_target = self.ctx.polymorph_target;
        self.ctx.polymorph_target = false;

        let start = self.source_info.top.range.start;

        self.pop(); // `fn`

        let name = if anonymous {
            None
        } else {
            Some(self.parse_symbol()?)
        };
        let name_sym = name.map(|n| self.ctx.nodes[n].as_symbol().unwrap());

        // open a new scope
        let pushed_scope = self.ctx.push_scope();

        self.in_fn_params_decl = true;
        self.expect(Token::LParen)?;
        let params = self.parse_decl_params(DeclParamParseType::Fn)?;
        self.expect(Token::RParen)?;
        self.in_fn_params_decl = false;

        let return_ty = if self.source_info.top.tok != Token::LCurly {
            Some(self.parse_type()?)
        } else {
            None
        };

        self.expect(Token::LCurly)?;

        self.ctx.returns.push(Vec::new());

        let mut stmts = Vec::new();
        while self.source_info.top.tok != Token::RCurly {
            let stmt = self.parse_block_stmt()?;
            stmts.push(stmt);
        }
        let stmts = self.ctx.push_id_vec(stmts);

        let range = self.expect_range(start, Token::RCurly)?;

        let returns = self.ctx.returns.pop().unwrap();
        let returns = self.ctx.push_id_vec(returns);
        let func = self.ctx.push_node(
            range,
            Node::FnDefinition {
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
            self.ctx.polymorph_sources.insert(func, func);
        }

        self.ctx.polymorph_target = old_polymorph_target;

        self.ctx.funcs.push(func);

        Ok(func)
    }

    #[instrument(skip_all)]
    pub fn parse_impl_fn(&mut self, for_id: NodeId) -> DraftResult<NodeId> {
        let start = self.source_info.top.range.start;

        self.pop(); // `fn`

        let name = self.parse_symbol()?;
        let name_sym = self.ctx.nodes[name].as_symbol().unwrap();

        // open a new scope
        let pushed_scope = self.ctx.push_scope();

        self.in_fn_params_decl = true;

        let self_param_range = self.source_info.top.range;
        let self_symbol = Sym(self.ctx.string_interner.get_or_intern("self"));
        let self_symbol_node = self
            .ctx
            .push_node(self_param_range, Node::Symbol(self_symbol));
        let self_ptr_ty = self
            .ctx
            .push_node(self_param_range, Node::Type(Type::Pointer(for_id)));
        let self_param = self.ctx.push_node(
            self_param_range,
            Node::FnDeclParam {
                name: self_symbol_node,
                ty: Some(self_ptr_ty),
                default: None,
                index: 0,
                transparent: false,
            },
        );
        self.ctx.scope_insert(self_symbol, self_param);

        self.expect(Token::LParen)?;
        let params = self.parse_decl_params(DeclParamParseType::Fn)?;

        let params_vec = params.clone();
        params_vec.borrow_mut().insert(0, self_param);

        for (i, param) in params_vec.borrow().iter().enumerate() {
            let Node::FnDeclParam { index, .. } = &mut self.ctx.nodes[param] else {
                unreachable!()
            };
            *index = i as u16;
        }

        self.expect(Token::RParen)?;
        self.in_fn_params_decl = false;

        let return_ty = if self.source_info.top.tok != Token::LCurly {
            Some(self.parse_type()?)
        } else {
            None
        };

        self.expect(Token::LCurly)?;

        self.ctx.returns.push(Vec::new());

        let mut stmts = Vec::new();
        while self.source_info.top.tok != Token::RCurly {
            let stmt = self.parse_block_stmt()?;
            stmts.push(stmt);
        }
        let stmts = self.ctx.push_id_vec(stmts);

        let range = self.expect_range(start, Token::RCurly)?;

        let returns = self.ctx.returns.pop().unwrap();
        let returns = self.ctx.push_id_vec(returns);
        let func = self.ctx.push_node(
            range,
            Node::FnDefinition {
                name: Some(name),
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
            self.ctx.scope_insert(name_sym, func);
        }

        self.ctx.funcs.push(func);

        Ok(func)
    }

    #[instrument(skip_all)]
    pub fn parse_extern(&mut self) -> DraftResult<NodeId> {
        let start = self.source_info.top.range.start;

        self.pop(); // `extern`

        let name = self.parse_symbol()?;
        let name_sym = self.ctx.nodes[name].as_symbol().unwrap();

        let pushed_scope = self.ctx.push_scope();

        self.expect(Token::LParen)?;
        let params = self.parse_decl_params(DeclParamParseType::Fn)?;
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

    #[instrument(skip_all)]
    pub fn parse_expression(&mut self, struct_literals_allowed: bool) -> DraftResult<NodeId> {
        let mut operators = Vec::<Op>::new();
        let mut output = Vec::new();

        let (mut parsing_op, mut parsing_expr) = (false, true);

        loop {
            let _debug_tok = self.source_info.top.tok;

            match self.source_info.top.tok {
                Token::IntegerLiteral(_, _)
                | Token::FloatLiteral(_, _)
                | Token::StringLiteral(_)
                | Token::LCurly
                | Token::LParen
                | Token::LSquare
                | Token::Symbol(_)
                | Token::UnderscoreLCurly
                | Token::AddressOf
                | Token::Cast
                | Token::SizeOf
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
                | Token::F64
                | Token::True
                | Token::False
                | Token::If => {
                    if !parsing_expr {
                        break;
                    }

                    let id = self.parse_expression_piece(struct_literals_allowed, false)?;
                    output.push(Shunting::Id(id))
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

                        let expr = self.parse_expression_piece(struct_literals_allowed, false)?;
                        let id = self.ctx.push_node(
                            Range::new(start, self.ctx.ranges[expr].end, self.source_info.path),
                            Node::Deref(expr),
                        );

                        output.push(Shunting::Id(id))
                    }
                }
                Token::Thread => {
                    if !parsing_op {
                        break;
                    }

                    let Some(Shunting::Id(value)) = output.pop() else {
                        self.ctx.errors.push(CompileError::Generic(
                            "Expected value before `->`".to_string(),
                            self.source_info.top.range,
                        ));
                        break;
                    };

                    let mut values = vec![value];

                    // Threading function call?
                    while self.source_info.top.tok == Token::Thread {
                        self.pop(); // `|>`
                        let value = self.parse_expression_piece(false, true)?;
                        values.push(value);
                    }

                    let unrolled = self.unroll_threading_call(&mut values);

                    output.push(Shunting::Id(unrolled));
                }
                Token::Plus
                | Token::Dash
                | Token::Slash
                | Token::EqEq
                | Token::Neq
                | Token::Gt
                | Token::Lt
                | Token::GtEq
                | Token::LtEq
                | Token::And
                | Token::Or => {
                    if !parsing_op {
                        break;
                    }

                    let op = Op::from(self.source_info.top.tok);

                    while !operators.is_empty()
                        && operators.last().unwrap().precedence() >= op.precedence()
                    {
                        output.push(Shunting::Op(operators.pop().unwrap()));
                    }
                    operators.push(op);

                    self.pop(); // op
                }
                _ => break,
            }

            std::mem::swap(&mut parsing_op, &mut parsing_expr);
        }

        while !operators.is_empty() {
            output.push(Shunting::Op(operators.pop().unwrap()));
        }

        if output.len() == 1 {
            let Shunting::Id(id) = output[0] else {
                unreachable!()
            };
            return Ok(id);
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

    #[instrument(skip_all)]
    pub fn unroll_threading_call(&mut self, values: &mut Vec<NodeId>) -> NodeId {
        let inner_func_id = values.pop().unwrap();
        let param = if values.len() > 1 {
            self.unroll_threading_call(values)
        } else if values.len() == 1 {
            // for a single param, turn it into a value param node
            values.pop().unwrap()
        } else {
            panic!()
        };

        let param = self.ctx.push_node(
            self.ctx.ranges[param],
            Node::ValueParam {
                name: None,
                value: param,
                index: 0,
            },
        );

        let value_param_id = self.ctx.push_node(
            self.ctx.ranges[param],
            Node::ValueParam {
                name: None,
                value: param,
                index: 0,
            },
        );

        let (inner_func_id, params) = match self.ctx.nodes[inner_func_id].clone() {
            Node::Call { func, params } | Node::ThreadingCall { func, params } => {
                let params_vec = params.clone();

                // If one of the params is a threading target, replace it with the param in question
                // If more than one of the params is a threading target, that's an error
                // If none are a threading target, insert the param at position 0

                let mut threading_target = None;
                for (pid, param) in params_vec.borrow().iter().enumerate() {
                    match &mut self.ctx.nodes[param] {
                        Node::ThreadingParamTarget => {
                            if threading_target.is_some() {
                                self.ctx.errors.push(CompileError::Node("Multiple threading param targets specified - can have at most one in a threading call".to_string(), *param));
                            }
                            threading_target = Some(pid);
                        }
                        _ => (),
                    }
                }

                if let Some(tt) = threading_target {
                    params_vec.borrow_mut()[tt] = value_param_id;
                } else {
                    params_vec.borrow_mut().insert(0, value_param_id);

                    for (pid, param) in params_vec.borrow().iter().enumerate() {
                        match &mut self.ctx.nodes[param] {
                            Node::ValueParam { index, .. } => {
                                *index = pid as u16;
                            }
                            a => panic!("Expected ValueParam, got {}", a.ty()),
                        }
                    }
                }

                (func, params)
            }
            _ => (inner_func_id, self.ctx.push_id_vec(vec![param])),
        };

        self.ctx.push_node(
            Range::new(
                self.ctx.ranges[param].start,
                self.ctx.ranges[inner_func_id].end,
                self.source_info.path,
            ),
            Node::ThreadingCall {
                func: inner_func_id,
                params,
            },
        )
    }

    #[instrument(skip_all)]
    pub fn parse_expression_piece(
        &mut self,
        struct_literals_allowed: bool,
        is_threading_func_call: bool,
    ) -> DraftResult<NodeId> {
        let start = self.source_info.top.range.start;

        let mut value = match self.source_info.top.tok {
            Token::True => {
                self.pop(); // `true`
                let id = self
                    .ctx
                    .push_node(self.source_info.top.range, Node::BoolLiteral(true));
                Ok(id)
            }
            Token::False => {
                self.pop(); // `false`
                let id = self
                    .ctx
                    .push_node(self.source_info.top.range, Node::BoolLiteral(false));
                Ok(id)
            }
            Token::If => self.parse_if(),
            Token::IntegerLiteral(_, _) | Token::FloatLiteral(_, _) => self.parse_numeric_literal(),
            Token::StringLiteral(sym) => {
                self.pop();
                let id = self
                    .ctx
                    .push_node(self.source_info.top.range, Node::StringLiteral(sym));
                self.ctx.string_literals.push(id);
                Ok(id)
            }
            Token::Fn => self.parse_fn(true),
            Token::Star => {
                self.pop(); // `*`

                let expr = self.parse_expression_piece(true, false)?;
                let id = self.ctx.push_node(
                    Range::new(start, self.ctx.ranges[expr].end, self.source_info.path),
                    Node::Deref(expr),
                );

                Ok(id)
            }
            Token::AddressOf => {
                self.pop(); // `&`

                let expr = self.parse_expression_piece(true, false)?;

                let end = self.ctx.ranges[expr].end;
                let id = self.ctx.push_node(
                    Range::new(start, end, self.source_info.path),
                    Node::AddressOf(expr),
                );

                Ok(id)
            }
            Token::LParen => {
                self.pop(); // `(`
                let expr = self.parse_expression(true)?;
                self.expect(Token::RParen)?;

                Ok(expr)
            }
            Token::LSquare => {
                let start = self.source_info.top.range.start;

                self.pop(); // `[`
                let mut exprs = Vec::new();
                while self.source_info.top.tok != Token::RSquare {
                    let expr = self.parse_expression(true)?;
                    exprs.push(expr);

                    if self.source_info.top.tok == Token::Comma {
                        self.pop(); // `,`
                    } else {
                        break;
                    }
                }

                let members = self.ctx.push_id_vec(exprs);

                let range = self.expect_range(start, Token::RSquare)?;

                let ty = self.ctx.push_node(range, Node::Type(Type::Infer(None)));

                let id = self
                    .ctx
                    .push_node(range, Node::ArrayLiteral { members, ty });

                Ok(id)
            }
            Token::Cast => {
                let cast_range = self.source_info.top.range;

                self.pop(); // `#cast`

                let ty = if self.source_info.top.tok == Token::LParen {
                    self.pop(); // `(`
                    let ty = self.parse_type()?;
                    self.expect(Token::RParen)?;
                    ty
                } else {
                    self.ctx
                        .push_node(cast_range, Node::Type(Type::Infer(None)))
                };

                let value = self.parse_expression(true)?;

                let end = self.ctx.ranges[value].end;
                let id = self.ctx.push_node(
                    Range::new(start, end, self.source_info.path),
                    Node::Cast { ty, value },
                );

                Ok(id)
            }
            Token::SizeOf => {
                self.pop(); // `#sizeof`

                self.expect(Token::LParen)?;
                let ty = self.parse_type()?;
                let range = self.expect_range(start, Token::RParen)?;

                let id = self.ctx.push_node(range, Node::SizeOf(ty));

                Ok(id)
            }
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
            Token::UnderscoreLCurly => self.parse_struct_literal(),
            Token::LCurly => self.parse_block(true),
            Token::Symbol(_)
                if struct_literals_allowed && self.source_info.second.tok == Token::LCurly =>
            {
                self.parse_struct_literal()
            }
            Token::Symbol(sym) if self.source_info.second.tok == Token::Bang => {
                self.parse_poly_specialize(sym)
            }
            Token::Symbol(_) => self.parse_symbol(),
            _ => Err(CompileError::Generic(
                "Could not parse lvalue".to_string(),
                self.source_info.top.range,
            )),
        }?;

        while let Token::LParen
        | Token::LSquare
        | Token::Dot
        | Token::Arrow
        | Token::DoubleColon
        | Token::As = self.source_info.top.tok
        {
            // function call?
            while self.source_info.top.tok == Token::LParen {
                self.pop(); // `(`
                let params = self.parse_value_params(is_threading_func_call)?;
                let end = self.expect_range(start, Token::RParen)?.end;
                value = self.ctx.push_node(
                    Range::new(start, end, self.source_info.path),
                    Node::Call {
                        func: value,
                        params,
                    },
                );
            }

            // array access?
            while self.source_info.top.tok == Token::LSquare {
                self.pop(); // `[`

                let index = self.parse_expression(true)?;
                let end = self.expect_range(start, Token::RSquare)?.end;

                value = self.ctx.push_node(
                    Range::new(start, end, self.source_info.path),
                    Node::ArrayAccess {
                        array: value,
                        index,
                    },
                );
            }

            // member access?
            while self.source_info.top.tok == Token::Dot {
                self.pop(); // `.`

                let member = self.parse_symbol()?;
                let end = self.ctx.ranges[member].end;

                value = self.ctx.push_node(
                    Range::new(start, end, self.source_info.path),
                    Node::MemberAccess { value, member },
                );
            }

            // static member access?
            while self.source_info.top.tok == Token::DoubleColon {
                self.pop(); // `::`
                let member = self.parse_symbol()?;
                let end = self.ctx.ranges[member].end;
                value = self.ctx.push_node(
                    Range::new(start, end, self.source_info.path),
                    Node::StaticMemberAccess {
                        value,
                        member,
                        resolved: None,
                    },
                );
            }

            // as cast?
            if self.source_info.top.tok == Token::As {
                self.pop(); // `as`
                let ty = self.parse_type()?;
                let end = self.ctx.ranges[ty].end;
                value = self.ctx.push_node(
                    Range::new(start, end, self.source_info.path),
                    Node::AsCast {
                        value,
                        ty,
                        style: AsCastStyle::None,
                    },
                );
            }
        }

        Ok(value)
    }

    #[instrument(skip_all)]
    fn parse_struct_literal(&mut self) -> DraftResult<NodeId> {
        let start = self.source_info.top.range.start;

        let name = if self.source_info.top.tok == Token::UnderscoreLCurly {
            self.pop(); // `_{`
            None
        } else {
            let sym = self.parse_symbol()?;
            self.expect(Token::LCurly)?;
            Some(sym)
        };

        let params = self.parse_value_params(false)?;
        let range = self.expect_range(start, Token::RCurly)?;

        let struct_node = self
            .ctx
            .push_node(range, Node::StructLiteral { name, params });

        Ok(struct_node)
    }

    #[instrument(skip_all)]
    fn parse_numeric_literal(&mut self) -> DraftResult<NodeId> {
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

    #[instrument(skip_all)]
    pub fn parse_let(&mut self) -> DraftResult<NodeId> {
        let start = self.source_info.top.range.start;

        self.pop(); // `let`

        let transparent = self.source_info.top.tok == Token::Transparent;
        if transparent {
            self.pop(); // `#transparent`
        }

        let name = self.parse_symbol()?;
        let name_sym = self.ctx.nodes[name].as_symbol().unwrap();

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
                Some(self.parse_expression(true)?)
            }
            _ => {
                return Err(CompileError::Generic(
                    "Expected `;` or `=`".to_string(),
                    self.source_info.top.range,
                ));
            }
        };
        let range = self.expect_range(start, Token::Semicolon)?;
        let let_id = self.ctx.push_node(
            range,
            Node::Let {
                name,
                ty,
                expr,
                transparent,
            },
        );

        self.ctx.scope_insert(name_sym, let_id);

        Ok(let_id)
    }

    #[instrument(skip_all)]
    pub fn parse_block(&mut self, is_standalone: bool) -> DraftResult<NodeId> {
        let start = self.source_info.top.range.start;

        self.expect(Token::LCurly)?;

        let pushed_scope = self.ctx.push_scope();

        self.ctx.resolves.push(Vec::new());

        let mut stmts = Vec::new();
        while self.source_info.top.tok != Token::RCurly {
            let stmt = self.parse_block_stmt()?;
            stmts.push(stmt);
        }

        let range = self.expect_range(start, Token::RCurly)?;

        self.ctx.pop_scope(pushed_scope);

        let stmts = self.ctx.push_id_vec(stmts);
        let resolves = self.ctx.resolves.pop().unwrap();
        let resolves = self.ctx.push_id_vec(resolves);

        let block_id = self.ctx.push_node(
            range,
            Node::Block {
                stmts,
                resolves,
                is_standalone,
            },
        );

        Ok(block_id)
    }

    #[instrument(skip_all)]
    pub fn parse_if(&mut self) -> DraftResult<NodeId> {
        let start = self.source_info.top.range.start;

        self.pop(); // `if`

        let cond = self.parse_expression(false)?;

        let then_block = self.parse_block(false)?;

        let (else_block, end) = if self.source_info.top.tok == Token::Else {
            self.pop(); // `else`

            if self.source_info.top.tok == Token::LCurly {
                let else_block = self.parse_block(false)?;
                (NodeElse::Block(else_block), self.ctx.ranges[else_block].end)
            } else if self.source_info.top.tok == Token::If {
                let else_if = self.parse_if()?;
                (NodeElse::If(else_if), self.ctx.ranges[else_if].end)
            } else {
                return Err(CompileError::Generic(
                    "Expected `if` or `{`".to_string(),
                    self.source_info.top.range,
                ));
            }
        } else {
            (NodeElse::None, self.ctx.ranges[then_block].end)
        };

        Ok(self.ctx.push_node(
            Range::new(start, end, self.source_info.path),
            Node::If {
                cond,
                then_block,
                else_block,
            },
        ))
    }

    #[instrument(skip_all)]
    pub fn parse_for(&mut self) -> DraftResult<NodeId> {
        let start = self.source_info.top.range.start;

        self.pop(); // `for`

        let pushed_label_scope = self.ctx.push_scope();

        let label = self.parse_symbol()?;
        let name_sym = self.ctx.get_symbol(label);
        self.ctx.scope_insert(name_sym, label);

        self.expect(Token::In)?;

        let iterable = self.parse_expression(false)?;

        let block = self.parse_block(false)?;

        self.ctx.pop_scope(pushed_label_scope);

        Ok(self.ctx.push_node(
            Range::new(start, self.ctx.ranges[block].end, self.source_info.path),
            Node::For {
                label,
                iterable,
                block,
            },
        ))
    }

    #[instrument(skip_all)]
    pub fn parse_while(&mut self) -> DraftResult<NodeId> {
        let start = self.source_info.top.range.start;

        self.pop(); // `while`

        let pushed_label_scope = self.ctx.push_scope();

        let cond = self.parse_expression(false)?;

        let block = self.parse_block(false)?;

        self.ctx.pop_scope(pushed_label_scope);

        Ok(self.ctx.push_node(
            Range::new(start, self.ctx.ranges[block].end, self.source_info.path),
            Node::While { cond, block },
        ))
    }

    #[instrument(skip_all)]
    pub fn parse_block_stmt(&mut self) -> DraftResult<NodeId> {
        let start = self.source_info.top.range.start;

        let r = match self.source_info.top.tok {
            Token::Return => {
                self.pop(); // `return`

                let expr = if self.source_info.top.tok != Token::Semicolon {
                    Some(self.parse_expression(true)?)
                } else {
                    None
                };

                let range = self.expect_range(start, Token::Semicolon)?;

                let ret_id = self.ctx.push_node(range, Node::Return(expr));
                self.ctx.returns.last_mut().unwrap().push(ret_id);
                Ok(ret_id)
            }
            Token::Resolve => {
                self.pop(); // `resolve`

                let expr = if self.source_info.top.tok != Token::Semicolon {
                    Some(self.parse_expression(true)?)
                } else {
                    None
                };

                let range = self.expect_range(start, Token::Semicolon)?;

                let ret_id = self.ctx.push_node(range, Node::Resolve(expr));
                self.ctx.resolves.last_mut().unwrap().push(ret_id);
                Ok(ret_id)
            }
            Token::Let => self.parse_let(),
            Token::If => self.parse_if(),
            Token::For => self.parse_for(),
            Token::While => self.parse_while(),
            Token::Struct => self.parse_struct_definition(),
            Token::Enum => self.parse_enum_definition(),
            Token::Fn => self.parse_fn(false),
            _ => {
                let lvalue = self.parse_expression(true)?;

                match self.source_info.top.tok {
                    // Assignment?
                    Token::Eq => {
                        // parsing something like "foo = expr;";
                        self.expect(Token::Eq)?;
                        let expr = self.parse_expression(true)?;
                        let range = self.expect_range(start, Token::Semicolon)?;

                        Ok(self.ctx.push_node(
                            range,
                            Node::Assign {
                                name: lvalue,
                                expr,
                                is_store: false,
                            },
                        ))
                    }
                    _ => {
                        self.ctx.ranges[lvalue] = self.expect_range(start, Token::Semicolon)?;
                        Ok(lvalue)
                    }
                }
            }
        };

        r
    }

    #[instrument(skip_all)]
    fn parse_type(&mut self) -> DraftResult<NodeId> {
        match self.source_info.top.tok {
            Token::Bool => {
                let range = self.source_info.top.range;
                self.pop();
                Ok(self.ctx.push_node(range, Node::Type(Type::Bool)))
            }
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
            Token::String => {
                let range = self.source_info.top.range;
                self.pop();
                Ok(self.ctx.push_node(range, Node::Type(Type::String)))
            }
            Token::Fn => {
                let range = self.source_info.top.range;
                self.pop(); // `fn`

                self.expect(Token::LParen)?;
                let params = self.parse_decl_params(DeclParamParseType::Fn)?;
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

                let pushed_scope = self.ctx.push_scope();

                let params = self.parse_decl_params(DeclParamParseType::Struct)?;
                let range = self.expect_range(range.start, Token::RCurly)?;

                self.ctx.pop_scope(pushed_scope);

                let id = self.ctx.push_node(
                    range,
                    Node::Type(Type::Struct {
                        name: None,
                        params,
                        decl: None,
                    }),
                );

                Ok(id)
            }
            Token::Enum => {
                let range = self.source_info.top.range;
                self.pop(); // `enum`

                self.expect(Token::LCurly)?;

                let pushed_scope = self.ctx.push_scope();

                let params = self.parse_decl_params(DeclParamParseType::Enum)?;
                let range = self.expect_range(range.start, Token::RCurly)?;

                self.ctx.pop_scope(pushed_scope);

                Ok(self
                    .ctx
                    .push_node(range, Node::Type(Type::Enum { name: None, params })))
            }
            Token::LSquare => {
                let start = self.source_info.top.range.start;
                self.pop(); // `[`
                let len = if let Token::IntegerLiteral(len, _) = self.source_info.top.tok {
                    self.pop();
                    ArrayLen::Some(len as usize)
                } else if let Token::Underscore = self.source_info.top.tok {
                    self.pop();
                    ArrayLen::Infer
                } else {
                    ArrayLen::None
                };
                self.expect(Token::RSquare)?; // `]`
                let ty = self.parse_type()?;
                let range = Range::new(start, self.ctx.ranges[ty].end, self.source_info.path);
                Ok(self.ctx.push_node(range, Node::Type(Type::Array(ty, len))))
            }
            Token::Star => {
                let mut range = self.source_info.top.range;
                self.pop(); // `*`
                let ty = self.parse_type()?;
                range.end = self.ctx.ranges[ty].end;
                Ok(self.ctx.push_node(range, Node::Type(Type::Pointer(ty))))
            }
            Token::Underscore => {
                let mut range = self.source_info.top.range;
                self.pop(); // `_`

                if self.source_info.top.tok == Token::Bang {
                    range.end = self.source_info.top.range.end;
                    self.pop(); // `!`
                    self.ctx.polymorph_target = true;
                }

                Ok(self.ctx.push_node(range, Node::Type(Type::Infer(None))))
            }
            Token::Bang => {
                let range = self.source_info.top.range;
                self.pop(); // `!`
                self.ctx.polymorph_target = true;
                Ok(self.ctx.push_node(range, Node::Type(Type::Infer(None))))
            }
            Token::UnderscoreSymbol(sym) => {
                let mut range = self.source_info.top.range;
                self.pop(); // `_T`

                if self.source_info.top.tok == Token::Bang {
                    range.end = self.source_info.top.range.end;
                    self.pop(); // `!`
                    self.ctx.polymorph_target = true;
                }

                let id = self
                    .ctx
                    .push_node(range, Node::Type(Type::Infer(Some(sym))));
                self.ctx.scope_insert(sym, id);
                Ok(id)
            }
            Token::Symbol(sym) => {
                let range = self.source_info.top.range;
                if self.source_info.second.tok == Token::Bang {
                    self.parse_poly_specialize(sym)
                } else {
                    self.pop();
                    Ok(self.ctx.push_node(range, Node::Symbol(sym)))
                }
            }
            _ => Err(CompileError::Generic(
                "Expected type".to_string(),
                self.source_info.top.range,
            )),
        }
    }

    #[instrument(skip_all)]
    fn shunting_unroll(
        &mut self,
        output: &mut Vec<Shunting>,
        err_node_id: NodeId,
    ) -> DraftResult<NodeId> {
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

                Ok(value)
            }
        }
    }

    #[instrument(skip_all)]
    pub fn make_range_spanning(&self, start: NodeId, end: NodeId) -> Range {
        let start = self.ctx.ranges[start];
        let end = self.ctx.ranges[end];
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

impl Context {
    #[instrument(skip_all)]
    pub fn parse_file(&mut self, file_name: &str) -> DraftResult<()> {
        let mut source = self.make_source_info_from_file(file_name);
        let mut parser = Parser::from_source(self, &mut source);

        parser.parse()
    }

    #[instrument(skip_all)]
    pub fn ropey_parse_file(&mut self, file_name: &str) -> DraftResult<()> {
        let mut source = self.make_ropey_source_info_from_file(file_name);
        let mut parser = Parser::from_source(self, &mut source);
        parser.parse()
    }

    #[instrument(skip_all)]
    pub fn parse_str(&mut self, source: &'static str) -> DraftResult<()> {
        let mut source = SourceInfo::<StaticStrSource>::from_static_str(source);
        let mut parser = Parser::from_source(self, &mut source);
        parser.parse()
    }

    #[instrument(skip_all)]
    pub fn parse_source<W: Source>(&mut self, source: &mut SourceInfo<W>) -> DraftResult<()> {
        let mut parser = Parser::<W>::from_source(self, source);
        parser.parse()
    }

    #[instrument(skip_all)]
    pub fn push_node(&mut self, range: Range, node: Node) -> NodeId {
        self.nodes.push(node);
        self.ranges.push(range);
        self.node_scopes.push(self.top_scope);

        NodeId(self.nodes.len() - 1)
    }

    #[instrument(skip_all)]
    pub fn push_id_vec(&mut self, vec: Vec<NodeId>) -> IdVec {
        Rc::new(RefCell::new(vec))
    }

    pub fn debug_tokens<W: Source>(&mut self, source: &mut SourceInfo<W>) -> DraftResult<()> {
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

    #[instrument(skip_all)]
    pub fn polymorph_copy(&mut self, id: NodeId, target: ParseTarget) -> DraftResult<NodeId> {
        // Re-parse the region of the source code that contains the id
        let range = self.ranges[id];

        let mut source = self.make_source_info_from_range(range);

        let mut parser = Parser::from_source(self, &mut source);
        parser.is_polymorph_copying = true;
        parser.pop();
        parser.pop();

        let copied = match target {
            ParseTarget::FnDefinition => parser.parse_fn(false)?,
            ParseTarget::StructDefinition => {
                let parsed = parser.parse_struct_definition()?;

                // if the struct has generic params, we need to copy those too
                let Node::StructDefinition { params, .. } = self.nodes[parsed].clone() else {
                    panic!()
                };
                for param in params.borrow().iter() {
                    let Node::StructDeclParam { ty, .. } = self.nodes[param] else {
                        panic!()
                    };
                    if let Some(ty) = ty {
                        if let Node::Symbol(_) = self.nodes[ty] {
                            let copied = self.copy_polymorph_if_needed(ty)?;
                            self.nodes[ty] = self.nodes[copied].clone();
                        }
                    }
                }

                parsed
            }
            ParseTarget::EnumDefinition => parser.parse_enum_definition()?,
            ParseTarget::Type => parser.parse_type()?,
        };

        self.polymorph_sources.remove(&copied);
        self.polymorph_copies.insert(copied);
        Ok(copied)
    }
}

pub enum ParseTarget {
    FnDefinition,
    StructDefinition,
    EnumDefinition,
    Type,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Op {
    Add,
    Sub,
    Mul,
    Div,
    EqEq,
    Neq,
    Gt,
    Lt,
    GtEq,
    LtEq,
    And,
    Or,
}

impl Op {
    fn precedence(self) -> u8 {
        match self {
            Op::And | Op::Or | Op::Add | Op::Sub => 1,
            Op::Mul | Op::Div => 2,
            Op::Gt | Op::Lt | Op::GtEq | Op::LtEq | Op::EqEq | Op::Neq => 3,
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
            Token::EqEq => Op::EqEq,
            Token::Neq => Op::Neq,
            Token::Gt => Op::Gt,
            Token::Lt => Op::Lt,
            Token::GtEq => Op::GtEq,
            Token::LtEq => Op::LtEq,
            Token::And => Op::And,
            Token::Or => Op::Or,
            _ => unreachable!(),
        }
    }
}
