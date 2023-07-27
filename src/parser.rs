use std::{cell::RefCell, rc::Rc};

use string_interner::symbol::SymbolU32;

use crate::{
    CompileError, Context, IdVec, Node, NodeElse, NodeId, Source, SourceInfo, StaticStrSource, Type,
};

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash, PartialOrd, Ord)]
pub struct Sym(pub SymbolU32);

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
    Underscore,
    UnderscoreSymbol(Sym),
    EqEq,
    Eq,
    Fn,
    Extern,
    Let,
    If,
    Else,
    For,
    In,
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
    True,
    False,
    Symbol(Sym),
    IntegerLiteral(i64, NumericSpecification),
    FloatLiteral(f64, NumericSpecification),
    Plus,
    Dash,
    Star,
    Slash,
    Return,
    Resolve,
    Struct,
    Enum,
    AddressOf,
    Bang,
    UnderscoreLCurly,
    Cast,
    SizeOf,
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

    pub fn pop(&mut self) {
        self.source_info.eat_spaces();

        let start = self.source_info.loc;
        self.source_info.top = self.source_info.second;

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
        if self.source_info.prefix_keyword("in", Token::In) {
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
            Token::Enum => Ok(self.parse_enum_definition()?),
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

    fn parse_decl_params(&mut self, parse_type: DeclParamParseType) -> Result<IdVec, CompileError> {
        let mut params = Vec::new();

        while self.source_info.top.tok != Token::RParen && self.source_info.top.tok != Token::RCurly
        {
            let input_start = self.source_info.top.range.start;

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
                todo!("parse default parameters")
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
                },
                DeclParamParseType::Struct => Node::StructDeclParam {
                    name,
                    ty,
                    default,
                    index: params.len() as u16,
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

            if parse_type == DeclParamParseType::Struct {
                self.ctx.addressable_nodes.insert(param);
            }

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

        self.in_struct_decl = true;

        let old_polymorph_target = self.ctx.polymorph_target;
        self.ctx.polymorph_target = false;

        let name = self.parse_symbol()?;
        let name_sym = self.ctx.nodes[name].as_symbol().unwrap();

        let name_str = self.ctx.string_interner.resolve(name_sym.0).unwrap();
        let is_array_decl = name_str == "Array";

        let pushed_scope = self.ctx.push_scope();
        let struct_scope = self.ctx.top_scope;

        self.expect(Token::LCurly)?;
        let fields = self.parse_decl_params(DeclParamParseType::Struct)?;
        let range = self.expect_range(start, Token::RCurly)?;

        self.ctx.pop_scope(pushed_scope);

        let struct_node = self.ctx.push_node(
            range,
            Node::StructDefinition {
                name,
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

        if is_array_decl {
            self.ctx.array_declaration = Some(struct_node);
        }

        Ok(struct_node)
    }

    pub fn parse_enum_definition(&mut self) -> Result<NodeId, CompileError> {
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

    pub fn parse_extern(&mut self) -> Result<NodeId, CompileError> {
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

    pub fn parse_expression(
        &mut self,
        struct_literals_allowed: bool,
    ) -> Result<NodeId, CompileError> {
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

                    let id = self.parse_expression_piece(struct_literals_allowed)?;
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

                        let expr = self.parse_expression_piece(struct_literals_allowed)?;
                        let id = self.ctx.push_node(
                            Range::new(start, self.ctx.ranges[expr].end, self.source_info.path),
                            Node::Deref(expr),
                        );

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
                Token::EqEq => {
                    if !parsing_op {
                        break;
                    }

                    while !operators.is_empty()
                        && operators.last().unwrap().precedence()
                            >= Op::from(self.source_info.top.tok).precedence()
                    {
                        output.push(Shunting::Op(operators.pop().unwrap()));
                    }
                    operators.push(Op::EqEq);

                    self.pop(); // `==`
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

    pub fn parse_expression_piece(
        &mut self,
        struct_literals_allowed: bool,
    ) -> Result<NodeId, CompileError> {
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
            Token::Fn => self.parse_fn(true),
            Token::Star => {
                self.pop(); // `*`

                let expr = self.parse_expression_piece(true)?;
                let id = self.ctx.push_node(
                    Range::new(start, self.ctx.ranges[expr].end, self.source_info.path),
                    Node::Deref(expr),
                );

                Ok(id)
            }
            Token::AddressOf => {
                self.pop(); // `&`

                let expr = self.parse_expression_piece(true)?;
                self.ctx.addressable_nodes.insert(expr);

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

                self.ctx.addressable_nodes.insert(id);

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
            Token::UnderscoreLCurly => Ok(self.parse_struct_literal()?),
            Token::LCurly => Ok(self.parse_block(true)?),
            Token::Symbol(_) => {
                if struct_literals_allowed && self.source_info.second.tok == Token::LCurly {
                    Ok(self.parse_struct_literal()?)
                } else {
                    Ok(self.parse_symbol()?)
                }
            }
            _ => Err(CompileError::Generic(
                "Could not parse lvalue".to_string(),
                self.source_info.top.range,
            )),
        }?;

        while let Token::LParen | Token::LSquare | Token::Dot | Token::DoubleColon =
            self.source_info.top.tok
        {
            // function call?
            while self.source_info.top.tok == Token::LParen {
                self.pop(); // `(`
                let params = self.parse_value_params()?;
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

                self.ctx.addressable_nodes.insert(value);
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

                self.ctx.addressable_nodes.insert(value);
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
        }

        Ok(value)
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

        let params = self.parse_value_params()?;
        let range = self.expect_range(start, Token::RCurly)?;

        let struct_node = self
            .ctx
            .push_node(range, Node::StructLiteral { name, params });

        self.ctx.addressable_nodes.insert(struct_node);

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

    pub fn parse_let(&mut self) -> Result<NodeId, CompileError> {
        let start = self.source_info.top.range.start;

        self.pop(); // `let`
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
        let let_id = self.ctx.push_node(range, Node::Let { name, ty, expr });

        self.ctx.addressable_nodes.insert(let_id);
        self.ctx.scope_insert(name_sym, let_id);

        Ok(let_id)
    }

    pub fn parse_block(&mut self, is_standalone: bool) -> Result<NodeId, CompileError> {
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

        if is_standalone {
            self.ctx.addressable_nodes.insert(block_id);
        }

        Ok(block_id)
    }

    pub fn parse_if(&mut self) -> Result<NodeId, CompileError> {
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

    pub fn parse_for(&mut self) -> Result<NodeId, CompileError> {
        let start = self.source_info.top.range.start;

        self.pop(); // `for`

        let pushed_label_scope = self.ctx.push_scope();

        let label = self.parse_symbol()?;
        self.ctx.addressable_nodes.insert(label);
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

    pub fn parse_block_stmt(&mut self) -> Result<NodeId, CompileError> {
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

                        // Assignment lhs never needs to be addressable
                        self.ctx.addressable_nodes.remove(&lvalue);

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

                Ok(self
                    .ctx
                    .push_node(range, Node::Type(Type::Struct { name: None, params })))
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
                self.expect(Token::RSquare)?;
                let ty = self.parse_type()?;
                let range = Range::new(start, self.ctx.ranges[ty].end, self.source_info.path);
                Ok(self.ctx.push_node(range, Node::Type(Type::Array(ty))))
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
                let mut range = self.source_info.top.range;
                self.pop();
                if self.source_info.top.tok == Token::Bang {
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

                    Ok(self
                        .ctx
                        .push_node(range, Node::PolySpecialize { sym, overrides }))
                } else {
                    Ok(self.ctx.push_node(range, Node::Symbol(sym)))
                }
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

                Ok(value)
            }
        }
    }

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
    pub fn parse_file(&mut self, file_name: &str) -> Result<(), CompileError> {
        let mut source = self.make_source_info_from_file(file_name);
        let mut parser = Parser::from_source(self, &mut source);

        parser.parse()
    }

    pub fn ropey_parse_file(&mut self, file_name: &str) -> Result<(), CompileError> {
        let mut source = self.make_ropey_source_info_from_file(file_name);
        let mut parser = Parser::from_source(self, &mut source);
        parser.parse()
    }

    pub fn parse_str(&mut self, source: &'static str) -> Result<(), CompileError> {
        let mut source = SourceInfo::<StaticStrSource>::from_static_str(source);
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

    pub fn polymorph_copy(
        &mut self,
        id: NodeId,
        target: ParseTarget,
    ) -> Result<NodeId, CompileError> {
        // Re-parse the region of the source code that contains the id
        // todo(chad): @performance
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
                // todo(chad): @hack_polymorph
                let Node::StructDefinition { params, .. } = self.nodes[parsed] else { panic!() };
                let params = self.id_vecs[params].clone();
                for param in params.borrow().iter() {
                    let Node::StructDeclParam { ty, .. } = self.nodes[param] else { panic!() };
                    if let Some(ty) = ty {
                        // todo(chad): @hack_polymorph
                        if let Node::Symbol(_) = self.nodes[ty] {
                            let copied = self.copy_polymorph_if_needed(ty);
                            self.nodes[ty] = self.nodes[copied];
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
}

impl Op {
    fn precedence(self) -> u8 {
        match self {
            Op::Add | Op::Sub => 1,
            Op::Mul | Op::Div => 2,
            Op::EqEq => 3,
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
            _ => unreachable!(),
        }
    }
}
