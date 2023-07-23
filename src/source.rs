use std::path::PathBuf;

use crate::{breaks_symbol, Lexeme, Location, Range, Token};

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
    pub path: &'static str,
    pub source: W,
    pub chars_left: usize,

    pub loc: Location,
    pub top: Lexeme,
    pub second: Lexeme,
}

impl<W: Source> SourceInfo<W> {
    pub fn from_file(file_name: &str) -> SourceInfo<StrSource> {
        let path = PathBuf::from(file_name);
        let source = std::fs::read_to_string(&path).unwrap();
        let source: &'static str = Box::leak(source.into_boxed_str());
        let source = StrSource::from_str(source);
        let chars_left = source.char_count();
        let path = Box::leak(path.into_boxed_path().to_str().unwrap().into());

        SourceInfo {
            path,
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
            path: source_path,
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
            path: "<none>",
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
            path: "<none>",
            source,
            chars_left,
            loc: Default::default(),
            top: Default::default(),
            second: Default::default(),
        }
    }

    pub fn from_range(range: Range) -> SourceInfo<StrSource<'static>> {
        let source_path = PathBuf::from(range.source_path);
        let source = std::fs::read_to_string(&source_path).unwrap();
        let source = source[range.start.char_offset..range.end.char_offset].to_string();
        let source: &'static str = Box::leak(source.into_boxed_str());
        let source = StrSource::from_str(source);
        let chars_left = source.char_count();
        let source_path = Box::leak(source_path.into_boxed_path().to_str().unwrap().into());

        SourceInfo {
            path: source_path,
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
        Range::new(start, end, self.path)
    }

    pub fn prefix(&mut self, pat: &str, tok: Token) -> bool {
        if self.source.char_count() >= pat.len() && self.source.starts_with(pat) {
            let start = self.loc;
            self.eat(pat.len());
            self.second = Lexeme::new(tok, self.make_range(start, self.loc));
            true
        } else {
            false
        }
    }

    pub fn prefix_keyword(&mut self, pat: &str, tok: Token) -> bool {
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

    pub fn eat_chars(&mut self, chars: usize) {
        let chars = chars.min(self.chars_left);
        self.source.pop_chars(chars);
        self.chars_left -= chars;
    }

    pub fn eat(&mut self, chars: usize) {
        self.eat_chars(chars);
        self.loc.col += chars;
        self.loc.char_offset += chars;
    }

    pub fn newline(&mut self) {
        self.eat_chars(1);
        self.loc.line += 1;
        self.loc.col = 1;
        self.loc.char_offset += 1;
    }

    pub fn eat_rest_of_line(&mut self) {
        let chars = self
            .source
            .position_of(|c| c == '\n')
            .unwrap_or_else(|| self.source.char_count());
        self.eat(chars);

        if !self.source.is_empty() {
            self.newline();
        }
    }

    pub fn eat_spaces(&mut self) {
        loop {
            let mut br = true;

            // eat spaces
            while let Some(' ') = self.source.next_char() {
                br = false;
                self.eat(1);
            }

            // eat newlines
            while let Some('\r') | Some('\n') = self.source.next_char() {
                br = false;
                self.newline();
            }

            // eat comments
            while self.source.starts_with("//") {
                br = false;
                self.eat_rest_of_line();
            }

            if br {
                break;
            }
        }
    }
}
