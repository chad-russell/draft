# Bugs
- [ ] should be able to type `return` at the end of a function with no return type even though it's pointless, without a cranelift error
- [ ] ```
    let a: [3]i64 = [1, 2, 3]; 
    a::len + a[0] |> print_i64;
    ```
    - this is a type inference issue, combined with precedence. The precedence interprets this as `a::len + (a[0] |> print_i64)`,
    and because the `print_i64` call has return type unassigned, it is automatically infers it (incorrectly) as i64.

# Features
- [-] modules, imports
    - [x] import bare module
    - [x] import with double colon
    - [x] import with curly braces
    - [ ] import file modules
    - [x] import aliases
    - [ ] import *
    - [ ] import url
- [ ] debugger
- [ ] top-level/struct-level global constants
    - [ ] numerics / bools
    - [ ] strings
    - [ ] arrays of constant values
    - [ ] structs with constant field values
    - [ ] enums
    - [ ] pointers
    - [ ] use these constants in polymorphs (?)
- [ ] c interop
    - [x] extern functions POC
    - [x] make sure extern functions works with libs/dylibs when compiling to binary
    - [ ] pass structs as arguments
- [ ] type alignment
- [ ] enum discriminant type should be variable, based on how many members there are
- [ ] aoc problems
- [ ] figure out what stance to take on variable shadowing

# Experimental Features
- [ ] string interpolation
- [ ] coroutines
- [ ] stack allocation should work the same way as any other allocation, and produce a pointer. If a struct is put on the stack, 
      why should it not be a pointer, and copy, but if it's allocated through an arena it's suddenly a pointer and a reference?
    - [ ] `let f: *Foo = _{ 3, 4 };`
    - [ ] `let f: *Foo = _{ 3, 4 } @ arena!; fn arena() -> *_T! { ... }`
        - literally shorthand for `let f: *Foo = arena!(); *f = _{ 3, 4 };`
        - A pointer should be more than just a 64-bit integer. Conceptually, it can be any data imaginable where there is a way to translate that data into the precise location of another piece of data.
            - Relative pointers (from a statically known offset)
            - Array indices
            - Map keys

# Testing
- [ ] Write rust tests which check for compile errors

# Performance
- [ ] consider disabling bounds checking on array (DenseStorage) accesses
- [ ] multithreading?

# LSP / Tooling
- [ ] drafmt auto-formatter
- [ ] create rust tests which each mimic one of the capabilities of the lsp - for easy reproducing of errors
- [ ] error reporting
- [ ] symbol outline
- [-] completion
    - [ ] for symbols in scope
    - [ ] for member access
    - [ ] for functions (params)
- [-] go to definition
- [ ] hover
- [ ] go to type definition
- [ ] run button over main
- [ ] generate missing function
- [ ] import suggestions
- [-] syntax highlighter
