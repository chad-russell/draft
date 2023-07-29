# Bugs
- [x] matching things against polymorph sources will pretty much just accept anything
    - ```
        struct Array {
            data: *_T!,
            len: i64,
        }

        fn main() i64 {
            let a: Array = _{ foobar: 3i64 };
            return 0;
        }
    ```
- [ ] shouldn't have to type `return` at the end of every function with no return type

# Features
- [x] test returning structs from a function
- [x] clean up 'store if addressable' logic if possible
- [x] test suite
    - [x] test for recursive structs (linked list)
- [x] test recursive functions
- [x] blocks as expressions
- [x] structs
- [x] pointers
- [x] polymorphism
    - [x] functions
    - [x] structs
    - [x] enums
- [x] booleans
- [x] if/else
    - [x] as expressions
- [x] `==` operator
- [x] enums
- [x] #cast
    - [x] restrict to only pointers (?)
- [x] arrays
- [x] for loops
- [x] !=, <, >, <=, >= operators
- [x] while loops
- [x] arrays assignable to Array type
- [x] pipe/threading/ufcs operator
    - [ ] threading operator can choose where the argument gets threaded into
- [x] strings
- [ ] interfaces
- [ ] modules, imports
    - [ ] import "whatever/bar"::{ A, B::{C, D}, E::* }
- [ ] short-circuit and/or operators
- [ ] defer
- [ ] question mark operator (how would this work? do we need interfaces after all?)
- [ ] string interpolation (need a default stringification function for everything, how does that work? interfaces again?)
- [ ] pattern matching (on enums only at first, very simple single-depth match)
    - [ ] `matches(...)`
    - [ ] `match`
    - [ ] `if let`
    - [ ] `let else`
- [ ] coroutines
- [ ] any / #astof(...)
- [ ] c interop
    - [x] extern functions POC
    - [ ] make sure extern functions works with libs/dylibs when compiling to binary
    - [ ] pass structs as arguments
- [ ] implicit function arguments
- [ ] immutability (?)
- [ ] debugger

# Performance
- [ ] use the `stack_store` cranelift ir instruction
- [ ] when doing codegen for aggregate types, they don't always need their own slot. Usually it's going to get copied into the slot of somethign else like a let binding, so we can just directly codegen it into that slot
- [ ] consider disabling bounds checking on array (DenseStorage) accesses
- [ ] multithreading?

# LSP / Tooling
- [ ] brewfmt
- [ ] create rust tests which each mimic one of the capabilities of the lsp
- [x] error reporting
- [ ] symbol outline
- [-] completion
    - [x] for symbols in scope
    - [ ] for dots on structs/enums/modules/etc.
    - [ ] for functions (params)
- [x] go to definition
- [ ] hover
- [ ] go to type definition
- [ ] run button over main
- [ ] generate missing function
- [ ] import suggestions
- [x] syntax highlighter