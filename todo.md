# Bugs
- [ ] shouldn't have to type `return` at the end of every function with no return type
- [ ] ```
    let a: [3]i64 = [1, 2, 3]; 
    a::len + a[0] |> print_i64;
    ```

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
- [x] slices
    - [x] Make a distinction between arrays and slices:
        - [x] `let a: [3]i64 = [1, 2, 3];` is basically identical memory-wise to a struct with 3 members. Can be indexed like a slice
        - [x] `let a: [_]i64 = [1, 2, 3];` is the same thing, inferring 3 as the len
        - [x] `let a: []i64 = _{ data: &[1, 2, 3], len: 3 } as _;` is a slice. So an array literal expression can be type-coerced to an array or a slice
        - [x] `let a: []i64 = [1, 2, 3] as []i64;` is the same as above
        - [x] `let a: []i64 = [1, 2, 3] as []_;` is the same as above
        - [x] `let a: []i64 = [1, 2, 3] as _;` is the same as above
- [x] for loops
- [x] !=, <, >, <=, >= operators
- [x] while loops
- [x] arrays assignable to Array type
- [x] pipe/threading/ufcs operator
    - [x] threading operator can choose where the argument gets threaded into
- [ ] #transparent
- [ ] arrow operator
- [ ] #this
- [ ] const struct members
    - [ ] functions
    - [ ] other stuff??
- [x] strings
    - [x] store in program's data segment
- [x] short-circuit and/or operators
- [ ] pattern matching (on enums only at first, very simple single-depth match)
    - [ ] `matches(...)`
    - [ ] `match`
    - [ ] `if let`
    - [ ] `let else`
- [ ] labelled blocks (use as target for resolve/continue/defer)
- [ ] continue
- [ ] defer
- [ ] any / #astof(...) / type info stuff
- [ ] question mark operator
- [ ] string interpolation
- [ ] coroutines
- [ ] c interop
    - [x] extern functions POC
    - [ ] make sure extern functions works with libs/dylibs when compiling to binary
    - [ ] pass structs as arguments
- [ ] implicit function arguments
- [ ] debugger
- [ ] modules, imports
    - [ ] import { A, B::{C, D}, E::* } from "whatever/bar"

# Performance
- [ ] investigate performance hit of using Box/Rc instead of the IdVec convention
- [ ] use the `stack_store` cranelift ir instruction
- [ ] use cranelift's frontend for `let` bindings - they don't need to always have stack storage by default unless something is specifically taking a reference to them later
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