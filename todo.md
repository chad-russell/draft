# Bugs
- [ ] shouldn't have to type `return` at the end of every function with no return type
- [ ] ```
    let a: [3]i64 = [1, 2, 3]; 
    a::len + a[0] |> print_i64;
    ```
    - this is a type inference issue, combined with precedence. The precedence interprets this as `a::len + (a[0] |> print_i64)`,
    and because the `print_i64` call has return type unassigned, it is automatically infers it (incorrectly) as i64.
- [x] putting `continue` or `break` where they don't belong causes errors. Need to catch this at semantic analysis time
- [x] having statements after a `return`, `continue` or `break` shouldn't cause a compile error
- [x] semantic checking for putting `return`, `continue` or `break` in the wrong context
- [x] semantic checking for referencing in a nonexistent label

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
- [x] arrays assignable to [] type
- [x] pipe/threading/ufcs operator
    - [x] threading operator can choose where the argument gets threaded into
- [x] #transparent
    - [x] keyword on struct members
    - [x] keyword on let stmts
    - [x] keyword on fn parameters
    - [x] keyword on fn declarations
    - [x] structs can be transparent
    - [x] unnamed struct types can be transparent
    - [x] functions can be transparent
    - [x] enums can be transparent
- [x] arrow operator
- [x] strings
    - [x] store in program's data segment
- [x] short-circuit and/or operators
- [x] implicit function arguments
- [x] labelled blocks (use as target for break/continue/defer)
    - [x] labelled while blocks
    - [x] labelled if blocks
    - [x] labelled for blocks
    - [x] labelled general blocks
    - [x] `break` can reference labels
    - [x] `continue` can reference labels
- [x] continue
- [ ] any / #astof(...) / type info stuff
- [-] pattern matching (on enums only at first, very simple single-depth match)
    - [x] `match`
        - [x] underscore for catch-all case
            - [ ] error for when trying to specify two catch-all cases
            - [ ] error when specifying a catch-all case but all the other cases are already covered 
    - [x] `if let`
- [ ] stack allocation should work the same way as any other allocation, and produce a pointer. If a struct is put on the stack, 
      why should it not be a pointer, and copy, but if it's allocated through an arena it's suddenly a pointer and a reference?
    - [ ] `let f: Foo = _{ 3, 4 };`
    - [ ] `let f: Foo = _{ 3, 4 } @ arena!; fn arena() -> *_T! { ... }`
        - literally shorthand for `let f: *Foo = arena!(); *f = #{ 3, 4 };`
- [ ] type alignment
- [ ] enum discriminant type should be variable, based on how many members there are (but still alignment comes into play, so often we'll have a small type but also enough padding to make it not really matter)
- [ ] defer
    - [ ] can reference a label
- [ ] c interop
    - [x] extern functions POC
    - [ ] make sure extern functions works with libs/dylibs when compiling to binary
    - [ ] pass structs as arguments
- [ ] debugger
- [ ] modules, imports
    - [ ] import { A, B::{C, D}, E::* } from "whatever/bar"
- [ ] const struct members
    - [ ] functions
    - [ ] other stuff (??)
- [ ] string interpolation
- [ ] coroutines
- [ ] question mark operator (maybe, not sure about this one yet)

# Testing
- [ ] Write rust tests which check for compile errors

# Performance
- [x] investigate performance hit of using Box/Rc instead of the IdVec convention (it was about 10-15%)
- [x] when doing codegen for aggregate types, they don't always need their own slot. Usually it's going to get copied into the slot of somethign else like a let binding, so we can just directly codegen it into that slot
- [ ] consider disabling bounds checking on array (DenseStorage) accesses
- [ ] multithreading?

# LSP / Tooling
- [ ] brewfmt
- [ ] create rust tests which each mimic one of the capabilities of the lsp - for easy reproducing of errors
- [x] error reporting
- [ ] symbol outline
- [-] completion
    - [x] for symbols in scope
    - [ ] for member access
    - [ ] for functions (params)
- [x] go to definition
- [ ] hover
- [ ] go to type definition
- [ ] run button over main
- [ ] generate missing function
- [ ] import suggestions
- [x] syntax highlighter

# New Pointer Approach (??)
- Everything is a pointer, unless it's specifically declared to be a register
- Any function returning a pointer to a T can act as an T-allocator for purposes of declaring variables
    - The function must be callable with no arguments. It may however be polymorphic, and may take arguments as long as they are all implicitly available at every call site
    - Examples (*all* of the below are of type *Foo):
        - let a = 'stack Foo { 3, 4 };
        - let b = 'malloc Foo { 3, 4 };
        - let c = 'my_alloc Foo { 3, 4 };
        - let d = 'array_push Foo { 3, 4 };
- A pointer should be more than just a 64-bit integer. Conceptually, it can be any data imaginable where there is a way to translate that data into the precise location of another piece of data.
    - Relative pointers (from a statically known offset)
    - Array indices
    - Map keys
