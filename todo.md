# Bugs
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
    - [x] threading operator can choose where the argument gets threaded into
- [x] strings
    - [x] store in program's data segment
- [x] short-circuit and/or operators
- [ ] pattern matching (on enums only at first, very simple single-depth match)
    - [ ] `matches(...)`
    - [ ] `match`
    - [ ] `if let`
    - [ ] `let else`
- [ ] interfaces
- [ ] continue
- [ ] defer
- [ ] labelled blocks (use as target for resolve/continue/defer)
- [ ] any / #astof(...) / type info stuff
- [ ] question mark operator
- [ ] string interpolation (need a default stringification function for everything, how does that work? interfaces again?)
- [ ] coroutines
- [ ] c interop
    - [x] extern functions POC
    - [ ] make sure extern functions works with libs/dylibs when compiling to binary
    - [ ] pass structs as arguments
- [ ] implicit function arguments
- [ ] closures (can this be fully done using implicit function arguments?)
- [ ] immutability (?)
- [ ] debugger
- [ ] modules, imports
    - [ ] import { A, B::{C, D}, E::* } from "whatever/bar"

# Performance
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

# Interface Ideas

---

interface SayHello {
    fn say_hello();
}

// struct SayHello {
//   data: *_T!,
//   vtable: *struct{
//     fn say_hello(self: *T),
//   }
// }

struct Foo {
    x: i64,
    y: i64,
}

impl SayHello for Foo {
    fn say_hello() {
        println("Foo says hello");
    }
}

fn dynamic_say_hello(s: SayHello!(*)) {
    s..say_hello();
}

fn static_say_hello_1(s: SayHello!(_T!)) {
    s..say_hello();
}

fn static_say_hello_2(s: _T!) {
    (impl SayHello for T)::say_hello(&s);
}

fn main() {
    let f = Foo { 3, 4 };

    let fsh = impl SayHello for Foo;
    dynamic_say_hello(fsh);

    static_say_hello(f);
}

---