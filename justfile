bench:
    hyperfine -N "./target/release/draft tests.dr"

bench-run:
    hyperfine -N "./target/release/draft tests.dr --run"
