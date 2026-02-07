test-rust:
    cargo nextest run --workspace
    cargo nextest run -p okmain --features image
    cargo nextest run -p okmain --features image,unstable
    cargo test --doc --workspace
    cargo test --doc -p okmain --features image

maturin-develop:
    cd python && uv run --package okmain maturin develop

test-python:
    uv run --package okmain pytest python/tests/ -v

test: test-rust test-python

lint-rust:
    cargo fmt --all -- --check
    cargo clippy --workspace --all-features -- -D warnings

lint-python:
    uv run --package okmain ruff check python/
    uv run --package okmain ruff format --check python/
    uv run --package okmain mypy python/okmain

lint: lint-rust lint-python

bench:
    cargo bench -p okmain --features image,_debug

fmt:
    cargo fmt --all
    uv run --package okmain ruff format python/

debug_colors:
    cargo run --bin debug_colors --features="_debug" --release test_images/ 
