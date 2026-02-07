# Run all Rust tests
test-rust:
    cargo nextest run --workspace
    cargo nextest run -p okmain --features image

# Build the Python extension in dev mode
develop:
    cd python && uv run maturin develop

# Run Python tests
test-python:
    uv run pytest python/tests/ -v

# Run all tests
test: test-rust test-python

# Rust lint
lint-rust:
    cargo fmt --all -- --check
    cargo clippy --workspace --all-features -- -D warnings

# Python lint + type check
lint-python:
    uv run ruff check python/
    uv run ruff format --check python/
    uv run mypy python/okmain

# All lints
lint: lint-rust lint-python

# Run benchmarks
bench:
    cargo bench -p okmain --features _bench

# Format all
fmt:
    cargo fmt --all
    uv run ruff format python/
