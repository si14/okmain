test-rust:
    cargo nextest run --workspace
    cargo nextest run -p okmain --features image
    cargo test --doc --workspace
    cargo test --doc -p okmain --features image

# TODO: what is this generated nonsense

# Build the Python extension in dev mode
develop:
    cd python && uv run maturin develop

test-python:
    uv run pytest python/tests/ -v

test: test-rust test-python

lint-rust:
    cargo fmt --all -- --check
    cargo clippy --workspace --all-features -- -D warnings

lint-python:
    uv run ruff check python/
    uv run ruff format --check python/
    uv run mypy python/okmain

lint: lint-rust lint-python

bench:
    cargo bench -p okmain --features image,_debug

fmt:
    cargo fmt --all
    uv run ruff format python/
