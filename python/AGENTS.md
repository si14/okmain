# Python/PyO3 Wrapper — `okmain_py`

## Rules

- The Rust PyO3 crate lives at `crates/okmain_py/` and is **never published** to crates.io (`publish = false`).
- `crates/okmain_py/src/lib.rs` is a **thin PyO3 wrapper only** — no business logic.
- Python source lives in `python/okmain/`.
- RGBA rejection happens in the Python layer (`__init__.py`), not in Rust.

## Development

All commands run from the **repo root** (uv workspace):

```sh
uv sync                                # set up venv + deps
just develop                           # build extension (maturin)
uv run pytest python/tests/ -v         # run tests
uv run ruff check python/              # lint
uv run ruff format --check python/     # format check
uv run mypy python/okmain              # type check
```

## Build

```sh
cd python && uv run maturin develop    # dev build (editable)
cd python && uv run maturin build      # release wheel
```
