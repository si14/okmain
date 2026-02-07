# okmain — Dominant Color Extraction

Extract dominant colors from images. Ships as a Rust crate (`okmain` on crates.io) and Python package (`okmain` on PyPI, built with maturin/PyO3).

## Workspace layout

```
okmain/
├── crates/okmain/      # Pure Rust library (published to crates.io)
├── crates/okmain_py/   # PyO3 wrapper (never published to crates.io)
└── python/             # Python package (published to PyPI)
```

## Invariants — never break these

- `crates/okmain` has **zero PyO3 knowledge**. All Python bindings live in `crates/okmain_py/`.
- `crates/okmain_py/src/lib.rs` is a **thin translation layer only** — no business logic.
- Input validation for Python callers (e.g., RGBA rejection) lives in `python/okmain/__init__.py`, not in Rust.

## Commands

```sh
just test         # all tests (Rust + Python)
just lint         # all linters (Rust + Python)
just develop      # rebuild maturin extension (required before Python tests after any Rust change)
just fmt          # auto-format everything
```

Individual targets: `just test-rust`, `just test-python`, `just lint-rust`, `just lint-python`.
Run `uv sync`, `uv run pytest`, etc. from the **repo root** (uv workspace).

## Settings

- **Rust edition:** 2024 · **MSRV:** 1.93 · **resolver:** 3 · **license:** MIT OR Apache-2.0
- **Python:** ≥3.12 · **mypy:** strict · **line length:** 99

## Sub-guides

- [`crates/okmain/AGENTS.md`](crates/okmain/AGENTS.md) — Core Rust crate
- [`crates/okmain_py/AGENTS.md`](crates/okmain_py/AGENTS.md) — PyO3 wrapper
- [`python/AGENTS.md`](python/AGENTS.md) — Python package
