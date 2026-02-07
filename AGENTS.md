# okmain — Dominant Color Extraction

## Overview

Extract dominant colors from images. Ships as:
- A **Rust crate** (`okmain`) on crates.io
- A **Python package** (`okmain`) on PyPI via maturin/PyO3

## Workspace Layout

```
okmain/
├── crates/okmain/      # Pure Rust library (published to crates.io)
├── crates/okmain_py/   # PyO3 wrapper (NOT published to crates.io)
└── python/             # Python package (published to PyPI)
```

**Key architectural rule:** the core Rust crate (`crates/okmain`) has **zero PyO3 knowledge**. All PyO3 bindings live in `crates/okmain_py/`. This keeps core tests independent of the Python runtime, improves compile times, and separates concerns cleanly.

## Tooling

| Tool      | Purpose                          |
|-----------|----------------------------------|
| `just`    | Task runner (`justfile` at root) |
| `uv`      | Python project/env management    |
| `maturin` | Build PyO3 extension             |
| `ruff`    | Python linting & formatting      |
| `mypy`    | Python type checking             |

**uv workspace:** The root `pyproject.toml` defines a uv workspace with `python/` as a member. Run `uv sync`, `uv run pytest`, `uv run ruff`, etc. from the **repo root** — no need to `cd python` first.

## Conventions

- **Rust edition:** 2024, resolver 3, MSRV 1.93
- **License:** MIT OR Apache-2.0 (dual license, both files in repo root)
- **Error handling (core crate):** `snafu` 0.8 internally; public API is snafu-agnostic by default

## Sub-directory Guides

- [`crates/okmain/AGENTS.md`](crates/okmain/AGENTS.md) — Core Rust crate guidance
- [`python/AGENTS.md`](python/AGENTS.md) — Python/PyO3 wrapper guidance
