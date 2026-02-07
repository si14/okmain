# PyO3 Wrapper — `okmain_py`

Translates between Rust types and Python-friendly types. Contains **no business logic**.

## Constraints

- One file only: `src/lib.rs`. No sub-modules, no algorithm code.
- Never published to crates.io (`publish = false`).
- All symbols exported to Python use a `_` prefix (e.g., `_colors_debug`, `_ScoredCentroid`, `_DebugInfo`) to signal
  they are internal to `okmain._core`.

## Type translation rules

| Rust                         | Python (in `_core.pyi`)                                |
|------------------------------|--------------------------------------------------------|
| Named struct (e.g., `Oklab`) | Plain tuple (e.g., `tuple[float, float, float]`)       |
| `Vec<T>`                     | `list[T]`                                              |
| `Result<_, E>`               | Raise `PyValueError` with the error's `Display` string |

The Python layer (`python/okmain/__init__.py`) is responsible for wrapping raw tuples into typed frozen dataclasses.

## Update checklist

When changing the Rust interface:

1. Edit `src/lib.rs`.
2. Update `python/okmain/_core.pyi` to match — this is what mypy checks.
3. Update `python/okmain/__init__.py` if new data needs wrapping.
4. `just develop && just test-python`

## Lint notes

`#[allow(clippy::too_many_arguments)]` is expected on functions mirroring the `colors_with_config` signature.
