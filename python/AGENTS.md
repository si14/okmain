# Python Package — `okmain`

## Architecture

- `python/okmain/__init__.py` — the entire public API
- `python/okmain/_core.pyi` — type stub for the compiled Rust extension (keep in sync with `crates/okmain_py/src/lib.rs`)
- `crates/okmain_py/src/lib.rs` — the PyO3 Rust code that backs `_core`

**Validation rule:** input validation (e.g., `image.mode != "RGB"`) lives in `__init__.py`, not in Rust.

## Commands (all from repo root)

```sh
just develop                         # rebuild Rust extension after any Rust change
uv run pytest python/tests/ -v       # tests
uv run mypy python/okmain            # type check (strict mode)
uv run ruff check python/            # lint
uv run ruff format python/           # format
```

## Code style

### Public types — frozen dataclasses with slots

```python
@dataclass(frozen=True, slots=True)
class RGB:
    r: int
    g: int
    b: int
```

Every public type follows this pattern. No mutable classes, no plain dicts.

### Converting `_core` types — `_from_core` classmethod

Raw `_core` tuples are wrapped into typed dataclasses via a `_from_core` classmethod:

```python
@classmethod
def _from_core(cls, sc: _ScoredCentroid) -> Self:
    rgb_r, rgb_g, rgb_b = sc.rgb
    lab_l, lab_a, lab_b = sc.oklab
    return cls(rgb=RGB(rgb_r, rgb_g, rgb_b), oklab=Oklab(lab_l, lab_a, lab_b), ...)
```

### Overloads for type-narrowing polymorphism

When a boolean flag changes the return type, use `@overload` + `Literal`:

```python
@overload
def colors(..., with_debug: Literal[True]) -> tuple[list[RGB], DebugInfo]: ...
@overload
def colors(..., with_debug: Literal[False] = ...) -> list[RGB]: ...
```

### Other conventions

- `from __future__ import annotations` at the top of every module.
- All optional config parameters are keyword-only (after `*`).
- Explicit `__all__` in `__init__.py` listing every public symbol.

## Tests

pytest, no fixtures, all test functions annotated `-> None`.

```python
def test_rgba_raises() -> None:
    img = Image.new("RGBA", (2, 2), (255, 0, 0, 255))
    with pytest.raises(ValueError, match="RGBA"):
        okmain.colors(img)
```

Cover: happy path, error cases (wrong mode, invalid config), debug mode, result ordering. Use `match=` on `pytest.raises` to assert the error message text.

## Recipe: adding a new public type

1. Add to `crates/okmain_py/src/lib.rs` (return as tuple, not a Python class).
2. Update `python/okmain/_core.pyi` with the matching stub.
3. Add a `@dataclass(frozen=True, slots=True)` class in `__init__.py` with a `_from_core` classmethod.
4. Add to `__all__`.
5. Wire into `__init__.py` conversion code.
6. Add tests.
7. `just develop && just test-python && uv run mypy python/okmain`
