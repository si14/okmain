# Core Crate — `okmain`

## Rules

- **No PyO3 dependency.** This crate must never depend on `pyo3`. All Python bindings live in `python/`.
- **Error handling:** Use `snafu` 0.8 (with `rust_1_81` feature) internally. The public `Error` type is a normal
  `std::error::Error` — consumers don't need snafu.
- **Optional `snafu` feature:** When enabled, re-exports `snafu` and makes context selectors public.
- **Optional `image` feature:** Gates `dominant_color_from_image` and the `image` dependency.
- Tests use `pretty_assertions`
- **RNG:** Abstracted behind `rng::new()`. To change the RNG algorithm or seed, only modify `src/rng.rs`.

## Testing

```sh
cargo nextest -p okmain                   # default features
cargo nextest -p okmain --features image  # with image support
```

## Public API Surface

Keep it snafu-agnostic by default. Only expose snafu internals behind the `snafu` crate feature.
