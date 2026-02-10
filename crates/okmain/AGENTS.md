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

## Code Style

Style reference for the k-means module: `lloyds.rs`.

### Be concise
Every line should do real work. Don't add intermediate variables, wrapper structs, or control flow that doesn't earn its keep. Directly index SoA arrays instead of constructing a temporary struct to pass to a helper. Fewer lines of straightforward code beat more lines of "clean" code.

### Factor out hot paths for benchmarking
Expensive inner loops (`assign_points`, `update_centroids`) must be separate `pub` functions so criterion benchmarks can target them individually. Don't bury performance-critical work inside a monolithic function where it can't be measured in isolation.

### Match the data representation
Data here is SoA (`SampledOklabSoA`). Write functions that operate on SoA directly. Don't convert between layouts at call boundaries — e.g. don't build an AoS struct from SoA fields just to pass it to a function that immediately destructures it back into components.

### No scaffolding in shipped code
No `println!` debug output. No dead code kept around with `let _ = unused;`. No commented-out alternative implementations. Code should be clean enough to ship as-is.

### Don't over-engineer edge cases
If `count == 0` is the right check, write `count == 0`. Don't build a threshold heuristic with several intermediate variables that ultimately does the same thing with more code.

### Separate algorithm steps from policy
The iteration loop does the math. It doesn't also do initialization or output format conversion. Callers own the policy — which init strategy to use, how to package results, when to retry. Inner functions own the computation.

### Common LLM anti-patterns to avoid

- **Monolithic "do everything" functions** that bundle initialization, iteration, convergence checking, and output conversion into one block. Split them so each piece is testable and benchmarkable.
- **Layout conversion at call boundaries.** Writing `nearest_centroid(point: &Oklab, ...)` when the caller has SoA data forces a struct construction at every call site. Pass scalar components or slice references instead.
- **Allocating inside loops.** `Vec::with_capacity(n)` + `.push()` in every iteration of a convergence loop. Pre-allocate once, pass `&mut [u8]`, reuse.
- **Hedging with dead code.** Computing a value "just in case" and then suppressing the unused warning with `let _ = x;`. Delete it.
- **Complexity as a proxy for correctness.** Adding threshold heuristics, smoothing, or multi-step logic where a trivial check suffices. Simple code is easier to verify.
