# Core Crate — `okmain`

## Constraints

- **No PyO3 dependency** — ever. All Python bindings live in `crates/okmain_py/`.
- **Error handling:** `snafu` 0.8 internally (`rust_1_81` feature). Validate with `ensure!`. Public `Error` types are plain `std::error::Error`.
- Mark all public enums `#[non_exhaustive]`.
- Feature gate conventions:
  - `image` — optional `image` crate dep; gates `dominant_color_from_image`
  - `unstable` — public but experimental API (`colors_debug`, `ScoredCentroid`, etc.)
  - `_debug` — internal only (debug binaries + benches); never mention in public docs
- **RNG:** only touch `src/rng.rs` to change the algorithm or seed.

## Testing

```sh
cargo nextest run -p okmain                              # default features
cargo nextest run -p okmain --features image             # with image support
cargo nextest run -p okmain --features image,unstable
cargo test --doc -p okmain --features image              # doc tests
```

Use `pretty_assertions` for all equality assertions. Test helpers live in the same file as the tests they support.

## Code style

Style reference for the k-means module: `src/kmeans/lloyds.rs`.

### Data layout
Data is SoA (`SampledOklabSoA`, `CentroidSoA`). Write functions that operate on SoA directly — don't convert to AoS at call boundaries.

### Conciseness
Every line earns its keep. Index SoA arrays directly instead of constructing a temporary struct to pass to a helper. Fewer clear lines beat more "clean" lines.

### Hot paths
Expensive inner loops (`assign_points`, `update_centroids`) must be `pub fn` so benchmarks can target them individually. Use `#[inline(always)]` on leaf scalar functions called in the innermost loop (e.g., `squared_distance_flat`); use `#[inline]` on larger but still hot functions.

### Algorithm/policy separation
The iteration loop does math. Initialization, output formatting, and retry policy belong to the caller.

## Common LLM anti-patterns

- **Monolithic functions** bundling init + iteration + convergence check + output conversion. Split them — each piece should be testable and benchmarkable independently.
- **AoS at call boundaries** — writing `fn nearest_centroid(point: &Oklab, ...)` when the caller has SoA forces a struct construction at every call site. Pass scalar components or slice references instead.
- **Allocating inside loops** — `Vec::with_capacity` + `.push()` in every convergence iteration. Pre-allocate once, pass `&mut [T]`, reuse.
- **Spurious complexity** — threshold heuristics and multi-step logic where a simple check suffices. Simple code is easier to verify.
