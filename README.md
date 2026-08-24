# Okmain: OK main colors

`okmain` finds the main colors of an image and makes sure they look good.

Sometimes you need to show a "dominant" color (or colors) of an image. It can be a background or a placeholder.
There are several ways of doing that; a popular quick-and-dirty method is to resize the image
to a handful of pixels, or even just one.

However, this method tends to produce muted, dirty-looking colors. Most images have clusters of colors:
the dominant colors of an image of a lush green field with a clear sky above it are not a muddy average
of blue and green, it's a bright blue and green. Okmain clusters colors explicitly, recovering and ranking main colors
while keeping them sharp and clean.

Here's a comparison:

[<img src="https://dgroshev.com/img/okmain/comparison01.png" width="500" title="Comparison of colors extracted via 1x1 resize and Okmain" alt="Comparison of colors extracted via 1x1 resize and Okmain"/>](https://dgroshev.com/img/okmain/comparison01.png)

## Technical highlights

- Color operations in a state-of-the-art perceptually linear color space
  ([Oklab](https://bottosson.github.io/posts/oklab/))
- Rust implementation for speed and safety
- Finding main colors of a reasonably sized image takes about 100ms
- Minimal and stable dependencies
- Fast custom K-means color clustering, optimized for auto-vectorization (confirmed with disassembly)
- Position- and visual prominence-based color prioritization (more central and higher Oklab chroma pixels tend to be
  more important)
- Tunable parameters

Read more about Okmain in the [blog post](https://dgroshev.com/blog/okmain/).

## Packages

`okmain` is available in:

* [Rust](https://github.com/si14/okmain/tree/main/crates/okmain)
* [Python](https://github.com/si14/okmain/tree/main/python) (a wrapper over the Rust crate)
* JavaScript/WebAssembly proof-of-concept in [`wasm-poc/`](wasm-poc)

## LLM disclosure

LLMs are used extensively in the development of Okmain, but all generated code is reviewed and rewritten by a human.