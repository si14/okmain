# Okmain: OK main colors (Rust edition)

`okmain` finds the main colors of an image and makes sure they look good.

[Crate](https://crates.io/crates/okmain) •
[GitHub](https://github.com/si14/okmain) •
[Rust Docs](https://docs.rs/okmain/) •
[Python package](https://pypi.org/project/okmain/)

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
- Finding main colors of a reasonably sized image takes about 100ms
- Fast custom K-means color clustering, optimized for auto-vectorization (confirmed with disassembly)
- Position- and visual prominence-based color prioritization (more central and higher Oklab chroma pixels tend to be
  more important)
- Tunable parameters (see `colors_with_config`)
- Minimal and stable dependencies
- [Python wrapper](https://github.com/si14/okmain/tree/main/python)

Read more about Okmain in the [blog post](https://dgroshev.com/blog/okmain/).

## Usage

Add the dependency in your `Cargo.toml`:

```toml
[dependencies]
okmain = "0.1"
```

Then call [`colors`] on image bytes:

```rust
let input = okmain::InputImage::from_bytes(
    2, 2,
    &[255, 0, 0, 0, 255, 0,
      255, 0, 0, 255, 0, 0]
).unwrap();

let output = okmain::colors(input);

let green = rgb::Rgb { r: 0, g: 255, b: 0 };
let red = rgb::Rgb { r: 255, g: 0, b: 0 };
assert_eq!(vec![red, green], output)
```

Or if you need interop with the [`image`](https://crates.io/crates/image) crate, add the `image` feature:

```rust,ignore
let img = image::ImageBuffer::from_raw(
    1, 2,
    vec![255, 0, 0,
         255, 0, 0]
).unwrap();
let input = okmain::InputImage::try_from(&img).unwrap();

let output = okmain::colors(input);

let red = rgb::Rgb { r: 255, g: 0, b: 0 };
assert_eq!(vec![red], output);
```

## Features

- `image`: interop with the [`image`](https://crates.io/crates/image) crate
- `unstable`: features with no stability guarantees (currently, debug information)