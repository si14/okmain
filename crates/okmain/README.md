# Okmain: OK main colors of your image

`okmain` finds the main colors of an image and makes sure they look good.

[Crate](https://crates.io/crates/okmain) •
[GitHub](https://github.com/si14/okmain) •
[Docs](https://docs.rs/okmain/)

Technical highlights:

- Color operations in a state-of-the-art perceptually linear color
  space ([Oklab](https://bottosson.github.io/posts/oklab/))
- Fast custom K-means color clustering, optimized for auto-vectorization (confirmed with disassembly)
- Position- and visual prominence-based color prioritization (more central and higher Oklab chroma pixels are more
  important)
- Tunable parameters (see [`colors_with_config`])
- Minimal and stable dependencies
- [Python interface](https://github.com/si14/okmain/tree/main/python)

Read more about the Okmain algorithm in the [blog post](https://dgroshev.com/blog/okmain/).

## Usage

Add the dependency in your `Cargo.toml`:

```toml
[dependencies]
okmain = { version = "0.1", features = [] }
```

Then call [`colors`] on image bytes:

```rust
fn demo() {
    let input = okmain::InputImage::from_bytes(
        2, 2,
        &[255, 0, 0, 0, 255, 0,
            255, 0, 0, 255, 0, 0]
    ).unwrap();

    let output = okmain::colors(input);

    let green = rgb::Rgb { r: 0, g: 255, b: 0 };
    let red = rgb::Rgb { r: 255, g: 0, b: 0 };
    assert_eq!(vec![red, green], output)
}
```

Or if you need interop with the [`image`](https://crates.io/crates/image) crate:

```rust
fn image_demo() {
    let img = image::ImageBuffer::from_raw(
        1, 2,
        vec![255, 0, 0,
             255, 0, 0]
    ).unwrap();
    let input = okmain::InputImage::try_from(&img).unwrap();

    let output = okmain::colors(input);

    let red = rgb::Rgb { r: 255, g: 0, b: 0 };
    assert_eq!(vec![red], output);
}
```

## Features

- `image`: interop with the [`image`](https://crates.io/crates/image) crate
- `unstable`: features with no stability guarantees (currently, debug information)