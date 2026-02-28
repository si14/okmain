# Okmain: OK main colors (Python edition)

`okmain` finds the main colors of an image and makes sure they look good.

[Package](https://pypi.org/project/okmain/) •
[GitHub](https://github.com/si14/okmain)

Sometimes you need to show a "dominant" color (or colors) of an image. It can be a background or a placeholder.
There are several ways of doing that; a popular quick-and-dirty method is to
resize the image to a handful of pixels, or even just one.

However, this method tends to produce muted, dirty-looking colors. Most images have clusters of colors:
the dominant color of a lush green field with a clear sky above it is not an average of blue and green,
it's either blue or green. Okmain clusters colors explicitly, recovering the dominant ones while keeping them sharp and
clear.

[<img src="https://github.com/si14/okmain/blob/main/comparison.png?raw=true" width="500" title="Comparison of colors extracted via 1x1 resize and Okmain" alt="Comparison of colors extracted via 1x1 resize and Okmain"/>](https://github.com/si14/okmain/blob/main/comparison.png?raw=true)

Technical highlights:

- Color operations in a state-of-the-art perceptually linear color space
  ([Oklab](https://bottosson.github.io/posts/oklab/))
- Rust implementation for speed and safety
- Fast custom K-means color clustering, optimized for auto-vectorization (confirmed with disassembly)
- Position- and visual prominence-based color prioritization (more central and higher Oklab chroma pixels tend to be
  more important)
- Tunable parameters (see `colors` kwargs)

Read more about the Okmain algorithm in the [blog post](https://dgroshev.com/blog/okmain/).

## Usage

Just call `okmain.colors()` on a PIL/Pillow image.

(documentation coming soon)