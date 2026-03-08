# Okmain: OK main colors (Python edition)

`okmain` finds the main colors of an image and makes sure they look good.

[PyPI Package](https://pypi.org/project/okmain/) •
[GitHub](https://github.com/si14/okmain) •
[Rust crate](https://crates.io/crates/okmain)

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
- Fast custom K-means color clustering, optimized for auto-vectorization (confirmed with disassembly)
- Position- and visual prominence-based color prioritization (more central and higher Oklab chroma pixels tend to be
  more important)
- Tunable parameters (see optional kwargs on `colors`)

Read more about Okmain in the [blog post](https://dgroshev.com/blog/okmain/).

## Usage

Install the package:

```
uv add okmain
```

Call `okmain.colors()` on a PIL/Pillow image to get back a list of `RGB` colors:

```python
import okmain
from PIL import Image

test_image = Image.open("test_image.jpeg")
dominant_colors = okmain.colors(test_image)
# dominant_colors are [okmain.RGB(r=..., g=..., b=...), ...)

css_hex = dominant_colors[0].to_hex()
# css_hex is a string like "#AABBCC"
```
