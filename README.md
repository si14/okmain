# Okmain: OK main colors

`okmain` finds the main colors of an image and makes sure they look good.

Sometimes you need to show a "dominant" color (or colors) of an image. It can be a background or a placeholder.
There are several ways of doing that; a popular quick-and-dirty method is to
resize the image to a handful of pixels, or even just one.

However, this method tends to produce muted, dirty-looking colors. Most images have clusters of colors:
the dominant color of a lush green field with a clear sky above it is not an average of blue and green,
it's either blue or green. Okmain clusters colors explicitly, recovering the dominant ones while keeping them sharp and
clear.

[<img src="https://github.com/si14/okmain/blob/main/comparison.png?raw=true" width="500" title="Comparison of colors extracted via 1x1 resize and Okmain" alt="Comparison of colors extracted via 1x1 resize and Okmain"/>](https://github.com/si14/okmain/blob/main/comparison.png?raw=true)

`okmain` is available in:

* [Rust](https://github.com/si14/okmain/tree/main/crates/okmain)
* [Python wrapper over native Rust code](https://github.com/si14/okmain/tree/main/python)
