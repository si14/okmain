#[cfg(feature = "_debug")]
pub mod kmeans;
#[cfg(not(feature = "_debug"))]
mod kmeans;
#[cfg(feature = "_debug")]
pub mod rng;
#[cfg(not(feature = "_debug"))]
mod rng;
#[cfg(feature = "_debug")]
pub mod sample;
#[cfg(not(feature = "_debug"))]
mod sample;

use oklab::{oklab_to_srgb, Oklab};
pub use rgb::RGB8;
use snafu::prelude::*;
#[cfg(feature = "image")]
use std::ops::Deref;

const U16_MAX: u16 = u16::MAX;

#[derive(Debug, Snafu)]
#[non_exhaustive]
pub enum InputImageError {
    #[snafu(display("image size must be positive"))]
    ZeroImageSize,

    #[snafu(display("buffer length {len} is not a multiple of 3"))]
    InvalidBufferLength { len: usize },

    #[snafu(display("buffer is empty"))]
    EmptyBuffer,

    #[snafu(display("image size ({width}x{height}) doesn't match the buffer size ({buf_size})"))]
    ImageSizeMismatch {
        width: u16,
        height: u16,
        buf_size: usize,
    },

    #[snafu(display(
        "image dimensions are too large, max image size is {U16_MAX}x{U16_MAX}, \
        got {width}x{height}"
    ))]
    ImageDimensionsTooLarge { width: u32, height: u32 },
}

// The mask is a 1.0-weight rectangle starting at 30% and finishing at 70% on both axes,
// with linear falloff from 1.0 at the border of the rectangle to 0.1 at the border
// of the image
const DEFAULT_MASK_SATURATED_MIDDLE_THRESHOLD: f32 = 0.3;

pub const DEFAULT_MASK_WEIGHT: f32 = 1.0;
pub const DEFAULT_MASK_WEIGHTED_COUNTS_WEIGHT: f32 = 0.3;
pub const DEFAULT_CHROMA_WEIGHT: f32 = 0.7;

// todo: verify
#[inline]
fn distance_mask(saturated_middle_threshold: f32, width: u16, height: u16, x: u16, y: u16) -> f32 {
    let width = width as f32;
    let height = height as f32;
    let x = x as f32;
    let y = y as f32;

    // For simplicity, let's operate in the top left quadrant only
    let middle_x = width / 2.0;
    let x = if x <= middle_x { x } else { width - x };

    let middle_y = height / 2.0;
    let y = if y <= middle_y { y } else { height - y };

    let x_threshold = middle_x * saturated_middle_threshold;
    let y_threshold = middle_y * saturated_middle_threshold;

    let x_contribution = f32::min(0.1 + 0.9 * (x / x_threshold), 1.0);
    let y_contribution = f32::min(0.1 + 0.9 * (y / y_threshold), 1.0);

    f32::min(x_contribution, y_contribution)
}

/// A structure used as a façade for the image bytes.
#[derive(Debug, Copy, Clone)]
pub struct InputImage<'a> {
    width: u16,
    height: u16,
    buf: &'a [u8],
}

impl InputImage<'_> {
    pub fn from_bytes(
        width: u16,
        height: u16,
        buf: &[u8],
    ) -> Result<InputImage<'_>, InputImageError> {
        ensure!(!buf.is_empty(), EmptyBufferSnafu);
        ensure!(width > 0 && height > 0, ZeroImageSizeSnafu);
        ensure!(
            buf.len().is_multiple_of(3),
            InvalidBufferLengthSnafu { len: buf.len() }
        );
        ensure!(
            buf.len() == (width as usize) * (height as usize) * 3,
            ImageSizeMismatchSnafu {
                width,
                height,
                buf_size: buf.len()
            }
        );

        Ok(InputImage { width, height, buf })
    }
}

#[cfg(feature = "image")]
impl<'a, Container> TryFrom<&'a image::ImageBuffer<image::Rgb<u8>, Container>> for InputImage<'a>
where
    Container: Deref<Target = [<image::Rgb<u8> as image::Pixel>::Subpixel]> + 'a,
{
    type Error = InputImageError;

    fn try_from(
        img: &'a image::ImageBuffer<image::Rgb<u8>, Container>,
    ) -> Result<Self, Self::Error> {
        let width = u16::try_from(img.width()).map_err(|_| {
            ImageDimensionsTooLargeSnafu {
                width: img.width(),
                height: img.height(),
            }
            .build()
        })?;
        let height = u16::try_from(img.height()).map_err(|_| {
            ImageDimensionsTooLargeSnafu {
                width: img.width(),
                height: img.height(),
            }
            .build()
        })?;
        Self::from_bytes(width, height, img.as_raw().deref())
    }
}

/// Extract dominant colors from something that can provide sRGB bytes.
///
/// For example, from a raw byte buffer:
///
/// ```
/// let input = okmain::InputImage::from_bytes(2, 2,
///   &[255, 0, 0, 0, 255, 0,
///     255, 0, 0, 255, 0, 0]
/// ).unwrap();
///
/// let output = okmain::colors(input);
///
/// let green = rgb::Rgb { r: 0, g: 255, b: 0 };
/// let red = rgb::Rgb { r: 255, g: 0, b: 0 };
/// assert_eq!(vec![red, green], output)
/// ```
///
/// The buffer must be non-empty, and its length must be a multiple of 3. The byte layout
/// is assumed to be RGBRGBRGB….
///
/// Or, if the `image` feature is enabled, from [`image::ImageBuffer`]:
///
#[cfg_attr(
    not(feature = "image"),
    doc = r##"
```compile_fail
let img = image::ImageBuffer::from_raw(2, 1, vec![255, 0, 0, 255, 0, 0]).unwrap();
let input = okmain::InputImage::try_from(&img).unwrap();

let output = okmain::colors(input);

let red = rgb::Rgb { r: 255, g: 0, b: 0 };
assert_eq!(vec![red], output);
```
"##
)]
#[cfg_attr(
    feature = "image",
    doc = r##"
```
let img = image::ImageBuffer::from_raw(2, 1, vec![255, 0, 0, 255, 0, 0]).unwrap();
let input = okmain::InputImage::try_from(&img).unwrap();

let output = okmain::colors(input);

let red = rgb::Rgb { r: 255, g: 0, b: 0 };
assert_eq!(vec![red], output);
```
"##
)]
///
/// Returns a vector of up to four dominant colors in [`RGB8`], sorted by dominance
/// (the most dominant color first). If some colors are too close, fewer colors
/// might be returned.
///
/// See also [`colors_extra`] for the same function with more tuning parameters
/// and debug info.
pub fn colors(input: InputImage) -> Vec<RGB8> {
    colors_extra(
        input,
        DEFAULT_MASK_SATURATED_MIDDLE_THRESHOLD,
        DEFAULT_MASK_WEIGHT,
        DEFAULT_MASK_WEIGHTED_COUNTS_WEIGHT,
        DEFAULT_CHROMA_WEIGHT,
    )
}

pub fn colors_extra(
    input: InputImage,
    mask_saturated_middle_threshold: f32,
    mask_weight: f32,
    mask_weighted_counts_weight: f32,
    chroma_weight: f32,
) -> Vec<RGB8> {
    colors_extra_debug(
        input,
        mask_saturated_middle_threshold,
        mask_weight,
        mask_weighted_counts_weight,
        chroma_weight,
    )
    .0
}

#[derive(Debug, Clone, Copy)]
pub struct ScoredCentroid {
    pub oklab: Oklab,
    pub rgb: RGB8,
    pub score: f32,
}

#[derive(Debug)]
pub struct DebugInfo {
    pub scored_centroids: Vec<ScoredCentroid>,
    pub kmeans_loop_iterations: Vec<usize>,
    pub kmeans_converged: Vec<bool>,
}

pub fn colors_extra_debug(
    input: InputImage,
    mask_saturated_middle_threshold: f32,
    mask_weight: f32,
    mask_weighted_counts_weight: f32,
    chroma_weight: f32,
) -> (Vec<RGB8>, DebugInfo) {
    let mut rng = rng::new();

    let oklab_soa = sample::sample(input.width, input.height, input.buf);

    let centroids_result = kmeans::adaptive::find_centroids(&mut rng, &oklab_soa);

    let num_centroids = centroids_result.centroids.len();

    // Number of (downsampled) pixels assigned to each centroid, weighed by
    // how central the pixels are (the impact of centrality is controlled
    // by mask_weight)
    let mut weighted_counts = vec![0.0f32; num_centroids];
    for (i, &assignment) in centroids_result.assignments.iter().enumerate() {
        let x = (i % oklab_soa.width as usize) as u16;
        let y = (i / oklab_soa.width as usize) as u16;
        let mask_value = distance_mask(
            mask_saturated_middle_threshold,
            oklab_soa.width,
            oklab_soa.height,
            x,
            y,
        );
        // Mask goes from 1.0 of full weight to 0.1, so to dampen its impact with
        // mask_weight, we need to invert it and then invert it back
        let w = 1.0 - mask_weight * (1.0 - mask_value);
        weighted_counts[assignment] += w;
    }

    // Normalize the weighted counts, because chroma is (0..1) and we need to make
    // the weighted sum meaningful
    let total: f32 = weighted_counts.iter().sum();
    if total > 0.0 {
        for wc in &mut weighted_counts {
            *wc /= total;
        }
    }

    let mut scored_centroids = centroids_result
        .centroids
        .iter()
        .copied()
        .enumerate()
        .map(|(i, oklab)| {
            let rgb = oklab_to_srgb(oklab);

            // Chroma goes up to 0.5 in practice
            let chroma = (oklab.a * oklab.a + oklab.b * oklab.b).sqrt() * 2f32;
            let score = mask_weighted_counts_weight * weighted_counts[i] + chroma_weight * chroma;
            ScoredCentroid { oklab, rgb, score }
        })
        .collect::<Vec<_>>();

    // Sort by score descending
    scored_centroids.sort_by(|a, b| a.score.total_cmp(&b.score));

    let result = scored_centroids.iter().map(|sc| sc.rgb).collect();

    (
        result,
        DebugInfo {
            scored_centroids,
            kmeans_loop_iterations: centroids_result.loop_iterations,
            kmeans_converged: centroids_result.converged,
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    #[test]
    fn single_pixel() {
        let buf = [128, 64, 32];
        let input = InputImage::from_bytes(1, 1, &buf).unwrap();
        let colors = colors(input);
        assert_eq!(colors.len(), 1);
        // Round-trip through Oklab should be within ±1
        assert!((colors[0].r as i16 - 128).abs() <= 1);
        assert!((colors[0].g as i16 - 64).abs() <= 1);
        assert!((colors[0].b as i16 - 32).abs() <= 1);
    }

    #[test]
    fn uniform_image() {
        // 10x10 image of a single color
        let buf = [200, 100, 50].repeat(100);
        let input = InputImage::from_bytes(10, 10, &buf).unwrap();
        let colors = colors(input);
        assert_eq!(colors.len(), 1);
        assert!((colors[0].r as i16 - 200).abs() <= 1);
        assert!((colors[0].g as i16 - 100).abs() <= 1);
        assert!((colors[0].b as i16 - 50).abs() <= 1);
    }

    #[test]
    fn dominant_color_is_first() {
        // Saturated red fills most of the image, dim gray only at far edges
        let w = 20u16;
        let h = 20u16;
        let mut buf = vec![0u8; (w as usize) * (h as usize) * 3];
        for y in 0..h as usize {
            for x in 0..w as usize {
                let idx = (y * w as usize + x) * 3;
                if (2..18).contains(&x) && (2..18).contains(&y) {
                    buf[idx] = 255;
                    buf[idx + 1] = 0;
                    buf[idx + 2] = 0;
                } else {
                    buf[idx] = 40;
                    buf[idx + 1] = 40;
                    buf[idx + 2] = 40;
                }
            }
        }
        let input = InputImage::from_bytes(w, h, &buf).unwrap();
        let colors = colors(input);
        assert!(!colors.is_empty());
        // Dominant color should be reddish (high r, low g and b)
        // Red has both high chroma and high center weight
        assert!(colors[0].r > 150);
        assert!(colors[0].g < 80);
    }

    #[test]
    fn deterministic() {
        let buf = vec![255, 0, 0, 0, 255, 0, 0, 0, 255];
        let input = InputImage::from_bytes(3, 1, &buf).unwrap();
        let a = colors(input);
        let b = colors(input);
        assert_eq!(a, b);
    }

    #[test]
    fn empty_buffer() {
        let result = InputImage::from_bytes(0, 0, &[]);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("empty"));
    }

    #[test]
    fn invalid_length() {
        let result = InputImage::from_bytes(1, 2, &[1, 2]);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("of 3"));
    }

    #[test]
    fn distance_weight_center() {
        // Center of a 100x100 image
        let w = distance_mask(0.3, 50, 50, 101, 101);
        assert!((w - 1.0).abs() < 1e-6);
    }

    #[test]
    fn distance_weight_corner() {
        // Corner of image → d=1.0 → weight = 0.01
        let w = distance_mask(0.3, 0, 0, 100, 100);
        assert!((w - 0.01).abs() < 1e-6);
    }

    #[test]
    fn distance_weight_1x1() {
        // Single pixel → normalized to center → weight 1.0
        let w = distance_mask(0.3, 0, 0, 1, 1);
        assert!((w - 1.0).abs() < 1e-6);
    }

    #[cfg(feature = "image")]
    #[test]
    fn image_buffer() {
        let img = image::ImageBuffer::from_raw(2, 1, vec![255, 0, 0, 0, 255, 0]).unwrap();
        let input = InputImage::from_image(&img).unwrap();
        let colors = colors(input);
        assert!(!colors.is_empty());
    }
}
