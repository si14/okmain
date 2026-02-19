#![doc = include_str!("lib.md")]
#![cfg_attr(docsrs, feature(doc_cfg))]

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
pub use rgb;
use snafu::prelude::*;
#[cfg(feature = "image")]
use std::ops::Deref;

#[cfg(feature = "image")]
pub use image;

const U16_MAX: u16 = u16::MAX;

/// Default saturation threshold for the mask. Pixels within the central rectangle defined by
/// this threshold have a weight of 1.0, with linear falloff to 0.1 at the image borders.
/// See [`colors_with_config`] for details.
pub const DEFAULT_MASK_SATURATED_THRESHOLD: f32 = 0.3;

/// The default weight of the center-priority mask. The closer this value is
/// to 1.0, the more the mask reduces the contribution of peripheral pixels to color scoring.
/// See [`colors_with_config`] for details.
pub const DEFAULT_MASK_WEIGHT: f32 = 1.0;

/// Default weight of the number of pixels assigned to each color.
/// The more pixels are assigned to a color, the more likely it is to be the dominant color.
/// See [`colors_with_config`] for details.
pub const DEFAULT_WEIGHTED_COUNTS_WEIGHT: f32 = 0.3;

/// Default weight of the chroma component of the color.
/// The more saturated the color is, the more prominent it is. This is more true in Oklab,
/// in Oklab, since it accounts, to an extent, for low luminance affecting perceived saturation.
/// The more prominent the color is, the more likely it is to be the dominant color.
/// See [`colors_with_config`] for details.
pub const DEFAULT_CHROMA_WEIGHT: f32 = 0.7;

// todo: verify
#[inline]
fn distance_mask(saturated_threshold: f32, width: u16, height: u16, x: u16, y: u16) -> f32 {
    let width = width as f32;
    let height = height as f32;
    let x = x as f32;
    let y = y as f32;

    // For simplicity, let's operate in the top left quadrant only
    let middle_x = width / 2.0;
    let x = if x <= middle_x { x } else { width - x };

    let middle_y = height / 2.0;
    let y = if y <= middle_y { y } else { height - y };

    let x_threshold = middle_x * saturated_threshold;
    let y_threshold = middle_y * saturated_threshold;

    let x_contribution = f32::min(0.1 + 0.9 * (x / x_threshold), 1.0);
    let y_contribution = f32::min(0.1 + 0.9 * (y / y_threshold), 1.0);

    f32::min(x_contribution, y_contribution)
}

/// Errors that can occur when constructing an [`InputImage`].
#[derive(Debug, Snafu)]
#[non_exhaustive]
pub enum InputImageError {
    /// Image width and height must be larger than 0.
    #[snafu(display("image size must be positive"))]
    ZeroImageSize,

    /// Buffer length must be a multiple of 3.
    #[snafu(display("buffer length {len} is not a multiple of 3"))]
    InvalidBufferLength { len: usize },

    /// Buffer must not be empty.
    #[snafu(display("buffer is empty"))]
    EmptyBuffer,

    /// Image size doesn't match the buffer size.
    #[snafu(display("image size ({width}x{height}) doesn't match the buffer size ({buf_size})"))]
    ImageSizeMismatch {
        width: u16,
        height: u16,
        buf_size: usize,
    },

    /// Image dimensions are too large, max image size is [`u16::MAX`] x [`u16::MAX`].
    #[snafu(display(
        "image dimensions are too large, max image size is {U16_MAX}x{U16_MAX}, \
        got {width}x{height}"
    ))]
    ImageDimensionsTooLarge { width: u32, height: u32 },
}

/// A reference to image bytes with the image size attached.
///
/// Can either be constructed directly using a slice of `RGBRGBRGB…` image bytes with
/// [`InputImage::from_bytes`], or, if the `image` feature is enabled, with [`InputImage::try_from`]
/// from [`image::ImageBuffer`].
///
/// Validates the parameters during construction, see [`InputImageError`].
#[derive(Debug, Copy, Clone)]
pub struct InputImage<'a> {
    width: u16,
    height: u16,
    buf: &'a [u8],
}

impl InputImage<'_> {
    /// Creates an [`InputImage`] from raw byte slice in `RGBRGBRGB…` format. The bytes
    /// are assumed to be in sRGB. [`InputImage`] only holds a reference to the image bytes,
    /// so it's cheap to create.
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
#[cfg_attr(docsrs, doc(cfg(feature = "image")))]
impl<'a, Container> TryFrom<&'a image::ImageBuffer<image::Rgb<u8>, Container>> for InputImage<'a>
where
    Container: Deref<Target = [<image::Rgb<u8> as image::Pixel>::Subpixel]> + 'a,
{
    type Error = InputImageError;

    /// Creates an [`InputImage`] from an [`image::ImageBuffer`]. [`InputImage`] only holds
    /// a reference to the image bytes, so it's cheap to create.
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

/// Extract dominant colors from something that can provide sRGB bytes with default parameters.
///
/// For example, from a raw byte buffer:
///
/// ```
/// let input = okmain::InputImage::from_bytes(
///   2, 2,
///   &[255, 0, 0,   0, 255, 0,
///     255, 0, 0,   255, 0, 0]
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
/// is assumed to be `RGBRGBRGB…`.
///
/// Or, if the `image` feature is enabled, from [`image::ImageBuffer`]:
///
#[cfg_attr(
    not(feature = "image"),
    doc = r##"
```compile_fail
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
"##
)]
#[cfg_attr(
    feature = "image",
    doc = r##"
```
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
"##
)]
///
/// Returns a vector of up to four dominant colors in [`rgb::RGB8`], sorted by dominance
/// (the most dominant color first). If some colors are too close, fewer colors
/// might be returned.
///
/// See [`colors_with_config`] for the same function with more tuning parameters
/// and [`colors_debug`] for extra debug info.
pub fn colors(input: InputImage) -> Vec<rgb::RGB8> {
    colors_with_config(
        input,
        DEFAULT_MASK_SATURATED_THRESHOLD,
        DEFAULT_MASK_WEIGHT,
        DEFAULT_WEIGHTED_COUNTS_WEIGHT,
        DEFAULT_CHROMA_WEIGHT,
    )
    .expect("hardcoded config values should never fail")
}

/// Config errors that [`colors_with_config`] can return.
///
/// Color extraction itself is infallible.
#[derive(Debug, Snafu)]
#[non_exhaustive]
pub enum ConfigError {
    /// Invalid `mask_saturated_threshold` (must be in `[0, 0.5)`)
    #[snafu(display("invalid mask_saturated_threshold: {threshold} (must be in [0, 0.5))"))]
    InvalidMaskSaturatedThreshold { threshold: f32 },
    /// Invalid `mask_weight` (must be in `[0, 1]`)
    #[snafu(display("invalid mask_weight: {weight} (must be in [0, 1])"))]
    InvalidMaskWeight { weight: f32 },
    /// Invalid `weighted_counts_weight` (must be in `[0, 1]`)
    #[snafu(display("invalid weighted_counts_weight: {weight} (must be in [0, 1])"))]
    InvalidWeightedCountsWeight { weight: f32 },
    /// Invalid `chroma_weight` (must be in `[0, 1]`)
    #[snafu(display("invalid chroma_weight: {weight} (must be in [0, 1])"))]
    InvalidChromaWeight { weight: f32 },
    /// `mask_weighted_counts_weight` and `chroma_weight` don't add up to 1.0
    #[snafu(display("mask_weighted_counts_weight and chroma_weight don't add up to 1.0"))]
    WeightsDontAddUp {
        mask_weighted_counts_weight: f32,
        chroma_weight: f32,
    },
}

/// Extract dominant colors from something that can provide sRGB bytes, providing
/// explicit config values.
///
/// See [`colors`] for a description of the basic input and output.
///
/// **`mask_saturated_threshold`**: The algorithm uses a mask to prioritize central pixels while
/// considering the relative color dominance. The mask is a 1.0-weight rectangle starting
/// at `mask_saturated_threshold * 100%` and finishing at `(1.0 - mask_saturated_threshold) * 100%`
/// on both axes, with linear weight falloff from 1.0 at the border of the rectangle
/// to 0.1 at the border of the image. Must be in the `[0.0, 0.5)` range.
///
/// **`mask_weight`**: The weight of the mask, which can be used to reduce the impact of the mask
/// on less-central pixels. By default, it's set to 1.0, but by reducing this number you can
/// increase the relative contribution of peripheral pixels. Must be in the `[0.0, 1.0]` range.
///
/// **`mask_weighted_counts_weight`**: After the number of pixels belonging to every color
/// is added up (with the mask reducing the contribution of peripheral pixels), the sums are
/// normalized to add up to 1.0, and used as a part of the final score that decides the
/// ordering of the colors. This parameter sets the relative weight of this component
/// in the final score.
/// Must be in the `[0.0, 1.0]` range and add up to 1.0 together with `chroma_weight`.
///
/// **`chroma_weight`**: For each color its saturation (Oklab chroma) is used to prioritize
/// colors that are visually more prominent. This parameter controls the relative contribution
/// of chroma into the final score. Must be in the `[0.0, 1.0]` range and
/// add up to 1.0 together with `mask_weighted_counts_weight`.
pub fn colors_with_config(
    input: InputImage,
    mask_saturated_threshold: f32,
    mask_weight: f32,
    mask_weighted_counts_weight: f32,
    chroma_weight: f32,
) -> Result<Vec<rgb::RGB8>, ConfigError> {
    colors_debug(
        input,
        mask_saturated_threshold,
        mask_weight,
        mask_weighted_counts_weight,
        chroma_weight,
    )
    .map(|(colors, _)| colors)
}

/// Debug details about every "centroid" in the Oklab color space and its score.
/// Available only if the `unstable` feature is enabled.
#[cfg(feature = "unstable")]
#[cfg_attr(docsrs, doc(cfg(feature = "unstable")))]
#[derive(Debug, Clone, Copy)]
pub struct ScoredCentroid {
    /// Oklab color of the centroid
    pub oklab: Oklab,
    /// sRGB color of the centroid
    pub rgb: rgb::RGB8,
    /// The fraction of pixels assigned to this centroid, with a mask reducing the impact of
    /// peripheral pixels applied
    pub mask_weighted_counts: f32,
    /// The score of the centroid based on mask-weighted pixel counts
    pub mask_weighted_counts_score: f32,
    /// Centroid's Oklab chroma (calculated from the Oklab value and normalized to `[0, 1]`)
    pub chroma: f32,
    /// The score of the centroid based on chroma
    pub chroma_score: f32,
    /// The final score of the centroid, combining two scores based on provided weights
    pub final_score: f32,
}

#[cfg(not(feature = "unstable"))]
#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
struct ScoredCentroid {
    oklab: Oklab,
    rgb: rgb::RGB8,
    chroma: f32,
    mask_weighted_counts: f32,
    chroma_score: f32,
    mask_weighted_counts_score: f32,
    final_score: f32,
}

/// Debug info returned by [`colors_debug`]. Available only if the `unstable` feature is enabled.
#[cfg(feature = "unstable")]
#[cfg_attr(docsrs, doc(cfg(feature = "unstable")))]
#[derive(Debug)]
pub struct DebugInfo {
    /// The Okmain algorithm looks for k-means centroids in the Oklab color space. This
    /// field contains details about the centroids that were found in the image.
    pub scored_centroids: Vec<ScoredCentroid>,
    /// The number of iterations the k-means algorithm took until the position
    /// of centroids stopped changing. `kmeans_loop_iterations` is a vector, because Okmain
    /// can re-run k-means with a lower number of centroids if some of the discovered centroids
    /// are too close.
    pub kmeans_loop_iterations: Vec<usize>,
    /// Did kmeans search converge? If not, it was cut off by the maximum number of iterations.
    /// It's a vector for the same reason `kmeans_loop_iterations` is.
    pub kmeans_converged: Vec<bool>,
}
#[cfg(not(feature = "unstable"))]
#[allow(dead_code)]
#[derive(Debug)]
struct DebugInfo {
    scored_centroids: Vec<ScoredCentroid>,
    kmeans_loop_iterations: Vec<usize>,
    kmeans_converged: Vec<bool>,
}

/// Same as [`colors_with_config`], but also returns debug info that can be used to
/// debug the algorithm's behavior. See [`DebugInfo`] for details. No stability guarantees
/// are provided for [`DebugInfo`], so this function is opt-in via the `unstable` feature.
///
/// See [`colors`] and [`colors_with_config`] for arguments.
#[cfg(feature = "unstable")]
#[cfg_attr(docsrs, doc(cfg(feature = "unstable")))]
pub fn colors_debug(
    input: InputImage,
    mask_saturated_threshold: f32,
    mask_weight: f32,
    mask_weighted_counts_weight: f32,
    chroma_weight: f32,
) -> Result<(Vec<rgb::RGB8>, DebugInfo), ConfigError> {
    colors_internal(
        input,
        mask_saturated_threshold,
        mask_weight,
        mask_weighted_counts_weight,
        chroma_weight,
    )
}
#[cfg(not(feature = "unstable"))]
fn colors_debug(
    input: InputImage,
    mask_saturated_threshold: f32,
    mask_weight: f32,
    mask_weighted_counts_weight: f32,
    chroma_weight: f32,
) -> Result<(Vec<rgb::RGB8>, DebugInfo), ConfigError> {
    colors_internal(
        input,
        mask_saturated_threshold,
        mask_weight,
        mask_weighted_counts_weight,
        chroma_weight,
    )
}

fn colors_internal(
    input: InputImage,
    mask_saturated_threshold: f32,
    mask_weight: f32,
    mask_weighted_counts_weight: f32,
    chroma_weight: f32,
) -> Result<(Vec<rgb::RGB8>, DebugInfo), ConfigError> {
    ensure!(
        (0.0..0.5).contains(&mask_saturated_threshold),
        InvalidMaskSaturatedThresholdSnafu {
            threshold: mask_saturated_threshold
        }
    );
    ensure!(
        (0.0..=1.0).contains(&mask_weight),
        InvalidMaskWeightSnafu {
            weight: mask_weight
        }
    );
    ensure!(
        (0.0..=1.0).contains(&mask_weighted_counts_weight),
        InvalidWeightedCountsWeightSnafu {
            weight: mask_weighted_counts_weight
        }
    );
    ensure!(
        (0.0..=1.0).contains(&chroma_weight),
        InvalidChromaWeightSnafu {
            weight: chroma_weight
        }
    );
    ensure!(
        mask_weighted_counts_weight + chroma_weight == 1.0,
        WeightsDontAddUpSnafu {
            mask_weighted_counts_weight,
            chroma_weight,
        }
    );

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
            mask_saturated_threshold,
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

            let mask_weighted_counts = weighted_counts[i];
            let mask_weighted_counts_score = mask_weighted_counts * mask_weighted_counts_weight;
            // todo: try to normalise/stretch the chromas
            // Chroma goes up to 0.5 in practice
            let chroma = (oklab.a * oklab.a + oklab.b * oklab.b).sqrt() * 2f32;
            let chroma_score = chroma * chroma_weight;
            let final_score =
                mask_weighted_counts_weight * weighted_counts[i] + chroma_weight * chroma;
            ScoredCentroid {
                oklab,
                rgb,
                mask_weighted_counts,
                mask_weighted_counts_score,
                chroma,
                chroma_score,
                final_score,
            }
        })
        .collect::<Vec<_>>();

    // Sort by score descending
    scored_centroids.sort_by(|a, b| a.final_score.total_cmp(&b.final_score).reverse());

    let result = scored_centroids.iter().map(|sc| sc.rgb).collect();

    Ok((
        result,
        DebugInfo {
            scored_centroids,
            kmeans_loop_iterations: centroids_result.loop_iterations,
            kmeans_converged: centroids_result.converged,
        },
    ))
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
        let input = InputImage::try_from(&img).unwrap();
        let colors = colors(input);
        assert!(!colors.is_empty());
    }
}
