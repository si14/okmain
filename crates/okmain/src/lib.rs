#[cfg(any(feature = "_bench", feature = "_debug"))]
pub mod kmeans;
#[cfg(not(any(feature = "_bench", feature = "_debug")))]
mod kmeans;
#[cfg(any(feature = "_bench", feature = "_debug"))]
pub mod rng;
#[cfg(not(any(feature = "_bench", feature = "_debug")))]
mod rng;
#[cfg(any(feature = "_bench", feature = "_debug"))]
pub mod sample;
#[cfg(not(any(feature = "_bench", feature = "_debug")))]
mod sample;

use snafu::prelude::*;

const U16_MAX: u16 = u16::MAX;

#[derive(Debug, Snafu)]
pub enum Error {
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

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct SRGB {
    pub r: u8,
    pub g: u8,
    pub b: u8,
}

#[derive(Debug, Copy, Clone, PartialEq)]
#[cfg(any(feature = "_bench", feature = "_debug"))]
pub struct Oklab {
    pub l: f32,
    pub a: f32,
    pub b: f32,
}

#[derive(Debug, Copy, Clone, PartialEq)]
#[cfg(not(any(feature = "_bench", feature = "_debug")))]
pub(crate) struct Oklab {
    l: f32,
    a: f32,
    b: f32,
}

impl Oklab {
    pub(crate) fn squared_distance(self, other: Self) -> f32 {
        let dl = self.l - other.l;
        let da = self.a - other.a;
        let db = self.b - other.b;
        dl.mul_add(dl, da.mul_add(da, db * db))
    }
}

const DISTANCE_WEIGHT_SATURATED_MIDDLE_THRESHOLD: f32 = 0.3;
const WEIGHTED_COUNTS_WEIGHT: f32 = 0.3;
const CHROMA_WEIGHT: f32 = 0.7;

// todo: verify
fn distance_weight(x: u16, y: u16, width: u16, height: u16) -> f32 {
    let nx = if width <= 1 {
        0.0
    } else {
        2.0 * (x as f32) / (width as f32 - 1.0) - 1.0
    };
    let ny = if height <= 1 {
        0.0
    } else {
        2.0 * (y as f32) / (height as f32 - 1.0) - 1.0
    };

    let d = nx.abs().max(ny.abs());

    if d <= DISTANCE_WEIGHT_SATURATED_MIDDLE_THRESHOLD {
        1.0
    } else {
        0.01 + 0.99
            * (1.0
                - (d - DISTANCE_WEIGHT_SATURATED_MIDDLE_THRESHOLD)
                    / (1.0 - DISTANCE_WEIGHT_SATURATED_MIDDLE_THRESHOLD))
    }
}

/// Extract dominant colors from raw sRGB bytes.
///
/// `buf` must be non-empty, and its length must be a multiple of 3.
///
/// Returns a vector of dominant colors, sorted by score (most dominant first).
pub fn colors_from_rgb_buffer(width: u16, height: u16, buf: &[u8]) -> Result<Vec<SRGB>, Error> {
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

    let oklab_soa = sample::sample(width, height, buf);

    let mut rng = rng::new();
    let result = kmeans::adaptive::find_centroids(&mut rng, &oklab_soa);

    let num_centroids = result.centroids.len();

    // Weighted counts per cluster
    let mut weighted_counts = vec![0.0f32; num_centroids];
    for (i, &assignment) in result.assignments.iter().enumerate() {
        let bx = (i % oklab_soa.width as usize) as u16;
        let by = (i / oklab_soa.width as usize) as u16;
        let w = distance_weight(bx, by, oklab_soa.width, oklab_soa.height);
        weighted_counts[assignment] += w;
    }
    let total: f32 = weighted_counts.iter().sum();
    if total > 0.0 {
        for wc in &mut weighted_counts {
            *wc /= total;
        }
    }

    // Score each centroid: 0.3 * weighted_count + 0.7 * chroma
    let mut scored: Vec<(f32, &Oklab)> = result
        .centroids
        .iter()
        .enumerate()
        .map(|(i, c)| {
            let chroma = (c.a * c.a + c.b * c.b).sqrt();
            let score = WEIGHTED_COUNTS_WEIGHT * weighted_counts[i] + CHROMA_WEIGHT * chroma;
            (score, c)
        })
        .collect();

    // Sort by score descending
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    // Convert Oklab → sRGB
    let colors = scored
        .iter()
        .map(|(_, c)| {
            let rgb = oklab::oklab_to_srgb(oklab::Oklab {
                l: c.l,
                a: c.a,
                b: c.b,
            });
            SRGB {
                r: rgb.r,
                g: rgb.g,
                b: rgb.b,
            }
        })
        .collect();

    Ok(colors)
}

/// Extract the dominant color from an [`image::ImageBuffer`] of RGB pixels.
#[cfg(feature = "image")]
pub fn colors_from_image<
    Container: std::ops::Deref<Target = [<image::Rgb<u8> as image::Pixel>::Subpixel]>,
>(
    img: &image::ImageBuffer<image::Rgb<u8>, Container>,
) -> Result<Vec<SRGB>, Error> {
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

    colors_from_rgb_buffer(width, height, img.as_raw().deref())
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    #[test]
    fn single_pixel() {
        let buf = vec![128, 64, 32];
        let colors = colors_from_rgb_buffer(1, 1, &buf).unwrap();
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
        let colors = colors_from_rgb_buffer(10, 10, &buf).unwrap();
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
        let colors = colors_from_rgb_buffer(w, h, &buf).unwrap();
        assert!(!colors.is_empty());
        // Dominant color should be reddish (high r, low g and b)
        // Red has both high chroma and high center weight
        assert!(colors[0].r > 150);
        assert!(colors[0].g < 80);
    }

    #[test]
    fn deterministic() {
        let buf = vec![255, 0, 0, 0, 255, 0, 0, 0, 255];
        let a = colors_from_rgb_buffer(3, 1, &buf).unwrap();
        let b = colors_from_rgb_buffer(3, 1, &buf).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn empty_buffer() {
        let result = colors_from_rgb_buffer(0, 0, &[]);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("empty"));
    }

    #[test]
    fn invalid_length() {
        let result = colors_from_rgb_buffer(1, 2, &[1, 2]);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("of 3"));
    }

    #[test]
    fn distance_weight_center() {
        // Center of a 100x100 image
        let w = distance_weight(50, 50, 101, 101);
        assert!((w - 1.0).abs() < 1e-6);
    }

    #[test]
    fn distance_weight_corner() {
        // Corner of image → d=1.0 → weight = 0.01
        let w = distance_weight(0, 0, 100, 100);
        assert!((w - 0.01).abs() < 1e-6);
    }

    #[test]
    fn distance_weight_1x1() {
        // Single pixel → normalized to center → weight 1.0
        let w = distance_weight(0, 0, 1, 1);
        assert!((w - 1.0).abs() < 1e-6);
    }

    #[cfg(feature = "image")]
    #[test]
    fn image_buffer() {
        let img = image::ImageBuffer::from_raw(2, 1, vec![255, 0, 0, 0, 255, 0]).unwrap();
        let colors = colors_from_image(&img).unwrap();
        assert!(!colors.is_empty());
    }
}
