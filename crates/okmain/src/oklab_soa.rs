const MAX_SAMPLE_SIZE: usize = 250_000; // 500x500

#[derive(Debug)]
pub struct SampledOklabSoA {
    pub width: u16,
    pub height: u16,
    pub l: Vec<f32>,
    pub a: Vec<f32>,
    pub b: Vec<f32>,
}

impl SampledOklabSoA {
    pub fn new(width: u16, height: u16, sample_size: usize) -> Self {
        Self {
            width,
            height,
            l: Vec::with_capacity(sample_size),
            a: Vec::with_capacity(sample_size),
            b: Vec::with_capacity(sample_size),
        }
    }

    #[inline(always)]
    pub fn push(&mut self, l: f32, a: f32, b: f32) {
        self.l.push(l);
        self.a.push(a);
        self.b.push(b);
    }
}

/// Compute the block size N for block averaging.
///
/// - If total pixels <= MAX_SAMPLE_SIZE, returns 1 (no averaging).
/// - Otherwise, ceil(sqrt(total / MAX_SAMPLE_SIZE)), rounded up to the next multiple of 4.
pub fn block_size(width: u16, height: u16) -> usize {
    let total = width as usize * height as usize;
    if total <= MAX_SAMPLE_SIZE {
        return 1;
    }
    let n = ((total as f64 / MAX_SAMPLE_SIZE as f64).sqrt()).ceil() as usize;
    // Round up to next multiple of 4
    (n + 3) & !3
}

pub fn sample(width: u16, height: u16, buf: &[u8]) -> SampledOklabSoA {
    assert!(!buf.is_empty());
    assert!(width > 0);
    assert!(height > 0);
    assert!(buf.len().is_multiple_of(3));
    assert_eq!(buf.len(), width as usize * height as usize * 3);

    let w = width as usize;
    let h = height as usize;
    let n = block_size(width, height);

    let blocks_x = w.div_ceil(n);
    let blocks_y = h.div_ceil(n);
    let num_blocks = blocks_x * blocks_y;

    let mut result = SampledOklabSoA::new(blocks_x as u16, blocks_y as u16, num_blocks);

    // Per-block-column accumulators for the current block row
    let mut acc_r = vec![0.0f32; blocks_x];
    let mut acc_g = vec![0.0f32; blocks_x];
    let mut acc_b = vec![0.0f32; blocks_x];
    let mut acc_count = vec![0u32; blocks_x];

    for by in 0..blocks_y {
        let y_start = by * n;
        let y_end = (y_start + n).min(h);

        for y in y_start..y_end {
            let row_offset = y * w * 3;
            let row = &buf[row_offset..row_offset + w * 3];

            for bx in 0..blocks_x {
                let x_start = bx * n;
                let x_end = (x_start + n).min(w);
                let chunk_start = x_start * 3;
                let chunk_end = x_end * 3;

                for pixel in row[chunk_start..chunk_end].chunks_exact(3) {
                    acc_r[bx] += fast_srgb8::srgb8_to_f32(pixel[0]);
                    acc_g[bx] += fast_srgb8::srgb8_to_f32(pixel[1]);
                    acc_b[bx] += fast_srgb8::srgb8_to_f32(pixel[2]);
                    acc_count[bx] += 1;
                }
            }
        }

        // Emit one averaged Oklab sample per block column, then reset accumulators
        for bx in 0..blocks_x {
            let count = acc_count[bx] as f32;
            let avg_r = acc_r[bx] / count;
            let avg_g = acc_g[bx] / count;
            let avg_b = acc_b[bx] / count;

            let oklab = oklab::linear_srgb_to_oklab(oklab::LinearRgb {
                r: avg_r,
                g: avg_g,
                b: avg_b,
            });

            result.push(oklab.l, oklab.a, oklab.b);

            acc_r[bx] = 0.0;
            acc_g[bx] = 0.0;
            acc_b[bx] = 0.0;
            acc_count[bx] = 0;
        }
    }

    assert_eq!(result.l.len(), num_blocks);
    assert_eq!(result.a.len(), num_blocks);
    assert_eq!(result.b.len(), num_blocks);

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    fn assert_all_len(soa: &SampledOklabSoA, len: usize) {
        assert_eq!(soa.l.len(), len);
        assert_eq!(soa.a.len(), len);
        assert_eq!(soa.b.len(), len);
    }

    #[test]
    fn new_allocates_capacity() {
        let soa = SampledOklabSoA::new(0, 0, 100);
        assert_eq!(soa.l.capacity(), 100);
        assert_all_len(&soa, 0);
    }

    #[test]
    fn push_adds_values() {
        let mut soa = SampledOklabSoA::new(0, 0, 2);
        soa.push(0.5, 0.1, 0.2);
        soa.push(0.7, 0.3, 0.4);
        assert_eq!(soa.l, vec![0.5, 0.7]);
    }

    #[test]
    fn block_size_small_image() {
        // 100*100 = 10_000 < MAX_SAMPLE_SIZE => block_size = 1
        assert_eq!(block_size(100, 100), 1);
    }

    #[test]
    fn block_size_at_boundary() {
        // 499*499 = 249,001 < MAX_SAMPLE_SIZE => block_size = 1
        assert_eq!(block_size(499, 499), 1);
    }

    #[test]
    fn block_size_large_image() {
        // 1000*1000 = 1_000_000, ratio = 2, sqrt(2) ≈ 1.41, ceil = 2, round to 4
        assert_eq!(block_size(1000, 1000), 4);
    }

    #[test]
    fn sample_1x1() {
        let result = sample(1, 1, &[255, 0, 0]);
        assert_all_len(&result, 1);
        assert_eq!(result.width, 1);
        assert_eq!(result.height, 1);
    }

    #[test]
    fn sample_2x2() {
        let buf = vec![255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255];
        let result = sample(2, 2, &buf);
        // block_size=1 for 4 pixels, so 4 blocks
        assert_all_len(&result, 4);
        assert_eq!(result.width, 2);
        assert_eq!(result.height, 2);
    }

    #[test]
    fn sample_small_image() {
        let result = sample(3, 3, &[0u8; 27]);
        assert_all_len(&result, 9);
    }

    #[test]
    fn sample_medium_image() {
        let result = sample(100, 100, &vec![128u8; 30_000]);
        assert_all_len(&result, 10_000);
    }

    #[test]
    fn sample_large_image_deterministic_count() {
        let result = sample(1000, 1000, &vec![64u8; 3_000_000]);
        // block_size = 4, blocks_x = 250, blocks_y = 250
        let expected = 250 * 250;
        assert_all_len(&result, expected);
        assert_eq!(result.width, 250);
        assert_eq!(result.height, 250);
    }

    #[test]
    fn sample_at_boundary() {
        let pixels = 499 * 499;
        let result = sample(499, 499, &vec![50u8; pixels * 3]);
        assert_all_len(&result, pixels);
    }

    #[test]
    #[should_panic(expected = "assertion failed: !buf.is_empty()")]
    fn panics_on_empty_buffer() {
        sample(1, 1, &[]);
    }

    #[test]
    #[should_panic(expected = "assertion failed: width > 0")]
    fn panics_on_zero_width() {
        sample(0, 1, &[0, 0, 0]);
    }

    #[test]
    #[should_panic(expected = "assertion failed: height > 0")]
    fn panics_on_zero_height() {
        sample(1, 0, &[0, 0, 0]);
    }

    #[test]
    #[should_panic]
    fn panics_on_mismatched_size() {
        sample(2, 2, &[0, 0, 0]);
    }

    #[test]
    #[should_panic]
    fn panics_on_non_multiple_of_3() {
        sample(1, 1, &[0, 0, 0, 0]);
    }
}
