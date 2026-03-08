use clap::Parser;
use image::{Rgb, RgbImage};
use std::path::PathBuf;

#[derive(Parser)]
struct Args {
    /// Number of hue steps (also the square mixing area side length)
    #[arg(long, default_value_t = 512)]
    size: u32,

    /// Directory to write color_mixing.png into (created if absent)
    #[arg(long)]
    output_dir: PathBuf,

    /// OKLab lightness for all sweep colours (0..1)
    #[arg(long, default_value_t = 0.7)]
    lightness: f32,

    /// OKLCh chroma for all sweep colours
    #[arg(long, default_value_t = 0.17)]
    chroma: f32,
}

// 5×7 pixel font. Each u8 encodes one row; bit 7 = leftmost (col 0) pixel.
fn glyph(c: char) -> [u8; 7] {
    match c {
        'O' => [
            0b01110000, 0b10001000, 0b10001000, 0b10001000, 0b10001000, 0b10001000, 0b01110000,
        ],
        'K' => [
            0b10001000, 0b10010000, 0b10100000, 0b11000000, 0b10100000, 0b10010000, 0b10001000,
        ],
        'L' => [
            0b10000000, 0b10000000, 0b10000000, 0b10000000, 0b10000000, 0b10000000, 0b11111000,
        ],
        'A' => [
            0b01110000, 0b10001000, 0b10001000, 0b11111000, 0b10001000, 0b10001000, 0b10001000,
        ],
        'B' => [
            0b11110000, 0b10001000, 0b10001000, 0b11110000, 0b10001000, 0b10001000, 0b11110000,
        ],
        'S' => [
            0b01110000, 0b10000000, 0b10000000, 0b01110000, 0b00001000, 0b00001000, 0b01110000,
        ],
        'R' => [
            0b11110000, 0b10001000, 0b10001000, 0b11110000, 0b10100000, 0b10010000, 0b10001000,
        ],
        'G' => [
            0b01110000, 0b10000000, 0b10000000, 0b10011000, 0b10001000, 0b10001000, 0b01110000,
        ],
        _ => [0; 7],
    }
}

fn draw_str(img: &mut RgbImage, text: &str, x0: u32, y0: u32, scale: u32, color: Rgb<u8>) {
    let stride = 6 * scale; // 5px char + 1px gap, scaled
    for (i, c) in text.chars().enumerate() {
        let g = glyph(c.to_ascii_uppercase());
        let cx = x0 + i as u32 * stride;
        for (row, g_row) in g.iter().enumerate() {
            for col in 0..5usize {
                if (g_row >> (7 - col)) & 1 != 0 {
                    for dy in 0..scale {
                        for dx in 0..scale {
                            let fx = cx + col as u32 * scale + dx;
                            let fy = y0 + row as u32 * scale + dy;
                            if fx < img.width() && fy < img.height() {
                                img.put_pixel(fx, fy, color);
                            }
                        }
                    }
                }
            }
        }
    }
}

fn text_dims(text: &str, scale: u32) -> (u32, u32) {
    let n = text.len() as u32;
    let w = if n == 0 {
        0
    } else {
        n * 5 * scale + (n - 1) * scale
    };
    (w, 7 * scale)
}

/// Draw text centred at (cx, cy) with a 1-pixel black drop shadow.
fn draw_label(img: &mut RgbImage, text: &str, cx: u32, cy: u32, scale: u32) {
    let (tw, th) = text_dims(text, scale);
    let x0 = cx.saturating_sub(tw / 2);
    let y0 = cy.saturating_sub(th / 2);
    draw_str(img, text, x0 + 1, y0 + 1, scale, Rgb([0, 0, 0]));
    draw_str(img, text, x0, y0, scale, Rgb([255, 255, 255]));
}

fn generate_sweep_lab(n: u32, l: f32, c: f32) -> Vec<oklab::Oklab> {
    (0..n)
        .map(|i| {
            let h = std::f32::consts::TAU * i as f32 / n as f32;
            oklab::Oklab {
                l,
                a: c * h.cos(),
                b: c * h.sin(),
            }
        })
        .collect()
}

fn main() {
    let args = Args::parse();
    std::fs::create_dir_all(&args.output_dir).unwrap();

    let n = args.size;
    let sweep_srgb: Vec<oklab::Rgb<u8>> = generate_sweep_lab(n, args.lightness, args.chroma)
        .into_iter()
        .map(oklab::oklab_to_srgb)
        .collect();

    // Layout: white top-left corner | top swatch | 3px white sep | mixing area
    //         left swatch           |            |               |
    //         3px white sep         |            |               |
    //         mixing area           |            |               |
    let swatch = (n / 20).max(8); // ~5% of mixing area
    let sep = 3u32;
    let total = swatch + sep + n;
    let off = swatch + sep; // pixel offset to mixing area origin

    let mut img = RgbImage::new(total, total);

    // Fill with white (covers corner, separators)
    for px in img.pixels_mut() {
        *px = Rgb([255, 255, 255]);
    }

    // Top swatch: rows 0..swatch, cols off..total
    for x_m in 0..n {
        let rgb = sweep_srgb[x_m as usize];
        for y in 0..swatch {
            img.put_pixel(off + x_m, y, Rgb([rgb.r, rgb.g, rgb.b]));
        }
    }

    // Left swatch: cols 0..swatch, rows off..total
    for y_m in 0..n {
        let rgb = sweep_srgb[y_m as usize];
        for x in 0..swatch {
            img.put_pixel(x, off + y_m, Rgb([rgb.r, rgb.g, rgb.b]));
        }
    }

    // Mixing area: upper-right triangle (x_m >= y_m) = sRGB,
    //              lower-left  triangle (x_m <  y_m) = OKLab
    for y_m in 0..n {
        for x_m in 0..n {
            let pixel = if x_m >= y_m {
                let cx = sweep_srgb[x_m as usize];
                let cy = sweep_srgb[y_m as usize];
                Rgb([
                    ((cx.r as u16 + cy.r as u16) / 2) as u8,
                    ((cx.g as u16 + cy.g as u16) / 2) as u8,
                    ((cx.b as u16 + cy.b as u16) / 2) as u8,
                ])
            } else {
                let lx = oklab::srgb_to_oklab(sweep_srgb[x_m as usize]);
                let ly = oklab::srgb_to_oklab(sweep_srgb[y_m as usize]);
                let mixed = oklab::Oklab {
                    l: (lx.l + ly.l) * 0.5,
                    a: (lx.a + ly.a) * 0.5,
                    b: (lx.b + ly.b) * 0.5,
                };
                let rgb = oklab::oklab_to_srgb(mixed);
                Rgb([rgb.r, rgb.g, rgb.b])
            };
            img.put_pixel(off + x_m, off + y_m, pixel);
        }
    }

    // Diagonal separator line (white)
    for i in 0..n {
        img.put_pixel(off + i, off + i, Rgb([255, 255, 255]));
    }

    // Text labels
    let scale = (n / 128).max(2);
    draw_label(&mut img, "SRGB", off + n * 2 / 3, off + n / 4, scale);
    draw_label(&mut img, "OKLAB", off + n / 3, off + n * 3 / 4, scale);

    img.save(args.output_dir.join("color_mixing.png")).unwrap();
    println!("Wrote color_mixing.png to {}", args.output_dir.display());
}
