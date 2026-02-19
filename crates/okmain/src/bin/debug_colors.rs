use clap::Parser;
use image::RgbImage;
use okmain::debug_helpers::{ensure_out_dir, find_jpg_files, load_rgb8, FolderArgs};
use okmain::{colors, InputImage};
use std::time::Instant;

#[derive(Parser)]
struct Args {
    #[command(flatten)]
    folder: FolderArgs,
}

fn main() {
    let args = Args::parse();
    let files = find_jpg_files(&args.folder.folder);
    let out_dir = ensure_out_dir(&args.folder.folder, "colors");

    for path in &files {
        let filename = path.file_name().unwrap();

        let t = Instant::now();
        let img = load_rgb8(path);
        let input = InputImage::try_from(&img).unwrap();
        let result = colors(input);
        let elapsed = t.elapsed();

        let (w, h) = (img.width(), img.height());
        let swatch_w = w * 2 / 10;
        let out_w = w + swatch_w;
        let mut out = RgbImage::new(out_w, h);

        // Left: original image
        for y in 0..h {
            for x in 0..w {
                out.put_pixel(x, y, *img.get_pixel(x, y));
            }
        }

        // Right: swatch strip (sorted by score, first = dominant)
        let num_colors = result.len();
        let swatch_h = h / num_colors as u32;

        for (i, color) in result.iter().enumerate() {
            let pixel = image::Rgb([color.r, color.g, color.b]);
            let y_start = i as u32 * swatch_h;
            let y_end = if i == num_colors - 1 {
                h
            } else {
                y_start + swatch_h
            };
            for y in y_start..y_end {
                for x in w..(w + swatch_w) {
                    out.put_pixel(x, y, pixel);
                }
            }
        }

        out.save(out_dir.join(filename)).unwrap();
        println!(
            "{}: {} colors, {:?}",
            filename.to_string_lossy(),
            num_colors,
            elapsed,
        );
    }
}
