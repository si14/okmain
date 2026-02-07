use image::RgbImage;
use okmain::colors_from_image;
use std::path::PathBuf;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 2 {
        eprintln!("usage: {} <folder>", args[0]);
        std::process::exit(1);
    }
    let folder = PathBuf::from(&args[1]);

    let mut files: Vec<PathBuf> = std::fs::read_dir(&folder)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.extension()
                .map_or(false, |ext| ext.eq_ignore_ascii_case("jpg"))
        })
        .collect();
    files.sort();

    let out_dir = folder.join("debug_results/colors");
    std::fs::create_dir_all(&out_dir).unwrap();

    for path in &files {
        let filename = path.file_name().unwrap();

        let t = Instant::now();
        let img = image::open(path).unwrap().to_rgb8();
        let colors = colors_from_image(&img).unwrap();
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
        let num_colors = colors.len();
        let swatch_h = h / num_colors as u32;

        for (i, color) in colors.iter().enumerate() {
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
