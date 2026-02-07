use clap::Parser;
use image::RgbImage;
use okmain::debug_helpers::{FolderArgs, ensure_out_dir, find_jpg_files, load_rgb8};
use okmain::{InputImage, colors};
use std::time::Instant;

#[derive(Parser)]
struct Args {
    #[command(flatten)]
    folder: FolderArgs,
}

fn main() {
    let args = Args::parse();
    let files = find_jpg_files(&args.folder.folder);
    let out_dir = ensure_out_dir(&args.folder.folder, "comparison_with_resize");

    for path in &files {
        let filename = path.file_name().unwrap();

        let img = load_rgb8(path);

        // 1px resize color
        let t_resize = Instant::now();
        let resized = image::imageops::resize(&img, 1, 1, image::imageops::FilterType::Lanczos3);
        let resize_pixel = *resized.get_pixel(0, 0);
        let elapsed_resize = t_resize.elapsed();

        // okmain dominant color
        let t_okmain = Instant::now();
        let input = InputImage::try_from(&img).unwrap();
        let result = colors(input);
        let okmain_color = result[0];
        let okmain_pixel = image::Rgb([okmain_color.r, okmain_color.g, okmain_color.b]);
        let elapsed_okmain = t_okmain.elapsed();

        let (w, h) = (img.width(), img.height());
        let mut out = RgbImage::new(w * 3, h);

        // Left third: solid resize color
        for y in 0..h {
            for x in 0..w {
                out.put_pixel(x, y, resize_pixel);
            }
        }

        // Centre third: original image
        for y in 0..h {
            for x in 0..w {
                out.put_pixel(w + x, y, *img.get_pixel(x, y));
            }
        }

        // Right third: solid okmain color
        for y in 0..h {
            for x in 0..w {
                out.put_pixel(w * 2 + x, y, okmain_pixel);
            }
        }

        let out_path = out_dir.join(filename).with_extension("jpg");
        out.save(&out_path).unwrap();
        println!(
            "{}: resize=#{:02x}{:02x}{:02x} ({:?}) okmain=#{:02x}{:02x}{:02x} ({:?})",
            filename.to_string_lossy(),
            resize_pixel[0],
            resize_pixel[1],
            resize_pixel[2],
            elapsed_resize,
            okmain_pixel[0],
            okmain_pixel[1],
            okmain_pixel[2],
            elapsed_okmain,
        );
    }
}
