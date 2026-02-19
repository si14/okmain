use clap::Parser;
use image::RgbImage;
use okmain::debug_helpers::{FolderArgs, ensure_out_dir, find_jpg_files, load_rgb8};
use okmain::sample;
use std::time::Instant;

#[derive(Parser)]
struct Args {
    #[command(flatten)]
    folder: FolderArgs,
}

fn main() {
    let args = Args::parse();
    let files = find_jpg_files(&args.folder.folder);
    let out_dir = ensure_out_dir(&args.folder.folder, "sample");

    for path in &files {
        let filename = path.file_name().unwrap();
        let img = load_rgb8(path);
        let (w, h) = (img.width() as u16, img.height() as u16);

        let t = Instant::now();
        let sample = sample::sample(w, h, img.as_raw());
        let elapsed = t.elapsed();

        let mut out = RgbImage::new(sample.width as u32, sample.height as u32);
        for i in 0..sample.l.len() {
            let rgb = oklab::oklab_to_srgb(oklab::Oklab {
                l: sample.l[i],
                a: sample.a[i],
                b: sample.b[i],
            });
            let px = (i as u32) % sample.width as u32;
            let py = (i as u32) / sample.width as u32;
            out.put_pixel(px, py, image::Rgb([rgb.r, rgb.g, rgb.b]));
        }

        out.save(out_dir.join(filename)).unwrap();
        println!(
            "{}: {} samples, {:?}",
            filename.to_string_lossy(),
            sample.l.len(),
            elapsed,
        );
    }
}
