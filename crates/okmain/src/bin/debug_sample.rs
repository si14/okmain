use image::RgbImage;
use okmain::sample;
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

    let out_dir = folder.join("debug_results/sample");
    std::fs::create_dir_all(&out_dir).unwrap();

    for path in &files {
        let filename = path.file_name().unwrap();
        let img = image::open(path).unwrap().to_rgb8();
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
