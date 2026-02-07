use image::RgbImage;
use okmain::oklab_soa;
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
        let sample = oklab_soa::sample(w, h, img.as_raw());
        let elapsed = t.elapsed();

        let n = oklab_soa::block_size(w, h) as u32;
        let blocks_x = (w as u32 + n - 1) / n;
        let mut out = RgbImage::new(w as u32, h as u32);
        for i in 0..sample.l.len() {
            let rgb = oklab::oklab_to_srgb(oklab::Oklab {
                l: sample.l[i],
                a: sample.a[i],
                b: sample.b[i],
            });
            let pixel = image::Rgb([rgb.r, rgb.g, rgb.b]);
            let bx = (i as u32) % blocks_x;
            let by = (i as u32) / blocks_x;
            let x0 = bx * n;
            let y0 = by * n;
            let x1 = (x0 + n).min(w as u32);
            let y1 = (y0 + n).min(h as u32);
            for py in y0..y1 {
                for px in x0..x1 {
                    out.put_pixel(px, py, pixel);
                }
            }
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
