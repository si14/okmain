use image::RgbImage;
use okmain::{kmeans, rng, sample};
use std::path::PathBuf;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 || args.len() > 3 {
        eprintln!("usage: {} <folder> [k]", args[0]);
        std::process::exit(1);
    }
    let folder = PathBuf::from(&args[1]);
    let _k: usize = args
        .get(2)
        .map_or(4, |s| s.parse().expect("k must be a number"));

    let mut files: Vec<PathBuf> = std::fs::read_dir(&folder)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.extension()
                .is_some_and(|ext| ext.eq_ignore_ascii_case("jpg"))
        })
        .collect();
    files.sort();

    let out_dir = folder.join("debug_results/centroids");
    std::fs::create_dir_all(&out_dir).unwrap();

    for path in &files {
        let filename = path.file_name().unwrap();

        let t = Instant::now();
        let img = image::open(path).unwrap().to_rgb8();
        let (w, h) = (img.width() as u16, img.height() as u16);
        let sample = sample::sample(w, h, img.as_raw());

        let mut rng = rng::new();
        let result = kmeans::adaptive::find_centroids(&mut rng, &sample);
        let elapsed = t.elapsed();

        let out_w = w as u32 * 22 / 10;
        let out_h = h as u32;
        let mut out = RgbImage::new(out_w, out_h);

        // left: original
        for y in 0..out_h {
            for x in 0..(w as u32) {
                out.put_pixel(x, y, *img.get_pixel(x, y));
            }
        }

        // centre: centroid swatches
        let num_centroids = result.centroids.len();
        let swatch_x = w as u32;
        let swatch_w = w as u32 * 2 / 10;
        let swatch_h = out_h / num_centroids as u32;

        let centroid_rgbs: Vec<image::Rgb<u8>> = result
            .centroids
            .iter()
            .map(|c| {
                let rgb = oklab::oklab_to_srgb(oklab::Oklab {
                    l: c.l,
                    a: c.a,
                    b: c.b,
                });
                image::Rgb([rgb.r, rgb.g, rgb.b])
            })
            .collect();

        for (i, &pixel) in centroid_rgbs.iter().enumerate() {
            let y_start = i as u32 * swatch_h;
            let y_end = if i == num_centroids - 1 {
                out_h
            } else {
                y_start + swatch_h
            };
            for y in y_start..y_end {
                for x in swatch_x..(swatch_x + swatch_w) {
                    out.put_pixel(x, y, pixel);
                }
            }
        }

        // right: remapped image (every pixel replaced by nearest centroid)
        let remap_x = swatch_x + swatch_w;
        for py in 0..h as u32 {
            for px in 0..w as u32 {
                let src = img.get_pixel(px, py);
                let ok = oklab::srgb_to_oklab(oklab::Rgb {
                    r: src[0],
                    g: src[1],
                    b: src[2],
                });
                let mut best = 0;
                let mut best_d = f32::MAX;
                for (i, c) in result.centroids.iter().enumerate() {
                    let dl = ok.l - c.l;
                    let da = ok.a - c.a;
                    let db = ok.b - c.b;
                    let d = dl * dl + da * da + db * db;
                    if d < best_d {
                        best_d = d;
                        best = i;
                    }
                }
                out.put_pixel(remap_x + px, py, centroid_rgbs[best]);
            }
        }

        out.save(out_dir.join(filename)).unwrap();
        println!(
            "{}: k={}, {:?}",
            filename.to_string_lossy(),
            num_centroids,
            elapsed,
        );
    }
}
