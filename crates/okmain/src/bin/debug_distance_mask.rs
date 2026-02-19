use clap::Parser;
use image::GrayImage;
use okmain::DEFAULT_MASK_SATURATED_THRESHOLD;
use std::path::PathBuf;

#[derive(Parser)]
struct Args {
    /// Image width in pixels
    #[arg(long, default_value_t = 500)]
    width: u16,

    /// Image height in pixels
    #[arg(long, default_value_t = 700)]
    height: u16,

    /// Saturated threshold for the mask
    #[arg(long, default_value_t = DEFAULT_MASK_SATURATED_THRESHOLD)]
    threshold: f32,

    /// Output file path
    #[arg(short, long, default_value = "distance_mask.png")]
    output: PathBuf,
}

fn main() {
    let args = Args::parse();

    let mut img = GrayImage::new(args.width as u32, args.height as u32);

    for y in 0..args.height {
        for x in 0..args.width {
            let value = okmain::distance_mask(args.threshold, args.width, args.height, x, y);
            let pixel = (value * 255.0).round() as u8;
            img.put_pixel(x as u32, y as u32, image::Luma([pixel]));
        }
    }

    img.save(&args.output).unwrap();
    println!(
        "Saved {}x{} distance mask (threshold={}) to {}",
        args.width,
        args.height,
        args.threshold,
        args.output.display(),
    );
}
