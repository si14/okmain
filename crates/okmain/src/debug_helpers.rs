use clap::Parser;
use image::RgbImage;
use std::path::{Path, PathBuf};

#[derive(Parser)]
pub struct FolderArgs {
    /// Path to a folder of images
    pub folder: PathBuf,
}

pub fn find_jpg_files(folder: &Path) -> Vec<PathBuf> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(folder)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.extension()
                .is_some_and(|ext| ext.eq_ignore_ascii_case("jpg"))
        })
        .collect();
    files.sort();
    files
}

pub fn ensure_out_dir(folder: &Path, name: &str) -> PathBuf {
    let dir = folder.join("debug_results").join(name);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

pub fn load_rgb8(path: &Path) -> RgbImage {
    image::open(path).unwrap().to_rgb8()
}
