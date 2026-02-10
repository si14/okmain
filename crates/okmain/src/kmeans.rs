pub mod adaptive;
pub mod lloyds;
pub mod plus_plus_init;

use crate::Oklab;

pub const MAX_CENTROIDS: usize = 4;
pub const ADAPTIVE_MIN_CENTROID_DISTANCE_SQUARED: f32 = 0.005;

#[derive(Debug)]
pub struct Centroids {
    pub centroids: Vec<Oklab>,
    pub assignments: Vec<usize>,
}
