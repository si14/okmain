use super::{lloyds, ADAPTIVE_MIN_CENTROID_DISTANCE_SQUARED, MAX_CENTROIDS};
use crate::sample::SampledOklabSoA;
use oklab::Oklab;
use rand::RngExt;

fn squared_distance(x: Oklab, y: Oklab) -> f32 {
    let dl = x.l - y.l;
    let da = x.a - y.a;
    let db = x.b - y.b;
    dl.mul_add(dl, da.mul_add(da, db * db))
}

fn count_similar_clusters(centroids: &[Oklab]) -> usize {
    let mut count = 0;
    for i in 0..centroids.len() {
        for j in (i + 1)..centroids.len() {
            if squared_distance(centroids[i], centroids[j]) < ADAPTIVE_MIN_CENTROID_DISTANCE_SQUARED
            {
                count += 1;
            }
        }
    }
    count
}

#[derive(Debug)]
pub struct Result {
    pub centroids: Vec<Oklab>,
    pub assignments: Vec<usize>,
    pub loop_iterations: Vec<usize>,
    pub converged: Vec<bool>,
}

pub fn find_centroids(rng: &mut impl RngExt, sample: &SampledOklabSoA) -> Result {
    let mut k = MAX_CENTROIDS;
    let mut loop_iterations = Vec::with_capacity(MAX_CENTROIDS);
    let mut converged = Vec::with_capacity(MAX_CENTROIDS);

    loop {
        let result = lloyds::find_centroids(rng, sample, k);

        loop_iterations.push(result.loop_iterations);
        converged.push(result.converged);

        let similar_count = count_similar_clusters(&result.centroids);
        if similar_count == 0 || k <= 1 {
            break Result {
                centroids: result.centroids,
                assignments: result.assignments,
                loop_iterations,
                converged,
            };
        }
        k -= 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_similar_well_separated() {
        let centroids = [
            Oklab {
                l: 0.0,
                a: 0.0,
                b: 0.0,
            },
            Oklab {
                l: 1.0,
                a: 0.0,
                b: 0.0,
            },
            Oklab {
                l: 0.0,
                a: 1.0,
                b: 0.0,
            },
        ];
        assert_eq!(count_similar_clusters(&centroids), 0);
    }

    #[test]
    fn one_similar_pair_near_duplicates() {
        let centroids = [
            Oklab {
                l: 0.5,
                a: 0.3,
                b: 0.2,
            },
            Oklab {
                l: 0.5,
                a: 0.3,
                b: 0.2,
            },
        ];
        assert_eq!(count_similar_clusters(&centroids), 1);
    }

    #[test]
    fn no_similar_single_centroid() {
        let centroids = [Oklab {
            l: 0.5,
            a: 0.3,
            b: 0.2,
        }];
        assert_eq!(count_similar_clusters(&centroids), 0);
    }

    #[test]
    fn multiple_similar_clusters() {
        let centroids = [
            Oklab {
                l: 0.5,
                a: 0.3,
                b: 0.2,
            },
            Oklab {
                l: 0.5,
                a: 0.3,
                b: 0.2,
            },
            Oklab {
                l: 0.6,
                a: 0.4,
                b: 0.3,
            },
            Oklab {
                l: 0.6,
                a: 0.4,
                b: 0.3,
            },
        ];
        assert_eq!(count_similar_clusters(&centroids), 2);
    }

    #[test]
    fn squared_distance_known_values() {
        let a = Oklab {
            l: 1.0,
            a: 2.0,
            b: 3.0,
        };
        let b = Oklab {
            l: 4.0,
            a: 5.0,
            b: 6.0,
        };
        // (3^2 + 3^2 + 3^2) = 27
        let d = squared_distance(a, b);
        assert!((d - 27.0).abs() < 1e-6);
    }
}
