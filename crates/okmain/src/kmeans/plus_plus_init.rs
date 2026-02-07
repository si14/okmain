use crate::oklab_soa::SampledOklabSoA;
use rand::RngExt;

#[inline(always)]
pub(crate) fn squared_distance(l: &[f32], a: &[f32], b: &[f32], x1: usize, x2: usize) -> f32 {
    let dl = l[x1] - l[x2];
    let da = a[x1] - a[x2];
    let db = b[x1] - b[x2];

    dl.mul_add(dl, da.mul_add(da, db * db))
}

#[inline(always)]
fn sample_by_distance(rng: &mut impl RngExt, min_distances: &[f32]) -> usize {
    let sum_min_distances = min_distances.iter().sum::<f32>();

    // I don't really need to normalise the probabilities to sample from them,
    // multiplying the threshold by the sum of the distances is exactly the same
    let random_threshold = rng.random::<f32>() * sum_min_distances;
    let mut cumsum = 0.0;

    for (i, &distance) in min_distances.iter().enumerate() {
        cumsum += distance;
        if cumsum > random_threshold {
            return i;
        }
    }

    min_distances.len() - 1
}

pub fn find_initial(
    rng: &mut impl RngExt,
    SampledOklabSoA {
        l,
        a,
        b,
        x: _,
        y: _,
    }: &SampledOklabSoA,
    k: usize,
) -> Vec<usize> {
    let n = l.len();
    assert_eq!(a.len(), n);
    assert_eq!(b.len(), n);

    // More clusters than points => silent clamping
    let k = k.min(n);

    let mut init_points = Vec::<usize>::with_capacity(k);
    let c0 = rng.random_range(0..n);
    init_points.push(c0);

    let mut min_distances: Vec<f32> = (0..n).map(|x| squared_distance(l, a, b, c0, x)).collect();

    // Greedy k-means++: sample multiple candidates per step and pick the one
    // that minimises the total potential (sum of min distances).
    let n_candidates = (2.0 + (k as f64).ln()).floor() as usize;

    for _ in 1..k {
        let mut best_candidate = 0usize;
        let mut best_potential = f32::INFINITY;

        for _ in 0..n_candidates {
            let candidate = sample_by_distance(rng, &min_distances);

            // Compute what the potential would be if we picked this candidate
            let potential: f32 = min_distances
                .iter()
                .enumerate()
                .map(|(x, &cur_dist)| {
                    let cand_dist = squared_distance(l, a, b, candidate, x);
                    cur_dist.min(cand_dist)
                })
                .sum();

            if potential < best_potential {
                best_potential = potential;
                best_candidate = candidate;
            }
        }

        // Update min_distances with the winner
        for (x, distance) in min_distances.iter_mut().enumerate() {
            let new_distance = squared_distance(l, a, b, best_candidate, x);
            *distance = distance.min(new_distance);
        }

        init_points.push(best_candidate);
    }

    init_points
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng;
    use pretty_assertions::{assert_eq, assert_ne};

    #[test]
    fn basic_invariants() {
        let mut rng = rng::new();

        let data = SampledOklabSoA {
            l: vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            a: vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            b: vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            x: vec![],
            y: vec![],
        };

        for k in 1..=4 {
            let result = find_initial(&mut rng, &data, k);
            assert_eq!(result.len(), k, "must return exactly k indices");

            for &idx in &result {
                assert!(idx < data.l.len(), "index must be valid");
            }

            let mut sorted = result.clone();
            sorted.sort();
            sorted.dedup();
            assert_eq!(sorted.len(), k, "indices must be distinct");
        }
    }

    #[test]
    fn k_equals_one() {
        let mut rng = rng::new();

        let data = SampledOklabSoA {
            l: vec![0.0, 1.0, 2.0],
            a: vec![0.0, 1.0, 2.0],
            b: vec![0.0, 1.0, 2.0],
            x: vec![],
            y: vec![],
        };

        let result = find_initial(&mut rng, &data, 1);
        assert_eq!(result.len(), 1);
        assert!(result[0] < 3);
    }

    #[test]
    fn k_equals_n() {
        let mut rng = rng::new();

        let data = SampledOklabSoA {
            l: vec![0.0, 10.0, 20.0, 30.0],
            a: vec![0.0, 10.0, 20.0, 30.0],
            b: vec![0.0, 10.0, 20.0, 30.0],
            x: vec![],
            y: vec![],
        };

        let result = find_initial(&mut rng, &data, 4);
        assert_eq!(result.len(), 4);

        let mut sorted = result.clone();
        sorted.sort();
        assert_eq!(sorted, vec![0, 1, 2, 3]);
    }

    #[test]
    fn k_greater_than_n_clamped() {
        let mut rng = rng::new();

        let data = SampledOklabSoA {
            l: vec![0.0, 10.0, 20.0],
            a: vec![0.0, 10.0, 20.0],
            b: vec![0.0, 10.0, 20.0],
            x: vec![],
            y: vec![],
        };

        // k=4 > n=3, should behave like k=3 and return all indices
        let result = find_initial(&mut rng, &data, 4);
        assert_eq!(result.len(), 3);

        let mut sorted = result.clone();
        sorted.sort();
        assert_eq!(sorted, vec![0, 1, 2]);
    }

    #[test]
    fn unequal_cluster_sizes() {
        let mut rng = rng::new();

        // Dense cluster near origin (indices 0..10) + two distant outliers (10, 11)
        let mut l = vec![0.0; 12];
        let mut a = vec![0.0; 12];
        let mut b = vec![0.0; 12];

        for i in 0..10 {
            l[i] = (i as f32) * 0.01;
            a[i] = (i as f32) * 0.01;
            b[i] = (i as f32) * 0.01;
        }

        l[10] = 100.0;
        a[10] = 100.0;
        b[10] = 100.0;

        l[11] = -100.0;
        a[11] = -100.0;
        b[11] = -100.0;

        let data = SampledOklabSoA {
            l,
            a,
            b,
            x: vec![],
            y: vec![],
        };

        let result = find_initial(&mut rng, &data, 3);
        assert!(
            result.contains(&10),
            "outlier at index 10 should be selected"
        );
        assert!(
            result.contains(&11),
            "outlier at index 11 should be selected"
        );
    }

    #[test]
    fn duplicate_coordinates() {
        let mut rng = rng::new();

        // Two points at the same location + one elsewhere
        let data = SampledOklabSoA {
            l: vec![0.0, 0.0, 10.0],
            a: vec![0.0, 0.0, 10.0],
            b: vec![0.0, 0.0, 10.0],
            x: vec![],
            y: vec![],
        };

        let result = find_initial(&mut rng, &data, 2);
        assert_eq!(result.len(), 2);

        // The two selected points must have distinct coordinates
        let coords: Vec<(f32, f32, f32)> = result
            .iter()
            .map(|&i| (data.l[i], data.a[i], data.b[i]))
            .collect();
        assert_ne!(
            coords[0], coords[1],
            "selected centroids should have distinct coordinates"
        );
    }

    #[test]
    fn three_clusters() {
        let mut rng = rng::new();

        // Indices:            0    1    2     3    4    5
        // Clusters:           0    1    2     0    1    2
        let data = SampledOklabSoA {
            l: vec![0.0, 1.0, -1.0, 0.1, 1.1, -1.1],
            a: vec![0.0, 1.0, -1.0, 0.1, 1.1, -1.1],
            b: vec![0.0, 1.0, -1.0, 0.1, 1.1, -1.1],
            x: vec![],
            y: vec![],
        };

        let result = find_initial(&mut rng, &data, 3);
        assert!(
            result.contains(&0) || result.contains(&3),
            "The result contains the first cluster"
        );
        assert!(
            result.contains(&1) || result.contains(&4),
            "The result contains the second cluster"
        );
        assert!(
            result.contains(&2) || result.contains(&5),
            "The result contains the third cluster"
        );
    }
}
