use crate::kmeans::lloyds::squared_distance;
use crate::sample::SampledOklabSoA;
use rand::RngExt;
use std::array;

// Scikit uses (2+log(k)), which is 3 or 4 for k=1..4, we can settle on 3
const N_CANDIDATES: usize = 3;

#[inline(always)]
fn sample_by_distance(rng: &mut impl RngExt, min_distances: &[f32], sum: f32) -> usize {
    let random_threshold = rng.random::<f32>() * sum;
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
    SampledOklabSoA { l, a, b, .. }: &SampledOklabSoA,
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

    let (c0l, c0a, c0b) = (l[c0], a[c0], b[c0]);
    let mut min_distances = vec![0.0f32; n];
    let mut min_distances_sum = 0.0f32;
    for i in 0..n {
        let d = squared_distance(l[i], a[i], b[i], c0l, c0a, c0b);
        min_distances[i] = d;
        min_distances_sum += d;
    }

    let mut candidate_min_distances: [_; N_CANDIDATES] = array::from_fn(|_| vec![0.0f32; n]);

    // Greedy k-means++: sample multiple candidates per step and pick the one
    // that minimises the total potential (sum of min distances).

    for _ in 1..k {
        // Sample all candidates upfront (uses cached sum)
        let mut candidates = [0usize; N_CANDIDATES];
        for candidate in candidates.iter_mut() {
            *candidate = sample_by_distance(rng, &min_distances, min_distances_sum);
        }

        let candidates_l = candidates.map(|c| l[c]);
        let candidates_a = candidates.map(|c| a[c]);
        let candidates_b = candidates.map(|c| b[c]);

        let mut potentials = [0.0f32; N_CANDIDATES];

        // mut slices seem to reassure the compiler that the vectors don't alias
        let candidate_min_d_slices = candidate_min_distances.each_mut().map(|v| v.as_mut_slice());

        for i in 0..n {
            let (li, ai, bi, current_min) = (l[i], a[i], b[i], min_distances[i]);
            for j in 0..N_CANDIDATES {
                let d = squared_distance(
                    candidates_l[j],
                    candidates_a[j],
                    candidates_b[j],
                    li,
                    ai,
                    bi,
                )
                .min(current_min);
                candidate_min_d_slices[j][i] = d;
                potentials[j] += d;
            }
        }

        let mut best_potential = f32::INFINITY;
        let mut best = 0;
        for (i, potential) in potentials.iter().copied().enumerate().take(N_CANDIDATES) {
            if potential < best_potential {
                best_potential = potential;
                best = i;
            }
        }

        std::mem::swap(
            &mut min_distances,
            &mut candidate_min_distances.as_mut()[best],
        );
        min_distances_sum = best_potential;
        init_points.push(candidates[best]);
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
            width: 0,
            height: 0,
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
            width: 0,
            height: 0,
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
            width: 0,
            height: 0,
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
            width: 0,
            height: 0,
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
            width: 0,
            height: 0,
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
            width: 0,
            height: 0,
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
            width: 0,
            height: 0,
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
