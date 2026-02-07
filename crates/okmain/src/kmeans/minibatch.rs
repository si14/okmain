use super::{Centroids, MAX_CENTROIDS};
use crate::kmeans::plus_plus_init::find_initial;
use crate::oklab_soa::SampledOklabSoA;
use crate::Oklab;
use rand::RngExt;
// References:
// - https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MiniBatchKMeans.html
// - Web-Scale K-Means Clustering (D. Sculley)
//   https://dl.acm.org/doi/epdf/10.1145/1772690.1772862
// - Noisy, Greedy and Not so Greedy k-Means++ (A. Bhattacharya et al)
//   https://drops.dagstuhl.de/storage/00lipics/lipics-vol173-esa2020/LIPIcs.ESA.2020.18/LIPIcs.ESA.2020.18.pdf

// scikit default
const BATCH_SIZE: usize = 1024;

const MAX_PASSES: usize = 100;
const MAX_NO_IMPROVEMENT: u32 = 10;

#[derive(Debug, Copy, Clone)]
struct CentroidSoA {
    l: [f32; MAX_CENTROIDS],
    a: [f32; MAX_CENTROIDS],
    b: [f32; MAX_CENTROIDS],
}

#[inline(always)]
fn nearest_centroid(point: &Oklab, centroids: &CentroidSoA) -> (usize, f32) {
    let mut distances = [0.0f32; MAX_CENTROIDS];

    for (i, d) in distances.iter_mut().enumerate().take(MAX_CENTROIDS) {
        let dl = point.l - centroids.l[i];
        let da = point.a - centroids.a[i];
        let db = point.b - centroids.b[i];

        *d = dl.mul_add(dl, da.mul_add(da, db * db));
    }

    let mut best = 0;
    for j in 1..MAX_CENTROIDS {
        if distances[j] < distances[best] {
            best = j;
        }
    }
    (best, distances[best])
}

pub fn find_centroids(rng: &mut impl RngExt, sample: &SampledOklabSoA, k: usize) -> Centroids {
    let SampledOklabSoA {
        l,
        a,
        b,
        x: _,
        y: _,
    } = sample;

    let n = l.len();
    assert_eq!(a.len(), n);
    assert_eq!(b.len(), n);

    assert!(k <= MAX_CENTROIDS);

    // More clusters than points => silent clamping
    let k = k.min(n);

    let init_points = find_initial(rng, sample, k);

    let mut centroids = CentroidSoA {
        l: [f32::MAX; MAX_CENTROIDS],
        a: [f32::MAX; MAX_CENTROIDS],
        b: [f32::MAX; MAX_CENTROIDS],
    };
    for (j, &idx) in init_points.iter().enumerate() {
        centroids.l[j] = l[idx];
        centroids.a[j] = a[idx];
        centroids.b[j] = b[idx];
    }

    // Per-centroid cumulative counts
    let mut counts = [0u32; MAX_CENTROIDS];

    // Batch buffers for the hot loop
    let mut batch_l = Vec::with_capacity(BATCH_SIZE);
    let mut batch_a = Vec::with_capacity(BATCH_SIZE);
    let mut batch_b = Vec::with_capacity(BATCH_SIZE);

    let selection_prob = BATCH_SIZE as f32 / n as f32;

    // Change 1: iteration count = full-dataset passes, not mini-batch steps
    let n_steps = (MAX_PASSES * n).div_ceil(BATCH_SIZE).max(1);

    // Change 4: adaptive EMA alpha
    let ema_alpha = (BATCH_SIZE as f32 * 2.0 / (n as f32 + 1.0)).min(1.0);

    // Change 5: sklearn-style early stopping — track all-time minimum
    let mut ema_inertia: f32 = f32::INFINITY;
    let mut ema_inertia_min: f32 = f32::INFINITY;
    let mut no_improvement: u32 = 0;

    let mut last_iteration = 0;
    let mut points_processed: usize = 0;

    for _iter in 0..n_steps {
        // 1. Bernoulli sampling (unchanged)
        batch_l.clear();
        batch_a.clear();
        batch_b.clear();
        for i in 0..n {
            if rng.random::<f32>() < selection_prob && batch_l.len() < BATCH_SIZE {
                batch_l.push(l[i]);
                batch_a.push(a[i]);
                batch_b.push(b[i]);
            }
        }

        let batch_len = batch_l.len();
        if batch_len == 0 {
            continue;
        }
        points_processed += batch_len;

        // 2. E-step: assign all batch points to nearest centroid (frozen centers)
        let mut batch_sums_l = [0.0f32; MAX_CENTROIDS];
        let mut batch_sums_a = [0.0f32; MAX_CENTROIDS];
        let mut batch_sums_b = [0.0f32; MAX_CENTROIDS];
        let mut batch_counts = [0u32; MAX_CENTROIDS];
        let mut batch_inertia: f32 = 0.0;

        for i in 0..batch_len {
            let (nearest_idx, dist_sq) = nearest_centroid(
                &Oklab {
                    l: batch_l[i],
                    a: batch_a[i],
                    b: batch_b[i],
                },
                &centroids,
            );

            batch_inertia += dist_sq;
            batch_sums_l[nearest_idx] += batch_l[i];
            batch_sums_a[nearest_idx] += batch_a[i];
            batch_sums_b[nearest_idx] += batch_b[i];
            batch_counts[nearest_idx] += 1;
        }

        batch_inertia /= batch_len as f32;

        // 3. M-step: update centroids using batch statistics
        for j in 0..k {
            if batch_counts[j] == 0 {
                continue;
            }
            let bc = batch_counts[j] as f32;
            let batch_mean_l = batch_sums_l[j] / bc;
            let batch_mean_a = batch_sums_a[j] / bc;
            let batch_mean_b = batch_sums_b[j] / bc;

            let alpha = bc / (counts[j] as f32 + bc);
            centroids.l[j] += (batch_mean_l - centroids.l[j]) * alpha;
            centroids.a[j] += (batch_mean_a - centroids.a[j]) * alpha;
            centroids.b[j] += (batch_mean_b - centroids.b[j]) * alpha;
            counts[j] += batch_counts[j];
        }

        // 4. Empty cluster reassignment (threshold 0.005%)
        let max_count = counts[..k].iter().copied().max().unwrap_or(0);
        let empty_threshold = (0.00005 * max_count as f32) as u32;
        let min_nonempty_count = counts[..k]
            .iter()
            .copied()
            .filter(|&c| c > empty_threshold)
            .min()
            .unwrap_or(1)
            .max(1);

        for j in 0..k {
            if counts[j] <= empty_threshold {
                // Reassign to a random data point
                let idx = rng.random_range(0..n);
                centroids.l[j] = l[idx];
                centroids.a[j] = a[idx];
                centroids.b[j] = b[idx];
                counts[j] = min_nonempty_count;
            }
        }

        // 5. EMA inertia update + sklearn-style early stopping
        if ema_inertia == f32::INFINITY {
            ema_inertia = batch_inertia;
        } else {
            ema_inertia = ema_inertia * (1.0 - ema_alpha) + batch_inertia * ema_alpha;
        }

        if ema_inertia < ema_inertia_min {
            ema_inertia_min = ema_inertia;
            no_improvement = 0;
        } else {
            no_improvement += 1;
            if no_improvement >= MAX_NO_IMPROVEMENT {
                break;
            }
        }

        last_iteration += 1;
    }

    println!("last_iteration: {}, points_processed: {}", last_iteration, points_processed);

    // Final assignment pass (full scan)
    let mut assignments = Vec::with_capacity(n);
    for i in 0..n {
        let (idx, _dist) = nearest_centroid(
            &Oklab {
                l: l[i],
                a: a[i],
                b: b[i],
            },
            &centroids,
        );
        assignments.push(idx);
    }

    let centroids: Vec<Oklab> = (0..k)
        .map(|j| Oklab {
            l: centroids.l[j],
            a: centroids.a[j],
            b: centroids.b[j],
        })
        .collect();

    Centroids {
        centroids,
        assignments,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng;
    use pretty_assertions::{assert_eq, assert_ne};

    fn make_soa(points: &[(f32, f32, f32)]) -> SampledOklabSoA {
        SampledOklabSoA {
            x: vec![],
            y: vec![],
            l: points.iter().map(|p| p.0).collect(),
            a: points.iter().map(|p| p.1).collect(),
            b: points.iter().map(|p| p.2).collect(),
        }
    }

    #[test]
    fn single_point() {
        let mut rng = rng::new();

        let soa = make_soa(&[(0.5, 0.3, 1.0)]);
        let result = find_centroids(&mut rng, &soa, 1);
        assert_eq!(result.centroids.len(), 1);
        assert_eq!(result.assignments, vec![0]);
        assert!((result.centroids[0].l - 0.5).abs() < 1e-3);
        assert!((result.centroids[0].a - 0.3).abs() < 1e-3);
        assert!((result.centroids[0].b - 1.0).abs() < 1e-3);
    }

    #[test]
    fn k_equals_one() {
        let mut rng = rng::new();

        let soa = make_soa(&[(0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (2.0, 2.0, 2.0)]);
        let result = find_centroids(&mut rng, &soa, 1);
        assert_eq!(result.centroids.len(), 1);
        assert!(result.assignments.iter().all(|&a| a == 0));
    }

    #[test]
    fn two_well_separated_clusters() {
        let mut rng = rng::new();

        // Two tight groups far apart
        let mut points = Vec::new();
        for i in 0..50 {
            let offset = i as f32 * 0.001;
            points.push((0.0 + offset, 0.0 + offset, 0.0 + offset));
        }
        for i in 0..50 {
            let offset = i as f32 * 0.001;
            points.push((100.0 + offset, 100.0 + offset, 100.0 + offset));
        }
        let soa = make_soa(&points);
        let result = find_centroids(&mut rng, &soa, 2);

        assert_eq!(result.centroids.len(), 2);

        // All points in the first group should have the same label
        let label_a = result.assignments[0];
        assert!(result.assignments[..50].iter().all(|&a| a == label_a));

        // All points in the second group should have the same label
        let label_b = result.assignments[50];
        assert!(result.assignments[50..].iter().all(|&a| a == label_b));

        // The two groups should have different labels
        assert_ne!(label_a, label_b);

        // Centroids should be near cluster centers
        for centroid in &result.centroids {
            let near_zero = centroid.l < 1.0 && centroid.a < 1.0 && centroid.b < 1.0;
            let near_hundred = centroid.l > 99.0 && centroid.a > 99.0 && centroid.b > 99.0;
            assert!(near_zero || near_hundred);
        }
    }

    #[test]
    fn three_clusters() {
        let mut rng = rng::new();

        let mut points = Vec::new();
        // Cluster near (0,0,0)
        for i in 0..30 {
            let o = i as f32 * 0.001;
            points.push((o, o, o));
        }
        // Cluster near (50,50,50)
        for i in 0..30 {
            let o = i as f32 * 0.001;
            points.push((50.0 + o, 50.0 + o, 50.0 + o));
        }
        // Cluster near (100,100,100)
        for i in 0..30 {
            let o = i as f32 * 0.001;
            points.push((100.0 + o, 100.0 + o, 100.0 + o));
        }
        let soa = make_soa(&points);
        let result = find_centroids(&mut rng, &soa, 3);

        assert_eq!(result.centroids.len(), 3);

        // Each group should be internally consistent
        let label_a = result.assignments[0];
        let label_b = result.assignments[30];
        let label_c = result.assignments[60];
        assert!(result.assignments[..30].iter().all(|&a| a == label_a));
        assert!(result.assignments[30..60].iter().all(|&a| a == label_b));
        assert!(result.assignments[60..].iter().all(|&a| a == label_c));

        // All labels should be distinct
        assert_ne!(label_a, label_b);
        assert_ne!(label_b, label_c);
        assert_ne!(label_a, label_c);
    }

    #[test]
    fn k_clamped_when_greater_than_n() {
        let mut rng = rng::new();

        let soa = make_soa(&[(0.0, 0.0, 0.0), (50.0, 50.0, 50.0), (100.0, 100.0, 100.0)]);
        // k=4 with n=3 should work, returns 3 centroids
        let result = find_centroids(&mut rng, &soa, 4);
        assert_eq!(result.centroids.len(), 3);
        assert_eq!(result.assignments.len(), 3);
        assert!(result.assignments.iter().all(|&a| a < 3));
    }

    #[test]
    fn deterministic() {
        let soa = make_soa(&[
            (0.0, 0.0, 0.0),
            (1.0, 1.0, 1.0),
            (50.0, 50.0, 50.0),
            (51.0, 51.0, 51.0),
        ]);

        let mut rng1 = rng::new();
        let result1 = find_centroids(&mut rng1, &soa, 2);

        let mut rng2 = rng::new();
        let result2 = find_centroids(&mut rng2, &soa, 2);

        assert_eq!(result1.assignments, result2.assignments);
        assert_eq!(result1.centroids.len(), result2.centroids.len());
        for (a, b) in result1.centroids.iter().zip(result2.centroids.iter()) {
            assert_eq!(a, b);
        }
    }

    #[test]
    fn assignments_in_range() {
        let mut rng = rng::new();

        let soa = make_soa(&[
            (0.0, 0.0, 0.0),
            (10.0, 10.0, 10.0),
            (20.0, 20.0, 20.0),
            (30.0, 30.0, 30.0),
            (40.0, 40.0, 40.0),
        ]);

        for k in 1..=4 {
            let result = find_centroids(&mut rng, &soa, k);
            assert_eq!(result.centroids.len(), k);
            assert!(
                result.assignments.iter().all(|&a| a < k),
                "k={k}: all assignments must be in 0..{k}, got {:?}",
                result.assignments,
            );
        }
    }

    #[test]
    fn larger_than_batch() {
        let mut rng = rng::new();

        // n=5000 > BATCH_SIZE=1024
        let mut points = Vec::with_capacity(5000);
        for i in 0..5000 {
            let v = (i % 4) as f32 * 25.0 + (i as f32 * 0.0001);
            points.push((v, v, v));
        }
        let soa = make_soa(&points);
        let result = find_centroids(&mut rng, &soa, 4);

        assert_eq!(result.centroids.len(), 4);
        assert_eq!(result.assignments.len(), 5000);
        assert!(result.assignments.iter().all(|&a| a < 4));
    }
}
