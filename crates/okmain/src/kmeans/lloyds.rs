use super::{LLOYDS_CONVERGENCE_TOLERANCE, LLOYDS_MAX_ITERATIONS, MAX_CENTROIDS};
use crate::kmeans::plus_plus_init::find_initial;
use crate::sample::SampledOklabSoA;
use oklab::Oklab;
use rand::RngExt;

#[derive(Debug, Copy, Clone)]
pub struct CentroidSoA {
    pub l: [f32; MAX_CENTROIDS],
    pub a: [f32; MAX_CENTROIDS],
    pub b: [f32; MAX_CENTROIDS],
}

#[inline(always)]
pub(crate) fn squared_distance_flat(c_l: f32, c_a: f32, c_b: f32, l: f32, a: f32, b: f32) -> f32 {
    let dl = c_l - l;
    let da = c_a - a;
    let db = c_b - b;

    dl.mul_add(dl, da.mul_add(da, db * db))
}

#[inline]
pub fn assign_points(sample: &SampledOklabSoA, centroids: &CentroidSoA, assignments: &mut [u8]) {
    for (i, assignment) in assignments.iter_mut().enumerate() {
        let mut min = f32::MAX;
        let mut min_idx = 0;
        for j in 0..MAX_CENTROIDS {
            let d = squared_distance_flat(
                centroids.l[j],
                centroids.a[j],
                centroids.b[j],
                sample.l[i],
                sample.a[i],
                sample.b[i],
            );
            if d < min {
                min = d;
                min_idx = j;
            }
        }

        *assignment = min_idx as u8;
    }
}

#[derive(Debug)]
pub struct UpdateResult {
    shift_squared: f32,
    counts: [u32; MAX_CENTROIDS],
}

#[inline]
pub fn update_centroids(
    sample: &SampledOklabSoA,
    _k: usize,
    assignments: &[u8],
    centroids: &mut CentroidSoA,
) -> UpdateResult {
    let mut counts_f = [0f32; MAX_CENTROIDS];
    let mut sums_l = [0f32; MAX_CENTROIDS];
    let mut sums_a = [0f32; MAX_CENTROIDS];
    let mut sums_b = [0f32; MAX_CENTROIDS];

    for (i, &assigned_c) in assignments.iter().enumerate() {
        let l = sample.l[i];
        let a = sample.a[i];
        let b = sample.b[i];
        for k in 0..MAX_CENTROIDS {
            // This construction auto-vectorises a bit better
            let mask = ((assigned_c == k as u8) as u32) as f32;

            counts_f[k] += mask;
            sums_l[k] = mask.mul_add(l, sums_l[k]);
            sums_a[k] = mask.mul_add(a, sums_a[k]);
            sums_b[k] = mask.mul_add(b, sums_b[k]);
        }
    }

    let mut counts = [0u32; MAX_CENTROIDS];
    for i in 0..MAX_CENTROIDS {
        counts[i] = counts_f[i] as u32;
    }

    let mut shift_squared = 0f32;

    for i in 0..MAX_CENTROIDS {
        if counts[i] == 0 {
            // It's an empty cluster, ignore
            continue;
        }

        let new_l = sums_l[i] / counts_f[i];
        let new_a = sums_a[i] / counts_f[i];
        let new_b = sums_b[i] / counts_f[i];

        let dl = centroids.l[i] - new_l;
        let da = centroids.a[i] - new_a;
        let db = centroids.b[i] - new_b;

        centroids.l[i] = new_l;
        centroids.a[i] = new_a;
        centroids.b[i] = new_b;

        shift_squared += dl.mul_add(dl, da.mul_add(da, db * db));
    }

    UpdateResult {
        shift_squared,
        counts,
    }
}

#[allow(dead_code)] // used for debug output
pub struct LloydsLoopResult {
    pub loop_iterations: usize,
    pub converged: bool,
}

pub fn lloyds_loop(
    rng: &mut impl RngExt,
    sample: &SampledOklabSoA,
    k: usize,
    assignments: &mut [u8],
    centroids: &mut CentroidSoA,
) -> LloydsLoopResult {
    assert_eq!(sample.l.len(), sample.a.len());
    assert_eq!(sample.l.len(), sample.b.len());
    assert_eq!(sample.l.len(), assignments.len());
    assert!(k <= MAX_CENTROIDS);
    assert!(k <= sample.l.len());

    for i in k..MAX_CENTROIDS {
        // Verify that the unused centroids are set to f32::MAX as padding (it might be used later)
        assert_eq!(centroids.l[i], f32::MAX);
        assert_eq!(centroids.a[i], f32::MAX);
        assert_eq!(centroids.b[i], f32::MAX);
    }

    for i in 0..LLOYDS_MAX_ITERATIONS {
        assign_points(sample, centroids, assignments);
        let update_result = update_centroids(sample, k, assignments, centroids);

        // Empty cluster reassignment (but only when it's not a dummy one)
        for (i, count) in update_result.counts.iter().copied().take(k).enumerate() {
            if count == 0 {
                let random_point = rng.random_range(0..sample.l.len());
                centroids.l[i] = sample.l[random_point];
                centroids.a[i] = sample.a[random_point];
                centroids.b[i] = sample.b[random_point];
            }
        }

        if update_result.shift_squared < LLOYDS_CONVERGENCE_TOLERANCE {
            return LloydsLoopResult {
                loop_iterations: i + 1,
                converged: true,
            };
        }
    }

    LloydsLoopResult {
        loop_iterations: LLOYDS_MAX_ITERATIONS,
        converged: false,
    }
}

pub struct Result {
    pub centroids: Vec<Oklab>,
    pub assignments: Vec<usize>,
    pub loop_iterations: usize,
    pub converged: bool,
}

pub fn find_centroids(rng: &mut impl RngExt, sample: &SampledOklabSoA, k: usize) -> Result {
    let n = sample.l.len();
    let k = k.min(n);

    let init_points = find_initial(rng, sample, k);

    let mut centroids = CentroidSoA {
        l: [f32::MAX; MAX_CENTROIDS],
        a: [f32::MAX; MAX_CENTROIDS],
        b: [f32::MAX; MAX_CENTROIDS],
    };
    for (j, &idx) in init_points.iter().enumerate() {
        centroids.l[j] = sample.l[idx];
        centroids.a[j] = sample.a[idx];
        centroids.b[j] = sample.b[idx];
    }

    let mut assignments = vec![0u8; n];
    let LloydsLoopResult {
        loop_iterations: iterations,
        converged,
    } = lloyds_loop(rng, sample, k, &mut assignments, &mut centroids);

    let centroids_vec: Vec<Oklab> = (0..k)
        .map(|j| Oklab {
            l: centroids.l[j],
            a: centroids.a[j],
            b: centroids.b[j],
        })
        .collect();

    Result {
        centroids: centroids_vec,
        assignments: assignments.iter().map(|&a| a as usize).collect(),
        loop_iterations: iterations,
        converged,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sample::SampledOklabSoA;
    use rand::RngExt;

    const N_PER_CLUSTER: usize = 4096;
    const NOISE: f32 = 0.5;
    const CENTERS: [(f32, f32, f32); 4] = [
        (0.0, 0.0, 0.0),
        (10.0, 0.0, 0.0),
        (0.0, 10.0, 0.0),
        (0.0, 0.0, 10.0),
    ];

    fn make_four_cluster_soa(rng: &mut impl RngExt) -> (SampledOklabSoA, Vec<u8>) {
        let n = N_PER_CLUSTER * 4;
        let mut points: Vec<(u8, f32, f32, f32)> = Vec::with_capacity(n);
        for (ci, &(cl, ca, cb)) in CENTERS.iter().enumerate() {
            for _ in 0..N_PER_CLUSTER {
                points.push((
                    ci as u8,
                    cl + rng.random::<f32>() * NOISE,
                    ca + rng.random::<f32>() * NOISE,
                    cb + rng.random::<f32>() * NOISE,
                ));
            }
        }

        // Fisher-Yates shuffle
        for i in (1..points.len()).rev() {
            let j = rng.random_range(0..(i + 1));
            points.swap(i, j);
        }

        let mut soa = SampledOklabSoA::new(0, 0, n);
        let mut truth = Vec::with_capacity(n);
        for &(cluster_id, l, a, b) in &points {
            soa.push(l, a, b);
            truth.push(cluster_id);
        }

        (soa, truth)
    }

    #[test]
    fn test_assign_points() {
        let mut rng = crate::rng::new();
        let (soa, truth) = make_four_cluster_soa(&mut rng);
        let n = soa.l.len();

        let mut centroids = CentroidSoA {
            l: [f32::MAX; MAX_CENTROIDS],
            a: [f32::MAX; MAX_CENTROIDS],
            b: [f32::MAX; MAX_CENTROIDS],
        };
        for (i, &(cl, ca, cb)) in CENTERS.iter().enumerate() {
            centroids.l[i] = cl;
            centroids.a[i] = ca;
            centroids.b[i] = cb;
        }

        let mut assignments = vec![0u8; n];
        assign_points(&soa, &centroids, &mut assignments);

        // All points sharing the same ground-truth cluster get the same label
        let mut label_for_cluster = [None::<u8>; 4];
        for (i, &assigned) in assignments.iter().enumerate() {
            let gt = truth[i] as usize;
            match label_for_cluster[gt] {
                None => label_for_cluster[gt] = Some(assigned),
                Some(expected) => assert_eq!(
                    assigned, expected,
                    "point {i}: ground-truth cluster {gt} got different labels",
                ),
            }
        }

        // The 4 labels are distinct
        let labels: Vec<u8> = label_for_cluster.iter().map(|l| l.unwrap()).collect();
        for i in 0..4 {
            for j in (i + 1)..4 {
                assert_ne!(
                    labels[i], labels[j],
                    "clusters {i} and {j} should have different labels",
                );
            }
        }
    }

    #[test]
    fn test_update_centroids() {
        let mut rng = crate::rng::new();
        let (soa, truth) = make_four_cluster_soa(&mut rng);
        let k = 4;

        // Start centroids far away
        let mut centroids = CentroidSoA {
            l: [f32::MAX; MAX_CENTROIDS],
            a: [f32::MAX; MAX_CENTROIDS],
            b: [f32::MAX; MAX_CENTROIDS],
        };
        for i in 0..k {
            centroids.l[i] = 99.0;
            centroids.a[i] = 99.0;
            centroids.b[i] = 99.0;
        }

        let result = update_centroids(&soa, k, &truth, &mut centroids);

        // Each centroid should be near its cluster's true center (tolerance 1.0 for noise)
        for (i, &(cl, ca, cb)) in CENTERS.iter().enumerate() {
            assert!(
                (centroids.l[i] - cl).abs() < 1.0,
                "centroid {i} l: expected ~{cl}, got {}",
                centroids.l[i],
            );
            assert!(
                (centroids.a[i] - ca).abs() < 1.0,
                "centroid {i} a: expected ~{ca}, got {}",
                centroids.a[i],
            );
            assert!(
                (centroids.b[i] - cb).abs() < 1.0,
                "centroid {i} b: expected ~{cb}, got {}",
                centroids.b[i],
            );
        }

        assert!(
            result.shift_squared > 0.0,
            "shift_squared should be > 0, got {}",
            result.shift_squared,
        );

        let total_count: u32 = result.counts.iter().take(k).sum();
        assert_eq!(total_count, (N_PER_CLUSTER * 4) as u32);
    }

    #[test]
    fn test_find_centroids_k1() {
        let mut rng = crate::rng::new();
        let (soa, _) = make_four_cluster_soa(&mut rng);

        let result = find_centroids(&mut crate::rng::new(), &soa, 1);
        assert_eq!(result.centroids.len(), 1);
        assert!(result.assignments.iter().all(|&a| a == 0));
    }

    #[test]
    fn test_find_centroids_k_equals_n() {
        let soa = SampledOklabSoA {
            l: vec![0.0, 10.0, 20.0],
            a: vec![0.0, 10.0, 20.0],
            b: vec![0.0, 10.0, 20.0],
            width: 0,
            height: 0,
        };

        // k=5 > n=3, gets clamped to n=3
        let result = find_centroids(&mut crate::rng::new(), &soa, 5);
        assert_eq!(result.centroids.len(), 3);
    }

    #[test]
    fn test_find_centroids_well_separated() {
        let mut rng = crate::rng::new();

        let n_per = 100;
        let mut points: Vec<(u8, f32, f32, f32)> = Vec::with_capacity(n_per * 4);
        for (ci, &(cl, ca, cb)) in CENTERS.iter().enumerate() {
            for _ in 0..n_per {
                points.push((
                    ci as u8,
                    cl + rng.random::<f32>() * NOISE,
                    ca + rng.random::<f32>() * NOISE,
                    cb + rng.random::<f32>() * NOISE,
                ));
            }
        }

        // Fisher-Yates shuffle
        for i in (1..points.len()).rev() {
            let j = rng.random_range(0..(i + 1));
            points.swap(i, j);
        }

        let mut soa = SampledOklabSoA::new(0, 0, n_per * 4);
        let mut truth = Vec::with_capacity(n_per * 4);
        for &(cluster_id, l, a, b) in &points {
            soa.push(l, a, b);
            truth.push(cluster_id);
        }

        let result = find_centroids(&mut crate::rng::new(), &soa, 4);
        assert_eq!(result.centroids.len(), 4);

        // Cluster consistency: points with same ground-truth get same assignment
        let mut label_for_cluster = [None::<usize>; 4];
        for (i, &assigned) in result.assignments.iter().enumerate() {
            let gt = truth[i] as usize;
            match label_for_cluster[gt] {
                None => label_for_cluster[gt] = Some(assigned),
                Some(expected) => assert_eq!(
                    assigned, expected,
                    "point {i}: ground-truth cluster {gt} got different assignments",
                ),
            }
        }

        // 4 distinct labels
        let labels: Vec<usize> = label_for_cluster.iter().map(|l| l.unwrap()).collect();
        for i in 0..4 {
            for j in (i + 1)..4 {
                assert_ne!(labels[i], labels[j]);
            }
        }

        // Centroids near centers
        for ci in 0..4 {
            let (cl, ca, cb) = CENTERS[ci];
            let ri = labels[ci];
            let c = &result.centroids[ri];
            assert!((c.l - cl).abs() < 1.0, "centroid l for cluster {ci}");
            assert!((c.a - ca).abs() < 1.0, "centroid a for cluster {ci}");
            assert!((c.b - cb).abs() < 1.0, "centroid b for cluster {ci}");
        }
    }
}
