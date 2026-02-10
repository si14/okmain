use super::{Centroids, MAX_CENTROIDS};
use crate::kmeans::plus_plus_init::find_initial;
use crate::oklab_soa::SampledOklabSoA;
use crate::Oklab;
use rand::RngExt;

// sklearn KMeans defaults
const MAX_ITER: usize = 300;
const CONVERGENCE_TOLERANCE: f32 = 1e-4;

#[derive(Debug, Copy, Clone)]
pub struct CentroidSoA {
    pub l: [f32; MAX_CENTROIDS],
    pub a: [f32; MAX_CENTROIDS],
    pub b: [f32; MAX_CENTROIDS],
}

// gemini suggestion
//
//use pulp::{Arch, Simd, WithSimd};
//
// pub struct PointData<'a> {
//     pub x: &'a [f32],
//     pub y: &'a [f32],
//     pub z: &'a [f32],
// }
//
// pub fn assign_centroids_padded(
//     arch: impl Arch,
//     points: PointData,
//     centroids: &[[f32; 3]],
//     assignments: &mut [u8],
// ) {
//     arch.dispatch(
//         #[inline(always)]
//         move |simd| {
//             // 1. PREPARE FIXED BUFFER (Padding)
//             // Copy centroids to a fixed-size stack array of 4.
//             // Fill unused slots with f32::MAX (infinity).
//             let mut padded_c = [[f32::MAX; 3]; 4];
//             for (i, c) in centroids.iter().enumerate() {
//                 padded_c[i] = *c;
//             }
//
//             // Constants for indices
//             let idx_0 = simd.splat_f32s(0.0);
//             let idx_1 = simd.splat_f32s(1.0);
//             let idx_2 = simd.splat_f32s(2.0);
//             let idx_3 = simd.splat_f32s(3.0);
//
//             // Helpful array to access indices by constant offset
//             let indices = [idx_0, idx_1, idx_2, idx_3];
//
//             let n = points.x.len();
//             let lane_count = simd.f32s_lane_count();
//             let chunks = n / lane_count;
//
//             // --- THE HOT LOOP ---
//             for i in 0..chunks {
//                 let offset = i * lane_count;
//
//                 // Load Point Data
//                 let px = simd.f32s_from_slice(&points.x[offset..]);
//                 let py = simd.f32s_from_slice(&points.y[offset..]);
//                 let pz = simd.f32s_from_slice(&points.z[offset..]);
//
//                 // Initialize state with Centroid 0
//                 // We assume at least 1 centroid exists.
//                 let cx0 = simd.splat_f32s(padded_c[0][0]);
//                 let cy0 = simd.splat_f32s(padded_c[0][1]);
//                 let cz0 = simd.splat_f32s(padded_c[0][2]);
//
//                 let mut min_dist = dist_sq(simd, px, py, pz, cx0, cy0, cz0);
//                 let mut best_idx = idx_0;
//
//                 // Unrolled check for 1, 2, 3
//                 // The compiler knows this loop runs exactly 3 times and will
//                 // flatten it into straight-line assembly.
//                 for k in 1..4 {
//                     // Load broadcasted centroid coordinates
//                     let cx = simd.splat_f32s(padded_c[k][0]);
//                     let cy = simd.splat_f32s(padded_c[k][1]);
//                     let cz = simd.splat_f32s(padded_c[k][2]);
//
//                     let d = dist_sq(simd, px, py, pz, cx, cy, cz);
//
//                     // Branchless update
//                     let mask = simd.f32s_less_than(d, min_dist);
//                     min_dist = simd.m32s_select_f32s(mask, d, min_dist);
//                     best_idx = simd.m32s_select_f32s(mask, indices[k], best_idx);
//                 }
//
//                 // Write Output (same as before)
//                 let best_idx_int = simd.cast_f32s_to_i32s(best_idx);
//                 let mut buffer = [0i32; 16];
//                 simd.write_i32s_to_slice(&best_idx_int, &mut buffer);
//                 for j in 0..lane_count {
//                     assignments[offset + j] = buffer[j] as u8;
//                 }
//             }
//         },
//     );
// }
//
// #[inline(always)]
// fn dist_sq<S: Simd>(
//     simd: S,
//     px: S::f32s, py: S::f32s, pz: S::f32s,
//     cx: S::f32s, cy: S::f32s, cz: S::f32s
// ) -> S::f32s {
//     let dx = simd.sub_f32s(px, cx);
//     let dy = simd.sub_f32s(py, cy);
//     let dz = simd.sub_f32s(pz, cz);
//     let d2 = simd.mul_f32s(dx, dx);
//     let d2 = simd.mul_add_f32s(dy, dy, d2);
//     simd.mul_add_f32s(dz, dz, d2)
// }
//
// pub fn compute_sums_only(
//     arch: impl Arch,
//     data: &[f32],
//     assignments: &[u8],
//     sums_out: &mut [f32; 4],
// ) {
//     arch.dispatch(
//         #[inline(always)]
//         move |simd| {
//             // 4 Float Accumulators (4 Registers)
//             let mut sums = [simd.splat_f32s(0.0); 4];
//
//             let n = data.len();
//             let lane_count = simd.f32s_lane_count();
//             let chunks = n / lane_count;
//
//             for i in 0..chunks {
//                 let offset = i * lane_count;
//
//                 // Load Data (1 Register)
//                 let val = simd.f32s_from_slice(&data[offset..]);
//
//                 // Load Assignments (Same logic as above)
//                 let mut idx_buf = [0i32; 16];
//                 for j in 0..lane_count {
//                     idx_buf[j] = assignments[offset + j] as i32;
//                 }
//                 let current_indices = simd.i32s_from_slice(&idx_buf[0..lane_count]);
//
//                 for k in 0..4 {
//                     let k_splat = simd.splat_i32s(k as i32);
//                     let mask = simd.i32s_equal(current_indices, k_splat);
//                     let mask_f = simd.cast_i32s_to_f32s(mask);
//
//                     // Float Masked Add
//                     let added = simd.add_f32s(sums[k], val);
//                     sums[k] = simd.m32s_select_f32s(mask_f, added, sums[k]);
//                 }
//             }
//
//             for k in 0..4 {
//                 sums_out[k] = simd.f32s_reduce_sum(sums[k]);
//             }
//         }
//     )
// }
//
//use pulp::{Arch, Simd, WithSimd};
//
// pub fn compute_counts_only(
//     arch: impl Arch,
//     assignments: &[u8],
//     counts_out: &mut [f32; 4], // Output as floats for easier division later
// ) {
//     arch.dispatch(
//         #[inline(always)]
//         move |simd| {
//             // 1. Integer Accumulators (4 Registers)
//             // We accumulate in i32 first, then convert to f32 at the very end.
//             // This avoids domain crossing penalties inside the loop.
//             let mut cnts = [simd.splat_i32s(0); 4];
//
//             // Constant "1" for adding
//             let one = simd.splat_i32s(1);
//
//             let n = assignments.len();
//             let lane_count = simd.u32s_lane_count(); // usually same as f32
//             let chunks = n / lane_count;
//
//             for i in 0..chunks {
//                 let offset = i * lane_count;
//
//                 // Load Assignments & Expand to i32 (1 Register)
//                 let mut idx_buf = [0i32; 16];
//                 for j in 0..lane_count {
//                     idx_buf[j] = assignments[offset + j] as i32;
//                 }
//                 let current_indices = simd.i32s_from_slice(&idx_buf[0..lane_count]);
//
//                 // The Loop 0..4
//                 // We generate the comparison constant on the fly.
//                 // It's just an integer splat, which is virtually free.
//                 for k in 0..4 {
//                     let k_splat = simd.splat_i32s(k as i32);
//
//                     // Compare: mask is -1 (All 1s) if equal, 0 otherwise
//                     let mask = simd.i32s_equal(current_indices, k_splat);
//
//                     // Integer Masked Add
//                     // In 2's complement, subtracting -1 is the same as adding 1.
//                     // If mask is -1 (True), we subtract -1 -> +1.
//                     // If mask is 0 (False), we subtract 0 -> +0.
//                     cnts[k] = simd.sub_i32s(cnts[k], simd.and_i32s(mask, one));
//
//                     // Alternatively, simpler logic if 'sub' is confusing:
//                     // let inc = simd.and_i32s(mask, one); // gives 1 or 0
//                     // cnts[k] = simd.add_i32s(cnts[k], inc);
//                 }
//             }
//
//             // Horizontal Reduction
//             for k in 0..4 {
//                 // Sum the lanes
//                 let sum_int = simd.i32s_reduce_sum(cnts[k]);
//                 counts_out[k] = sum_int as f32;
//             }
//         },
//     );
// }

#[inline(always)]
fn distance(c_l: f32, c_a: f32, c_b: f32, l: f32, a: f32, b: f32) -> f32 {
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
            let d = distance(
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
    k: usize,
    assignments: &[u8],
    centroids: &mut CentroidSoA,
) -> UpdateResult {
    let mut counts = [0u32; MAX_CENTROIDS];
    let mut sums_l = [0f32; MAX_CENTROIDS];
    let mut sums_a = [0f32; MAX_CENTROIDS];
    let mut sums_b = [0f32; MAX_CENTROIDS];

    for (i, assigned_c) in assignments.iter().copied().enumerate() {
        debug_assert!(assigned_c < k as u8);

        let assigned_c = assigned_c as usize;
        counts[assigned_c] += 1;
        sums_l[assigned_c] += sample.l[i];
        sums_a[assigned_c] += sample.a[i];
        sums_b[assigned_c] += sample.b[i];
    }

    let mut shift_squared = 0f32;

    for i in 0..MAX_CENTROIDS {
        if counts[i] == 0 {
            // It's an empty cluster, ignore
            continue;
        }

        let new_l = sums_l[i] / counts[i] as f32;
        let new_a = sums_a[i] / counts[i] as f32;
        let new_b = sums_b[i] / counts[i] as f32;

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
    pub iterations: usize,
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

    for i in 0..MAX_ITER {
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

        if update_result.shift_squared < CONVERGENCE_TOLERANCE {
            return LloydsLoopResult { iterations: i + 1 };
        }
    }

    LloydsLoopResult {
        iterations: MAX_ITER,
    }
}

pub fn find_centroids(rng: &mut impl RngExt, sample: &SampledOklabSoA, k: usize) -> Centroids {
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
    lloyds_loop(rng, sample, k, &mut assignments, &mut centroids);

    let centroids_vec: Vec<Oklab> = (0..k)
        .map(|j| Oklab {
            l: centroids.l[j],
            a: centroids.a[j],
            b: centroids.b[j],
        })
        .collect();

    Centroids {
        centroids: centroids_vec,
        assignments: assignments.iter().map(|&a| a as usize).collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::oklab_soa::SampledOklabSoA;
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
