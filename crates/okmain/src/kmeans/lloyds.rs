use super::MAX_CENTROIDS;
use crate::oklab_soa::SampledOklabSoA;
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
        assert!(assigned_c < k as u8);

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::oklab_soa::SampledOklabSoA;

    const N_PER_CLUSTER: usize = 4096;
    const CENTERS: [(f32, f32, f32); 4] = [
        (0.0, 0.0, 0.0),
        (10.0, 0.0, 0.0),
        (0.0, 10.0, 0.0),
        (0.0, 0.0, 10.0),
    ];

    fn make_four_cluster_soa() -> SampledOklabSoA {
        let n = N_PER_CLUSTER * 4;
        let mut soa = SampledOklabSoA::new(n);
        for &(cl, ca, cb) in &CENTERS {
            for i in 0..N_PER_CLUSTER {
                let offset = i as f32 * 0.0001;
                soa.push(0, 0, cl + offset, ca + offset, cb + offset);
            }
        }
        soa
    }

    #[test]
    fn test_assign_points() {
        let soa = make_four_cluster_soa();
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

        // Each cluster's points should all get the same label
        for ci in 0..4 {
            let start = ci * N_PER_CLUSTER;
            let end = start + N_PER_CLUSTER;
            let label = assignments[start];
            assert!(
                assignments[start..end].iter().all(|&a| a == label),
                "cluster {ci}: not all points assigned to same centroid",
            );
        }

        // The four cluster labels should be distinct
        let labels: Vec<u8> = (0..4).map(|ci| assignments[ci * N_PER_CLUSTER]).collect();
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
        let soa = make_four_cluster_soa();
        let n = soa.l.len();
        let k = 4;

        // Correct assignments: point i belongs to cluster i / N_PER_CLUSTER
        let assignments: Vec<u8> = (0..n).map(|i| (i / N_PER_CLUSTER) as u8).collect();

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

        let result = update_centroids(&soa, k, &assignments, &mut centroids);

        // Each centroid should be near its cluster's true center.
        // Per-cluster offsets are 0..4096 * 0.0001, so mean offset ~0.2.
        for (i, &(cl, ca, cb)) in CENTERS.iter().enumerate() {
            assert!(
                (centroids.l[i] - cl).abs() < 0.3,
                "centroid {i} l: expected ~{cl}, got {}",
                centroids.l[i],
            );
            assert!(
                (centroids.a[i] - ca).abs() < 0.3,
                "centroid {i} a: expected ~{ca}, got {}",
                centroids.a[i],
            );
            assert!(
                (centroids.b[i] - cb).abs() < 0.3,
                "centroid {i} b: expected ~{cb}, got {}",
                centroids.b[i],
            );
        }

        // Centroids moved from (99,99,99), so shift must be large
        assert!(
            result.shift_squared > 0.0,
            "shift_squared should be > 0, got {}",
            result.shift_squared,
        );

        // Each cluster has exactly N_PER_CLUSTER points
        for i in 0..k {
            assert_eq!(
                result.counts[i], N_PER_CLUSTER as u32,
                "cluster {i} count mismatch",
            );
        }
    }
}
