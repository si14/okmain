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

        let label_a = result.assignments[0];
        assert!(result.assignments[..50].iter().all(|&a| a == label_a));

        let label_b = result.assignments[50];
        assert!(result.assignments[50..].iter().all(|&a| a == label_b));

        assert_ne!(label_a, label_b);

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
        for i in 0..30 {
            let o = i as f32 * 0.001;
            points.push((o, o, o));
        }
        for i in 0..30 {
            let o = i as f32 * 0.001;
            points.push((50.0 + o, 50.0 + o, 50.0 + o));
        }
        for i in 0..30 {
            let o = i as f32 * 0.001;
            points.push((100.0 + o, 100.0 + o, 100.0 + o));
        }
        let soa = make_soa(&points);
        let result = find_centroids(&mut rng, &soa, 3);

        assert_eq!(result.centroids.len(), 3);

        let label_a = result.assignments[0];
        let label_b = result.assignments[30];
        let label_c = result.assignments[60];
        assert!(result.assignments[..30].iter().all(|&a| a == label_a));
        assert!(result.assignments[30..60].iter().all(|&a| a == label_b));
        assert!(result.assignments[60..].iter().all(|&a| a == label_c));

        assert_ne!(label_a, label_b);
        assert_ne!(label_b, label_c);
        assert_ne!(label_a, label_c);
    }

    #[test]
    fn k_clamped_when_greater_than_n() {
        let mut rng = rng::new();

        let soa = make_soa(&[(0.0, 0.0, 0.0), (50.0, 50.0, 50.0), (100.0, 100.0, 100.0)]);
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
    fn large_dataset() {
        let mut rng = rng::new();

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
