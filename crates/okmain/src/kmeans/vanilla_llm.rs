use super::{Centroids, MAX_CENTROIDS};
use crate::kmeans::plus_plus_init::find_initial;
use crate::oklab_soa::SampledOklabSoA;
use crate::Oklab;
use rand::RngExt;

// sklearn KMeans defaults
const MAX_ITER: usize = 300;
const TOL: f32 = 1e-4;

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

    let mut last_iteration = 0;
    let mut converged_assignments: Option<Vec<usize>> = None;

    for _iter in 0..MAX_ITER {
        // E-step: assign ALL points to nearest centroid, accumulate sums + counts
        let mut sums_l = [0.0f32; MAX_CENTROIDS];
        let mut sums_a = [0.0f32; MAX_CENTROIDS];
        let mut sums_b = [0.0f32; MAX_CENTROIDS];
        let mut counts = [0u32; MAX_CENTROIDS];
        let mut assignments = Vec::with_capacity(n);

        for i in 0..n {
            let (nearest_idx, _dist_sq) = nearest_centroid(
                &Oklab {
                    l: l[i],
                    a: a[i],
                    b: b[i],
                },
                &centroids,
            );

            assignments.push(nearest_idx);
            sums_l[nearest_idx] += l[i];
            sums_a[nearest_idx] += a[i];
            sums_b[nearest_idx] += b[i];
            counts[nearest_idx] += 1;
        }

        // M-step: new centroids = sum / count
        let mut shift_sq = 0.0f32;
        for j in 0..k {
            if counts[j] == 0 {
                continue;
            }
            let c = counts[j] as f32;
            let new_l = sums_l[j] / c;
            let new_a = sums_a[j] / c;
            let new_b = sums_b[j] / c;

            let dl = new_l - centroids.l[j];
            let da = new_a - centroids.a[j];
            let db = new_b - centroids.b[j];
            shift_sq += dl.mul_add(dl, da.mul_add(da, db * db));

            centroids.l[j] = new_l;
            centroids.a[j] = new_a;
            centroids.b[j] = new_b;
        }

        // Empty cluster reassignment
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
                println!("Reassigning cluster {} to random point", j);
                let idx = rng.random_range(0..n);
                centroids.l[j] = l[idx];
                centroids.a[j] = a[idx];
                centroids.b[j] = b[idx];
            }
        }
        // Suppress unused variable warning
        let _ = min_nonempty_count;

        last_iteration += 1;

        // Convergence check
        if shift_sq < TOL {
            converged_assignments = Some(assignments);
            break;
        }
    }

    let points_processed = last_iteration * n;
    println!(
        "last_iteration: {}, points_processed: {}",
        last_iteration, points_processed
    );

    // Final assignment pass ONLY if we did NOT converge
    let assignments = if let Some(a) = converged_assignments {
        a
    } else {
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
        assignments
    };

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
