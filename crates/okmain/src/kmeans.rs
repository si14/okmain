pub mod adaptive;
pub mod lloyds;
pub mod plus_plus_init;

// References:
// - https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
// - https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MiniBatchKMeans.html
// - Web-Scale K-Means Clustering (D. Sculley)
//   https://dl.acm.org/doi/epdf/10.1145/1772690.1772862
// - Noisy, Greedy and Not so Greedy k-Means++ (A. Bhattacharya et al)
//   https://drops.dagstuhl.de/storage/00lipics/lipics-vol173-esa2020/LIPIcs.ESA.2020.18/LIPIcs.ESA.2020.18.pdf
//
// Observations:
// - Mini-batch K-means is not worth it. It's slower than classic K-means on the relevant
//   datasets.

pub const MAX_CENTROIDS: usize = 4;
pub const ADAPTIVE_MIN_CENTROID_DISTANCE_SQUARED: f32 = 0.005;

// sklearn KMeans defaults + inspired by testing
const LLOYDS_MAX_ITERATIONS: usize = 300;

const LLOYDS_CONVERGENCE_TOLERANCE: f32 = 1e-3;

// Scikit uses (2+log(k)), which is 3 or 4 for k=1..4, we can settle on 3
const GREEDY_INIT_N_CANDIDATES: usize = 3;
