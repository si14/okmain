use criterion::{Criterion, criterion_group, criterion_main};
use okmain::kmeans::{MAX_CENTROIDS, lloyds, plus_plus_init};
use okmain::rng;
use okmain::sample::SampledOklabSoA;
use rand::RngExt;

fn generate_random_points(n: usize) -> SampledOklabSoA {
    let mut rng = rng::new();
    let mut sampled = SampledOklabSoA::new(0, 0, n);
    for _ in 0..n {
        sampled.l.push(rng.random::<f32>());
        sampled.a.push(rng.random::<f32>());
        sampled.b.push(rng.random::<f32>());
    }
    sampled
}

fn bench(c: &mut Criterion) {
    let sample = generate_random_points(500_000);
    let k = 4;

    // Pre-compute initial centroids for assign_points bench
    let init_indices = plus_plus_init::find_initial(&mut rng::new(), &sample, k);
    let mut centroids = lloyds::CentroidSoA {
        l: [f32::MAX; MAX_CENTROIDS],
        a: [f32::MAX; MAX_CENTROIDS],
        b: [f32::MAX; MAX_CENTROIDS],
    };
    for (j, &idx) in init_indices.iter().enumerate() {
        centroids.l[j] = sample.l[idx];
        centroids.a[j] = sample.a[idx];
        centroids.b[j] = sample.b[idx];
    }

    // Pre-compute assignments for update_centroids bench
    let mut assignments = vec![0u8; sample.l.len()];
    lloyds::assign_points(&sample, &centroids, &mut assignments);

    c.bench_function("kmeans/plus_plus_init", |b| {
        b.iter(|| plus_plus_init::find_initial(&mut rng::new(), &sample, k))
    });

    c.bench_function("kmeans/assign_points", |b| {
        let mut assign_buf = vec![0u8; sample.l.len()];
        b.iter(|| lloyds::assign_points(&sample, &centroids, &mut assign_buf))
    });

    c.bench_function("kmeans/update_centroids", |b| {
        b.iter(|| {
            let mut centroids_copy = centroids;
            lloyds::update_centroids(&sample, &assignments, &mut centroids_copy)
        })
    });

    c.bench_function("kmeans/find_centroids", |b| {
        b.iter(|| lloyds::find_centroids(&mut rng::new(), &sample, k))
    });
}

criterion_group!(benches, bench);
criterion_main!(benches);
