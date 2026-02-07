use criterion::{
    criterion_group, criterion_main, AxisScale, BenchmarkId, Criterion, PlotConfiguration,
};
use okmain::kmeans::{minibatch, plus_plus_init};
use okmain::oklab_soa::SampledOklabSoA;
use okmain::rng;
use rand::RngExt;
use std::collections::HashMap;

fn generate_random_points(n: usize) -> SampledOklabSoA {
    let mut rng = rng::new();

    let mut sampled = SampledOklabSoA::new(n);

    sampled.x.resize(n, 0);
    sampled.y.resize(n, 0);

    for _ in 0..n {
        sampled.l.push(rng.random::<f32>());
        sampled.a.push(rng.random::<f32>());
        sampled.b.push(rng.random::<f32>());
    }

    sampled
}

fn generate_clustered_points(n: usize, k: usize) -> SampledOklabSoA {
    let mut rng = rng::new();

    let mut sampled = SampledOklabSoA::new(n);

    let centers: Vec<(f32, f32, f32)> = vec![
        (0.2, 0.1, 0.5),
        (0.8, 0.3, 2.0),
        (0.5, 0.2, 4.0),
        (0.3, 0.4, 5.5),
    ];
    let noise = 0.01;

    let mut gen_noise = || (rng.random::<f32>() - 0.5) * noise;

    sampled.x.resize(n, 0);
    sampled.y.resize(n, 0);

    for i in 0..n {
        let (c_l, c_a, c_b) = centers[i % k];
        sampled.l.push(c_l + gen_noise());
        sampled.a.push(c_a + gen_noise());
        sampled.b.push(c_b + gen_noise());
    }

    sampled
}

struct Input<'a> {
    pub label: String,
    pub k: usize,
    pub samples: &'a HashMap<usize, SampledOklabSoA>,
}

fn bench(c: &mut Criterion) {
    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);

    let mut random_samples: HashMap<usize, SampledOklabSoA> = HashMap::new();
    random_samples.insert(100_000, generate_random_points(100_000));
    random_samples.insert(1_000_000, generate_random_points(1_000_000));
    random_samples.insert(10_000_000, generate_random_points(10_000_000));

    let mut clustered_samples: HashMap<usize, SampledOklabSoA> = HashMap::new();
    clustered_samples.insert(100_000, generate_clustered_points(100_000, 4));
    clustered_samples.insert(1_000_000, generate_clustered_points(1_000_000, 4));
    clustered_samples.insert(10_000_000, generate_clustered_points(10_000_000, 4));

    let ks = [2usize, 4usize];
    let sizes = [
        ("100k", 100_000usize),
        ("1M", 1_000_000usize),
        ("10M", 10_000_000usize),
    ];

    let group_inputs = ks
        .iter()
        .flat_map(|&k| {
            [
                ("random", &random_samples),
                ("clustered", &clustered_samples),
            ]
            .into_iter()
            .map(move |(sample_label, samples)| Input {
                label: format!("{sample_label}-k{k}"),
                k,
                samples,
            })
        })
        .collect::<Vec<_>>();

    for group_input in group_inputs {
        let mut group = c.benchmark_group(format!("plus_plus_init/{}", group_input.label));
        group.plot_config(plot_config.clone());

        for &(size_name, size) in sizes.iter() {
            group.bench_with_input(BenchmarkId::from_parameter(size_name), &size, |b, size| {
                let sample = group_input.samples.get(size).unwrap();
                b.iter_with_large_drop(|| {
                    let rng = &mut rng::new();
                    plus_plus_init::find_initial(rng, sample, group_input.k)
                })
            });
        }
        group.finish();

        let mut group = c.benchmark_group(format!("minibatch/{}", group_input.label));
        group.plot_config(plot_config.clone());

        for &(size_name, size) in sizes.iter() {
            group.bench_with_input(BenchmarkId::from_parameter(size_name), &size, |b, size| {
                let sample = group_input.samples.get(size).unwrap();
                b.iter_with_large_drop(|| {
                    let rng = &mut rng::new();
                    minibatch::find_centroids(rng, sample, group_input.k)
                })
            });
        }
        group.finish();
    }
}

criterion_group!(benches, bench);
criterion_main!(benches);
