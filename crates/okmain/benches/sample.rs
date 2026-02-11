use criterion::{
    AxisScale, BenchmarkId, Criterion, PlotConfiguration, criterion_group, criterion_main,
};
use okmain::{rng, sample};
use rand::Rng;

struct Size {
    label: &'static str,
    width: u16,
    height: u16,
}

const SIZES: &[Size] = &[
    Size {
        label: "small",
        width: 100,
        height: 100,
    },
    Size {
        label: "medium",
        width: 1000,
        height: 1500,
    },
    Size {
        label: "large",
        width: 2796,
        height: 1290,
    },
];

fn bench(c: &mut Criterion) {
    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);

    let buffers: Vec<Vec<u8>> = SIZES
        .iter()
        .map(|size| {
            let len = size.width as usize * size.height as usize * 3;
            let mut buf = vec![0u8; len];
            rng::new().fill_bytes(&mut buf);
            buf
        })
        .collect();

    let mut group = c.benchmark_group("sample");
    group.plot_config(plot_config);

    for (size, buf) in SIZES.iter().zip(buffers.iter()) {
        group.bench_with_input(BenchmarkId::from_parameter(size.label), buf, |b, buf| {
            b.iter_with_large_drop(|| sample::sample(size.width, size.height, buf))
        });
    }
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
