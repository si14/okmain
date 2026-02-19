use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pyclass(name = "_ScoredCentroid")]
struct PyScoredCentroid {
    #[pyo3(get)]
    rgb: (u8, u8, u8),
    #[pyo3(get)]
    oklab: (f32, f32, f32),
    #[pyo3(get)]
    mask_weighted_counts: f32,
    #[pyo3(get)]
    mask_weighted_counts_score: f32,
    #[pyo3(get)]
    chroma: f32,
    #[pyo3(get)]
    chroma_score: f32,
    #[pyo3(get)]
    final_score: f32,
}

#[pyclass(name = "_DebugInfo")]
struct PyDebugInfo {
    #[pyo3(get)]
    scored_centroids: Vec<Py<PyScoredCentroid>>,
    #[pyo3(get)]
    kmeans_loop_iterations: Vec<usize>,
    #[pyo3(get)]
    kmeans_converged: Vec<bool>,
}

fn parse_dimensions(width: u32, height: u32) -> PyResult<(u16, u16)> {
    let w = u16::try_from(width).map_err(|_| {
        PyValueError::new_err(format!(
            "image width {} is too large (max {})",
            width,
            u16::MAX
        ))
    })?;
    let h = u16::try_from(height).map_err(|_| {
        PyValueError::new_err(format!(
            "image height {} is too large (max {})",
            height,
            u16::MAX
        ))
    })?;
    Ok((w, h))
}

#[pyfunction]
#[pyo3(name = "_colors")]
fn py_colors(
    buf: &[u8],
    width: u32,
    height: u32,
    mask_saturated_threshold: f32,
    mask_weight: f32,
    mask_weighted_counts_weight: f32,
    chroma_weight: f32,
) -> PyResult<Vec<(u8, u8, u8)>> {
    let (w, h) = parse_dimensions(width, height)?;
    let input = okmain::InputImage::from_bytes(w, h, buf)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let colors = okmain::colors_with_config(
        input,
        mask_saturated_threshold,
        mask_weight,
        mask_weighted_counts_weight,
        chroma_weight,
    )
    .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(colors.into_iter().map(|c| (c.r, c.g, c.b)).collect())
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
#[pyfunction]
#[pyo3(name = "_colors_debug")]
fn py_colors_debug(
    py: Python<'_>,
    buf: &[u8],
    width: u32,
    height: u32,
    mask_saturated_threshold: f32,
    mask_weight: f32,
    mask_weighted_counts_weight: f32,
    chroma_weight: f32,
) -> PyResult<(Vec<(u8, u8, u8)>, Py<PyDebugInfo>)> {
    let (w, h) = parse_dimensions(width, height)?;
    let input = okmain::InputImage::from_bytes(w, h, buf)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let (colors, debug) = okmain::colors_debug(
        input,
        mask_saturated_threshold,
        mask_weight,
        mask_weighted_counts_weight,
        chroma_weight,
    )
    .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let colors_out: Vec<(u8, u8, u8)> = colors.into_iter().map(|c| (c.r, c.g, c.b)).collect();

    let scored_centroids = debug
        .scored_centroids
        .into_iter()
        .map(|sc| {
            Py::new(
                py,
                PyScoredCentroid {
                    rgb: (sc.rgb.r, sc.rgb.g, sc.rgb.b),
                    oklab: (sc.oklab.l, sc.oklab.a, sc.oklab.b),
                    mask_weighted_counts: sc.mask_weighted_counts,
                    mask_weighted_counts_score: sc.mask_weighted_counts_score,
                    chroma: sc.chroma,
                    chroma_score: sc.chroma_score,
                    final_score: sc.final_score,
                },
            )
        })
        .collect::<PyResult<Vec<_>>>()?;

    let debug_out = Py::new(
        py,
        PyDebugInfo {
            scored_centroids,
            kmeans_loop_iterations: debug.kmeans_loop_iterations,
            kmeans_converged: debug.kmeans_converged,
        },
    )?;

    Ok((colors_out, debug_out))
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyScoredCentroid>()?;
    m.add_class::<PyDebugInfo>()?;
    m.add_function(wrap_pyfunction!(py_colors, m)?)?;
    m.add_function(wrap_pyfunction!(py_colors_debug, m)?)?;
    m.add(
        "DEFAULT_MASK_SATURATED_THRESHOLD",
        okmain::DEFAULT_MASK_SATURATED_THRESHOLD,
    )?;
    m.add("DEFAULT_MASK_WEIGHT", okmain::DEFAULT_MASK_WEIGHT)?;
    m.add(
        "DEFAULT_WEIGHTED_COUNTS_WEIGHT",
        okmain::DEFAULT_WEIGHTED_COUNTS_WEIGHT,
    )?;
    m.add("DEFAULT_CHROMA_WEIGHT", okmain::DEFAULT_CHROMA_WEIGHT)?;
    Ok(())
}
