use pyo3::prelude::*;

#[pyfunction]
fn dominant_color_from_rgb_bytes(_rgb_bytes: &[u8]) -> PyResult<(u8, u8, u8)> {
    // okmain::colors_from_rgb_bytes(rgb_bytes)
    //     .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    todo!()
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(dominant_color_from_rgb_bytes, m)?)?;
    Ok(())
}
