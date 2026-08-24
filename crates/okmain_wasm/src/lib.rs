use wasm_bindgen::prelude::*;

#[wasm_bindgen(start)]
pub fn init_panic_hook() {
    console_error_panic_hook::set_once();
}

#[wasm_bindgen]
pub fn extract_dominant_colors_rgb(width: u16, height: u16, rgb: &[u8]) -> Result<Vec<u8>, JsValue> {
    let input = okmain::InputImage::from_bytes(width, height, rgb)
        .map_err(|err| JsValue::from_str(&err.to_string()))?;
    let colors = okmain::colors(input);

    let mut out = Vec::with_capacity(colors.len() * 3);
    for color in colors {
        out.extend_from_slice(&[color.r, color.g, color.b]);
    }

    Ok(out)
}

#[wasm_bindgen]
pub fn extract_dominant_colors_rgba(
    width: u16,
    height: u16,
    rgba: &[u8],
) -> Result<Vec<u8>, JsValue> {
    let expected_len = (width as usize) * (height as usize) * 4;
    if rgba.len() != expected_len {
        return Err(JsValue::from_str(&format!(
            "RGBA buffer length mismatch: expected {}, got {}",
            expected_len,
            rgba.len()
        )));
    }

    let mut rgb = Vec::with_capacity((width as usize) * (height as usize) * 3);
    for px in rgba.chunks_exact(4) {
        rgb.extend_from_slice(&px[0..3]);
    }

    extract_dominant_colors_rgb(width, height, &rgb)
}
