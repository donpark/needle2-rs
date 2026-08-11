//! Transparent Needle 2 browser entry point.
//! The current implementation owns the model bytes per call; caching and KV
//! reuse are deliberately left to the host until numerical parity is complete.

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn needle2_complete(
    model_bytes: &[u8],
    tools_json: &str,
    query: &str,
    max_new_tokens: usize,
) -> Result<JsValue, JsValue> {
    let model = needle2_format::CactModel::parse(model_bytes)
        .map_err(|error| JsValue::from_str(&error.to_string()))?;
    let calls = needle2_infer::generate_constrained(&model, tools_json, query, max_new_tokens)
        .map_err(|error| JsValue::from_str(&error))?;
    JsValue::from_serde(&calls).map_err(|error| JsValue::from_str(&error.to_string()))
}

#[cfg(not(target_arch = "wasm32"))]
pub fn needle2_complete(
    model_bytes: &[u8],
    tools_json: &str,
    query: &str,
    max_new_tokens: usize,
) -> Result<serde_json::Value, String> {
    let model = needle2_format::CactModel::parse(model_bytes).map_err(|error| error.to_string())?;
    let calls = needle2_infer::generate_constrained(&model, tools_json, query, max_new_tokens)?;
    Ok(serde_json::Value::Array(calls))
}
