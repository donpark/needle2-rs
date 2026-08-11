use needle2_format::CactModel;
use needle2_infer::{attention_block, LayerWeights, D_MODEL};

#[test]
fn loads_and_runs_one_real_attention_block() {
    let Some(path) = std::env::var_os("NEEDLE2_CACT") else {
        eprintln!("SKIP: NEEDLE2_CACT is not set");
        return;
    };
    let bytes = std::fs::read(path).expect("read model");
    let model = CactModel::parse(&bytes).expect("parse model");
    let weights = LayerWeights::from_cact(&model, 0).expect("layer 0 weights");
    let input = vec![0.0; 2 * D_MODEL];
    let mut output = vec![0.0; input.len()];
    attention_block(&input, 2, &weights, &mut output);
    assert!(output.iter().all(|value| value.is_finite()));

    let Some(fixture) = std::env::var_os("NEEDLE2_ATTENTION_FIXTURE") else {
        return;
    };
    let fixture: serde_json::Value =
        serde_json::from_slice(&std::fs::read(fixture).expect("read attention fixture"))
            .expect("parse attention fixture");
    let input: Vec<f32> = fixture["input"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect();
    let expected: Vec<f32> = fixture["expected"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect();
    let mut actual = vec![0.0; input.len()];
    attention_block(&input, 2, &weights, &mut actual);
    let max_diff = actual
        .iter()
        .zip(expected)
        .map(|(a, e)| (a - e).abs())
        .fold(0.0, f32::max);
    assert!(max_diff < 1e-2, "attention max diff {max_diff}");
}
