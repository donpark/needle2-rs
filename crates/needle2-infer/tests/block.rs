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
}
