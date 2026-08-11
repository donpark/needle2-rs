use needle2_format::CactModel;
use needle2_infer::{
    engram_forward, infer_logits, stack_forward, stack_mhc_forward, ConfidenceWeights,
    EngramWeights, D_MODEL, VOCAB_SIZE,
};

#[test]
fn runs_the_full_sequential_stack_on_official_model() {
    let Some(path) = std::env::var_os("NEEDLE2_CACT") else {
        eprintln!("SKIP: NEEDLE2_CACT is not set");
        return;
    };
    let bytes = std::fs::read(path).expect("read model");
    let model = CactModel::parse(&bytes).expect("parse model");
    let input = vec![0.0; D_MODEL];
    let mut output = vec![0.0; D_MODEL];
    stack_forward(&model, &input, 1, &mut output).expect("run stack");
    assert!(output.iter().all(|value| value.is_finite()));
    let mut mhc_output = vec![0.0; D_MODEL];
    stack_mhc_forward(&model, &[101], &input, 1, &mut mhc_output).expect("run mHC stack");
    assert!(mhc_output.iter().all(|value| value.is_finite()));
    let confidence = ConfidenceWeights::from_cact(&model).expect("load confidence head");
    assert_eq!(confidence.probes.len(), 8 * D_MODEL);
    assert_eq!(confidence.proj.len(), 4096);
    let engram = EngramWeights::from_cact(&model, 0).expect("load engram");
    let mut key = vec![0.0; 3 * D_MODEL];
    let mut value = vec![0.0; key.len()];
    engram_forward(&[101, 202, 303], &engram, &mut key, &mut value);
    assert!(key.iter().chain(&value).all(|number| number.is_finite()));
    let mut logits = vec![0.0; VOCAB_SIZE];
    infer_logits(&model, &[101], &mut logits).expect("run logits");
    assert!(logits.iter().all(|number| number.is_finite()));
}
