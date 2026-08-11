use needle2_format::CactModel;
use needle2_infer::{stack_forward, stack_mhc_forward, D_MODEL};

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
    stack_mhc_forward(&model, &input, 1, &mut mhc_output).expect("run mHC stack");
    assert!(mhc_output.iter().all(|value| value.is_finite()));
}
