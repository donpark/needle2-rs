use needle2_format::CactModel;

#[test]
fn parses_official_needle2_model_when_provided() {
    let Some(path) = std::env::var_os("NEEDLE2_CACT") else {
        eprintln!("SKIP: NEEDLE2_CACT is not set");
        return;
    };
    let bytes = std::fs::read(path).expect("read NEEDLE2_CACT");
    let model = CactModel::parse(&bytes).expect("parse .cact");
    assert_eq!(model.header.tensor_count as usize, model.tensors.len());
    assert_eq!(model.tokenizer.pieces.len(), 8192);
    assert_eq!(
        model.tokenizer.encode("what's it like in Lagos right now?"),
        [1039, 8075, 8049, 506, 848, 301, 441, 493, 370, 2553, 3170, 8100,]
    );
    for (index, tensor) in model.tensors.iter().enumerate() {
        if tensor.dtype == 3 {
            let values = model.tensor_f32(index).expect("decode CQ tensor");
            let count = tensor.shape[..tensor.ndim as usize].iter().product::<u32>() as usize;
            assert_eq!(values.len(), count, "tensor {index}");
        }
    }

    let Some(expected_path) = std::env::var_os("NEEDLE2_CACT_EXPECTED") else {
        eprintln!("SKIP: NEEDLE2_CACT_EXPECTED is not set");
        return;
    };
    let expected: serde_json::Value =
        serde_json::from_slice(&std::fs::read(expected_path).expect("read expected tensor values"))
            .expect("parse expected tensor values");
    for (key, entry) in expected.as_object().expect("expected object") {
        let index: usize = key.parse().expect("tensor index");
        let actual = model.tensor_f32(index).expect("decode expected tensor");
        let values = entry["values"].as_array().expect("expected values");
        let max_diff = values
            .iter()
            .enumerate()
            .map(|(i, value)| (actual[i] - value.as_f64().unwrap() as f32).abs())
            .fold(0.0f32, f32::max);
        assert!(max_diff < 1e-3, "tensor {index} max diff {max_diff}");
    }
}
