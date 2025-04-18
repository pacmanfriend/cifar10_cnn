use crate::tensor;

pub fn softmax(logits: &tensor::Tensor) -> tensor::Tensor {
    let max_val = logits
        .data
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);

    let exps: Vec<f32> = logits.data.iter().map(|x| (x - max_val).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let data = exps.iter().map(|e| e / sum).collect();

    tensor::Tensor::from_data(data, logits.shape.clone())
}

pub fn cross_entropy(probs: &tensor::Tensor, target: usize) -> f32 {
    -(probs.data[target].max(1e-12)).ln()
}

pub fn argmax(t: &tensor::Tensor) -> usize {
    t.data
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0
}
