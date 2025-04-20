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

#[cfg(test)]
mod tests {
    use super::{argmax, cross_entropy, softmax};
    use crate::tensor::Tensor;

    #[test]
    fn softmax_outputs_probabilities() {
        let logits = Tensor::from_data(vec![1.0, 2.0, 3.0], vec![3]);
        let probs = softmax(&logits);
        let sum: f32 = probs.data.iter().sum();

        assert!((sum - 1.0).abs() < 1e-6);
        assert_eq!(probs.shape, vec![3]);
        assert_eq!(argmax(&probs), 2);
    }

    #[test]
    fn cross_entropy_uses_target_probability() {
        let probs = Tensor::from_data(vec![0.1, 0.8, 0.1], vec![3]);
        let loss = cross_entropy(&probs, 1);
        let expected = -0.8_f32.ln();

        assert!((loss - expected).abs() < 1e-6);
    }
}
