use crate::compute::tensor;

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

pub fn softmax_batch(logits: &tensor::Tensor) -> tensor::Tensor {
    debug_assert_eq!(logits.rank(), 2);

    let n = logits.shape[0];
    let classes = logits.shape[1];
    let mut data = vec![0.0; logits.numel()];

    for batch in 0..n {
        let row = &logits.data[batch * classes..(batch + 1) * classes];
        let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0;

        for class in 0..classes {
            let value = (row[class] - max_val).exp();
            data[batch * classes + class] = value;
            sum += value;
        }

        for class in 0..classes {
            data[batch * classes + class] /= sum;
        }
    }

    tensor::Tensor::from_data(data, logits.shape.clone())
}

pub fn cross_entropy(probs: &tensor::Tensor, target: usize) -> f32 {
    -(probs.data[target].max(1e-12)).ln()
}

pub fn cross_entropy_batch(probs: &tensor::Tensor, targets: &[usize]) -> f32 {
    debug_assert_eq!(probs.rank(), 2);

    let n = probs.shape[0];
    let classes = probs.shape[1];
    debug_assert_eq!(targets.len(), n);

    let mut total = 0.0;
    for (batch, &target) in targets.iter().enumerate() {
        debug_assert!(target < classes);
        total += -(probs.data[batch * classes + target].max(1e-12)).ln();
    }

    total / n as f32
}

pub fn argmax(t: &tensor::Tensor) -> usize {
    t.data
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0
}

pub fn argmax_batch(probs: &tensor::Tensor) -> Vec<usize> {
    debug_assert_eq!(probs.rank(), 2);

    let n = probs.shape[0];
    let classes = probs.shape[1];
    let mut predictions = Vec::with_capacity(n);

    for batch in 0..n {
        let row = &probs.data[batch * classes..(batch + 1) * classes];
        let predicted = row
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        predictions.push(predicted);
    }

    predictions
}

pub fn softmax_ce_grad_batch(probs: &tensor::Tensor, targets: &[usize]) -> tensor::Tensor {
    debug_assert_eq!(probs.rank(), 2);

    let n = probs.shape[0];
    let classes = probs.shape[1];
    debug_assert_eq!(targets.len(), n);

    let mut grad = probs.clone();
    for (batch, &target) in targets.iter().enumerate() {
        debug_assert!(target < classes);
        grad.data[batch * classes + target] -= 1.0;
    }

    let scale = 1.0 / n as f32;
    for value in grad.data.iter_mut() {
        *value *= scale;
    }

    grad
}

#[cfg(test)]
mod tests {
    use super::{
        argmax, argmax_batch, cross_entropy, cross_entropy_batch, softmax, softmax_batch,
        softmax_ce_grad_batch,
    };
    use crate::compute::tensor::Tensor;

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

    #[test]
    fn softmax_batch_outputs_probabilities_per_row() {
        let logits = Tensor::from_data(vec![1.0, 2.0, 3.0, 3.0, 2.0, 1.0], vec![2, 3]);
        let probs = softmax_batch(&logits);

        assert_eq!(probs.shape, vec![2, 3]);
        assert!((probs.data[0..3].iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!((probs.data[3..6].iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert_eq!(argmax_batch(&probs), vec![2, 0]);
    }

    #[test]
    fn cross_entropy_batch_averages_losses() {
        let probs = Tensor::from_data(vec![0.1, 0.8, 0.1, 0.7, 0.2, 0.1], vec![2, 3]);
        let loss = cross_entropy_batch(&probs, &[1, 0]);
        let expected = (-0.8_f32.ln() - 0.7_f32.ln()) / 2.0;

        assert!((loss - expected).abs() < 1e-6);
    }

    #[test]
    fn softmax_ce_grad_batch_scales_by_batch_size() {
        let probs = Tensor::from_data(vec![0.1, 0.8, 0.1, 0.7, 0.2, 0.1], vec![2, 3]);
        let grad = softmax_ce_grad_batch(&probs, &[1, 0]);

        assert_eq!(grad.shape, vec![2, 3]);
        let expected = [0.05, -0.1, 0.05, -0.15, 0.1, 0.05];
        for (actual, expected) in grad.data.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-6);
        }
    }
}
