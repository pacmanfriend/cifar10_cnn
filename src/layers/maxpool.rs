use crate::compute::tensor;

pub struct MaxPool2x2;

pub struct MaxPool2x2Cache {
    max_indices: Vec<usize>,
    input_shape: Vec<usize>,
}

impl Default for MaxPool2x2 {
    fn default() -> Self {
        Self::new()
    }
}

impl MaxPool2x2 {
    pub fn new() -> Self {
        MaxPool2x2
    }

    pub fn forward(&self, input: &tensor::Tensor) -> (tensor::Tensor, MaxPool2x2Cache) {
        debug_assert_eq!(input.rank(), 4);

        let n = input.shape[0];
        let c = input.shape[1];
        let h = input.shape[2];
        let w = input.shape[3];
        let h_out = h / 2;
        let w_out = w / 2;

        let mut output = tensor::Tensor::zeros(vec![n, c, h_out, w_out]);
        let mut max_indices = vec![0; n * c * h_out * w_out];

        for batch in 0..n {
            for ch in 0..c {
                for i in 0..h_out {
                    for j in 0..w_out {
                        let mut best_val = f32::NEG_INFINITY;
                        let mut best_idx = 0;

                        for di in 0..2 {
                            for dj in 0..2 {
                                let idx = batch * c * h * w
                                    + ch * h * w
                                    + (2 * i + di) * w
                                    + (2 * j + dj);
                                if input.data[idx] > best_val {
                                    best_val = input.data[idx];
                                    best_idx = idx;
                                }
                            }
                        }

                        let out_idx =
                            batch * c * h_out * w_out + ch * h_out * w_out + i * w_out + j;

                        output.data[out_idx] = best_val;
                        max_indices[out_idx] = best_idx;
                    }
                }
            }
        }

        (
            output,
            MaxPool2x2Cache {
                max_indices,
                input_shape: input.shape.clone(),
            },
        )
    }

    pub fn backward(
        &self,
        cache: &MaxPool2x2Cache,
        grad_output: &tensor::Tensor,
    ) -> tensor::Tensor {
        let mut grad_input = tensor::Tensor::zeros(cache.input_shape.clone());
        // Градиент течёт только в ту позицию, которая дала максимум.
        for (out_idx, &in_idx) in cache.max_indices.iter().enumerate() {
            grad_input.data[in_idx] += grad_output.data[out_idx];
        }
        grad_input
    }
}

#[cfg(test)]
mod tests {
    use super::MaxPool2x2;
    use crate::compute::tensor::Tensor;

    #[test]
    fn backward_uses_explicit_cache() {
        let pool = MaxPool2x2::new();
        let first = Tensor::from_data(vec![1.0, 4.0, 2.0, 3.0], vec![1, 1, 2, 2]);
        let second = Tensor::from_data(vec![9.0, 1.0, 2.0, 3.0], vec![1, 1, 2, 2]);

        let (_, first_cache) = pool.forward(&first);
        let _ = pool.forward(&second);

        let grad = Tensor::from_data(vec![1.0], vec![1, 1, 1, 1]);
        let grad_input = pool.backward(&first_cache, &grad);

        assert_eq!(grad_input.data, vec![0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn forward_and_backward_support_batches() {
        let pool = MaxPool2x2::new();
        let input = Tensor::from_data(
            vec![1.0, 4.0, 2.0, 3.0, 9.0, 1.0, 2.0, 3.0],
            vec![2, 1, 2, 2],
        );

        let (output, cache) = pool.forward(&input);
        let grad = Tensor::from_data(vec![1.0, 2.0], vec![2, 1, 1, 1]);
        let grad_input = pool.backward(&cache, &grad);

        assert_eq!(output.shape, vec![2, 1, 1, 1]);
        assert_eq!(output.data, vec![4.0, 9.0]);
        assert_eq!(
            grad_input.data,
            vec![0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0]
        );
    }
}
