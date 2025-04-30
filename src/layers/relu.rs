use crate::compute::tensor;

pub struct ReLU;

pub struct ReLUCache {
    input: tensor::Tensor,
}

impl Default for ReLU {
    fn default() -> Self {
        Self::new()
    }
}

impl ReLU {
    pub fn new() -> Self {
        ReLU
    }

    pub fn forward(&self, input: &tensor::Tensor) -> (tensor::Tensor, ReLUCache) {
        let mut output = input.clone();
        for v in output.data.iter_mut() {
            if *v < 0.0 {
                *v = 0.0;
            }
        }

        (
            output,
            ReLUCache {
                input: input.clone(),
            },
        )
    }

    pub fn backward(&self, cache: &ReLUCache, grad_output: &tensor::Tensor) -> tensor::Tensor {
        let input = &cache.input;
        let mut grad_input = grad_output.clone();

        for i in 0..grad_input.data.len() {
            if input.data[i] <= 0.0 {
                grad_input.data[i] = 0.0;
            }
        }
        grad_input
    }
}

#[cfg(test)]
mod tests {
    use super::ReLU;
    use crate::compute::tensor::Tensor;

    #[test]
    fn backward_uses_explicit_cache() {
        let relu = ReLU::new();
        let first = Tensor::from_data(vec![-1.0, 2.0], vec![2]);
        let second = Tensor::from_data(vec![3.0, -4.0], vec![2]);

        let (_, first_cache) = relu.forward(&first);
        let _ = relu.forward(&second);

        let grad = Tensor::from_data(vec![1.0, 1.0], vec![2]);
        let grad_input = relu.backward(&first_cache, &grad);

        assert_eq!(grad_input.data, vec![0.0, 1.0]);
    }
}
