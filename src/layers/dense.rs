use crate::{
    compute::{random, tensor},
    training::optimizer,
};

pub struct Dense {
    params: DenseParams,
    grads: DenseGrads,
}

struct DenseParams {
    weights: tensor::Tensor,
    bias: tensor::Tensor,
}

struct DenseGrads {
    weights: tensor::Tensor,
    bias: tensor::Tensor,
}

pub struct DenseCache {
    input: tensor::Tensor,
}

impl Dense {
    pub fn new(in_features: usize, out_features: usize, rng: &mut random::Rng) -> Self {
        let scale = (2.0 / in_features as f32).sqrt();

        Dense {
            params: DenseParams {
                weights: tensor::Tensor::random(vec![out_features, in_features], rng, scale),
                bias: tensor::Tensor::zeros(vec![out_features]),
            },
            grads: DenseGrads {
                weights: tensor::Tensor::zeros(vec![out_features, in_features]),
                bias: tensor::Tensor::zeros(vec![out_features]),
            },
        }
    }

    pub fn forward(&self, input: &tensor::Tensor) -> (tensor::Tensor, DenseCache) {
        debug_assert_eq!(input.rank(), 2);

        let n = input.shape[0];
        let in_f = self.params.weights.shape[1];
        let out_f = self.params.weights.shape[0];
        let mut output = tensor::Tensor::zeros(vec![n, out_f]);

        for batch in 0..n {
            for i in 0..out_f {
                let mut sum = self.params.bias.data[i];
                for j in 0..in_f {
                    sum += self.params.weights.data[i * in_f + j] * input.data[batch * in_f + j];
                }
                output.data[batch * out_f + i] = sum;
            }
        }

        (
            output,
            DenseCache {
                input: input.clone(),
            },
        )
    }

    pub fn backward(&mut self, cache: &DenseCache, grad_output: &tensor::Tensor) -> tensor::Tensor {
        let input = &cache.input;
        let n = input.shape[0];
        let in_f = self.params.weights.shape[1];
        let out_f = self.params.weights.shape[0];
        let mut grad_input = tensor::Tensor::zeros(vec![n, in_f]);

        for v in self.grads.weights.data.iter_mut() {
            *v = 0.0;
        }
        for v in self.grads.bias.data.iter_mut() {
            *v = 0.0;
        }

        for batch in 0..n {
            for i in 0..out_f {
                let g = grad_output.data[batch * out_f + i];
                self.grads.bias.data[i] += g;
                for j in 0..in_f {
                    self.grads.weights.data[i * in_f + j] += g * input.data[batch * in_f + j];
                    grad_input.data[batch * in_f + j] += g * self.params.weights.data[i * in_f + j];
                }
            }
        }

        grad_input
    }

    pub fn trainable_parameters_mut(&mut self) -> [optimizer::ParamGrad<'_>; 2] {
        [
            optimizer::ParamGrad::new(&mut self.params.weights, &self.grads.weights),
            optimizer::ParamGrad::new(&mut self.params.bias, &self.grads.bias),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::Dense;
    use crate::compute::{random, tensor::Tensor};

    #[test]
    fn forward_supports_batches() {
        let mut rng = random::Rng::new(1);
        let mut dense = Dense::new(2, 2, &mut rng);
        dense.params.weights.data = vec![1.0, 2.0, 3.0, 4.0];
        dense.params.bias.data = vec![0.5, -0.5];
        let input = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);

        let (output, _) = dense.forward(&input);

        assert_eq!(output.shape, vec![2, 2]);
        assert_eq!(output.data, vec![5.5, 10.5, 11.5, 24.5]);
    }

    #[test]
    fn backward_accumulates_gradients_across_batch() {
        let mut rng = random::Rng::new(1);
        let mut dense = Dense::new(2, 1, &mut rng);
        dense.params.weights.data = vec![2.0, 3.0];
        let input = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let (_, cache) = dense.forward(&input);
        let grad_output = Tensor::from_data(vec![0.5, 1.5], vec![2, 1]);

        let grad_input = dense.backward(&cache, &grad_output);

        assert_eq!(grad_input.shape, vec![2, 2]);
        assert_eq!(grad_input.data, vec![1.0, 1.5, 3.0, 4.5]);
        assert_eq!(dense.grads.bias.data, vec![2.0]);
        assert_eq!(dense.grads.weights.data, vec![5.0, 7.0]);
    }
}
