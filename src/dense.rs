use crate::{optimizer, random, tensor};

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
        let in_f = self.params.weights.shape[1];
        let out_f = self.params.weights.shape[0];
        let mut output = tensor::Tensor::zeros(vec![out_f]);

        for i in 0..out_f {
            let mut sum = self.params.bias.data[i];
            for j in 0..in_f {
                sum += self.params.weights.data[i * in_f + j] * input.data[j];
            }
            output.data[i] = sum;
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
        let in_f = self.params.weights.shape[1];
        let out_f = self.params.weights.shape[0];
        let mut grad_input = tensor::Tensor::zeros(vec![in_f]);

        // Прямое вычисление трёх производных:
        //   dL/db_i  = g_i
        //   dL/dW_ij = g_i * x_j
        //   dL/dx_j  = Σ_i W_ij * g_i
        for i in 0..out_f {
            let g = grad_output.data[i];
            self.grads.bias.data[i] = g;
            for j in 0..in_f {
                self.grads.weights.data[i * in_f + j] = g * input.data[j];
                grad_input.data[j] += g * self.params.weights.data[i * in_f + j];
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
