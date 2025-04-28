use crate::{random, tensor};

pub struct Conv2D {
    weights: tensor::Tensor,
    bias: tensor::Tensor,
    grad_w: tensor::Tensor,
    grad_b: tensor::Tensor,
}

pub struct Conv2DCache {
    input: tensor::Tensor,
}

impl Conv2D {
    pub fn new(c_in: usize, c_out: usize, k: usize, rng: &mut random::Rng) -> Self {
        let fan_in = c_in * k * k;
        let scale = (2.0 / fan_in as f32).sqrt();

        Conv2D {
            weights: tensor::Tensor::random(vec![c_out, c_in, k, k], rng, scale),
            bias: tensor::Tensor::zeros(vec![c_out]),
            grad_w: tensor::Tensor::zeros(vec![c_out, c_in, k, k]),
            grad_b: tensor::Tensor::zeros(vec![c_out]),
        }
    }

    pub fn forward(&self, input: &tensor::Tensor) -> (tensor::Tensor, Conv2DCache) {
        debug_assert_eq!(input.rank(), 3);

        let c_in = input.shape[0];
        let h = input.shape[1];
        let w = input.shape[2];
        let c_out = self.weights.shape[0];
        let k = self.weights.shape[2];
        let h_out = h - k + 1;
        let w_out = w - k + 1;

        let mut output = tensor::Tensor::zeros(vec![c_out, h_out, w_out]);

        for co in 0..c_out {
            for i in 0..h_out {
                for j in 0..w_out {
                    let mut sum = self.bias.data[co];
                    for ci in 0..c_in {
                        for ki in 0..k {
                            for kj in 0..k {
                                let in_idx = ci * h * w + (i + ki) * w + (j + kj);
                                let w_idx = co * c_in * k * k + ci * k * k + ki * k + kj;
                                sum += input.data[in_idx] * self.weights.data[w_idx];
                            }
                        }
                    }
                    output.data[co * h_out * w_out + i * w_out + j] = sum;
                }
            }
        }

        (
            output,
            Conv2DCache {
                input: input.clone(),
            },
        )
    }

    pub fn backward(
        &mut self,
        cache: &Conv2DCache,
        grad_output: &tensor::Tensor,
    ) -> tensor::Tensor {
        let input = &cache.input;
        let c_in = input.shape[0];
        let h = input.shape[1];
        let w = input.shape[2];
        let c_out = self.weights.shape[0];
        let k = self.weights.shape[2];
        let h_out = h - k + 1;
        let w_out = w - k + 1;

        let mut grad_input = tensor::Tensor::zeros(vec![c_in, h, w]);

        for v in self.grad_w.data.iter_mut() {
            *v = 0.0;
        }
        for v in self.grad_b.data.iter_mut() {
            *v = 0.0;
        }

        for co in 0..c_out {
            for i in 0..h_out {
                for j in 0..w_out {
                    let g = grad_output.data[co * h_out * w_out + i * w_out + j];
                    self.grad_b.data[co] += g;
                    for ci in 0..c_in {
                        for ki in 0..k {
                            for kj in 0..k {
                                let in_idx = ci * h * w + (i + ki) * w + (j + kj);
                                let w_idx = co * c_in * k * k + ci * k * k + ki * k + kj;
                                self.grad_w.data[w_idx] += input.data[in_idx] * g;
                                grad_input.data[in_idx] += self.weights.data[w_idx] * g;
                            }
                        }
                    }
                }
            }
        }

        grad_input
    }

    pub fn step(&mut self, lr: f32) {
        for i in 0..self.weights.data.len() {
            self.weights.data[i] -= lr * self.grad_w.data[i];
        }
        for i in 0..self.bias.data.len() {
            self.bias.data[i] -= lr * self.grad_b.data[i];
        }
    }
}
