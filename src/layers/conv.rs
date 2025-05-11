use crate::{
    compute::{random, tensor},
    training::optimizer,
};

pub struct Conv2D {
    params: Conv2DParams,
    grads: Conv2DGrads,
    padding: usize,
}

struct Conv2DParams {
    weights: tensor::Tensor,
    bias: tensor::Tensor,
}

struct Conv2DGrads {
    weights: tensor::Tensor,
    bias: tensor::Tensor,
}

pub struct Conv2DCache {
    input: tensor::Tensor,
}

impl Conv2D {
    pub fn new(c_in: usize, c_out: usize, k: usize, padding: usize, rng: &mut random::Rng) -> Self {
        let fan_in = c_in * k * k;
        let scale = (2.0 / fan_in as f32).sqrt();

        Conv2D {
            params: Conv2DParams {
                weights: tensor::Tensor::random(vec![c_out, c_in, k, k], rng, scale),
                bias: tensor::Tensor::zeros(vec![c_out]),
            },
            grads: Conv2DGrads {
                weights: tensor::Tensor::zeros(vec![c_out, c_in, k, k]),
                bias: tensor::Tensor::zeros(vec![c_out]),
            },
            padding,
        }
    }

    pub fn forward(&self, input: &tensor::Tensor) -> (tensor::Tensor, Conv2DCache) {
        debug_assert_eq!(input.rank(), 4);

        let n = input.shape[0];
        let c_in = input.shape[1];
        let h = input.shape[2];
        let w = input.shape[3];
        let c_out = self.params.weights.shape[0];
        let k = self.params.weights.shape[2];
        let h_out = h + 2 * self.padding - k + 1;
        let w_out = w + 2 * self.padding - k + 1;

        let mut output = tensor::Tensor::zeros(vec![n, c_out, h_out, w_out]);
        let padding = self.padding as isize;

        for batch in 0..n {
            for co in 0..c_out {
                for i in 0..h_out {
                    for j in 0..w_out {
                        let mut sum = self.params.bias.data[co];
                        for ci in 0..c_in {
                            for ki in 0..k {
                                for kj in 0..k {
                                    let i_in = i as isize + ki as isize - padding;
                                    let j_in = j as isize + kj as isize - padding;
                                    if i_in < 0
                                        || i_in >= h as isize
                                        || j_in < 0
                                        || j_in >= w as isize
                                    {
                                        continue;
                                    }

                                    let i_in = i_in as usize;
                                    let j_in = j_in as usize;
                                    let in_idx =
                                        batch * c_in * h * w + ci * h * w + i_in * w + j_in;
                                    let w_idx = co * c_in * k * k + ci * k * k + ki * k + kj;
                                    sum += input.data[in_idx] * self.params.weights.data[w_idx];
                                }
                            }
                        }
                        output.data
                            [batch * c_out * h_out * w_out + co * h_out * w_out + i * w_out + j] =
                            sum;
                    }
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
        let n = input.shape[0];
        let c_in = input.shape[1];
        let h = input.shape[2];
        let w = input.shape[3];
        let c_out = self.params.weights.shape[0];
        let k = self.params.weights.shape[2];
        let h_out = h + 2 * self.padding - k + 1;
        let w_out = w + 2 * self.padding - k + 1;

        let mut grad_input = tensor::Tensor::zeros(vec![n, c_in, h, w]);
        let padding = self.padding as isize;

        for v in self.grads.weights.data.iter_mut() {
            *v = 0.0;
        }
        for v in self.grads.bias.data.iter_mut() {
            *v = 0.0;
        }

        for batch in 0..n {
            for co in 0..c_out {
                for i in 0..h_out {
                    for j in 0..w_out {
                        let g = grad_output.data
                            [batch * c_out * h_out * w_out + co * h_out * w_out + i * w_out + j];
                        self.grads.bias.data[co] += g;
                        for ci in 0..c_in {
                            for ki in 0..k {
                                for kj in 0..k {
                                    let i_in = i as isize + ki as isize - padding;
                                    let j_in = j as isize + kj as isize - padding;
                                    if i_in < 0
                                        || i_in >= h as isize
                                        || j_in < 0
                                        || j_in >= w as isize
                                    {
                                        continue;
                                    }

                                    let i_in = i_in as usize;
                                    let j_in = j_in as usize;
                                    let in_idx =
                                        batch * c_in * h * w + ci * h * w + i_in * w + j_in;
                                    let w_idx = co * c_in * k * k + ci * k * k + ki * k + kj;
                                    self.grads.weights.data[w_idx] += input.data[in_idx] * g;
                                    grad_input.data[in_idx] += self.params.weights.data[w_idx] * g;
                                }
                            }
                        }
                    }
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
    use super::Conv2D;
    use crate::compute::{random, tensor::Tensor};

    #[test]
    fn forward_without_padding_matches_existing_shape() {
        let mut rng = random::Rng::new(1);
        let conv = Conv2D::new(1, 2, 3, 0, &mut rng);
        let input = Tensor::zeros(vec![1, 1, 5, 5]);

        let (output, _) = conv.forward(&input);

        assert_eq!(output.shape, vec![1, 2, 3, 3]);
    }

    #[test]
    fn forward_with_padding_preserves_spatial_dims() {
        let mut rng = random::Rng::new(1);
        let conv = Conv2D::new(1, 2, 3, 1, &mut rng);
        let input = Tensor::zeros(vec![1, 1, 3, 3]);

        let (output, _) = conv.forward(&input);

        assert_eq!(output.shape, vec![1, 2, 3, 3]);
    }

    #[test]
    fn backward_with_padding_returns_input_shape_and_valid_gradients() {
        let mut rng = random::Rng::new(1);
        let mut conv = Conv2D::new(1, 1, 3, 1, &mut rng);
        conv.params.weights.data.fill(1.0);
        conv.params.bias.data.fill(0.0);

        let input = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 2, 2]);
        let (output, cache) = conv.forward(&input);
        let grad_output = Tensor::from_data(vec![1.0; output.numel()], output.shape);

        let grad_input = conv.backward(&cache, &grad_output);

        assert_eq!(grad_input.shape, vec![1, 1, 2, 2]);
        assert_eq!(grad_input.data, vec![4.0, 4.0, 4.0, 4.0]);
    }

    #[test]
    fn forward_supports_multiple_batch_items() {
        let mut rng = random::Rng::new(1);
        let mut conv = Conv2D::new(1, 1, 1, 0, &mut rng);
        conv.params.weights.data.fill(2.0);
        conv.params.bias.data.fill(1.0);
        let input = Tensor::from_data(vec![1.0, 2.0], vec![2, 1, 1, 1]);

        let (output, _) = conv.forward(&input);

        assert_eq!(output.shape, vec![2, 1, 1, 1]);
        assert_eq!(output.data, vec![3.0, 5.0]);
    }
}
