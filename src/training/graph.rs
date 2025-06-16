use crate::compute::tensor;

pub type NodeId = usize;

#[derive(Clone)]
pub enum Op {
    Leaf,
    Conv2d {
        input: NodeId,
        weights: NodeId,
        bias: NodeId,
        padding: usize,
        k: usize,
    },
    Relu {
        input: NodeId,
    },
    MaxPool {
        input: NodeId,
    },
    Flatten {
        input: NodeId,
        original_shape: Vec<usize>,
    },
    Linear {
        input: NodeId,
        weights: NodeId,
        bias: NodeId,
    },
    SoftmaxCE {
        logits: NodeId,
        targets: Vec<usize>,
    },
}

pub struct Node {
    pub data: tensor::Tensor,
    pub shape: Vec<usize>,
    pub grad: Option<tensor::Tensor>,
    pub op: Op,
    pub is_param: bool,
    pub aux_argmax: Option<Vec<usize>>,
    pub aux_probs: Option<tensor::Tensor>,
}

pub struct Graph {
    nodes: Vec<Node>,
    parameters: Vec<NodeId>,
    param_count: usize,
}

impl Default for Graph {
    fn default() -> Self {
        Self::new()
    }
}

impl Graph {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            parameters: Vec::new(),
            param_count: 0,
        }
    }

    pub fn finalize_params(&mut self) {
        self.param_count = self.nodes.len();
    }

    pub fn reset_for_iteration(&mut self) {
        self.nodes.truncate(self.param_count);
        for id in self.parameters.iter().copied() {
            self.nodes[id].grad = None;
        }
    }

    pub fn add_leaf(&mut self, data: tensor::Tensor, is_param: bool) -> NodeId {
        let id = self.nodes.len();
        if is_param {
            self.parameters.push(id);
        }
        let shape = data.shape.clone();
        self.nodes.push(Node {
            data,
            shape,
            grad: None,
            op: Op::Leaf,
            is_param,
            aux_argmax: None,
            aux_probs: None,
        });
        id
    }

    pub fn data(&self, id: NodeId) -> &tensor::Tensor {
        &self.nodes[id].data
    }

    pub fn conv2d(
        &mut self,
        input: NodeId,
        weights: NodeId,
        bias: NodeId,
        padding: usize,
    ) -> NodeId {
        let x = &self.nodes[input].data;
        let w = &self.nodes[weights].data;
        let b = &self.nodes[bias].data;
        debug_assert_eq!(x.rank(), 4);
        debug_assert_eq!(w.rank(), 4);

        let n = x.shape[0];
        let c_in = x.shape[1];
        let h = x.shape[2];
        let width = x.shape[3];
        let c_out = w.shape[0];
        let k = w.shape[2];
        let h_out = h + 2 * padding - k + 1;
        let w_out = width + 2 * padding - k + 1;
        let mut output = tensor::Tensor::zeros(vec![n, c_out, h_out, w_out]);
        let pad = padding as isize;

        for batch in 0..n {
            for co in 0..c_out {
                for i in 0..h_out {
                    for j in 0..w_out {
                        let mut sum = b.data[co];
                        for ci in 0..c_in {
                            for ki in 0..k {
                                for kj in 0..k {
                                    let i_in = i as isize + ki as isize - pad;
                                    let j_in = j as isize + kj as isize - pad;
                                    if i_in < 0
                                        || i_in >= h as isize
                                        || j_in < 0
                                        || j_in >= width as isize
                                    {
                                        continue;
                                    }
                                    let in_idx = batch * c_in * h * width
                                        + ci * h * width
                                        + i_in as usize * width
                                        + j_in as usize;
                                    let w_idx = co * c_in * k * k + ci * k * k + ki * k + kj;
                                    sum += x.data[in_idx] * w.data[w_idx];
                                }
                            }
                        }
                        let out_idx =
                            batch * c_out * h_out * w_out + co * h_out * w_out + i * w_out + j;
                        output.data[out_idx] = sum;
                    }
                }
            }
        }

        self.push_op(
            output,
            Op::Conv2d {
                input,
                weights,
                bias,
                padding,
                k,
            },
            None,
            None,
        )
    }

    pub fn relu(&mut self, input: NodeId) -> NodeId {
        let mut output = self.nodes[input].data.clone();
        for value in output.data.iter_mut() {
            if *value < 0.0 {
                *value = 0.0;
            }
        }
        self.push_op(output, Op::Relu { input }, None, None)
    }

    pub fn maxpool2x2(&mut self, input: NodeId) -> NodeId {
        let x = &self.nodes[input].data;
        debug_assert_eq!(x.rank(), 4);

        let n = x.shape[0];
        let c = x.shape[1];
        let h = x.shape[2];
        let w = x.shape[3];
        let h_out = h / 2;
        let w_out = w / 2;
        let mut output = tensor::Tensor::zeros(vec![n, c, h_out, w_out]);
        let mut argmax = vec![0; n * c * h_out * w_out];

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
                                if x.data[idx] > best_val {
                                    best_val = x.data[idx];
                                    best_idx = idx;
                                }
                            }
                        }
                        let out_idx =
                            batch * c * h_out * w_out + ch * h_out * w_out + i * w_out + j;
                        output.data[out_idx] = best_val;
                        argmax[out_idx] = best_idx;
                    }
                }
            }
        }

        self.push_op(output, Op::MaxPool { input }, Some(argmax), None)
    }

    pub fn flatten(&mut self, input: NodeId) -> NodeId {
        let x = &self.nodes[input].data;
        debug_assert_eq!(x.rank(), 4);
        let original_shape = x.shape.clone();
        let batch_size = x.shape[0];
        let features = x.numel() / batch_size;
        let output = x.clone().reshape(vec![batch_size, features]);
        self.push_op(
            output,
            Op::Flatten {
                input,
                original_shape,
            },
            None,
            None,
        )
    }

    pub fn linear(&mut self, input: NodeId, weights: NodeId, bias: NodeId) -> NodeId {
        let x = &self.nodes[input].data;
        let w = &self.nodes[weights].data;
        let b = &self.nodes[bias].data;
        debug_assert_eq!(x.rank(), 2);

        let n = x.shape[0];
        let in_f = w.shape[1];
        let out_f = w.shape[0];
        let mut output = tensor::Tensor::zeros(vec![n, out_f]);

        for batch in 0..n {
            for out in 0..out_f {
                let mut sum = b.data[out];
                for input_feature in 0..in_f {
                    sum +=
                        w.data[out * in_f + input_feature] * x.data[batch * in_f + input_feature];
                }
                output.data[batch * out_f + out] = sum;
            }
        }

        self.push_op(
            output,
            Op::Linear {
                input,
                weights,
                bias,
            },
            None,
            None,
        )
    }

    pub fn softmax_ce(&mut self, logits: NodeId, targets: &[usize]) -> NodeId {
        let logits_data = &self.nodes[logits].data;
        debug_assert_eq!(logits_data.rank(), 2);

        let n = logits_data.shape[0];
        let classes = logits_data.shape[1];
        debug_assert_eq!(targets.len(), n);

        let mut probs = tensor::Tensor::zeros(logits_data.shape.clone());
        let mut total = 0.0;
        for (batch, target) in targets.iter().copied().enumerate().take(n) {
            let row = &logits_data.data[batch * classes..(batch + 1) * classes];
            let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0;
            for (class, logit) in row.iter().copied().enumerate().take(classes) {
                let value = (logit - max_val).exp();
                probs.data[batch * classes + class] = value;
                sum += value;
            }
            for class in 0..classes {
                probs.data[batch * classes + class] /= sum;
            }
            total += -(probs.data[batch * classes + target].max(1e-12)).ln();
        }

        let loss = tensor::Tensor::from_data(vec![total / n as f32], vec![1]);
        self.push_op(
            loss,
            Op::SoftmaxCE {
                logits,
                targets: targets.to_vec(),
            },
            None,
            Some(probs),
        )
    }

    pub fn predictions_for_loss(&self, loss: NodeId) -> Vec<usize> {
        let probs = self.nodes[loss]
            .aux_probs
            .as_ref()
            .expect("loss node must store probabilities");
        let n = probs.shape[0];
        let classes = probs.shape[1];
        let mut predictions = Vec::with_capacity(n);
        for batch in 0..n {
            let row = &probs.data[batch * classes..(batch + 1) * classes];
            predictions.push(
                row.iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .unwrap()
                    .0,
            );
        }
        predictions
    }

    pub fn backward(&mut self, loss: NodeId) {
        self.nodes[loss].grad = Some(tensor::Tensor::from_data(vec![1.0], vec![1]));

        for id in (0..=loss).rev() {
            if self.nodes[id].grad.is_none() {
                continue;
            }
            self.backward_node(id);
        }
    }

    pub fn sgd_step(&mut self, lr: f32) {
        for id in self.parameters.iter().copied() {
            if let Some(grad) = self.nodes[id].grad.clone() {
                let param = &mut self.nodes[id].data;
                debug_assert_eq!(param.shape, grad.shape);
                for (value, grad) in param.data.iter_mut().zip(grad.data.iter()) {
                    *value -= lr * grad;
                }
            }
        }
    }

    fn push_op(
        &mut self,
        data: tensor::Tensor,
        op: Op,
        aux_argmax: Option<Vec<usize>>,
        aux_probs: Option<tensor::Tensor>,
    ) -> NodeId {
        let id = self.nodes.len();
        let shape = data.shape.clone();
        self.nodes.push(Node {
            data,
            shape,
            grad: None,
            op,
            is_param: false,
            aux_argmax,
            aux_probs,
        });
        id
    }

    fn add_grad(&mut self, id: NodeId, grad: tensor::Tensor) {
        if let Some(existing) = self.nodes[id].grad.as_mut() {
            debug_assert_eq!(existing.shape, grad.shape);
            for (dst, src) in existing.data.iter_mut().zip(grad.data.iter()) {
                *dst += src;
            }
        } else {
            self.nodes[id].grad = Some(grad);
        }
    }

    fn backward_node(&mut self, id: NodeId) {
        let grad = self.nodes[id].grad.clone().expect("gradient must exist");
        match self.nodes[id].op.clone() {
            Op::Leaf => {}
            Op::Relu { input } => {
                let x = &self.nodes[input].data;
                let mut grad_input = grad.clone();
                for i in 0..grad_input.data.len() {
                    if x.data[i] <= 0.0 {
                        grad_input.data[i] = 0.0;
                    }
                }
                self.add_grad(input, grad_input);
            }
            Op::MaxPool { input } => {
                let input_shape = self.nodes[input].shape.clone();
                let mut grad_input = tensor::Tensor::zeros(input_shape);
                let argmax = self.nodes[id]
                    .aux_argmax
                    .as_ref()
                    .expect("maxpool node must store argmax indices");
                for (out_idx, &in_idx) in argmax.iter().enumerate() {
                    grad_input.data[in_idx] += grad.data[out_idx];
                }
                self.add_grad(input, grad_input);
            }
            Op::Flatten {
                input,
                original_shape,
            } => {
                self.add_grad(input, grad.reshape(original_shape));
            }
            Op::SoftmaxCE { logits, targets } => {
                let probs = self.nodes[id]
                    .aux_probs
                    .as_ref()
                    .expect("softmax CE node must store probabilities");
                let n = probs.shape[0];
                let classes = probs.shape[1];
                let mut grad_logits = probs.clone();
                for (batch, target) in targets.iter().copied().enumerate() {
                    grad_logits.data[batch * classes + target] -= 1.0;
                }
                let scale = grad.data[0] / n as f32;
                for value in grad_logits.data.iter_mut() {
                    *value *= scale;
                }
                self.add_grad(logits, grad_logits);
            }
            Op::Linear {
                input,
                weights,
                bias,
            } => self.backward_linear(input, weights, bias, &grad),
            Op::Conv2d {
                input,
                weights,
                bias,
                padding,
                k,
            } => self.backward_conv2d(input, weights, bias, padding, k, &grad),
        }
    }

    fn backward_linear(
        &mut self,
        input: NodeId,
        weights: NodeId,
        bias: NodeId,
        grad: &tensor::Tensor,
    ) {
        let x = &self.nodes[input].data;
        let w = &self.nodes[weights].data;
        let n = x.shape[0];
        let in_f = w.shape[1];
        let out_f = w.shape[0];
        let mut grad_input = tensor::Tensor::zeros(x.shape.clone());
        let mut grad_weights = tensor::Tensor::zeros(w.shape.clone());
        let mut grad_bias = tensor::Tensor::zeros(self.nodes[bias].shape.clone());

        for batch in 0..n {
            for out in 0..out_f {
                let g = grad.data[batch * out_f + out];
                grad_bias.data[out] += g;
                for input_feature in 0..in_f {
                    grad_weights.data[out * in_f + input_feature] +=
                        g * x.data[batch * in_f + input_feature];
                    grad_input.data[batch * in_f + input_feature] +=
                        g * w.data[out * in_f + input_feature];
                }
            }
        }

        self.add_grad(input, grad_input);
        self.add_grad(weights, grad_weights);
        self.add_grad(bias, grad_bias);
    }

    fn backward_conv2d(
        &mut self,
        input: NodeId,
        weights: NodeId,
        bias: NodeId,
        padding: usize,
        k: usize,
        grad: &tensor::Tensor,
    ) {
        let x = &self.nodes[input].data;
        let w = &self.nodes[weights].data;
        let n = x.shape[0];
        let c_in = x.shape[1];
        let h = x.shape[2];
        let width = x.shape[3];
        let c_out = w.shape[0];
        let h_out = h + 2 * padding - k + 1;
        let w_out = width + 2 * padding - k + 1;
        let pad = padding as isize;
        let mut grad_input = tensor::Tensor::zeros(x.shape.clone());
        let mut grad_weights = tensor::Tensor::zeros(w.shape.clone());
        let mut grad_bias = tensor::Tensor::zeros(self.nodes[bias].shape.clone());

        for batch in 0..n {
            for co in 0..c_out {
                for i in 0..h_out {
                    for j in 0..w_out {
                        let g = grad.data
                            [batch * c_out * h_out * w_out + co * h_out * w_out + i * w_out + j];
                        grad_bias.data[co] += g;
                        for ci in 0..c_in {
                            for ki in 0..k {
                                for kj in 0..k {
                                    let i_in = i as isize + ki as isize - pad;
                                    let j_in = j as isize + kj as isize - pad;
                                    if i_in < 0
                                        || i_in >= h as isize
                                        || j_in < 0
                                        || j_in >= width as isize
                                    {
                                        continue;
                                    }
                                    let in_idx = batch * c_in * h * width
                                        + ci * h * width
                                        + i_in as usize * width
                                        + j_in as usize;
                                    let w_idx = co * c_in * k * k + ci * k * k + ki * k + kj;
                                    grad_weights.data[w_idx] += x.data[in_idx] * g;
                                    grad_input.data[in_idx] += w.data[w_idx] * g;
                                }
                            }
                        }
                    }
                }
            }
        }

        self.add_grad(input, grad_input);
        self.add_grad(weights, grad_weights);
        self.add_grad(bias, grad_bias);
    }
}

#[cfg(test)]
mod tests {
    use super::Graph;
    use crate::compute::tensor::Tensor;

    #[test]
    fn linear_softmax_ce_backward_populates_parameter_grads() {
        let mut graph = Graph::new();
        let weights = graph.add_leaf(
            Tensor::from_data(vec![0.1, 0.2, 0.3, 0.4], vec![2, 2]),
            true,
        );
        let bias = graph.add_leaf(Tensor::zeros(vec![2]), true);
        graph.finalize_params();
        graph.reset_for_iteration();

        let input = graph.add_leaf(Tensor::from_data(vec![1.0, 2.0], vec![1, 2]), false);
        let logits = graph.linear(input, weights, bias);
        let loss = graph.softmax_ce(logits, &[1]);
        let before = graph.data(weights).data.clone();

        graph.backward(loss);
        graph.sgd_step(0.1);

        assert!(graph.data(loss).data[0].is_finite());
        assert_ne!(graph.data(weights).data, before);
    }

    #[test]
    fn conv2d_padding_preserves_spatial_shape() {
        let mut graph = Graph::new();
        let weights = graph.add_leaf(Tensor::from_data(vec![1.0; 9], vec![1, 1, 3, 3]), true);
        let bias = graph.add_leaf(Tensor::zeros(vec![1]), true);
        graph.finalize_params();
        graph.reset_for_iteration();

        let input = graph.add_leaf(Tensor::zeros(vec![1, 1, 3, 3]), false);
        let output = graph.conv2d(input, weights, bias, 1);

        assert_eq!(graph.data(output).shape, vec![1, 1, 3, 3]);
    }

    #[test]
    fn reset_keeps_parameters_and_drops_activations() {
        let mut graph = Graph::new();
        let weights = graph.add_leaf(Tensor::from_data(vec![1.0, 2.0], vec![1, 2]), true);
        let bias = graph.add_leaf(Tensor::zeros(vec![1]), true);
        graph.finalize_params();

        let input = graph.add_leaf(Tensor::from_data(vec![3.0, 4.0], vec![1, 2]), false);
        let output = graph.linear(input, weights, bias);

        assert!(graph.nodes.len() > graph.param_count);
        assert_eq!(graph.data(output).shape, vec![1, 1]);

        graph.nodes[weights].grad = Some(Tensor::from_data(vec![1.0, 1.0], vec![1, 2]));
        graph.reset_for_iteration();

        assert_eq!(graph.nodes.len(), graph.param_count);
        assert_eq!(graph.nodes[weights].data.data, vec![1.0, 2.0]);
        assert!(graph.nodes[weights].grad.is_none());
        assert!(graph.nodes[bias].grad.is_none());
    }

    #[test]
    fn accumulate_grad_adds_multiple_contributions() {
        let mut graph = Graph::new();
        let input = graph.add_leaf(Tensor::zeros(vec![2]), false);

        graph.add_grad(input, Tensor::from_data(vec![1.0, 2.0], vec![2]));
        graph.add_grad(input, Tensor::from_data(vec![3.0, 4.0], vec![2]));

        assert_eq!(
            graph.nodes[input].grad.as_ref().unwrap().data,
            vec![4.0, 6.0]
        );
    }

    #[test]
    fn backward_covers_conv_relu_pool_flatten_linear_chain() {
        let mut graph = Graph::new();
        let conv_w = graph.add_leaf(Tensor::from_data(vec![0.1; 9], vec![1, 1, 3, 3]), true);
        let conv_b = graph.add_leaf(Tensor::zeros(vec![1]), true);
        let linear_w = graph.add_leaf(
            Tensor::from_data(vec![0.1, -0.2, 0.3, -0.4, -0.1, 0.2, -0.3, 0.4], vec![2, 4]),
            true,
        );
        let linear_b = graph.add_leaf(Tensor::zeros(vec![2]), true);
        graph.finalize_params();
        graph.reset_for_iteration();

        let input = graph.add_leaf(
            Tensor::from_data(
                vec![
                    1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 1.5, -1.5, 2.0, 1.0, -1.0, -2.0, 0.0, 0.25,
                    -0.25, 0.75,
                ],
                vec![1, 1, 4, 4],
            ),
            false,
        );
        let x = graph.conv2d(input, conv_w, conv_b, 1);
        let x = graph.relu(x);
        let x = graph.maxpool2x2(x);
        let x = graph.flatten(x);
        let logits = graph.linear(x, linear_w, linear_b);
        let loss = graph.softmax_ce(logits, &[1]);
        let before = graph.data(conv_w).data.clone();

        graph.backward(loss);
        graph.sgd_step(0.01);

        for id in [input, conv_w, conv_b, linear_w, linear_b] {
            let grad = graph.nodes[id]
                .grad
                .as_ref()
                .expect("node must receive a backward gradient");
            assert!(
                grad.data.iter().all(|value| value.is_finite()),
                "non-finite gradient for node {id}"
            );
        }
        assert_ne!(graph.data(conv_w).data, before);
    }
}
