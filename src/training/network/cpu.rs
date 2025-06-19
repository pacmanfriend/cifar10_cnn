use crate::{
    compute::{random, tensor},
    config,
    training::graph::{Graph, NodeId},
    training::loss,
};

pub(super) struct CpuNetwork {
    graph: Graph,
    conv1_w: NodeId,
    conv1_b: NodeId,
    conv2_w: Option<NodeId>,
    conv2_b: Option<NodeId>,
    fc_w: NodeId,
    fc_b: NodeId,
    config: config::ModelConfig,
}

impl CpuNetwork {
    pub(super) fn new(config: config::ModelConfig, rng: &mut random::Rng) -> Self {
        let mut graph = Graph::new();

        let conv1_w = add_weight(
            &mut graph,
            vec![
                config.conv_out_channels,
                config.input_channels,
                config.conv_kernel,
                config.conv_kernel,
            ],
            config.input_channels * config.conv_kernel * config.conv_kernel,
            rng,
        );
        let conv1_b = graph.add_leaf(tensor::Tensor::zeros(vec![config.conv_out_channels]), true);

        let (conv2_w, conv2_b) = match config.conv2_out_channels {
            Some(out_channels) => {
                let weights = add_weight(
                    &mut graph,
                    vec![
                        out_channels,
                        config.conv_out_channels,
                        config.conv_kernel,
                        config.conv_kernel,
                    ],
                    config.conv_out_channels * config.conv_kernel * config.conv_kernel,
                    rng,
                );
                let bias = graph.add_leaf(tensor::Tensor::zeros(vec![out_channels]), true);
                (Some(weights), Some(bias))
            }
            None => (None, None),
        };

        let fc_w = add_weight(
            &mut graph,
            vec![config.num_classes, config.flat_dim()],
            config.flat_dim(),
            rng,
        );
        let fc_b = graph.add_leaf(tensor::Tensor::zeros(vec![config.num_classes]), true);

        graph.finalize_params();

        CpuNetwork {
            graph,
            conv1_w,
            conv1_b,
            conv2_w,
            conv2_b,
            fc_w,
            fc_b,
            config,
        }
    }

    pub(super) fn train_step_batch(
        &mut self,
        input: &tensor::Tensor,
        targets: &[usize],
        lr: f32,
    ) -> (f32, usize) {
        let (loss, predictions) = self.train_step_batch_with_momentum(input, targets, lr, 0.0);
        let correct = predictions
            .iter()
            .zip(targets.iter())
            .filter(|(predicted, target)| predicted == target)
            .count();

        (loss, correct)
    }

    pub(super) fn train_step_batch_with_momentum(
        &mut self,
        input: &tensor::Tensor,
        targets: &[usize],
        lr: f32,
        momentum: f32,
    ) -> (f32, Vec<usize>) {
        self.train_step_batch_with_predictions_and_momentum(input, targets, lr, momentum)
    }

    pub(super) fn train_step(
        &mut self,
        input: &tensor::Tensor,
        target: usize,
        lr: f32,
    ) -> (f32, usize) {
        let batch_input = input.clone().reshape(vec![
            1,
            self.config.input_channels,
            self.config.input_height,
            self.config.input_width,
        ]);
        let (loss, predictions) =
            self.train_step_batch_with_predictions(&batch_input, &[target], lr);

        (loss, predictions[0])
    }

    pub(super) fn predict_batch(&mut self, input: &tensor::Tensor) -> Vec<usize> {
        let logits = self.forward_logits(input);
        let probs = loss::softmax_batch(&logits);
        loss::argmax_batch(&probs)
    }

    pub(super) fn train_step_batch_with_predictions(
        &mut self,
        input: &tensor::Tensor,
        targets: &[usize],
        lr: f32,
    ) -> (f32, Vec<usize>) {
        self.train_step_batch_with_predictions_and_momentum(input, targets, lr, 0.0)
    }

    fn train_step_batch_with_predictions_and_momentum(
        &mut self,
        input: &tensor::Tensor,
        targets: &[usize],
        lr: f32,
        momentum: f32,
    ) -> (f32, Vec<usize>) {
        debug_assert_eq!(input.rank(), 4);
        debug_assert_eq!(input.shape[0], targets.len());
        debug_assert_eq!(input.shape[1], self.config.input_channels);
        debug_assert_eq!(input.shape[2], self.config.input_height);
        debug_assert_eq!(input.shape[3], self.config.input_width);
        debug_assert!(targets
            .iter()
            .all(|&target| target < self.config.num_classes));

        self.graph.reset_for_iteration();

        let logits = self.forward_logits_node(input);
        let loss = self.graph.softmax_ce(logits, targets);
        let predictions = self.graph.predictions_for_loss(loss);
        let loss_value = self.graph.data(loss).data[0];

        self.graph.backward(loss);
        self.graph.momentum_sgd_step(lr, momentum);

        (loss_value, predictions)
    }

    fn forward_logits(&mut self, input: &tensor::Tensor) -> tensor::Tensor {
        let logits = self.forward_logits_node(input);
        self.graph.data(logits).clone()
    }

    fn forward_logits_node(&mut self, input: &tensor::Tensor) -> NodeId {
        debug_assert_eq!(input.rank(), 4);
        debug_assert_eq!(input.shape[1], self.config.input_channels);
        debug_assert_eq!(input.shape[2], self.config.input_height);
        debug_assert_eq!(input.shape[3], self.config.input_width);

        self.graph.reset_for_iteration();

        let input = self.graph.add_leaf(input.clone(), false);
        let x = self
            .graph
            .conv2d(input, self.conv1_w, self.conv1_b, self.config.conv_padding);
        let x = self.graph.relu(x);
        let x = self.graph.maxpool2x2(x);

        let x = if let (Some(conv2_w), Some(conv2_b)) = (self.conv2_w, self.conv2_b) {
            let x = self
                .graph
                .conv2d(x, conv2_w, conv2_b, self.config.conv_padding);
            let x = self.graph.relu(x);
            self.graph.maxpool2x2(x)
        } else {
            x
        };

        let x = self.graph.flatten(x);
        self.graph.linear(x, self.fc_w, self.fc_b)
    }
}

fn add_weight(
    graph: &mut Graph,
    shape: Vec<usize>,
    fan_in: usize,
    rng: &mut random::Rng,
) -> NodeId {
    let scale = (2.0 / fan_in as f32).sqrt();
    graph.add_leaf(tensor::Tensor::random(shape, rng, scale), true)
}
