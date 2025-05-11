use crate::{
    compute::{random, tensor},
    config,
    layers::{conv, dense, maxpool, relu},
    training::{loss, optimizer},
};

pub(super) struct CpuNetwork {
    conv: conv::Conv2D,
    relu: relu::ReLU,
    pool: maxpool::MaxPool2x2,
    fc: dense::Dense,
    config: config::ModelConfig,
}

impl CpuNetwork {
    pub(super) fn new(config: config::ModelConfig, rng: &mut random::Rng) -> Self {
        CpuNetwork {
            conv: conv::Conv2D::new(
                config.input_channels,
                config.conv_out_channels,
                config.conv_kernel,
                config.conv_padding,
                rng,
            ),
            relu: relu::ReLU::new(),
            pool: maxpool::MaxPool2x2::new(),
            fc: dense::Dense::new(config.flat_dim(), config.num_classes, rng),
            config,
        }
    }

    pub(super) fn train_step_batch(
        &mut self,
        input: &tensor::Tensor,
        targets: &[usize],
        lr: f32,
    ) -> (f32, usize) {
        let (loss, predictions) = self.train_step_batch_with_predictions(input, targets, lr);
        let correct = predictions
            .iter()
            .zip(targets.iter())
            .filter(|(predicted, target)| predicted == target)
            .count();

        (loss, correct)
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

    fn train_step_batch_with_predictions(
        &mut self,
        input: &tensor::Tensor,
        targets: &[usize],
        lr: f32,
    ) -> (f32, Vec<usize>) {
        debug_assert_eq!(input.rank(), 4);
        debug_assert_eq!(input.shape[0], targets.len());
        debug_assert!(targets
            .iter()
            .all(|&target| target < self.config.num_classes));

        let batch_size = input.shape[0];
        debug_assert_eq!(
            input.numel(),
            batch_size
                * self.config.input_channels
                * self.config.input_height
                * self.config.input_width
        );

        let (x, conv_cache) = self.conv.forward(input);
        let (x, relu_cache) = self.relu.forward(&x);
        let (x, pool_cache) = self.pool.forward(&x);

        let flat = x.reshape(vec![batch_size, self.config.flat_dim()]);

        let (logits, fc_cache) = self.fc.forward(&flat);
        let probs = loss::softmax_batch(&logits);

        let loss = loss::cross_entropy_batch(&probs, targets);
        let predictions = loss::argmax_batch(&probs);
        let grad_logits = loss::softmax_ce_grad_batch(&probs, targets);

        let grad_flat = self.fc.backward(&fc_cache, &grad_logits);

        let grad_pool_in = grad_flat.reshape(vec![
            batch_size,
            self.config.conv_out_channels,
            self.config.pool_height(),
            self.config.pool_width(),
        ]);

        let grad_relu_in = self.pool.backward(&pool_cache, &grad_pool_in);
        let grad_conv_in = self.relu.backward(&relu_cache, &grad_relu_in);
        let _ = self.conv.backward(&conv_cache, &grad_conv_in);

        let optimizer = optimizer::Sgd::new(lr);
        optimizer.step(self.conv.trainable_parameters_mut());
        optimizer.step(self.fc.trainable_parameters_mut());

        (loss, predictions)
    }
}
