use crate::{
    compute::{random, tensor},
    config,
    layers::{conv, dense, maxpool, relu},
    training::{loss, optimizer},
};

pub(super) struct CpuNetwork {
    conv1: conv::Conv2D,
    relu1: relu::ReLU,
    pool1: maxpool::MaxPool2x2,
    conv2: Option<conv::Conv2D>,
    relu2: Option<relu::ReLU>,
    pool2: Option<maxpool::MaxPool2x2>,
    fc: dense::Dense,
    config: config::ModelConfig,
}

impl CpuNetwork {
    pub(super) fn new(config: config::ModelConfig, rng: &mut random::Rng) -> Self {
        let conv1 = conv::Conv2D::new(
            config.input_channels,
            config.conv_out_channels,
            config.conv_kernel,
            config.conv_padding,
            rng,
        );
        let conv2 = config.conv2_out_channels.map(|out_channels| {
            conv::Conv2D::new(
                config.conv_out_channels,
                out_channels,
                config.conv_kernel,
                config.conv_padding,
                rng,
            )
        });
        let relu2 = config.conv2_out_channels.map(|_| relu::ReLU::new());
        let pool2 = config
            .conv2_out_channels
            .map(|_| maxpool::MaxPool2x2::new());

        CpuNetwork {
            conv1,
            relu1: relu::ReLU::new(),
            pool1: maxpool::MaxPool2x2::new(),
            conv2,
            relu2,
            pool2,
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
        debug_assert_eq!(input.shape[1], self.config.input_channels);
        debug_assert_eq!(input.shape[2], self.config.input_height);
        debug_assert_eq!(input.shape[3], self.config.input_width);
        debug_assert_eq!(
            input.numel(),
            batch_size
                * self.config.input_channels
                * self.config.input_height
                * self.config.input_width
        );

        let (x, conv1_cache) = self.conv1.forward(input);
        let (x, relu1_cache) = self.relu1.forward(&x);
        let (x, pool1_cache) = self.pool1.forward(&x);

        if let (Some(conv2), Some(relu2), Some(pool2), Some(conv2_out_channels)) = (
            self.conv2.as_ref(),
            self.relu2.as_ref(),
            self.pool2.as_ref(),
            self.config.conv2_out_channels,
        ) {
            let (x, conv2_cache) = conv2.forward(&x);
            let (x, relu2_cache) = relu2.forward(&x);
            let (x, pool2_cache) = pool2.forward(&x);

            let flat = x.reshape(vec![batch_size, self.config.flat_dim()]);

            let (logits, fc_cache) = self.fc.forward(&flat);
            let probs = loss::softmax_batch(&logits);

            let loss = loss::cross_entropy_batch(&probs, targets);
            let predictions = loss::argmax_batch(&probs);
            let grad_logits = loss::softmax_ce_grad_batch(&probs, targets);

            let grad_flat = self.fc.backward(&fc_cache, &grad_logits);

            let grad_pool2_in = grad_flat.reshape(vec![
                batch_size,
                conv2_out_channels,
                self.config.pool2_height(),
                self.config.pool2_width(),
            ]);

            let grad_relu2_in = pool2.backward(&pool2_cache, &grad_pool2_in);
            let grad_conv2_in = relu2.backward(&relu2_cache, &grad_relu2_in);
            let grad_pool1_in = self
                .conv2
                .as_mut()
                .unwrap()
                .backward(&conv2_cache, &grad_conv2_in);
            let grad_relu1_in = self.pool1.backward(&pool1_cache, &grad_pool1_in);
            let grad_conv1_in = self.relu1.backward(&relu1_cache, &grad_relu1_in);
            let _ = self.conv1.backward(&conv1_cache, &grad_conv1_in);

            let optimizer = optimizer::Sgd::new(lr);
            optimizer.step(self.conv1.trainable_parameters_mut());
            optimizer.step(self.conv2.as_mut().unwrap().trainable_parameters_mut());
            optimizer.step(self.fc.trainable_parameters_mut());

            return (loss, predictions);
        }

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

        let grad_relu_in = self.pool1.backward(&pool1_cache, &grad_pool_in);
        let grad_conv_in = self.relu1.backward(&relu1_cache, &grad_relu_in);
        let _ = self.conv1.backward(&conv1_cache, &grad_conv_in);

        let optimizer = optimizer::Sgd::new(lr);
        optimizer.step(self.conv1.trainable_parameters_mut());
        optimizer.step(self.fc.trainable_parameters_mut());

        (loss, predictions)
    }
}
