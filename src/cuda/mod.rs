use crate::{
    compute::{random, tensor},
    config,
};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, DriverError, LaunchConfig,
    PushKernelArg,
};
use cudarc::nvrtc::compile_ptx;
use std::{error::Error, sync::Arc};

mod kernels;

pub struct CudaNetwork {
    config: config::ModelConfig,
    stream: Arc<CudaStream>,
    module: Arc<CudaModule>,
    conv1_w: CudaSlice<f32>,
    conv1_b: CudaSlice<f32>,
    conv2_w: Option<CudaSlice<f32>>,
    conv2_b: Option<CudaSlice<f32>>,
    linear_w: CudaSlice<f32>,
    linear_b: CudaSlice<f32>,
    grad_conv1_w: CudaSlice<f32>,
    grad_conv1_b: CudaSlice<f32>,
    grad_conv2_w: Option<CudaSlice<f32>>,
    grad_conv2_b: Option<CudaSlice<f32>>,
    grad_linear_w: CudaSlice<f32>,
    grad_linear_b: CudaSlice<f32>,
    velocity_conv1_w: CudaSlice<f32>,
    velocity_conv1_b: CudaSlice<f32>,
    velocity_conv2_w: Option<CudaSlice<f32>>,
    velocity_conv2_b: Option<CudaSlice<f32>>,
    velocity_linear_w: CudaSlice<f32>,
    velocity_linear_b: CudaSlice<f32>,
}

#[derive(Clone, Copy)]
struct ConvShape {
    n: usize,
    c_in: usize,
    h: usize,
    w: usize,
    c_out: usize,
    h_out: usize,
    w_out: usize,
    kernel: usize,
    padding: usize,
}

impl ConvShape {
    fn output_len(self) -> usize {
        self.n * self.c_out * self.h_out * self.w_out
    }

    fn weight_len(self) -> usize {
        self.c_out * self.c_in * self.kernel * self.kernel
    }
}

struct Conv2Cache {
    shape: ConvShape,
    input: CudaSlice<f32>,
    relu_out: CudaSlice<f32>,
    max_indices: CudaSlice<i32>,
    pool_len: usize,
}

impl CudaNetwork {
    pub fn new(config: config::ModelConfig, rng: &mut random::Rng) -> Result<Self, Box<dyn Error>> {
        let ctx = CudaContext::new(0)?;
        let stream = ctx.default_stream();
        println!("GPU 0: {}", ctx.name()?);

        let ptx = compile_ptx(kernels::SOURCE)?;
        let module = ctx.load_module(ptx)?;
        println!("Compiled {} CUDA kernels with NVRTC", kernels::NAMES.len());

        let conv1_w_host = init_weight(config.conv1_weight_len(), config.input_channels, rng);
        let conv1_b_host = vec![0.0_f32; config.conv_out_channels];

        let (conv2_w, conv2_b, grad_conv2_w, grad_conv2_b, velocity_conv2_w, velocity_conv2_b) =
            match config.conv2_out_channels {
                Some(out_channels) => {
                    let weights =
                        init_weight(config.conv2_weight_len(), config.conv_out_channels, rng);
                    let bias = vec![0.0_f32; out_channels];
                    (
                        Some(stream.memcpy_stod(&weights)?),
                        Some(stream.memcpy_stod(&bias)?),
                        Some(stream.alloc_zeros::<f32>(config.conv2_weight_len())?),
                        Some(stream.alloc_zeros::<f32>(out_channels)?),
                        Some(stream.alloc_zeros::<f32>(config.conv2_weight_len())?),
                        Some(stream.alloc_zeros::<f32>(out_channels)?),
                    )
                }
                None => (None, None, None, None, None, None),
            };

        let linear_w_host = init_weight(
            config.num_classes * config.flat_dim(),
            config.flat_dim(),
            rng,
        );
        let linear_b_host = vec![0.0_f32; config.num_classes];

        Ok(Self {
            config,
            stream: stream.clone(),
            module,
            conv1_w: stream.memcpy_stod(&conv1_w_host)?,
            conv1_b: stream.memcpy_stod(&conv1_b_host)?,
            conv2_w,
            conv2_b,
            linear_w: stream.memcpy_stod(&linear_w_host)?,
            linear_b: stream.memcpy_stod(&linear_b_host)?,
            grad_conv1_w: stream.alloc_zeros::<f32>(config.conv1_weight_len())?,
            grad_conv1_b: stream.alloc_zeros::<f32>(config.conv_out_channels)?,
            grad_conv2_w,
            grad_conv2_b,
            grad_linear_w: stream.alloc_zeros::<f32>(config.num_classes * config.flat_dim())?,
            grad_linear_b: stream.alloc_zeros::<f32>(config.num_classes)?,
            velocity_conv1_w: stream.alloc_zeros::<f32>(config.conv1_weight_len())?,
            velocity_conv1_b: stream.alloc_zeros::<f32>(config.conv_out_channels)?,
            velocity_conv2_w,
            velocity_conv2_b,
            velocity_linear_w: stream.alloc_zeros::<f32>(config.num_classes * config.flat_dim())?,
            velocity_linear_b: stream.alloc_zeros::<f32>(config.num_classes)?,
        })
    }

    fn kernel(&self, name: &str) -> Result<CudaFunction, DriverError> {
        self.module.load_function(name)
    }

    pub fn train_step(
        &mut self,
        input_host: &[f32],
        target: usize,
        lr: f32,
    ) -> Result<(f32, usize), DriverError> {
        debug_assert_eq!(input_host.len(), self.config.input_dim());
        let input = tensor::Tensor::from_data(
            input_host.to_vec(),
            vec![
                1,
                self.config.input_channels,
                self.config.input_height,
                self.config.input_width,
            ],
        );
        let (loss, predictions) = self.train_step_batch_with_predictions(&input, &[target], lr)?;

        Ok((loss, predictions[0]))
    }

    pub fn train_step_batch(
        &mut self,
        input: &tensor::Tensor,
        targets: &[usize],
        lr: f32,
    ) -> Result<(f32, usize), DriverError> {
        self.train_step_batch_with_momentum(input, targets, lr, 0.0)
    }

    pub fn train_step_batch_with_momentum(
        &mut self,
        input: &tensor::Tensor,
        targets: &[usize],
        lr: f32,
        momentum: f32,
    ) -> Result<(f32, usize), DriverError> {
        let (loss, predictions) =
            self.train_step_batch_with_predictions_and_momentum(input, targets, lr, momentum)?;
        let correct = predictions
            .iter()
            .zip(targets.iter())
            .filter(|(predicted, target)| predicted == target)
            .count();

        Ok((loss, correct))
    }

    pub fn predict_batch(&mut self, input: &tensor::Tensor) -> Result<Vec<usize>, DriverError> {
        debug_assert_eq!(input.rank(), 4);
        debug_assert_eq!(input.shape[1], self.config.input_channels);
        debug_assert_eq!(input.shape[2], self.config.input_height);
        debug_assert_eq!(input.shape[3], self.config.input_width);

        let batch_size = input.shape[0];
        let input_dev = self.stream.memcpy_stod(&input.data)?;

        let conv1 = ConvShape {
            n: batch_size,
            c_in: self.config.input_channels,
            h: self.config.input_height,
            w: self.config.input_width,
            c_out: self.config.conv_out_channels,
            h_out: self.config.conv1_height(),
            w_out: self.config.conv1_width(),
            kernel: self.config.conv_kernel,
            padding: self.config.conv_padding,
        };
        let mut conv1_out = self.stream.alloc_zeros::<f32>(conv1.output_len())?;
        let mut relu1_out = self.stream.alloc_zeros::<f32>(conv1.output_len())?;
        let pool1_len = batch_size
            * self.config.conv_out_channels
            * self.config.pool1_height()
            * self.config.pool1_width();
        let mut pool1_out = self.stream.alloc_zeros::<f32>(pool1_len)?;
        let mut max_indices1 = self.stream.alloc_zeros::<i32>(pool1_len)?;

        launch_conv_forward(
            &self.stream,
            &self.kernel("conv2d_forward")?,
            &mut conv1_out,
            &input_dev,
            &self.conv1_w,
            &self.conv1_b,
            conv1,
        )?;
        launch_relu_forward(
            &self.stream,
            &self.kernel("relu_forward")?,
            &mut relu1_out,
            &conv1_out,
            conv1.output_len(),
        )?;
        launch_maxpool_forward(
            &self.stream,
            &self.kernel("maxpool2x2_forward")?,
            &mut pool1_out,
            &mut max_indices1,
            &relu1_out,
            batch_size,
            self.config.conv_out_channels,
            self.config.conv1_height(),
            self.config.conv1_width(),
        )?;

        let linear_input = if let (Some(conv2_w), Some(conv2_b)) =
            (self.conv2_w.as_ref(), self.conv2_b.as_ref())
        {
            let conv2_out_channels = self.config.conv2_out_channels.unwrap();
            let conv2 = ConvShape {
                n: batch_size,
                c_in: self.config.conv_out_channels,
                h: self.config.pool1_height(),
                w: self.config.pool1_width(),
                c_out: conv2_out_channels,
                h_out: self.config.conv2_height(),
                w_out: self.config.conv2_width(),
                kernel: self.config.conv_kernel,
                padding: self.config.conv_padding,
            };
            let mut conv2_out = self.stream.alloc_zeros::<f32>(conv2.output_len())?;
            let mut relu2_out = self.stream.alloc_zeros::<f32>(conv2.output_len())?;
            let pool2_len = batch_size
                * conv2_out_channels
                * self.config.pool2_height()
                * self.config.pool2_width();
            let mut pool2_out = self.stream.alloc_zeros::<f32>(pool2_len)?;
            let mut max_indices2 = self.stream.alloc_zeros::<i32>(pool2_len)?;

            launch_conv_forward(
                &self.stream,
                &self.kernel("conv2d_forward")?,
                &mut conv2_out,
                &pool1_out,
                conv2_w,
                conv2_b,
                conv2,
            )?;
            launch_relu_forward(
                &self.stream,
                &self.kernel("relu_forward")?,
                &mut relu2_out,
                &conv2_out,
                conv2.output_len(),
            )?;
            launch_maxpool_forward(
                &self.stream,
                &self.kernel("maxpool2x2_forward")?,
                &mut pool2_out,
                &mut max_indices2,
                &relu2_out,
                batch_size,
                conv2_out_channels,
                self.config.conv2_height(),
                self.config.conv2_width(),
            )?;
            pool2_out
        } else {
            pool1_out
        };

        let mut logits = self
            .stream
            .alloc_zeros::<f32>(batch_size * self.config.num_classes)?;
        let mut probs = self
            .stream
            .alloc_zeros::<f32>(batch_size * self.config.num_classes)?;
        let mut losses = self.stream.alloc_zeros::<f32>(batch_size)?;
        let dummy_targets = vec![0_i32; batch_size];
        let targets_dev = self.stream.memcpy_stod(&dummy_targets)?;

        launch_linear_forward(
            &self.stream,
            &self.kernel("linear_forward")?,
            &mut logits,
            &linear_input,
            &self.linear_w,
            &self.linear_b,
            batch_size,
            self.config.flat_dim(),
            self.config.num_classes,
        )?;
        launch_softmax_forward(
            &self.stream,
            &self.kernel("softmax_ce_forward")?,
            &mut probs,
            &mut losses,
            &logits,
            &targets_dev,
            batch_size,
            self.config.num_classes,
        )?;

        let probs_host = self.stream.memcpy_dtov(&probs)?;
        Ok(predictions_from_probs(&probs_host, self.config.num_classes))
    }

    pub fn train_step_batch_with_predictions(
        &mut self,
        input: &tensor::Tensor,
        targets: &[usize],
        lr: f32,
    ) -> Result<(f32, Vec<usize>), DriverError> {
        self.train_step_batch_with_predictions_and_momentum(input, targets, lr, 0.0)
    }

    fn train_step_batch_with_predictions_and_momentum(
        &mut self,
        input: &tensor::Tensor,
        targets: &[usize],
        lr: f32,
        momentum: f32,
    ) -> Result<(f32, Vec<usize>), DriverError> {
        debug_assert_eq!(input.rank(), 4);
        debug_assert_eq!(input.shape[0], targets.len());
        debug_assert_eq!(input.shape[1], self.config.input_channels);
        debug_assert_eq!(input.shape[2], self.config.input_height);
        debug_assert_eq!(input.shape[3], self.config.input_width);
        debug_assert!(targets
            .iter()
            .all(|&target| target < self.config.num_classes));

        let batch_size = targets.len();
        let targets_i32 = targets
            .iter()
            .map(|&target| target as i32)
            .collect::<Vec<_>>();
        let input_dev = self.stream.memcpy_stod(&input.data)?;
        let targets_dev = self.stream.memcpy_stod(&targets_i32)?;

        let conv1 = ConvShape {
            n: batch_size,
            c_in: self.config.input_channels,
            h: self.config.input_height,
            w: self.config.input_width,
            c_out: self.config.conv_out_channels,
            h_out: self.config.conv1_height(),
            w_out: self.config.conv1_width(),
            kernel: self.config.conv_kernel,
            padding: self.config.conv_padding,
        };
        let mut conv1_out = self.stream.alloc_zeros::<f32>(conv1.output_len())?;
        let mut relu1_out = self.stream.alloc_zeros::<f32>(conv1.output_len())?;
        let pool1_len = batch_size
            * self.config.conv_out_channels
            * self.config.pool1_height()
            * self.config.pool1_width();
        let mut pool1_out = self.stream.alloc_zeros::<f32>(pool1_len)?;
        let mut max_indices1 = self.stream.alloc_zeros::<i32>(pool1_len)?;

        launch_conv_forward(
            &self.stream,
            &self.kernel("conv2d_forward")?,
            &mut conv1_out,
            &input_dev,
            &self.conv1_w,
            &self.conv1_b,
            conv1,
        )?;
        launch_relu_forward(
            &self.stream,
            &self.kernel("relu_forward")?,
            &mut relu1_out,
            &conv1_out,
            conv1.output_len(),
        )?;
        launch_maxpool_forward(
            &self.stream,
            &self.kernel("maxpool2x2_forward")?,
            &mut pool1_out,
            &mut max_indices1,
            &relu1_out,
            batch_size,
            self.config.conv_out_channels,
            self.config.conv1_height(),
            self.config.conv1_width(),
        )?;

        let (linear_input, conv2_cache) = if let (Some(conv2_w), Some(conv2_b)) =
            (self.conv2_w.as_ref(), self.conv2_b.as_ref())
        {
            let conv2_out_channels = self.config.conv2_out_channels.unwrap();
            let conv2 = ConvShape {
                n: batch_size,
                c_in: self.config.conv_out_channels,
                h: self.config.pool1_height(),
                w: self.config.pool1_width(),
                c_out: conv2_out_channels,
                h_out: self.config.conv2_height(),
                w_out: self.config.conv2_width(),
                kernel: self.config.conv_kernel,
                padding: self.config.conv_padding,
            };
            let mut conv2_out = self.stream.alloc_zeros::<f32>(conv2.output_len())?;
            let mut relu2_out = self.stream.alloc_zeros::<f32>(conv2.output_len())?;
            let pool2_len = batch_size
                * conv2_out_channels
                * self.config.pool2_height()
                * self.config.pool2_width();
            let mut pool2_out = self.stream.alloc_zeros::<f32>(pool2_len)?;
            let mut max_indices2 = self.stream.alloc_zeros::<i32>(pool2_len)?;

            launch_conv_forward(
                &self.stream,
                &self.kernel("conv2d_forward")?,
                &mut conv2_out,
                &pool1_out,
                conv2_w,
                conv2_b,
                conv2,
            )?;
            launch_relu_forward(
                &self.stream,
                &self.kernel("relu_forward")?,
                &mut relu2_out,
                &conv2_out,
                conv2.output_len(),
            )?;
            launch_maxpool_forward(
                &self.stream,
                &self.kernel("maxpool2x2_forward")?,
                &mut pool2_out,
                &mut max_indices2,
                &relu2_out,
                batch_size,
                conv2_out_channels,
                self.config.conv2_height(),
                self.config.conv2_width(),
            )?;

            (
                pool2_out,
                Some(Conv2Cache {
                    shape: conv2,
                    input: pool1_out,
                    relu_out: relu2_out,
                    max_indices: max_indices2,
                    pool_len: pool2_len,
                }),
            )
        } else {
            (pool1_out, None)
        };

        let mut logits = self
            .stream
            .alloc_zeros::<f32>(batch_size * self.config.num_classes)?;
        let mut probs = self
            .stream
            .alloc_zeros::<f32>(batch_size * self.config.num_classes)?;
        let mut losses = self.stream.alloc_zeros::<f32>(batch_size)?;
        let mut grad_logits = self
            .stream
            .alloc_zeros::<f32>(batch_size * self.config.num_classes)?;
        let mut grad_flat = self
            .stream
            .alloc_zeros::<f32>(batch_size * self.config.flat_dim())?;

        launch_linear_forward(
            &self.stream,
            &self.kernel("linear_forward")?,
            &mut logits,
            &linear_input,
            &self.linear_w,
            &self.linear_b,
            batch_size,
            self.config.flat_dim(),
            self.config.num_classes,
        )?;
        launch_softmax_forward(
            &self.stream,
            &self.kernel("softmax_ce_forward")?,
            &mut probs,
            &mut losses,
            &logits,
            &targets_dev,
            batch_size,
            self.config.num_classes,
        )?;
        launch_softmax_backward(
            &self.stream,
            &self.kernel("softmax_ce_backward")?,
            &mut grad_logits,
            &probs,
            &targets_dev,
            batch_size,
            self.config.num_classes,
        )?;

        let losses_host = self.stream.memcpy_dtov(&losses)?;
        let probs_host = self.stream.memcpy_dtov(&probs)?;
        let loss = losses_host.iter().sum::<f32>() / batch_size as f32;
        let predictions = predictions_from_probs(&probs_host, self.config.num_classes);

        launch_linear_backward_input(
            &self.stream,
            &self.kernel("linear_backward_input")?,
            &mut grad_flat,
            &grad_logits,
            &self.linear_w,
            batch_size,
            self.config.flat_dim(),
            self.config.num_classes,
        )?;
        launch_linear_backward_weight(
            &self.stream,
            &self.kernel("linear_backward_weight")?,
            &mut self.grad_linear_w,
            &grad_logits,
            &linear_input,
            batch_size,
            self.config.flat_dim(),
            self.config.num_classes,
        )?;
        launch_linear_backward_bias(
            &self.stream,
            &self.kernel("linear_backward_bias")?,
            &mut self.grad_linear_b,
            &grad_logits,
            batch_size,
            self.config.num_classes,
        )?;

        let grad_pool1 = if let Some(cache) = conv2_cache {
            let mut grad_relu2 = self.stream.alloc_zeros::<f32>(cache.shape.output_len())?;
            let mut grad_conv2 = self.stream.alloc_zeros::<f32>(cache.shape.output_len())?;
            let mut grad_pool1 = self.stream.alloc_zeros::<f32>(pool1_len)?;

            launch_zero(
                &self.stream,
                &self.kernel("zero_buffer")?,
                &mut grad_relu2,
                cache.shape.output_len(),
            )?;
            launch_maxpool_backward(
                &self.stream,
                &self.kernel("maxpool2x2_backward")?,
                &mut grad_relu2,
                &grad_flat,
                &cache.max_indices,
                cache.pool_len,
            )?;
            launch_relu_backward(
                &self.stream,
                &self.kernel("relu_backward")?,
                &mut grad_conv2,
                &grad_relu2,
                &cache.relu_out,
                cache.shape.output_len(),
            )?;
            launch_conv_backward_input(
                &self.stream,
                &self.kernel("conv2d_backward_input")?,
                &mut grad_pool1,
                &grad_conv2,
                self.conv2_w.as_ref().unwrap(),
                cache.shape,
            )?;
            launch_conv_backward_weight(
                &self.stream,
                &self.kernel("conv2d_backward_weight")?,
                self.grad_conv2_w.as_mut().unwrap(),
                &grad_conv2,
                &cache.input,
                cache.shape,
            )?;
            launch_conv_backward_bias(
                &self.stream,
                &self.kernel("conv2d_backward_bias")?,
                self.grad_conv2_b.as_mut().unwrap(),
                &grad_conv2,
                cache.shape,
            )?;

            grad_pool1
        } else {
            grad_flat
        };

        let mut grad_relu1 = self.stream.alloc_zeros::<f32>(conv1.output_len())?;
        let mut grad_conv1 = self.stream.alloc_zeros::<f32>(conv1.output_len())?;
        launch_zero(
            &self.stream,
            &self.kernel("zero_buffer")?,
            &mut grad_relu1,
            conv1.output_len(),
        )?;
        launch_maxpool_backward(
            &self.stream,
            &self.kernel("maxpool2x2_backward")?,
            &mut grad_relu1,
            &grad_pool1,
            &max_indices1,
            pool1_len,
        )?;
        launch_relu_backward(
            &self.stream,
            &self.kernel("relu_backward")?,
            &mut grad_conv1,
            &grad_relu1,
            &relu1_out,
            conv1.output_len(),
        )?;
        launch_conv_backward_weight(
            &self.stream,
            &self.kernel("conv2d_backward_weight")?,
            &mut self.grad_conv1_w,
            &grad_conv1,
            &input_dev,
            conv1,
        )?;
        launch_conv_backward_bias(
            &self.stream,
            &self.kernel("conv2d_backward_bias")?,
            &mut self.grad_conv1_b,
            &grad_conv1,
            conv1,
        )?;

        self.launch_sgd_all(lr, momentum)?;

        Ok((loss, predictions))
    }

    fn launch_sgd_all(&mut self, lr: f32, momentum: f32) -> Result<(), DriverError> {
        let sgd = self.kernel("momentum_sgd_update")?;
        launch_sgd(
            &self.stream,
            &sgd,
            &mut self.conv1_w,
            &self.grad_conv1_w,
            &mut self.velocity_conv1_w,
            lr,
            momentum,
            self.config.conv1_weight_len(),
        )?;
        launch_sgd(
            &self.stream,
            &sgd,
            &mut self.conv1_b,
            &self.grad_conv1_b,
            &mut self.velocity_conv1_b,
            lr,
            momentum,
            self.config.conv_out_channels,
        )?;
        if let (Some(conv2_w), Some(grad_conv2_w), Some(velocity_conv2_w)) = (
            self.conv2_w.as_mut(),
            self.grad_conv2_w.as_ref(),
            self.velocity_conv2_w.as_mut(),
        ) {
            launch_sgd(
                &self.stream,
                &sgd,
                conv2_w,
                grad_conv2_w,
                velocity_conv2_w,
                lr,
                momentum,
                self.config.conv2_weight_len(),
            )?;
        }
        if let (Some(conv2_b), Some(grad_conv2_b), Some(velocity_conv2_b), Some(out_channels)) = (
            self.conv2_b.as_mut(),
            self.grad_conv2_b.as_ref(),
            self.velocity_conv2_b.as_mut(),
            self.config.conv2_out_channels,
        ) {
            launch_sgd(
                &self.stream,
                &sgd,
                conv2_b,
                grad_conv2_b,
                velocity_conv2_b,
                lr,
                momentum,
                out_channels,
            )?;
        }
        launch_sgd(
            &self.stream,
            &sgd,
            &mut self.linear_w,
            &self.grad_linear_w,
            &mut self.velocity_linear_w,
            lr,
            momentum,
            self.config.num_classes * self.config.flat_dim(),
        )?;
        launch_sgd(
            &self.stream,
            &sgd,
            &mut self.linear_b,
            &self.grad_linear_b,
            &mut self.velocity_linear_b,
            lr,
            momentum,
            self.config.num_classes,
        )
    }
}

fn init_weight(len: usize, fan_in: usize, rng: &mut random::Rng) -> Vec<f32> {
    let scale = (2.0 / fan_in as f32).sqrt();
    (0..len).map(|_| rng.normal() * scale).collect()
}

fn conv_grid(height: usize, width: usize, depth: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (
            width.div_ceil(8) as u32,
            height.div_ceil(8) as u32,
            depth as u32,
        ),
        block_dim: (8, 8, 1),
        shared_mem_bytes: 0,
    }
}

fn linear_grid(out_dim: usize, batch_size: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (out_dim.div_ceil(256) as u32, batch_size as u32, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    }
}

fn predictions_from_probs(probs: &[f32], classes: usize) -> Vec<usize> {
    probs
        .chunks_exact(classes)
        .map(|row| {
            row.iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0
        })
        .collect()
}

fn launch_conv_forward(
    stream: &CudaStream,
    kernel: &CudaFunction,
    output: &mut CudaSlice<f32>,
    input: &CudaSlice<f32>,
    weights: &CudaSlice<f32>,
    bias: &CudaSlice<f32>,
    shape: ConvShape,
) -> Result<(), DriverError> {
    let n = shape.n as i32;
    let c_in = shape.c_in as i32;
    let h = shape.h as i32;
    let w = shape.w as i32;
    let c_out = shape.c_out as i32;
    let k = shape.kernel as i32;
    let pad = shape.padding as i32;
    let mut args = stream.launch_builder(kernel);
    args.arg(output);
    args.arg(input);
    args.arg(weights);
    args.arg(bias);
    args.arg(&n);
    args.arg(&c_in);
    args.arg(&h);
    args.arg(&w);
    args.arg(&c_out);
    args.arg(&k);
    args.arg(&pad);
    unsafe { args.launch(conv_grid(shape.h_out, shape.w_out, shape.n * shape.c_out))? };
    Ok(())
}

fn launch_conv_backward_input(
    stream: &CudaStream,
    kernel: &CudaFunction,
    grad_input: &mut CudaSlice<f32>,
    grad_output: &CudaSlice<f32>,
    weights: &CudaSlice<f32>,
    shape: ConvShape,
) -> Result<(), DriverError> {
    let n = shape.n as i32;
    let c_in = shape.c_in as i32;
    let h = shape.h as i32;
    let w = shape.w as i32;
    let c_out = shape.c_out as i32;
    let k = shape.kernel as i32;
    let pad = shape.padding as i32;
    let mut args = stream.launch_builder(kernel);
    args.arg(grad_input);
    args.arg(grad_output);
    args.arg(weights);
    args.arg(&n);
    args.arg(&c_in);
    args.arg(&h);
    args.arg(&w);
    args.arg(&c_out);
    args.arg(&k);
    args.arg(&pad);
    unsafe { args.launch(conv_grid(shape.h, shape.w, shape.n * shape.c_in))? };
    Ok(())
}

fn launch_conv_backward_weight(
    stream: &CudaStream,
    kernel: &CudaFunction,
    grad_weights: &mut CudaSlice<f32>,
    grad_output: &CudaSlice<f32>,
    input: &CudaSlice<f32>,
    shape: ConvShape,
) -> Result<(), DriverError> {
    let n = shape.n as i32;
    let c_in = shape.c_in as i32;
    let h = shape.h as i32;
    let w = shape.w as i32;
    let c_out = shape.c_out as i32;
    let k = shape.kernel as i32;
    let pad = shape.padding as i32;
    let mut args = stream.launch_builder(kernel);
    args.arg(grad_weights);
    args.arg(grad_output);
    args.arg(input);
    args.arg(&n);
    args.arg(&c_in);
    args.arg(&h);
    args.arg(&w);
    args.arg(&c_out);
    args.arg(&k);
    args.arg(&pad);
    unsafe { args.launch(LaunchConfig::for_num_elems(shape.weight_len() as u32))? };
    Ok(())
}

fn launch_conv_backward_bias(
    stream: &CudaStream,
    kernel: &CudaFunction,
    grad_bias: &mut CudaSlice<f32>,
    grad_output: &CudaSlice<f32>,
    shape: ConvShape,
) -> Result<(), DriverError> {
    let n = shape.n as i32;
    let c_out = shape.c_out as i32;
    let h_out = shape.h_out as i32;
    let w_out = shape.w_out as i32;
    let mut args = stream.launch_builder(kernel);
    args.arg(grad_bias);
    args.arg(grad_output);
    args.arg(&n);
    args.arg(&c_out);
    args.arg(&h_out);
    args.arg(&w_out);
    unsafe { args.launch(LaunchConfig::for_num_elems(shape.c_out as u32))? };
    Ok(())
}

fn launch_relu_forward(
    stream: &CudaStream,
    kernel: &CudaFunction,
    output: &mut CudaSlice<f32>,
    input: &CudaSlice<f32>,
    len: usize,
) -> Result<(), DriverError> {
    let len = len as i32;
    let mut args = stream.launch_builder(kernel);
    args.arg(output);
    args.arg(input);
    args.arg(&len);
    unsafe { args.launch(LaunchConfig::for_num_elems(len as u32))? };
    Ok(())
}

fn launch_relu_backward(
    stream: &CudaStream,
    kernel: &CudaFunction,
    grad_input: &mut CudaSlice<f32>,
    grad_output: &CudaSlice<f32>,
    post_act: &CudaSlice<f32>,
    len: usize,
) -> Result<(), DriverError> {
    let len = len as i32;
    let mut args = stream.launch_builder(kernel);
    args.arg(grad_input);
    args.arg(grad_output);
    args.arg(post_act);
    args.arg(&len);
    unsafe { args.launch(LaunchConfig::for_num_elems(len as u32))? };
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn launch_maxpool_forward(
    stream: &CudaStream,
    kernel: &CudaFunction,
    output: &mut CudaSlice<f32>,
    max_indices: &mut CudaSlice<i32>,
    input: &CudaSlice<f32>,
    n: usize,
    c: usize,
    h: usize,
    w: usize,
) -> Result<(), DriverError> {
    let n_i = n as i32;
    let c_i = c as i32;
    let h_i = h as i32;
    let w_i = w as i32;
    let mut args = stream.launch_builder(kernel);
    args.arg(output);
    args.arg(max_indices);
    args.arg(input);
    args.arg(&n_i);
    args.arg(&c_i);
    args.arg(&h_i);
    args.arg(&w_i);
    unsafe { args.launch(conv_grid(h / 2, w / 2, n * c))? };
    Ok(())
}

fn launch_maxpool_backward(
    stream: &CudaStream,
    kernel: &CudaFunction,
    grad_input: &mut CudaSlice<f32>,
    grad_output: &CudaSlice<f32>,
    max_indices: &CudaSlice<i32>,
    total: usize,
) -> Result<(), DriverError> {
    let total = total as i32;
    let mut args = stream.launch_builder(kernel);
    args.arg(grad_input);
    args.arg(grad_output);
    args.arg(max_indices);
    args.arg(&total);
    unsafe { args.launch(LaunchConfig::for_num_elems(total as u32))? };
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn launch_linear_forward(
    stream: &CudaStream,
    kernel: &CudaFunction,
    output: &mut CudaSlice<f32>,
    input: &CudaSlice<f32>,
    weights: &CudaSlice<f32>,
    bias: &CudaSlice<f32>,
    batch_size: usize,
    in_dim: usize,
    out_dim: usize,
) -> Result<(), DriverError> {
    let n = batch_size as i32;
    let in_dim_i = in_dim as i32;
    let out_dim_i = out_dim as i32;
    let mut args = stream.launch_builder(kernel);
    args.arg(output);
    args.arg(input);
    args.arg(weights);
    args.arg(bias);
    args.arg(&n);
    args.arg(&in_dim_i);
    args.arg(&out_dim_i);
    unsafe { args.launch(linear_grid(out_dim, batch_size))? };
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn launch_linear_backward_input(
    stream: &CudaStream,
    kernel: &CudaFunction,
    grad_input: &mut CudaSlice<f32>,
    grad_output: &CudaSlice<f32>,
    weights: &CudaSlice<f32>,
    batch_size: usize,
    in_dim: usize,
    out_dim: usize,
) -> Result<(), DriverError> {
    let n = batch_size as i32;
    let in_dim_i = in_dim as i32;
    let out_dim_i = out_dim as i32;
    let mut args = stream.launch_builder(kernel);
    args.arg(grad_input);
    args.arg(grad_output);
    args.arg(weights);
    args.arg(&n);
    args.arg(&in_dim_i);
    args.arg(&out_dim_i);
    unsafe {
        args.launch(LaunchConfig {
            grid_dim: (in_dim.div_ceil(256) as u32, batch_size as u32, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        })?
    };
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn launch_linear_backward_weight(
    stream: &CudaStream,
    kernel: &CudaFunction,
    grad_weights: &mut CudaSlice<f32>,
    grad_output: &CudaSlice<f32>,
    input: &CudaSlice<f32>,
    batch_size: usize,
    in_dim: usize,
    out_dim: usize,
) -> Result<(), DriverError> {
    let n = batch_size as i32;
    let in_dim_i = in_dim as i32;
    let out_dim_i = out_dim as i32;
    let mut args = stream.launch_builder(kernel);
    args.arg(grad_weights);
    args.arg(grad_output);
    args.arg(input);
    args.arg(&n);
    args.arg(&in_dim_i);
    args.arg(&out_dim_i);
    unsafe { args.launch(LaunchConfig::for_num_elems((out_dim * in_dim) as u32))? };
    Ok(())
}

fn launch_linear_backward_bias(
    stream: &CudaStream,
    kernel: &CudaFunction,
    grad_bias: &mut CudaSlice<f32>,
    grad_output: &CudaSlice<f32>,
    batch_size: usize,
    out_dim: usize,
) -> Result<(), DriverError> {
    let n = batch_size as i32;
    let out_dim_i = out_dim as i32;
    let mut args = stream.launch_builder(kernel);
    args.arg(grad_bias);
    args.arg(grad_output);
    args.arg(&n);
    args.arg(&out_dim_i);
    unsafe { args.launch(LaunchConfig::for_num_elems(out_dim as u32))? };
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn launch_softmax_forward(
    stream: &CudaStream,
    kernel: &CudaFunction,
    probs: &mut CudaSlice<f32>,
    losses: &mut CudaSlice<f32>,
    logits: &CudaSlice<f32>,
    targets: &CudaSlice<i32>,
    batch_size: usize,
    classes: usize,
) -> Result<(), DriverError> {
    let n = batch_size as i32;
    let classes_i = classes as i32;
    let mut args = stream.launch_builder(kernel);
    args.arg(probs);
    args.arg(losses);
    args.arg(logits);
    args.arg(targets);
    args.arg(&n);
    args.arg(&classes_i);
    unsafe { args.launch(LaunchConfig::for_num_elems(batch_size as u32))? };
    Ok(())
}

fn launch_softmax_backward(
    stream: &CudaStream,
    kernel: &CudaFunction,
    grad_logits: &mut CudaSlice<f32>,
    probs: &CudaSlice<f32>,
    targets: &CudaSlice<i32>,
    batch_size: usize,
    classes: usize,
) -> Result<(), DriverError> {
    let n = batch_size as i32;
    let classes_i = classes as i32;
    let mut args = stream.launch_builder(kernel);
    args.arg(grad_logits);
    args.arg(probs);
    args.arg(targets);
    args.arg(&n);
    args.arg(&classes_i);
    unsafe { args.launch(LaunchConfig::for_num_elems((batch_size * classes) as u32))? };
    Ok(())
}

fn launch_zero(
    stream: &CudaStream,
    kernel: &CudaFunction,
    buffer: &mut CudaSlice<f32>,
    len: usize,
) -> Result<(), DriverError> {
    let len = len as i32;
    let mut args = stream.launch_builder(kernel);
    args.arg(buffer);
    args.arg(&len);
    unsafe { args.launch(LaunchConfig::for_num_elems(len as u32))? };
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn launch_sgd(
    stream: &CudaStream,
    kernel: &CudaFunction,
    param: &mut CudaSlice<f32>,
    grad: &CudaSlice<f32>,
    velocity: &mut CudaSlice<f32>,
    lr: f32,
    momentum: f32,
    len: usize,
) -> Result<(), DriverError> {
    let len = len as i32;
    let mut args = stream.launch_builder(kernel);
    args.arg(param);
    args.arg(grad);
    args.arg(velocity);
    args.arg(&lr);
    args.arg(&momentum);
    args.arg(&len);
    unsafe { args.launch(LaunchConfig::for_num_elems(len as u32))? };
    Ok(())
}
