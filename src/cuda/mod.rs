use crate::{compute::random, config};
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
    conv_w: CudaSlice<f32>,
    conv_b: CudaSlice<f32>,
    dense_w: CudaSlice<f32>,
    dense_b: CudaSlice<f32>,
    grad_conv_w: CudaSlice<f32>,
    grad_conv_b: CudaSlice<f32>,
    grad_dense_w: CudaSlice<f32>,
    grad_dense_b: CudaSlice<f32>,
    act: CudaSlice<f32>,
    act_grad: CudaSlice<f32>,
    pooled: CudaSlice<f32>,
    pooled_grad: CudaSlice<f32>,
    max_indices: CudaSlice<i32>,
    logits: CudaSlice<f32>,
    probs: CudaSlice<f32>,
    grad_logits: CudaSlice<f32>,
}

impl CudaNetwork {
    pub fn new(config: config::ModelConfig, rng: &mut random::Rng) -> Result<Self, Box<dyn Error>> {
        let ctx = CudaContext::new(0)?;
        let stream = ctx.default_stream();
        println!("GPU 0: {}", ctx.name()?);

        let ptx = compile_ptx(kernels::SOURCE)?;
        let module = ctx.load_module(ptx)?;
        println!("Compiled {} CUDA kernels with NVRTC", kernels::NAMES.len());

        let conv_w_scale =
            (2.0 / (config.input_channels * config.conv_kernel * config.conv_kernel) as f32).sqrt();
        let conv_w_host: Vec<f32> = (0..config.conv_weight_len())
            .map(|_| rng.normal() * conv_w_scale)
            .collect();
        let conv_b_host = vec![0.0_f32; config.conv_out_channels];

        let dense_w_scale = (2.0 / config.flat_dim() as f32).sqrt();
        let dense_w_host: Vec<f32> = (0..config.num_classes * config.flat_dim())
            .map(|_| rng.normal() * dense_w_scale)
            .collect();
        let dense_b_host = vec![0.0_f32; config.num_classes];

        Ok(Self {
            config,
            conv_w: stream.memcpy_stod(&conv_w_host)?,
            conv_b: stream.memcpy_stod(&conv_b_host)?,
            dense_w: stream.memcpy_stod(&dense_w_host)?,
            dense_b: stream.memcpy_stod(&dense_b_host)?,
            grad_conv_w: stream.alloc_zeros::<f32>(config.conv_weight_len())?,
            grad_conv_b: stream.alloc_zeros::<f32>(config.conv_out_channels)?,
            grad_dense_w: stream.alloc_zeros::<f32>(config.num_classes * config.flat_dim())?,
            grad_dense_b: stream.alloc_zeros::<f32>(config.num_classes)?,
            act: stream.alloc_zeros::<f32>(config.conv_dim())?,
            act_grad: stream.alloc_zeros::<f32>(config.conv_dim())?,
            pooled: stream.alloc_zeros::<f32>(config.flat_dim())?,
            pooled_grad: stream.alloc_zeros::<f32>(config.flat_dim())?,
            max_indices: stream.alloc_zeros::<i32>(config.flat_dim())?,
            logits: stream.alloc_zeros::<f32>(config.num_classes)?,
            probs: stream.alloc_zeros::<f32>(config.num_classes)?,
            grad_logits: stream.alloc_zeros::<f32>(config.num_classes)?,
            stream,
            module,
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
        let target_index = target;
        debug_assert_eq!(input_host.len(), self.config.input_dim());
        debug_assert!(target < self.config.num_classes);

        let input = self.stream.memcpy_stod(input_host)?;
        let config = self.config;

        let cfg_conv_fwd = LaunchConfig {
            grid_dim: (
                config.conv_width().div_ceil(8) as u32,
                config.conv_height().div_ceil(8) as u32,
                config.conv_out_channels as u32,
            ),
            block_dim: (8, 8, 1),
            shared_mem_bytes: 0,
        };
        let cfg_act = LaunchConfig::for_num_elems(config.conv_dim() as u32);
        let cfg_pool = LaunchConfig::for_num_elems(config.flat_dim() as u32);
        let cfg_dense_out = LaunchConfig::for_num_elems(config.num_classes as u32);
        let cfg_dense_in = LaunchConfig::for_num_elems(config.flat_dim() as u32);
        let cfg_dense_w =
            LaunchConfig::for_num_elems((config.num_classes * config.flat_dim()) as u32);
        let cfg_conv_w = LaunchConfig::for_num_elems(config.conv_weight_len() as u32);
        let cfg_conv_b = LaunchConfig::for_num_elems(config.conv_out_channels as u32);
        let cfg_softmax = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };

        let c_in = config.input_channels as i32;
        let h = config.input_height as i32;
        let w = config.input_width as i32;
        let c_out = config.conv_out_channels as i32;
        let k = config.conv_kernel as i32;
        let conv_dim = config.conv_dim() as i32;
        let conv_h = config.conv_height() as i32;
        let conv_w = config.conv_width() as i32;
        let flat_dim = config.flat_dim() as i32;
        let n_classes = config.num_classes as i32;
        let target = target as i32;
        let h_out = config.conv_height() as i32;
        let w_out = config.conv_width() as i32;
        let conv_w_len = config.conv_weight_len() as i32;
        let dense_w_len = (config.num_classes * config.flat_dim()) as i32;

        let conv_fwd = self.kernel("conv2d_forward")?;
        let mut args = self.stream.launch_builder(&conv_fwd);
        args.arg(&mut self.act);
        args.arg(&input);
        args.arg(&self.conv_w);
        args.arg(&self.conv_b);
        args.arg(&c_in);
        args.arg(&h);
        args.arg(&w);
        args.arg(&c_out);
        args.arg(&k);
        unsafe { args.launch(cfg_conv_fwd) }?;

        let relu_fwd = self.kernel("relu_forward")?;
        let mut args = self.stream.launch_builder(&relu_fwd);
        args.arg(&mut self.act);
        args.arg(&conv_dim);
        unsafe { args.launch(cfg_act) }?;

        let pool_fwd = self.kernel("maxpool2x2_forward")?;
        let mut args = self.stream.launch_builder(&pool_fwd);
        args.arg(&mut self.pooled);
        args.arg(&mut self.max_indices);
        args.arg(&self.act);
        args.arg(&c_out);
        args.arg(&conv_h);
        args.arg(&conv_w);
        unsafe { args.launch(cfg_pool) }?;

        let dense_fwd = self.kernel("dense_forward")?;
        let mut args = self.stream.launch_builder(&dense_fwd);
        args.arg(&mut self.logits);
        args.arg(&self.dense_w);
        args.arg(&self.pooled);
        args.arg(&self.dense_b);
        args.arg(&flat_dim);
        args.arg(&n_classes);
        unsafe { args.launch(cfg_dense_out) }?;

        let softmax = self.kernel("softmax_and_grad")?;
        let mut args = self.stream.launch_builder(&softmax);
        args.arg(&mut self.probs);
        args.arg(&mut self.grad_logits);
        args.arg(&self.logits);
        args.arg(&target);
        args.arg(&n_classes);
        unsafe { args.launch(cfg_softmax) }?;

        let probs_host = self.stream.memcpy_dtov(&self.probs)?;
        let loss = -probs_host[target_index].max(1e-12).ln();
        let predicted = probs_host
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;

        let dense_bwd_in = self.kernel("dense_backward_input")?;
        let mut args = self.stream.launch_builder(&dense_bwd_in);
        args.arg(&mut self.pooled_grad);
        args.arg(&self.dense_w);
        args.arg(&self.grad_logits);
        args.arg(&flat_dim);
        args.arg(&n_classes);
        unsafe { args.launch(cfg_dense_in) }?;

        let dense_bwd_w = self.kernel("dense_backward_weight")?;
        let mut args = self.stream.launch_builder(&dense_bwd_w);
        args.arg(&mut self.grad_dense_w);
        args.arg(&mut self.grad_dense_b);
        args.arg(&self.grad_logits);
        args.arg(&self.pooled);
        args.arg(&flat_dim);
        args.arg(&n_classes);
        unsafe { args.launch(cfg_dense_w) }?;

        let pool_bwd = self.kernel("maxpool2x2_backward")?;
        let mut args = self.stream.launch_builder(&pool_bwd);
        args.arg(&mut self.act_grad);
        args.arg(&self.pooled_grad);
        args.arg(&self.max_indices);
        args.arg(&c_out);
        args.arg(&conv_h);
        args.arg(&conv_w);
        unsafe { args.launch(cfg_act) }?;

        let relu_bwd = self.kernel("relu_backward")?;
        let mut args = self.stream.launch_builder(&relu_bwd);
        args.arg(&mut self.act_grad);
        args.arg(&self.act);
        args.arg(&conv_dim);
        unsafe { args.launch(cfg_act) }?;

        let conv_bwd_w = self.kernel("conv2d_backward_weight")?;
        let mut args = self.stream.launch_builder(&conv_bwd_w);
        args.arg(&mut self.grad_conv_w);
        args.arg(&self.act_grad);
        args.arg(&input);
        args.arg(&c_in);
        args.arg(&h);
        args.arg(&w);
        args.arg(&c_out);
        args.arg(&k);
        unsafe { args.launch(cfg_conv_w) }?;

        let conv_bwd_b = self.kernel("conv2d_backward_bias")?;
        let mut args = self.stream.launch_builder(&conv_bwd_b);
        args.arg(&mut self.grad_conv_b);
        args.arg(&self.act_grad);
        args.arg(&c_out);
        args.arg(&h_out);
        args.arg(&w_out);
        unsafe { args.launch(cfg_conv_b) }?;

        let sgd = self.kernel("sgd_update")?;
        let mut args = self.stream.launch_builder(&sgd);
        args.arg(&mut self.conv_w);
        args.arg(&self.grad_conv_w);
        args.arg(&lr);
        args.arg(&conv_w_len);
        unsafe { args.launch(cfg_conv_w) }?;

        let mut args = self.stream.launch_builder(&sgd);
        args.arg(&mut self.conv_b);
        args.arg(&self.grad_conv_b);
        args.arg(&lr);
        args.arg(&c_out);
        unsafe { args.launch(cfg_conv_b) }?;

        let mut args = self.stream.launch_builder(&sgd);
        args.arg(&mut self.dense_w);
        args.arg(&self.grad_dense_w);
        args.arg(&lr);
        args.arg(&dense_w_len);
        unsafe { args.launch(cfg_dense_w) }?;

        let mut args = self.stream.launch_builder(&sgd);
        args.arg(&mut self.dense_b);
        args.arg(&self.grad_dense_b);
        args.arg(&lr);
        args.arg(&n_classes);
        unsafe { args.launch(cfg_dense_out) }?;

        Ok((loss, predicted))
    }
}
