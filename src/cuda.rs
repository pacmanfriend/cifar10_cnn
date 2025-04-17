use crate::random;
use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, DriverError, LaunchConfig,
    PushKernelArg,
};
use cudarc::nvrtc::compile_ptx;
use std::{error::Error, sync::Arc};

const C_IN: usize = 1;
const H: usize = 8;
const W: usize = 8;
const K: usize = 3;
const C_OUT: usize = 4;
const H_OUT: usize = H - K + 1;
const W_OUT: usize = W - K + 1;
const CONV_DIM: usize = C_OUT * H_OUT * W_OUT;
const POOL_H: usize = H_OUT / 2;
const POOL_W: usize = W_OUT / 2;
const FLAT_DIM: usize = C_OUT * POOL_H * POOL_W;
const N_CLASSES: usize = 3;

const KERNEL_NAMES: [&str; 12] = [
    "conv2d_forward",
    "conv2d_backward_weight",
    "conv2d_backward_bias",
    "relu_forward",
    "relu_backward",
    "maxpool2x2_forward",
    "maxpool2x2_backward",
    "dense_forward",
    "dense_backward_input",
    "dense_backward_weight",
    "softmax_and_grad",
    "sgd_update",
];

const KERNELS_SRC: &str = r#"
extern "C" __global__ void conv2d_forward(
    float* output,
    const float* input,
    const float* weights,
    const float* bias,
    int C_in, int H, int W, int C_out, int K
) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int co = blockIdx.z;

    int H_out = H - K + 1;
    int W_out = W - K + 1;
    if (i >= H_out || j >= W_out || co >= C_out) return;

    float sum = bias[co];
    for (int ci = 0; ci < C_in; ++ci) {
        for (int ki = 0; ki < K; ++ki) {
            for (int kj = 0; kj < K; ++kj) {
                int in_idx = ci * H * W + (i + ki) * W + (j + kj);
                int w_idx = ((co * C_in + ci) * K + ki) * K + kj;
                sum += input[in_idx] * weights[w_idx];
            }
        }
    }
    output[co * H_out * W_out + i * W_out + j] = sum;
}

extern "C" __global__ void conv2d_backward_weight(
    float* grad_weights,
    const float* grad_output,
    const float* input,
    int C_in, int H, int W, int C_out, int K
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = C_out * C_in * K * K;
    if (idx >= total) return;

    int co = idx / (C_in * K * K);
    int rem = idx - co * (C_in * K * K);
    int ci = rem / (K * K);
    int rem2 = rem - ci * (K * K);
    int ki = rem2 / K;
    int kj = rem2 - ki * K;

    int H_out = H - K + 1;
    int W_out = W - K + 1;

    float sum = 0.0f;
    for (int i = 0; i < H_out; ++i) {
        for (int j = 0; j < W_out; ++j) {
            int g_idx = co * H_out * W_out + i * W_out + j;
            int in_idx = ci * H * W + (i + ki) * W + (j + kj);
            sum += grad_output[g_idx] * input[in_idx];
        }
    }
    grad_weights[idx] = sum;
}

extern "C" __global__ void conv2d_backward_bias(
    float* grad_bias,
    const float* grad_output,
    int C_out, int H_out, int W_out
) {
    int co = blockIdx.x * blockDim.x + threadIdx.x;
    if (co >= C_out) return;

    float sum = 0.0f;
    int n = H_out * W_out;
    for (int k = 0; k < n; ++k) {
        sum += grad_output[co * n + k];
    }
    grad_bias[co] = sum;
}

extern "C" __global__ void relu_forward(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (x[i] < 0.0f) x[i] = 0.0f;
}

extern "C" __global__ void relu_backward(float* grad, const float* post_act, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (post_act[i] <= 0.0f) grad[i] = 0.0f;
}

extern "C" __global__ void maxpool2x2_forward(
    float* output,
    int* max_indices,
    const float* input,
    int C, int H, int W
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int H_out = H / 2;
    int W_out = W / 2;
    int total = C * H_out * W_out;
    if (idx >= total) return;

    int ch = idx / (H_out * W_out);
    int rem = idx - ch * H_out * W_out;
    int i = rem / W_out;
    int j = rem - i * W_out;

    float best_val = -3.4028234663852886e38f;
    int best_idx = 0;
    for (int di = 0; di < 2; ++di) {
        for (int dj = 0; dj < 2; ++dj) {
            int in_idx = ch * H * W + (2 * i + di) * W + (2 * j + dj);
            float v = input[in_idx];
            if (v > best_val) {
                best_val = v;
                best_idx = in_idx;
            }
        }
    }

    output[idx] = best_val;
    max_indices[idx] = best_idx;
}

extern "C" __global__ void maxpool2x2_backward(
    float* grad_input,
    const float* grad_output,
    const int* max_indices,
    int C, int H, int W
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = C * H * W;
    if (idx >= total) return;

    int H_out = H / 2;
    int W_out = W / 2;
    int ch = idx / (H * W);
    int rem = idx - ch * H * W;
    int i = rem / W;
    int j = rem - i * W;
    int out_idx = ch * H_out * W_out + (i / 2) * W_out + (j / 2);

    grad_input[idx] = max_indices[out_idx] == idx ? grad_output[out_idx] : 0.0f;
}

extern "C" __global__ void dense_forward(
    float* y,
    const float* W,
    const float* x,
    const float* b,
    int in_dim, int out_dim
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= out_dim) return;
    float sum = b[i];
    for (int j = 0; j < in_dim; ++j) {
        sum += W[i * in_dim + j] * x[j];
    }
    y[i] = sum;
}

extern "C" __global__ void dense_backward_input(
    float* dx,
    const float* W,
    const float* dy,
    int in_dim, int out_dim
) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= in_dim) return;
    float sum = 0.0f;
    for (int i = 0; i < out_dim; ++i) {
        sum += W[i * in_dim + j] * dy[i];
    }
    dx[j] = sum;
}

extern "C" __global__ void dense_backward_weight(
    float* dW,
    float* db,
    const float* dy,
    const float* x,
    int in_dim, int out_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = in_dim * out_dim;
    if (idx >= total) return;

    int i = idx / in_dim;
    int j = idx - i * in_dim;
    dW[idx] = dy[i] * x[j];
    if (j == 0) db[i] = dy[i];
}

extern "C" __global__ void softmax_and_grad(
    float* probs,
    float* grad,
    const float* logits,
    int target,
    int n
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    float mx = logits[0];
    for (int i = 1; i < n; ++i) {
        if (logits[i] > mx) mx = logits[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        float e = __expf(logits[i] - mx);
        probs[i] = e;
        sum += e;
    }
    float inv = 1.0f / sum;
    for (int i = 0; i < n; ++i) {
        probs[i] *= inv;
        grad[i] = probs[i] - (i == target ? 1.0f : 0.0f);
    }
}

extern "C" __global__ void sgd_update(float* param, const float* grad, float lr, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    param[i] -= lr * grad[i];
}
"#;

pub struct CudaNetwork {
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
    pub fn new(rng: &mut random::Rng) -> Result<Self, Box<dyn Error>> {
        let ctx = CudaContext::new(0)?;
        let stream = ctx.default_stream();
        println!("GPU 0: {}", ctx.name()?);

        let ptx = compile_ptx(KERNELS_SRC)?;
        let module = ctx.load_module(ptx)?;
        println!("Compiled {} CUDA kernels with NVRTC", KERNEL_NAMES.len());

        let conv_w_scale = (2.0 / (C_IN * K * K) as f32).sqrt();
        let conv_w_host: Vec<f32> = (0..C_OUT * C_IN * K * K)
            .map(|_| rng.normal() * conv_w_scale)
            .collect();
        let conv_b_host = vec![0.0_f32; C_OUT];

        let dense_w_scale = (2.0 / FLAT_DIM as f32).sqrt();
        let dense_w_host: Vec<f32> = (0..N_CLASSES * FLAT_DIM)
            .map(|_| rng.normal() * dense_w_scale)
            .collect();
        let dense_b_host = vec![0.0_f32; N_CLASSES];

        Ok(Self {
            conv_w: stream.memcpy_stod(&conv_w_host)?,
            conv_b: stream.memcpy_stod(&conv_b_host)?,
            dense_w: stream.memcpy_stod(&dense_w_host)?,
            dense_b: stream.memcpy_stod(&dense_b_host)?,
            grad_conv_w: stream.alloc_zeros::<f32>(C_OUT * C_IN * K * K)?,
            grad_conv_b: stream.alloc_zeros::<f32>(C_OUT)?,
            grad_dense_w: stream.alloc_zeros::<f32>(N_CLASSES * FLAT_DIM)?,
            grad_dense_b: stream.alloc_zeros::<f32>(N_CLASSES)?,
            act: stream.alloc_zeros::<f32>(CONV_DIM)?,
            act_grad: stream.alloc_zeros::<f32>(CONV_DIM)?,
            pooled: stream.alloc_zeros::<f32>(FLAT_DIM)?,
            pooled_grad: stream.alloc_zeros::<f32>(FLAT_DIM)?,
            max_indices: stream.alloc_zeros::<i32>(FLAT_DIM)?,
            logits: stream.alloc_zeros::<f32>(N_CLASSES)?,
            probs: stream.alloc_zeros::<f32>(N_CLASSES)?,
            grad_logits: stream.alloc_zeros::<f32>(N_CLASSES)?,
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
        let input = self.stream.memcpy_stod(input_host)?;

        let cfg_conv_fwd = LaunchConfig {
            grid_dim: (
                W_OUT.div_ceil(8) as u32,
                H_OUT.div_ceil(8) as u32,
                C_OUT as u32,
            ),
            block_dim: (8, 8, 1),
            shared_mem_bytes: 0,
        };
        let cfg_act = LaunchConfig::for_num_elems(CONV_DIM as u32);
        let cfg_pool = LaunchConfig::for_num_elems(FLAT_DIM as u32);
        let cfg_dense_out = LaunchConfig::for_num_elems(N_CLASSES as u32);
        let cfg_dense_in = LaunchConfig::for_num_elems(FLAT_DIM as u32);
        let cfg_dense_w = LaunchConfig::for_num_elems((N_CLASSES * FLAT_DIM) as u32);
        let cfg_conv_w = LaunchConfig::for_num_elems((C_OUT * C_IN * K * K) as u32);
        let cfg_conv_b = LaunchConfig::for_num_elems(C_OUT as u32);
        let cfg_softmax = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };

        let c_in = C_IN as i32;
        let h = H as i32;
        let w = W as i32;
        let c_out = C_OUT as i32;
        let k = K as i32;
        let conv_dim = CONV_DIM as i32;
        let conv_h = H_OUT as i32;
        let conv_w = W_OUT as i32;
        let flat_dim = FLAT_DIM as i32;
        let n_classes = N_CLASSES as i32;
        let target = target as i32;
        let h_out = H_OUT as i32;
        let w_out = W_OUT as i32;
        let conv_w_len = (C_OUT * C_IN * K * K) as i32;
        let dense_w_len = (N_CLASSES * FLAT_DIM) as i32;

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
