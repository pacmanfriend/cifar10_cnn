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
