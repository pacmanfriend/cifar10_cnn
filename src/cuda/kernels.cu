extern "C" __global__ void conv2d_forward(
    float* output,
    const float* input,
    const float* weights,
    const float* bias,
    int N,
    int C_in,
    int H,
    int W,
    int C_out,
    int K,
    int pad
) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int n_co = blockIdx.z;
    int n = n_co / C_out;
    int co = n_co - n * C_out;

    int H_out = H + 2 * pad - K + 1;
    int W_out = W + 2 * pad - K + 1;
    if (i >= H_out || j >= W_out || n >= N) return;

    float sum = bias[co];
    for (int ci = 0; ci < C_in; ++ci) {
        for (int ki = 0; ki < K; ++ki) {
            for (int kj = 0; kj < K; ++kj) {
                int i_in = i + ki - pad;
                int j_in = j + kj - pad;
                if (i_in < 0 || i_in >= H || j_in < 0 || j_in >= W) continue;

                int in_idx = ((n * C_in + ci) * H + i_in) * W + j_in;
                int w_idx = ((co * C_in + ci) * K + ki) * K + kj;
                sum += input[in_idx] * weights[w_idx];
            }
        }
    }

    output[((n * C_out + co) * H_out + i) * W_out + j] = sum;
}

extern "C" __global__ void conv2d_backward_input(
    float* grad_input,
    const float* grad_output,
    const float* weights,
    int N,
    int C_in,
    int H,
    int W,
    int C_out,
    int K,
    int pad
) {
    int j_in = blockIdx.x * blockDim.x + threadIdx.x;
    int i_in = blockIdx.y * blockDim.y + threadIdx.y;
    int n_ci = blockIdx.z;
    int n = n_ci / C_in;
    int ci = n_ci - n * C_in;
    if (i_in >= H || j_in >= W || n >= N) return;

    int H_out = H + 2 * pad - K + 1;
    int W_out = W + 2 * pad - K + 1;
    float sum = 0.0f;

    for (int co = 0; co < C_out; ++co) {
        for (int ki = 0; ki < K; ++ki) {
            for (int kj = 0; kj < K; ++kj) {
                int i_out = i_in - ki + pad;
                int j_out = j_in - kj + pad;
                if (i_out < 0 || i_out >= H_out || j_out < 0 || j_out >= W_out) continue;

                int g_idx = ((n * C_out + co) * H_out + i_out) * W_out + j_out;
                int w_idx = ((co * C_in + ci) * K + ki) * K + kj;
                sum += grad_output[g_idx] * weights[w_idx];
            }
        }
    }

    grad_input[((n * C_in + ci) * H + i_in) * W + j_in] = sum;
}

extern "C" __global__ void conv2d_backward_weight(
    float* grad_weights,
    const float* grad_output,
    const float* input,
    int N,
    int C_in,
    int H,
    int W,
    int C_out,
    int K,
    int pad
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

    int H_out = H + 2 * pad - K + 1;
    int W_out = W + 2 * pad - K + 1;
    float sum = 0.0f;

    for (int n = 0; n < N; ++n) {
        for (int i = 0; i < H_out; ++i) {
            for (int j = 0; j < W_out; ++j) {
                int i_in = i + ki - pad;
                int j_in = j + kj - pad;
                if (i_in < 0 || i_in >= H || j_in < 0 || j_in >= W) continue;

                int g_idx = ((n * C_out + co) * H_out + i) * W_out + j;
                int in_idx = ((n * C_in + ci) * H + i_in) * W + j_in;
                sum += grad_output[g_idx] * input[in_idx];
            }
        }
    }

    grad_weights[idx] = sum;
}

extern "C" __global__ void conv2d_backward_bias(
    float* grad_bias,
    const float* grad_output,
    int N,
    int C_out,
    int H_out,
    int W_out
) {
    int co = blockIdx.x * blockDim.x + threadIdx.x;
    if (co >= C_out) return;

    int spatial = H_out * W_out;
    float sum = 0.0f;
    for (int n = 0; n < N; ++n) {
        const float* base = grad_output + (n * C_out + co) * spatial;
        for (int k = 0; k < spatial; ++k) {
            sum += base[k];
        }
    }

    grad_bias[co] = sum;
}

extern "C" __global__ void relu_forward(float* output, const float* input, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    output[i] = fmaxf(0.0f, input[i]);
}

extern "C" __global__ void relu_backward(
    float* grad_input,
    const float* grad_output,
    const float* post_act,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    grad_input[i] = post_act[i] > 0.0f ? grad_output[i] : 0.0f;
}

extern "C" __global__ void maxpool2x2_forward(
    float* output,
    int* max_indices,
    const float* input,
    int N,
    int C,
    int H,
    int W
) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int n_c = blockIdx.z;
    int n = n_c / C;
    int c = n_c - n * C;

    int H_out = H / 2;
    int W_out = W / 2;
    if (i >= H_out || j >= W_out || n >= N) return;

    float best_val = -3.4028234663852886e38f;
    int best_idx = 0;
    for (int di = 0; di < 2; ++di) {
        for (int dj = 0; dj < 2; ++dj) {
            int in_idx = ((n * C + c) * H + 2 * i + di) * W + 2 * j + dj;
            float v = input[in_idx];
            if (v > best_val) {
                best_val = v;
                best_idx = in_idx;
            }
        }
    }

    int out_idx = ((n * C + c) * H_out + i) * W_out + j;
    output[out_idx] = best_val;
    max_indices[out_idx] = best_idx;
}

extern "C" __global__ void maxpool2x2_backward(
    float* grad_input,
    const float* grad_output,
    const int* max_indices,
    int total
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    atomicAdd(&grad_input[max_indices[idx]], grad_output[idx]);
}

extern "C" __global__ void linear_forward(
    float* y,
    const float* x,
    const float* weights,
    const float* bias,
    int N,
    int in_dim,
    int out_dim
) {
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    int n = blockIdx.y;
    if (o >= out_dim || n >= N) return;

    float sum = bias[o];
    for (int i = 0; i < in_dim; ++i) {
        sum += weights[o * in_dim + i] * x[n * in_dim + i];
    }

    y[n * out_dim + o] = sum;
}

extern "C" __global__ void linear_backward_input(
    float* dx,
    const float* dy,
    const float* weights,
    int N,
    int in_dim,
    int out_dim
) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int n = blockIdx.y;
    if (j >= in_dim || n >= N) return;

    float sum = 0.0f;
    for (int o = 0; o < out_dim; ++o) {
        sum += dy[n * out_dim + o] * weights[o * in_dim + j];
    }

    dx[n * in_dim + j] = sum;
}

extern "C" __global__ void linear_backward_weight(
    float* dW,
    const float* dy,
    const float* x,
    int N,
    int in_dim,
    int out_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = out_dim * in_dim;
    if (idx >= total) return;

    int o = idx / in_dim;
    int j = idx - o * in_dim;
    float sum = 0.0f;
    for (int n = 0; n < N; ++n) {
        sum += dy[n * out_dim + o] * x[n * in_dim + j];
    }

    dW[idx] = sum;
}

extern "C" __global__ void linear_backward_bias(
    float* db,
    const float* dy,
    int N,
    int out_dim
) {
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= out_dim) return;

    float sum = 0.0f;
    for (int n = 0; n < N; ++n) {
        sum += dy[n * out_dim + o];
    }

    db[o] = sum;
}

extern "C" __global__ void softmax_ce_forward(
    float* probs,
    float* loss_per_sample,
    const float* logits,
    const int* targets,
    int N,
    int C
) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;

    const float* row_logits = logits + n * C;
    float* row_probs = probs + n * C;
    float mx = row_logits[0];
    for (int c = 1; c < C; ++c) {
        if (row_logits[c] > mx) mx = row_logits[c];
    }

    float sum = 0.0f;
    for (int c = 0; c < C; ++c) {
        float e = __expf(row_logits[c] - mx);
        row_probs[c] = e;
        sum += e;
    }

    float inv = 1.0f / sum;
    int target = targets[n];
    for (int c = 0; c < C; ++c) {
        row_probs[c] *= inv;
    }

    loss_per_sample[n] = -__logf(row_probs[target] + 1e-12f);
}

extern "C" __global__ void softmax_ce_backward(
    float* grad_logits,
    const float* probs,
    const int* targets,
    int N,
    int C
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C;
    if (idx >= total) return;

    int n = idx / C;
    int c = idx - n * C;
    float one_hot = c == targets[n] ? 1.0f : 0.0f;
    grad_logits[idx] = (probs[idx] - one_hot) / (float)N;
}

extern "C" __global__ void sgd_update(float* param, const float* grad, float lr, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    param[i] -= lr * grad[i];
}

extern "C" __global__ void zero_buffer(float* buffer, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    buffer[i] = 0.0f;
}
