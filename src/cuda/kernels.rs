pub const NAMES: [&str; 12] = [
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

pub const SOURCE: &str = include_str!("kernels.cu");
