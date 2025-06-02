pub const NAMES: [&str; 16] = [
    "conv2d_forward",
    "conv2d_backward_input",
    "conv2d_backward_weight",
    "conv2d_backward_bias",
    "relu_forward",
    "relu_backward",
    "maxpool2x2_forward",
    "maxpool2x2_backward",
    "linear_forward",
    "linear_backward_input",
    "linear_backward_weight",
    "linear_backward_bias",
    "softmax_ce_forward",
    "softmax_ce_backward",
    "sgd_update",
    "zero_buffer",
];

pub const SOURCE: &str = include_str!("kernels.cu");

#[cfg(test)]
mod tests {
    use super::{NAMES, SOURCE};

    #[test]
    fn kernel_names_are_declared_in_source() {
        for name in NAMES {
            let declaration = format!("void {name}(");
            assert!(
                SOURCE.contains(&declaration),
                "kernel source is missing declaration for {name}"
            );
        }
    }

    #[test]
    fn cuda_source_contains_batched_padding_contracts() {
        assert!(SOURCE.contains("int N,"));
        assert!(SOURCE.contains("int pad"));
        assert!(SOURCE.contains("blockIdx.z"));
        assert!(SOURCE.contains("atomicAdd(&grad_input[max_indices[idx]], grad_output[idx])"));
        assert!(SOURCE.contains("grad_logits[idx] = (probs[idx] - one_hot) / (float)N"));
    }
}
