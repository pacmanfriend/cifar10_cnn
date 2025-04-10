use crate::tensor;

pub struct MaxPool2x2 {
    max_indices: Vec<usize>,
    input_shape: Vec<usize>,
}

impl MaxPool2x2 {
    pub fn new() -> Self {
        MaxPool2x2 {
            max_indices: vec![],
            input_shape: vec![],
        }
    }

    pub fn forward(&mut self, input: &tensor::Tensor) -> tensor::Tensor {
        let c = input.shape[0];
        let h = input.shape[1];
        let w = input.shape[2];
        let h_out = h / 2;
        let w_out = w / 2;

        let mut output = tensor::Tensor::zeros(vec![c, h_out, w_out]);
        self.max_indices = vec![0; c * h_out * w_out];
        self.input_shape = input.shape.clone();

        for ch in 0..c {
            for i in 0..h_out {
                for j in 0..w_out {
                    let mut best_val = f32::NEG_INFINITY;
                    let mut best_idx = 0;

                    for di in 0..2 {
                        for dj in 0..2 {
                            let idx = ch * h * w + (2 * i + di) * w + (2 * j + dj);
                            if input.data[idx] > best_val {
                                best_val = input.data[idx];
                                best_idx = idx;
                            }
                        }
                    }

                    let out_idx = ch * h_out * w_out + i * w_out + j;

                    output.data[out_idx] = best_val;
                    self.max_indices[out_idx] = best_idx;
                }
            }
        }

        output
    }

    pub fn backward(&self, grad_output: &tensor::Tensor) -> tensor::Tensor {
        let mut grad_input = tensor::Tensor::zeros(self.input_shape.clone());
        // Градиент течёт только в ту позицию, которая дала максимум.
        for (out_idx, &in_idx) in self.max_indices.iter().enumerate() {
            grad_input.data[in_idx] += grad_output.data[out_idx];
        }
        grad_input
    }
}
