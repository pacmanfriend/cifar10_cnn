use crate::tensor;

pub struct ReLU {
    last_input: Option<tensor::Tensor>,
}

impl ReLU {
    pub fn new() -> Self {
        ReLU { last_input: None }
    }

    pub fn forward(&mut self, input: &tensor::Tensor) -> tensor::Tensor {
        let mut output = input.clone();
        for v in output.data.iter_mut() {
            if *v < 0.0 {
                *v = 0.0;
            }
        }

        self.last_input = Some(input.clone());
        output
    }

    pub fn backward(&mut self, grad_output: &tensor::Tensor) -> tensor::Tensor {
        let input = self.last_input.as_ref().unwrap();
        let mut grad_input = grad_output.clone();

        for i in 0..grad_input.data.len() {
            if input.data[i] <= 0.0 {
                grad_input.data[i] = 0.0;
            }
        }
        grad_input
    }
}
