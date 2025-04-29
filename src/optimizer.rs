use crate::tensor;

pub struct ParamGrad<'a> {
    param: &'a mut tensor::Tensor,
    grad: &'a tensor::Tensor,
}

impl<'a> ParamGrad<'a> {
    pub fn new(param: &'a mut tensor::Tensor, grad: &'a tensor::Tensor) -> Self {
        Self { param, grad }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Sgd {
    lr: f32,
}

impl Sgd {
    pub fn new(lr: f32) -> Self {
        Self { lr }
    }

    pub fn step<'a>(&self, params: impl IntoIterator<Item = ParamGrad<'a>>) {
        for param_grad in params {
            self.update(param_grad.param, param_grad.grad);
        }
    }

    fn update(&self, param: &mut tensor::Tensor, grad: &tensor::Tensor) {
        assert_eq!(param.shape, grad.shape);
        assert_eq!(param.data.len(), grad.data.len());

        for (value, grad) in param.data.iter_mut().zip(grad.data.iter()) {
            *value -= self.lr * grad;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{ParamGrad, Sgd};
    use crate::tensor::Tensor;

    #[test]
    fn sgd_updates_parameter_tensor() {
        let mut param = Tensor::from_data(vec![1.0, -2.0], vec![2]);
        let grad = Tensor::from_data(vec![0.5, -0.25], vec![2]);

        Sgd::new(0.1).step([ParamGrad::new(&mut param, &grad)]);

        assert!((param.data[0] - 0.95).abs() < 1e-6);
        assert!((param.data[1] - -1.975).abs() < 1e-6);
    }
}
