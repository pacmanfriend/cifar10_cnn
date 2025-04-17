use crate::{conv, cuda, dense, maxpool, random, relu, tensor};
use std::error::Error;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Backend {
    Cpu,
    Gpu,
}

pub struct Network {
    inner: NetworkInner,
}

enum NetworkInner {
    Cpu(Box<CpuNetwork>),
    Gpu(Box<cuda::CudaNetwork>),
}

struct CpuNetwork {
    conv: conv::Conv2D,
    relu: relu::ReLU,
    pool: maxpool::MaxPool2x2,
    fc: dense::Dense,
}

impl Network {
    pub fn new(rng: &mut random::Rng, backend: Backend) -> Result<Self, Box<dyn Error>> {
        let inner = match backend {
            Backend::Cpu => NetworkInner::Cpu(Box::new(CpuNetwork::new(rng))),
            Backend::Gpu => NetworkInner::Gpu(Box::new(cuda::CudaNetwork::new(rng)?)),
        };

        Ok(Network { inner })
    }

    pub fn train_step(
        &mut self,
        input: &tensor::Tensor,
        target: usize,
        lr: f32,
    ) -> Result<(f32, usize), Box<dyn Error>> {
        match &mut self.inner {
            NetworkInner::Cpu(net) => Ok(net.train_step(input, target, lr)),
            NetworkInner::Gpu(net) => Ok(net.train_step(&input.data, target, lr)?),
        }
    }
}

impl CpuNetwork {
    fn new(rng: &mut random::Rng) -> Self {
        CpuNetwork {
            conv: conv::Conv2D::new(1, 4, 3, rng),
            relu: relu::ReLU::new(),
            pool: maxpool::MaxPool2x2::new(),
            fc: dense::Dense::new(4 * 3 * 3, 3, rng),
        }
    }

    fn train_step(&mut self, input: &tensor::Tensor, target: usize, lr: f32) -> (f32, usize) {
        let x = self.conv.forward(input);
        let x = self.relu.forward(&x);
        let x = self.pool.forward(&x);

        let flat = tensor::Tensor {
            data: x.data,
            shape: vec![4 * 3 * 3],
        };

        let logits = self.fc.forward(&flat);
        let probs = softmax(&logits);

        let loss = cross_entropy_loss(&probs, target);
        let predicted = argmax(&probs);

        let mut grad_logits = probs.clone();
        grad_logits.data[target] -= 1.0;

        let grad_flat = self.fc.backward(&grad_logits);

        let grad_pool_in = tensor::Tensor {
            data: grad_flat.data,
            shape: vec![4, 3, 3],
        };

        let grad_relu_in = self.pool.backward(&grad_pool_in);
        let grad_conv_in = self.relu.backward(&grad_relu_in);
        let _ = self.conv.backward(&grad_conv_in);

        // Step SGD.
        self.conv.step(lr);
        self.fc.step(lr);

        (loss, predicted)
    }
}

fn softmax(logits: &tensor::Tensor) -> tensor::Tensor {
    let max_val = logits
        .data
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);

    let exps: Vec<f32> = logits.data.iter().map(|x| (x - max_val).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let data = exps.iter().map(|e| e / sum).collect();

    tensor::Tensor {
        data,
        shape: logits.shape.clone(),
    }
}

fn cross_entropy_loss(probs: &tensor::Tensor, target: usize) -> f32 {
    -(probs.data[target].max(1e-12)).ln()
}

fn argmax(t: &tensor::Tensor) -> usize {
    t.data
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0
}
