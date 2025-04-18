use crate::{config, conv, cuda, dense, loss, maxpool, random, relu, tensor};
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
    config: config::ModelConfig,
}

impl Network {
    pub fn new(
        config: config::ModelConfig,
        rng: &mut random::Rng,
        backend: Backend,
    ) -> Result<Self, Box<dyn Error>> {
        let inner = match backend {
            Backend::Cpu => NetworkInner::Cpu(Box::new(CpuNetwork::new(config, rng))),
            Backend::Gpu => NetworkInner::Gpu(Box::new(cuda::CudaNetwork::new(config, rng)?)),
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
    fn new(config: config::ModelConfig, rng: &mut random::Rng) -> Self {
        CpuNetwork {
            conv: conv::Conv2D::new(
                config.input_channels,
                config.conv_out_channels,
                config.conv_kernel,
                rng,
            ),
            relu: relu::ReLU::new(),
            pool: maxpool::MaxPool2x2::new(),
            fc: dense::Dense::new(config.flat_dim(), config.num_classes, rng),
            config,
        }
    }

    fn train_step(&mut self, input: &tensor::Tensor, target: usize, lr: f32) -> (f32, usize) {
        debug_assert_eq!(input.numel(), self.config.input_dim());
        debug_assert!(target < self.config.num_classes);

        let x = self.conv.forward(input);
        let x = self.relu.forward(&x);
        let x = self.pool.forward(&x);

        let flat = x.reshape(vec![self.config.flat_dim()]);

        let logits = self.fc.forward(&flat);
        let probs = loss::softmax(&logits);

        let loss = loss::cross_entropy(&probs, target);
        let predicted = loss::argmax(&probs);

        let mut grad_logits = probs.clone();
        grad_logits.data[target] -= 1.0;

        let grad_flat = self.fc.backward(&grad_logits);

        let grad_pool_in = grad_flat.reshape(vec![
            self.config.conv_out_channels,
            self.config.pool_height(),
            self.config.pool_width(),
        ]);

        let grad_relu_in = self.pool.backward(&grad_pool_in);
        let grad_conv_in = self.relu.backward(&grad_relu_in);
        let _ = self.conv.backward(&grad_conv_in);

        // Step SGD.
        self.conv.step(lr);
        self.fc.step(lr);

        (loss, predicted)
    }
}
