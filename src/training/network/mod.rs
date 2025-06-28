use crate::{
    compute::{random, tensor},
    config, cuda,
};
use std::error::Error;

mod cpu;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Backend {
    Cpu,
    Gpu,
}

pub struct Network {
    inner: NetworkInner,
}

enum NetworkInner {
    Cpu(Box<cpu::CpuNetwork>),
    Gpu(Box<cuda::CudaNetwork>),
}

impl Network {
    pub fn new(
        config: config::ModelConfig,
        rng: &mut random::Rng,
        backend: Backend,
    ) -> Result<Self, Box<dyn Error>> {
        let inner = match backend {
            Backend::Cpu => NetworkInner::Cpu(Box::new(cpu::CpuNetwork::new(config, rng))),
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

    pub fn train_step_batch(
        &mut self,
        input: &tensor::Tensor,
        targets: &[usize],
        lr: f32,
    ) -> Result<(f32, usize), Box<dyn Error>> {
        match &mut self.inner {
            NetworkInner::Cpu(net) => Ok(net.train_step_batch(input, targets, lr)),
            NetworkInner::Gpu(net) => Ok(net.train_step_batch(input, targets, lr)?),
        }
    }

    pub fn train_step_batch_with_momentum(
        &mut self,
        input: &tensor::Tensor,
        targets: &[usize],
        lr: f32,
        momentum: f32,
    ) -> Result<(f32, usize), Box<dyn Error>> {
        match &mut self.inner {
            NetworkInner::Cpu(net) => {
                let (loss, predictions) =
                    net.train_step_batch_with_momentum(input, targets, lr, momentum);
                let correct = predictions
                    .iter()
                    .zip(targets.iter())
                    .filter(|(predicted, target)| predicted == target)
                    .count();
                Ok((loss, correct))
            }
            NetworkInner::Gpu(net) => {
                Ok(net.train_step_batch_with_momentum(input, targets, lr, momentum)?)
            }
        }
    }

    pub fn save_weights(&self, path: &std::path::Path) -> Result<(), Box<dyn Error>> {
        match &self.inner {
            NetworkInner::Cpu(net) => net.save_weights(path),
            NetworkInner::Gpu(net) => net.save_weights(path),
        }
    }

    pub fn load_weights(&mut self, path: &std::path::Path) -> Result<(), Box<dyn Error>> {
        match &mut self.inner {
            NetworkInner::Cpu(net) => net.load_weights(path),
            NetworkInner::Gpu(net) => net.load_weights(path),
        }
    }

    pub fn predict_batch(&mut self, input: &tensor::Tensor) -> Result<Vec<usize>, Box<dyn Error>> {
        match &mut self.inner {
            NetworkInner::Cpu(net) => Ok(net.predict_batch(input)),
            NetworkInner::Gpu(net) => Ok(net.predict_batch(input)?),
        }
    }

    pub fn predict_batch_with_scores(
        &mut self,
        input: &tensor::Tensor,
    ) -> Result<(Vec<usize>, Vec<f32>), Box<dyn Error>> {
        match &mut self.inner {
            NetworkInner::Cpu(net) => Ok(net.predict_batch_with_scores(input)),
            NetworkInner::Gpu(net) => Ok(net.predict_batch_with_scores(input)?),
        }
    }

    #[cfg(test)]
    fn train_step_batch_with_predictions(
        &mut self,
        input: &tensor::Tensor,
        targets: &[usize],
        lr: f32,
    ) -> Result<(f32, Vec<usize>), Box<dyn Error>> {
        match &mut self.inner {
            NetworkInner::Cpu(net) => Ok(net.train_step_batch_with_predictions(input, targets, lr)),
            NetworkInner::Gpu(net) => {
                Ok(net.train_step_batch_with_predictions(input, targets, lr)?)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Backend, Network};
    use crate::{compute::random, config, data::datasets};

    #[test]
    fn save_and_load_weights_preserves_predictions() -> Result<(), Box<dyn std::error::Error>> {
        let config = config::ModelConfig::demo();
        let dataset = datasets::make_fake_dataset();
        let input = crate::compute::tensor::Tensor::from_data(
            dataset.iter().take(2).flat_map(|(x, _)| x.data.iter().copied()).collect(),
            vec![2, 1, 8, 8],
        );
        let targets: Vec<usize> = dataset.iter().take(2).map(|(_, t)| *t).collect();

        let mut rng = random::Rng::new(99);
        let mut net = Network::new(config, &mut rng, Backend::Cpu)?;
        for _ in 0..5 {
            net.train_step_batch(&input, &targets, 0.05)?;
        }

        let path = std::env::temp_dir().join("cifar10_net_test.ck10");
        net.save_weights(&path)?;

        let mut rng2 = random::Rng::new(0);
        let mut fresh = Network::new(config, &mut rng2, Backend::Cpu)?;
        fresh.load_weights(&path)?;

        let original = net.predict_batch(&input)?;
        let restored = fresh.predict_batch(&input)?;
        assert_eq!(original, restored);

        Ok(())
    }

    #[test]
    fn cpu_train_step_is_deterministic_for_fixed_seed() -> Result<(), Box<dyn std::error::Error>> {
        let config = config::ModelConfig::demo();
        let dataset = datasets::make_fake_dataset();
        let mut first_rng = random::Rng::new(7);
        let mut second_rng = random::Rng::new(7);
        let mut first = Network::new(config, &mut first_rng, Backend::Cpu)?;
        let mut second = Network::new(config, &mut second_rng, Backend::Cpu)?;

        for (input, target) in dataset.iter() {
            let (first_loss, first_predicted) = first.train_step(input, *target, 0.05)?;
            let (second_loss, second_predicted) = second.train_step(input, *target, 0.05)?;

            assert_eq!(first_predicted, second_predicted);
            assert!(
                (first_loss - second_loss).abs() <= f32::EPSILON,
                "loss mismatch: {first_loss} != {second_loss}"
            );
        }

        Ok(())
    }

    #[test]
    fn cpu_train_step_batch_is_deterministic_for_fixed_seed(
    ) -> Result<(), Box<dyn std::error::Error>> {
        let config = config::ModelConfig::demo();
        let dataset = datasets::make_fake_dataset();
        let input = crate::compute::tensor::Tensor::from_data(
            dataset
                .iter()
                .take(2)
                .flat_map(|(input, _)| input.data.iter().copied())
                .collect(),
            vec![2, 1, 8, 8],
        );
        let targets: Vec<usize> = dataset.iter().take(2).map(|(_, target)| *target).collect();
        let mut first_rng = random::Rng::new(7);
        let mut second_rng = random::Rng::new(7);
        let mut first = Network::new(config, &mut first_rng, Backend::Cpu)?;
        let mut second = Network::new(config, &mut second_rng, Backend::Cpu)?;

        let (first_loss, first_correct) = first.train_step_batch(&input, &targets, 0.05)?;
        let (second_loss, second_correct) = second.train_step_batch(&input, &targets, 0.05)?;

        assert_eq!(first_correct, second_correct);
        assert!(
            (first_loss - second_loss).abs() <= f32::EPSILON,
            "loss mismatch: {first_loss} != {second_loss}"
        );

        Ok(())
    }

    #[test]
    fn cpu_cifar10_train_step_accepts_two_layer_batch() -> Result<(), Box<dyn std::error::Error>> {
        let config = config::ModelConfig::cifar10();
        let input = crate::compute::tensor::Tensor::zeros(vec![1, 3, 32, 32]);
        let targets = [0];
        let mut rng = random::Rng::new(13);
        let mut net = Network::new(config, &mut rng, Backend::Cpu)?;

        let (loss, correct) = net.train_step_batch(&input, &targets, 0.01)?;

        assert!(loss.is_finite());
        assert!(correct <= targets.len());

        Ok(())
    }

    #[test]
    #[ignore]
    fn cpu_and_gpu_train_step_match_for_fixed_seed() -> Result<(), Box<dyn std::error::Error>> {
        let config = config::ModelConfig::demo();
        let dataset = datasets::make_fake_dataset();
        let input = crate::compute::tensor::Tensor::from_data(
            dataset
                .iter()
                .take(2)
                .flat_map(|(input, _)| input.data.iter().copied())
                .collect(),
            vec![2, 1, 8, 8],
        );
        let targets: Vec<usize> = dataset.iter().take(2).map(|(_, target)| *target).collect();
        let mut cpu_rng = random::Rng::new(11);
        let mut gpu_rng = random::Rng::new(11);
        let mut cpu = Network::new(config, &mut cpu_rng, Backend::Cpu)?;
        let mut gpu = match Network::new(config, &mut gpu_rng, Backend::Gpu) {
            Ok(net) => net,
            Err(err) => {
                eprintln!("skipping CUDA parity test: {err}");
                return Ok(());
            }
        };

        let (cpu_loss, cpu_predictions) =
            cpu.train_step_batch_with_predictions(&input, &targets, 0.05)?;
        let (gpu_loss, gpu_predictions) =
            gpu.train_step_batch_with_predictions(&input, &targets, 0.05)?;

        assert_eq!(cpu_predictions, gpu_predictions);
        assert!(
            (cpu_loss - gpu_loss).abs() < 1e-3,
            "loss mismatch: cpu={cpu_loss}, gpu={gpu_loss}"
        );

        Ok(())
    }
}
