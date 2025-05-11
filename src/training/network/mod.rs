use crate::{
    compute::{random, tensor},
    config, cuda,
};
use std::{error::Error, io};

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
            NetworkInner::Gpu(_) => Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "batched train_step is not implemented for GPU backend yet",
            )
            .into()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Backend, Network};
    use crate::{compute::random, config, data::datasets};

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
    #[ignore]
    fn cpu_and_gpu_train_step_match_for_fixed_seed() -> Result<(), Box<dyn std::error::Error>> {
        let config = config::ModelConfig::demo();
        let dataset = datasets::make_fake_dataset();
        let (input, target) = &dataset[0];
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

        let (cpu_loss, cpu_predicted) = cpu.train_step(input, *target, 0.05)?;
        let (gpu_loss, gpu_predicted) = gpu.train_step(input, *target, 0.05)?;

        assert_eq!(cpu_predicted, gpu_predicted);
        assert!(
            (cpu_loss - gpu_loss).abs() < 1e-4,
            "loss mismatch: cpu={cpu_loss}, gpu={gpu_loss}"
        );

        Ok(())
    }
}
