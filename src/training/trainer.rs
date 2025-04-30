use crate::{compute::random, config, data::datasets, training::network};
use std::error::Error;

#[derive(Clone, Copy, Debug)]
pub struct TrainOptions {
    pub epochs: usize,
    pub learning_rate: f32,
    pub seed: u64,
}

impl TrainOptions {
    pub const fn demo() -> Self {
        Self {
            epochs: 50,
            learning_rate: 0.05,
            seed: 42,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct EpochMetrics {
    pub epoch: usize,
    pub avg_loss: f32,
    pub correct: usize,
    pub total: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TrainingHistory {
    pub dataset_len: usize,
    pub metrics: Vec<EpochMetrics>,
}

pub fn demo_dataset_len() -> usize {
    datasets::make_fake_dataset().len()
}

pub fn train_demo(
    backend: network::Backend,
    options: TrainOptions,
) -> Result<TrainingHistory, Box<dyn Error>> {
    let mut rng = random::Rng::new(options.seed);
    let config = config::ModelConfig::demo();
    let mut net = network::Network::new(config, &mut rng, backend)?;
    let dataset = datasets::make_fake_dataset();
    let dataset_len = dataset.len();
    let mut metrics = Vec::with_capacity(options.epochs);

    for epoch in 0..options.epochs {
        let mut total_loss = 0.0;
        let mut correct = 0;

        for (input, target) in dataset.iter() {
            let (loss, predicted) = net.train_step(input, *target, options.learning_rate)?;
            total_loss += loss;
            if predicted == *target {
                correct += 1;
            }
        }

        metrics.push(EpochMetrics {
            epoch,
            avg_loss: total_loss / dataset_len as f32,
            correct,
            total: dataset_len,
        });
    }

    Ok(TrainingHistory {
        dataset_len,
        metrics,
    })
}
