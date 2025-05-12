use crate::{compute::random, config, data::datasets, training::network};
use std::error::Error;

#[derive(Clone, Copy, Debug)]
pub struct TrainOptions {
    pub epochs: usize,
    pub learning_rate: f32,
    pub batch_size: usize,
    pub seed: u64,
}

impl TrainOptions {
    pub const fn demo() -> Self {
        Self {
            epochs: 50,
            learning_rate: 0.05,
            batch_size: 1,
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

pub fn make_batch(
    samples: &[(crate::compute::tensor::Tensor, usize)],
    indices: &[usize],
    start: usize,
    batch_size: usize,
) -> (crate::compute::tensor::Tensor, Vec<usize>) {
    let end = (start + batch_size).min(indices.len());
    let actual_batch = end - start;
    assert!(actual_batch > 0, "batch must contain at least one sample");

    let first = &samples[indices[start]].0;
    assert_eq!(first.rank(), 3);

    let sample_len = first.numel();
    let mut data = Vec::with_capacity(actual_batch * sample_len);
    let mut targets = Vec::with_capacity(actual_batch);

    for &sample_index in &indices[start..end] {
        let (input, target) = &samples[sample_index];
        assert_eq!(input.shape, first.shape);
        data.extend_from_slice(&input.data);
        targets.push(*target);
    }

    let mut shape = Vec::with_capacity(first.shape.len() + 1);
    shape.push(actual_batch);
    shape.extend_from_slice(&first.shape);

    (
        crate::compute::tensor::Tensor::from_data(data, shape),
        targets,
    )
}

pub fn train_demo(
    backend: network::Backend,
    options: TrainOptions,
) -> Result<TrainingHistory, Box<dyn Error>> {
    assert!(
        options.batch_size > 0,
        "batch_size must be greater than zero"
    );

    let mut rng = random::Rng::new(options.seed);
    let config = config::ModelConfig::demo();
    let mut net = network::Network::new(config, &mut rng, backend)?;
    let dataset = datasets::make_fake_dataset();
    let dataset_len = dataset.len();
    let mut metrics = Vec::with_capacity(options.epochs);

    for epoch in 0..options.epochs {
        let mut total_loss = 0.0;
        let mut correct = 0;

        let indices: Vec<usize> = (0..dataset_len).collect();

        match backend {
            network::Backend::Cpu => {
                for start in (0..dataset_len).step_by(options.batch_size) {
                    let (input, targets) =
                        make_batch(&dataset, &indices, start, options.batch_size);
                    let (loss, batch_correct) =
                        net.train_step_batch(&input, &targets, options.learning_rate)?;
                    total_loss += loss * targets.len() as f32;
                    correct += batch_correct;
                }
            }
            network::Backend::Gpu => {
                for (input, target) in dataset.iter() {
                    let (loss, predicted) =
                        net.train_step(input, *target, options.learning_rate)?;
                    total_loss += loss;
                    if predicted == *target {
                        correct += 1;
                    }
                }
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

#[cfg(test)]
mod tests {
    use super::make_batch;
    use crate::data::datasets;

    #[test]
    fn make_batch_copies_samples_as_contiguous_nchw() {
        let dataset = datasets::make_fake_dataset();
        let indices = [2, 0];

        let (batch, targets) = make_batch(&dataset, &indices, 0, 2);

        assert_eq!(batch.shape, vec![2, 1, 8, 8]);
        assert_eq!(targets, vec![dataset[2].1, dataset[0].1]);
        assert_eq!(&batch.data[0..64], &dataset[2].0.data[..]);
        assert_eq!(&batch.data[64..128], &dataset[0].0.data[..]);
    }
}
