use crate::{compute::random, config, data::datasets, training::network};
use std::{error::Error, path::Path};

#[derive(Clone, Copy, Debug)]
pub struct TrainOptions {
    pub epochs: usize,
    pub learning_rate: f32,
    pub lr_decay_epochs: usize,
    pub lr_decay_factor: f32,
    pub batch_size: usize,
    pub momentum: f32,
    pub seed: u64,
}

impl TrainOptions {
    pub const fn demo() -> Self {
        Self {
            epochs: 50,
            learning_rate: 0.05,
            lr_decay_epochs: 0,
            lr_decay_factor: 1.0,
            batch_size: 1,
            momentum: 0.0,
            seed: 42,
        }
    }

    pub const fn cifar10() -> Self {
        Self {
            epochs: 10,
            learning_rate: 0.003,
            lr_decay_epochs: 5,
            lr_decay_factor: 0.5,
            batch_size: 64,
            momentum: 0.9,
            seed: 42,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct EpochMetrics {
    pub epoch: usize,
    pub learning_rate: f32,
    pub train_avg_loss: f32,
    pub train_correct: usize,
    pub train_total: usize,
    pub test_correct: Option<usize>,
    pub test_total: Option<usize>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TrainingHistory {
    pub train_len: usize,
    pub test_len: Option<usize>,
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

pub fn shuffle_indices(indices: &mut [usize], rng: &mut random::Rng) {
    for i in (1..indices.len()).rev() {
        let j = rng.gen_range(i + 1);
        indices.swap(i, j);
    }
}

pub fn learning_rate_for_epoch(options: TrainOptions, epoch: usize) -> f32 {
    if options.lr_decay_epochs == 0 {
        return options.learning_rate;
    }

    let decay_steps = epoch / options.lr_decay_epochs;
    options.learning_rate * options.lr_decay_factor.powi(decay_steps as i32)
}

pub fn train_demo(
    backend: network::Backend,
    options: TrainOptions,
) -> Result<TrainingHistory, Box<dyn Error>> {
    train_demo_with_checkpoints(backend, options, None, None)
}

pub fn train_demo_with_checkpoints(
    backend: network::Backend,
    options: TrainOptions,
    load_path: Option<&Path>,
    save_path: Option<&Path>,
) -> Result<TrainingHistory, Box<dyn Error>> {
    train_dataset(
        backend,
        config::ModelConfig::demo(),
        datasets::make_fake_dataset(),
        None,
        options,
        false,
        load_path,
        save_path,
    )
}

pub fn train_cifar10(
    backend: network::Backend,
    options: TrainOptions,
    data_dir: &Path,
) -> Result<TrainingHistory, Box<dyn Error>> {
    train_cifar10_with_checkpoints(backend, options, data_dir, None, None)
}

pub fn train_cifar10_with_checkpoints(
    backend: network::Backend,
    options: TrainOptions,
    data_dir: &Path,
    load_path: Option<&Path>,
    save_path: Option<&Path>,
) -> Result<TrainingHistory, Box<dyn Error>> {
    let (train, test) = datasets::load_cifar10(data_dir)?;
    train_dataset(
        backend,
        config::ModelConfig::cifar10(),
        train,
        Some(test),
        options,
        true,
        load_path,
        save_path,
    )
}

fn train_dataset(
    backend: network::Backend,
    config: config::ModelConfig,
    train: datasets::Dataset,
    test: Option<datasets::Dataset>,
    options: TrainOptions,
    shuffle_each_epoch: bool,
    load_path: Option<&Path>,
    save_path: Option<&Path>,
) -> Result<TrainingHistory, Box<dyn Error>> {
    assert!(
        options.batch_size > 0,
        "batch_size must be greater than zero"
    );
    assert!(
        (0.0..=1.0).contains(&options.momentum),
        "momentum must be in the range [0, 1]"
    );
    assert!(
        options.lr_decay_factor > 0.0,
        "lr_decay_factor must be greater than zero"
    );

    let mut rng = random::Rng::new(options.seed);
    let mut net = network::Network::new(config, &mut rng, backend)?;

    if let Some(path) = load_path {
        net.load_weights(path)?;
        println!("Loaded weights from {}", path.display());
    }
    let train_len = train.len();
    let test_len = test.as_ref().map(Vec::len);
    let mut metrics = Vec::with_capacity(options.epochs);

    for epoch in 0..options.epochs {
        let learning_rate = learning_rate_for_epoch(options, epoch);
        let mut total_loss = 0.0;
        let mut correct = 0;

        let mut indices: Vec<usize> = (0..train_len).collect();
        if shuffle_each_epoch {
            shuffle_indices(&mut indices, &mut rng);
        }

        for start in (0..train_len).step_by(options.batch_size) {
            let (input, targets) = make_batch(&train, &indices, start, options.batch_size);
            let (loss, batch_correct) = net.train_step_batch_with_momentum(
                &input,
                &targets,
                learning_rate,
                options.momentum,
            )?;
            total_loss += loss * targets.len() as f32;
            correct += batch_correct;
        }

        let (test_correct, test_total) = match test.as_ref() {
            Some(test) => {
                let (correct, total) = evaluate(&mut net, test, options.batch_size)?;
                (Some(correct), Some(total))
            }
            None => (None, None),
        };

        metrics.push(EpochMetrics {
            epoch,
            learning_rate,
            train_avg_loss: total_loss / train_len as f32,
            train_correct: correct,
            train_total: train_len,
            test_correct,
            test_total,
        });
    }

    if let Some(path) = save_path {
        net.save_weights(path)?;
        println!("Saved weights to {}", path.display());
    }

    Ok(TrainingHistory {
        train_len,
        test_len,
        metrics,
    })
}

fn evaluate(
    net: &mut network::Network,
    dataset: &datasets::Dataset,
    batch_size: usize,
) -> Result<(usize, usize), Box<dyn Error>> {
    assert!(batch_size > 0, "batch_size must be greater than zero");

    let indices: Vec<usize> = (0..dataset.len()).collect();
    let mut correct = 0;

    for start in (0..dataset.len()).step_by(batch_size) {
        let (input, targets) = make_batch(dataset, &indices, start, batch_size);
        let predictions = net.predict_batch(&input)?;
        correct += predictions
            .iter()
            .zip(targets.iter())
            .filter(|(predicted, target)| predicted == target)
            .count();
    }

    Ok((correct, dataset.len()))
}

#[cfg(test)]
mod tests {
    use super::{learning_rate_for_epoch, make_batch, shuffle_indices, train_demo, TrainOptions};
    use crate::{compute::random, data::datasets, training::network};

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

    #[test]
    fn shuffle_is_deterministic_for_fixed_seed() {
        let mut first = (0..8).collect::<Vec<_>>();
        let mut second = (0..8).collect::<Vec<_>>();
        let mut first_rng = random::Rng::new(123);
        let mut second_rng = random::Rng::new(123);

        shuffle_indices(&mut first, &mut first_rng);
        shuffle_indices(&mut second, &mut second_rng);

        assert_eq!(first, second);
        assert_ne!(first, (0..8).collect::<Vec<_>>());
    }

    #[test]
    fn cifar10_options_enable_momentum_sgd() {
        let options = TrainOptions::cifar10();

        assert_eq!(options.learning_rate, 0.003);
        assert_eq!(options.lr_decay_epochs, 5);
        assert_eq!(options.lr_decay_factor, 0.5);
        assert_eq!(options.momentum, 0.9);
        assert_eq!(options.batch_size, 64);
    }

    #[test]
    fn learning_rate_schedule_uses_step_decay() {
        let mut options = TrainOptions::demo();
        options.learning_rate = 0.1;
        options.lr_decay_epochs = 3;
        options.lr_decay_factor = 0.5;

        assert!((learning_rate_for_epoch(options, 0) - 0.1).abs() < 1e-6);
        assert!((learning_rate_for_epoch(options, 2) - 0.1).abs() < 1e-6);
        assert!((learning_rate_for_epoch(options, 3) - 0.05).abs() < 1e-6);
        assert!((learning_rate_for_epoch(options, 7) - 0.025).abs() < 1e-6);
    }

    #[test]
    fn zero_decay_epochs_disables_learning_rate_schedule() {
        let mut options = TrainOptions::demo();
        options.learning_rate = 0.1;
        options.lr_decay_epochs = 0;
        options.lr_decay_factor = 0.5;

        assert!((learning_rate_for_epoch(options, 100) - 0.1).abs() < 1e-6);
    }

    #[test]
    fn train_demo_still_runs_after_batch_migration() -> Result<(), Box<dyn std::error::Error>> {
        let mut options = super::TrainOptions::demo();
        options.epochs = 2;
        options.batch_size = 3;

        let history = train_demo(network::Backend::Cpu, options)?;

        assert_eq!(history.train_len, datasets::make_fake_dataset().len());
        assert_eq!(history.test_len, None);
        assert_eq!(history.metrics.len(), 2);
        assert!(history
            .metrics
            .iter()
            .all(|metric| metric.train_total == history.train_len));

        Ok(())
    }

    #[test]
    fn train_history_records_scheduled_learning_rate() -> Result<(), Box<dyn std::error::Error>> {
        let mut options = super::TrainOptions::demo();
        options.epochs = 3;
        options.batch_size = 12;
        options.learning_rate = 0.1;
        options.lr_decay_epochs = 2;
        options.lr_decay_factor = 0.5;

        let history = train_demo(network::Backend::Cpu, options)?;

        assert_eq!(history.metrics.len(), 3);
        assert!((history.metrics[0].learning_rate - 0.1).abs() < 1e-6);
        assert!((history.metrics[1].learning_rate - 0.1).abs() < 1e-6);
        assert!((history.metrics[2].learning_rate - 0.05).abs() < 1e-6);

        Ok(())
    }
}
