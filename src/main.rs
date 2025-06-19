use cifar10_cnn::training::{network, trainer};
use std::path::PathBuf;

const DEFAULT_CIFAR10_DIR: &str = "data/cifar-10-batches-bin";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args(std::env::args().skip(1))?;

    let backend = match args.backend.as_str() {
        "cpu" => network::Backend::Cpu,
        "gpu" | "cuda" => network::Backend::Gpu,
        _ => {
            return Err(format!(
                "unknown mode '{}', use 'cpu', 'gpu', or 'cuda'",
                args.backend
            )
            .into())
        }
    };

    let default_data_dir;
    let data_dir = if let Some(data_dir) = args.data_dir.as_ref() {
        Some(data_dir)
    } else if args.cifar10 {
        default_data_dir = PathBuf::from(DEFAULT_CIFAR10_DIR);
        Some(&default_data_dir)
    } else {
        None
    };

    let (options, history) = if let Some(data_dir) = data_dir {
        let mut options = trainer::TrainOptions::cifar10();
        apply_overrides(&mut options, &args);
        println!("Dataset: CIFAR-10 at {}", data_dir.display());
        println!("Start {backend:?} CIFAR-10 training...\n");
        let history = trainer::train_cifar10(backend, options, data_dir)?;
        (options, history)
    } else {
        let mut options = trainer::TrainOptions::demo();
        apply_overrides(&mut options, &args);
        let dataset_len = trainer::demo_dataset_len();
        println!("Dataset size: {dataset_len}");
        println!("Start {backend:?} demo training...\n");
        let history = trainer::train_demo(backend, options)?;
        (options, history)
    };

    for metric in history.metrics {
        match (metric.test_correct, metric.test_total) {
            (Some(test_correct), Some(test_total)) => println!(
                "Epoch {:>2}: train loss = {:.4}, train accuracy = {}/{}, test accuracy = {}/{}",
                metric.epoch,
                metric.train_avg_loss,
                metric.train_correct,
                metric.train_total,
                test_correct,
                test_total
            ),
            _ if metric.epoch % 5 == 0 || metric.epoch + 1 == options.epochs => {
                println!(
                    "Epoch {:>2}: avg loss = {:.4}, accuracy = {}/{}",
                    metric.epoch, metric.train_avg_loss, metric.train_correct, metric.train_total
                )
            }
            _ => {}
        }
    }

    Ok(())
}

#[derive(Debug)]
struct Args {
    backend: String,
    data_dir: Option<PathBuf>,
    cifar10: bool,
    epochs: Option<usize>,
    learning_rate: Option<f32>,
    batch_size: Option<usize>,
    momentum: Option<f32>,
}

fn parse_args(mut args: impl Iterator<Item = String>) -> Result<Args, Box<dyn std::error::Error>> {
    let mut parsed = Args {
        backend: args.next().unwrap_or_else(|| "cpu".to_string()),
        data_dir: None,
        cifar10: false,
        epochs: None,
        learning_rate: None,
        batch_size: None,
        momentum: None,
    };

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--data" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--data requires a path argument".to_string())?;
                parsed.data_dir = Some(PathBuf::from(value));
            }
            "--cifar" | "--cifar10" => {
                parsed.cifar10 = true;
            }
            "--epochs" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--epochs requires a number".to_string())?;
                parsed.epochs = Some(value.parse()?);
            }
            "--lr" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--lr requires a number".to_string())?;
                parsed.learning_rate = Some(value.parse()?);
            }
            "--batch-size" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--batch-size requires a number".to_string())?;
                parsed.batch_size = Some(value.parse()?);
            }
            "--momentum" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--momentum requires a number".to_string())?;
                parsed.momentum = Some(value.parse()?);
            }
            _ => return Err(format!("unknown argument '{arg}'").into()),
        }
    }

    Ok(parsed)
}

fn apply_overrides(options: &mut trainer::TrainOptions, args: &Args) {
    if let Some(epochs) = args.epochs {
        options.epochs = epochs;
    }
    if let Some(learning_rate) = args.learning_rate {
        options.learning_rate = learning_rate;
    }
    if let Some(batch_size) = args.batch_size {
        options.batch_size = batch_size;
    }
    if let Some(momentum) = args.momentum {
        options.momentum = momentum;
    }
}
