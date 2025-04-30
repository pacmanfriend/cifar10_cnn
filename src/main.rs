use cifar10_cnn::training::{network, trainer};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mode = std::env::args().nth(1).unwrap_or_else(|| "cpu".to_string());

    let backend = match mode.as_str() {
        "cpu" => network::Backend::Cpu,
        "gpu" | "cuda" => network::Backend::Gpu,
        _ => return Err(format!("unknown mode '{mode}', use 'cpu' or 'gpu'").into()),
    };

    let options = trainer::TrainOptions::demo();
    let dataset_len = trainer::demo_dataset_len();

    println!("Dataset size: {dataset_len}");
    println!("Start {backend:?} training...\n");

    let history = trainer::train_demo(backend, options)?;

    for metric in history.metrics {
        if metric.epoch % 5 == 0 || metric.epoch + 1 == options.epochs {
            println!(
                "Epoch {:>2}: avg loss = {:.4}, accuracy = {}/{}",
                metric.epoch, metric.avg_loss, metric.correct, metric.total
            );
        }
    }

    Ok(())
}
