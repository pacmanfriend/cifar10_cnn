use cifar10_cnn::{config, datasets, network, random};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mode = std::env::args().nth(1).unwrap_or_else(|| "cpu".to_string());

    let backend = match mode.as_str() {
        "cpu" => network::Backend::Cpu,
        "gpu" | "cuda" => network::Backend::Gpu,
        _ => return Err(format!("unknown mode '{mode}', use 'cpu' or 'gpu'").into()),
    };

    train(backend)
}

fn train(backend: network::Backend) -> Result<(), Box<dyn std::error::Error>> {
    let mut rng = random::Rng::new(42);
    let config = config::ModelConfig::demo();
    let mut net = network::Network::new(config, &mut rng, backend)?;
    let dataset = datasets::make_fake_dataset();

    println!("Dataset size: {}", dataset.len());
    println!("Start {backend:?} training...\n");

    let lr = 0.05;
    for epoch in 0..50 {
        let mut total_loss = 0.0;
        let mut correct = 0;

        for (input, target) in dataset.iter() {
            let (loss, predicted) = net.train_step(input, *target, lr)?;
            total_loss += loss;
            if predicted == *target {
                correct += 1;
            }
        }

        if epoch % 5 == 0 || epoch == 49 {
            println!(
                "Epoch {:>2}: avg loss = {:.4}, accuracy = {}/{}",
                epoch,
                total_loss / dataset.len() as f32,
                correct,
                dataset.len()
            );
        }
    }

    Ok(())
}
