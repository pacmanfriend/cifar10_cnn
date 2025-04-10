mod conv;
mod datasets;
mod dense;
mod maxpool;
mod network;
mod random;
mod relu;
mod tensor;

// use std::f32;

fn main() {
    let mut rng = random::Rng::new(42);
    let mut net = network::Network::new(&mut rng);
    let dataset = datasets::make_fake_dataset();

    println!("Dataset size: {}", dataset.len());
    println!("Start training...\n");

    let lr = 0.05;
    for epoch in 0..50 {
        let mut total_loss = 0.0;
        let mut correct = 0;

        for (input, target) in dataset.iter() {
            let (loss, predicted) = net.train_step(input, *target, lr);
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
}
