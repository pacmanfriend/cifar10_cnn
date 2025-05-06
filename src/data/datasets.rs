use crate::compute::tensor;
use std::{error::Error, fs, io, path::Path};

pub const CIFAR10_CLASSES: usize = 10;
pub const CIFAR10_CHANNELS: usize = 3;
pub const CIFAR10_HEIGHT: usize = 32;
pub const CIFAR10_WIDTH: usize = 32;
pub const CIFAR10_IMAGE_LEN: usize = CIFAR10_CHANNELS * CIFAR10_HEIGHT * CIFAR10_WIDTH;
pub const CIFAR10_RECORD_LEN: usize = 1 + CIFAR10_IMAGE_LEN;
pub const CIFAR10_MEAN: [f32; CIFAR10_CHANNELS] = [0.4914, 0.4822, 0.4465];
pub const CIFAR10_STD: [f32; CIFAR10_CHANNELS] = [0.2470, 0.2435, 0.2616];

pub type Dataset = Vec<(tensor::Tensor, usize)>;

pub fn make_fake_dataset() -> Dataset {
    let mut data = vec![];

    for offset in 0..4 {
        // Vertical line
        let mut img = vec![0.0_f32; 64];
        for i in 0..8 {
            img[i * 8 + offset + 2] = 1.0;
        }
        data.push((
            tensor::Tensor {
                data: img,
                shape: vec![1, 8, 8],
            },
            0,
        ));

        // Horizontal line
        let mut img = vec![0.0_f32; 64];
        for j in 0..8 {
            img[(offset + 2) * 8 + j] = 1.0;
        }
        data.push((
            tensor::Tensor {
                data: img,
                shape: vec![1, 8, 8],
            },
            1,
        ));

        // Diagonal line
        let mut img = vec![0.0_f32; 64];
        for i in 0..8 {
            let j = (i + offset) % 8;
            img[i * 8 + j] = 1.0;
        }
        data.push((
            tensor::Tensor {
                data: img,
                shape: vec![1, 8, 8],
            },
            2,
        ));
    }

    data
}

pub fn load_cifar10_batch(path: &Path) -> Result<Dataset, Box<dyn Error>> {
    let bytes = fs::read(path)?;
    if bytes.len() % CIFAR10_RECORD_LEN != 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "CIFAR-10 batch length {} is not divisible by record length {}",
                bytes.len(),
                CIFAR10_RECORD_LEN
            ),
        )
        .into());
    }

    let mut samples = Vec::with_capacity(bytes.len() / CIFAR10_RECORD_LEN);

    for record in bytes.chunks_exact(CIFAR10_RECORD_LEN) {
        let target = record[0] as usize;
        if target >= CIFAR10_CLASSES {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("CIFAR-10 label {target} is outside 0..{CIFAR10_CLASSES}"),
            )
            .into());
        }

        let mut data = vec![0.0_f32; CIFAR10_IMAGE_LEN];
        for channel in 0..CIFAR10_CHANNELS {
            let channel_offset = channel * CIFAR10_HEIGHT * CIFAR10_WIDTH;
            for pixel in 0..(CIFAR10_HEIGHT * CIFAR10_WIDTH) {
                let raw = record[1 + channel_offset + pixel] as f32 / 255.0;
                data[channel_offset + pixel] = (raw - CIFAR10_MEAN[channel]) / CIFAR10_STD[channel];
            }
        }

        samples.push((
            tensor::Tensor::from_data(data, vec![CIFAR10_CHANNELS, CIFAR10_HEIGHT, CIFAR10_WIDTH]),
            target,
        ));
    }

    Ok(samples)
}

pub fn load_cifar10(dir: &Path) -> Result<(Dataset, Dataset), Box<dyn Error>> {
    let mut train = Vec::new();

    for batch_index in 1..=5 {
        let path = dir.join(format!("data_batch_{batch_index}.bin"));
        train.extend(load_cifar10_batch(&path)?);
    }

    let test = load_cifar10_batch(&dir.join("test_batch.bin"))?;

    Ok((train, test))
}

#[cfg(test)]
mod tests {
    use super::{
        load_cifar10, load_cifar10_batch, make_fake_dataset, CIFAR10_CHANNELS, CIFAR10_HEIGHT,
        CIFAR10_MEAN, CIFAR10_RECORD_LEN, CIFAR10_STD, CIFAR10_WIDTH,
    };
    use std::{
        fs,
        path::PathBuf,
        time::{SystemTime, UNIX_EPOCH},
    };

    fn temp_path(name: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time must be after unix epoch")
            .as_nanos();
        std::env::temp_dir().join(format!(
            "cifar10_cnn_{name}_{}_{}",
            std::process::id(),
            nanos
        ))
    }

    fn normalized(raw: u8, channel: usize) -> f32 {
        (raw as f32 / 255.0 - CIFAR10_MEAN[channel]) / CIFAR10_STD[channel]
    }

    fn record(label: u8, red: u8, green: u8, blue: u8) -> Vec<u8> {
        let mut bytes = vec![0; CIFAR10_RECORD_LEN];
        bytes[0] = label;
        bytes[1] = red;
        bytes[1 + CIFAR10_HEIGHT * CIFAR10_WIDTH] = green;
        bytes[1 + 2 * CIFAR10_HEIGHT * CIFAR10_WIDTH + CIFAR10_HEIGHT * CIFAR10_WIDTH - 1] = blue;
        bytes
    }

    #[test]
    fn fake_dataset_has_expected_shape_and_classes() {
        let dataset = make_fake_dataset();

        assert_eq!(dataset.len(), 12);

        for (input, target) in dataset {
            assert_eq!(input.shape, vec![1, 8, 8]);
            assert_eq!(input.data.len(), 64);
            assert!((0..3).contains(&target), "unexpected class: {target}");
        }
    }

    #[test]
    fn load_cifar10_batch_reads_shape_target_and_normalizes_channels() {
        let path = temp_path("single_batch.bin");
        let mut bytes = record(7, 255, 128, 64);
        bytes.extend(record(3, 0, 255, 128));
        fs::write(&path, bytes).expect("failed to write temporary CIFAR-10 batch");

        let dataset = load_cifar10_batch(&path).expect("batch must load");
        fs::remove_file(&path).expect("failed to remove temporary CIFAR-10 batch");

        assert_eq!(dataset.len(), 2);
        assert_eq!(dataset[0].1, 7);
        assert_eq!(dataset[1].1, 3);
        assert_eq!(
            dataset[0].0.shape,
            vec![CIFAR10_CHANNELS, CIFAR10_HEIGHT, CIFAR10_WIDTH]
        );

        let first = &dataset[0].0;
        let channel_len = CIFAR10_HEIGHT * CIFAR10_WIDTH;
        assert!((first.data[0] - normalized(255, 0)).abs() < 1e-6);
        assert!((first.data[channel_len] - normalized(128, 1)).abs() < 1e-6);
        assert!((first.data[3 * channel_len - 1] - normalized(64, 2)).abs() < 1e-6);
    }

    #[test]
    fn load_cifar10_batch_rejects_incomplete_record() {
        let path = temp_path("bad_len.bin");
        fs::write(&path, vec![0; CIFAR10_RECORD_LEN - 1])
            .expect("failed to write temporary CIFAR-10 batch");

        let err = match load_cifar10_batch(&path) {
            Ok(_) => panic!("batch length must be rejected"),
            Err(err) => err,
        };
        fs::remove_file(&path).expect("failed to remove temporary CIFAR-10 batch");

        assert!(err.to_string().contains("not divisible by record length"));
    }

    #[test]
    fn load_cifar10_batch_rejects_invalid_label() {
        let path = temp_path("bad_label.bin");
        fs::write(&path, record(10, 0, 0, 0)).expect("failed to write temporary CIFAR-10 batch");

        let err = match load_cifar10_batch(&path) {
            Ok(_) => panic!("invalid label must be rejected"),
            Err(err) => err,
        };
        fs::remove_file(&path).expect("failed to remove temporary CIFAR-10 batch");

        assert!(err.to_string().contains("outside 0..10"));
    }

    #[test]
    fn load_cifar10_reads_train_and_test_files() {
        let dir = temp_path("dataset_dir");
        fs::create_dir(&dir).expect("failed to create temporary CIFAR-10 directory");
        for batch_index in 1..=5 {
            fs::write(
                dir.join(format!("data_batch_{batch_index}.bin")),
                record(batch_index as u8, 0, 0, 0),
            )
            .expect("failed to write temporary train batch");
        }
        fs::write(dir.join("test_batch.bin"), record(9, 0, 0, 0))
            .expect("failed to write temporary test batch");

        let (train, test) = load_cifar10(&dir).expect("dataset must load");
        fs::remove_dir_all(&dir).expect("failed to remove temporary CIFAR-10 directory");

        assert_eq!(train.len(), 5);
        assert_eq!(test.len(), 1);
        assert_eq!(train[0].1, 1);
        assert_eq!(train[4].1, 5);
        assert_eq!(test[0].1, 9);
    }
}
