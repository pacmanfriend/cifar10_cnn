use crate::compute::tensor;

pub fn make_fake_dataset() -> Vec<(tensor::Tensor, usize)> {
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

#[cfg(test)]
mod tests {
    use super::make_fake_dataset;

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
}
