use crate::{
    compute::tensor::Tensor,
    data::datasets::{CIFAR10_CHANNELS, CIFAR10_HEIGHT, CIFAR10_MEAN, CIFAR10_STD, CIFAR10_WIDTH},
};
use image::{imageops::FilterType, ImageError};

pub fn image_bytes_to_tensor(bytes: &[u8]) -> Result<Tensor, ImageError> {
    let image = image::load_from_memory(bytes)?;
    let resized = image
        .resize_exact(
            CIFAR10_WIDTH as u32,
            CIFAR10_HEIGHT as u32,
            FilterType::Lanczos3,
        )
        .to_rgb8();

    let channel_len = CIFAR10_HEIGHT * CIFAR10_WIDTH;
    let mut data = vec![0.0; CIFAR10_CHANNELS * channel_len];

    for y in 0..CIFAR10_HEIGHT {
        for x in 0..CIFAR10_WIDTH {
            let pixel = resized.get_pixel(x as u32, y as u32).0;
            let pixel_index = y * CIFAR10_WIDTH + x;
            for channel in 0..CIFAR10_CHANNELS {
                let raw = pixel[channel] as f32 / 255.0;
                data[channel * channel_len + pixel_index] =
                    (raw - CIFAR10_MEAN[channel]) / CIFAR10_STD[channel];
            }
        }
    }

    Ok(Tensor::from_data(
        data,
        vec![1, CIFAR10_CHANNELS, CIFAR10_HEIGHT, CIFAR10_WIDTH],
    ))
}

#[cfg(test)]
mod tests {
    use super::image_bytes_to_tensor;
    use crate::data::datasets::{CIFAR10_CHANNELS, CIFAR10_HEIGHT, CIFAR10_WIDTH};
    use image::{DynamicImage, ImageFormat, Rgb, RgbImage};
    use std::io::Cursor;

    #[test]
    fn image_to_tensor_shape_and_range() {
        let mut image = RgbImage::new(4, 4);
        for pixel in image.pixels_mut() {
            *pixel = Rgb([128, 64, 32]);
        }

        let mut bytes = Vec::new();
        DynamicImage::ImageRgb8(image)
            .write_to(&mut Cursor::new(&mut bytes), ImageFormat::Png)
            .unwrap();

        let tensor = image_bytes_to_tensor(&bytes).unwrap();

        assert_eq!(
            tensor.shape,
            vec![1, CIFAR10_CHANNELS, CIFAR10_HEIGHT, CIFAR10_WIDTH]
        );
        assert_eq!(
            tensor.data.len(),
            CIFAR10_CHANNELS * CIFAR10_HEIGHT * CIFAR10_WIDTH
        );
        assert!(tensor.data.iter().all(|value| value.is_finite()));
    }
}
