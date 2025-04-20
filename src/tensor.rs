use crate::random;

#[derive(Clone)]
pub struct Tensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}

impl Tensor {
    pub fn from_data(data: Vec<f32>, shape: Vec<usize>) -> Self {
        let expected_len: usize = shape.iter().product();
        assert_eq!(
            data.len(),
            expected_len,
            "tensor data length does not match shape"
        );
        Tensor { data, shape }
    }

    pub fn zeros(shape: Vec<usize>) -> Self {
        let n: usize = shape.iter().product();
        Tensor {
            data: vec![0.0; n],
            shape,
        }
    }

    pub fn random(shape: Vec<usize>, rng: &mut random::Rng, scale: f32) -> Self {
        let n: usize = shape.iter().product();
        let data = (0..n).map(|_| rng.normal() * scale).collect();
        Tensor { data, shape }
    }

    pub fn numel(&self) -> usize {
        self.data.len()
    }

    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    pub fn reshape(mut self, shape: Vec<usize>) -> Self {
        let expected_len: usize = shape.iter().product();
        assert_eq!(
            self.data.len(),
            expected_len,
            "reshape must preserve tensor element count"
        );
        self.shape = shape;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::Tensor;

    #[test]
    fn from_data_accepts_matching_shape() {
        let tensor = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);

        assert_eq!(tensor.numel(), 4);
        assert_eq!(tensor.rank(), 3);
        assert_eq!(tensor.shape, vec![1, 2, 2]);
    }

    #[test]
    fn reshape_preserves_data_and_updates_shape() {
        let tensor = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
        let reshaped = tensor.reshape(vec![2, 2]);

        assert_eq!(reshaped.data, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(reshaped.shape, vec![2, 2]);
    }

    #[test]
    #[should_panic(expected = "reshape must preserve tensor element count")]
    fn reshape_rejects_mismatched_shape() {
        let tensor = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![4]);

        let _ = tensor.reshape(vec![3]);
    }
}
