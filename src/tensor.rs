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
