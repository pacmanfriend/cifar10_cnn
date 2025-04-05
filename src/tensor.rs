use crate::random;

#[derive(Clone)]
pub struct Tensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}

impl Tensor {
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
}
