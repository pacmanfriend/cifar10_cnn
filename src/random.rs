pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng { state: seed }
    }

    fn next(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.state
    }

    fn uniform(&mut self) -> f32 {
        (self.next() >> 32) as f32 / u32::MAX as f32
    }

    pub fn normal(&mut self) -> f32 {
        let u1 = self.uniform().max(1e-7); // избегаем log(0)
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }
}
