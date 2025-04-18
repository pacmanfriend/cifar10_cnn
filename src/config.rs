#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ModelConfig {
    pub input_channels: usize,
    pub input_height: usize,
    pub input_width: usize,
    pub conv_out_channels: usize,
    pub conv_kernel: usize,
    pub num_classes: usize,
}

impl ModelConfig {
    pub const fn demo() -> Self {
        Self {
            input_channels: 1,
            input_height: 8,
            input_width: 8,
            conv_out_channels: 4,
            conv_kernel: 3,
            num_classes: 3,
        }
    }

    pub const fn conv_height(self) -> usize {
        self.input_height - self.conv_kernel + 1
    }

    pub const fn conv_width(self) -> usize {
        self.input_width - self.conv_kernel + 1
    }

    pub const fn conv_dim(self) -> usize {
        self.conv_out_channels * self.conv_height() * self.conv_width()
    }

    pub const fn pool_height(self) -> usize {
        self.conv_height() / 2
    }

    pub const fn pool_width(self) -> usize {
        self.conv_width() / 2
    }

    pub const fn flat_dim(self) -> usize {
        self.conv_out_channels * self.pool_height() * self.pool_width()
    }

    pub const fn input_dim(self) -> usize {
        self.input_channels * self.input_height * self.input_width
    }

    pub const fn conv_weight_len(self) -> usize {
        self.conv_out_channels * self.input_channels * self.conv_kernel * self.conv_kernel
    }
}
