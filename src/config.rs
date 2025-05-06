#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ModelConfig {
    pub input_channels: usize,
    pub input_height: usize,
    pub input_width: usize,
    pub conv_out_channels: usize,
    pub conv2_out_channels: Option<usize>,
    pub conv_kernel: usize,
    pub conv_padding: usize,
    pub num_classes: usize,
}

impl ModelConfig {
    pub const fn demo() -> Self {
        Self {
            input_channels: 1,
            input_height: 8,
            input_width: 8,
            conv_out_channels: 4,
            conv2_out_channels: None,
            conv_kernel: 3,
            conv_padding: 0,
            num_classes: 3,
        }
    }

    pub const fn cifar10() -> Self {
        Self {
            input_channels: 3,
            input_height: 32,
            input_width: 32,
            conv_out_channels: 32,
            conv2_out_channels: Some(64),
            conv_kernel: 3,
            conv_padding: 1,
            num_classes: 10,
        }
    }

    pub const fn conv_height(self) -> usize {
        self.conv1_height()
    }

    pub const fn conv_width(self) -> usize {
        self.conv1_width()
    }

    pub const fn conv_dim(self) -> usize {
        self.conv_out_channels * self.conv1_height() * self.conv1_width()
    }

    pub const fn pool_height(self) -> usize {
        self.pool1_height()
    }

    pub const fn pool_width(self) -> usize {
        self.pool1_width()
    }

    pub const fn flat_dim(self) -> usize {
        match self.conv2_out_channels {
            Some(channels) => channels * self.pool2_height() * self.pool2_width(),
            None => self.conv_out_channels * self.pool1_height() * self.pool1_width(),
        }
    }

    pub const fn input_dim(self) -> usize {
        self.input_channels * self.input_height * self.input_width
    }

    pub const fn conv_weight_len(self) -> usize {
        self.conv1_weight_len()
    }

    pub const fn conv1_height(self) -> usize {
        self.input_height + 2 * self.conv_padding - self.conv_kernel + 1
    }

    pub const fn conv1_width(self) -> usize {
        self.input_width + 2 * self.conv_padding - self.conv_kernel + 1
    }

    pub const fn pool1_height(self) -> usize {
        self.conv1_height() / 2
    }

    pub const fn pool1_width(self) -> usize {
        self.conv1_width() / 2
    }

    pub const fn conv2_height(self) -> usize {
        self.pool1_height() + 2 * self.conv_padding - self.conv_kernel + 1
    }

    pub const fn conv2_width(self) -> usize {
        self.pool1_width() + 2 * self.conv_padding - self.conv_kernel + 1
    }

    pub const fn pool2_height(self) -> usize {
        self.conv2_height() / 2
    }

    pub const fn pool2_width(self) -> usize {
        self.conv2_width() / 2
    }

    pub const fn conv1_weight_len(self) -> usize {
        self.conv_out_channels * self.input_channels * self.conv_kernel * self.conv_kernel
    }

    pub const fn conv2_weight_len(self) -> usize {
        match self.conv2_out_channels {
            Some(channels) => {
                channels * self.conv_out_channels * self.conv_kernel * self.conv_kernel
            }
            None => 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::ModelConfig;

    #[test]
    fn demo_config_derives_expected_dimensions() {
        let config = ModelConfig::demo();

        assert_eq!(config.conv_padding, 0);
        assert_eq!(config.conv_height(), 6);
        assert_eq!(config.conv_width(), 6);
        assert_eq!(config.conv_dim(), 144);
        assert_eq!(config.pool_height(), 3);
        assert_eq!(config.pool_width(), 3);
        assert_eq!(config.flat_dim(), 36);
        assert_eq!(config.input_dim(), 64);
        assert_eq!(config.conv_weight_len(), 36);
        assert_eq!(config.conv1_weight_len(), 36);
        assert_eq!(config.conv2_weight_len(), 0);
    }

    #[test]
    fn cifar10_config_derives_expected_dimensions() {
        let config = ModelConfig::cifar10();

        assert_eq!(config.input_channels, 3);
        assert_eq!(config.input_height, 32);
        assert_eq!(config.input_width, 32);
        assert_eq!(config.conv_out_channels, 32);
        assert_eq!(config.conv2_out_channels, Some(64));
        assert_eq!(config.conv_kernel, 3);
        assert_eq!(config.conv_padding, 1);
        assert_eq!(config.num_classes, 10);

        assert_eq!(config.conv1_height(), 32);
        assert_eq!(config.conv1_width(), 32);
        assert_eq!(config.pool1_height(), 16);
        assert_eq!(config.pool1_width(), 16);
        assert_eq!(config.conv2_height(), 16);
        assert_eq!(config.conv2_width(), 16);
        assert_eq!(config.pool2_height(), 8);
        assert_eq!(config.pool2_width(), 8);
        assert_eq!(config.flat_dim(), 4096);
        assert_eq!(config.conv1_weight_len(), 864);
        assert_eq!(config.conv2_weight_len(), 18_432);
    }
}
