pub mod api;
pub mod compute;
pub mod config;
pub mod cuda;
pub mod data;
pub mod layers;
pub mod training;

pub use compute::{random, tensor};
pub use data::datasets;
pub use layers::{conv, dense, maxpool, relu};
pub use training::{loss, network, optimizer};
