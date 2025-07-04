use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ApiBackend {
    Cpu,
    Gpu,
}

impl From<ApiBackend> for crate::training::network::Backend {
    fn from(value: ApiBackend) -> Self {
        match value {
            ApiBackend::Cpu => Self::Cpu,
            ApiBackend::Gpu => Self::Gpu,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct SystemGpuInfo {
    pub available: bool,
    pub name: Option<String>,
    pub vram_mb: Option<usize>,
    pub driver_version: Option<String>,
    pub error: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct SystemCpuInfo {
    pub name: String,
    pub physical_cores: Option<usize>,
    pub logical_cores: usize,
    pub usage_percent: f32,
}

#[derive(Clone, Debug, Serialize)]
pub struct ModelInfo {
    pub architecture: String,
    pub param_count: usize,
    pub backend: ApiBackend,
    pub loaded: bool,
}

#[derive(Clone, Debug, Deserialize)]
pub struct LoadRequest {
    pub path: String,
}

#[derive(Clone, Debug, Deserialize)]
pub struct SaveRequest {
    pub path: String,
}

#[derive(Clone, Debug, Deserialize)]
pub struct BackendRequest {
    pub backend: ApiBackend,
}

#[derive(Clone, Debug, Deserialize)]
pub struct TrainRequest {
    pub data_dir: String,
    pub backend: Option<ApiBackend>,
    pub epochs: Option<usize>,
    pub learning_rate: Option<f32>,
    pub lr_decay_epochs: Option<usize>,
    pub lr_decay_factor: Option<f32>,
    pub batch_size: Option<usize>,
    pub momentum: Option<f32>,
    pub load_checkpoint: Option<String>,
    pub save_checkpoint: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct TrainStartResponse {
    pub job_id: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct TrainStatus {
    pub status: String,
    pub epoch: usize,
    pub loss: Option<f32>,
    pub accuracy: Option<f32>,
    pub error: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct PredictResponse {
    pub class_id: usize,
    pub class_name: String,
    pub confidence: f32,
    pub scores: Vec<f32>,
}

#[derive(Clone, Debug, Serialize)]
pub struct ErrorResponse {
    pub error: String,
}
