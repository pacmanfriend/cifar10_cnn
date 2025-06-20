pub mod image_util;
pub mod routes;
pub mod types;

use crate::api::types::{ApiBackend, ModelInfo, TrainStatus};
use axum::{routing::get, Router};
use std::{collections::HashMap, net::SocketAddr, sync::Arc};
use tokio::sync::Mutex;
use tower_http::services::{ServeDir, ServeFile};

#[derive(Clone)]
pub struct ApiState {
    inner: Arc<Mutex<AppState>>,
}

pub struct AppState {
    pub backend: ApiBackend,
    pub model_loaded: bool,
    pub jobs: HashMap<String, TrainStatus>,
}

impl Default for AppState {
    fn default() -> Self {
        Self {
            backend: ApiBackend::Cpu,
            model_loaded: true,
            jobs: HashMap::new(),
        }
    }
}

impl ApiState {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(AppState::default())),
        }
    }

    pub async fn model_info(&self) -> ModelInfo {
        let state = self.inner.lock().await;
        ModelInfo {
            architecture: "Input [1,3,32,32] -> Conv2D(3->32, 3x3, pad=1) -> ReLU -> MaxPool2x2 -> Conv2D(32->64, 3x3, pad=1) -> ReLU -> MaxPool2x2 -> Dense(4096->10)".to_string(),
            param_count: cifar10_param_count(),
            backend: state.backend,
            loaded: state.model_loaded,
        }
    }

    pub async fn backend(&self) -> ApiBackend {
        self.inner.lock().await.backend
    }

    pub async fn set_backend(&self, backend: ApiBackend) {
        let mut state = self.inner.lock().await;
        state.backend = backend;
        state.model_loaded = true;
    }

    pub async fn insert_job(&self, job_id: String, status: TrainStatus) {
        self.inner.lock().await.jobs.insert(job_id, status);
    }

    pub async fn update_job(&self, job_id: &str, status: TrainStatus) {
        self.inner
            .lock()
            .await
            .jobs
            .insert(job_id.to_string(), status);
    }

    pub async fn job(&self, job_id: &str) -> Option<TrainStatus> {
        self.inner.lock().await.jobs.get(job_id).cloned()
    }
}

impl Default for ApiState {
    fn default() -> Self {
        Self::new()
    }
}

pub fn router() -> Router {
    let spa = ServeDir::new("web/build").not_found_service(ServeFile::new("web/build/index.html"));

    Router::new()
        .route("/api/system/cpu", get(routes::system::cpu_info))
        .route("/api/system/gpu", get(routes::system::gpu_info))
        .route(
            "/api/model",
            get(routes::model::model_info).post(routes::model::set_backend_not_allowed),
        )
        .route(
            "/api/model/load",
            axum::routing::post(routes::model::load_model),
        )
        .route(
            "/api/model/save",
            axum::routing::post(routes::model::save_model),
        )
        .route(
            "/api/model/backend",
            axum::routing::post(routes::model::set_backend),
        )
        .route(
            "/api/train",
            axum::routing::post(routes::train::start_train),
        )
        .route("/api/train/{job_id}", get(routes::train::train_status))
        .route(
            "/api/predict",
            axum::routing::post(routes::predict::predict),
        )
        .fallback_service(spa)
        .with_state(ApiState::new())
}

pub async fn serve(port: u16) -> Result<(), Box<dyn std::error::Error>> {
    let app = router();
    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    let listener = tokio::net::TcpListener::bind(addr).await?;
    println!("Serving API at http://{addr}");
    axum::serve(listener, app).await?;
    Ok(())
}

fn cifar10_param_count() -> usize {
    let config = crate::config::ModelConfig::cifar10();
    config.conv1_weight_len()
        + config.conv_out_channels
        + config.conv2_weight_len()
        + config.conv2_out_channels.unwrap_or(0)
        + config.num_classes * config.flat_dim()
        + config.num_classes
}

pub const CIFAR10_CLASS_NAMES: [&str; 10] = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
];

#[cfg(test)]
mod tests {
    use super::router;
    use axum::{
        body::Body,
        http::{Request, StatusCode},
    };
    use tower::ServiceExt;

    #[tokio::test]
    async fn get_cpu_info_returns_json() {
        let response = router()
            .oneshot(
                Request::builder()
                    .uri("/api/system/cpu")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn get_model_returns_json() {
        let response = router()
            .oneshot(
                Request::builder()
                    .uri("/api/model")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }
}
