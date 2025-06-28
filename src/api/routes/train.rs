use crate::{
    api::{
        types::{ErrorResponse, TrainRequest, TrainStartResponse, TrainStatus},
        ApiState,
    },
    training::trainer,
};
use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
    Json,
};
use std::path::Component;
use uuid::Uuid;

fn validate_data_dir(path: &str) -> Result<(), String> {
    if path.is_empty() {
        return Err("data_dir must not be empty".to_string());
    }
    for component in std::path::Path::new(path).components() {
        if matches!(component, Component::ParentDir) {
            return Err("data_dir must not contain '..' components".to_string());
        }
    }
    Ok(())
}

pub async fn start_train(
    State(state): State<ApiState>,
    Json(request): Json<TrainRequest>,
) -> impl IntoResponse {
    if let Err(err) = validate_data_dir(&request.data_dir) {
        return (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse { error: err }),
        )
            .into_response();
    }

    let mut options = trainer::TrainOptions::cifar10();
    if let Some(epochs) = request.epochs {
        options.epochs = epochs;
    }
    if let Some(learning_rate) = request.learning_rate {
        options.learning_rate = learning_rate;
    }
    if let Some(lr_decay_epochs) = request.lr_decay_epochs {
        options.lr_decay_epochs = lr_decay_epochs;
    }
    if let Some(lr_decay_factor) = request.lr_decay_factor {
        options.lr_decay_factor = lr_decay_factor;
    }
    if let Some(batch_size) = request.batch_size {
        options.batch_size = batch_size;
    }
    if let Some(momentum) = request.momentum {
        options.momentum = momentum;
    }
    
    if let Err(err) = trainer::validate_train_options(&options) {
        return (
            StatusCode::UNPROCESSABLE_ENTITY,
            Json(ErrorResponse { error: err }),
        )
            .into_response();
    }

    let backend = request.backend.unwrap_or(state.backend().await);
    let data_dir = request.data_dir;
    let job_id = Uuid::new_v4().to_string();
    
    let job_arc = state.insert_job(job_id.clone()).await;
    let job_arc_for_progress = job_arc.clone();

    let job_state = state.clone();
    tokio::task::spawn_blocking(move || {
        let result = trainer::train_cifar10_take_net(
            backend.into(),
            options,
            std::path::Path::new(&data_dir),
            Some(Box::new(move |metrics: &trainer::EpochMetrics| {
                if let Ok(mut status) = job_arc_for_progress.lock() {
                    status.epoch = metrics.epoch;
                    status.loss = Some(metrics.train_avg_loss);
                    status.accuracy =
                        Some(metrics.train_correct as f32 / metrics.train_total as f32);
                }
            })),
        );

        let final_status = match result {
            Ok((history, net)) => {
                tokio::runtime::Handle::current().block_on(async {
                    job_state.set_trained_model(net).await;
                });
                match history.metrics.last() {
                    Some(metric) => TrainStatus {
                        status: "done".to_string(),
                        epoch: metric.epoch,
                        loss: Some(metric.train_avg_loss),
                        accuracy: Some(metric.train_correct as f32 / metric.train_total as f32),
                        error: None,
                    },
                    None => TrainStatus {
                        status: "done".to_string(),
                        epoch: 0,
                        loss: None,
                        accuracy: None,
                        error: None,
                    },
                }
            }
            Err(err) => TrainStatus {
                status: "error".to_string(),
                epoch: 0,
                loss: None,
                accuracy: None,
                error: Some(err.to_string()),
            },
        };

        if let Ok(mut status) = job_arc.lock() {
            *status = final_status;
        }
    });

    (StatusCode::ACCEPTED, Json(TrainStartResponse { job_id })).into_response()
}

pub async fn train_status(
    State(state): State<ApiState>,
    Path(job_id): Path<String>,
) -> impl IntoResponse {
    match state.job(&job_id).await {
        Some(status) => (StatusCode::OK, Json(status)).into_response(),
        None => (
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: format!("unknown train job: {job_id}"),
            }),
        )
            .into_response(),
    }
}
