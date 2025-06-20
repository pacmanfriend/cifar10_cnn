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
use uuid::Uuid;

pub async fn start_train(
    State(state): State<ApiState>,
    Json(request): Json<TrainRequest>,
) -> impl IntoResponse {
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

    let backend = request.backend.unwrap_or(state.backend().await);
    let data_dir = request.data_dir;
    let job_id = Uuid::new_v4().to_string();

    state
        .insert_job(
            job_id.clone(),
            TrainStatus {
                status: "running".to_string(),
                epoch: 0,
                loss: None,
                accuracy: None,
                error: None,
            },
        )
        .await;

    let job_state = state.clone();
    let job_id_for_task = job_id.clone();
    tokio::task::spawn_blocking(move || {
        let result =
            trainer::train_cifar10(backend.into(), options, std::path::Path::new(&data_dir));

        let status = match result {
            Ok(history) => match history.metrics.last() {
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
            },
            Err(err) => TrainStatus {
                status: "error".to_string(),
                epoch: 0,
                loss: None,
                accuracy: None,
                error: Some(err.to_string()),
            },
        };

        tokio::runtime::Handle::current().block_on(async move {
            job_state.update_job(&job_id_for_task, status).await;
        });
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
