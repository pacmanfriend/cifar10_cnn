use crate::api::{
    types::{BackendRequest, ErrorResponse, LoadRequest, ModelInfo, SaveRequest},
    ApiState,
};
use axum::{extract::State, http::StatusCode, response::IntoResponse, Json};
use std::path::{Component, Path};

pub async fn model_info(State(state): State<ApiState>) -> Json<ModelInfo> {
    Json(state.model_info().await)
}

pub async fn set_backend(
    State(state): State<ApiState>,
    Json(request): Json<BackendRequest>,
) -> Json<ModelInfo> {
    state.set_backend(request.backend).await;
    Json(state.model_info().await)
}

pub async fn set_backend_not_allowed() -> impl IntoResponse {
    (
        StatusCode::METHOD_NOT_ALLOWED,
        Json(ErrorResponse {
            error: "use POST /api/model/backend to change backend".to_string(),
        }),
    )
}

pub async fn load_model(
    State(state): State<ApiState>,
    Json(request): Json<LoadRequest>,
) -> impl IntoResponse {
    if let Err(err) = validate_checkpoint_path(&request.path) {
        return (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse { error: err }),
        )
            .into_response();
    }

    match state.load_model_weights(Path::new(&request.path)).await {
        Ok(()) => Json(state.model_info().await).into_response(),
        Err(err) => (
            StatusCode::UNPROCESSABLE_ENTITY,
            Json(ErrorResponse { error: err }),
        )
            .into_response(),
    }
}

pub async fn save_model(
    State(state): State<ApiState>,
    Json(request): Json<SaveRequest>,
) -> impl IntoResponse {
    if let Err(err) = validate_checkpoint_path(&request.path) {
        return (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse { error: err }),
        )
            .into_response();
    }

    match state.save_model_weights(Path::new(&request.path)).await {
        Ok(()) => Json(state.model_info().await).into_response(),
        Err(err) => (
            StatusCode::UNPROCESSABLE_ENTITY,
            Json(ErrorResponse { error: err }),
        )
            .into_response(),
    }
}

fn validate_checkpoint_path(path: &str) -> Result<(), String> {
    if path.is_empty() {
        return Err("path must not be empty".to_string());
    }
    for component in Path::new(path).components() {
        if matches!(component, Component::ParentDir) {
            return Err("path must not contain '..' components".to_string());
        }
    }
    Ok(())
}
