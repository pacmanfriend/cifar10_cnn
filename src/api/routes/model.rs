use crate::api::{
    types::{BackendRequest, ErrorResponse, LoadRequest, ModelInfo, SaveRequest},
    ApiState,
};
use axum::{extract::State, http::StatusCode, response::IntoResponse, Json};

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

pub async fn load_model(Json(request): Json<LoadRequest>) -> impl IntoResponse {
    (
        StatusCode::NOT_IMPLEMENTED,
        Json(ErrorResponse {
            error: format!(
                "checkpoint loading is not implemented yet; requested path: {}",
                request.path
            ),
        }),
    )
}

pub async fn save_model(Json(request): Json<SaveRequest>) -> impl IntoResponse {
    (
        StatusCode::NOT_IMPLEMENTED,
        Json(ErrorResponse {
            error: format!(
                "checkpoint saving is not implemented yet; requested path: {}",
                request.path
            ),
        }),
    )
}
