use crate::{
    api::{
        image_util::image_bytes_to_tensor,
        types::{ErrorResponse, PredictResponse},
        ApiState, CIFAR10_CLASS_NAMES,
    },
    compute::random,
    config,
    training::network::Network,
};
use axum::{
    extract::{Multipart, State},
    http::StatusCode,
    response::IntoResponse,
    Json,
};

pub async fn predict(State(state): State<ApiState>, mut multipart: Multipart) -> impl IntoResponse {
    let mut image_bytes = None;

    while let Ok(Some(field)) = multipart.next_field().await {
        if field.name() == Some("image") {
            match field.bytes().await {
                Ok(bytes) => {
                    image_bytes = Some(bytes.to_vec());
                    break;
                }
                Err(err) => {
                    return (
                        StatusCode::BAD_REQUEST,
                        Json(ErrorResponse {
                            error: format!("failed to read image field: {err}"),
                        }),
                    )
                        .into_response()
                }
            }
        }
    }

    let Some(image_bytes) = image_bytes else {
        return (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "multipart field 'image' is required".to_string(),
            }),
        )
            .into_response();
    };

    let input = match image_bytes_to_tensor(&image_bytes) {
        Ok(input) => input,
        Err(err) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: format!("failed to decode image: {err}"),
                }),
            )
                .into_response()
        }
    };

    let backend = state.backend().await;
    let mut rng = random::Rng::new(42);
    let mut network = match Network::new(config::ModelConfig::cifar10(), &mut rng, backend.into()) {
        Ok(network) => network,
        Err(err) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: format!("failed to initialize model: {err}"),
                }),
            )
                .into_response()
        }
    };

    let predictions = match network.predict_batch(&input) {
        Ok(predictions) => predictions,
        Err(err) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: format!("prediction failed: {err}"),
                }),
            )
                .into_response()
        }
    };
    let class_id = predictions[0];
    let mut scores = vec![0.0; CIFAR10_CLASS_NAMES.len()];
    scores[class_id] = 1.0;

    Json(PredictResponse {
        class_id,
        class_name: CIFAR10_CLASS_NAMES[class_id].to_string(),
        confidence: 1.0,
        scores,
    })
    .into_response()
}
