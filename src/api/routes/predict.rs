use crate::api::{
    image_util::image_bytes_to_tensor,
    types::{ErrorResponse, PredictResponse},
    ApiState, CIFAR10_CLASS_NAMES,
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

    match state.predict_with_scores(&input).await {
        Err(err) => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(ErrorResponse { error: err }),
        )
            .into_response(),
        Ok((predictions, scores)) => {
            let class_id = predictions[0];
            let num_classes = CIFAR10_CLASS_NAMES.len();
            // scores содержит softmax-вероятности для всего батча [N * num_classes];
            // берём первый (и единственный) сэмпл
            let sample_scores: Vec<f32> = scores[..num_classes].to_vec();
            let confidence = sample_scores[class_id];

            Json(PredictResponse {
                class_id,
                class_name: CIFAR10_CLASS_NAMES[class_id].to_string(),
                confidence,
                scores: sample_scores,
            })
            .into_response()
        }
    }
}
