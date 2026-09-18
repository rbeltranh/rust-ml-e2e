pub mod inference;
pub mod types;

use std::sync::Arc;

use inference::InferenceModel;
use lambda_http::{Body, Error, Request, Response};
use types::{InferenceRequest, InferenceResponse};

/// Handles an HTTP inference request using the already-initialized model.
pub async fn function_handler(
    model: Arc<InferenceModel>,
    event: Request,
) -> Result<Response<Body>, Error> {
    let payload: InferenceRequest = match serde_json::from_slice(event.body()) {
        Ok(payload) => payload,
        Err(error) => {
            return json_response(
                400,
                serde_json::json!({
                    "error": format!("Invalid JSON request payload: {error}")
                }),
            )
        }
    };

    match model.predict(&payload.features) {
        Ok((prediction, probabilities)) => {
            let response = InferenceResponse {
                prediction,
                probabilities,
            };
            json_response(200, serde_json::to_value(response)?)
        }
        Err(error) => json_response(
            400,
            serde_json::json!({
                "error": format!("Inference error: {error}")
            }),
        ),
    }
}

fn json_response(status: u16, body: serde_json::Value) -> Result<Response<Body>, Error> {
    Ok(Response::builder()
        .status(status)
        .header("content-type", "application/json")
        .body(Body::Text(body.to_string()))?)
}
