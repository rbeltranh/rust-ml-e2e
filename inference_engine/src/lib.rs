pub mod inference;
pub mod types;

use std::sync::Arc;
use std::time::Instant;

use inference::InferenceModel;
use lambda_http::{Body, Error, Request, Response};
use types::{InferenceRequest, InferenceResponse};

/// Handles an HTTP inference request using the already-initialized model.
pub async fn function_handler(
    model: Arc<InferenceModel>,
    event: Request,
) -> Result<Response<Body>, Error> {
    let started_at = Instant::now();

    let response = match serde_json::from_slice::<InferenceRequest>(event.body()) {
        Err(error) => json_response(
            400,
            serde_json::json!({
                "error": format!("Invalid JSON request payload: {error}")
            }),
        ),
        Ok(payload) => match model.predict(&payload.features) {
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
        },
    };

    println!(
        "request_execution_ms={:.3}",
        started_at.elapsed().as_secs_f64() * 1_000.0
    );

    response
}

fn json_response(status: u16, body: serde_json::Value) -> Result<Response<Body>, Error> {
    Ok(Response::builder()
        .status(status)
        .header("content-type", "application/json")
        .body(Body::Text(body.to_string()))?)
}
