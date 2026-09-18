use lambda_http::{http::Request, Body};
use serde_json::json;
use std::sync::Arc;

use inference_engine::{function_handler, inference::InferenceModel, types::InferenceResponse};

static MODEL_BYTES: &[u8] = include_bytes!("../../model_pipeline/model.onnx");

fn setup_test_model() -> Arc<InferenceModel> {
    Arc::new(InferenceModel::load(MODEL_BYTES).expect("Failed to load embedded test model"))
}

#[tokio::test]
async fn test_valid_inference_request() {
    let model = setup_test_model();

    // Construct valid HTTP POST payload matching a feature vector
    let payload = json!({
        "features": [0.25, -1.10, 0.85, 0.42]
    });

    let request = Request::builder()
        .method("POST")
        .header("content-type", "application/json")
        .body(Body::Text(payload.to_string()))
        .unwrap();

    let response = function_handler(model, request).await.unwrap();
    assert_eq!(response.status(), 200);

    let body_bytes = response.body().to_vec();
    let response_data: InferenceResponse = serde_json::from_slice(&body_bytes).unwrap();

    // Confirm outputs contain expected prediction label and 2 class probabilities
    assert!(response_data.prediction == 0 || response_data.prediction == 1);
    assert_eq!(response_data.probabilities.len(), 2);
}

#[tokio::test]
async fn test_invalid_json_payload() {
    let model = setup_test_model();

    let request = Request::builder()
        .method("POST")
        .header("content-type", "application/json")
        .body(Body::Text("invalid_json_string".to_string()))
        .unwrap();

    let response = function_handler(model, request).await.unwrap();
    assert_eq!(response.status(), 400);
}

#[tokio::test]
async fn test_incorrect_feature_length() {
    let model = setup_test_model();

    // Send 3 features instead of expected 4
    let payload = json!({
        "features": [1.0, 2.0, 3.0]
    });

    let request = Request::builder()
        .method("POST")
        .header("content-type", "application/json")
        .body(Body::Text(payload.to_string()))
        .unwrap();

    let response = function_handler(model, request).await.unwrap();
    assert_eq!(response.status(), 400);
}
