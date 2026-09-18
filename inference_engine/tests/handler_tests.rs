use lambda_http::{
    http::{header::CONTENT_TYPE, Request},
    Body,
};
use serde_json::{json, Value};
use std::sync::Arc;

use inference_engine::{function_handler, inference::InferenceModel, types::InferenceResponse};

static MODEL_BYTES: &[u8] = include_bytes!("../../model_pipeline/model.onnx");

fn setup_test_model() -> Arc<InferenceModel> {
    Arc::new(InferenceModel::load(MODEL_BYTES).expect("Failed to load embedded test model"))
}

fn json_request(payload: Value) -> Request<Body> {
    Request::builder()
        .method("POST")
        .header(CONTENT_TYPE, "application/json")
        .body(Body::Text(payload.to_string()))
        .unwrap()
}

fn assert_json_error(response: &lambda_http::Response<Body>) {
    assert_eq!(
        response.headers().get(CONTENT_TYPE).unwrap(),
        "application/json"
    );
    let body: Value = serde_json::from_slice(response.body()).unwrap();
    assert!(body["error"]
        .as_str()
        .is_some_and(|message| !message.is_empty()));
}

#[test]
fn model_loads_embedded_artifact() {
    let _model = setup_test_model();
}

#[test]
fn predict_rejects_invalid_feature_lengths() {
    let model = setup_test_model();

    for features in [vec![], vec![1.0, 2.0, 3.0], vec![1.0, 2.0, 3.0, 4.0, 5.0]] {
        assert!(model.predict(&features).is_err());
    }
}

#[test]
fn predict_returns_valid_probabilities() {
    let model = setup_test_model();
    let (_, probabilities) = model.predict(&[0.25, -1.10, 0.85, 0.42]).unwrap();

    assert_eq!(probabilities.len(), 2);
    assert!(probabilities
        .iter()
        .all(|probability| (0.0..=1.0).contains(probability)));
    assert!((probabilities.iter().sum::<f32>() - 1.0).abs() < 1e-6);
}

#[test]
fn predict_is_deterministic() {
    let model = setup_test_model();
    let features = [0.25, -1.10, 0.85, 0.42];

    assert_eq!(
        model.predict(&features).unwrap(),
        model.predict(&features).unwrap()
    );
}

#[tokio::test]
async fn test_valid_inference_request() {
    let model = setup_test_model();

    // Construct valid HTTP POST payload matching a feature vector
    let payload = json!({
        "features": [0.25, -1.10, 0.85, 0.42]
    });

    let request = json_request(payload);

    let response = function_handler(model, request).await.unwrap();
    assert_eq!(response.status(), 200);
    assert_eq!(
        response.headers().get(CONTENT_TYPE).unwrap(),
        "application/json"
    );

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
    assert_json_error(&response);
}

#[tokio::test]
async fn test_incorrect_feature_length() {
    let model = setup_test_model();

    // Send 3 features instead of expected 4
    let payload = json!({
        "features": [1.0, 2.0, 3.0]
    });

    let request = json_request(payload);

    let response = function_handler(model, request).await.unwrap();
    assert_eq!(response.status(), 400);
    assert_json_error(&response);
}

#[tokio::test]
async fn handler_rejects_missing_features() {
    let response = function_handler(setup_test_model(), json_request(json!({})))
        .await
        .unwrap();

    assert_eq!(response.status(), 400);
    assert_json_error(&response);
}

#[tokio::test]
async fn handler_rejects_non_numeric_features() {
    let response = function_handler(
        setup_test_model(),
        json_request(json!({ "features": [1.0, "bad", 3.0, 4.0] })),
    )
    .await
    .unwrap();

    assert_eq!(response.status(), 400);
    assert_json_error(&response);
}

#[tokio::test]
async fn handler_rejects_null_features() {
    let response = function_handler(
        setup_test_model(),
        json_request(json!({ "features": null })),
    )
    .await
    .unwrap();

    assert_eq!(response.status(), 400);
    assert_json_error(&response);
}

#[tokio::test]
async fn handler_rejects_a_json_array_payload() {
    let response = function_handler(setup_test_model(), json_request(json!([])))
        .await
        .unwrap();

    assert_eq!(response.status(), 400);
    assert_json_error(&response);
}

#[tokio::test]
async fn handler_supports_concurrent_predictions() {
    let model = setup_test_model();
    let request = || json_request(json!({ "features": [0.25, -1.10, 0.85, 0.42] }));
    let (first, second, third, fourth) = tokio::join!(
        function_handler(Arc::clone(&model), request()),
        function_handler(Arc::clone(&model), request()),
        function_handler(Arc::clone(&model), request()),
        function_handler(model, request()),
    );

    for response in [first, second, third, fourth] {
        assert_eq!(response.unwrap().status(), 200);
    }
}
