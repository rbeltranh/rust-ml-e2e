use inference_engine::{function_handler, inference::InferenceModel};
use lambda_http::{service_fn, Error, Request};
use std::sync::Arc;

// Embed ONNX model at compile-time (eliminates disk read on cold start)
static MODEL_BYTES: &[u8] = include_bytes!("../../model_pipeline/model.onnx");

#[tokio::main]
async fn main() -> Result<(), Error> {
    // 1. Cold start: Load model ONCE outside the request loop
    let model = InferenceModel::load(MODEL_BYTES)
        .map_err(|e| Error::from(format!("Failed to initialize ONNX model: {e}")))?;
    let shared_model = Arc::new(model);

    // 2. Service loop
    lambda_http::run(service_fn(move |req: Request| {
        let model = Arc::clone(&shared_model);
        async move { function_handler(model, req).await }
    }))
    .await
}
