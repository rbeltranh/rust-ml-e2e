use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Serialize)]
pub struct InferenceRequest {
    pub features: Vec<f32>,
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct InferenceResponse {
    pub prediction: i64,
    pub probabilities: Vec<f32>,
}
