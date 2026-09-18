use std::sync::Arc;
use tract_onnx::{prelude::*, tract_core::plan::SimplePlan};

type SimpleModel = SimplePlan<TypedFact, Box<dyn TypedOp>>;

#[derive(Clone)]
pub struct InferenceModel {
    plan: Arc<SimpleModel>,
}

impl InferenceModel {
    /// Loads and optimizes the ONNX model graph once during initialization
    pub fn load(onnx_bytes: &[u8]) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let mut cursor = std::io::Cursor::new(onnx_bytes);

        let model = tract_onnx::onnx()
            .model_for_read(&mut cursor)?
            .with_input_fact(0, InferenceFact::dt_shape(f32::datum_type(), [1, 4]))?
            .into_optimized()?
            .into_runnable()?;

        Ok(Self { plan: model })
    }

    /// Runs tensor prediction on an incoming feature slice
    pub fn predict(
        &self,
        features: &[f32],
    ) -> Result<(i64, Vec<f32>), Box<dyn std::error::Error + Send + Sync>> {
        if features.len() != 4 {
            return Err(format!("Expected 4 features, got {}", features.len()).into());
        }

        // Convert slice to tract 2D tensor [1, 4]
        let input_matrix = tract_ndarray::Array2::from_shape_vec((1, 4), features.to_vec())?;
        let tensor = Tensor::from(input_matrix);

        // Execute inference graph
        let outputs = self.plan.run(tvec!(tensor.into()))?;

        // Output 0: predicted class label (int64)
        let prediction = outputs[0].to_plain_array_view::<i64>()?[0];

        // Output 1: probability distribution tensor [1, 2]
        let probabilities = outputs[1]
            .to_plain_array_view::<f32>()?
            .iter()
            .copied()
            .collect::<Vec<f32>>();

        Ok((prediction, probabilities))
    }
}
