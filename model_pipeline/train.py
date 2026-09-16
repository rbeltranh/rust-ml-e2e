import os
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

def main():
    print("1. Generating synthetic dataset (4 features, binary classification)...")
    X, y = make_classification(
        n_samples=1200,
        n_features=4,
        n_informative=3,
        n_redundant=1,
        n_classes=2,
        random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("2. Training RandomForestClassifier...")
    model = RandomForestClassifier(
        n_estimators=50,
        max_depth=5,
        random_state=42
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)
    print(f"   Model Accuracy: {accuracy * 100:.2f}%")

    # Sample input for verification/reference
    sample_input = X_test[0].astype(np.float32)
    sample_pred = model.predict([sample_input])[0]
    print(f"   Sample Input: {sample_input.tolist()}")
    print(f"   Sample Output Prediction: {sample_pred}")

    print("3. Exporting model to ONNX format...")
    # Define input schema for ONNX: Float tensor of shape [batch_size, 4]
    initial_type = [('float_input', FloatTensorType([None, 4]))]
    
    # Convert scikit-learn model to ONNX
    onnx_model = convert_sklearn(
        model, 
        initial_types=initial_type,
        target_opset=15
    )

    # Save artifact
    output_path = os.path.join(os.path.dirname(__file__), "model.onnx")
    with open(output_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    print(f"4. Successfully exported ONNX model to: {output_path}")

if __name__ == "__main__":
    main()