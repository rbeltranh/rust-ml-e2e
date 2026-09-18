# rust-ml-e2e

An end-to-end Rust machine-learning inference service deployed to AWS Lambda.
The service accepts an HTTP JSON payload containing four numeric features, runs
an embedded ONNX model, and returns a predicted class plus its probabilities.

The Lambda function is built for ARM64 with Cargo Lambda and exposed through an
AWS Lambda Function URL. GitHub Actions runs tests, builds the ARM64 release
artifact, and deploys updates to AWS Lambda.

## Test the deployed Lambda

Send a `POST` request with the feature vector:

```bash
curl --silent --show-error --fail-with-body \
  -X POST 'https://amyqfw3bcu2wohjbm4c7ge6ywu0gcayt.lambda-url.us-east-1.on.aws/' \
  -H 'Content-Type: application/json' \
  -d '{"features":[5.1,3.5,1.4,0.2]}'
```

Example response:

```json
{"prediction":0,"probabilities":[0.0011150240898132324,0.9988849759101868]}
```

Opening the Function URL directly in a browser sends an empty `GET` request.
Use a JSON `POST` request as above instead.

## How a request is processed

```text
GitHub Actions ── build ARM64 binary / deploy ──► AWS Lambda
                                                  src/main.rs
                                                       │
                                                       │ Cold start
                                                       ▼
                                      include_bytes!(model.onnx)
                                                       │
                                                       ▼
                                    InferenceModel::load(...)
                                    Model reused through Arc
                                                       │
Client ── POST Function URL ──► lambda_http::run(...) │
  {"features":[5.1,3.5,1.4,0.2]}                      ▼
                                     function_handler(...) in src/lib.rs
                                                       │
                          ┌────────────────────────────┼────────────────────────────┐
                          ▼                            ▼                            ▼
                 Parse JSON with serde        model.predict(features)       Invalid JSON or
                 into InferenceRequest        in src/inference.rs           invalid features
                          │                            │                            │
                          ▼                            ▼                            ▼
                   feature vector         HTTP 200 JSON response       HTTP 400 JSON error
                                               prediction + probabilities
```

`src/main.rs` initializes the Lambda runtime and loads the ONNX model once per
warm Lambda environment. `src/lib.rs` validates each HTTP request and returns
the JSON response. `src/inference.rs` runs model inference.
