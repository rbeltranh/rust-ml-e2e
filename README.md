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

## Performance measurements

Measurements captured on 2026-09-18 from the `rust-ml-inference` Lambda in
`us-east-1`. Lambda execution time and cold-start initialization are read from
CloudWatch `REPORT` log entries; end-to-end timings include the public network
round trip from the benchmark client.

| Metric | Result | Target | Source |
| --- | ---: | ---: | --- |
| Lambda execution duration | 1.29–1.42 ms | <20 ms | CloudWatch `REPORT` logs |
| Cold-start initialization | 87.88–122.64 ms | <150 ms | CloudWatch `Init Duration` |
| HTTP median latency (p50) | 106.1 ms | — | `hey` load benchmark |
| HTTP p95 latency | 187.5 ms | — | `hey` load benchmark |
| Throughput | 80.62 requests/s | — | `hey` load benchmark |

The 1,000-request benchmark used concurrency 10. It completed with 999 HTTP
200 responses and one HTTP 429 response; the latter is counted separately from
successful inference responses.

```bash
hey -n 1000 -c 10 \
  -m POST \
  -H 'Content-Type: application/json' \
  -d '{"features":[5.1,3.5,1.4,0.2]}' \
  'https://amyqfw3bcu2wohjbm4c7ge6ywu0gcayt.lambda-url.us-east-1.on.aws/'
```

The handler logs `request_execution_ms` for every request. After deploying the
current code, query that field in the `/aws/lambda/rust-ml-inference`
CloudWatch log group to track request-level execution time.

```bash
aws logs filter-log-events \
  --region us-east-1 \
  --log-group-name /aws/lambda/rust-ml-inference \
  --filter-pattern '"request_execution_ms"'
```
