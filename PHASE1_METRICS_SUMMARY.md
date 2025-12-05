# Phase 1: Metrics Infrastructure - Implementation Summary

## Overview

Phase 1 of the telemetry metrics implementation is now complete. This phase establishes the foundational infrastructure for OpenTelemetry metrics in llama-stack, setting up the framework for tracking request-level and token-level metrics.

## What Was Implemented

### 1. New Metrics Module (`src/llama_stack/telemetry/metrics.py`)

Created a centralized metrics module that:
- Initializes an OpenTelemetry Meter for the `llama_stack.inference` scope
- Defines 6 metric instruments (3 request-level, 3 token-level)
- Provides a helper function for creating consistent metric attributes

**Metrics Defined:**

| Metric Name | Type | Description |
|-------------|------|-------------|
| `llama_stack.inference.requests_total` | Counter | Total number of inference requests processed |
| `llama_stack.inference.request_duration_seconds` | Histogram | Duration of inference requests from start to completion |
| `llama_stack.inference.concurrent_requests` | UpDownCounter | Number of concurrent inference requests being processed |
| `llama_stack.inference.tokens_per_second` | Histogram | Token generation rate (total_tokens / duration) |
| `llama_stack.inference.inference_duration_seconds` | Histogram | Time spent in model inference |
| `llama_stack.inference.time_to_first_token_seconds` | Histogram | Time from request start until first token is generated |

### 2. Updated Constants (`src/llama_stack/telemetry/constants.py`)

Added naming constants for all metrics to ensure consistency across the codebase:
- `REQUESTS_TOTAL`
- `REQUEST_DURATION`
- `CONCURRENT_REQUESTS`
- `TOKENS_PER_SECOND`
- `INFERENCE_DURATION`
- `TIME_TO_FIRST_TOKEN`

### 3. Demo Script (`examples/telemetry_metrics_demo.py`)

Created a demonstration script that shows:
- How to import and use the metrics
- How to create metric attributes
- How to record metric values for different scenarios (streaming, non-streaming, errors)
- Example Prometheus queries for analyzing the metrics

## How to Use

### Basic Usage

```python
from llama_stack.telemetry.metrics import (
    requests_total,
    request_duration,
    concurrent_requests,
    tokens_per_second,
    inference_duration,
    time_to_first_token,
    create_metric_attributes,
)

# Create attributes for the request
attributes = create_metric_attributes(
    model="meta-llama/Llama-3.2-3B-Instruct",
    provider="inline::meta-reference",
    endpoint_type="chat_completion",
    stream=True,
    status="success",
)

# Record metrics
requests_total.add(1, attributes)
request_duration.record(0.5, attributes)
concurrent_requests.add(1, attributes)  # Increment
concurrent_requests.add(-1, attributes)  # Decrement
tokens_per_second.record(100.5, attributes)
inference_duration.record(0.45, attributes)
time_to_first_token.record(0.1, attributes)
```

### Running the Demo

```bash
# Basic demo (metrics recorded but not exported)
uv run python examples/telemetry_metrics_demo.py

# With OTEL auto-instrumentation (metrics exported to collector)
export OTEL_EXPORTER_OTLP_ENDPOINT="http://127.0.0.1:4318"
export OTEL_SERVICE_NAME="llama-stack-server"
uv run opentelemetry-instrument python examples/telemetry_metrics_demo.py
```

## Integration Points

The metrics infrastructure is now ready to be integrated into the actual inference pipeline. The next phases will:

**Phase 2: Request-Level Metrics**
- Add metrics tracking to `InferenceRouter.openai_chat_completion()`
- Add metrics tracking to `InferenceRouter.openai_completion()`
- Implement decorator or middleware pattern for automatic tracking

**Phase 3: Token-Level Metrics**
- Extract token counts from response objects
- Calculate tokens/second for non-streaming requests
- Track TTFT and streaming metrics for streaming requests

**Phase 4: Testing & Validation**
- Update existing tests in `tests/integration/telemetry/test_completions.py`
- Add new tests for the new metrics
- Verify export to OTLP collector and Prometheus

## Metric Attributes

All metrics support the following attributes for filtering and grouping:

- `model`: Model identifier (e.g., "meta-llama/Llama-3.2-3B-Instruct")
- `provider`: Provider ID (e.g., "inline::meta-reference")
- `endpoint_type`: Type of endpoint ("chat_completion", "completion", "embeddings")
- `stream`: Whether request is streaming (true/false)
- `status`: Request outcome ("success", "error")

## Prometheus Queries

Once integrated and exported, you can query these metrics in Prometheus:

```promql
# Request rate by model
rate(llama_stack_inference_requests_total[5m])

# 95th percentile request duration
histogram_quantile(0.95, llama_stack_inference_request_duration_seconds_bucket)

# Current concurrent requests
llama_stack_inference_concurrent_requests

# Average tokens per second
rate(llama_stack_inference_tokens_per_second_sum[5m]) /
rate(llama_stack_inference_tokens_per_second_count[5m])

# 99th percentile time to first token
histogram_quantile(0.99, llama_stack_inference_time_to_first_token_seconds_bucket)
```

## Code Quality

All code has been:
- ✅ Linted with `ruff`
- ✅ Formatted with `black`
- ✅ Type hints included
- ✅ Documented with docstrings
- ✅ Tested with demo script

## Files Modified/Created

**Created:**
- `src/llama_stack/telemetry/metrics.py` (new module)
- `examples/telemetry_metrics_demo.py` (demo script)
- `PHASE1_METRICS_SUMMARY.md` (this document)

**Modified:**
- `src/llama_stack/telemetry/constants.py` (added metric name constants)

## Next Steps

To continue with the implementation:

1. **Phase 2**: Implement request-level metrics in the inference router
2. **Phase 3**: Implement token-level metrics for both streaming and non-streaming
3. **Phase 4**: Add comprehensive tests
4. **Phase 5**: Update documentation and deployment guides

## Notes

- The metrics use OpenTelemetry's global MeterProvider, which is configured via OTEL auto-instrumentation or manually via `metrics.set_meter_provider()`
- Metrics are exported via OTLP HTTP protocol to the configured collector endpoint
- The current OTEL collector configuration (`scripts/telemetry/otel-collector-config.yaml`) already supports metrics export to Prometheus
- All metric names follow the `llama_stack.inference.*` pattern for consistency
