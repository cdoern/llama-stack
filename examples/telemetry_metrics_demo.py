#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Demo script showing how to use the llama-stack telemetry metrics.

This demonstrates the Phase 1 metrics infrastructure that was created,
showing how to:
1. Import the metrics
2. Create metric attributes
3. Record metric values

To run this demo with actual OTEL export:
    export OTEL_EXPORTER_OTLP_ENDPOINT="http://127.0.0.1:4318"
    export OTEL_SERVICE_NAME="llama-stack-metrics-demo"
    uv run opentelemetry-instrument python examples/telemetry_metrics_demo.py
"""

import time

from llama_stack.telemetry.metrics import (
    concurrent_requests,
    create_metric_attributes,
    inference_duration,
    request_duration,
    requests_total,
    time_to_first_token,
    tokens_per_second,
)


def simulate_inference_request(model: str, streaming: bool = False):
    """Simulate an inference request with metrics recording."""
    print(f"\n{'='*60}")
    print(f"Simulating {'streaming' if streaming else 'non-streaming'} request")
    print(f"Model: {model}")
    print(f"{'='*60}")

    # Create metric attributes
    attributes = create_metric_attributes(
        model=model,
        provider="inline::meta-reference",
        endpoint_type="chat_completion",
        stream=streaming,
    )

    # Track concurrent requests
    print("→ Incrementing concurrent requests counter")
    concurrent_requests.add(1, attributes)

    # Simulate request processing
    start_time = time.perf_counter()

    if streaming:
        # Simulate time to first token
        time.sleep(0.1)  # 100ms to first token
        ttft = time.perf_counter() - start_time
        print(f"→ Recording TTFT: {ttft:.3f}s")
        time_to_first_token.record(ttft, attributes)

        # Simulate token generation
        time.sleep(0.4)  # Additional 400ms for remaining tokens
        total_duration = time.perf_counter() - start_time

        # Simulate 50 tokens in total
        token_count = 50
        tps = token_count / total_duration
        print(f"→ Recording tokens/second: {tps:.2f}")
        tokens_per_second.record(tps, attributes)
    else:
        # Simulate non-streaming inference
        time.sleep(0.3)  # 300ms total
        total_duration = time.perf_counter() - start_time

        # Simulate 40 tokens
        token_count = 40
        tps = token_count / total_duration
        print(f"→ Recording tokens/second: {tps:.2f}")
        tokens_per_second.record(tps, attributes)

    # Record inference duration
    print(f"→ Recording inference duration: {total_duration:.3f}s")
    inference_duration.record(total_duration, attributes)

    # Record request duration
    print(f"→ Recording request duration: {total_duration:.3f}s")
    request_duration.record(total_duration, attributes)

    # Track successful completion
    success_attributes = {**attributes, "status": "success"}
    print("→ Incrementing requests_total counter (success)")
    requests_total.add(1, success_attributes)

    # Decrement concurrent requests
    print("→ Decrementing concurrent requests counter")
    concurrent_requests.add(-1, attributes)

    print(f"✓ Request completed in {total_duration:.3f}s")


def simulate_failed_request(model: str):
    """Simulate a failed inference request."""
    print(f"\n{'='*60}")
    print("Simulating FAILED request")
    print(f"Model: {model}")
    print(f"{'='*60}")

    attributes = create_metric_attributes(
        model=model,
        provider="inline::meta-reference",
        endpoint_type="chat_completion",
        stream=False,
    )

    # Track concurrent requests
    print("→ Incrementing concurrent requests counter")
    concurrent_requests.add(1, attributes)

    start_time = time.perf_counter()

    try:
        # Simulate some processing before error
        time.sleep(0.05)
        raise ValueError("Simulated model error")
    except ValueError as e:
        total_duration = time.perf_counter() - start_time

        # Record error metrics
        error_attributes = {**attributes, "status": "error"}
        print(f"→ Recording error: {e}")
        print("→ Incrementing requests_total counter (error)")
        requests_total.add(1, error_attributes)

        # Still record duration for failed requests
        print(f"→ Recording request duration: {total_duration:.3f}s")
        request_duration.record(total_duration, error_attributes)

        print("✗ Request failed")
    finally:
        # Always decrement concurrent requests
        print("→ Decrementing concurrent requests counter")
        concurrent_requests.add(-1, attributes)


def main():
    """Run the metrics demo."""
    print("\n" + "=" * 60)
    print("Llama Stack Telemetry Metrics Demo")
    print("Phase 1: Metrics Infrastructure")
    print("=" * 60)

    print("\nThis demo shows how the new OTEL metrics are recorded.")
    print("The metrics will be exported to the configured OTLP endpoint.")
    print("\nMetrics being demonstrated:")
    print("  - llama_stack.inference.requests_total")
    print("  - llama_stack.inference.request_duration_seconds")
    print("  - llama_stack.inference.concurrent_requests")
    print("  - llama_stack.inference.tokens_per_second")
    print("  - llama_stack.inference.inference_duration_seconds")
    print("  - llama_stack.inference.time_to_first_token_seconds")

    # Simulate various request scenarios
    simulate_inference_request("meta-llama/Llama-3.2-3B-Instruct", streaming=False)
    simulate_inference_request("meta-llama/Llama-3.2-3B-Instruct", streaming=True)
    simulate_inference_request("meta-llama/Llama-3.1-8B-Instruct", streaming=False)
    simulate_failed_request("meta-llama/Llama-3.2-3B-Instruct")

    print("\n" + "=" * 60)
    print("Demo completed!")
    print("=" * 60)
    print("\nMetrics have been recorded and exported (if OTEL is configured).")
    print("Check your OTEL collector/Prometheus for the metrics.")
    print("\nExample Prometheus queries:")
    print('  rate(llama_stack_inference_requests_total[5m])')
    print('  histogram_quantile(0.95, llama_stack_inference_request_duration_seconds)')
    print('  llama_stack_inference_concurrent_requests')
    print('  rate(llama_stack_inference_tokens_per_second_sum[5m]) / rate(llama_stack_inference_tokens_per_second_count[5m])')


if __name__ == "__main__":
    main()
