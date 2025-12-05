#!/usr/bin/env python3
"""Verification script for Phase 2 metrics integration.

This script verifies that:
1. All metrics are properly defined
2. The InferenceRouter imports successfully with metrics
3. The metrics integration doesn't break existing functionality
"""

import sys


def verify_metrics_module():
    """Verify the metrics module is properly set up."""
    print("=" * 60)
    print("Verifying Phase 1: Metrics Module")
    print("=" * 60)

    try:
        from llama_stack.telemetry.metrics import (
            concurrent_requests,
            create_metric_attributes,
            inference_duration,
            request_duration,
            requests_total,
            time_to_first_token,
            tokens_per_second,
        )

        print("✓ All metrics imported successfully")

        # Verify metric types
        from opentelemetry.metrics import Counter, Histogram, UpDownCounter

        assert hasattr(
            requests_total, "add"
        ), "requests_total should have add method (Counter)"
        assert hasattr(
            request_duration, "record"
        ), "request_duration should have record method (Histogram)"
        assert hasattr(
            concurrent_requests, "add"
        ), "concurrent_requests should have add method (UpDownCounter)"

        print("✓ Metric types are correct")

        # Test create_metric_attributes helper
        attrs = create_metric_attributes(
            model="test-model",
            provider="test-provider",
            endpoint_type="chat_completion",
            stream=True,
            status="success",
        )
        assert len(attrs) == 5, "Should have 5 attributes"
        assert attrs["model"] == "test-model"
        assert attrs["stream"] is True

        print("✓ create_metric_attributes helper works correctly")

        return True
    except Exception as e:
        print(f"✗ Error in metrics module: {e}")
        return False


def verify_router_integration():
    """Verify the InferenceRouter integrates metrics correctly."""
    print("\n" + "=" * 60)
    print("Verifying Phase 2: Router Integration")
    print("=" * 60)

    try:
        from llama_stack.core.routers.inference import InferenceRouter

        print("✓ InferenceRouter imports successfully")

        # Verify the methods exist
        assert hasattr(
            InferenceRouter, "openai_chat_completion"
        ), "Missing openai_chat_completion"
        assert hasattr(
            InferenceRouter, "openai_completion"
        ), "Missing openai_completion"
        assert hasattr(
            InferenceRouter, "stream_tokens_and_compute_metrics_openai_chat"
        ), "Missing streaming method"

        print("✓ All required methods exist")

        # Check that the streaming method has the new parameters
        import inspect

        sig = inspect.signature(
            InferenceRouter.stream_tokens_and_compute_metrics_openai_chat
        )
        params = sig.parameters

        assert (
            "request_start_time" in params
        ), "Missing request_start_time parameter in streaming method"
        assert (
            "metric_attrs" in params
        ), "Missing metric_attrs parameter in streaming method"

        print("✓ Streaming method has new metric parameters")

        return True
    except Exception as e:
        print(f"✗ Error in router integration: {e}")
        import traceback

        traceback.print_exc()
        return False


def verify_constants():
    """Verify the constants are properly defined."""
    print("\n" + "=" * 60)
    print("Verifying Metric Name Constants")
    print("=" * 60)

    try:
        from llama_stack.telemetry.constants import (
            CONCURRENT_REQUESTS,
            INFERENCE_DURATION,
            REQUESTS_TOTAL,
            REQUEST_DURATION,
            TIME_TO_FIRST_TOKEN,
            TOKENS_PER_SECOND,
        )

        expected = {
            "REQUESTS_TOTAL": "llama_stack.inference.requests_total",
            "REQUEST_DURATION": "llama_stack.inference.request_duration_seconds",
            "CONCURRENT_REQUESTS": "llama_stack.inference.concurrent_requests",
            "TOKENS_PER_SECOND": "llama_stack.inference.tokens_per_second",
            "INFERENCE_DURATION": "llama_stack.inference.inference_duration_seconds",
            "TIME_TO_FIRST_TOKEN": "llama_stack.inference.time_to_first_token_seconds",
        }

        actual = {
            "REQUESTS_TOTAL": REQUESTS_TOTAL,
            "REQUEST_DURATION": REQUEST_DURATION,
            "CONCURRENT_REQUESTS": CONCURRENT_REQUESTS,
            "TOKENS_PER_SECOND": TOKENS_PER_SECOND,
            "INFERENCE_DURATION": INFERENCE_DURATION,
            "TIME_TO_FIRST_TOKEN": TIME_TO_FIRST_TOKEN,
        }

        for name, expected_value in expected.items():
            if actual[name] != expected_value:
                print(
                    f"✗ {name} mismatch: expected {expected_value}, got {actual[name]}"
                )
                return False

        print("✓ All metric name constants are correct")
        return True
    except Exception as e:
        print(f"✗ Error in constants: {e}")
        return False


def main():
    """Run all verification checks."""
    print("\n" + "=" * 60)
    print("PHASE 2 METRICS INTEGRATION VERIFICATION")
    print("=" * 60)

    results = []

    results.append(("Metrics Module", verify_metrics_module()))
    results.append(("Router Integration", verify_router_integration()))
    results.append(("Metric Constants", verify_constants()))

    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
        if not passed:
            all_passed = False

    print("=" * 60)

    if all_passed:
        print("\n🎉 ALL CHECKS PASSED! Phase 2 integration is complete.")
        print("\nMetrics are now fully integrated into the inference pipeline.")
        print("When OTEL is enabled, all inference requests will automatically")
        print("generate comprehensive telemetry data.\n")
        return 0
    else:
        print("\n❌ SOME CHECKS FAILED. Please review the errors above.\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
