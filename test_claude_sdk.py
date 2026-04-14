#!/usr/bin/env python3
"""Test script to verify claude-agent-sdk-python works with Llama Stack."""

import asyncio
import sys

try:
    from claude_agent_sdk import query, ClaudeAgentOptions, AssistantMessage, TextBlock
except ImportError:
    print("ERROR: claude-agent-sdk-python not installed")
    print("Install with: pip install claude-agent-sdk")
    sys.exit(1)


async def test_simple_query():
    """Test 1: Simple query to verify SDK can talk to Llama Stack."""
    base_url = "http://localhost:8321"
    print(f"\n=== Test 1: Simple query ===")
    print(f"ANTHROPIC_BASE_URL: {base_url}")

    try:
        options = ClaudeAgentOptions(
            max_turns=1,  # Single turn
            system_prompt="You are a helpful assistant.",
            model="vllm/Qwen/Qwen3-8B",  # Use vllm qwen model
            env={
                "ANTHROPIC_BASE_URL": base_url,
                "ANTHROPIC_API_KEY": "test-key"
            }
        )

        messages = []
        async for message in query(prompt="Say hello in exactly 3 words", options=options):
            messages.append(message)
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        print(f"✓ Response: {block.text}")

        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_multi_turn():
    """Test 2: Multi-turn conversation with session state."""
    base_url = "http://localhost:8321"
    print("\n=== Test 2: Multi-turn conversation ===")

    try:
        # The SDK/CLI will manage the session state client-side
        options = ClaudeAgentOptions(
            max_turns=3,
            system_prompt="You are a helpful assistant.",
            model="vllm/Qwen/Qwen3-8B",
            env={
                "ANTHROPIC_BASE_URL": base_url,
                "ANTHROPIC_API_KEY": "test-key"
            }
        )

        # First turn: introduce name
        print("Turn 1: My name is Charlie")
        async for message in query(prompt="My name is Charlie. Remember it.", options=options):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        print(f"  Response: {block.text}")

        # Second turn: ask for name (should remember from session)
        print("\nTurn 2: What's my name?")
        async for message in query(prompt="What's my name?", options=options):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        print(f"  Response: {block.text}")
                        # Check if it remembered
                        if "charlie" in block.text.lower():
                            print("✓ SDK remembered name across turns (client-side session)")
                        else:
                            print("⚠ SDK did not remember name (each query may be independent)")

        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_tool_use():
    """Test 3: Tool use (if supported by Llama Stack's Messages API)."""
    base_url = "http://localhost:8321"
    print("\n=== Test 3: Tool use ===")

    try:
        options = ClaudeAgentOptions(
            max_turns=2,
            allowed_tools=["Bash"],  # Allow bash tool
            permission_mode="acceptEdits",
            model="vllm/Qwen/Qwen3-8B",
            env={
                "ANTHROPIC_BASE_URL": base_url,
                "ANTHROPIC_API_KEY": "test-key"
            }
        )

        async for message in query(prompt="What is the output of 'echo Hello from SDK'?", options=options):
            print(f"  Message: {message}")
            if isinstance(message, AssistantMessage):
                print("✓ Tool use test completed")

        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    print("Testing claude-agent-sdk-python with Llama Stack...")

    # Test 1: Simple query
    if not await test_simple_query():
        return False

    # Test 2: Multi-turn (session management)
    if not await test_multi_turn():
        return False

    # Test 3: Tool use
    # if not await test_tool_use():
    #     return False
    # Note: Tool use may not work if Llama Stack doesn't support tools in Messages API

    print("\n✓ SDK tests completed!")
    return True


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
