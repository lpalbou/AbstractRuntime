from abstractruntime.integrations.abstractcore.tool_executor import MappingToolExecutor


def test_mapping_tool_executor_sanitizes_wrapper_kwargs_on_typeerror() -> None:
    def add(x: int, y: int) -> int:
        return x + y

    executor = MappingToolExecutor.from_tools([add])

    # Simulate a common model-emitted wrapper payload inside `arguments`.
    result = executor.execute(
        tool_calls=[
            {
                "name": "add",
                "call_id": "c1",
                "arguments": {
                    "name": "add",
                    "arguments": {"x": 2, "y": 3},
                    "overwrite": True,  # stray key should be ignored after sanitization
                },
            }
        ]
    )

    assert result["mode"] == "executed"
    assert result["results"][0]["success"] is True
    assert result["results"][0]["output"] == 5

