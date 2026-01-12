def test_visualflow_add_message_builtin_builds_message_with_timestamp_and_id():
    from abstractruntime.visualflow_compiler.visual.builtins import get_builtin_handler

    add_message = get_builtin_handler("add_message")
    assert add_message is not None

    out = add_message({"role": "user", "content": "hello"})
    assert isinstance(out, dict)

    msg = out.get("message")
    assert isinstance(msg, dict)
    assert msg.get("role") == "user"
    assert msg.get("content") == "hello"

    ts = msg.get("timestamp")
    assert isinstance(ts, str)
    assert ts

    meta = msg.get("metadata")
    assert isinstance(meta, dict)
    mid = meta.get("message_id")
    assert isinstance(mid, str)
    assert mid.startswith("msg_")

