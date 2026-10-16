"""Imports in the Code-node sandbox: Python's own standard library, except the
modules that reach the machine (processes, files, network, code loading).

One rule, applied twice: validate_code refuses at compile time (early, clear
message), and the sandbox's ``__import__`` refuses at run time (what actually
gates, in both the RestrictedPython lane and the basic fallback lane).
"""

from __future__ import annotations

import pytest

from abstractruntime.visualflow_compiler.visual import code_executor as ce
from abstractruntime.visualflow_compiler.visual.code_executor import (
    CodeExecutionError,
    SANDBOX_DENIED_MODULES,
    create_code_handler,
    get_code_execution_policy,
    sandbox_import_reason,
    validate_code,
)
from abstractruntime.visualflow_compiler.visual.function_library import FunctionLibraryError, compile_function_library

STDLIB_CODE = (
    "import time\n"
    "import json\n"
    "from datetime import datetime\n"
    "import textwrap\n"
    "from urllib.parse import quote\n"
    "from urllib import parse as urlparse\n"
    "import collections, re\n"
    "\n"
    "def transform(_input):\n"
    "    inside = __import_in_body(_input)\n"
    "    return {\n"
    "        'ctime': len(time.ctime()) > 0,\n"
    "        'json': json.dumps({'a': 1}),\n"
    "        'year': datetime(2026, 1, 2).year,\n"
    "        'wrap': textwrap.shorten('a b c d', 7),\n"
    "        'from_pkg': urlparse.quote('c d'),\n"
    "        'quote': quote('a b'),\n"
    "        'counter': dict(collections.Counter('aab')),\n"
    "        're': re.sub('a', 'b', 'aa'),\n"
    "        'inside': inside,\n"
    "    }\n"
    "\n"
    "def __import_in_body(x):\n"
    "    import math\n"
    "    from statistics import mean\n"
    "    return math.floor(mean([1, 2, 4]))\n"
).replace("__import_in_body", "helper")

EXPECTED = {"ctime": True, "json": '{"a": 1}', "year": 2026, "wrap": "a b c d", "from_pkg": "c%20d", "quote": "a%20b", "counter": {"a": 2, "b": 1}, "re": "bb", "inside": 2}


@pytest.fixture(params=["restricted", "basic"])
def lane(request, monkeypatch):
    """Both sandbox lanes: RestrictedPython, and the fallback used without it."""
    if request.param == "restricted":
        pytest.importorskip("RestrictedPython")
    else:
        monkeypatch.setattr(ce, "RESTRICTED_PYTHON_AVAILABLE", False)
    return request.param


def test_standard_library_imports_run_in_the_sandbox(lane) -> None:
    assert create_code_handler(STDLIB_CODE)({"input": None}) == EXPECTED


def test_the_operator_example_import_time_then_ctime(lane) -> None:
    code = "import time; os_time = time.ctime()\n\ndef transform(_input):\n    return str(_input) + ' - ' + os_time[:3]\n"
    out = create_code_handler(code)({"input": None})
    assert out.startswith("{'input': None} - ") and len(out) == len("{'input': None} - ") + 3


@pytest.mark.parametrize(
    "code, fragment",
    [
        ("import os\n", "import of 'os' is not allowed in the sandbox: it reaches processes and the operating system"),
        ("from os import path\n", "import of 'os' is not allowed in the sandbox"),
        ("import os.path\n", "import of 'os.path' is not allowed in the sandbox: it reaches processes"),
        ("import subprocess\n", "it reaches processes and the operating system"),
        ("import pathlib\n", "it reaches files on the machine"),
        ("import io\n", "it reaches files on the machine"),
        ("import socket\n", "it reaches the network"),
        ("import urllib.request\n", "it reaches the network; only urllib.parse is importable"),
        ("import urllib\n", "only urllib.parse is importable"),
        ("from urllib import request\n", "it reaches the network; only urllib.parse is importable"),
        ("from urllib import parse, request\n", "only urllib.parse is importable"),
        ("import operator\n", "it reaches around the sandbox's checks (attributes by name, evaluated annotations, frames)"),
        ("import typing\n", "it reaches around the sandbox's checks"),
        ("import functools\n", "it reaches around the sandbox's checks"),
        ("import traceback\n", "it reaches around the sandbox's checks"),
        ("import xml.etree.ElementTree as ET\n", "import of 'xml.etree.ElementTree' is not allowed in the sandbox: it reaches files on the machine"),
        ("import configparser\n", "it reaches files on the machine"),
        ("import argparse\n", "it reaches files on the machine"),
        ("import importlib\n", "it reaches loading or running code"),
        ("import sys\n", "it reaches loading or running code"),
        ("import pickle\n", "it reaches loading or running code"),
        ("import numpy\n", "import of 'numpy' is not allowed in the sandbox: only Python's own standard library is importable"),
        ("import _io\n", "import of '_io' is not allowed in the sandbox: a private module"),
        ("from json import *\n", "'from json import *' is not allowed in the sandbox"),
        ("from . import x\n", "relative imports are not allowed in the sandbox"),
    ],
)
def test_refused_imports_fail_at_compile_time_with_the_reason(code: str, fragment: str) -> None:
    with pytest.raises(CodeExecutionError, match=fragment.replace("(", "\\(").replace(")", "\\)").replace("*", "\\*").replace(".", "\\.")):
        validate_code(code + "\ndef transform(_input):\n    return 1\n")
    # ...and every refusal names full_access as the way out, except the syntactic ones.
    if "not allowed in the sandbox:" in fragment:
        with pytest.raises(CodeExecutionError, match="full_access lifts this limit"):
            validate_code(code + "\ndef transform(_input):\n    return 1\n")


def test_dynamic_import_call_is_refused_with_the_alternative() -> None:
    with pytest.raises(CodeExecutionError, match="'__import__' is not allowed in the sandbox: write `import <module>` instead"):
        validate_code("def transform(_input):\n    return __import__('time')\n")


def test_the_runtime_guard_applies_the_same_rule(lane) -> None:
    """The AST check can be sidestepped only by code the sandbox itself runs;
    the __import__ bound into the sandbox builtins refuses the same names."""
    assert ce._sandbox_import("json").dumps({"a": 1}) == '{"a": 1}'
    with pytest.raises(CodeExecutionError, match="import of 'subprocess' is not allowed in the sandbox"):
        ce._sandbox_import("subprocess")
    with pytest.raises(CodeExecutionError, match="relative imports are not allowed"):
        ce._sandbox_import("x", None, None, (), 1)
    # `from urllib import parse` reaches the guard as the package + a fromlist: allowed for that one.
    assert ce._sandbox_import("urllib", None, None, ("parse",), 0).parse.quote("a b") == "a%20b"
    with pytest.raises(CodeExecutionError, match="only urllib.parse is importable"):
        ce._sandbox_import("urllib", None, None, ("request",), 0)
    # The handler's builtins carry that guard, not the interpreter's __import__.
    code = "def transform(_input):\n    import json\n    return json.dumps(_input)\n"
    assert create_code_handler(code)({"k": 1}) == '{"k": 1}'


def test_full_access_imports_anything(monkeypatch) -> None:
    monkeypatch.setenv("ABSTRACTRUNTIME_CODE_FULL_ACCESS", "1")
    code = "import os\n\ndef transform(_input):\n    return os.sep\n"
    assert create_code_handler(code, permissions="full_access")({}) in ("/", "\\")
    # The sandbox still refuses it under the same host policy.
    with pytest.raises(CodeExecutionError, match="import of 'os' is not allowed in the sandbox"):
        create_code_handler(code)


def test_import_rule_reasons() -> None:
    assert sandbox_import_reason("json") is None
    assert sandbox_import_reason("collections.abc") is None
    assert sandbox_import_reason("email.utils") is None
    assert sandbox_import_reason("urllib.parse") is None
    assert sandbox_import_reason("") == "an empty module name"
    assert sandbox_import_reason("os") == "it reaches processes and the operating system"
    assert sandbox_import_reason("tempfile") == "it reaches files on the machine"
    assert sandbox_import_reason("http.client") == "it reaches the network"
    assert sandbox_import_reason("types") == "it reaches loading or running code"
    assert sandbox_import_reason("operator") == "it reaches around the sandbox's checks (attributes by name, evaluated annotations, frames)"
    assert sandbox_import_reason("sched") is None and sandbox_import_reason("platform") is None
    assert "sched_h" not in SANDBOX_DENIED_MODULES and "platform_h" not in SANDBOX_DENIED_MODULES
    assert sandbox_import_reason("requests") == "only Python's own standard library is importable"
    # Every denied module names a reason, and the list is the standard library's (or a
    # historical member: modules removed by newer Pythons stay refused for older ones).
    assert all(SANDBOX_DENIED_MODULES.values())


def test_policy_tells_thin_clients_what_the_sandbox_imports() -> None:
    sandbox = next(m for m in get_code_execution_policy()["modes"] if m["id"] == "sandbox")
    assert sandbox["imports"]["allowed"] == "python standard library"
    assert {"os", "subprocess", "operator", "typing", "functools", "traceback", "xml"} <= set(sandbox["imports"]["denied_modules"])
    assert sandbox["safety"] in ("restricted_python", "basic")
    assert sandbox["imports"]["allowed_submodules"] == ["urllib.parse"]
    assert "standard library is importable" in sandbox["description"]


def test_function_library_shares_the_rule() -> None:
    lib = compile_function_library([{"name": "slug", "code": "def slug(s):\n    import re\n    return re.sub('[^a-z]+', '-', s.lower())\n"}])
    assert lib["slug"]("Hello World") == "hello-world"
    with pytest.raises(FunctionLibraryError, match="import of 'os' is not allowed in the sandbox"):
        compile_function_library([{"name": "evil", "code": "def evil():\n    import os\n    return os.getcwd()\n"}])
