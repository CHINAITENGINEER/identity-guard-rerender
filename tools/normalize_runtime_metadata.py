"""Remove machine-specific traceback paths from the recovered CPython runtime.

Only code-object filename metadata changes. Executable instructions, constants,
names and the original pyc header are retained. This is not a decompiler.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import marshal
import sys
from pathlib import Path
from types import CodeType


RUNTIME = Path(__file__).resolve().parents[1] / "strict_compositor" / "_render_runtime.cpython-313.pyc"


def normalize(code: CodeType) -> tuple[CodeType, int]:
    constants = []
    count = 1
    for value in code.co_consts:
        if isinstance(value, CodeType):
            value, child_count = normalize(value)
            count += child_count
        constants.append(value)
    result = code.replace(
        co_filename="strict_compositor/render_runtime.py",
        co_consts=tuple(constants),
    )
    assert result.co_code == code.co_code
    return result, count


def main() -> None:
    if sys.version_info[:2] != (3, 13):
        raise SystemExit("Run this metadata utility with CPython 3.13.")
    original = RUNTIME.read_bytes()
    if original[:4] != importlib.util.MAGIC_NUMBER:
        raise SystemExit("Unexpected bytecode magic; refusing to rewrite runtime.")
    code = marshal.loads(original[16:])
    if not isinstance(code, CodeType):
        raise SystemExit("Runtime payload is not a code object.")
    normalized, count = normalize(code)
    released = original[:16] + marshal.dumps(normalized)
    RUNTIME.write_bytes(released)
    print(json.dumps({
        "code_objects": count,
        "original_sha256": hashlib.sha256(original).hexdigest(),
        "release_sha256": hashlib.sha256(released).hexdigest(),
        "change": "traceback filename metadata only",
    }, indent=2))


if __name__ == "__main__":
    main()
