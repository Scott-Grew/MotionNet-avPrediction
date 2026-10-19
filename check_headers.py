"""Fails when any function or method header spans more than one line; the
Makefile's check target runs it over the whole tree.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

Function = ast.FunctionDef | ast.AsyncFunctionDef


def spans_lines(lines: list[str], function: Function) -> bool:
    """Whether code, not comments, sits between the def line and the body."""
    between = lines[function.lineno : function.body[0].lineno - 1]
    for line in between:
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            return True
    return False


def spanning_headers(path: Path) -> list[str]:
    """Each offending function in the file as 'path:line name'."""
    source = path.read_text()
    lines = source.splitlines()
    tree = ast.parse(source, filename=str(path))
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if spans_lines(lines, node):
            offenders.append(f"{path}:{node.lineno} {node.name}")
    return offenders


def main() -> None:
    offenders = []
    for argument in sys.argv[1:]:
        offenders.extend(spanning_headers(Path(argument)))
    for offender in offenders:
        print(offender)
    if offenders:
        raise SystemExit(f"{len(offenders)} function headers span lines")


if __name__ == "__main__":
    main()
