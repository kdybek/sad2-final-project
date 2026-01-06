from __future__ import annotations

import re
from pathlib import Path
from typing import List

from boolean_network import BN


def _normalize_expr(expr: str) -> str:
    e = expr.strip()
    if e.endswith(';'):
        e = e[:-1].rstrip()


    e = re.sub(r"\bAND\b", "&", e, flags=re.IGNORECASE)
    e = re.sub(r"\bOR\b", "|", e, flags=re.IGNORECASE)
    e = re.sub(r"\bNOT\b", "~", e, flags=re.IGNORECASE)
    e = re.sub(r"\bXOR\b", "^", e, flags=re.IGNORECASE)

    e = e.replace("!", "~")
    e = e.replace("+", "|")
    e = e.replace("*", "&")

    e = re.sub(r"\s*([&|~^()])\s*", r" \1 ", e)
    e = re.sub(r"\s+", " ", e).strip()

    if e == "":
        return "0"

    return e


def parse_bnet_content(content: str) -> BN:
    """Parse the content of a .bnet file and return a BN instance.

    The parser accepts lines of the form:
      target, expression
    or
      target = expression

    Lines starting with '#' or '//' (after optional whitespace) are treated as
    comments. Empty lines are ignored.

    First line is also ignored.

    Returns:
      BN: Boolean network constructed from the file content.
    """
    targets: List[str] = []
    factors: List[str] = []

    lines = content.splitlines()
    for raw_line in lines[1:]:
        line = raw_line.strip()
        if not line:
            continue
        # Skip common comment styles
        if line.startswith('#') or line.startswith('%') or line.startswith('//'):
            continue

        left = None
        right = None

        if ',' in line:
            parts = line.split(',', 1)
            left, right = parts[0].strip(), parts[1].strip()
        elif '=' in line:
            parts = line.split('=', 1)
            left, right = parts[0].strip(), parts[1].strip()
        elif ':' in line:
            parts = line.split(':', 1)
            left, right = parts[0].strip(), parts[1].strip()
        else:
            parts = line.split(None, 1)
            if len(parts) == 2:
                left, right = parts[0].strip(), parts[1].strip()

        if left is None or right is None:
            continue

        node = left
        expr = _normalize_expr(right)

        targets.append(node)
        factors.append(expr)

    if not targets:
        raise ValueError("No targets found in provided .bnet content")

    return BN(targets, factors)


def parse_bnet_file(path: str | Path) -> BN:
    """Read a .bnet file from disk and parse it into a BN object.

    Args:
      path: path to the .bnet file.

    Returns:
      BN: parsed Boolean network.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f".bnet file not found: {p}")

    content = p.read_text(encoding='utf-8')
    return parse_bnet_content(content)
