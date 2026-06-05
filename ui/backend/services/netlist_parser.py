"""Lightweight extractor for `.param name=val` lines from a SPICE netlist.

Used by the Setup wizard to pre-fill DUT and testbench parameter rows from an
uploaded netlist. We intentionally avoid a full SPICE parser — only the
`.param` directive form is recognised, which matches the convention used
throughout `examples/`.
"""
from __future__ import annotations

import re
from typing import List, Dict


_PARAM_LINE = re.compile(
    r"""^\s*\.param\s+         # directive
        ([A-Za-z_][A-Za-z0-9_]*) # capture name
        \s*=\s*
        ([^\s*;$]+)              # capture value (stop at whitespace, `*`/`;`/`$` comment markers)
    """,
    re.IGNORECASE | re.VERBOSE,
)


def parse_params(netlist_text: str) -> List[Dict[str, str]]:
    """Return a list of `{name, default_val}` rows in order of first appearance.

    Duplicate names are deduplicated (first occurrence wins) so a testbench
    that overrides DUT defaults doesn't generate phantom rows.
    """
    seen: Dict[str, str] = {}
    for raw in netlist_text.splitlines():
        line = raw.split("//", 1)[0]  # strip `//` comments if any
        m = _PARAM_LINE.match(line)
        if not m:
            continue
        name, val = m.group(1), m.group(2).strip()
        if name not in seen:
            seen[name] = val
    return [{"name": n, "default_val": v} for n, v in seen.items()]
