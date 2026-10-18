"""Create the three BEACON indices with their explicit mappings.

Idempotent: an index that already exists is left alone. Run this before the
first write -- AOSS creates an index implicitly on first write and infers the
mapping from that first document, which is the guess these explicit mappings
exist to prevent.

Usage
-----
    set -a; . <(grep -E '^[A-Z][A-Z0-9_]*=' .exp.env); set +a
    python scripts/provision.py [--verify-only]
"""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from maths_tools.aoss import config  # noqa: E402
from maths_tools.aoss import mappings  # noqa: E402
from maths_tools.aoss.client import AOSSAccessDenied, AOSSClient, AOSSError  # noqa: E402


def verify_mapping(client: AOSSClient, index: str, expected: dict) -> list:
    """Compare the live mapping against what we asked for.

    Parameters
    ----------
    client : AOSSClient
        Connected client.
    index : str
        Index name.
    expected : dict
        The creation body that was sent.

    Returns
    -------
    list of str
        Human-readable discrepancies; empty when the mapping is as intended.
    """
    live = client.get_mapping(index).get(index, {}).get("mappings", {})
    want = expected["mappings"]
    problems = []

    if live.get("dynamic") != want.get("dynamic"):
        problems.append(
            f"dynamic is {live.get('dynamic')!r}, expected "
            f"{want.get('dynamic')!r} -- AOSS will GUESS new fields")

    live_props = live.get("properties", {})
    for name, spec in want["properties"].items():
        got = live_props.get(name)
        if got is None:
            if spec.get("enabled") is False:
                continue      # a disabled object may not be echoed back
            problems.append(f"field {name!r} is missing from the live mapping")
            continue
        if spec.get("type") and got.get("type") != spec.get("type"):
            problems.append(
                f"field {name!r} is {got.get('type')!r}, expected "
                f"{spec.get('type')!r}")
    return problems


def main(argv: list) -> int:
    verify_only = "--verify-only" in argv
    cfg = config.load()
    client = AOSSClient(cfg, timeout=60, max_retries=2)
    names = mappings.all_index_names(cfg.index_prefix)

    print(f"region={cfg.region} prefix={cfg.index_prefix}")
    print()

    status = {}
    for kind in sorted(names):
        index = names[kind]
        body = mappings.mapping_for(kind)
        try:
            if client.index_exists(index):
                status[kind] = "exists"
            elif verify_only:
                status[kind] = "ABSENT"
            else:
                client.create_index(index, body)
                status[kind] = "created"
            print(f"  {index:28s} {status[kind]}")
        except AOSSAccessDenied as exc:
            print(f"  {index:28s} 403 ACCESS DENIED")
            print(f"      {exc.body[:300]}")
            print("      -> data access policy must grant aoss:APIAccessAll")
            return 3
        except AOSSError as exc:
            print(f"  {index:28s} FAIL {type(exc).__name__}: {str(exc)[:300]}")
            return 4

    print()
    print("verifying the live mappings are what we asked for, not a guess:")
    bad = 0
    for kind in sorted(names):
        index = names[kind]
        if status[kind] == "ABSENT":
            print(f"  {index:28s} skipped (absent)")
            continue
        problems = verify_mapping(client, index, mappings.mapping_for(kind))
        if problems:
            bad += 1
            print(f"  {index:28s} {len(problems)} PROBLEM(S)")
            for p in problems:
                print(f"      - {p}")
        else:
            print(f"  {index:28s} matches")

    print()
    print("PASS" if bad == 0 else f"FAIL ({bad} index/indices mis-mapped)")
    return 0 if bad == 0 else 5


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
