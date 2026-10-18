"""Smallest possible live check: can we reach the collection, and are we
authorised to use it?

Prints nothing but shapes and status codes. No credential value, no endpoint
host, ever reaches stdout.

Usage
-----
    set -a; . .exp.env; set +a
    python scripts/probe.py
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from maths_tools.aoss import config  # noqa: E402
from maths_tools.aoss.client import AOSSAccessDenied, AOSSClient, AOSSError  # noqa: E402


def main() -> int:
    try:
        cfg = config.load()
    except config.ConfigError as exc:
        print(f"CONFIG FAIL: {exc}")
        return 2

    shape = cfg.describe()
    print(f"config  region={shape['region']} prefix={shape['index_prefix']} "
          f"access_key_len={shape['access_key_len']} "
          f"has_secret={shape['has_secret']} "
          f"has_session_token={shape['has_session_token']}")
    print(f"host    <{len(cfg.host)} chars, ends .on.aws: "
          f"{cfg.host.endswith('.on.aws')}>")

    client = AOSSClient(cfg, timeout=30, max_retries=1)

    # A request that needs no index and no write permission.
    try:
        info = client.request("GET", "/", ok_status=(200,))
        version = (info or {}).get("version", {}).get("number", "?")
        print(f"REACH   OK  (engine version {version})")
    except AOSSAccessDenied as exc:
        print("REACH   403 ACCESS DENIED")
        print(f"        body: {exc.body[:300]}")
        print("        -> signed as 'aoss'. If the signature is right, the "
              "principal lacks aoss:APIAccessAll in the data access policy.")
        return 3
    except AOSSError as exc:
        print(f"REACH   FAIL {type(exc).__name__}: {str(exc)[:300]}")
        return 4

    return 0


if __name__ == "__main__":
    sys.exit(main())
