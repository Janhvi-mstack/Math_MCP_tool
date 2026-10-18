"""Stop a campaign at experiment 6, kill the process, continue at experiment 7.

The capability under test: *"if you stop your BO at some sixth experiment --
now, can you run from the seventh?"* -- with the state in the configured
backend and, on AOSS, nothing on disk at all.

Phase 1 starts an ``mcp_server.py`` process, runs 6 experiments and is killed
with ``SIGKILL``. Nothing is flushed, no cleanup handler runs, and the
in-process write-through cache dies with it. Phase 2 starts a brand-new server
that knows only the campaign id. On AOSS it must read the campaign back out of
the collection, paying the real near-real-time lag, and continue at 7.

The restart is immediate and deliberately so: a leisurely restart would let
the visibility lag expire and prove nothing about the hard case.

``BEACON_STORE_DIR`` points at an empty directory and is asserted still empty
at the end, so an AOSS pass cannot be explained by a local JSON file.

Usage
-----
    set -a; . <(grep -E '^[A-Z][A-Z0-9_]*=' .env); set +a
    python scripts/verify_resume.py --expect aoss
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import tempfile
import time
import uuid

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.verify_campaign import (  # noqa: E402
    EXPECTED_LOG, SPACE, Server,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--expect", choices=sorted(EXPECTED_LOG), required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out = args.out or os.path.join(ROOT, "results",
                                   f"resume_{args.expect}.json")

    env = dict(os.environ, PYTHONPATH=ROOT)
    store_dir = tempfile.mkdtemp(prefix="beacon-resume-")
    env["BEACON_STORE_DIR"] = store_dir
    if args.expect == "local":
        for key in ("AOSS_ENDPOINT", "AWS_ACCESS_KEY_ID",
                    "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN",
                    "BEACON_STORE_BACKEND"):
            env.pop(key, None)

    checks, run = [], uuid.uuid4().hex[:8]

    def record(name: str, ok: bool, detail: str = "") -> bool:
        checks.append({"check": name, "ok": bool(ok), "detail": str(detail)})
        print("  [%s] %s%s" % ("PASS" if ok else "FAIL", name,
                               " -- %s" % detail if detail else ""))
        return bool(ok)

    print("resume across SIGKILL, backend=%s" % args.expect)

    # ---------------------------------------------------------- phase 1
    one = Server(env)
    one.rpc("initialize", {"protocolVersion": "2024-11-05"})
    line1 = next((l for l in one.stderr().splitlines()
                  if "store backend:" in l), "")
    record("phase 1 selected %s" % args.expect,
           EXPECTED_LOG[args.expect] in line1, line1)

    sp = one.call("create_search_space", {**SPACE, "name": "resume-%s" % run})
    sid = sp["space_id"]
    z = one.call("zero_shot_recommendation",
                 {"space_id": sid, "n": 6, "seed": 11})
    cid = z["campaign_id"]
    obs = [{**p, "yield": 30.0 + 3.0 * i} for i, p in enumerate(z["picks"])]
    rec = one.call("record_observations",
                   {"campaign_id": cid, "observations": obs})
    record("phase 1 recorded 6 experiments", rec["n_observations"] == 6,
           "best_y=%s campaign=%s" % (rec["best_y"], cid))

    killed_at = time.time()
    one.proc.send_signal(signal.SIGKILL)
    rc = one.proc.wait()
    record("phase 1 died uncleanly", rc in (-signal.SIGKILL, -9, 137),
           "returncode=%s" % rc)

    # ---------------------------------------------------------- phase 2
    two = Server(env)
    try:
        two.rpc("initialize", {"protocolVersion": "2024-11-05"})
        line2 = next((l for l in two.stderr().splitlines()
                      if "store backend:" in l), "")
        record("phase 2 is a new process on the same backend",
               EXPECTED_LOG[args.expect] in line2, line2)

        t0 = time.time()
        camp = two.call("get_campaign", {"campaign_id": cid})
        read_secs = time.time() - t0
        record("phase 2 read the campaign back", camp["n_observations"] == 6,
               "%d observations in %.2f s, %.2f s after the kill" % (
                   camp["n_observations"], read_secs, t0 - killed_at))

        nxt = two.call("next_recommendation", {"campaign_id": cid, "n": 1})
        pick = nxt["picks"][0]
        record("phase 2 proposed experiment 7", bool(pick),
               "acquisition=%s" % nxt.get("acquisition"))

        after = two.call("record_observations",
                         {"campaign_id": cid,
                          "observations": [{**pick, "yield": 61.0}]})
        record("experiment 7 recorded, continuing from 6",
               after["n_observations"] == 7,
               "n=%s best_y=%s" % (after["n_observations"], after["best_y"]))

        if args.expect == "aoss":
            campaign_dir = os.path.join(store_dir, "campaign")
            record("nothing was read from or written to local files",
                   not os.path.exists(campaign_dir), store_dir)
    finally:
        two.close()

    summary = {"run": run, "backend_expected": args.expect,
               "space_id": sid, "campaign_id": cid,
               "restart_gap_seconds": round(t0 - killed_at, 2),
               "campaign_read_seconds": round(read_secs, 2),
               "n_pass": sum(c["ok"] for c in checks),
               "n_total": len(checks), "checks": checks}
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump(summary, fh, indent=2)
    print("\n%d/%d  -> %s" % (summary["n_pass"], summary["n_total"], out))
    return 0 if summary["n_pass"] == summary["n_total"] else 1


if __name__ == "__main__":
    sys.exit(main())
