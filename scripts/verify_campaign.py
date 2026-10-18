"""Drive a complete campaign through the real MCP endpoint, on either backend.

This talks to ``maths_tools/mcp_server.py`` as a subprocess over stdio
JSON-RPC -- the way an agent talks to it -- rather than importing the handlers.
That is deliberate: it is the only way to check the thing most likely to go
wrong silently, which is *which backend the server chose*. The startup line on
stderr is asserted, not assumed.

Sequence: create_search_space -> zero_shot_recommendation ->
record_observations -> skip_recommendation -> next_recommendation ->
get_campaign -> list_campaigns.

Local JSON (no AWS variables in the environment)::

    python scripts/verify_campaign.py --expect local

AOSS::

    set -a; . <(grep -E '^[A-Z][A-Z0-9_]*=' .env); set +a
    python scripts/verify_campaign.py --expect aoss
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import uuid

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

SERVER = os.path.join(ROOT, "maths_tools", "mcp_server.py")

#: A small Buchwald-Hartwig-shaped space: every factor kind the tool supports.
SPACE = {"factors": [
    {"name": "ligand", "type": "categorical",
     "levels": ["c1ccc(P(c2ccccc2)c2ccccc2)cc1",
                "CC(C)(C)P(C(C)(C)C)C(C)(C)C",
                "COc1cccc(OC)c1-c1ccccc1P(C1CCCCC1)C1CCCCC1"]},
    {"name": "base", "type": "categorical", "levels": ["CCN(CC)CC", "CN(C)C=O"]},
    {"name": "temperature", "type": "integer", "bounds": [60, 120], "unit": "C"},
    {"name": "loading", "type": "ordinal", "levels": [1, 2, 5], "unit": "mol%"},
]}

#: What the startup line must contain for each backend.
EXPECTED_LOG = {"local": "local JSON", "aoss": "aoss"}


class ToolFailed(RuntimeError):
    """The endpoint returned isError."""


class Server:
    """A live ``mcp_server.py`` process speaking stdio JSON-RPC.

    stderr is redirected to a file rather than a pipe: the startup log is
    written before the first request, and a pipe nobody drains can fill and
    block the server.
    """

    def __init__(self, env: dict) -> None:
        self._err = tempfile.NamedTemporaryFile(
            mode="w+", suffix=".stderr", delete=False)
        self.proc = subprocess.Popen(
            [sys.executable, SERVER], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=self._err, text=True, bufsize=1,
            env=env)
        self.n = 0

    def rpc(self, method: str, params=None) -> dict:
        self.n += 1
        req = {"jsonrpc": "2.0", "id": self.n, "method": method}
        if params is not None:
            req["params"] = params
        self.proc.stdin.write(json.dumps(req) + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError("server closed the stream\n" + self.stderr())
        return json.loads(line)

    def call(self, name: str, args: dict) -> dict:
        res = self.rpc("tools/call", {"name": name, "arguments": args})
        payload = res["result"]["content"][0]["text"]
        if res["result"].get("isError"):
            raise ToolFailed(payload)
        return json.loads(payload)

    def stderr(self) -> str:
        self._err.flush()
        with open(self._err.name) as fh:
            return fh.read()

    def close(self) -> None:
        try:
            self.proc.stdin.close()
            self.proc.wait(timeout=10)
        except Exception:                                        # noqa: BLE001
            self.proc.kill()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--expect", choices=sorted(EXPECTED_LOG), required=True,
                    help="which backend the server is expected to select")
    ap.add_argument("--out", default=None, help="where to write the result JSON")
    args = ap.parse_args()

    out = args.out or os.path.join(
        ROOT, "results", f"campaign_{args.expect}.json")

    env = dict(os.environ, PYTHONPATH=ROOT)
    store_dir = tempfile.mkdtemp(prefix="beacon-verify-")
    env["BEACON_STORE_DIR"] = store_dir
    if args.expect == "local":
        # a local run must be local because nothing is configured, not because
        # an override hid a misconfiguration
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

    print("campaign through the MCP endpoint, expecting backend=%s" % args.expect)
    t0 = time.time()
    server = Server(env)
    try:
        init = server.rpc("initialize", {"protocolVersion": "2024-11-05"})
        record("server initialised", "result" in init,
               init.get("result", {}).get("protocolVersion", ""))

        # -- the startup line. This is the whole point of driving the server.
        log = server.stderr()
        want = EXPECTED_LOG[args.expect]
        backend_line = next((l for l in log.splitlines()
                             if "store backend:" in l), "")
        record("startup log names a backend", bool(backend_line), backend_line)
        record("startup log says %r" % want, want in backend_line,
               backend_line or log[-300:])
        other = EXPECTED_LOG["aoss" if args.expect == "local" else "local"]
        record("startup log does not name the other backend",
               other not in backend_line, backend_line)

        # -- the eight functions
        sp = server.call("create_search_space",
                         {**SPACE, "name": "verify-%s" % run})
        sid = sp["space_id"]
        record("create_search_space", bool(sid),
               "space_id ok, cardinality=%s" % sp.get("cardinality"))

        got = server.call("get_search_space", {"space_id": sid})
        record("get_search_space round-trips", got.get("space_id") == sid, sid)

        z = server.call("zero_shot_recommendation",
                        {"space_id": sid, "n": 6, "seed": 11})
        cid, picks = z["campaign_id"], z["picks"]
        record("zero_shot_recommendation", len(picks) == 6,
               "%d picks, campaign ok" % len(picks))

        obs = [{**p, "yield": 30.0 + 4.5 * i} for i, p in enumerate(picks[:5])]
        rec = server.call("record_observations",
                          {"campaign_id": cid, "observations": obs})
        record("record_observations", rec["n_observations"] == 5,
               "n=%s best_y=%s rejected=%s" % (rec["n_observations"],
                                               rec["best_y"], rec["n_rejected"]))

        sk = server.call("skip_recommendation",
                         {"campaign_id": cid, "picks": [picks[5]],
                          "reason": "ligand unavailable"})
        record("skip_recommendation", sk["n_skipped"] == 1,
               "n_skipped=%s" % sk["n_skipped"])

        nxt = server.call("next_recommendation",
                          {"campaign_id": cid, "n": 1})
        record("next_recommendation", bool(nxt.get("picks")),
               "acquisition=%s, %s pick(s)" % (nxt.get("acquisition"),
                                               len(nxt.get("picks") or [])))

        camp = server.call("get_campaign", {"campaign_id": cid})
        record("get_campaign sees 5 observed and 1 skipped",
               camp["n_observations"] == 5 and camp["n_skipped"] == 1,
               "observed=%s skipped=%s best_y=%s" % (
                   camp["n_observations"], camp["n_skipped"], camp["best_y"]))

        listed = server.call("list_campaigns", {"limit": 50})
        record("list_campaigns finds it",
               any(c["campaign_id"] == cid for c in listed["campaigns"]),
               "%d campaign(s) listed" % listed["n"])

        if args.expect == "local":
            on_disk = os.path.join(store_dir, "campaign", cid + ".json")
            record("local JSON wrote the campaign to disk",
                   os.path.exists(on_disk), store_dir)
        else:
            empty = not os.path.exists(os.path.join(store_dir, "campaign"))
            record("nothing was written to the local store dir", empty,
                   store_dir)

        # The comparable part. Ids are random per run, so they are reported
        # but excluded from `fingerprint`, which is what two runs are diffed
        # on: seeded picks, the recommendation, and the resulting state.
        fingerprint = {
            "cardinality": sp.get("cardinality"),
            "zero_shot_seed": 11,
            "zero_shot_picks": picks,
            "observed_yields": [o["yield"] for o in obs],
            "best_y": camp["best_y"],
            "n_observations": camp["n_observations"],
            "n_skipped": camp["n_skipped"],
            "acquisition": nxt.get("acquisition"),
            "next_picks": nxt.get("picks"),
        }
        summary = {"run": run, "backend_expected": args.expect,
                   "backend_line": backend_line, "space_id": sid,
                   "campaign_id": cid, "seconds": round(time.time() - t0, 2),
                   "fingerprint": fingerprint,
                   "n_pass": sum(c["ok"] for c in checks),
                   "n_total": len(checks), "checks": checks}
    finally:
        server.close()

    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump(summary, fh, indent=2)
    print("\n%d/%d  -> %s" % (summary["n_pass"], summary["n_total"], out))
    return 0 if summary["n_pass"] == summary["n_total"] else 1


if __name__ == "__main__":
    sys.exit(main())
