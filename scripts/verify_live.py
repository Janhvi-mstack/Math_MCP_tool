"""Live round-trip verification against the real AOSS collection.

Checks, in order: write, read back by id, search visibility (and the actual
near-real-time lag), list with pagination, deterministic-id replacement,
strict-mapping rejection, and delete. Writes a frozen JSON result file.

Every document it creates is removed at the end, and every id is prefixed
``verify_`` so a crashed run leaves obvious, sweepable debris.

Usage
-----
    set -a; . <(grep -E '^[A-Z][A-Z0-9_]*=' .exp.env); set +a
    python scripts/verify_live.py
"""
from __future__ import annotations

import json
import os
import sys
import time
import uuid

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from maths_tools.aoss import config  # noqa: E402
from maths_tools.aoss import mappings  # noqa: E402
from maths_tools.aoss.client import AOSSClient, AOSSRequestError  # noqa: E402
from maths_tools.aoss.store import AOSSStore, StoreError  # noqa: E402

RESULTS = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "results", "live_round_trip.json")

SPACE = {
    "space": {"name": "verify", "description": "live round trip",
              "factors": [
                  {"name": "ligand", "kind": "categorical",
                   "levels": ["XPhos", "SPhos"]},
                  {"name": "temp", "kind": "continuous",
                   "low": 60.0, "high": 110.0, "unit": "C"}]},
    "normalisation": {"needs_review": False},
    "created": 1760000000.42,
}


def _campaign(cid: str) -> dict:
    return {
        "campaign_id": cid, "space_id": "verify_space",
        "observations": [
            {"point": {"ligand": "XPhos", "temp": 90.0}, "y": 41.7,
             "recorded": 1760000100.5},
            {"point": {"ligand": "SPhos", "temp": 70.0}, "y": 0,
             "recorded": 1760000200.5}],
        "skipped": [{"point": {"ligand": "XPhos", "temp": 110.0},
                     "reason": "no material that day", "ts": 1760000300.5}],
        "pending": [{"ligand": "SPhos", "temp": 100.0}],
        "config": {"init_method": "random", "seed": 7, "acquisition": "ei"},
        "created": 1760000000.0,
    }


def main() -> int:
    cfg = config.load()
    store = AOSSStore(cfg)
    client = store.client
    run = uuid.uuid4().hex[:8]
    checks, created = [], []

    def record(name, ok, detail=""):
        checks.append({"check": name, "ok": bool(ok), "detail": str(detail)})
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}"
              + (f" -- {detail}" if detail else ""))
        return ok

    print(f"run={run}  region={cfg.region}  prefix={cfg.index_prefix}")
    print()

    try:
        # 1 -- indices present with explicit mappings ----------------------
        state = store.ensure_indices()
        record("indices present with explicit mappings",
               set(state.values()) <= {"created", "exists"}, json.dumps(state))

        # 2 -- write -------------------------------------------------------
        sid = f"verify_space_{run}"
        store.put("space", sid, SPACE)
        created.append(("space", sid))
        record("write a search space", True, sid)

        # 3 -- read back by id ---------------------------------------------
        t0 = time.monotonic()
        got = store.get("space", sid)
        by_id_ms = (time.monotonic() - t0) * 1000
        ok = got.get("space", {}).get("name") == "verify"
        record("read back by id immediately after write", ok,
               f"{by_id_ms:.0f} ms, round-trip intact")

        # 4 -- the actual near-real-time lag -------------------------------
        cid = f"verify_camp_{run}"
        store.put("campaign", cid, _campaign(cid))
        created.append(("campaign", cid))
        lag = store.wait_until_searchable("campaign", cid, timeout=120)
        record("document becomes searchable", lag is not None,
               f"observed NRT lag {lag:.2f} s" if lag is not None
               else "never became searchable within 120 s")

        # 5 -- resume path reads by id, not by search -----------------------
        cid2 = f"verify_resume_{run}"
        store.put("campaign", cid2, _campaign(cid2))
        created.append(("campaign", cid2))
        reread = store.get("campaign", cid2)        # no wait at all
        record("read-your-writes on the resume path",
               reread.get("campaign_id") == cid2,
               "get-by-id returned the campaign with no refresh wait")

        # 6 -- search by the query surface ----------------------------------
        res = client.search(mappings.index_name("observation", cfg.index_prefix),
                            {"size": 10, "query": {"bool": {"filter": [
                                {"term": {"campaign_id": cid}},
                                {"term": {"outcome": "skipped"}}]}}})
        hits = res.get("hits", {}).get("hits", [])
        ok = len(hits) == 1 and "y" not in (hits[0].get("_source") or {})
        record("search by campaign + outcome finds the skip", ok,
               f"{len(hits)} hit(s); skip carries no y value")

        # 7 -- the fan-out ---------------------------------------------------
        res = client.search(mappings.index_name("observation", cfg.index_prefix),
                            {"size": 10,
                             "query": {"term": {"campaign_id": cid}}})
        total = res.get("hits", {}).get("total", {}).get("value", 0)
        record("observations fanned out one document per experiment",
               total == 3, f"{total} documents (2 observed + 1 skipped)")

        # 8 -- deterministic id replaces rather than duplicates --------------
        for _ in range(2):
            store.put("campaign", cid, _campaign(cid))
        store.wait_until_searchable("campaign", cid, timeout=60)
        time.sleep(3)
        res = client.search(mappings.index_name("observation", cfg.index_prefix),
                            {"size": 20,
                             "query": {"term": {"campaign_id": cid}}})
        total = res.get("hits", {}).get("total", {}).get("value", 0)
        record("re-writing a campaign does not duplicate documents",
               total == 3, f"{total} documents after 3 writes")

        # 9 -- list ----------------------------------------------------------
        ids = store.list("campaign")
        record("list returns ids via paginated search", cid in ids,
               f"{len(ids)} campaign id(s) listed")

        # 10 -- strict mapping refuses a surprise field -----------------------
        try:
            client.put_doc(mappings.index_name("campaign", cfg.index_prefix),
                           f"verify_strict_{run}",
                           {"doc_id": "x", "undeclared_field": 1})
            record("strict mapping refuses an undeclared field", False,
                   "AOSS ACCEPTED an undeclared field -- mapping is not strict")
            created.append(("campaign", f"verify_strict_{run}"))
        except AOSSRequestError as exc:
            ok = "strict_dynamic_mapping_exception" in (exc.body or "")
            record("strict mapping refuses an undeclared field", ok,
                   "rejected with strict_dynamic_mapping_exception" if ok
                   else (exc.body or "")[:160])

        # 11 -- delete ---------------------------------------------------------
        store.delete("space", sid)
        created.remove(("space", sid))
        record("delete removes the document",
               not store.exists("space", sid), "get after delete raises")

    finally:
        for kind, key in created:
            try:
                store.delete(kind, key)
            except Exception:                                    # noqa: BLE001
                pass
        # sweep the fanned-out observation documents for this run
        try:
            idx = mappings.index_name("observation", cfg.index_prefix)
            for cid_ in {k for kind, k in created if kind == "campaign"} | {
                    f"verify_camp_{run}", f"verify_resume_{run}"}:
                res = client.search(idx, {"size": 50, "_source": ["doc_id"],
                                          "query": {"term":
                                                    {"campaign_id": cid_}}})
                for hit in res.get("hits", {}).get("hits", []):
                    client.delete_doc(idx, hit["_id"])
        except Exception:                                        # naoqa: BLE001
            pass

    passed = sum(1 for c in checks if c["ok"])
    print()
    print(f"{passed}/{len(checks)} checks passed")

    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    with open(RESULTS, "w") as fh:
        json.dump({"run": run, "region": cfg.region,
                   "index_prefix": cfg.index_prefix,
                   "indices": mappings.all_index_names(cfg.index_prefix),
                   "checks": checks,
                   "passed": passed, "total": len(checks),
                   "recorded_at": time.time()}, fh, indent=2)
    print(f"frozen -> {os.path.relpath(RESULTS)}")
    return 0 if passed == len(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
