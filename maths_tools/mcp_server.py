#!/usr/bin/env python
"""The BO tool, as one MCP endpoint.

One server, because both halves take the same search space -- the zero-shot
recommendation and the next recommendation are two functions of one tool, not
two tools.

    create_search_space      define it once, get a space_id, see what was
                             normalised
    zero_shot_recommendation the opening experiments, before any data exists
    record_observations      evidence comes back
    skip_recommendation      a suggestion nobody ran -- dropped, not guessed
    next_recommendation      what to run next, fitted on the evidence
    get_campaign             full state; this is how you resume at experiment 7
    list_campaigns

Every call may be made two ways:

    stateful   pass campaign_id; state is loaded and saved for you
    stateless  pass `space` and `observed` inline; nothing is stored

Both produce identical recommendations for identical inputs, because the
surrogate is refitted from the evidence on every call. There is no hidden
model state, which is why resume is exact.

stdio JSON-RPC. Nothing may go to stdout except protocol frames.
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from typing import Any, Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(HERE) not in sys.path:
    sys.path.insert(0, os.path.dirname(HERE))

from maths_tools.normalise import normalise_space
from maths_tools.store import CampaignStore, StoreError, default_store, new_id

PROTOCOL = "2024-11-05"
CALL_LOG = os.path.join(HERE, "logs", "mcp_calls.jsonl")

# The v1.5 default is `random`. Every other method is implemented and
# selectable; the benchmark behind them is in MCP_TOOLS.md.
DEFAULT_INIT_METHOD = "random"
DEFAULT_ACQUISITION = "ei"

_store: Optional[CampaignStore] = None


def store() -> CampaignStore:
    global _store
    if _store is None:
        _store = default_store()
    return _store


def log(rec: dict):
    os.makedirs(os.path.dirname(CALL_LOG), exist_ok=True)
    rec["ts"] = time.time()
    with open(CALL_LOG, "a") as fh:
        fh.write(json.dumps(rec, default=str) + "\n")


def err(msg: str):
    print(msg, file=sys.stderr, flush=True)


class ToolError(ValueError):
    """Raised with a message meant for whoever called the tool."""


# ------------------------------------------------------------------ helpers

def _space_from(args: dict) -> tuple:
    """Resolve a search space from space_id, inline space, or a campaign."""
    from maths_tools.space import SearchSpace

    if args.get("campaign_id"):
        camp = _load_campaign(args["campaign_id"])
        rec = store().get("space", camp["space_id"])
        return SearchSpace.from_dict(rec["space"]), rec, camp
    if args.get("space_id"):
        rec = store().get("space", args["space_id"])
        return SearchSpace.from_dict(rec["space"]), rec, None
    if args.get("space"):
        d = args["space"]
        if isinstance(d, str):
            d = json.loads(d)
        return SearchSpace.from_dict(d), {"space_id": None, "space": d}, None
    raise ToolError("provide one of: campaign_id, space_id, or an inline space")


def _load_campaign(cid: str) -> dict:
    try:
        return store().get("campaign", cid)
    except StoreError:
        raise ToolError(f"campaign {cid!r} not found") from None


def _encoder_for(space, mode=None):
    from maths_tools.encode import AUTO, Encoder
    return Encoder(space, mode=mode or AUTO)


def _fit(space, observations: List[dict], surrogate_kind: str, seed: int):
    """Rebuild the model from evidence. No cached state is used, ever."""
    import numpy as np
    from maths_tools.surrogate import AUTO as SUR_AUTO
    from maths_tools.surrogate import build_surrogate

    enc = _encoder_for(space)
    pts = [o["point"] for o in observations]
    ys = np.array([float(o["y"]) for o in observations], dtype=float)
    X = enc.encode(pts, validate=False)
    kw = {"seed": seed} if (surrogate_kind or SUR_AUTO) == "deep" else {}
    sur = build_surrogate(enc, kind=surrogate_kind or SUR_AUTO, **kw)
    info = sur.fit(X, ys)
    return enc, sur, info


def _point_key(p: dict):
    return tuple(sorted((k, round(v, 12) if isinstance(v, float) else v)
                        for k, v in p.items()))


# -------------------------------------------------------------------- tools

def t_create_search_space(a: dict) -> dict:
    """Define the space. Normalises first and reports what it did."""
    from maths_tools.space import SearchSpace, SpaceError

    if not a.get("factors"):
        raise ToolError("factors is required")
    raw = {"factors": a["factors"]}
    if a.get("constraints"):
        raw["constraints"] = a["constraints"]
    if a.get("name"):
        raw["name"] = a["name"]

    normalised, report = normalise_space(raw)

    if report.collisions:
        raise ToolError(
            "normalisation found level(s) that denote the same structure, so "
            "the space would be degenerate. Resolve these before continuing: "
            + json.dumps(report.collisions))

    try:
        space = SearchSpace.from_dict(normalised)
    except SpaceError as e:
        raise ToolError(f"search space is not valid: {e}") from None

    sid = new_id("space")
    store().put("space", sid, {"space_id": sid, "space": space.to_dict(),
                               "normalisation": report.to_dict(),
                               "created": time.time()})
    card = space.cardinality
    return {
        "space_id": sid,
        "normalisation": report.to_dict(),
        "n_factors": len(space),
        "factors": [{"name": f.name, "type": f.type,
                     "n_levels": f.n_levels, "bounds": f.bounds}
                    for f in space.factors],
        "cardinality": card,
        "enumerable": space.is_enumerable,
        "summary": space.summary(),
        "note": ("Normalisation is reported above and nothing was resolved by "
                 "guessing. Any entry under `unresolved` was passed through "
                 "verbatim and needs a human decision."),
    }


def t_get_search_space(a: dict) -> dict:
    if not a.get("space_id"):
        raise ToolError("space_id is required")
    try:
        return store().get("space", a["space_id"])
    except StoreError:
        raise ToolError(f"space {a['space_id']!r} not found") from None


def t_zero_shot(a: dict) -> dict:
    """Opening experiments, before any data exists."""
    from maths_tools.design import initial_design

    space, srec, _ = _space_from(a)
    n = int(a.get("n", 5))
    method = a.get("method") or DEFAULT_INIT_METHOD
    seed = int(a.get("seed", 0))

    d = initial_design(space, n, method=method, seed=seed)
    out = {"picks": d.points, "method": d.method, "seed": seed,
           "n_requested": n, "n_returned": len(d.points), "notes": list(d.notes)}

    if a.get("create_campaign", True) and srec.get("space_id"):
        cid = a.get("campaign_id") or new_id("camp")
        store().put("campaign", cid, {
            "campaign_id": cid, "space_id": srec["space_id"],
            "observations": [], "skipped": [],
            "pending": d.points,
            "config": {"init_method": d.method, "seed": seed,
                       "acquisition": a.get("acquisition", DEFAULT_ACQUISITION),
                       "surrogate": a.get("surrogate")},
            "created": time.time()})
        out["campaign_id"] = cid
    return out


def t_record(a: dict) -> dict:
    """Evidence comes back."""
    cid = a.get("campaign_id")
    if not cid:
        raise ToolError("campaign_id is required")
    camp = _load_campaign(cid)
    space, _, _ = _space_from({"campaign_id": cid})

    obs = a.get("observations") or []
    if not obs:
        raise ToolError("observations is required and must be non-empty")

    seen = {_point_key(o["point"]) for o in camp["observations"]}
    added, rejected = 0, []
    for o in obs:
        pt = {k: v for k, v in o.items() if k != "yield"}
        if "yield" not in o:
            rejected.append({"point": pt, "reason": "no 'yield' field"})
            continue
        y = o["yield"]
        if y is None or (isinstance(y, float) and y != y):
            rejected.append({"point": pt, "reason":
                             "yield is null/NaN. An experiment that was not run "
                             "must go to skip_recommendation, not be recorded "
                             "with a placeholder."})
            continue
        try:
            space.validate_point(pt)
        except Exception as e:                                   # noqa: BLE001
            rejected.append({"point": pt, "reason": str(e)})
            continue
        k = _point_key(pt)
        if k in seen:
            rejected.append({"point": pt, "reason": "already recorded"})
            continue
        seen.add(k)
        # If this was deferred earlier and has now been run, it stops being
        # skipped. The deferral is kept for audit.
        still = [sk for sk in camp["skipped"] if _point_key(sk["point"]) != k]
        if len(still) != len(camp["skipped"]):
            camp.setdefault("deferral_history", []).extend(
                [sk for sk in camp["skipped"] if _point_key(sk["point"]) == k])
            camp["skipped"] = still
        camp["observations"].append({"point": pt, "y": float(y),
                                     "recorded": time.time()})
        added += 1

    camp["pending"] = [p for p in camp.get("pending", [])
                       if _point_key(p) not in seen]
    store().put("campaign", cid, camp)
    ys = [o["y"] for o in camp["observations"]]
    return {"campaign_id": cid, "n_added": added,
            "n_rejected": len(rejected), "rejected": rejected,
            "n_observations": len(ys),
            "best_y": max(ys) if ys else None}


def t_skip(a: dict) -> dict:
    """A suggestion nobody ran. Dropped -- never recorded as a value."""
    cid = a.get("campaign_id")
    if not cid:
        raise ToolError("campaign_id is required")
    camp = _load_campaign(cid)
    picks = a.get("picks") or []
    if not picks:
        raise ToolError("picks is required")
    reason = a.get("reason", "not run")
    obs_keys = {_point_key(o["point"]) for o in camp["observations"]}
    for p in picks:
        if _point_key(p) in obs_keys:
            raise ToolError(f"cannot skip {p}: already recorded as run")
        camp["skipped"].append({"point": p, "reason": reason, "ts": time.time()})
    camp["pending"] = [q for q in camp.get("pending", [])
                       if _point_key(q) not in {_point_key(p) for p in picks}]
    store().put("campaign", cid, camp)
    return {"campaign_id": cid, "n_skipped": len(camp["skipped"]),
            "note": ("Dropped from the training data entirely -- not recorded "
                     "as zero, not filled with a guess, not down-weighted. The "
                     "model is in exactly the state it would be in had the "
                     "experiment never been proposed.\n\n"
                     "These conditions REMAIN IN THE SEARCH SPACE and can be "
                     "recommended again in a later round. Because the model is "
                     "unchanged, the very next call may well propose them "
                     "again -- that is mathematically correct, since no new "
                     "information arrived. Pass exclude_skipped=true to "
                     "next_recommendation if you want them held back.")}


def t_next(a: dict) -> dict:
    """What to run next, fitted on the evidence."""
    from maths_tools.acquisition import next_recommendation

    space, srec, camp = _space_from(a)
    n = int(a.get("n", 1))
    acq = a.get("acquisition") or (camp or {}).get("config", {}).get(
        "acquisition", DEFAULT_ACQUISITION)
    seed = int(a.get("seed", (camp or {}).get("config", {}).get("seed", 0)))
    kind = a.get("surrogate") or (camp or {}).get("config", {}).get("surrogate")

    if camp is not None:
        observations = list(camp["observations"])
        if a.get("observed"):                    # record-then-recommend
            t_record({"campaign_id": camp["campaign_id"],
                      "observations": a["observed"]})
            camp = _load_campaign(camp["campaign_id"])
            observations = list(camp["observations"])
        # A skipped experiment STAYS IN THE SEARCH SPACE and may be
        # recommended again later. It is only kept out of the training data.
        # Confirmed three times in review: "can our tool recommend the same
        # conditions again also in the next titration?" -- yes.
        excluded = [o["point"] for o in observations]
        if a.get("exclude_skipped"):
            excluded += [s["point"] for s in camp.get("skipped", [])]
    else:
        raw = a.get("observed") or []
        if not raw:
            raise ToolError(
                "stateless use needs `observed`; with no evidence at all use "
                "zero_shot_recommendation instead")
        observations = []
        for o in raw:
            pt = {k: v for k, v in o.items() if k != "yield"}
            if "yield" not in o or o["yield"] is None:
                raise ToolError(
                    f"observation {pt} has no yield. An experiment that was "
                    f"never run must be omitted entirely, not sent with a "
                    f"placeholder.")
            space.validate_point(pt)
            observations.append({"point": pt, "y": float(o["yield"])})
        excluded = [o["point"] for o in observations]

    if len(observations) < 2:
        raise ToolError(
            f"need at least 2 observations to fit a model, have "
            f"{len(observations)}. Use zero_shot_recommendation for the opening "
            f"experiments.")

    enc, sur, info = _fit(space, observations, kind, seed)
    rec = next_recommendation(space, enc, sur, n=n, acquisition=acq, seed=seed,
                              observed=excluded, exclude_observed=True)

    out = {"picks": rec.points, "acquisition": rec.acquisition,
           "path": rec.path, "surrogate": sur.kind,
           "n_observations": len(observations),
           # How many candidates were in play and how many were held back.
           # n_excluded counts the experiments already run, plus the skipped
           # ones ONLY when exclude_skipped was requested -- so a caller can
           # confirm from the response alone that a skip did not quietly
           # blacklist anything.
           "n_candidates": rec.n_candidates,
           "n_excluded": rec.n_excluded,
           "excluded_skipped": bool(a.get("exclude_skipped")),
           "best_observed": round(max(o["y"] for o in observations), 4),
           "predicted_mean": [round(v, 3) for v in rec.predicted_mean],
           "predicted_sd": [round(v, 3) for v in rec.predicted_std],
           "fit_failures": info.get("fit_failures", 0),
           "notes": list(rec.notes)}
    if camp is not None:
        camp["pending"] = list(camp.get("pending", [])) + rec.points
        store().put("campaign", camp["campaign_id"], camp)
        out["campaign_id"] = camp["campaign_id"]
    return out


def t_get_campaign(a: dict) -> dict:
    """Full state. Reload this and experiment 7 continues from experiment 6."""
    cid = a.get("campaign_id")
    if not cid:
        raise ToolError("campaign_id is required")
    camp = _load_campaign(cid)
    ys = [o["y"] for o in camp["observations"]]
    best, trace = None, []
    for y in ys:
        best = y if best is None else max(best, y)
        trace.append(best)
    return {**camp, "n_observations": len(ys), "n_skipped": len(camp["skipped"]),
            "best_y": max(ys) if ys else None, "best_so_far_trace": trace,
            "resume_note": ("The model is refitted from these observations on "
                            "every call, so nothing else needs restoring. Call "
                            "next_recommendation with this campaign_id to "
                            "continue.")}


def t_list_campaigns(a: dict) -> dict:
    ids = store().list("campaign")
    out = []
    for cid in ids[-int(a.get("limit", 50)):]:
        try:
            c = store().get("campaign", cid)
        except StoreError:
            continue
        ys = [o["y"] for o in c.get("observations", [])]
        out.append({"campaign_id": cid, "space_id": c.get("space_id"),
                    "n_observations": len(ys), "best_y": max(ys) if ys else None})
    return {"n": len(out), "campaigns": out}


HANDLERS = {
    "create_search_space": t_create_search_space,
    "get_search_space": t_get_search_space,
    "zero_shot_recommendation": t_zero_shot,
    "record_observations": t_record,
    "skip_recommendation": t_skip,
    "next_recommendation": t_next,
    "get_campaign": t_get_campaign,
    "list_campaigns": t_list_campaigns,
}

_FACTOR_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "type": {"type": "string",
                 "enum": ["continuous", "integer", "ordinal", "categorical"]},
        "bounds": {"type": "array", "items": {"type": "number"},
                   "description": "[low, high] for continuous/integer"},
        "levels": {"type": "array",
                   "description": "allowed values for ordinal/categorical"},
        "descriptors": {"type": "object",
                        "description": "{level: [numbers]} e.g. DFT or "
                                       "fingerprints. Without these a "
                                       "categorical factor is one-hot encoded, "
                                       "which is weaker."},
        "transform": {"type": "string", "enum": ["linear", "log"]},
        "unit": {"type": "string"},
    },
    "required": ["name", "type"],
}

TOOLS = [
    {"name": "create_search_space",
     "description": ("Define the search space once and get a space_id. "
                     "Normalises every chemical level to a canonical form "
                     "first and returns exactly what it changed, what it could "
                     "not resolve, and any two levels that turn out to be the "
                     "same structure. Nothing is resolved by guessing."),
     "inputSchema": {"type": "object", "properties": {
         "factors": {"type": "array", "items": _FACTOR_SCHEMA},
         "constraints": {"type": "array", "items": {
             "type": "object",
             "properties": {"weights": {"type": "object"},
                            "rhs": {"type": "number"},
                            "name": {"type": "string"}},
             "required": ["weights", "rhs"]},
             "description": "linear inequalities over numeric factors, in the "
                            "chemist's own units: sum(w_i * x_i) <= rhs"},
         "name": {"type": "string"}},
         "required": ["factors"]}},

    {"name": "get_search_space",
     "description": "Retrieve a stored search space and its normalisation report.",
     "inputSchema": {"type": "object",
                     "properties": {"space_id": {"type": "string"}},
                     "required": ["space_id"]}},

    {"name": "zero_shot_recommendation",
     "description": (
         "The opening experiments, before any results exist. Returns picks and "
         "a campaign_id to carry forward.\n\n"
         "method:\n"
         "  random   (default) uniform over the space\n"
         "  kmedoids groups similar candidates, returns one real member of each\n"
         "  maxmin   each pick furthest from those already chosen\n"
         "  sobol    low-discrepancy, spreads evenly; required when any factor "
         "is continuous\n"),
     "inputSchema": {"type": "object", "properties": {
         "space_id": {"type": "string"},
         "space": {"type": "object", "description": "inline space, for stateless use"},
         "n": {"type": "integer", "default": 5},
         "method": {"type": "string",
                    "enum": ["random", "kmedoids", "maxmin", "sobol"],
                    "default": DEFAULT_INIT_METHOD},
         "seed": {"type": "integer", "default": 0},
         "create_campaign": {"type": "boolean", "default": True}}}},

    {"name": "record_observations",
     "description": ("Record experiments that were actually run, with their "
                     "yields. A result that is null or NaN is refused: an "
                     "experiment that was not run belongs in "
                     "skip_recommendation."),
     "inputSchema": {"type": "object", "properties": {
         "campaign_id": {"type": "string"},
         "observations": {"type": "array", "items": {
             "type": "object",
             "description": "one factor-name:value pair per factor, plus 'yield'"}}},
         "required": ["campaign_id", "observations"]}},

    {"name": "skip_recommendation",
     "description": ("Declare that a recommended experiment was never run. It "
                     "is dropped from the model entirely -- not recorded as "
                     "zero, not filled with a guess, not down-weighted. The "
                     "model ends up exactly as if it had never been proposed. "
                     "The conditions stay in the search space and may be "
                     "recommended again later."),
     "inputSchema": {"type": "object", "properties": {
         "campaign_id": {"type": "string"},
         "picks": {"type": "array", "items": {"type": "object"}},
         "reason": {"type": "string", "default": "not run"}},
         "required": ["campaign_id", "picks"]}},

    {"name": "next_recommendation",
     "description": (
         "What to run next, fitted on the evidence so far.\n\n"
         "Stateful:  pass campaign_id (optionally with `observed` to record "
         "and recommend in one call).\n"
         "Skipped experiments stay in the search space and remain eligible.\n"
         "Stateless: pass `space` and `observed`; nothing is stored.\n\n"
         "acquisition:\n"
         "  ei   (default) expected improvement\n"
         "  qei  sampled equivalent, for batches\n"
         "  qnei for noisy measurements\n"
         "  ts   Thompson sampling, faster on very large spaces\n"),
     "inputSchema": {"type": "object", "properties": {
         "campaign_id": {"type": "string"},
         "space": {"type": "object"},
         "observed": {"type": "array", "items": {"type": "object"},
                      "description": "factor values plus 'yield'"},
         "n": {"type": "integer", "default": 1},
         "acquisition": {"type": "string", "enum": ["ei", "qei", "qnei", "ts"],
                         "default": DEFAULT_ACQUISITION},
         "surrogate": {"type": "string", "enum": ["deep", "mixed", "plain"]},
         "exclude_skipped": {"type": "boolean", "default": False,
                             "description": "By default skipped experiments "
                             "stay eligible and may be recommended again. Set "
                             "true to hold them back for this call."},
         "seed": {"type": "integer", "default": 0}}}},

    {"name": "get_campaign",
     "description": ("Full campaign state: every observation, every skip, and "
                     "the best-so-far trace. This is how a stopped campaign is "
                     "resumed -- reload and continue; the model is refitted "
                     "from the evidence, so nothing else needs restoring."),
     "inputSchema": {"type": "object",
                     "properties": {"campaign_id": {"type": "string"}},
                     "required": ["campaign_id"]}},

    {"name": "list_campaigns",
     "description": "Known campaigns with their progress.",
     "inputSchema": {"type": "object",
                     "properties": {"limit": {"type": "integer", "default": 50}}}},
]


# ----------------------------------------------------------------- protocol

def handle(req: dict) -> Optional[dict]:
    mid, method = req.get("id"), req.get("method")

    if method == "initialize":
        return {"jsonrpc": "2.0", "id": mid, "result": {
            "protocolVersion": PROTOCOL, "capabilities": {"tools": {}},
            "serverInfo": {"name": "beacon-bo", "version": "1.0.0"}}}
    if method in ("notifications/initialized", "initialized"):
        return None
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": mid, "result": {"tools": TOOLS}}
    if method == "ping":
        return {"jsonrpc": "2.0", "id": mid, "result": {}}

    if method == "tools/call":
        p = req.get("params") or {}
        name, args = p.get("name"), (p.get("arguments") or {})
        t0 = time.time()
        if name not in HANDLERS:
            return {"jsonrpc": "2.0", "id": mid, "result": {
                "content": [{"type": "text", "text": f"ERROR unknown tool {name!r}"}],
                "isError": True}}
        try:
            out = HANDLERS[name](args)
            log({"tool": name, "ok": True, "secs": round(time.time() - t0, 3),
                 "args_keys": sorted(args)})
            return {"jsonrpc": "2.0", "id": mid, "result": {
                "content": [{"type": "text", "text": json.dumps(out, default=str)}]}}
        except ToolError as e:
            log({"tool": name, "ok": False, "error": str(e)})
            return {"jsonrpc": "2.0", "id": mid, "result": {
                "content": [{"type": "text", "text": f"ERROR {e}"}],
                "isError": True}}
        except Exception as e:                                   # noqa: BLE001
            err(traceback.format_exc())
            log({"tool": name, "ok": False, "error": f"{type(e).__name__}: {e}"})
            return {"jsonrpc": "2.0", "id": mid, "result": {
                "content": [{"type": "text",
                             "text": f"ERROR {type(e).__name__}: {e}"}],
                "isError": True}}

    if method and method.startswith("notifications/"):
        return None
    return {"jsonrpc": "2.0", "id": mid,
            "error": {"code": -32601, "message": f"method not found: {method}"}}


def main():
    # Resolve the backend before the first request rather than lazily on the
    # first call that touches it, so the selection is logged at startup and
    # a misconfiguration fails here instead of mid-campaign.
    store()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue
        try:
            out = handle(req)
        except Exception as e:                                   # noqa: BLE001
            err(traceback.format_exc())
            out = {"jsonrpc": "2.0", "id": req.get("id"),
                   "error": {"code": -32603, "message": str(e)}}
        if out is not None:
            sys.stdout.write(json.dumps(out, default=str) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
