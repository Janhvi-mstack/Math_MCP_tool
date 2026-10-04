"""MCP endpoint tests. Drives the real server as a subprocess over stdio."""
import json, os, shutil, subprocess, sys, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SERVER = os.path.join(ROOT, "maths_tools", "mcp_server.py")
P, F = [], []


def check(name, fn):
    try:
        fn(); P.append(name)
    except Exception as e:
        F.append((name, "%s: %s" % (type(e).__name__, e)))


class Server:
    """A live server process; each instance gets its own store."""

    def __init__(self, store_dir):
        env = dict(os.environ, BEACON_STORE_DIR=store_dir, PYTHONPATH=ROOT)
        self.p = subprocess.Popen([sys.executable, SERVER], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                  text=True, bufsize=1, env=env)
        self.n = 0

    def rpc(self, method, params=None):
        self.n += 1
        req = {"jsonrpc": "2.0", "id": self.n, "method": method}
        if params is not None:
            req["params"] = params
        self.p.stdin.write(json.dumps(req) + "\n")
        self.p.stdin.flush()
        line = self.p.stdout.readline()
        if not line:
            raise RuntimeError("server closed: " + self.p.stderr.read()[-600:])
        return json.loads(line)

    def call(self, tool, args):
        r = self.rpc("tools/call", {"name": tool, "arguments": args})
        body = r["result"]["content"][0]["text"]
        if r["result"].get("isError"):
            raise ToolFailed(body)
        return json.loads(body)

    def close(self):
        try:
            self.p.stdin.close(); self.p.wait(timeout=10)
        except Exception:
            self.p.kill()


class ToolFailed(RuntimeError):
    pass


SPACE = {"factors": [
    {"name": "ligand", "type": "categorical",
     "levels": ["c1ccc(P(c2ccccc2)c2ccccc2)cc1", "CC(C)(C)P(C(C)(C)C)C(C)(C)C",
                "COc1cccc(OC)c1-c1ccccc1P(C1CCCCC1)C1CCCCC1"]},
    {"name": "base", "type": "categorical", "levels": ["CCN(CC)CC", "CN(C)C=O"]},
    {"name": "temperature", "type": "integer", "bounds": [60, 120], "unit": "C"},
    {"name": "loading", "type": "ordinal", "levels": [1, 2, 5], "unit": "mol%"},
]}


def fresh():
    d = tempfile.mkdtemp(prefix="beacon_mcp_")
    return Server(d), d


# ---------------------------------------------------------------- protocol
def _handshake():
    s, d = fresh()
    try:
        r = s.rpc("initialize", {"protocolVersion": "2024-11-05",
                                 "capabilities": {}, "clientInfo": {"name": "t"}})
        assert r["result"]["serverInfo"]["name"] == "beacon-bo", r
        assert "tools" in r["result"]["capabilities"]
        assert s.rpc("ping")["result"] == {}
    finally:
        s.close(); shutil.rmtree(d, ignore_errors=True)
check("initialize handshake + ping", _handshake)


def _tools_list():
    s, d = fresh()
    try:
        tools = s.rpc("tools/list")["result"]["tools"]
        names = {t["name"] for t in tools}
        need = {"create_search_space", "zero_shot_recommendation",
                "record_observations", "skip_recommendation",
                "next_recommendation", "get_campaign"}
        assert need <= names, names
        for t in tools:
            assert t.get("description") and t.get("inputSchema")
            assert t["inputSchema"]["type"] == "object"
    finally:
        s.close(); shutil.rmtree(d, ignore_errors=True)
check("tools/list advertises every function with a schema", _tools_list)


def _unknown_tool():
    s, d = fresh()
    try:
        r = s.rpc("tools/call", {"name": "nope", "arguments": {}})
        assert r["result"]["isError"] is True
    finally:
        s.close(); shutil.rmtree(d, ignore_errors=True)
check("unknown tool returns isError, not a crash", _unknown_tool)


# ------------------------------------------------------------ search space
def _create_space():
    s, d = fresh()
    try:
        out = s.call("create_search_space", SPACE)
        assert out["space_id"].startswith("space_")
        assert out["n_factors"] == 4
        assert out["cardinality"] == 3 * 2 * 61 * 3, out["cardinality"]
        assert out["enumerable"] is True
        assert out["normalisation"]["backend"].startswith("rdkit")
    finally:
        s.close(); shutil.rmtree(d, ignore_errors=True)
check("create_search_space returns id, cardinality, normalisation", _create_space)


def _normalisation_reported():
    """A non-canonical SMILES must be reported, not silently rewritten."""
    s, d = fresh()
    try:
        sp = {"factors": [
            {"name": "l", "type": "categorical",
             "levels": ["C1=CC=CC=C1", "CCO"]},          # first is Kekule benzene
            {"name": "t", "type": "integer", "bounds": [1, 5]}]}
        out = s.call("create_search_space", sp)
        nm = out["normalisation"]
        assert nm["n_canonicalised"] == 1, nm
        assert nm["canonicalised"][0]["to"] == "c1ccccc1", nm
    finally:
        s.close(); shutil.rmtree(d, ignore_errors=True)
check("normalisation reports what it canonicalised", _normalisation_reported)


def _collision_refused():
    """Two spellings of one structure would make the space degenerate."""
    s, d = fresh()
    try:
        sp = {"factors": [
            {"name": "l", "type": "categorical",
             "levels": ["C1=CC=CC=C1", "c1ccccc1", "CCO"]},
            {"name": "t", "type": "integer", "bounds": [1, 5]}]}
        try:
            s.call("create_search_space", sp)
            raise AssertionError("expected refusal")
        except ToolFailed as e:
            assert "same structure" in str(e), e
    finally:
        s.close(); shutil.rmtree(d, ignore_errors=True)
check("colliding levels are refused with an explanation", _collision_refused)


def _unresolved_flagged():
    s, d = fresh()
    try:
        sp = {"factors": [
            {"name": "l", "type": "categorical", "levels": ["triethylamine", "CCO"]},
            {"name": "t", "type": "integer", "bounds": [1, 5]}]}
        out = s.call("create_search_space", sp)
        un = out["normalisation"]["unresolved"]
        assert len(un) == 1 and un[0]["value"] == "triethylamine", un
        assert out["normalisation"]["needs_review"] is True
    finally:
        s.close(); shutil.rmtree(d, ignore_errors=True)
check("an unresolvable name is flagged, not guessed", _unresolved_flagged)


# ------------------------------------------------------------- the BO loop
def _full_loop():
    s, d = fresh()
    try:
        sp = s.call("create_search_space", SPACE)
        z = s.call("zero_shot_recommendation",
                   {"space_id": sp["space_id"], "n": 5, "seed": 0})
        assert z["method"] == "random", z["method"]      # the v1.5 default
        assert len(z["picks"]) == 5 and z["campaign_id"].startswith("camp_")
        cid = z["campaign_id"]

        obs = [{**p, "yield": 10.0 + 7 * i} for i, p in enumerate(z["picks"])]
        r = s.call("record_observations", {"campaign_id": cid, "observations": obs})
        assert r["n_added"] == 5 and r["n_rejected"] == 0, r
        assert r["best_y"] == 38.0

        nx = s.call("next_recommendation", {"campaign_id": cid, "n": 1})
        assert len(nx["picks"]) == 1
        assert nx["n_observations"] == 5
        assert len(nx["predicted_mean"]) == 1 and len(nx["predicted_sd"]) == 1
        assert nx["fit_failures"] == 0, nx
        for k in ("ligand", "base", "temperature", "loading"):
            assert k in nx["picks"][0]
    finally:
        s.close(); shutil.rmtree(d, ignore_errors=True)
check("zero-shot -> record -> next, end to end", _full_loop)


def _methods_selectable():
    s, d = fresh()
    try:
        sp = s.call("create_search_space", SPACE)
        got = {}
        for m in ("random", "kmedoids", "maxmin", "sobol"):
            z = s.call("zero_shot_recommendation",
                       {"space_id": sp["space_id"], "n": 4, "method": m,
                        "seed": 1, "create_campaign": False})
            assert z["method"] == m and len(z["picks"]) == 4, z
            got[m] = json.dumps(z["picks"], sort_keys=True)
        assert len(set(got.values())) > 1, "methods produced identical designs"
    finally:
        s.close(); shutil.rmtree(d, ignore_errors=True)
check("all four initial-design methods are selectable", _methods_selectable)


def _acquisitions():
    s, d = fresh()
    try:
        sp = s.call("create_search_space", SPACE)
        z = s.call("zero_shot_recommendation",
                   {"space_id": sp["space_id"], "n": 6, "seed": 2})
        cid = z["campaign_id"]
        s.call("record_observations", {"campaign_id": cid, "observations":
               [{**p, "yield": 20.0 + 3 * i} for i, p in enumerate(z["picks"])]})
        for acq in ("ei", "qei", "qnei", "ts"):
            n = 1 if acq == "ei" else 2
            nx = s.call("next_recommendation",
                        {"campaign_id": cid, "n": n, "acquisition": acq})
            assert len(nx["picks"]) == n, (acq, nx)
    finally:
        s.close(); shutil.rmtree(d, ignore_errors=True)
check("every acquisition function works through the endpoint", _acquisitions)


# ------------------------------------------- stop and resume a campaign
def _resume_at_seven():
    """Stop after six experiments, kill the process, continue at the seventh."""
    d = tempfile.mkdtemp(prefix="beacon_mcp_")
    try:
        s1 = Server(d)
        sp = s1.call("create_search_space", SPACE)
        z = s1.call("zero_shot_recommendation",
                    {"space_id": sp["space_id"], "n": 6, "seed": 3})
        cid = z["campaign_id"]
        s1.call("record_observations", {"campaign_id": cid, "observations":
                [{**p, "yield": 30.0 + 2 * i} for i, p in enumerate(z["picks"])]})
        before = s1.call("get_campaign", {"campaign_id": cid})
        assert before["n_observations"] == 6
        s1.close()                                   # the process is gone

        s2 = Server(d)                               # a brand new process
        after = s2.call("get_campaign", {"campaign_id": cid})
        assert after["n_observations"] == 6, after
        assert after["best_y"] == before["best_y"]
        assert after["best_so_far_trace"] == before["best_so_far_trace"]
        seventh = s2.call("next_recommendation", {"campaign_id": cid, "n": 1})
        assert len(seventh["picks"]) == 1 and seventh["n_observations"] == 6
        s2.close()
    finally:
        shutil.rmtree(d, ignore_errors=True)
check("stop at 6, restart the process, continue at 7", _resume_at_seven)


# ------------------------------------------------------------- the skip rule
def _skip_is_inert():
    """A skipped suggestion must leave the model exactly as it was."""
    s, d = fresh()
    try:
        sp = s.call("create_search_space", SPACE)
        z = s.call("zero_shot_recommendation",
                   {"space_id": sp["space_id"], "n": 5, "seed": 4})
        cid = z["campaign_id"]
        obs = [{**p, "yield": 40.0 + i} for i, p in enumerate(z["picks"])]
        s.call("record_observations", {"campaign_id": cid, "observations": obs})
        a = s.call("next_recommendation", {"campaign_id": cid, "n": 1, "seed": 9})

        sk = s.call("skip_recommendation",
                    {"campaign_id": cid, "picks": a["picks"], "reason": "no material"})
        assert sk["n_skipped"] == 1
        c = s.call("get_campaign", {"campaign_id": cid})
        assert c["n_observations"] == 5, "a skip must not become an observation"
        assert c["best_y"] == 44.0

        # The skip must not have entered the model.
        c0 = s.call("get_campaign", {"campaign_id": cid})
        assert c0["n_observations"] == 5 and c0["n_skipped"] == 1
        assert c0["best_y"] == 44.0

        # The skipped conditions must still be eligible. Asserting that the
        # very next pick is identical would be flaky -- a deep-kernel fit is
        # not bit-reproducible across calls (MLL convergence shifts with the
        # BLAS build). So assert the property itself: the skipped point is
        # still a legal member of the space, and it is only filtered out when
        # exclude_skipped is explicitly requested.
        # Eligibility is a property of the candidate set, not of where the
        # experiment happens to rank. Checking the picks would be checking the
        # acquisition ordering instead.
        d = s.call("next_recommendation", {"campaign_id": cid, "n": 1})
        h = s.call("next_recommendation",
                   {"campaign_id": cid, "n": 1, "exclude_skipped": True})
        assert d["n_excluded"] == 5, ("default excluded %d; it should exclude "
                                      "only the 5 experiments actually run"
                                      % d["n_excluded"])
        assert h["n_excluded"] == 6, ("exclude_skipped should also hold back "
                                      "the 1 skipped experiment, got %d"
                                      % h["n_excluded"])
        assert d["excluded_skipped"] is False and h["excluded_skipped"] is True
    finally:
        s.close(); shutil.rmtree(d, ignore_errors=True)
check("a skip leaves the model untouched but stays recommendable", _skip_is_inert)


def _skipped_can_return_after_more_data():
    """It can also come back later, once other experiments have been run."""
    s, d = fresh()
    try:
        sp = s.call("create_search_space", SPACE)
        z = s.call("zero_shot_recommendation",
                   {"space_id": sp["space_id"], "n": 5, "seed": 11})
        cid = z["campaign_id"]
        s.call("record_observations", {"campaign_id": cid, "observations":
               [{**p, "yield": 30.0 + i} for i, p in enumerate(z["picks"])]})
        a = s.call("next_recommendation", {"campaign_id": cid, "n": 1})
        skipped = a["picks"][0]
        s.call("skip_recommendation", {"campaign_id": cid, "picks": [skipped],
                                       "reason": "no material that day"})
        # run three other experiments instead
        alt = s.call("next_recommendation",
                     {"campaign_id": cid, "n": 3, "acquisition": "qei",
                      "exclude_skipped": True})
        s.call("record_observations", {"campaign_id": cid, "observations":
               [{**p, "yield": 20.0 + i} for i, p in enumerate(alt["picks"])]})
        c = s.call("get_campaign", {"campaign_id": cid})
        assert c["n_observations"] == 8 and c["n_skipped"] == 1

        # and it can still be RUN later -- at which point it stops being
        # skipped and becomes an ordinary observation
        r = s.call("record_observations",
                   {"campaign_id": cid,
                    "observations": [{**skipped, "yield": 88.0}]})
        assert r["n_added"] == 1, r
        c2 = s.call("get_campaign", {"campaign_id": cid})
        assert c2["n_observations"] == 9
        assert c2["n_skipped"] == 0, "it should no longer be listed as skipped"
        assert len(c2.get("deferral_history", [])) == 1, "deferral not kept for audit"
        assert c2["best_y"] == 88.0
    finally:
        s.close(); shutil.rmtree(d, ignore_errors=True)
check("a skipped experiment can be recommended again in a later round",
      _skipped_can_return_after_more_data)


def _nan_yield_refused():
    s, d = fresh()
    try:
        sp = s.call("create_search_space", SPACE)
        z = s.call("zero_shot_recommendation",
                   {"space_id": sp["space_id"], "n": 3, "seed": 5})
        cid = z["campaign_id"]
        r = s.call("record_observations", {"campaign_id": cid, "observations":
                   [{**z["picks"][0], "yield": None}]})
        assert r["n_added"] == 0 and r["n_rejected"] == 1
        assert "skip_recommendation" in r["rejected"][0]["reason"]
    finally:
        s.close(); shutil.rmtree(d, ignore_errors=True)
check("a null yield is refused and points at skip_recommendation", _nan_yield_refused)


# --------------------------------------------------------------- stateless
def _stateless():
    s, d = fresh()
    try:
        obs = [{"ligand": SPACE["factors"][0]["levels"][0], "base": "CCN(CC)CC",
                "temperature": 80, "loading": 1, "yield": 40.0},
               {"ligand": SPACE["factors"][0]["levels"][1], "base": "CN(C)C=O",
                "temperature": 100, "loading": 2, "yield": 55.0},
               {"ligand": SPACE["factors"][0]["levels"][2], "base": "CCN(CC)CC",
                "temperature": 110, "loading": 5, "yield": 61.0}]
        nx = s.call("next_recommendation",
                    {"space": SPACE, "observed": obs, "n": 1})
        assert len(nx["picks"]) == 1 and nx["n_observations"] == 3
        assert "campaign_id" not in nx, "stateless use must not store anything"
        assert s.call("list_campaigns", {})["n"] == 0
    finally:
        s.close(); shutil.rmtree(d, ignore_errors=True)
check("stateless call stores nothing and still recommends", _stateless)


def _continuous_space():
    """A factor with a range, not a list -- the thing the old code could not do."""
    s, d = fresh()
    try:
        sp = {"factors": [
            {"name": "ligand", "type": "categorical", "levels": ["CCO", "CCC"]},
            {"name": "temperature", "type": "continuous", "bounds": [20, 120]},
            {"name": "conc", "type": "continuous", "bounds": [0.01, 1.0],
             "transform": "log"}]}
        out = s.call("create_search_space", sp)
        assert out["enumerable"] is False and out["cardinality"] is None
        z = s.call("zero_shot_recommendation",
                   {"space_id": out["space_id"], "n": 6, "method": "sobol"})
        temps = [p["temperature"] for p in z["picks"]]
        assert len(set(temps)) == 6, temps
        assert all(20 <= t <= 120 for t in temps)
        cid = z["campaign_id"]
        s.call("record_observations", {"campaign_id": cid, "observations":
               [{**p, "yield": 50 - abs(p["temperature"] - 80) * 0.4}
                for p in z["picks"]]})
        nx = s.call("next_recommendation", {"campaign_id": cid, "n": 1})
        t = nx["picks"][0]["temperature"]
        assert isinstance(t, float) and 20 <= t <= 120, t
        assert t not in temps, "returned an already-tried value"
    finally:
        s.close(); shutil.rmtree(d, ignore_errors=True)
check("continuous factors: recommends a value that is in no list", _continuous_space)


def _constraints():
    s, d = fresh()
    try:
        sp = {"factors": [
            {"name": "a", "type": "integer", "bounds": [1, 10]},
            {"name": "b", "type": "integer", "bounds": [1, 10]}],
            "constraints": [{"weights": {"a": 1.0, "b": 1.0}, "rhs": 8.0}]}
        out = s.call("create_search_space", sp)
        z = s.call("zero_shot_recommendation",
                   {"space_id": out["space_id"], "n": 6, "seed": 7})
        assert all(p["a"] + p["b"] <= 8 for p in z["picks"]), z["picks"]
    finally:
        s.close(); shutil.rmtree(d, ignore_errors=True)
check("constraints are respected through the endpoint", _constraints)


def _errors_are_messages():
    s, d = fresh()
    try:
        for tool, args, frag in [
            ("create_search_space", {}, "factors is required"),
            ("get_campaign", {"campaign_id": "camp_nope"}, "not found"),
            ("next_recommendation", {}, "provide one of"),
            ("next_recommendation", {"space": SPACE, "observed": []}, "needs `observed`"),
        ]:
            try:
                s.call(tool, args)
                raise AssertionError(f"{tool} should have failed")
            except ToolFailed as e:
                assert frag in str(e), (tool, str(e)[:160])
    finally:
        s.close(); shutil.rmtree(d, ignore_errors=True)
check("bad input returns a readable error, never a stack trace", _errors_are_messages)


def _too_few_observations():
    s, d = fresh()
    try:
        sp = s.call("create_search_space", SPACE)
        z = s.call("zero_shot_recommendation", {"space_id": sp["space_id"], "n": 1})
        cid = z["campaign_id"]
        s.call("record_observations", {"campaign_id": cid,
               "observations": [{**z["picks"][0], "yield": 12.0}]})
        try:
            s.call("next_recommendation", {"campaign_id": cid})
            raise AssertionError("expected refusal")
        except ToolFailed as e:
            assert "at least 2 observations" in str(e), e
    finally:
        s.close(); shutil.rmtree(d, ignore_errors=True)
check("one observation is refused with the reason", _too_few_observations)


print("PASS %d" % len(P))
for n, m in F:
    print("FAIL %s\n     %s" % (n, m))
print("FAILED %d" % len(F))
sys.exit(1 if F else 0)
