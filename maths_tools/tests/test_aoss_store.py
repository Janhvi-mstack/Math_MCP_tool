"""Offline contract tests for :class:`maths_tools.aoss.store.AOSSStore`.

Everything here runs against :class:`fake_aoss.FakeAOSS`, which enforces
strict dynamic mapping, declared field types and near-real-time lag. No
network, no credentials, no cost.

The parity tests run the identical sequence against ``LocalJSONStore`` and
assert both backends agree, which is the real contract: the eight tool
functions must not be able to tell which one they are talking to.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from maths_tools.aoss import mappings  # noqa: E402
from maths_tools.aoss.client import AOSSAccessDenied, AOSSClient  # noqa: E402
from maths_tools.aoss.config import AOSSConfig  # noqa: E402
from fake_aoss import FakeAOSS  # noqa: E402
from maths_tools.aoss.store import AOSSStore, StoreError, canonical_point_key  # noqa: E402

CFG = AOSSConfig(endpoint="https://c.aoss.us-east-1.on.aws",
                 host="c.aoss.us-east-1.on.aws", region="us-east-1",
                 access_key="AKIA", secret_key="secret",
                 session_token=None, index_prefix="beacon")

SPACE = {
    "space_id": "space_abc123",
    "space": {"name": "buchwald", "description": "test space",
              "factors": [
                  {"name": "ligand", "kind": "categorical",
                   "levels": ["XPhos", "SPhos"], "descriptors": {"XPhos": [1.0]}},
                  {"name": "temp", "kind": "continuous",
                   "low": 60.0, "high": 110.0, "unit": "C"}]},
    "normalisation": {"needs_review": False},
    "created": 1760000000.42,
}


def _store(cache_writes=True, miss_grace=0.0, **kw):
    """A store wired to a fresh :class:`FakeAOSS`.

    ``miss_grace`` defaults to 0 here, not to the production default: the
    fake answers instantly unless a test asks it not to, so a grace period
    would only add real sleep to every miss. The tests that care about the
    grace pass it explicitly.
    """
    fake = FakeAOSS(**kw)
    s = AOSSStore(CFG, client=AOSSClient(CFG, transport=fake.transport,
                                         max_retries=0),
                  cache_writes=cache_writes, miss_grace=miss_grace)
    return s, fake


def _campaign(cid="camp_1"):
    return {
        "campaign_id": cid, "space_id": "space_abc123",
        "observations": [
            {"point": {"ligand": "XPhos", "temp": 90.0}, "y": 41.7,
             "recorded": 1760000100.5},
            {"point": {"ligand": "SPhos", "temp": 70.0}, "y": 0,
             "recorded": 1760000200.5}],
        "skipped": [{"point": {"ligand": "XPhos", "temp": 110.0},
                     "reason": "no material that day", "ts": 1760000300.5}],
        "pending": [{"ligand": "SPhos", "temp": 100.0}],
        "config": {"init_method": "random", "seed": 7,
                   "acquisition": "ei", "surrogate": None},
        "created": 1760000000.0,
    }


# -- provisioning -----------------------------------------------------------

def test_ensure_indices_creates_three_with_explicit_mappings():
    s, fake = _store()
    result = s.ensure_indices()
    assert result == {"space": "created", "campaign": "created",
                      "observation": "created"}, result
    assert sorted(fake.indices) == ["beacon-campaigns-v1",
                                    "beacon-observations-v1",
                                    "beacon-search-spaces-v1"], sorted(fake.indices)
    for body in fake.indices.values():
        assert body["mappings"]["dynamic"] == "strict"


def test_ensure_indices_is_idempotent():
    s, fake = _store()
    s.ensure_indices()
    assert s.ensure_indices() == {"space": "exists", "campaign": "exists",
                                  "observation": "exists"}


def test_mapping_exists_before_first_write():
    """A write must never be what creates the index."""
    s, fake = _store()
    s.ensure_indices()
    created_at = len(fake.calls)
    s.put("space", "space_abc123", SPACE)
    creations = [c for c in fake.calls[created_at:]
                 if c[0] == "PUT" and c[1].count("/") == 1]
    assert not creations, creations


def test_no_cat_or_cluster_apis_are_used():
    s, fake = _store()
    s.ensure_indices()
    s.put("space", "space_abc123", SPACE)
    s.list("space")
    bad = [c for c in fake.calls if "_cat" in c[1] or "_cluster" in c[1]]
    assert not bad, bad


# -- the three methods ------------------------------------------------------

def test_put_get_round_trip_preserves_the_record():
    s, _ = _store()
    s.ensure_indices()
    s.put("space", "space_abc123", SPACE)
    got = s.get("space", "space_abc123")
    assert got["space"]["factors"][0]["levels"] == ["XPhos", "SPhos"]
    assert got["space_id"] == "space_abc123"


def test_put_uses_deterministic_id_not_auto_id():
    s, fake = _store()
    s.ensure_indices()
    s.put("space", "space_abc123", SPACE)
    puts = [c for c in fake.calls if c[1].endswith("/_doc/space_abc123")]
    assert puts and puts[0][0] == "PUT", fake.calls
    assert not [c for c in fake.calls if c[0] == "POST" and c[1].endswith("/_doc")]


def test_repeated_put_replaces_rather_than_duplicates():
    s, fake = _store()
    s.ensure_indices()
    for _ in range(3):
        s.put("space", "space_abc123", SPACE)
    assert len(fake.docs["beacon-search-spaces-v1"]) == 1


def test_get_missing_raises_store_error():
    s, _ = _store()
    s.ensure_indices()
    try:
        s.get("campaign", "camp_nope")
    except StoreError:
        return
    raise AssertionError("expected StoreError")


def test_list_returns_sorted_ids():
    s, _ = _store()
    s.ensure_indices()
    for cid in ("camp_c", "camp_a", "camp_b"):
        s.put("campaign", cid, _campaign(cid))
    assert s.list("campaign") == ["camp_a", "camp_b", "camp_c"]


def test_list_paginates_beyond_one_page():
    s, _ = _store()
    s.ensure_indices()
    s.search_page_size = 3
    ids = [f"camp_{i:03d}" for i in range(10)]
    for cid in ids:
        s.put("campaign", cid, _campaign(cid))
    assert s.list("campaign") == sorted(ids)


def test_list_of_missing_index_is_empty_not_an_error():
    s, _ = _store()
    assert s.list("campaign") == []


def test_delete_removes_the_document():
    s, _ = _store()
    s.ensure_indices()
    s.put("campaign", "camp_1", _campaign())
    assert s.delete("campaign", "camp_1") is True
    assert s.exists("campaign", "camp_1") is False


# -- near-real-time lag -----------------------------------------------------

def test_get_by_id_works_while_search_is_still_blind():
    """The resume path reads by id, so search lag must not affect it."""
    s, _ = _store(cache_writes=False, searchable_after=30.0)
    s.ensure_indices()
    s.put("campaign", "camp_1", _campaign())
    got = s.get("campaign", "camp_1")          # immediate
    assert got["campaign_id"] == "camp_1"
    assert s.list("campaign") == []            # search cannot see it yet


def test_get_retries_when_get_by_id_also_lags():
    """AOSS get-by-id is not realtime (measured 2.5-6.1 s), so a miss by a
    process with no cache -- another worker, or after a restart -- is retried."""
    s, _ = _store(cache_writes=False, get_by_id_lag=1.0)
    s.ensure_indices()
    s.read_after_write_timeout = 10.0
    s.put("campaign", "camp_1", _campaign())
    started = time.monotonic()
    got = s.get("campaign", "camp_1")
    assert got["campaign_id"] == "camp_1"
    assert time.monotonic() - started >= 0.9


def test_cache_serves_read_your_writes_instantly():
    """The writing process must not pay the visibility lag at all."""
    s, _ = _store(get_by_id_lag=30.0)
    s.ensure_indices()
    s.put("campaign", "camp_1", _campaign())
    started = time.monotonic()
    got = s.get("campaign", "camp_1")
    assert got["campaign_id"] == "camp_1"
    assert time.monotonic() - started < 0.2


def test_cache_is_not_populated_when_the_write_fails():
    """A cache hit must never imply a document that is not durable."""
    s, fake = _store()
    s.ensure_indices()
    fake.deny_all = True
    try:
        s.put("campaign", "camp_ghost", _campaign("camp_ghost"))
    except Exception:                                            # noqa: BLE001
        pass
    fake.deny_all = False
    assert ("campaign", "camp_ghost") not in s._cache
    assert s.exists("campaign", "camp_ghost") is False


def test_delete_evicts_the_cache():
    """A delete must not be masked by a stale read-your-writes hit."""
    s, _ = _store()
    s.ensure_indices()
    s.put("campaign", "camp_1", _campaign())
    s.delete("campaign", "camp_1")
    assert s.exists("campaign", "camp_1") is False


def test_cache_is_bounded():
    """The cache covers a seconds-long gap; it is not a durable store."""
    s, _ = _store()
    s.ensure_indices()
    for i in range(AOSSStore._CACHE_LIMIT + 20):
        s.put("campaign", f"camp_{i:04d}", _campaign(f"camp_{i:04d}"))
    assert len(s._cache) <= AOSSStore._CACHE_LIMIT


def test_get_of_a_key_never_written_does_not_stall():
    """A genuine miss must stay fast, not burn the retry budget."""
    s, _ = _store(cache_writes=False, get_by_id_lag=5.0)
    s.ensure_indices()
    s.read_after_write_timeout = 10.0
    started = time.monotonic()
    try:
        s.get("campaign", "camp_never")
    except StoreError:
        pass
    assert time.monotonic() - started < 1.0


def test_wait_until_searchable_reports_the_lag():
    s, _ = _store(searchable_after=0.5)
    s.ensure_indices()
    s.put("campaign", "camp_1", _campaign())
    lag = s.wait_until_searchable("campaign", "camp_1", timeout=10)
    assert lag is not None and lag >= 0.4, lag


# -- the observations projection --------------------------------------------

def test_campaign_put_fans_out_observations_and_skips():
    s, fake = _store()
    s.ensure_indices()
    s.put("campaign", "camp_1", _campaign())
    obs = fake.docs["beacon-observations-v1"]
    assert len(obs) == 3, obs            # 2 observed + 1 skipped
    outcomes = sorted(src["outcome"] for src, _ in obs.values())
    assert outcomes == ["observed", "observed", "skipped"], outcomes


def test_a_skip_is_stored_without_a_value_never_as_zero():
    s, fake = _store()
    s.ensure_indices()
    s.put("campaign", "camp_1", _campaign())
    skips = [src for src, _ in fake.docs["beacon-observations-v1"].values()
             if src["outcome"] == "skipped"]
    assert len(skips) == 1
    assert "y" not in skips[0], skips[0]
    assert skips[0]["skip_reason"] == "no material that day"


def test_a_real_zero_yield_is_preserved_as_zero():
    """A measured 0 is data. Only an unrun experiment has no value."""
    s, fake = _store()
    s.ensure_indices()
    s.put("campaign", "camp_1", _campaign())
    observed = [src for src, _ in fake.docs["beacon-observations-v1"].values()
                if src["outcome"] == "observed"]
    assert sorted(o["y"] for o in observed) == [0, 41.7]


def test_deferred_then_run_is_marked_as_such():
    camp = _campaign()
    point = {"ligand": "XPhos", "temp": 110.0}
    camp["skipped"] = []
    camp["deferral_history"] = [{"point": point, "reason": "no material",
                                 "ts": 1760000300.5}]
    camp["observations"].append({"point": point, "y": 88.0,
                                 "recorded": 1760000400.5})
    s, fake = _store()
    s.ensure_indices()
    s.put("campaign", "camp_1", camp)
    marked = [src for src, _ in fake.docs["beacon-observations-v1"].values()
              if src["outcome"] == "deferred_then_observed"]
    assert len(marked) == 1 and marked[0]["was_deferred"] is True


def test_fan_out_ids_are_deterministic():
    s, fake = _store()
    s.ensure_indices()
    for _ in range(3):
        s.put("campaign", "camp_1", _campaign())
    assert len(fake.docs["beacon-observations-v1"]) == 3


def test_suggested_records_what_the_campaign_proposed():
    s, fake = _store()
    s.ensure_indices()
    s.put("campaign", "camp_1", _campaign())
    src, _ = fake.docs["beacon-campaigns-v1"]["camp_1"]
    states = sorted(x["state"] for x in src["suggested"])
    assert states == ["observed", "observed", "pending", "skipped"], states


# -- mapping discipline ------------------------------------------------------

def test_timestamps_are_written_as_epoch_millis_integers():
    """Float epoch seconds in a date field is the classic AOSS mis-type."""
    s, fake = _store()
    s.ensure_indices()
    s.put("campaign", "camp_1", _campaign())
    src, _ = fake.docs["beacon-campaigns-v1"]["camp_1"]
    assert src["created_at"] == 1760000000000
    assert isinstance(src["created_at"], int)


def test_strict_mapping_rejects_an_undeclared_field():
    """The fake enforces what AOSS enforces: a surprise field is a failure."""
    s, fake = _store()
    s.ensure_indices()
    client = s.client
    try:
        client.put_doc("beacon-campaigns-v1", "x",
                       {"doc_id": "x", "totally_new_field": 1})
    except Exception as exc:                                     # noqa: BLE001
        assert "strict_dynamic_mapping_exception" in str(exc), exc
        return
    raise AssertionError("expected a strict mapping rejection")


def test_arbitrary_factor_names_cannot_collide_with_the_mapping():
    """Payload is enabled:false, so user-chosen keys are never indexed."""
    weird = dict(SPACE)
    weird["space"] = dict(SPACE["space"])
    weird["space"]["factors"] = [
        {"name": "y", "kind": "categorical", "levels": ["a"]},
        {"name": "outcome", "kind": "categorical", "levels": ["b"]}]
    s, _ = _store()
    s.ensure_indices()
    s.put("space", "space_abc123", weird)
    got = s.get("space", "space_abc123")
    assert [f["name"] for f in got["space"]["factors"]] == ["y", "outcome"]


# -- errors ------------------------------------------------------------------

def test_403_is_reported_as_a_data_access_policy_problem():
    s, _ = _store(deny_all=True)
    try:
        s.ensure_indices()
    except AOSSAccessDenied as exc:
        assert "aoss:APIAccessAll" in str(exc)
        return
    raise AssertionError("expected AOSSAccessDenied")


def test_signing_with_the_wrong_service_is_rejected():
    """Guards the 'sign as aoss, not es' rule end to end."""
    fake = FakeAOSS()
    from maths_tools.aoss import signing as signing_mod
    original = signing_mod.sign_request

    def as_es(*args, **kwargs):
        kwargs["service"] = "es"
        return original(*args, **kwargs)

    from maths_tools.aoss import client as client_mod
    client_mod.sign_request = as_es
    try:
        s = AOSSStore(CFG, client=AOSSClient(
            CFG, transport=fake.transport, max_retries=0))
        s.ensure_indices()
    except AOSSAccessDenied:
        return
    finally:
        client_mod.sign_request = original
    raise AssertionError("expected a 403 when signing as es")


# -- parity with the local store ---------------------------------------------

def _local_json_store(root):
    """Return a :class:`maths_tools.store.LocalJSONStore` rooted at *root*.

    This used to be loaded by path under an alias: while the AOSS code lived
    in its own top-level package, both packages had a module called ``store``
    and a plain import picked the wrong one. Inside ``maths_tools.aoss`` the
    collision does not exist, so this is an ordinary import again.
    """
    from maths_tools.store import LocalJSONStore
    return LocalJSONStore(root)


def test_parity_with_local_json_store():
    """Both backends must be indistinguishable to the tool functions."""
    tmp = tempfile.mkdtemp()
    try:
        local = _local_json_store(tmp)
        remote, _ = _store()
        remote.ensure_indices()

        for backend in (local, remote):
            backend.put("space", "space_abc123", SPACE)
            backend.put("campaign", "camp_1", _campaign("camp_1"))
            backend.put("campaign", "camp_2", _campaign("camp_2"))

        assert local.list("campaign") == remote.list("campaign")
        assert local.list("space") == remote.list("space")

        a = local.get("campaign", "camp_1")
        b = remote.get("campaign", "camp_1")
        for field in ("campaign_id", "space_id", "pending", "config"):
            assert a[field] == b[field], (field, a[field], b[field])
        assert [o["y"] for o in a["observations"]] == \
               [o["y"] for o in b["observations"]]
        assert a["skipped"] == b["skipped"]
        assert local.exists("campaign", "nope") is False
        assert remote.exists("campaign", "nope") is False
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_point_key_matches_the_tool_implementation():
    """The fan-out id must agree with how the tool identifies a point."""
    a = canonical_point_key({"ligand": "XPhos", "temp": 90.0})
    b = canonical_point_key({"temp": 90.0, "ligand": "XPhos"})
    assert a == b


def test_network_timeout_is_retried_not_fatal():
    """A transient read timeout must not fail a campaign."""
    import socket
    fake = FakeAOSS()
    calls = {"n": 0}

    def flaky(method, url, headers, body, timeout):
        calls["n"] += 1
        if calls["n"] == 1:
            raise socket.timeout("The read operation timed out")
        return fake.transport(method, url, headers, body, timeout)

    s = AOSSStore(CFG, client=AOSSClient(CFG, transport=flaky, max_retries=2))
    s.ensure_indices()
    assert calls["n"] >= 2, calls
    assert "beacon-campaigns-v1" in fake.indices


def test_network_error_is_eventually_reported():
    """It is retried, not swallowed: persistent failure still raises."""
    import socket
    from maths_tools.aoss.client import AOSSRequestError

    def always_dead(method, url, headers, body, timeout):
        raise socket.timeout("down")

    s = AOSSStore(CFG, client=AOSSClient(CFG, transport=always_dead,
                                         max_retries=1))
    try:
        s.ensure_indices()
    except AOSSRequestError as exc:
        assert "network error" in str(exc)
        return
    raise AssertionError("expected AOSSRequestError")


# -- the fresh-process miss grace ---------------------------------------------
#
# Measured against the live collection: a document written by one process and
# read by a *fresh* one missed on the first get-by-id in 4 of 5 trials, and
# became visible 3.4-7.5 s after the write. A restarted server has no cache
# and no record that anything was written, so without a grace period it would
# report "not found" for a campaign that exists.

def test_fresh_reader_survives_get_by_id_lag():
    """A reader that never wrote the key still finds it, after waiting."""
    writer, fake = _store(get_by_id_lag=0.6)
    writer.ensure_indices()
    writer.put("campaign", "camp_1", _campaign())

    # a brand-new reader on the same collection: no cache, no _recent
    reader = AOSSStore(CFG, client=AOSSClient(CFG, transport=fake.transport,
                                              max_retries=0),
                       cache_writes=False, miss_grace=10.0)
    got = reader.get("campaign", "camp_1")
    assert got["campaign_id"] == "camp_1", got


def test_miss_grace_is_bounded_and_then_raises():
    """A key that genuinely does not exist is still reported, not hung."""
    s, _ = _store(miss_grace=0.5)
    s.ensure_indices()
    t0 = time.monotonic()
    try:
        s.get("campaign", "camp_never_written")
    except StoreError:
        waited = time.monotonic() - t0
        assert 0.4 <= waited < 5.0, waited
        return
    raise AssertionError("expected StoreError")


def test_miss_grace_can_be_switched_off():
    """miss_grace=0 restores fail-fast for callers that want it."""
    s, _ = _store(miss_grace=0.0)
    s.ensure_indices()
    t0 = time.monotonic()
    try:
        s.get("campaign", "camp_never_written")
    except StoreError:
        assert time.monotonic() - t0 < 0.4, time.monotonic() - t0
        return
    raise AssertionError("expected StoreError")


def test_exists_does_not_wait_out_the_grace():
    """A presence check answers now; only get() is worth waiting for."""
    s, _ = _store(miss_grace=30.0)
    s.ensure_indices()
    t0 = time.monotonic()
    assert s.exists("campaign", "camp_never_written") is False
    assert time.monotonic() - t0 < 0.4, time.monotonic() - t0


def test_default_miss_grace_covers_the_measured_lag():
    """The default is set from the measurement, not from taste."""
    from maths_tools.aoss.store import DEFAULT_MISS_GRACE
    assert DEFAULT_MISS_GRACE >= 7.5, DEFAULT_MISS_GRACE


# -- the dependency claim ------------------------------------------------------

def test_aoss_imports_only_stdlib():
    """The backend must add no third-party dependency.

    requirements.txt says so; this is what makes that true rather than
    aspirational. Anything imported by the backend has to resolve to the
    standard library or to maths_tools itself.
    """
    import ast
    import os
    import sys
    import sysconfig

    pkg = os.path.join(ROOT, "maths_tools", "aoss")
    stdlib = sysconfig.get_paths()["stdlib"]
    offenders = []
    for name in sorted(os.listdir(pkg)):
        if not name.endswith(".py"):
            continue
        tree = ast.parse(open(os.path.join(pkg, name)).read())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.level:                       # relative: our own package
                    continue
                roots = [(node.module or "").split(".")[0]]
            elif isinstance(node, ast.Import):
                roots = [a.name.split(".")[0] for a in node.names]
            else:
                continue
            for root in roots:
                if root in ("__future__", "maths_tools", ""):
                    continue
                module = sys.modules.get(root) or __import__(root)
                path = getattr(module, "__file__", "") or ""
                if path and not path.startswith(stdlib):
                    offenders.append("%s: %s (%s)" % (name, root, path))
    assert not offenders, offenders


TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]

if __name__ == "__main__":
    passed, failed = 0, []
    for fn in TESTS:
        try:
            fn()
            passed += 1
        except Exception as exc:                                 # noqa: BLE001
            failed.append((fn.__name__, exc))
    print("PASS %d" % passed)
    for name, exc in failed:
        print("FAIL %s" % name)
        print("     %s: %s" % (type(exc).__name__, exc))
    print("FAILED %d" % len(failed))
    sys.exit(1 if failed else 0)
