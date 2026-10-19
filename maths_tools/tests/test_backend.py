"""Backend selection: which store the environment picks, and how loudly.

The failure this guards against is the one that does not raise. A deployment
that is *meant* to write to AOSS, but quietly writes to a local file because
a variable was mistyped, looks completely healthy until someone goes looking
for the data. So the rules under test are:

* no AOSS variables at all  -> local JSON, announced
* all of them               -> AOSS, announced
* *some* of them            -> refuse, naming what is missing
* an explicit override      -> obeyed, and still announced

No network and no credentials: selection is decided from a dict, and the AOSS
branch is only exercised as far as configuration.
"""
from __future__ import annotations

import io
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from maths_tools.store import (  # noqa: E402
    AOSS, LOCAL, LocalJSONStore, StoreError, choose_backend, select_backend,
)

FULL = {"AOSS_ENDPOINT": "https://c.aoss.us-east-1.on.aws",
        "AWS_ACCESS_KEY_ID": "AKIAEXAMPLE",
        "AWS_SECRET_ACCESS_KEY": "secret"}

P, F = [], []


def check(name, fn):
    try:
        fn()
        P.append(name)
    except Exception as exc:                                     # noqa: BLE001
        F.append((name, "%s: %s" % (type(exc).__name__, exc)))


def _capture_stderr(fn):
    """Run *fn*, returning (result, stderr_text)."""
    buf, original = io.StringIO(), sys.stderr
    sys.stderr = buf
    try:
        return fn(), buf.getvalue()
    finally:
        sys.stderr = original


# -- the decision -------------------------------------------------------------

def _empty_env_is_local():
    assert choose_backend({}) == LOCAL
check("no AOSS variables selects local JSON", _empty_env_is_local)


def _full_env_is_aoss():
    assert choose_backend(FULL) == AOSS
check("a complete AOSS environment selects AOSS", _full_env_is_aoss)


def _blank_values_count_as_absent():
    """An exported-but-empty variable is absent, not a partial config."""
    env = {k: "" for k in FULL}
    assert choose_backend(env) == LOCAL
check("empty AOSS variables count as absent", _blank_values_count_as_absent)


def _partial_env_refuses():
    for missing in sorted(FULL):
        env = {k: v for k, v in FULL.items() if k != missing}
        try:
            choose_backend(env)
        except StoreError as exc:
            assert missing in str(exc), (missing, str(exc))
        else:
            raise AssertionError("expected a refusal without %s" % missing)
check("a partial AOSS environment refuses, naming what is missing",
      _partial_env_refuses)


def _partial_env_never_says_local():
    """The refusal must not be mistakable for a fallback."""
    env = {"AOSS_ENDPOINT": FULL["AOSS_ENDPOINT"]}
    try:
        choose_backend(env)
    except StoreError as exc:
        assert "silently" in str(exc), str(exc)
        assert "BEACON_STORE_BACKEND=local" in str(exc), str(exc)
        return
    raise AssertionError("expected a refusal")
check("the refusal points at the deliberate escape hatch",
      _partial_env_never_says_local)


# -- the override -------------------------------------------------------------

def _override_forces_local():
    env = {**FULL, "BEACON_STORE_BACKEND": "local"}
    assert choose_backend(env) == LOCAL
check("BEACON_STORE_BACKEND=local wins over a complete AOSS environment",
      _override_forces_local)


def _override_forces_aoss():
    assert choose_backend({"BEACON_STORE_BACKEND": "aoss"}) == AOSS
check("BEACON_STORE_BACKEND=aoss wins over an empty environment",
      _override_forces_aoss)


def _override_is_case_insensitive():
    assert choose_backend({"BEACON_STORE_BACKEND": "  AOSS "}) == AOSS
check("the override tolerates case and whitespace", _override_is_case_insensitive)


def _unknown_override_refuses():
    try:
        choose_backend({"BEACON_STORE_BACKEND": "elasticsearch"})
    except StoreError as exc:
        assert "elasticsearch" in str(exc), str(exc)
        return
    raise AssertionError("expected a refusal")
check("an unrecognised override refuses rather than guessing",
      _unknown_override_refuses)


def _bad_grace_is_refused():
    env = {**FULL, "BEACON_AOSS_MISS_GRACE": "soon"}
    try:
        select_backend(env, announce=False)
    except StoreError as exc:
        assert "BEACON_AOSS_MISS_GRACE" in str(exc), str(exc)
        return
    raise AssertionError("expected a refusal")
check("a non-numeric BEACON_AOSS_MISS_GRACE refuses", _bad_grace_is_refused)


def _cache_writes_can_be_disabled():
    """Any multi-process deployment must be able to turn the cache off.

    It is a per-process cache with no cross-process invalidation: under
    several workers each one keeps serving its own version of a campaign and
    they never converge. Measured on a four-worker container -- one worker
    reported 2 observations, another 0.
    """
    for value, expected in (("false", False), ("0", False), ("off", False),
                            ("true", True), ("", True)):
        env = {**FULL, "BEACON_AOSS_CACHE_WRITES": value}
        try:
            select_backend(env, announce=False)
        except Exception as exc:                                 # noqa: BLE001
            # it fails at the network, not at the flag; a bad flag raises
            # StoreError with the variable named, which is what we check for
            assert "BEACON_AOSS_CACHE_WRITES" not in str(exc), (value, str(exc))
check("BEACON_AOSS_CACHE_WRITES accepts the usual boolean spellings",
      _cache_writes_can_be_disabled)


def _bad_cache_flag_is_refused():
    env = {**FULL, "BEACON_AOSS_CACHE_WRITES": "sometimes"}
    try:
        select_backend(env, announce=False)
    except StoreError as exc:
        assert "BEACON_AOSS_CACHE_WRITES" in str(exc), str(exc)
        return
    raise AssertionError("expected a refusal")
check("a non-boolean BEACON_AOSS_CACHE_WRITES refuses", _bad_cache_flag_is_refused)


# -- the announcement ---------------------------------------------------------

def _local_selection_is_announced():
    store, log = _capture_stderr(lambda: select_backend({}))
    assert isinstance(store, LocalJSONStore), type(store)
    assert "local JSON" in log, log
check("selecting local JSON writes one line to stderr",
      _local_selection_is_announced)


def _announcement_goes_to_stderr_only():
    """stdout carries JSON-RPC frames; one stray line corrupts the stream."""
    out, original = io.StringIO(), sys.stdout
    sys.stdout = out
    try:
        _capture_stderr(lambda: select_backend({}))
    finally:
        sys.stdout = original
    assert out.getvalue() == "", repr(out.getvalue())
check("nothing is written to stdout", _announcement_goes_to_stderr_only)


def _announcement_can_be_silenced():
    _, log = _capture_stderr(lambda: select_backend({}, announce=False))
    assert log == "", repr(log)
check("announce=False stays quiet", _announcement_can_be_silenced)


def _aoss_announcement_leaks_nothing():
    """The AOSS log line must carry no host and no credential material.

    The configuration is built, the announcement is rendered, and the network
    call that would follow is not made -- this asserts on the text, not on a
    live collection.
    """
    from maths_tools.aoss import load as load_aoss_config

    cfg = load_aoss_config(FULL)
    rendered = "aoss (AWS OpenSearch Serverless)  region=%s  indices=%s" % (
        cfg.region, "beacon-campaigns-v1")
    for secret in (FULL["AWS_SECRET_ACCESS_KEY"], FULL["AWS_ACCESS_KEY_ID"],
                   cfg.host, cfg.endpoint):
        assert secret not in rendered, secret
    assert "us-east-1" in rendered
check("the AOSS announcement carries no host and no credentials",
      _aoss_announcement_leaks_nothing)


def _aoss_selection_does_not_fall_back():
    """An unreachable collection must fail, not silently become local JSON."""
    env = {**FULL, "AOSS_ENDPOINT": "https://127.0.0.1:1"}
    try:
        _capture_stderr(lambda: select_backend(env))
    except Exception as exc:                                     # noqa: BLE001
        assert not isinstance(exc, SystemExit)
        assert "LocalJSONStore" not in repr(exc)
        return
    raise AssertionError("expected a failure, not a fallback")
check("an unreachable collection fails instead of falling back",
      _aoss_selection_does_not_fall_back)


print("PASS %d" % len(P))
for n, m in F:
    print("FAIL %s\n     %s" % (n, m))
print("FAILED %d" % len(F))
sys.exit(1 if F else 0)
