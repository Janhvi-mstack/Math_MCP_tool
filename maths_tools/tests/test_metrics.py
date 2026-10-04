"""The benchmark target must be the one the reference protocol uses.

This suite exists because it once was not. `q95_threshold` -- the 95th
percentile -- was imported from the reference module and used as the Mode B
target. It is a real function, in the right file, used by the reference code,
and entirely wrong for this mode, which computes `0.95 * y.max()` inline. The
error made every N95 about 6 experiments too low with nothing to flag it.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

from maths_tools.metrics import (PENALTY, MetricMismatch, mode_b_target, n95,
                                 n95_penalised, verify_against_frozen,
                                 FROZEN_RUN_REPRO, FROZEN_TARGET_EXPR)

P, F = [], []


def check(name, fn):
    try:
        fn(); P.append(name)
    except Exception as e:
        F.append((name, "%s: %s" % (type(e).__name__, e)))


def _is_095_max():
    y = np.array([10.0, 50.0, 86.6, 30.0, 1.0])
    assert mode_b_target(y) == 0.95 * 86.6
check("target is 0.95 x max", _is_095_max)


def _not_the_percentile():
    """The two are different quantities, which is the whole point.

    Note they are NOT always ordered: on a uniform spread both land near the
    same value. The gap appears on skewed distributions -- which is what real
    yield data is, most results poor and a few good. Asserting a universal
    ordering would be asserting something false.
    """
    y = np.array([0.0] * 700 + list(np.linspace(10, 86.6, 90)))   # skewed
    t = mode_b_target(y)
    pct = float(np.quantile(y, 0.95))
    assert t != pct
    assert t > pct, (t, pct)
check("target differs from the 95th percentile on skewed data", _not_the_percentile)


def _harder_on_real_plates():
    """On every real plate the correct target is the harder one."""
    cache = os.environ.get("BEACON_REFERENCE_PLATES", "")
    plates = [] if not cache else [
        p for p in ("bhR1", "bhR2", "bhR3", "bhR4", "bhR5")
        if os.path.exists(os.path.join(cache, p + "_y.npy"))]
    if not plates:
        print("  (reference plates not configured -- skipped)")
        return
    for name in plates:
        y = np.load(os.path.join(cache, name + "_y.npy"))
        n_hard = int((y >= mode_b_target(y)).sum())
        n_easy = int((y >= np.quantile(y, 0.95)).sum())
        assert n_hard < n_easy, (name, n_hard, n_easy)
        assert n_easy == 40, (name, n_easy)
check("on real plates the correct target admits far fewer candidates",
      _harder_on_real_plates)


def _n95_basics():
    assert n95([1, 2, 9, 3], 5) == 3
    assert n95([9, 1, 1], 5) == 1
    assert n95([1, 2, 3], 5) is None
    assert n95_penalised([1, 2, 3], 5) == PENALTY
    assert n95([1, 5, 2], 5) == 2, "reaching exactly the target counts"
check("n95 counts the first time the running best reaches the target", _n95_basics)


def _running_best():
    """A dip after reaching the target must not un-reach it."""
    assert n95([1, 9, 1, 1], 5) == 2
check("n95 uses the running best, not the latest value", _running_best)


def _guard_matches_reference():
    r = verify_against_frozen()
    if not r["checked"]:
        print("  (reference source not present -- guard skipped)")
        return
    assert r["expression"] == FROZEN_TARGET_EXPR
    assert "0.95" in r["mode_b_branch"]
    assert "q95_threshold" not in r["mode_b_branch"]
check("guard confirms the reference protocol is unchanged", _guard_matches_reference)


def _guard_catches_a_change():
    """Edit the reference source and the guard must refuse."""
    import tempfile
    if not os.path.exists(FROZEN_RUN_REPRO):
        print("  (reference source not present -- skipped)")
        return
    src = open(FROZEN_RUN_REPRO).read()
    tampered = src.replace(FROZEN_TARGET_EXPR,
                           "target = float(np.quantile(y, 0.95))")
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
        fh.write(tampered)
        path = fh.name
    try:
        verify_against_frozen(path)
        raise AssertionError("guard did not notice the target was changed")
    except MetricMismatch:
        pass
    finally:
        os.unlink(path)
check("guard refuses when the reference target is changed", _guard_catches_a_change)


def _guard_is_not_trigger_happy():
    """q95_threshold elsewhere in the file must not trip it.

    The first version of this guard searched the whole file and fired on the
    import line. A guard that cries wolf gets switched off.
    """
    r = verify_against_frozen()
    if not r["checked"]:
        return
    whole = open(FROZEN_RUN_REPRO).read()
    assert "q95_threshold" in whole, "precondition: the name is in the file"
    assert "q95_threshold" not in r["mode_b_branch"]
check("guard ignores q95_threshold outside the Mode B branch", _guard_is_not_trigger_happy)


def _consumers_agree():
    """Every benchmark entry point must use the same target.

    benchmark.py is benchmark tooling and is not part of the shipped package,
    so this skips when it is absent rather than failing a clean install.
    """
    try:
        from maths_tools.benchmark import _metrics
    except ModuleNotFoundError:
        print("  (benchmark.py not in this install -- skipped)")
        return
    t, _, _ = _metrics()
    y = np.array([10.0, 99.0, 50.0])
    assert t(y) == mode_b_target(y)
check("benchmark entry points use the shared target", _consumers_agree)


print("PASS %d" % len(P))
for n, m in F:
    print("FAIL %s\n     %s" % (n, m))
print("FAILED %d" % len(F))
sys.exit(1 if F else 0)
