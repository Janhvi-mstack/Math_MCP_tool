"""Stage 7 tests -- campaign state and the unvalidated-recommendation rule."""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from maths_tools.space import *
from maths_tools.campaign import Campaign, CampaignError, Skipped
from maths_tools.priors import CategoricalPrior

P, F = [], []
def check(name, fn):
    try: fn(); P.append(name)
    except Exception as e: F.append((name, "%s: %s" % (type(e).__name__, e)))
def raises(exc, fn, frag=None):
    try: fn()
    except exc as e:
        if frag and frag.lower() not in str(e).lower():
            raise AssertionError("wrong message: %s" % e)
        return
    raise AssertionError("expected %s" % exc.__name__)

S = SearchSpace([Factor("l", CATEGORICAL, levels=list("abcd")),
                 Factor("t", INTEGER, bounds=(20, 120))])
TRUTH = {"a": 10.0, "b": 40.0, "c": 90.0, "d": 30.0}
f = lambda p: TRUTH[p["l"]] - 0.01 * (p["t"] - 80.0) ** 2

def run(skip_every=0, n_iter=12, seed=0, exclude_skipped=True):
    """A campaign with some suggestions declined.

    exclude_skipped defaults to True here because otherwise the model -- which
    is unchanged by a skip -- simply re-proposes the same experiment on the
    next call, the helper records it, and no skip ever persists. That
    re-proposal is correct behaviour and is tested separately in
    `_skipped_is_offered_again`; these tests need skips that stick.
    """
    c = Campaign(S, surrogate_kind="plain", n_init=5, seed=seed,
                 exclude_skipped=exclude_skipped)
    for p in c.suggest(5):
        c.record(p, f(p))
    for i in range(n_iter):
        p = c.suggest(1)[0]
        if skip_every and i % skip_every == 1:
            c.skip(p, "no material")
        else:
            c.record(p, f(p))
    return c

# ------------- THE CLAIM -------------
def _skip_is_inert():
    """The training data must be identical to a run where nothing was skipped.

    This is the whole claim on slide 11: the model ends up in exactly the state
    it would have been in if the experiment had never been proposed.
    """
    c = run(skip_every=3)
    assert c.n_skipped >= 3, c.n_skipped

    # rebuild a campaign that records exactly the same experiments and skips
    # nothing at all
    clean = Campaign(S, surrogate_kind="plain", n_init=5, seed=0)
    for o in c.observations:
        clean.record(o.point, o.y)

    Xa, ya = c._train_arrays()
    Xb, yb = clean._train_arrays()
    assert Xa.shape == Xb.shape, (Xa.shape, Xb.shape)
    assert np.array_equal(Xa, Xb), "training matrices differ"
    assert np.array_equal(ya, yb), "training targets differ"
check("a skipped suggestion leaves the training data byte-identical", _skip_is_inert)

def _skipped_carry_no_yield():
    c = run(skip_every=3)
    for s in c.skipped:
        assert isinstance(s, Skipped)
        assert not hasattr(s, "y")
    assert len(c.observed_y) == c.n_observations
check("skipped records have no yield field at all", _skipped_carry_no_yield)

def _skip_not_in_observed():
    c = run(skip_every=3)
    sk = {tuple(sorted(s.point.items())) for s in c.skipped}
    ob = {tuple(sorted(p.items())) for p in c.observed_points}
    assert not (sk & ob), sk & ob
check("no skipped experiment appears in the observed set", _skip_not_in_observed)

def _skipped_stays_in_the_space():
    """A skip withholds the experiment from the model, not from the space."""
    c = run(skip_every=2, n_iter=14)
    assert c.n_skipped >= 3
    sk = {tuple(sorted(s.point.items())) for s in c.skipped}
    pool = {tuple(sorted(p.items())) for p in S.enumerate()}
    assert sk <= pool, "skipped experiments vanished from the search space"
check("skipped experiments remain members of the search space",
      _skipped_stays_in_the_space)


def _skipped_is_offered_again():
    """The default: a skip does not blacklist anything.

    Asked three times in review -- "can our tool recommend the same conditions
    again also in the next titration?" The model is unchanged by a skip, so the
    very next call proposes the same experiment. Nothing is removed.
    """
    c = Campaign(S, surrogate_kind="plain", n_init=5, seed=0)   # default
    for p in c.suggest(5):
        c.record(p, f(p))
    first = c.suggest(1)[0]
    c.skip(first, "no material")
    again = c.suggest(1)[0]
    assert tuple(sorted(again.items())) == tuple(sorted(first.items())), \
        "a skipped experiment was blacklisted instead of staying eligible"

    # and running it later moves it out of the skipped list
    c.record(again, f(again))
    assert c.n_skipped == 0 and c.n_observations == 6
    assert len(c.deferral_history) == 1, "the deferral was not kept for audit"
    c.check_invariants()
check("default: a skipped experiment is offered again, and can then be run",
      _skipped_is_offered_again)


def _exclude_skipped_holds_back():
    c = Campaign(S, surrogate_kind="plain", n_init=5, seed=0,
                 exclude_skipped=True)
    for p in c.suggest(5):
        c.record(p, f(p))
    first = c.suggest(1)[0]
    c.skip(first, "no material")
    later = c.suggest(1)[0]
    assert tuple(sorted(later.items())) != tuple(sorted(first.items()))
check("exclude_skipped=True holds a skipped experiment back",
      _exclude_skipped_holds_back)

# ------------- refusals -------------
def _nan_yield_refused():
    c = Campaign(S, surrogate_kind="plain", seed=0)
    p = c.suggest(5)[0]
    raises(CampaignError, lambda: c.record(p, float("nan")), "call skip() instead")
check("recording a NaN yield is refused and points at skip()", _nan_yield_refused)

def _duplicate_record_refused():
    c = Campaign(S, surrogate_kind="plain", seed=0)
    p = c.suggest(5)[0]
    c.record(p, 1.0)
    raises(CampaignError, lambda: c.record(p, 2.0), "already been recorded")
check("recording the same experiment twice is refused", _duplicate_record_refused)

def _skip_after_record_refused():
    c = Campaign(S, surrogate_kind="plain", seed=0)
    p = c.suggest(5)[0]
    c.record(p, 1.0)
    raises(CampaignError, lambda: c.skip(p, "changed my mind"), "already been recorded")
check("skipping an experiment that already ran is refused", _skip_after_record_refused)

def _invalid_point_refused():
    c = Campaign(S, surrogate_kind="plain", seed=0)
    raises(SpaceError, lambda: c.record({"l": "zzz", "t": 50}, 1.0), "not one of")
    raises(SpaceError, lambda: c.skip({"l": "a", "t": 9999}, "x"), "outside bounds")
check("record/skip validate the point against the space", _invalid_point_refused)

# ------------- the guard itself -------------
def _guard_catches_contamination():
    """Force a leak past the public API; the guard must still catch it."""
    from maths_tools.campaign import Observation
    c = run(skip_every=3, n_iter=6)
    bad = c.skipped[0].point
    c._observations.append(Observation(dict(bad), 55.0, 99))
    raises(CampaignError, c.check_invariants, "skipped AND as observed")
check("guard catches a skipped point forced into observations", _guard_catches_contamination)

def _guard_catches_duplicates():
    from maths_tools.campaign import Observation
    c = run(n_iter=4)
    c._observations.append(Observation(dict(c.observations[0].point), 1.0, 99))
    raises(CampaignError, c.check_invariants, "more than once")
check("guard catches a duplicated observation", _guard_catches_duplicates)

def _guard_catches_nonfinite():
    from maths_tools.campaign import Observation
    c = run(n_iter=4)
    c._observations.append(Observation({"l": "a", "t": 21}, float("inf"), 99))
    raises(CampaignError, c.check_invariants, "non-finite yield")
check("guard catches a non-finite yield forced in", _guard_catches_nonfinite)

def _stale_model_is_allowed():
    """Between a fit and the next fit the model is legitimately behind."""
    c = run(n_iter=4)
    c.suggest(1)
    n_before = len(c.surrogate.train_X)
    c.record({"l": "d", "t": 119}, 12.0)
    c.check_invariants()                       # must NOT raise
    assert len(c.surrogate.train_X) == n_before
    raises(CampaignError, lambda: c.check_invariants(just_fitted=True), "diverged")
check("a stale model passes the normal guard but fails just_fitted", _stale_model_is_allowed)

# ------------- behaviour -------------
def _finds_optimum():
    c = run(n_iter=14)
    assert c.best_point["l"] == "c", c.best_point
    assert c.best_y > 85.0, c.best_y
check("campaign reaches the optimum region", _finds_optimum)

def _trace_monotone():
    c = run(skip_every=3, n_iter=12)
    tr = c.trace()
    assert len(tr) == c.n_observations
    assert all(b >= a for a, b in zip(tr, tr[1:])), tr
check("best-so-far trace never decreases", _trace_monotone)

def _initial_then_model():
    c = Campaign(S, surrogate_kind="plain", n_init=5, seed=0)
    d = c.suggest(5)
    assert c.history[-1]["kind"] == "design"
    for p in d:
        c.record(p, f(p))
    c.suggest(1)
    assert c.history[-1]["kind"] == "model"
check("initial design first, model once there is data", _initial_then_model)

def _summary_serialisable():
    c = run(skip_every=3, n_iter=8)
    d = json.loads(json.dumps(c.summary()))
    assert d["n_skipped"] >= 2
    assert d["skipped_reasons"]["no material"] == d["n_skipped"]
check("summary is json-serialisable and counts skip reasons", _summary_serialisable)

def _campaign_with_prior():
    pr = CategoricalPrior({"l": {"a": 1.0, "b": 2.0, "c": 4.0, "d": 1.0}})
    c = Campaign(S, surrogate_kind="plain", n_init=5, seed=0,
                 prior=pr, prior_beta=2.0, prior_confidence=1.0)
    for p in c.suggest(5):
        c.record(p, f(p))
    for _ in range(8):
        p = c.suggest(1)[0]
        c.record(p, f(p))
    assert c.best_point["l"] == "c", c.best_point
    assert c.n_observations == 13
check("campaign runs with a prior end to end", _campaign_with_prior)

def _all_skipped_still_works():
    """Every suggestion declined: the model must simply not advance."""
    c = Campaign(S, surrogate_kind="plain", n_init=5, seed=0)
    for p in c.suggest(5):
        c.record(p, f(p))
    n0 = c.n_observations
    for _ in range(5):
        c.skip(c.suggest(1)[0], "declined")
    assert c.n_observations == n0
    assert c.n_skipped == 5
    Xa, _ = c._train_arrays()
    assert len(Xa) == n0
check("declining every suggestion leaves the model unchanged", _all_skipped_still_works)

print("PASS %d" % len(P))
for n, m in F: print("FAIL %s\n     %s" % (n, m))
print("FAILED %d" % len(F))
sys.exit(1 if F else 0)
