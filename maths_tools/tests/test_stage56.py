"""Stage 5 + 6 tests -- surrogates and acquisition."""
import sys, os, math
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from maths_tools.space import *
from maths_tools.encode import Encoder, DESCRIPTOR, MIXED
from maths_tools.surrogate import build_surrogate, PlainGP, MixedGP, DeepKernelGP, SurrogateError
from maths_tools.design import initial_design
from maths_tools.acquisition import next_recommendation, AcquisitionError

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

TRUTH = {"a": 10.0, "b": 40.0, "c": 90.0, "d": 30.0}
def f(p): return TRUTH[p["l"]] - 0.01 * (p["t"] - 80.0) ** 2

ENUM = SearchSpace([Factor("l", CATEGORICAL, levels=list("abcd")),
                    Factor("t", INTEGER, bounds=(20, 120))])
CONT = SearchSpace([Factor("l", CATEGORICAL, levels=list("abcd")),
                    Factor("t", CONTINUOUS, bounds=(20, 120))])

def _fitted(space, mode, kind, n=10, seed=0, method="sobol"):
    enc = Encoder(space, mode=mode)
    pts = initial_design(space, n, method=method, seed=seed).points
    X, y = enc.encode(pts), np.array([f(p) for p in pts])
    kw = {"seed": 0} if kind == "deep" else {}
    sur = build_surrogate(enc, kind=kind, **kw)
    sur.fit(X, y)
    return enc, sur, pts, y

# ---------------- surrogate ----------------
def _all_three_fit():
    for mode, kind in ((DESCRIPTOR,"plain"), (DESCRIPTOR,"deep"), (MIXED,"mixed")):
        enc, sur, pts, y = _fitted(ENUM, mode, kind)
        assert sur.fit_info["fit_failures"] == 0, (kind, sur.fit_info)
        m, s = sur.predict(enc.encode(pts))
        assert m.shape == (len(pts),) and s.shape == (len(pts),)
        assert np.isfinite(m).all() and np.isfinite(s).all()
        assert (s > 0).all()
check("all three surrogates fit and predict finite mean/std", _all_three_fit)

def _auto_dispatch():
    assert build_surrogate(Encoder(CONT, mode=MIXED)).kind == "mixed"
    assert build_surrogate(Encoder(ENUM, mode=DESCRIPTOR)).kind == "deep"
    numeric = SearchSpace([Factor("t", CONTINUOUS, bounds=(0,1))])
    assert build_surrogate(Encoder(numeric, mode=MIXED)).kind == "plain"
check("surrogate auto-dispatch by encoding", _auto_dispatch)

def _nan_y_refused():
    """The unvalidated-recommendation guard, enforced at the model boundary."""
    enc = Encoder(ENUM, mode=DESCRIPTOR)
    pts = initial_design(ENUM, 6, seed=0).points
    X = enc.encode(pts)
    y = np.array([f(p) for p in pts]); y[2] = np.nan
    raises(SurrogateError, lambda: PlainGP().fit(X, y), "never run must be omitted")
check("NaN in y is refused, not imputed", _nan_y_refused)

def _too_few_obs():
    enc = Encoder(ENUM, mode=DESCRIPTOR)
    pts = initial_design(ENUM, 1, seed=0).points
    raises(SurrogateError, lambda: PlainGP().fit(enc.encode(pts), np.array([1.0])),
           "at least 2")
check("fitting on 1 observation raises", _too_few_obs)

def _mismatched_lengths():
    raises(SurrogateError, lambda: PlainGP().fit(np.zeros((5,3)), np.zeros(4)),
           "5 rows but y has 4")
check("X/y length mismatch raises", _mismatched_lengths)

def _held_out_skill():
    """The model must actually learn, not just interpolate its own points."""
    enc = Encoder(CONT, mode=MIXED)
    tr = initial_design(CONT, 24, method="sobol", seed=0).points
    te = initial_design(CONT, 40, method="sobol", seed=999).points
    sur = MixedGP(enc.cat_dims)
    sur.fit(enc.encode(tr), np.array([f(p) for p in tr]))
    m, _ = sur.predict(enc.encode(te))
    truth = np.array([f(p) for p in te])
    r = np.corrcoef(m, truth)[0, 1]
    assert r > 0.9, r
check("mixed GP predicts held-out points (r > 0.9)", _held_out_skill)

def _predict_before_fit():
    raises(SurrogateError, lambda: PlainGP().predict(np.zeros((2,3))), "before predict")
check("predict before fit raises", _predict_before_fit)

def _mixed_needs_cats():
    numeric = SearchSpace([Factor("t", CONTINUOUS, bounds=(0,1))])
    raises(SurrogateError, lambda: build_surrogate(Encoder(numeric, mode=MIXED),
                                                   kind="mixed"), "needs categorical")
check("MixedGP without categoricals raises", _mixed_needs_cats)

# ---------------- acquisition ----------------
def _finds_optimum_enumerated():
    enc, sur, pts, y = _fitted(ENUM, DESCRIPTOR, "plain", n=12, method="maxmin")
    r = next_recommendation(ENUM, enc, sur, n=1, acquisition="ei", seed=0, observed=pts)
    assert r.points[0]["l"] == "c", r.points[0]
check("EI picks the best ligand on the enumerated path", _finds_optimum_enumerated)

def _continuous_beats_the_grid():
    """The capability the current code cannot produce: a value in no list."""
    enc, sur, pts, y = _fitted(CONT, MIXED, "mixed", n=14)
    r = next_recommendation(CONT, enc, sur, n=1, acquisition="ei", seed=0)
    t = r.points[0]["t"]
    assert r.points[0]["l"] == "c", r.points[0]
    assert abs(t - 80.0) < 6.0, t
    assert t not in [p["t"] for p in pts]
    assert r.path == "mixed"
check("continuous path recommends a temperature in no list", _continuous_beats_the_grid)

def _ei_rejects_batch():
    enc, sur, pts, y = _fitted(ENUM, DESCRIPTOR, "plain")
    raises(ValueError, lambda: next_recommendation(ENUM, enc, sur, n=3,
           acquisition="ei", seed=0), "single-point only")
check("analytic EI refuses a batch", _ei_rejects_batch)

def _batch_is_diverse():
    """Sequential greedy must not return q copies of the same point."""
    enc, sur, pts, y = _fitted(ENUM, DESCRIPTOR, "plain", n=10, method="maxmin")
    for acq in ("qei", "qnei", "ts"):
        r = next_recommendation(ENUM, enc, sur, n=4, acquisition=acq, seed=0,
                                observed=pts)
        keys = {(p["l"], p["t"]) for p in r.points}
        assert len(keys) == 4, (acq, keys)
check("batches of 4 are 4 distinct experiments", _batch_is_diverse)

def _observed_excluded():
    enc, sur, pts, y = _fitted(ENUM, DESCRIPTOR, "plain", n=10, method="maxmin")
    r = next_recommendation(ENUM, enc, sur, n=5, acquisition="qei", seed=0,
                            observed=pts, exclude_observed=True)
    assert r.n_excluded == len(pts), (r.n_excluded, len(pts))
    seen = {(p["l"], p["t"]) for p in pts}
    assert not ({(p["l"], p["t"]) for p in r.points} & seen)
check("already-run experiments are excluded", _observed_excluded)

def _not_excluded_when_asked():
    enc, sur, pts, y = _fitted(ENUM, DESCRIPTOR, "plain", n=10, method="maxmin")
    r = next_recommendation(ENUM, enc, sur, n=2, acquisition="qei", seed=0,
                            observed=pts, exclude_observed=False)
    assert r.n_excluded == 0
check("exclude_observed=False keeps the full pool", _not_excluded_when_asked)

def _determinism():
    enc, sur, pts, y = _fitted(ENUM, DESCRIPTOR, "plain", n=10, method="maxmin")
    for acq in ("ei", "qei", "qnei", "ts"):
        n = 1 if acq == "ei" else 3
        a = next_recommendation(ENUM, enc, sur, n=n, acquisition=acq, seed=5,
                                observed=pts).points
        b = next_recommendation(ENUM, enc, sur, n=n, acquisition=acq, seed=5,
                                observed=pts).points
        assert a == b, acq
check("recommendations are deterministic for a fixed seed", _determinism)

def _candidate_pool_honoured():
    """Where a knowledge graph's narrowing would arrive."""
    enc, sur, pts, y = _fitted(ENUM, DESCRIPTOR, "plain", n=10, method="maxmin")
    pool = [p for p in ENUM.enumerate() if p["l"] in ("a", "b")]
    r = next_recommendation(ENUM, enc, sur, n=3, acquisition="qei", seed=0,
                            candidates=pool, observed=pts)
    assert all(p["l"] in ("a", "b") for p in r.points), r.points
    assert any("supplied pool" in n for n in r.notes)
check("a supplied candidate pool is respected", _candidate_pool_honoured)

def _descriptor_on_infinite_space():
    s = SearchSpace([Factor("t", CONTINUOUS, bounds=(0, 1)),
                     Factor("l", CATEGORICAL, levels=["a", "b"])])
    enc = Encoder(s, mode=DESCRIPTOR)
    pts = initial_design(s, 6, method="sobol", seed=0).points
    sur = PlainGP(); sur.fit(enc.encode(pts), np.array([p["t"] for p in pts]))
    raises(SpaceError, lambda: next_recommendation(s, enc, sur, n=1,
           acquisition="ei", seed=0), "not enumerable")
check("descriptor encoding refuses a non-enumerable space", _descriptor_on_infinite_space)

def _exhausted_pool():
    tiny = SearchSpace([Factor("l", CATEGORICAL, levels=["a", "b", "c"])])
    enc = Encoder(tiny, mode=DESCRIPTOR)
    pts = tiny.enumerate()
    sur = PlainGP(); sur.fit(enc.encode(pts), np.array([1.0, 2.0, 3.0]))
    r = next_recommendation(tiny, enc, sur, n=1, acquisition="ei", seed=0,
                            observed=pts)
    assert len(r.points) == 0 and not r.complete
    assert any("already been run" in n for n in r.notes)
check("exhausted pool returns empty with a note, not a crash", _exhausted_pool)

def _constraints_in_mixed_path():
    s = SearchSpace([Factor("l", CATEGORICAL, levels=["a", "b"]),
                     Factor("t", CONTINUOUS, bounds=(20, 120))],
                    constraints=[Constraint({"t": 1.0}, 60.0)])
    enc = Encoder(s, mode=MIXED)
    pts = initial_design(s, 10, method="sobol", seed=0).points
    assert all(p["t"] <= 60.0 + 1e-9 for p in pts)
    sur = MixedGP(enc.cat_dims)
    sur.fit(enc.encode(pts), np.array([f(p) for p in pts]))
    r = next_recommendation(s, enc, sur, n=1, acquisition="ei", seed=0)
    assert all(p["t"] <= 60.0 + 1e-6 for p in r.points), r.points
check("constraints hold on the optimised path", _constraints_in_mixed_path)

def _predictions_attached():
    enc, sur, pts, y = _fitted(ENUM, DESCRIPTOR, "plain", n=10, method="maxmin")
    r = next_recommendation(ENUM, enc, sur, n=3, acquisition="qei", seed=0,
                            observed=pts)
    assert len(r.predicted_mean) == 3 and len(r.predicted_std) == 3
    assert all(math.isfinite(v) for v in r.predicted_mean + r.predicted_std)
    assert r.best_f == float(np.max(y))
check("recommendation carries prediction, uncertainty and best_f", _predictions_attached)

def _serialisable():
    import json
    enc, sur, pts, y = _fitted(ENUM, DESCRIPTOR, "plain", n=10, method="maxmin")
    r = next_recommendation(ENUM, enc, sur, n=2, acquisition="qei", seed=0,
                            observed=pts)
    d = json.loads(json.dumps(r.to_dict()))
    assert d["n_returned"] == 2 and d["complete"] is True
check("recommendation is json-serialisable", _serialisable)

def _ts_cap_recorded():
    big = SearchSpace([Factor("a%d" % i, CATEGORICAL, levels=list("abcdefgh"))
                       for i in range(5)])          # 32768 candidates
    enc = Encoder(big, mode=DESCRIPTOR)
    pts = initial_design(big, 8, method="random", seed=0).points
    sur = PlainGP()
    sur.fit(enc.encode(pts), np.arange(8, dtype=float))
    r = next_recommendation(big, enc, sur, n=2, acquisition="ts", seed=0)
    assert any("capped" in n for n in r.notes), r.notes
    assert len(r.points) == 2
check("Thompson candidate cap is recorded, not silent", _ts_cap_recorded)

print("PASS %d" % len(P))
for n, m in F: print("FAIL %s\n     %s" % (n, m))
print("FAILED %d" % len(F))
sys.exit(1 if F else 0)
