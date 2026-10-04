"""Stage 4 tests -- priors, piBO decay, RGPE safety."""
import sys, os, math
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from maths_tools.space import *
from maths_tools.encode import Encoder, DESCRIPTOR, MIXED
from maths_tools.surrogate import PlainGP, MixedGP
from maths_tools.design import initial_design, RANDOM
from maths_tools.acquisition import next_recommendation, AcquisitionError
from maths_tools.priors import *

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

PERFECT = CategoricalPrior({"l": {"a": 1.0, "b": 2.0, "c": 4.0, "d": 1.5}})
REVERSED = CategoricalPrior({"l": {"a": 4.0, "b": 2.0, "c": 1.0, "d": 3.0}})

# ---------------- densities ----------------
def _uniform():
    pts = S.enumerate()[:5]
    assert np.allclose(UniformPrior().density(pts), 1.0)
check("uniform prior is all ones", _uniform)

def _categorical_product():
    s = SearchSpace([Factor("l", CATEGORICAL, levels=["a", "b"]),
                     Factor("b", CATEGORICAL, levels=["x", "y"])])
    pr = CategoricalPrior({"l": {"a": 2.0, "b": 1.0}, "b": {"x": 3.0, "y": 1.0}})
    d = pr.density([{"l": "a", "b": "x"}, {"l": "b", "b": "y"}])
    assert np.allclose(d, [6.0, 1.0]), d
check("categorical densities multiply across factors", _categorical_product)

def _default_weight():
    pr = CategoricalPrior({"l": {"a": 5.0}}, default=1.0)
    d = pr.density([{"l": "a", "t": 20}, {"l": "b", "t": 20}])
    assert np.allclose(d, [5.0, 1.0])
check("unmentioned levels take the default weight", _default_weight)

def _gaussian():
    pr = GaussianPrior(center={"t": 80.0}, width={"t": 10.0})
    d = pr.density([{"t": 80.0}, {"t": 90.0}, {"t": 120.0}])
    assert d[0] == 1.0 and d[0] > d[1] > d[2]
    assert abs(d[1] - math.exp(-0.5)) < 1e-12
check("gaussian prior peaks at its centre", _gaussian)

def _composite():
    c = CompositePrior([CategoricalPrior({"l": {"a": 2.0, "b": 1.0}}),
                        GaussianPrior(center={"t": 80.0}, width={"t": 10.0})])
    d = c.density([{"l": "a", "t": 80.0}, {"l": "b", "t": 80.0}])
    assert np.allclose(d, [2.0, 1.0])
check("composite prior multiplies its parts", _composite)

# ---------------- validation ----------------
def _validation():
    raises(PriorError, lambda: CategoricalPrior({"zzz": {"a": 1.0}}).validate_against(S),
           "not in the space")
    raises(PriorError, lambda: CategoricalPrior({"l": {"zzz": 1.0}}).validate_against(S),
           "does not declare")
    raises(PriorError, lambda: CategoricalPrior({"t": {1: 1.0}}).validate_against(S),
           "use GaussianPrior")
    raises(PriorError, lambda: GaussianPrior({"l": 1.0}, {"l": 1.0}).validate_against(S),
           "use CategoricalPrior")
    raises(PriorError, lambda: CategoricalPrior({"l": {"a": -1.0}}), "non-negative")
    raises(PriorError, lambda: GaussianPrior({"t": 1.0}, {"t": 0.0}), "must be positive")
check("priors reject mismatched or invalid definitions", _validation)

# ---------------- rank agreement ----------------
def _tau_extremes():
    pts, y = S.enumerate(), None
    pts = [{"l": k, "t": 20} for k in "abcd"]
    y = [TRUTH[p["l"]] for p in pts]
    assert abs(rank_agreement(PERFECT, pts, y) - 1.0) < 1e-12
    assert abs(rank_agreement(REVERSED, pts, y) + 1.0) < 1e-12
    assert rank_agreement(UniformPrior(), pts, y) == 0.0
check("tau is +1 for a perfect prior, -1 for a reversed one", _tau_extremes)

def _tau_third_of_four():
    """Our measured situation: the genuinely best option ranked third of four.

    Truth, best first:  c(90)  b(40)  d(30)  a(10)
    Prior, best first:  b(4)   d(3)   c(2)   a(1)     <- c demoted to third

    Pairs: a-b concordant, a-c concordant, a-d concordant, b-c DISCORDANT,
    b-d concordant, c-d DISCORDANT.  tau = (4 - 2) / 6 = 1/3.
    """
    pts = [{"l": k, "t": 20} for k in "abcd"]
    y = [TRUTH[p["l"]] for p in pts]
    kg = CategoricalPrior({"l": {"a": 1.0, "b": 4.0, "c": 2.0, "d": 3.0}})
    assert abs(rank_agreement(kg, pts, y) - 1.0 / 3.0) < 1e-12
    assert rgpe_weight(kg, pts, y) == rank_agreement(kg, pts, y)
check("tau computed by hand matches for best-ranked-third", _tau_third_of_four)

def _tau_length_mismatch():
    raises(PriorError, lambda: rank_agreement(PERFECT, [{"l": "a"}], [1.0, 2.0]),
           "1 points but 2")
check("tau refuses mismatched lengths", _tau_length_mismatch)

# ---------------- rgpe ----------------
def _rgpe():
    pts = [{"l": k, "t": 20} for k in "abcd"]
    y = [TRUTH[p["l"]] for p in pts]
    assert rgpe_weight(PERFECT, pts, y) == 1.0
    assert rgpe_weight(REVERSED, pts, y) == 0.0
    # below the evidence threshold the prior keeps full weight
    assert rgpe_weight(REVERSED, pts[:2], y[:2], min_observations=4) == 1.0
check("rgpe weights a good prior 1, a reversed prior 0", _rgpe)

def _pibo_decay():
    assert pibo_exponent(0) == 1.0 and pibo_exponent(1) == 1.0
    assert pibo_exponent(10) == 0.1
    assert pibo_exponent(10, beta=5.0) == 0.5
    xs = [pibo_exponent(t) for t in (1, 5, 20, 50)]
    assert all(a > b for a, b in zip(xs, xs[1:]))
    raises(PriorError, lambda: pibo_exponent(5, beta=-1.0), "beta must be")
check("piBO exponent decays monotonically with observations", _pibo_decay)

def _pibo_rgpe_zero():
    pts = [{"l": k, "t": 20} for k in "abcd"]
    assert np.allclose(pibo_weights(REVERSED, pts, 2, 1.0, rgpe=0.0), 1.0)
    raises(PriorError, lambda: pibo_weights(PERFECT, pts, 2, 1.0, rgpe=2.0), "in [0, 1]")
check("rgpe=0 makes the piBO multiplier exactly one", _pibo_rgpe_zero)

# ---------------- prior-led design ----------------
def _confidence_zero_is_random():
    """The no-prior control is the same code path, so the comparison is fair."""
    a = prior_initial_design(S, 6, PERFECT, confidence=0.0, seed=3).points
    b = initial_design(S, 6, method=RANDOM, seed=3).points
    assert a == b
check("confidence=0 reproduces random exactly", _confidence_zero_is_random)

def _prior_biases_selection():
    counts = {k: 0 for k in "abcd"}
    for seed in range(40):
        for p in prior_initial_design(S, 5, PERFECT, confidence=2.0, seed=seed):
            counts[p["l"]] += 1
    assert counts["c"] > counts["a"], counts
check("a prior favouring c selects c more often", _prior_biases_selection)

def _all_zero_prior_falls_back():
    zero = CategoricalPrior({"l": {k: 0.0 for k in "abcd"}}, default=0.0)
    d = prior_initial_design(S, 4, zero, confidence=1.0, seed=0)
    assert len(d) == 4
    assert any("zero weight to every candidate" in n for n in d.notes), d.notes
check("an all-zero prior falls back to uniform with a note", _all_zero_prior_falls_back)

def _prior_design_on_continuous():
    c = SearchSpace([Factor("t", CONTINUOUS, bounds=(20, 120))])
    pr = GaussianPrior(center={"t": 100.0}, width={"t": 5.0})
    d = prior_initial_design(c, 8, pr, confidence=1.0, seed=0)
    assert len(d) == 8
    assert np.mean([p["t"] for p in d]) > 70.0
    assert any("Sobol pool" in n for n in d.notes)
check("prior design works on a non-enumerable space", _prior_design_on_continuous)

# ---------------- the safety actually mattering ----------------
def _fit():
    enc = Encoder(S, mode=DESCRIPTOR)
    pts = initial_design(S, 8, method="maxmin", seed=0).points
    y = np.array([f(p) for p in pts])
    sur = PlainGP(); sur.fit(enc.encode(pts), y)
    return enc, sur, pts, y

def _bad_prior_does_damage():
    """With the dial turned up, a reversed prior pulls the pick off the optimum."""
    enc, sur, pts, y = _fit()
    clean = next_recommendation(S, enc, sur, n=1, acquisition="ei", seed=0,
                                observed=pts).points[0]
    harmed = next_recommendation(S, enc, sur, n=1, acquisition="ei", seed=0,
                                 observed=pts, prior=REVERSED, prior_beta=50.0,
                                 prior_rgpe=1.0).points[0]
    assert clean["l"] == "c", clean
    assert harmed["l"] != "c", harmed        # the damage is real
check("a reversed prior trusted blindly moves the pick off the optimum", _bad_prior_does_damage)

def _rgpe_prevents_the_damage():
    """Same prior, same dial -- but checked against the data first."""
    enc, sur, pts, y = _fit()
    r = next_recommendation(S, enc, sur, n=1, acquisition="ei", seed=0,
                            observed=pts, observed_y=y,
                            prior=REVERSED, prior_beta=50.0)
    assert r.points[0]["l"] == "c", r.points[0]
    assert any("switched off" in n for n in r.notes), r.notes
    assert any("RGPE weight of 0.00" in n for n in r.notes), r.notes
check("RGPE prevents that damage at the same dial setting", _rgpe_prevents_the_damage)

def _good_prior_survives_rgpe():
    """A good prior keeps most of its weight and does not move the answer.

    Not all of it: PERFECT ranks ligands only, and the yield also depends on
    temperature, so some observed pairs genuinely disagree with it. Around
    0.74 is the honest agreement here -- what matters is that it stays high
    and the recommendation is unharmed.
    """
    enc, sur, pts, y = _fit()
    w = rgpe_weight(PERFECT, pts, y)
    assert w > 0.7, w
    r = next_recommendation(S, enc, sur, n=1, acquisition="ei", seed=0,
                            observed=pts, observed_y=y,
                            prior=PERFECT, prior_beta=50.0)
    assert r.points[0]["l"] == "c", r.points[0]
    assert any("RGPE weight of %.2f" % w in n for n in r.notes), r.notes
check("a good prior keeps most of its weight through RGPE", _good_prior_survives_rgpe)

def _ligand_only_prior_is_exact():
    """On a space where the prior is genuinely complete, the weight is 1.00."""
    s1 = SearchSpace([Factor("l", CATEGORICAL, levels=list("abcd"))])
    pts = s1.enumerate()
    y = [TRUTH[p["l"]] for p in pts]
    assert rgpe_weight(PERFECT, pts, y) == 1.0
    assert rgpe_weight(REVERSED, pts, y) == 0.0
check("a complete prior earns exactly 1.00 on its own space", _ligand_only_prior_is_exact)

def _unchecked_prior_is_flagged():
    enc, sur, pts, y = _fit()
    r = next_recommendation(S, enc, sur, n=1, acquisition="ei", seed=0,
                            observed=pts, prior=PERFECT)
    assert any("without being checked" in n for n in r.notes), r.notes
check("a prior used without yields says so in the notes", _unchecked_prior_is_flagged)

# ---------------- refusals, not silent ignores ----------------
def _prior_with_ts_refused():
    enc, sur, pts, y = _fit()
    raises(AcquisitionError, lambda: next_recommendation(S, enc, sur, n=2,
           acquisition="ts", seed=0, observed=pts, prior=PERFECT),
           "cannot be applied to Thompson")
check("prior + Thompson is refused, not ignored", _prior_with_ts_refused)

def _prior_on_optimised_path_refused():
    c = SearchSpace([Factor("l", CATEGORICAL, levels=list("abcd")),
                     Factor("t", CONTINUOUS, bounds=(20, 120))])
    enc = Encoder(c, mode=MIXED)
    pts = initial_design(c, 10, method="sobol", seed=0).points
    sur = MixedGP(enc.cat_dims); sur.fit(enc.encode(pts), np.array([f(p) for p in pts]))
    raises(AcquisitionError, lambda: next_recommendation(c, enc, sur, n=1,
           acquisition="ei", seed=0, prior=PERFECT),
           "only supported on the table path")
check("prior on the optimised path is refused, not ignored", _prior_on_optimised_path_refused)

def _prior_with_candidates_works():
    """Supplying a pool is the documented way to use a prior with a mixed space."""
    c = SearchSpace([Factor("l", CATEGORICAL, levels=list("abcd")),
                     Factor("t", CONTINUOUS, bounds=(20, 120))])
    enc = Encoder(c, mode=MIXED)
    pts = initial_design(c, 10, method="sobol", seed=0).points
    y = np.array([f(p) for p in pts])
    sur = MixedGP(enc.cat_dims); sur.fit(enc.encode(pts), y)
    pool = initial_design(c, 200, method="sobol", seed=1).points
    r = next_recommendation(c, enc, sur, n=1, acquisition="ei", seed=0,
                            candidates=pool, observed=pts, observed_y=y,
                            prior=PERFECT, prior_beta=5.0)
    assert len(r.points) == 1 and r.path == "enumerated"
check("prior works on a mixed space via an explicit candidate pool", _prior_with_candidates_works)

print("PASS %d" % len(P))
for n, m in F: print("FAIL %s\n     %s" % (n, m))
print("FAILED %d" % len(F))
sys.exit(1 if F else 0)
