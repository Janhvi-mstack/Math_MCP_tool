"""Stage 3 tests -- initial design."""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from maths_tools.space import *
from maths_tools.design import *
from maths_tools.design import PAIRWISE_POOL_LIMIT

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

ENUM = SearchSpace([Factor("l", CATEGORICAL, levels=list("abcd")),
                    Factor("b", CATEGORICAL, levels=list("xyz")),
                    Factor("n", INTEGER, bounds=(1,4))])
CONT = SearchSpace([Factor("l", CATEGORICAL, levels=list("abcd")),
                    Factor("t", CONTINUOUS, bounds=(20,120)),
                    Factor("c", CONTINUOUS, bounds=(0.01,1.0), transform=LOG)])

def _determinism():
    for m in METHODS:
        sp = ENUM if m in (MAXMIN, KMEDOIDS) else CONT
        a = initial_design(sp, 5, method=m, seed=7).points
        b = initial_design(sp, 5, method=m, seed=7).points
        assert a == b, m
check("all methods deterministic for a fixed seed", _determinism)

def _seed_matters():
    for m in (RANDOM, SOBOL):
        a = initial_design(CONT, 6, method=m, seed=1).points
        b = initial_design(CONT, 6, method=m, seed=2).points
        assert a != b, m
check("seed changes random and sobol", _seed_matters)

def _auto_dispatch():
    """AUTO must follow the benchmark, not the original guess."""
    assert initial_design(ENUM, 4, method=AUTO).method == KMEDOIDS
    assert initial_design(CONT, 4, method=AUTO).method == SOBOL
check("auto picks k-medoids when enumerable, sobol otherwise", _auto_dispatch)

def _pool_methods_need_pool():
    for m in (MAXMIN, KMEDOIDS):
        raises(SpaceError, lambda m=m: initial_design(CONT, 4, method=m),
               "needs a finite pool")
check("maxmin/kmedoids refuse a non-enumerable space", _pool_methods_need_pool)

def _maxmin_is_diverse():
    """The property we actually claim: maxmin spreads across levels."""
    d = initial_design(ENUM, 4, method=MAXMIN, seed=0)
    assert len({p["l"] for p in d}) == 4, [p["l"] for p in d]
check("maxmin n=4 covers all 4 ligand levels", _maxmin_is_diverse)

def _maxmin_beats_random_on_coverage():
    """Averaged over seeds, maxmin covers more distinct levels than random."""
    def cover(method, seed):
        d = initial_design(ENUM, 4, method=method, seed=seed)
        return len({p["l"] for p in d})
    mm = np.mean([cover(MAXMIN, s) for s in range(20)])
    rd = np.mean([cover(RANDOM, s) for s in range(20)])
    assert mm > rd, (mm, rd)
    assert mm == 4.0, mm
check("maxmin > random on level coverage over 20 seeds", _maxmin_beats_random_on_coverage)

def _sobol_projection():
    """Every Sobol point gets its own value on each continuous axis."""
    d = initial_design(CONT, 16, method=SOBOL, seed=0)
    for name in ("t", "c"):
        vals = [p[name] for p in d]
        assert len(set(vals)) == len(vals), name
check("sobol gives distinct values on every continuous axis", _sobol_projection)

def _sobol_spread():
    """Sobol should cover the range more evenly than random.

    Measured as the largest gap between consecutive sorted values -- the
    clumping that random reliably produces.
    """
    def worst_gap(method, seed, n=16):
        d = initial_design(CONT, n, method=method, seed=seed)
        v = np.sort([(p["t"] - 20) / 100 for p in d])
        edges = np.concatenate([[0.0], v, [1.0]])
        return np.diff(edges).max()
    sob = np.mean([worst_gap(SOBOL, s) for s in range(20)])
    rnd = np.mean([worst_gap(RANDOM, s) for s in range(20)])
    assert sob < rnd, (sob, rnd)
check("sobol leaves smaller gaps than random over 20 seeds", _sobol_spread)

def _sobol_any_n():
    """You can stop at any count -- no committing to 4, 9, 16 like a grid."""
    for n in (1, 2, 3, 5, 7, 11, 13):
        d = initial_design(CONT, n, method=SOBOL, seed=0)
        assert len(d) == n and d.complete, n
check("sobol works for any n, not just powers of 2", _sobol_any_n)

def _sobol_is_prefix_stable():
    """The first k of an n-point Sobol design are the k-point design."""
    big = initial_design(CONT, 16, method=SOBOL, seed=3).points
    for k in (1, 4, 9):
        small = initial_design(CONT, k, method=SOBOL, seed=3).points
        assert small == big[:k], k
check("sobol prefixes are stable as n grows", _sobol_is_prefix_stable)

def _n_exceeds_pool():
    small = SearchSpace([Factor("l", CATEGORICAL, levels=["a","b"])])
    d = initial_design(small, 10, method=MAXMIN, seed=0)
    assert len(d) == 2 and not d.complete
    assert any("pool holds" in n for n in d.notes), d.notes
check("n > pool returns all and records a note", _n_exceeds_pool)

def _constraints_respected():
    s = SearchSpace([Factor("a", CONTINUOUS, bounds=(0,10)),
                     Factor("b", CONTINUOUS, bounds=(0,10))],
                    constraints=[Constraint({"a":1.0,"b":1.0}, 5.0)])
    for m in (SOBOL, RANDOM):
        d = initial_design(s, 8, method=m, seed=0)
        assert d.complete, (m, d.notes)
        assert all(p["a"] + p["b"] <= 5.0 + 1e-9 for p in d), m
        assert any("rejected by constraints" in n for n in d.notes), (m, d.notes)
check("constraints respected and rejections recorded", _constraints_respected)

def _enumerable_constraints():
    s = SearchSpace([Factor("a", INTEGER, bounds=(1,10)),
                     Factor("b", INTEGER, bounds=(1,10))],
                    constraints=[Constraint({"a":1.0,"b":1.0}, 6.0)])
    for m in (MAXMIN, KMEDOIDS, RANDOM):
        d = initial_design(s, 5, method=m, seed=0)
        assert all(p["a"] + p["b"] <= 6 for p in d), m
check("enumerated pool respects constraints for every method", _enumerable_constraints)

def _points_are_valid():
    for sp, ms in ((ENUM, METHODS), (CONT, (RANDOM, SOBOL))):
        for m in ms:
            for p in initial_design(sp, 6, method=m, seed=0):
                sp.validate_point(p)
check("every returned point is a legal experiment", _points_are_valid)

def _log_factor_spread():
    """On a log axis the low decade gets explored, unlike a linear one."""
    d = initial_design(CONT, 32, method=SOBOL, seed=0)
    low = sum(1 for p in d if p["c"] < 0.1)   # bottom 2 of 2 decades
    assert low >= 8, low       # ~half expected; linear scaling would give ~3
check("log factor explores the low decade", _log_factor_spread)

def _pool_cap_recorded():
    """A pool over the pairwise limit is subsampled, and says so."""
    big = SearchSpace([Factor("a%d"%i, CATEGORICAL, levels=list("abcdefgh"))
                       for i in range(5)])          # 32768 > 20000
    assert big.is_enumerable
    d = initial_design(big, 5, method=MAXMIN, seed=0)
    assert len(d) == 5
    assert any("subsampled" in n for n in d.notes), d.notes
check("oversized pool is subsampled with a recorded note", _pool_cap_recorded)

def _bad_args():
    raises(ValueError, lambda: initial_design(ENUM, 0), "n must be")
    raises(ValueError, lambda: initial_design(ENUM, 3, method="vibes"), "method must be")
check("bad arguments raise", _bad_args)

def _kmedoids_returns_real_members():
    pool = ENUM.enumerate()
    d = initial_design(ENUM, 5, method=KMEDOIDS, seed=0)
    assert len(d) == 5 and len({tuple(p.items()) for p in d}) == 5
    for p in d:
        assert p in pool
check("k-medoids returns distinct real pool members", _kmedoids_returns_real_members)

print("PASS %d" % len(P))
for n, m in F: print("FAIL %s\n     %s" % (n, m))
print("FAILED %d" % len(F))
sys.exit(1 if F else 0)
