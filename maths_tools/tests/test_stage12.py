"""Stage 1 + 2 tests. Plain asserts, no pytest dependency."""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from maths_tools.space import *
from maths_tools.encode import Encoder, EncodingError, DESCRIPTOR, MIXED

P, F = [], []
def check(name, fn):
    try:
        fn(); P.append(name)
    except Exception as e:
        F.append((name, "%s: %s" % (type(e).__name__, e)))

def raises(exc, fn, frag=None):
    try:
        fn()
    except exc as e:
        if frag and frag.lower() not in str(e).lower():
            raise AssertionError("wrong message: %s" % e)
        return
    raise AssertionError("expected %s" % exc.__name__)

# ---------- Factor validation ----------
check("continuous needs bounds", lambda: raises(SpaceError,
    lambda: Factor("t", CONTINUOUS), "requires bounds"))
check("continuous rejects levels", lambda: raises(SpaceError,
    lambda: Factor("t", CONTINUOUS, bounds=(0,1), levels=[1,2]), "must not have levels"))
check("categorical needs levels", lambda: raises(SpaceError,
    lambda: Factor("l", CATEGORICAL), "requires levels"))
check("categorical needs >=2 levels", lambda: raises(SpaceError,
    lambda: Factor("l", CATEGORICAL, levels=["a"]), "at least 2"))
check("duplicate levels rejected", lambda: raises(SpaceError,
    lambda: Factor("l", CATEGORICAL, levels=["a","a"]), "unique"))
check("bad bounds order", lambda: raises(SpaceError,
    lambda: Factor("t", CONTINUOUS, bounds=(5,5)), "must exceed"))
check("log needs positive low", lambda: raises(SpaceError,
    lambda: Factor("c", CONTINUOUS, bounds=(0,1), transform=LOG), "positive"))
check("integer bounds whole", lambda: raises(SpaceError,
    lambda: Factor("n", INTEGER, bounds=(1.5,4)), "whole"))
check("unknown type", lambda: raises(SpaceError,
    lambda: Factor("x", "vibes", bounds=(0,1)), "type must be"))

# ---------- descriptors ----------
check("partial descriptors refused", lambda: raises(SpaceError,
    lambda: Factor("l", CATEGORICAL, levels=["a","b"], descriptors={"a":[1.0]}),
    "missing for level"))
check("ragged descriptors refused", lambda: raises(SpaceError,
    lambda: Factor("l", CATEGORICAL, levels=["a","b"],
                   descriptors={"a":[1.0],"b":[1.0,2.0]}), "differing lengths"))
check("nan descriptors refused", lambda: raises(SpaceError,
    lambda: Factor("l", CATEGORICAL, levels=["a","b"],
                   descriptors={"a":[float('nan')],"b":[1.0]}), "non-finite"))
check("extra descriptor level refused", lambda: raises(SpaceError,
    lambda: Factor("l", CATEGORICAL, levels=["a","b"],
                   descriptors={"a":[1.0],"b":[2.0],"c":[3.0]}), "unknown level"))

# ---------- SearchSpace ----------
def _cardinality():
    s = SearchSpace([Factor("l", CATEGORICAL, levels=list("abcd")),
                     Factor("b", CATEGORICAL, levels=list("xyz")),
                     Factor("n", INTEGER, bounds=(1,4))])
    assert s.cardinality == 4*3*4 == 48, s.cardinality
    assert s.is_enumerable
    assert len(s.enumerate()) == 48
check("cardinality + enumerate", _cardinality)

def _infinite():
    s = SearchSpace([Factor("t", CONTINUOUS, bounds=(20,120))])
    assert s.cardinality is None
    assert not s.is_enumerable
    raises(SpaceError, s.enumerate, "cannot be enumerated")
check("continuous is not enumerable", _infinite)

def _limit():
    s = SearchSpace([Factor("a%d"%i, CATEGORICAL, levels=list("abcdefghij"))
                     for i in range(6)])
    assert s.cardinality == 10**6
    assert not s.is_enumerable          # above ENUMERATION_LIMIT
    raises(SpaceError, s.enumerate, "enumeration limit")
check("enumeration limit raises not truncates", _limit)

def _dupe_names():
    raises(SpaceError, lambda: SearchSpace([
        Factor("x", CONTINUOUS, bounds=(0,1)),
        Factor("x", CONTINUOUS, bounds=(0,1))]), "duplicate")
check("duplicate factor names", _dupe_names)

# ---------- constraints ----------
def _constraints():
    s = SearchSpace(
        [Factor("a", INTEGER, bounds=(1,10)), Factor("b", INTEGER, bounds=(1,10))],
        constraints=[Constraint({"a":1.0,"b":1.0}, 10.0)])
    rows = s.enumerate()
    assert all(r["a"]+r["b"] <= 10 for r in rows)
    assert len(rows) < 100
    assert len(s.enumerate(apply_constraints=False)) == 100
    raises(SpaceError, lambda: s.validate_point({"a":9,"b":9}), "violates a constraint")
check("linear constraint filters + validates", _constraints)

def _constraint_on_categorical():
    raises(SpaceError, lambda: SearchSpace(
        [Factor("l", CATEGORICAL, levels=["a","b"])],
        constraints=[Constraint({"l":1.0}, 1.0)]), "numeric factors only")
check("constraint on categorical refused", _constraint_on_categorical)

# ---------- validate_point ----------
def _validate_point():
    s = SearchSpace([Factor("t", CONTINUOUS, bounds=(20,120)),
                     Factor("l", CATEGORICAL, levels=["a","b"])])
    s.validate_point({"t": 50.0, "l": "a"})
    raises(SpaceError, lambda: s.validate_point({"t": 50.0}), "missing")
    raises(SpaceError, lambda: s.validate_point({"t":50.0,"l":"a","z":1}), "unknown")
    raises(SpaceError, lambda: s.validate_point({"t": 500.0, "l":"a"}), "outside bounds")
    raises(SpaceError, lambda: s.validate_point({"t": 50.0, "l":"q"}), "not one of")
check("validate_point", _validate_point)

# ---------- JSON round trip ----------
def _json():
    s = SearchSpace([
        Factor("t", CONTINUOUS, bounds=(20,120), unit="C"),
        Factor("c", CONTINUOUS, bounds=(0.01,1.0), transform=LOG),
        Factor("n", INTEGER, bounds=(1,5)),
        Factor("load", ORDINAL, levels=[1,2,5,10]),
        Factor("lig", CATEGORICAL, levels=["a","b"],
               descriptors={"a":[1.0,2.0],"b":[3.0,4.0]},
               descriptor_names=["cone","tep"]),
    ], constraints=[Constraint({"t":1.0}, 100.0)], name="rt")
    s2 = SearchSpace.from_json(s.to_json())
    assert s2.to_dict() == s.to_dict()
    assert s2.cardinality is None
check("space json round trip", _json)

# ---------- Encoder: descriptor mode ----------
def _enc_desc():
    s = SearchSpace([
        Factor("t", CONTINUOUS, bounds=(0,100)),
        Factor("lig", CATEGORICAL, levels=["a","b","c"]),
        Factor("load", ORDINAL, levels=[1,2,5,10]),
    ])
    e = Encoder(s, mode=DESCRIPTOR)
    assert e.n_columns == 1 + 3 + 1, e.n_columns
    assert e.cat_dims == []
    X = e.encode([{"t":50.0,"lig":"b","load":5}])
    assert np.allclose(X[0], [0.5, 0,1,0, 2/3]), X[0]
    assert e.column_names == ["t","lig=a","lig=b","lig=c","load"]
check("descriptor-mode encoding", _enc_desc)

def _onehot_order():
    """One-hot column order follows DECLARED level order, not first-seen."""
    s = SearchSpace([Factor("l", CATEGORICAL, levels=["a","b","c"])])
    e = Encoder(s, mode=DESCRIPTOR)
    X = e.encode([{"l":"c"},{"l":"a"}])
    assert np.allclose(X[0], [0,0,1]) and np.allclose(X[1], [1,0,0])
check("one-hot uses declared order", _onehot_order)

def _log_encoding():
    s = SearchSpace([Factor("c", CONTINUOUS, bounds=(0.01,1.0), transform=LOG)])
    e = Encoder(s, mode=DESCRIPTOR)
    X = e.encode([{"c":0.01},{"c":0.1},{"c":1.0}])
    assert np.allclose(X.ravel(), [0.0, 0.5, 1.0]), X.ravel()
check("log transform is evenly spaced in decades", _log_encoding)

# ---------- Encoder: mixed mode ----------
def _enc_mixed():
    s = SearchSpace([
        Factor("t", CONTINUOUS, bounds=(0,100)),
        Factor("lig", CATEGORICAL, levels=["a","b","c"]),
    ])
    e = Encoder(s, mode=MIXED)
    assert e.n_columns == 2
    assert e.cat_dims == [1] and e.cont_dims == [0]
    X = e.encode([{"t":25.0,"lig":"c"}])
    assert np.allclose(X[0], [0.25, 2.0]), X[0]
    b = e.bounds()
    assert np.allclose(b[0], [0,0]) and np.allclose(b[1], [1,2]), b
    assert e.categorical_choices() == {1:[0,1,2]}
check("mixed-mode encoding + cat_dims + bounds", _enc_mixed)

# ---------- round trip ----------
def _round_trip():
    s = SearchSpace([
        Factor("t", CONTINUOUS, bounds=(20,120)),
        Factor("c", CONTINUOUS, bounds=(0.01,1.0), transform=LOG),
        Factor("n", INTEGER, bounds=(1,5)),
        Factor("load", ORDINAL, levels=[1,2,5,10]),
        Factor("lig", CATEGORICAL, levels=["a","b","c"]),
        Factor("cat", CATEGORICAL, levels=["p","q"],
               descriptors={"p":[1.0,10.0],"q":[3.0,30.0]}),
    ])
    pts = [{"t":37.5,"c":0.05,"n":3,"load":5,"lig":"b","cat":"q"},
           {"t":20.0,"c":1.0,"n":1,"load":1,"lig":"a","cat":"p"},
           {"t":120.0,"c":0.01,"n":5,"load":10,"lig":"c","cat":"q"}]
    for mode in (DESCRIPTOR, MIXED):
        e = Encoder(s, mode=mode)
        back = e.decode(e.encode(pts))
        for a, b in zip(pts, back):
            for k in a:
                if isinstance(a[k], float):
                    assert abs(a[k]-b[k]) < 1e-9, (mode,k,a[k],b[k])
                else:
                    assert a[k] == b[k], (mode,k,a[k],b[k])
check("encode->decode round trip, both modes", _round_trip)

# ---------- THE KEY PROPERTY (D3) ----------
def _space_independence():
    """Encoding a point must not depend on what else is in the candidate set.

    This is the EDBO failure mode. Same factor definition, one space with 2
    extra ligands -- the shared levels must encode identically.
    """
    common = dict(descriptors={"a":[1.0,10.0],"b":[2.0,20.0]})
    small = SearchSpace([Factor("t", CONTINUOUS, bounds=(0,100)),
                         Factor("l", CATEGORICAL, levels=["a","b"], **common)])
    Xs = Encoder(small, mode=DESCRIPTOR).encode([{"t":50.0,"l":"a"}])

    # same 2 levels, but the candidate set is 100x bigger via another factor
    big = SearchSpace([Factor("t", CONTINUOUS, bounds=(0,100)),
                       Factor("l", CATEGORICAL, levels=["a","b"], **common),
                       Factor("extra", CATEGORICAL, levels=[str(i) for i in range(50)])])
    Xb = Encoder(big, mode=DESCRIPTOR).encode([{"t":50.0,"l":"a","extra":"0"}])
    assert np.allclose(Xs[0], Xb[0][:Xs.shape[1]]), (Xs[0], Xb[0][:Xs.shape[1]])
check("D3: encoding independent of candidate-set size", _space_independence)

def _pinned_constants():
    """Extending a level set later can reuse the original constants."""
    f1 = Factor("l", CATEGORICAL, levels=["a","b"],
                descriptors={"a":[1.0],"b":[2.0]})
    e1 = Encoder(SearchSpace([f1]), mode=DESCRIPTOR)
    X1 = e1.encode([{"l":"a"},{"l":"b"}])
    const = e1.constants()

    f2 = Factor("l", CATEGORICAL, levels=["a","b","c"],
                descriptors={"a":[1.0],"b":[2.0],"c":[9.0]})
    e_free = Encoder(SearchSpace([f2]), mode=DESCRIPTOR)
    e_pin  = Encoder(SearchSpace([f2]), mode=DESCRIPTOR, _constants=const)
    Xf = e_free.encode([{"l":"a"},{"l":"b"}])
    Xp = e_pin.encode([{"l":"a"},{"l":"b"}])
    assert not np.allclose(Xf, X1)      # free rescales, as expected
    assert np.allclose(Xp, X1)          # pinned reproduces exactly
check("pinned constants survive a level being added", _pinned_constants)

def _determinism():
    s = SearchSpace([Factor("t", CONTINUOUS, bounds=(0,1)),
                     Factor("l", CATEGORICAL, levels=["a","b","c"])])
    pt = [{"t":0.3,"l":"b"}]
    runs = [Encoder(s, mode=DESCRIPTOR).encode(pt) for _ in range(5)]
    assert all(np.array_equal(runs[0], r) for r in runs)
check("encoding is deterministic across constructions", _determinism)

def _constant_descriptor():
    """A descriptor constant over the levels becomes a 0 column, not a NaN."""
    s = SearchSpace([Factor("l", CATEGORICAL, levels=["a","b"],
                            descriptors={"a":[1.0,5.0],"b":[2.0,5.0]})])
    X = Encoder(s, mode=DESCRIPTOR).encode([{"l":"a"},{"l":"b"}])
    assert np.isfinite(X).all()
    assert np.allclose(X[:,1], 0.0)
    assert X.shape[1] == 2   # kept, so column count depends only on definition
check("constant descriptor -> zero column, no div-by-zero", _constant_descriptor)

def _encoder_json():
    s = SearchSpace([Factor("t", CONTINUOUS, bounds=(0,100)),
                     Factor("l", CATEGORICAL, levels=["a","b"])])
    e = Encoder(s, mode=MIXED)
    e2 = Encoder.from_dict(json.loads(json.dumps(e.to_dict())))
    pt = [{"t":42.0,"l":"b"}]
    assert np.allclose(e.encode(pt), e2.encode(pt))
    assert e2.cat_dims == e.cat_dims
check("encoder json round trip", _encoder_json)

def _bad_level_encode():
    s = SearchSpace([Factor("l", CATEGORICAL, levels=["a","b"])])
    e = Encoder(s, mode=DESCRIPTOR)
    raises(SpaceError, lambda: e.encode([{"l":"zzz"}]), "not one of")
check("encoding an undeclared level raises", _bad_level_encode)

# ---------- report ----------
print("PASS %d" % len(P))
for n, m in F:
    print("FAIL %s\n     %s" % (n, m))
print("FAILED %d" % len(F))
sys.exit(1 if F else 0)
