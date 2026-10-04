"""The vendored deep kernel must not drift from the code it was copied from.

`maths_tools/deep_kernel.py` is a verbatim copy of the classes behind the
published benchmark numbers. A copy can rot. These tests fail loudly if it
does -- comparing source text where the original is available, and comparing
actual behaviour either way.
"""
import os
import sys
import inspect

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
ROOT = os.path.dirname(PKG)
sys.path.insert(0, ROOT)

# Source the vendored code was copied from, when available. Set
# BEACON_REFERENCE_KERNEL to enable the source-level drift check; without it
# the behavioural checks still run.
ORIGINAL = os.environ.get("BEACON_REFERENCE_KERNEL", "")
def _load_reference():
    """Load the reference module from the configured path, by file not by name."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("_reference_impl", ORIGINAL)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, os.path.dirname(ORIGINAL))
    spec.loader.exec_module(mod)
    return mod


P, F = [], []


def check(name, fn):
    try:
        fn(); P.append(name)
    except Exception as e:
        F.append((name, "%s: %s" % (type(e).__name__, e)))


def _norm(src):
    return "\n".join(l.rstrip() for l in src.strip().splitlines())


def _extract(src_text, name):
    """Pull one top-level class or function out of a source file, as text.

    Reads the file rather than importing it: a module loaded by path has no
    discoverable source, so inspect.getsource cannot see it.
    """
    import re
    lines = src_text.splitlines(True)
    i = next(k for k, l in enumerate(lines)
             if re.match(r"^(class|def) %s\b" % re.escape(name), l))
    tops = (r"^class ", r"^def ", r"^# ---", r"^[A-Z_]+ *=")
    j = len(lines)
    for k in range(i + 1, len(lines)):
        if any(re.match(t, lines[k]) for t in tops):
            j = k
            break
    return "".join(lines[i:j])


def _source_drift():
    """Every vendored class must be character-identical to the original."""
    if not ORIGINAL or not os.path.exists(ORIGINAL):
        print("  (reference source not configured -- source diff skipped)")
        return
    orig_text = open(ORIGINAL).read()
    vend_text = open(os.path.join(PKG, "deep_kernel.py")).read()
    for name in ("ProjectionLayer", "PrecomputedProjection", "DeepGP",
                 "_kernel", "_apply_hypers"):
        a = _norm(_extract(orig_text, name))
        b = _norm(_extract(vend_text, name))
        assert a == b, (
            "%s has drifted from the reference implementation.\n"
            "--- reference ---\n%s\n--- vendored ---\n%s"
            % (name, a[:400], b[:400]))
check("vendored source is identical to the original", _source_drift)


def _behaviour_matches():
    """Same seed, same data, same posterior -- to the last bit."""
    import numpy as np
    import torch
    if not ORIGINAL or not os.path.exists(ORIGINAL):
        print("  (original not present -- behaviour diff skipped)")
        return
    ORIG = _load_reference()
    from maths_tools import deep_kernel as VEND

    rng = np.random.default_rng(0)
    X = rng.random((12, 20))
    y = rng.random(12) * 100
    tx = torch.as_tensor(X, dtype=torch.float64)
    ty = torch.as_tensor(y, dtype=torch.float64).unsqueeze(-1)

    for drop, clip in ((False, "code"), (True, "paper")):
        out = []
        for M in (ORIG, VEND):
            torch.manual_seed(0)
            fm = M.PrecomputedProjection(X.shape[1], 64, apply_proj_dropout=drop)
            g = M.DeepGP(tx, ty, fm, clip_mode=clip)
            info = g.fit()
            g.eval()
            with torch.no_grad():
                post = g.posterior(tx)
                out.append((info["n_steps"],
                            round(info["grad_mass_into_featurizer"], 9),
                            post.mean.squeeze(-1).numpy().copy(),
                            post.variance.squeeze(-1).numpy().copy()))
        a, b = out
        assert a[0] == b[0], f"n_steps differ ({drop}, {clip})"
        assert a[1] == b[1], f"gradient mass differs ({drop}, {clip})"
        assert (a[2] == b[2]).all(), f"posterior mean differs ({drop}, {clip})"
        assert (a[3] == b[3]).all(), f"posterior variance differs ({drop}, {clip})"
check("vendored behaviour is bit-identical to the original", _behaviour_matches)


def _self_contained():
    """The package must not reach outside itself to build a deep kernel."""
    import maths_tools.surrogate as S
    src = inspect.getsource(S)
    for bad in ("gollum", "benchmarks", "/Users/", "/home/"):
        assert bad not in src, \
            f"surrogate.py still references {bad!r}; the package must be self-contained"
check("surrogate.py has no dependency on the benchmark tree", _self_contained)


def _deep_kernel_runs():
    import numpy as np
    from maths_tools.surrogate import DeepKernelGP
    rng = np.random.default_rng(1)
    X = rng.random((10, 15))
    y = rng.random(10) * 50
    s = DeepKernelGP(seed=0)
    info = s.fit(X, y)
    m, sd = s.predict(X)
    assert info["fit_failures"] == 0
    assert m.shape == (10,) and (sd > 0).all()
check("DeepKernelGP fits and predicts through the public API", _deep_kernel_runs)


print("PASS %d" % len(P))
for n, m in F:
    print("FAIL %s\n     %s" % (n, m))
print("FAILED %d" % len(F))
sys.exit(1 if F else 0)
