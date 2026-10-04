"""The benchmark metric, with a guard against getting the wrong one.

WHY THIS FILE EXISTS
--------------------
The Mode B target was wrong for a while, and the way it went wrong is worth
recording because the same trap is easy to fall into again.

The reference module defines `q95_threshold(y)` -- the 95th percentile. It is
real, it is used by the reference code, and importing it felt safe: the metric
came from the same file as the published numbers, so it "could not drift".

But Mode B never calls it. Mode B computes its target inline:

    reference protocol    target = 0.95 * float(y.max())

Those are different targets. On bhR1 the percentile admits 40 candidates and
0.95*max admits 14. Measuring against the percentile and comparing the result
to a Mode B number makes our optimiser look about 6 experiments better than it
is, with nothing in the output to suggest anything is wrong.

So this module does NOT import the target. It defines it, and then verifies the
definition against the frozen source TEXT. If anyone edits run_repro.py, the
check fails loudly instead of our numbers quietly drifting.
"""
from __future__ import annotations

import os
import re
from typing import List, Optional, Sequence

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

# Where the reference protocol's source lives, when it is available. Set
# BEACON_REFERENCE_SRC to point the guard at it. Unset -- which is the normal
# case for an installed copy -- and `verify_against_frozen` reports that it
# could not check rather than failing.
FROZEN_RUN_REPRO = os.environ.get("BEACON_REFERENCE_SRC", "")

# The literal expression the frozen Mode B protocol uses, kept here so the
# guard below can confirm it is still what the frozen code does.
FROZEN_TARGET_EXPR = "target = 0.95 * float(y.max())"

PENALTY = 51          # a run that never reaches the target


class MetricMismatch(RuntimeError):
    """The frozen protocol changed, or we are looking at the wrong thing."""


def mode_b_target(y: Sequence[float]) -> float:
    """The Mode B target: 95% of the best yield on the plate.

    NOT the 95th percentile. See the module docstring.
    """
    return 0.95 * float(np.asarray(y, dtype=float).max())


def n95(trace: Sequence[float], target: float) -> Optional[int]:
    """1-based index at which the running best first reaches `target`.

    Same definition as the reference implementation's `n95`, reimplemented here
    so the package carries no external dependency.
    """
    best = -np.inf
    for i, v in enumerate(trace, start=1):
        best = max(best, float(v))
        if best >= target:
            return i
    return None


def n95_penalised(trace: Sequence[float], target: float) -> int:
    k = n95(trace, target)
    return k if k is not None else PENALTY


def verify_against_frozen(path: str = FROZEN_RUN_REPRO, strict: bool = False) -> dict:
    """Confirm our target still matches the reference Mode B protocol.

    Reads the source rather than importing anything: the bug this guards
    against was importing a function that existed, was used by the frozen code,
    and was still the wrong one for this mode.

    strict=True raises when the frozen tree is absent. Default is False so the
    shipped package -- which has no benchmark tree -- can still run this.
    """
    if not path or not os.path.exists(path):
        if strict:
            raise MetricMismatch(
                f"reference source not found at {path!r}. Set "
                f"BEACON_REFERENCE_SRC to the reference protocol's source file."
            )
        return {"checked": False,
                "reason": "reference source not configured "
                          "(set BEACON_REFERENCE_SRC to enable the check)"}

    src = open(path).read()
    if FROZEN_TARGET_EXPR not in src:
        found = re.findall(r"^\s*target\s*=\s*(.+)$", src, re.M)
        raise MetricMismatch(
            f"the frozen Mode B target is no longer {FROZEN_TARGET_EXPR!r}. "
            f"Target assignments now present: {found}. Our metric and the "
            f"published numbers have diverged -- fix this before quoting any "
            f"comparison."
        )

    # And confirm the percentile function is NOT what the Mode B branch uses,
    # since that is the specific mistake being guarded against. Scope this to
    # the else-branch only: q95_threshold legitimately appears elsewhere in the
    # file (the import line, and Mode A's coverage metric), and a guard that
    # fires on those gets switched off rather than heeded.
    lines = src.splitlines()
    try:
        i = next(k for k, l in enumerate(lines) if l.strip() == 'if mode == "A":')
    except StopIteration:
        raise MetricMismatch(
            "could not locate the Mode A/B branch in run_repro.py; the "
            "protocol has been restructured."
        ) from None
    try:
        j = next(k for k in range(i + 1, len(lines)) if lines[k].strip() == "else:")
    except StopIteration:
        raise MetricMismatch("Mode B branch not found in run_repro.py") from None
    indent = len(lines[j]) - len(lines[j].lstrip())
    end = len(lines)
    for k in range(j + 1, len(lines)):
        l = lines[k]
        if l.strip() and (len(l) - len(l.lstrip())) <= indent:
            end = k
            break
    mode_b = "\n".join(lines[j:end])

    if FROZEN_TARGET_EXPR not in mode_b:
        raise MetricMismatch(
            f"the Mode B branch no longer contains {FROZEN_TARGET_EXPR!r}. "
            f"It now reads:\n{mode_b}"
        )
    if "q95_threshold" in mode_b:
        raise MetricMismatch(
            "the Mode B branch now references q95_threshold; the protocol has "
            "changed and our target no longer matches it."
        )
    return {"checked": True, "expression": FROZEN_TARGET_EXPR,
            "mode_b_branch": mode_b.strip()}


if __name__ == "__main__":
    import json
    print(json.dumps(verify_against_frozen(), indent=2))
    y = np.array([10.0, 50.0, 86.6, 30.0])
    print("mode_b_target  :", mode_b_target(y))
    print("95th percentile:", float(np.quantile(y, 0.95)), " <- NOT the target")
