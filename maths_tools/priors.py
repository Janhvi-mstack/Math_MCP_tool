"""Tool 1 with priors -- starting from what we already know.

A prior is a belief about where the good region is, expressed as a probability
rather than as an instruction. It never rules anything out; it tilts the
choice. Two things make that safe:

  decay   the tilt weakens with every real result. piBO multiplies the
          acquisition value by density^(beta/t), where t counts observations.
          Early on the prior steers; after enough data the exponent is small
          enough that the evidence decides.

  RGPE    before trusting a source at all, check it against reality. Not on
          whether it predicted the right number -- on whether it got the
          ORDERING right. Take pairs of experiments that have actually been
          run and ask whether the source would have ranked them the same way.
          A source that ranks things backwards is weighted to zero within a
          handful of experiments.

The second one is not decoration. Our own knowledge graph ranks the genuinely
best ligand third of four on all five reactions, correlation -0.60 with truth.
Fed in with the confidence turned up it would make the search worse than
random. `rank_agreement` is the function that detects that, and
`rgpe_weight` is what acts on it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .design import Design, _from_unit
from .space import CATEGORICAL, CONTINUOUS, INTEGER, LOG, ORDINAL, SearchSpace, SpaceError

EPS = 1e-12


class PriorError(ValueError):
    pass


# ------------------------------------------------------------------- priors

class Prior:
    """A non-negative, unnormalised belief over the space."""

    name = "prior"

    def density(self, points: Sequence[Dict[str, Any]]) -> np.ndarray:
        raise NotImplementedError

    def _check(self, d: np.ndarray) -> np.ndarray:
        d = np.asarray(d, dtype=float).reshape(-1)
        if not np.isfinite(d).all():
            raise PriorError(f"{self.name}: density produced a non-finite value")
        if (d < 0).any():
            raise PriorError(f"{self.name}: density produced a negative value")
        return d

    def __repr__(self):
        return f"{type(self).__name__}()"


class UniformPrior(Prior):
    """No belief. The control -- with this, everything reduces to no prior."""

    name = "uniform"

    def density(self, points):
        return np.ones(len(points), dtype=float)


@dataclass
class CategoricalPrior(Prior):
    """Belief over which levels are good, e.g. straight from a knowledge graph.

    weights : {factor name: {level: non-negative number}}
        Levels not mentioned get `default`. The density of a point is the
        product of its factors' weights, so a belief about ligands and a belief
        about bases combine without either being rescaled.
    """

    weights: Dict[str, Dict[Any, float]]
    default: float = 1.0
    name: str = "categorical"

    def __post_init__(self):
        for fname, table in self.weights.items():
            for level, w in table.items():
                if w < 0 or not math.isfinite(w):
                    raise PriorError(
                        f"weight for {fname}={level!r} must be finite and "
                        f"non-negative, got {w}"
                    )
        if self.default < 0:
            raise PriorError("default weight must be non-negative")

    def validate_against(self, space: SearchSpace) -> None:
        """Catch a prior written against the wrong space before it is used."""
        for fname, table in self.weights.items():
            try:
                f = space[fname]
            except KeyError:
                raise PriorError(
                    f"prior mentions factor {fname!r}, which is not in the space"
                ) from None
            if f.is_numeric:
                raise PriorError(
                    f"CategoricalPrior cannot describe numeric factor {fname!r}; "
                    f"use GaussianPrior"
                )
            unknown = [l for l in table if l not in f.levels]
            if unknown:
                raise PriorError(
                    f"prior gives weights for {fname} level(s) {unknown!r}, "
                    f"which the space does not declare"
                )

    def density(self, points):
        out = np.ones(len(points), dtype=float)
        for i, p in enumerate(points):
            v = 1.0
            for fname, table in self.weights.items():
                v *= float(table.get(p[fname], self.default))
            out[i] = v
        return self._check(out)


@dataclass
class GaussianPrior(Prior):
    """"The optimum is probably around here" for numeric factors.

    center : {factor: value}
    width  : {factor: standard deviation in the factor's own units}
    """

    center: Dict[str, float]
    width: Dict[str, float]
    name: str = "gaussian"

    def __post_init__(self):
        for k, w in self.width.items():
            if w <= 0 or not math.isfinite(w):
                raise PriorError(f"width for {k!r} must be positive, got {w}")
        missing = set(self.center) - set(self.width)
        if missing:
            raise PriorError(f"no width given for {sorted(missing)}")

    def validate_against(self, space: SearchSpace) -> None:
        for fname in self.center:
            try:
                f = space[fname]
            except KeyError:
                raise PriorError(
                    f"prior mentions factor {fname!r}, which is not in the space"
                ) from None
            if not f.is_numeric:
                raise PriorError(
                    f"GaussianPrior cannot describe {f.type} factor {fname!r}; "
                    f"use CategoricalPrior"
                )

    def density(self, points):
        out = np.ones(len(points), dtype=float)
        for i, p in enumerate(points):
            v = 1.0
            for fname, mu in self.center.items():
                s = self.width[fname]
                z = (float(p[fname]) - float(mu)) / s
                v *= math.exp(-0.5 * z * z)
            out[i] = v
        return self._check(out)


@dataclass
class CompositePrior(Prior):
    """Several beliefs at once. Densities multiply."""

    parts: List[Prior]
    name: str = "composite"

    def validate_against(self, space):
        for p in self.parts:
            if hasattr(p, "validate_against"):
                p.validate_against(space)

    def density(self, points):
        out = np.ones(len(points), dtype=float)
        for p in self.parts:
            out = out * p.density(points)
        return self._check(out)


# ------------------------------------------------------- checking the prior

def rank_agreement(prior: Prior, points: Sequence[Dict[str, Any]],
                   y: Sequence[float]) -> float:
    """Kendall tau between what the prior believes and what actually happened.

    +1  the prior orders the observed experiments exactly as reality does
     0  no better than a coin flip
    -1  the prior orders them exactly backwards

    Ordering rather than absolute values, because yields sit on different
    scales across reactions but the ordering is the part that transfers.
    Ties on either side are skipped, which is the tau-b convention.
    """
    y = np.asarray(y, dtype=float).reshape(-1)
    if len(points) != len(y):
        raise PriorError(f"{len(points)} points but {len(y)} observations")
    if len(y) < 2:
        return 0.0
    d = prior.density(points)

    concordant = discordant = 0
    for i in range(len(y)):
        for j in range(i + 1, len(y)):
            dy = y[i] - y[j]
            dd = d[i] - d[j]
            if dy == 0 or dd == 0:
                continue
            if (dy > 0) == (dd > 0):
                concordant += 1
            else:
                discordant += 1
    total = concordant + discordant
    if total == 0:
        return 0.0
    return (concordant - discordant) / total


def rgpe_weight(prior: Prior, points: Sequence[Dict[str, Any]],
                y: Sequence[float], min_observations: int = 4) -> float:
    """How much this prior has earned the right to influence the search.

    Returns a number in [0, 1] that multiplies the confidence dial. A source
    that ranks things correctly keeps its influence; one that ranks them
    backwards falls to zero.

    Below `min_observations` there is not enough evidence to judge, so the
    prior keeps full weight -- that is exactly the regime it exists for.
    """
    if len(y) < min_observations:
        return 1.0
    return max(0.0, rank_agreement(prior, points, y))


# ---------------------------------------------------------- prior-led design

def prior_initial_design(
    space: SearchSpace,
    n: int,
    prior: Prior,
    confidence: float = 1.0,
    seed: int = 0,
    pool_size: int = 4096,
) -> Design:
    """The first experiments, drawn from the belief instead of uniformly.

    `confidence` is the dial. 0.0 reduces exactly to uniform sampling -- which
    makes the no-prior control a special case of this function rather than a
    different code path, so the comparison is honest.
    """
    from .design import RANDOM, initial_design

    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if confidence < 0 or not math.isfinite(confidence):
        raise PriorError(f"confidence must be finite and >= 0, got {confidence}")
    if hasattr(prior, "validate_against"):
        prior.validate_against(space)

    notes: List[str] = []
    if confidence == 0.0:
        d = initial_design(space, n, method=RANDOM, seed=seed)
        d.notes.append("confidence=0, so the prior was not applied at all")
        d.method = "prior(uniform)"
        return d

    rng = np.random.default_rng(seed)
    if space.is_enumerable:
        pool = space.enumerate()
    else:
        pool = initial_design(space, pool_size, method="sobol", seed=seed).points
        notes.append(
            f"space is not enumerable; sampled from a {len(pool):,}-point "
            f"Sobol pool"
        )

    if n >= len(pool):
        notes.append(f"requested {n} but the pool holds {len(pool)}; returning all")
        return Design(pool, "prior", seed, n, notes)

    w = prior.density(pool) ** confidence
    total = w.sum()
    if total <= 0 or not math.isfinite(total):
        notes.append(
            "the prior gives zero weight to every candidate; falling back to "
            "uniform sampling rather than failing"
        )
        w = np.ones(len(pool))
        total = w.sum()

    idx = rng.choice(len(pool), size=n, replace=False, p=w / total)
    notes.append(
        f"weighted sampling from {len(pool):,} candidates at confidence "
        f"{confidence:g}"
    )
    return Design([pool[int(i)] for i in sorted(idx)], "prior", seed, n, notes)


# ------------------------------------------------------------- piBO exponent

def pibo_exponent(n_observed: int, beta: float = 1.0) -> float:
    """The exponent the prior density is raised to: beta / max(t, 1).

    At t=0 the prior is at full strength. By t=beta it is at power 1, and it
    keeps shrinking. There is no cliff and nothing to tune per problem.
    """
    if beta < 0 or not math.isfinite(beta):
        raise PriorError(f"beta must be finite and >= 0, got {beta}")
    return float(beta) / float(max(n_observed, 1))


def pibo_weights(prior: Prior, points: Sequence[Dict[str, Any]],
                 n_observed: int, beta: float = 1.0,
                 rgpe: float = 1.0) -> np.ndarray:
    """Multiplier to apply to acquisition values.

    `rgpe` is the earned-trust factor from `rgpe_weight`. At 0 this returns all
    ones, i.e. the prior has been switched off by the evidence and the search
    proceeds on the data alone.
    """
    if not (0.0 <= rgpe <= 1.0):
        raise PriorError(f"rgpe weight must be in [0, 1], got {rgpe}")
    exp = pibo_exponent(n_observed, beta) * rgpe
    if exp == 0.0:
        return np.ones(len(points), dtype=float)
    d = prior.density(points)
    d = np.maximum(d, EPS)
    return d ** exp
