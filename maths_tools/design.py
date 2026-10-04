"""Tool 1 -- choosing the first experiments, before any data exists.

There is no model yet, so nothing here is clever. The whole job is to spread
the first few experiments out so that whatever model gets fitted afterwards has
something informative to learn from.

Methods
-------
random      uniform over the space. The control everything else is measured
            against -- not a straw man, it performs well on average.
maxmin      each pick is whichever candidate is furthest from everything
            already picked (Kennard-Stone). Needs a finite pool.
kmedoids    cluster the pool, take one real member per cluster. Needs a pool.
sobol       a low-discrepancy sequence: each point lands in the largest gap the
            earlier points left. Works with or without a pool, so this is the
            only option once a factor is continuous.
auto        maxmin if the space is enumerable, sobol otherwise.

Every method is deterministic given `seed`. Anything that bounds the work --
subsampling a huge pool, giving up on a constraint -- is recorded in
`Design.notes` rather than happening quietly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .encode import DESCRIPTOR, Encoder
from .space import (
    CATEGORICAL,
    CONTINUOUS,
    INTEGER,
    LOG,
    ORDINAL,
    SearchSpace,
    SpaceError,
)

RANDOM = "random"
MAXMIN = "maxmin"
KMEDOIDS = "kmedoids"
SOBOL = "sobol"
AUTO = "auto"

METHODS = (RANDOM, MAXMIN, KMEDOIDS, SOBOL)

# Pairwise-distance methods are O(pool^2) in memory. Above this the pool is
# subsampled (seeded) and the fact is recorded in Design.notes.
PAIRWISE_POOL_LIMIT = 20_000

# How many times to resample a constrained point before giving up.
CONSTRAINT_ATTEMPTS = 10_000


@dataclass
class Design:
    """The chosen first experiments, plus how they were chosen."""

    points: List[Dict[str, Any]]
    method: str
    seed: int
    n_requested: int
    notes: List[str] = field(default_factory=list)

    def __len__(self):
        return len(self.points)

    def __iter__(self):
        return iter(self.points)

    def __getitem__(self, i):
        return self.points[i]

    @property
    def complete(self) -> bool:
        return len(self.points) == self.n_requested

    def to_dict(self) -> dict:
        return {"points": self.points, "method": self.method, "seed": self.seed,
                "n_requested": self.n_requested, "n_returned": len(self.points),
                "complete": self.complete, "notes": list(self.notes)}

    def __repr__(self):
        s = f"Design({len(self.points)}/{self.n_requested}, {self.method}, seed={self.seed})"
        if self.notes:
            s += f" +{len(self.notes)} note(s)"
        return s


def initial_design(space: SearchSpace, n: int, method: str = AUTO,
                   seed: int = 0, encoder: Optional[Encoder] = None) -> Design:
    """Pick `n` experiments to start with.

    Parameters
    ----------
    space : SearchSpace
    n : int
        How many experiments. Must be >= 1.
    method : one of METHODS, or "auto"
    seed : int
        Fixes the result. Sobol is a fixed sequence so its seed only chooses
        the scramble; the others use it for their RNG.
    encoder : Encoder, optional
        Defines the geometry distances are measured in. Defaults to descriptor
        mode, which is what makes chemical descriptors count toward "far
        apart". Pass an encoder pinned to earlier constants if a level was
        added and you need the old geometry back.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")

    if method == AUTO:
        # k-medoids, not MaxMin: measured better than random on 10 of 10
        # plate/surrogate combinations, where MaxMin lost on 4 of 5 plates.
        # (This dispatch said MAXMIN until the benchmark was run.)
        method = KMEDOIDS if space.is_enumerable else SOBOL
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS} or {AUTO!r}, got {method!r}")

    if method in (MAXMIN, KMEDOIDS) and not space.is_enumerable:
        raise SpaceError(
            f"{method!r} needs a finite pool to choose from, but this space is "
            f"not enumerable (continuous factor, or above the enumeration "
            f"limit). Use {SOBOL!r}, which does not need a pool."
        )

    notes: List[str] = []
    if method == SOBOL:
        points = _sobol(space, n, seed, notes)
    elif method == RANDOM:
        points = _random(space, n, seed, notes)
    else:
        enc = encoder or Encoder(space, mode=DESCRIPTOR)
        pool = space.enumerate()
        pool, notes_ = _cap_pool(pool, seed)
        notes.extend(notes_)
        X = enc.encode(pool, validate=False)
        idx = (_maxmin(X, n, notes) if method == MAXMIN
               else _kmedoids(X, n, seed, notes))
        points = [pool[i] for i in idx]

    return Design(points=points, method=method, seed=seed,
                  n_requested=n, notes=notes)


# ---------------------------------------------------------------- pool utils

def _cap_pool(pool, seed):
    notes = []
    if len(pool) > PAIRWISE_POOL_LIMIT:
        rng = np.random.default_rng(seed)
        keep = np.sort(rng.choice(len(pool), PAIRWISE_POOL_LIMIT, replace=False))
        notes.append(
            f"pool of {len(pool):,} exceeds the pairwise limit of "
            f"{PAIRWISE_POOL_LIMIT:,}; subsampled to {PAIRWISE_POOL_LIMIT:,} "
            f"with seed {seed} before selection"
        )
        pool = [pool[i] for i in keep]
    return pool, notes


# ------------------------------------------------------------------- methods

def _maxmin(X: np.ndarray, n: int, notes: List[str]) -> List[int]:
    """Kennard-Stone.

    Starts from the two points furthest apart, then repeatedly adds whichever
    remaining point has the largest distance to its nearest already-chosen
    point. Fully deterministic -- ties break on the lower index, and the pool
    order is itself deterministic.
    """
    m = len(X)
    if n >= m:
        notes.append(f"requested {n} but the pool holds {m}; returning all of it")
        return list(range(m))

    # squared distances are enough for argmax and avoid the sqrt
    d2 = _sq_dists(X)
    first_pair = np.unravel_index(np.argmax(d2), d2.shape)
    chosen = [int(first_pair[0])]
    if n > 1:
        chosen.append(int(first_pair[1]))

    nearest = d2[chosen[0]].copy()
    for c in chosen[1:]:
        np.minimum(nearest, d2[c], out=nearest)

    while len(chosen) < n:
        nearest[chosen] = -1.0
        nxt = int(np.argmax(nearest))
        chosen.append(nxt)
        np.minimum(nearest, d2[nxt], out=nearest)

    return chosen[:n]


def _kmedoids(X: np.ndarray, n: int, seed: int, notes: List[str]) -> List[int]:
    """k-medoids by alternating assignment and re-selection.

    k-means would take the mean of each cluster, and the mean of four ligands
    is not a ligand you can buy. This keeps a real member.
    """
    m = len(X)
    if n >= m:
        notes.append(f"requested {n} but the pool holds {m}; returning all of it")
        return list(range(m))

    d2 = _sq_dists(X)
    rng = np.random.default_rng(seed)

    # k-means++ style seeding on the distance matrix: spread the starting
    # medoids out, so the alternation starts somewhere sensible.
    medoids = [int(rng.integers(m))]
    while len(medoids) < n:
        nearest = d2[medoids].min(axis=0)
        nearest[medoids] = 0.0
        total = nearest.sum()
        if total <= 0:
            remaining = [i for i in range(m) if i not in medoids]
            medoids.append(int(rng.choice(remaining)))
        else:
            medoids.append(int(rng.choice(m, p=nearest / total)))

    medoids = sorted(medoids)
    for it in range(100):
        labels = np.argmin(d2[medoids], axis=0)
        new = []
        for k in range(n):
            members = np.flatnonzero(labels == k)
            if len(members) == 0:
                new.append(medoids[k])
                continue
            sub = d2[np.ix_(members, members)]
            new.append(int(members[int(np.argmin(sub.sum(axis=1)))]))
        new = sorted(set(new))
        while len(new) < n:  # a cluster collapsed; refill deterministically
            cand = int(np.argmax(d2[new].min(axis=0)))
            if cand in new:
                break
            new.append(cand)
            new = sorted(new)
        if new == medoids:
            notes.append(f"k-medoids converged after {it + 1} iteration(s)")
            break
        medoids = new
    else:
        notes.append("k-medoids hit the 100-iteration cap without converging")

    return list(medoids)


def _sq_dists(X: np.ndarray) -> np.ndarray:
    g = X @ X.T
    sq = np.diag(g)
    d2 = sq[:, None] + sq[None, :] - 2 * g
    np.maximum(d2, 0, out=d2)
    return d2


def _sobol(space: SearchSpace, n: int, seed: int, notes: List[str]) -> List[Dict]:
    """Low-discrepancy sampling, one dimension per factor.

    Dimensions are per FACTOR, not per encoded column. That is what gives the
    projection property: every point gets its own value on every axis, so if
    one factor turns out not to matter the other values are still distinct.
    """
    from torch.quasirandom import SobolEngine

    d = len(space.factors)
    need = n
    out: List[Dict] = []
    eng = SobolEngine(dimension=d, scramble=True, seed=seed)

    drawn = 0
    rejected = 0
    while len(out) < need:
        batch = max(need - len(out), 16)
        U = eng.draw(batch).numpy().astype(float)
        drawn += batch
        for u in U:
            p = {f.name: _from_unit(f, u[j]) for j, f in enumerate(space.factors)}
            if not space.satisfies(p):
                rejected += 1
                continue
            out.append(p)
            if len(out) == need:
                break
        if drawn > CONSTRAINT_ATTEMPTS:
            notes.append(
                f"stopped after drawing {drawn:,} Sobol points; "
                f"{len(out)} of {need} satisfied the constraints "
                f"({rejected:,} rejected). The feasible region may be very small."
            )
            break

    if rejected:
        notes.append(f"{rejected:,} Sobol point(s) rejected by constraints")
    return out


def _random(space: SearchSpace, n: int, seed: int, notes: List[str]) -> List[Dict]:
    rng = np.random.default_rng(seed)

    if space.is_enumerable:
        pool = space.enumerate()
        if n >= len(pool):
            notes.append(
                f"requested {n} but only {len(pool)} feasible combination(s) "
                f"exist; returning all of them"
            )
            return pool
        idx = rng.choice(len(pool), n, replace=False)
        return [pool[int(i)] for i in sorted(idx)]

    out, attempts, rejected = [], 0, 0
    while len(out) < n and attempts < CONSTRAINT_ATTEMPTS:
        attempts += 1
        p = {f.name: _from_unit(f, rng.random()) for f in space.factors}
        if not space.satisfies(p):
            rejected += 1
            continue
        out.append(p)
    if rejected:
        notes.append(f"{rejected:,} random point(s) rejected by constraints")
    if len(out) < n:
        notes.append(
            f"only {len(out)} of {n} point(s) found in {attempts:,} attempts; "
            f"the feasible region may be very small"
        )
    return out


def _from_unit(f, u: float):
    """Map a uniform [0,1) draw onto one factor's allowed values."""
    u = float(min(max(u, 0.0), 1.0 - 1e-12))
    if f.type == CONTINUOUS:
        low, high = f.bounds
        if f.transform == LOG:
            return float(np.exp(np.log(low) + u * (np.log(high) - np.log(low))))
        return float(low + u * (high - low))
    if f.type == INTEGER:
        low, high = int(f.bounds[0]), int(f.bounds[1])
        return int(min(low + int(u * (high - low + 1)), high))
    # ordinal / categorical
    k = len(f.levels)
    return f.levels[int(min(int(u * k), k - 1))]
