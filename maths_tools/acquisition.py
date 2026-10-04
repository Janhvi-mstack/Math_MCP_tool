"""Tool 2, part two -- turning predictions into a choice.

Expected Improvement does not ask which candidate is predicted highest. It asks
which has the best chance of beating the best result so far, and by how much,
and it works that out from the uncertainty as well as the prediction. That is
the mechanism that stops the search settling into the first decent region it
finds.

Two optimisation paths, dispatched on the encoding rather than on a flag:

  descriptor encoding -> score every row of the table and take the best.
                         The continuous relaxation of a one-hot column is
                         meaningless, so gradient methods are refused here.
  mixed encoding      -> optimise the acquisition surface directly. Categorical
                         columns are held fixed at each level combination while
                         the numeric columns are climbed from several starts.

Batches are built by sequential greedy: pick one, tell the acquisition function
it is pending, pick the next. Taking the top q would return q near-identical
experiments, because the model's best guesses all sit in the same region.
"""

from __future__ import annotations

import itertools
import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .encode import DESCRIPTOR, MIXED, Encoder
from .space import SearchSpace, SpaceError
from .surrogate import DTYPE, Surrogate, SurrogateError

EI = "ei"
QEI = "qei"
QNEI = "qnei"
TS = "ts"
ACQUISITIONS = (EI, QEI, QNEI, TS)

# Scoring is chunked so a large table never materialises at once.
SCORE_CHUNK = 512

# A joint posterior sample needs the full covariance, so Thompson sampling is
# capped and the subsampling is recorded rather than done quietly.
TS_CANDIDATE_LIMIT = 5_000

# optimize_acqf_mixed enumerates every categorical combination. Above this the
# combinations are sampled, with a note.
MIXED_COMBO_LIMIT = 64

NUM_RESTARTS = 10
RAW_SAMPLES = 256
MC_SAMPLES = 128


class AcquisitionError(RuntimeError):
    pass


@dataclass
class Recommendation:
    """What to run next, and everything needed to defend the choice."""

    points: List[Dict[str, Any]]
    acq_values: List[float]
    acquisition: str
    path: str                      # "enumerated" | "mixed" | "continuous"
    n_requested: int
    seed: int
    best_f: Optional[float] = None
    predicted_mean: List[float] = field(default_factory=list)
    predicted_std: List[float] = field(default_factory=list)
    n_candidates: Optional[int] = None
    n_excluded: int = 0
    notes: List[str] = field(default_factory=list)
    fit_info: Dict[str, Any] = field(default_factory=dict)

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
        return {
            "points": self.points,
            "acq_values": [float(v) for v in self.acq_values],
            "acquisition": self.acquisition,
            "path": self.path,
            "n_requested": self.n_requested,
            "n_returned": len(self.points),
            "complete": self.complete,
            "seed": self.seed,
            "best_f": self.best_f,
            "predicted_mean": [float(v) for v in self.predicted_mean],
            "predicted_std": [float(v) for v in self.predicted_std],
            "n_candidates": self.n_candidates,
            "n_excluded": self.n_excluded,
            "notes": list(self.notes),
            "fit_info": dict(self.fit_info),
        }

    def __repr__(self):
        return (f"Recommendation({len(self.points)}/{self.n_requested}, "
                f"{self.acquisition}, path={self.path})")


# ------------------------------------------------------------------- public

def next_recommendation(
    space: SearchSpace,
    encoder: Encoder,
    surrogate: Surrogate,
    n: int = 1,
    acquisition: str = EI,
    seed: int = 0,
    observed: Optional[Sequence[Dict[str, Any]]] = None,
    exclude_observed: bool = True,
    candidates: Optional[Sequence[Dict[str, Any]]] = None,
    prior=None,
    prior_beta: float = 1.0,
    prior_rgpe: Optional[float] = None,
    observed_y: Optional[Sequence[float]] = None,
) -> Recommendation:
    """Suggest the next `n` experiments from a fitted surrogate.

    Parameters
    ----------
    surrogate : a Surrogate that has already had fit() called on it.
    n : batch size. 1 matches the benchmark protocol.
    acquisition : "ei" (analytic, q=1 only), "qei", "qnei" (use under noise),
        or "ts" (Thompson sampling -- about 4x faster, for large spaces).
    observed : the experiments already run. Only used to exclude them.
    candidates : restrict the choice to this pool. When the knowledge graph
        narrows the search, this is where that narrowing arrives.
    prior : a Prior from maths_tools.priors, or None.
        Applied piBO-style: acquisition values are multiplied by
        density ** (beta / t), so the tilt fades as observations arrive.
    prior_beta : the confidence dial. Larger means the prior holds on longer.
    prior_rgpe : earned-trust factor in [0, 1]. If left as None and
        `observed_y` is given, it is computed from the data by checking whether
        the prior ranks the observed experiments the way reality does. A prior
        that ranks them backwards is switched off automatically.
    observed_y : yields for `observed`, used only to earn the prior's trust.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if acquisition not in ACQUISITIONS:
        raise ValueError(f"acquisition must be one of {ACQUISITIONS}, got {acquisition!r}")
    if surrogate.model is None:
        raise SurrogateError("surrogate has not been fitted")
    if acquisition == EI and n > 1:
        raise ValueError(
            "analytic 'ei' is single-point only; use 'qei' (or 'qnei' under "
            "noise) for a batch"
        )

    torch.manual_seed(seed)
    notes: List[str] = []
    best_f = surrogate.best_f

    if prior is not None:
        if acquisition == TS:
            raise AcquisitionError(
                "a prior cannot be applied to Thompson sampling -- it picks the "
                "argmax of a posterior draw, so there is no acquisition value "
                "to tilt. Use 'ei'/'qei'/'qnei', or drop the prior. Refusing "
                "rather than ignoring it silently."
            )
        if hasattr(prior, "validate_against"):
            prior.validate_against(space)
        if prior_rgpe is None:
            if observed_y is not None and observed:
                from .priors import rgpe_weight
                prior_rgpe = rgpe_weight(prior, list(observed), list(observed_y))
                notes.append(
                    f"prior earned an RGPE weight of {prior_rgpe:.2f} against "
                    f"{len(observed_y)} observation(s)"
                )
            else:
                prior_rgpe = 1.0
                notes.append(
                    "no observed yields supplied, so the prior was trusted at "
                    "full weight without being checked"
                )
        if prior_rgpe == 0.0:
            notes.append(
                "the prior ranks the observed results no better than backwards; "
                "it has been switched off and the choice is made on data alone"
            )

    use_table = encoder.mode == DESCRIPTOR or candidates is not None
    if use_table:
        rec = _enumerated(space, encoder, surrogate, n, acquisition, seed,
                          observed, exclude_observed, candidates, best_f, notes,
                          prior, prior_beta, prior_rgpe)
    else:
        if prior is not None:
            raise AcquisitionError(
                "a prior is only supported on the table path (descriptor "
                "encoding, or an explicit `candidates` pool). On the optimised "
                "path the acquisition surface is climbed by gradient and the "
                "prior density is not differentiable through the decoder. "
                "Supply `candidates` to apply it. Refusing rather than "
                "ignoring it silently."
            )
        rec = _optimised(space, encoder, surrogate, n, acquisition, seed,
                         best_f, notes)

    if rec.points:
        m, s = surrogate.predict(encoder.encode(rec.points, validate=False))
        rec.predicted_mean = [float(v) for v in m]
        rec.predicted_std = [float(v) for v in s]
    rec.fit_info = dict(surrogate.fit_info)
    return rec


# -------------------------------------------------------------- enumerated

def _enumerated(space, encoder, surrogate, n, acquisition, seed, observed,
                exclude_observed, candidates, best_f, notes,
                prior=None, prior_beta=1.0, prior_rgpe=1.0) -> Recommendation:
    if candidates is not None:
        pool = [dict(p) for p in candidates]
        for p in pool:
            space.validate_point(p)
        notes.append(f"choosing from a supplied pool of {len(pool):,} candidate(s)")
    else:
        if not space.is_enumerable:
            raise SpaceError(
                "a descriptor encoding requires a table to score, but this "
                "space is not enumerable. Use a mixed encoder, or supply "
                "`candidates` explicitly."
            )
        pool = space.enumerate()

    n_total = len(pool)
    n_excluded = 0
    if exclude_observed and observed:
        seen = {_key(p) for p in observed}
        kept = [p for p in pool if _key(p) not in seen]
        n_excluded = n_total - len(kept)
        pool = kept

    if not pool:
        notes.append("every candidate has already been run; nothing left to suggest")
        return Recommendation([], [], acquisition, "enumerated", n, seed,
                              best_f=best_f, n_candidates=n_total,
                              n_excluded=n_excluded, notes=notes)

    X = torch.as_tensor(encoder.encode(pool, validate=False), dtype=DTYPE)

    if acquisition == TS:
        idx, vals = _thompson(surrogate, X, n, seed, notes)
    else:
        tilt = None
        if prior is not None:
            from .priors import pibo_exponent, pibo_weights
            n_obs = len(observed) if observed else len(surrogate.train_X)
            tilt = pibo_weights(prior, pool, n_obs, prior_beta, prior_rgpe)
            notes.append(
                f"piBO tilt applied at exponent "
                f"{pibo_exponent(n_obs, prior_beta) * prior_rgpe:.3f} "
                f"(beta={prior_beta:g}, rgpe={prior_rgpe:.2f}, t={n_obs})"
            )
        idx, vals = _greedy_table(surrogate, X, n, acquisition, best_f, seed,
                                  notes, tilt)

    if len(idx) < n:
        notes.append(f"only {len(idx)} of {n} distinct candidate(s) available")

    return Recommendation([pool[i] for i in idx], vals, acquisition, "enumerated",
                          n, seed, best_f=best_f, n_candidates=n_total,
                          n_excluded=n_excluded, notes=notes)


def _greedy_table(surrogate, X, n, acquisition, best_f, seed, notes, tilt=None):
    """Sequential greedy over a fixed table.

    q=1 is the analytic form. For a batch, each pick is added to X_pending so
    the next pick knows it is already going to be run -- otherwise the batch
    collapses onto one region.
    """
    from botorch.acquisition import (ExpectedImprovement, qExpectedImprovement,
                                     qNoisyExpectedImprovement)
    from botorch.sampling import SobolQMCNormalSampler

    chosen: List[int] = []
    values: List[float] = []
    pending: Optional[torch.Tensor] = None

    for step in range(n):
        if acquisition == EI:
            acq = ExpectedImprovement(surrogate.model, best_f=best_f)
        else:
            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([MC_SAMPLES]),
                                            seed=seed + step)
            if acquisition == QNEI:
                acq = qNoisyExpectedImprovement(
                    surrogate.model, X_baseline=surrogate.train_X,
                    sampler=sampler, X_pending=pending, prune_baseline=True)
            else:
                acq = qExpectedImprovement(surrogate.model, best_f=best_f,
                                           sampler=sampler, X_pending=pending)

        v = _score_all(acq, X)
        if tilt is not None:
            # EI is non-negative, so a multiplicative tilt preserves order
            # within equal-density groups and never makes a value negative.
            v = v * torch.as_tensor(tilt, dtype=DTYPE)
            if step == 0 and float(v.max()) <= 0.0:
                notes.append(
                    "every tilted acquisition value is zero; the pick is "
                    "effectively arbitrary this round"
                )
        v[chosen] = -float("inf")
        # stable argmax: ties break on the lower index, so the result does not
        # depend on the sort implementation
        order = torch.argsort(v, descending=True, stable=True)
        pick = int(order[0])
        if not math.isfinite(float(v[pick])):
            notes.append(f"stopped after {step} pick(s): no finite acquisition value left")
            break
        chosen.append(pick)
        values.append(float(v[pick]))
        nxt = X[pick].unsqueeze(0)
        pending = nxt if pending is None else torch.cat([pending, nxt], dim=0)

    return chosen, values


def _score_all(acq, X: torch.Tensor) -> torch.Tensor:
    out = torch.empty(len(X), dtype=DTYPE)
    with torch.no_grad(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(0, len(X), SCORE_CHUNK):
            chunk = X[i:i + SCORE_CHUNK].unsqueeze(1)   # (b, 1, d)
            out[i:i + SCORE_CHUNK] = acq(chunk).to(DTYPE)
    return out


def _thompson(surrogate, X, n, seed, notes):
    """One draw from the posterior; take its best points.

    A joint sample needs the full covariance, so the candidate set is capped.
    """
    idx_map = np.arange(len(X))
    if len(X) > TS_CANDIDATE_LIMIT:
        rng = np.random.default_rng(seed)
        idx_map = np.sort(rng.choice(len(X), TS_CANDIDATE_LIMIT, replace=False))
        notes.append(
            f"Thompson sampling capped at {TS_CANDIDATE_LIMIT:,} candidates; "
            f"sampled from {len(X):,} with seed {seed}"
        )
        X = X[idx_map]

    torch.manual_seed(seed)
    with torch.no_grad(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        post = surrogate.model.posterior(X)
        # one independent draw per batch member, so a batch is q genuinely
        # different beliefs about the world rather than q points from one
        draws = post.rsample(torch.Size([n])).squeeze(-1)     # (n, N)

    chosen, values = [], []
    for k in range(n):
        d = draws[k].clone()
        if chosen:
            d[torch.tensor(chosen, dtype=torch.long)] = -float("inf")
        order = torch.argsort(d, descending=True, stable=True)
        pick = int(order[0])
        chosen.append(pick)
        values.append(float(d[pick]))
    return [int(idx_map[i]) for i in chosen], values


# ----------------------------------------------------------------- optimised

def _optimised(space, encoder, surrogate, n, acquisition, seed, best_f, notes):
    """Climb the acquisition surface -- no table exists."""
    from botorch.acquisition import (ExpectedImprovement, qExpectedImprovement,
                                     qNoisyExpectedImprovement)
    from botorch.optim import optimize_acqf, optimize_acqf_mixed
    from botorch.sampling import SobolQMCNormalSampler

    if acquisition == TS:
        return _thompson_continuous(space, encoder, surrogate, n, seed, best_f, notes)

    if acquisition == EI:
        acq = ExpectedImprovement(surrogate.model, best_f=best_f)
    else:
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([MC_SAMPLES]), seed=seed)
        acq = (qNoisyExpectedImprovement(surrogate.model,
                                         X_baseline=surrogate.train_X,
                                         sampler=sampler, prune_baseline=True)
               if acquisition == QNEI
               else qExpectedImprovement(surrogate.model, best_f=best_f, sampler=sampler))

    bounds = torch.as_tensor(encoder.bounds(), dtype=DTYPE)
    ineq = _inequality_constraints(space, encoder, notes)
    cat_choices = encoder.categorical_choices()

    kw = dict(q=n, num_restarts=NUM_RESTARTS, raw_samples=RAW_SAMPLES,
              options={"seed": seed})
    if ineq:
        kw["inequality_constraints"] = ineq

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if cat_choices:
            fixed = _fixed_feature_list(cat_choices, seed, notes)
            cand, val = optimize_acqf_mixed(acq, bounds=bounds,
                                            fixed_features_list=fixed, **kw)
            path = "mixed"
        else:
            cand, val = optimize_acqf(acq, bounds=bounds, **kw)
            path = "continuous"

    pts = encoder.decode(cand.detach().cpu().numpy())
    kept, dropped = [], 0
    for p in pts:
        if space.satisfies(p):
            kept.append(p)
        else:
            dropped += 1
    if dropped:
        notes.append(
            f"{dropped} optimised point(s) violated a constraint after "
            f"decoding to the nearest legal level and were dropped"
        )

    v = float(val) if val.numel() == 1 else float(val.sum())
    return Recommendation(kept, [v] * len(kept), acquisition, path, n, seed,
                          best_f=best_f, notes=notes)


def _thompson_continuous(space, encoder, surrogate, n, seed, best_f, notes):
    """Thompson over a Sobol candidate set, since there is no table."""
    from .design import initial_design

    n_cand = min(TS_CANDIDATE_LIMIT, max(1024, 256 * n))
    pool = initial_design(space, n_cand, method="sobol", seed=seed).points
    notes.append(f"Thompson sampling over a {len(pool):,}-point Sobol candidate set")
    X = torch.as_tensor(encoder.encode(pool, validate=False), dtype=DTYPE)
    idx, vals = _thompson(surrogate, X, n, seed, notes)
    return Recommendation([pool[i] for i in idx], vals, TS, "continuous", n, seed,
                          best_f=best_f, n_candidates=len(pool), notes=notes)


def _fixed_feature_list(cat_choices: Dict[int, List[int]], seed: int, notes):
    """Every combination of categorical levels, or a sample if there are many."""
    cols = sorted(cat_choices)
    combos = list(itertools.product(*[cat_choices[c] for c in cols]))
    if len(combos) > MIXED_COMBO_LIMIT:
        rng = np.random.default_rng(seed)
        pick = rng.choice(len(combos), MIXED_COMBO_LIMIT, replace=False)
        notes.append(
            f"{len(combos):,} categorical combinations exceed the limit of "
            f"{MIXED_COMBO_LIMIT}; optimising over {MIXED_COMBO_LIMIT} sampled "
            f"combination(s) with seed {seed}. Some level combinations were "
            f"not considered this round."
        )
        combos = [combos[int(i)] for i in sorted(pick)]
    return [{c: float(v) for c, v in zip(cols, combo)} for combo in combos]


def _inequality_constraints(space, encoder, notes):
    """Translate schema constraints into botorch's encoded-space form.

    A constraint is written in the chemist's units (mol%, degrees). The encoder
    maps each numeric factor by x_enc = (x - offset) / scale, so
        sum_i w_i * x_i <= rhs
    becomes
        sum_i (w_i * scale_i) * x_enc_i <= rhs - sum_i w_i * offset_i
    and botorch wants it as >= , so both sides are negated.
    """
    out = []
    blocks = {b.factor.name: b for b in encoder._blocks}
    for c in space.constraints:
        cols, coefs, shift = [], [], 0.0
        ok = True
        for name, w in c.weights.items():
            b = blocks.get(name)
            if b is None or b.kind != "numeric":
                notes.append(
                    f"constraint on {name!r} could not be expressed in the "
                    f"encoded space and is enforced by filtering instead"
                )
                ok = False
                break
            if b.factor.transform != "linear":
                notes.append(
                    f"constraint on {name!r} is not linear in the encoded space "
                    f"(log transform); enforced by filtering instead"
                )
                ok = False
                break
            cols.append(b.start)
            coefs.append(-float(w) * float(b.scale[0]))
            shift += float(w) * float(b.offset[0])
        if ok:
            out.append((torch.tensor(cols, dtype=torch.long),
                        torch.tensor(coefs, dtype=DTYPE),
                        float(-(c.rhs - shift))))
    return out


def _key(p: Dict[str, Any]):
    return tuple(sorted((k, _h(v)) for k, v in p.items()))


def _h(v):
    if isinstance(v, float):
        return round(v, 12)
    if isinstance(v, (list, tuple)):
        return tuple(v)
    return v
