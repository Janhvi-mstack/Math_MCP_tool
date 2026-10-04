"""The run state -- what has been tried, what came back, what never ran.

This is where the unvalidated-recommendation rule lives, and it is enforced by
construction rather than by remembering to do the right thing.

A suggestion that never gets run is dropped. Not recorded as zero, not filled
with a predicted value, not included with inflated uncertainty. The model ends
up in exactly the state it would have been in if the experiment had never been
proposed.

The guarantee is structural: the training matrices are rebuilt from
`self._observations` alone, and `skip()` cannot reach that list. Every call
into the surrogate goes through `_train_arrays`, and `check_invariants` re-
derives the whole thing and raises if anything has leaked. Nothing relies on a
caller passing the right arguments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .acquisition import EI, Recommendation, next_recommendation, _key
from .design import AUTO as DESIGN_AUTO
from .design import Design, initial_design
from .encode import AUTO as ENC_AUTO
from .encode import Encoder
from .priors import Prior, prior_initial_design, rgpe_weight
from .space import SearchSpace
from .surrogate import AUTO as SUR_AUTO
from .surrogate import Surrogate, build_surrogate


class CampaignError(RuntimeError):
    pass


@dataclass
class Observation:
    point: Dict[str, Any]
    y: float
    iteration: int


@dataclass
class Skipped:
    """A suggestion that was never run. Carries no yield, by design."""
    point: Dict[str, Any]
    reason: str
    iteration: int


class Campaign:
    """One optimisation run.

    Typical use::

        c = Campaign(space, n_init=5, seed=0)
        for p in c.suggest(5):          # first call -> initial design
            c.record(p, run_in_lab(p))
        while c.n_observations < 50:
            rec = c.suggest(1)          # thereafter -> the model decides
            p = rec[0]
            if chemist_declines(p):
                c.skip(p, "no material")
            else:
                c.record(p, run_in_lab(p))
    """

    def __init__(self, space: SearchSpace, encoder: Optional[Encoder] = None,
                 surrogate_kind: str = SUR_AUTO, acquisition: str = EI,
                 design_method: str = DESIGN_AUTO, n_init: int = 5,
                 seed: int = 0, prior: Optional[Prior] = None,
                 prior_beta: float = 1.0, prior_confidence: float = 0.0,
                 exclude_observed: bool = True,
                 exclude_skipped: bool = False,
                 surrogate_kwargs: Optional[dict] = None):
        self.space = space
        self.encoder = encoder or Encoder(space, mode=ENC_AUTO)
        self.surrogate_kind = surrogate_kind
        self.acquisition = acquisition
        self.design_method = design_method
        self.n_init = n_init
        self.seed = seed
        self.prior = prior
        self.prior_beta = prior_beta
        self.prior_confidence = prior_confidence
        self.exclude_observed = exclude_observed
        self.exclude_skipped = exclude_skipped
        self.surrogate_kwargs = dict(surrogate_kwargs or {})

        self._observations: List[Observation] = []
        self._skipped: List[Skipped] = []
        self.deferral_history: List[Skipped] = []
        self._iteration = 0
        self._pending: List[Dict[str, Any]] = []
        self.history: List[Dict[str, Any]] = []
        self.surrogate: Optional[Surrogate] = None

        if prior is not None and hasattr(prior, "validate_against"):
            prior.validate_against(space)

    # ------------------------------------------------------------- state

    @property
    def n_observations(self) -> int:
        return len(self._observations)

    @property
    def n_skipped(self) -> int:
        return len(self._skipped)

    @property
    def observations(self) -> List[Observation]:
        return list(self._observations)

    @property
    def skipped(self) -> List[Skipped]:
        return list(self._skipped)

    @property
    def observed_points(self) -> List[Dict[str, Any]]:
        return [o.point for o in self._observations]

    @property
    def observed_y(self) -> List[float]:
        return [o.y for o in self._observations]

    @property
    def best_y(self) -> Optional[float]:
        return max((o.y for o in self._observations), default=None)

    @property
    def best_point(self) -> Optional[Dict[str, Any]]:
        if not self._observations:
            return None
        return max(self._observations, key=lambda o: o.y).point

    def trace(self) -> List[float]:
        """Best-so-far after each observation."""
        out, best = [], -float("inf")
        for o in self._observations:
            best = max(best, o.y)
            out.append(best)
        return out

    # ------------------------------------------------------- the guard

    def _train_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        """The ONLY way training data is produced.

        Built from `self._observations` and nothing else. `_skipped` is not in
        scope of this function by construction.
        """
        pts = [o.point for o in self._observations]
        y = np.array([o.y for o in self._observations], dtype=float)
        X = self.encoder.encode(pts, validate=False)
        return X, y

    def check_invariants(self, just_fitted: bool = False) -> None:
        """Re-derive everything and raise if a skipped suggestion has leaked.

        Called after every record() and around every fit. Cheap, and it turns a
        silent contamination into a loud failure.

        `just_fitted` adds the row-count check. That one is only true at the
        moment of fitting: between a fit and the next one the model is
        legitimately stale, because record() adds observations the model has
        not seen yet. Asserting it unconditionally would fail on the first
        normal record() after a suggest().
        """
        obs_keys = [_key(o.point) for o in self._observations]
        if len(set(obs_keys)) != len(obs_keys):
            dupes = {k for k in obs_keys if obs_keys.count(k) > 1}
            raise CampaignError(
                f"the same experiment was recorded more than once: {len(dupes)} "
                f"duplicate key(s)"
            )
        skipped_keys = {_key(s.point) for s in self._skipped}
        leaked = skipped_keys & set(obs_keys)
        if leaked:
            raise CampaignError(
                f"{len(leaked)} experiment(s) are listed as skipped AND as "
                f"observed at the same time. An experiment that was deferred "
                f"and has since been run must move out of the skipped list; "
                f"one that is still skipped must not be in the training data."
            )
        ys = np.array([o.y for o in self._observations], dtype=float)
        if ys.size and not np.isfinite(ys).all():
            raise CampaignError(
                "a recorded observation has a non-finite yield. An experiment "
                "that was not run must be passed to skip(), not recorded with "
                "a placeholder."
            )
        if just_fitted:
            if self.surrogate is None or self.surrogate.train_X is None:
                raise CampaignError("expected a fitted surrogate, found none")
            if len(self.surrogate.train_X) != len(self._observations):
                raise CampaignError(
                    f"the model was just fitted on "
                    f"{len(self.surrogate.train_X)} rows but "
                    f"{len(self._observations)} experiments have been "
                    f"observed. Training data and observations have diverged."
                )

    # ---------------------------------------------------------- recording

    def record(self, point: Dict[str, Any], y: float) -> None:
        """An experiment that actually ran, with the yield it gave."""
        self.space.validate_point(point)
        yf = float(y)
        if not np.isfinite(yf):
            raise CampaignError(
                f"yield for {point} is not finite. If the experiment was not "
                f"run, call skip() instead -- a placeholder value teaches the "
                f"model something we do not know."
            )
        if _key(point) in {_key(o.point) for o in self._observations}:
            raise CampaignError(f"this experiment has already been recorded: {point}")
        # An experiment that was skipped earlier and has now actually been run
        # stops being "not run". It moves out of the skipped list and into the
        # observations; the deferral is kept in `deferral_history` for audit.
        k = _key(point)
        still = [sk for sk in self._skipped if _key(sk.point) != k]
        if len(still) != len(self._skipped):
            self.deferral_history.extend(
                [sk for sk in self._skipped if _key(sk.point) == k])
            self._skipped = still
        self._observations.append(Observation(dict(point), yf, self._iteration))
        self._drop_pending(point)
        self.check_invariants()

    def skip(self, point: Dict[str, Any], reason: str = "not run") -> None:
        """A suggestion that never ran.

        Recorded for the audit trail only. It does not enter the training data
        now or later, and the model is never told it was suggested.

        It REMAINS IN THE SEARCH SPACE and may be recommended again in a later
        round. Set exclude_skipped=True on the campaign to hold skipped
        experiments back instead.
        """
        self.space.validate_point(point)
        if _key(point) in {_key(o.point) for o in self._observations}:
            raise CampaignError(
                f"cannot skip {point}: it has already been recorded as run"
            )
        self._skipped.append(Skipped(dict(point), str(reason), self._iteration))
        self._drop_pending(point)

    def _drop_pending(self, point):
        k = _key(point)
        self._pending = [p for p in self._pending if _key(p) != k]

    # --------------------------------------------------------- suggesting

    def suggest(self, n: int = 1, **kw) -> Design | Recommendation:
        """The initial design while there is no data, the model after that."""
        self._iteration += 1

        if self.n_observations < 2:
            d = self._initial(n)
            self._pending.extend(d.points)
            self.history.append({"iteration": self._iteration, "kind": "design",
                                 "n": len(d.points), "method": d.method,
                                 "notes": list(d.notes)})
            return d

        self.check_invariants()
        X, y = self._train_arrays()
        self.surrogate = build_surrogate(self.encoder, kind=self.surrogate_kind,
                                         **self.surrogate_kwargs)
        self.surrogate.fit(X, y)
        self.check_invariants(just_fitted=True)

        # Only experiments that were actually RUN are off the table, plus
        # anything currently outstanding. A skipped experiment stays in the
        # search space and may be recommended again in a later round -- it is
        # kept out of the training data, not out of the space.
        seen = self.observed_points + list(self._pending)
        if self.exclude_skipped:
            seen = seen + [s.point for s in self._skipped]

        call = dict(n=n, acquisition=self.acquisition,
                    seed=self.seed + self._iteration, observed=seen,
                    exclude_observed=self.exclude_observed,
                    prior=self.prior, prior_beta=self.prior_beta,
                    observed_y=(self.observed_y if self.prior is not None
                                else None))
        # a caller override wins over the campaign default, rather than
        # colliding with it
        call.update({k: v for k, v in kw.items() if v is not None})
        rec = next_recommendation(self.space, self.encoder, self.surrogate, **call)
        self._pending.extend(rec.points)
        self.history.append({"iteration": self._iteration, "kind": "model",
                             "n": len(rec.points), "acquisition": rec.acquisition,
                             "path": rec.path, "best_f": rec.best_f,
                             "notes": list(rec.notes)})
        return rec

    def _initial(self, n: int) -> Design:
        want = max(n, self.n_init - self.n_observations) if self.n_observations == 0 else n
        if self.prior is not None and self.prior_confidence > 0:
            return prior_initial_design(self.space, want, self.prior,
                                        confidence=self.prior_confidence,
                                        seed=self.seed)
        return initial_design(self.space, want, method=self.design_method,
                              seed=self.seed, encoder=self.encoder)

    # ------------------------------------------------------------- report

    def summary(self) -> dict:
        return {
            "n_observations": self.n_observations,
            "n_skipped": self.n_skipped,
            "best_y": self.best_y,
            "best_point": self.best_point,
            "iterations": self._iteration,
            "acquisition": self.acquisition,
            "surrogate": None if self.surrogate is None else self.surrogate.kind,
            "skipped_reasons": _counts(s.reason for s in self._skipped),
        }

    def __repr__(self):
        return (f"Campaign(n={self.n_observations}, skipped={self.n_skipped}, "
                f"best={self.best_y})")


def _counts(it):
    out: Dict[str, int] = {}
    for v in it:
        out[v] = out.get(v, 0) + 1
    return out
