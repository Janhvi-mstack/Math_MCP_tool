"""Tool 2, part one -- the model that predicts.

Three surrogates behind one interface. Which one you get depends on the space,
not on a flag the caller has to reason about:

  deep     EDBO-style deep kernel: a trainable projection sits inside the GP
           kernel and is fitted by the marginal likelihood. Our best performer
           on the benchmark. Needs a descriptor encoding.
  mixed    MixedSingleTaskGP -- Matern on the numeric columns, a categorical
           kernel on the level-index columns. The path once anything is
           continuous.
  plain    SingleTaskGP with Matern-5/2. All-numeric spaces, and the control.

Every one of them answers the same two questions: what do you predict here,
and how sure are you. The acquisition layer never needs to know which it got.

The deep kernel is VENDORED in `maths_tools/deep_kernel.py` -- a verbatim copy
of the classes that produced the published numbers. A product must not reach
into a benchmark directory to work. `tests/test_vendored.py` guards the copy
against drift by diffing it against the original whenever the original is
present.
"""

from __future__ import annotations

import os
import sys
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from .encode import DESCRIPTOR, MIXED, Encoder

DTYPE = torch.float64

DEEP = "deep"
MIXED_GP = "mixed"
PLAIN = "plain"
AUTO = "auto"
KINDS = (DEEP, MIXED_GP, PLAIN)




class SurrogateError(RuntimeError):
    pass


def _load_frozen():
    """The vendored deep kernel. Kept as a function so the import is lazy."""
    from maths_tools import deep_kernel as G
    return G


# --------------------------------------------------------------- base class

class Surrogate:
    """Common interface. `model` is the botorch Model the acquisition uses."""

    kind: str = "?"

    def __init__(self):
        self.model = None
        self.train_X: Optional[torch.Tensor] = None
        self.train_Y: Optional[torch.Tensor] = None
        self.fit_info: Dict[str, Any] = {}

    # -- fitting ------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        y = np.asarray(y, dtype=float).reshape(-1)
        if len(X) != len(y):
            raise SurrogateError(f"X has {len(X)} rows but y has {len(y)}")
        if len(X) < 2:
            raise SurrogateError(
                f"need at least 2 observations to fit a GP, got {len(X)}"
            )
        if not np.isfinite(X).all():
            raise SurrogateError("X contains non-finite values")
        if not np.isfinite(y).all():
            raise SurrogateError(
                "y contains non-finite values. An experiment that was never "
                "run must be omitted from X and y entirely, not passed as NaN."
            )
        self.train_X = torch.as_tensor(X, dtype=DTYPE)
        self.train_Y = torch.as_tensor(y, dtype=DTYPE).unsqueeze(-1)
        self.fit_info = self._fit()
        return self.fit_info

    def _fit(self) -> Dict[str, Any]:
        raise NotImplementedError

    # -- prediction ---------------------------------------------------------

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (mean, standard deviation), both shape (n,)."""
        if self.model is None:
            raise SurrogateError("call fit() before predict()")
        Xt = torch.as_tensor(np.atleast_2d(np.asarray(X, dtype=float)), dtype=DTYPE)
        self.model.eval()
        with torch.no_grad(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            post = self.model.posterior(Xt)
            mean = post.mean.squeeze(-1).cpu().numpy()
            std = post.variance.clamp_min(1e-12).sqrt().squeeze(-1).cpu().numpy()
        return mean.reshape(-1), std.reshape(-1)

    @property
    def best_f(self) -> float:
        if self.train_Y is None:
            raise SurrogateError("no observations yet")
        return float(self.train_Y.max())

    def __repr__(self):
        n = 0 if self.train_X is None else len(self.train_X)
        return f"{type(self).__name__}(kind={self.kind!r}, n_train={n})"


# ------------------------------------------------------------------- plain

class PlainGP(Surrogate):
    """SingleTaskGP with Matern-5/2, matching the frozen kernel settings."""

    kind = PLAIN

    def __init__(self, standardize_y: bool = True):
        super().__init__()
        self.standardize_y = standardize_y

    def _fit(self):
        from botorch.fit import fit_gpytorch_mll
        from botorch.models import SingleTaskGP
        from botorch.models.transforms import Standardize
        from gpytorch.constraints import GreaterThan
        from gpytorch.kernels import MaternKernel, ScaleKernel
        from gpytorch.likelihoods import GaussianLikelihood
        from gpytorch.means import ConstantMean
        from gpytorch.mlls import ExactMarginalLogLikelihood

        model = SingleTaskGP(
            train_X=self.train_X, train_Y=self.train_Y,
            likelihood=GaussianLikelihood(),
            covar_module=ScaleKernel(MaternKernel(nu=2.5, eps=1e-6), eps=1e-6),
            mean_module=ConstantMean(),
            outcome_transform=(Standardize(self.train_Y.shape[-1])
                               if self.standardize_y else None),
        )
        model.likelihood.noise_covar.register_constraint("raw_noise", GreaterThan(1e-4))
        model.to(DTYPE)

        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        failures = 0
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit_gpytorch_mll(mll)
        except Exception as e:  # noqa: BLE001
            # Counted, not swallowed silently -- a model that quietly fell back
            # to its initial hyperparameters still returns plausible numbers.
            failures = 1
            self._fit_error = f"{type(e).__name__}: {e}"
        self.model = model
        return {"fit_failures": failures, "n_train": len(self.train_X)}


# ------------------------------------------------------------------- mixed

class MixedGP(Surrogate):
    """MixedSingleTaskGP: Matern on numeric columns, categorical kernel on the
    level-index columns.

    A GP needs a notion of how similar two points are. For numbers that is
    distance. For categories there is no distance, so the categorical kernel
    asks only whether the levels match.
    """

    kind = MIXED_GP

    def __init__(self, cat_dims: List[int]):
        super().__init__()
        if not cat_dims:
            raise SurrogateError(
                "MixedGP needs at least one categorical column; use PlainGP "
                "for an all-numeric space"
            )
        self.cat_dims = list(cat_dims)

    def _fit(self):
        from botorch.fit import fit_gpytorch_mll
        from botorch.models import MixedSingleTaskGP
        from botorch.models.transforms import Standardize
        from gpytorch.mlls import ExactMarginalLogLikelihood

        d = self.train_X.shape[-1]
        bad = [c for c in self.cat_dims if not (0 <= c < d)]
        if bad:
            raise SurrogateError(f"cat_dims {bad} out of range for {d} columns")

        model = MixedSingleTaskGP(
            train_X=self.train_X, train_Y=self.train_Y,
            cat_dims=self.cat_dims,
            outcome_transform=Standardize(self.train_Y.shape[-1]),
        )
        model.to(DTYPE)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        failures = 0
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit_gpytorch_mll(mll)
        except Exception as e:  # noqa: BLE001
            failures = 1
            self._fit_error = f"{type(e).__name__}: {e}"
        self.model = model
        return {"fit_failures": failures, "n_train": len(self.train_X),
                "cat_dims": list(self.cat_dims)}


# -------------------------------------------------------------------- deep

class DeepKernelGP(Surrogate):
    """The frozen EDBO deep kernel, imported not copied.

    A Linear(d -> projection_dim) + ELU projection sits inside the kernel and
    is fitted jointly with the GP hyperparameters by the marginal likelihood.
    Hyperparameters default to the frozen benchmark's values so that a run here
    is comparable with the published numbers.
    """

    kind = DEEP

    def __init__(self, projection_dim: int = 64, gp_lr: float = 0.2,
                 ft_lr: float = 0.002, gp_step_lr: float = 0.95,
                 step_limit: Optional[int] = None, seed: Optional[int] = None,
                 proj_dropout: bool = False, clip_mode: str = "code"):
        """
        proj_dropout / clip_mode exist so this class can be configured to match
        the frozen Mode B baseline exactly, which uses dropout=True and
        clip_mode="paper" (run_repro.build_surrogate). The defaults here are
        dropout=False and clip_mode="code" -- that is what every existing
        maths_tools benchmark result was produced with, so the defaults must
        not change or 4,500 runs stop being reproducible.
        """
        super().__init__()
        self.projection_dim = projection_dim
        self.gp_lr, self.ft_lr, self.gp_step_lr = gp_lr, ft_lr, gp_step_lr
        self.step_limit = step_limit
        self.seed = seed
        self.proj_dropout = proj_dropout
        self.clip_mode = clip_mode

    def _fit(self):
        G = _load_frozen()
        if self.seed is not None:
            torch.manual_seed(self.seed)
        d = self.train_X.shape[-1]
        proj = G.PrecomputedProjection(d, projection_dim=self.projection_dim,
                                       apply_proj_dropout=self.proj_dropout)
        model = G.DeepGP(
            self.train_X, self.train_Y, proj,
            gp_lr=self.gp_lr, ft_lr=self.ft_lr, gp_step_lr=self.gp_step_lr,
            step_limit=self.step_limit, clip_mode=self.clip_mode,
        )
        failures = 0
        info = {}
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                info = model.fit() or {}
        except Exception as e:  # noqa: BLE001
            failures = 1
            self._fit_error = f"{type(e).__name__}: {e}"
        self.model = model
        out = {"fit_failures": failures, "n_train": len(self.train_X),
               "input_dim": int(d), "projection_dim": self.projection_dim,
               "proj_dropout": self.proj_dropout, "clip_mode": self.clip_mode}
        out.update(info)
        return out


# ---------------------------------------------------------------- dispatch

def build_surrogate(encoder: Encoder, kind: str = AUTO, **kw) -> Surrogate:
    """Pick the surrogate that suits the encoding.

    descriptor encoding -> deep kernel (our best performer)
    mixed encoding with categorical columns -> MixedGP
    anything all-numeric -> PlainGP
    """
    if kind == AUTO:
        if encoder.mode == MIXED:
            kind = MIXED_GP if encoder.cat_dims else PLAIN
        else:
            kind = DEEP
    if kind not in KINDS:
        raise SurrogateError(f"kind must be one of {KINDS} or {AUTO!r}, got {kind!r}")

    if kind == MIXED_GP:
        if not encoder.cat_dims:
            raise SurrogateError(
                "the mixed surrogate needs categorical columns, but this "
                "encoder reports none. Use kind='plain'."
            )
        return MixedGP(encoder.cat_dims, **kw)
    if kind == DEEP:
        return DeepKernelGP(**kw)
    return PlainGP(**kw)
