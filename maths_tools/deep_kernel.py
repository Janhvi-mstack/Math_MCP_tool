"""The deep-kernel GP, vendored.

A Linear(d -> 64) + ELU projection sits INSIDE the GP kernel and is fitted
jointly with the GP hyperparameters by the marginal likelihood, so the kernel
learns its own representation rather than taking fixed features.

This is a VERBATIM copy of the classes that produced the published benchmark
numbers. It is vendored rather than imported so the tool is self-contained: a
product must not reach into a benchmark directory in order to work.

Because it is a copy it can drift from the original. `tests/test_vendored.py`
guards that -- it asserts the source here is byte-identical to the original
wherever the original is present, and that both produce the same fit on the
same data.

PROVENANCE
    source : the reference implementation these numbers were produced with
    copied : ProjectionLayer, PrecomputedProjection, _kernel, _apply_hypers, DeepGP
    note   : the originals carry comments marking two deliberate paper-vs-code
             discrepancies. Both are preserved exactly; do not "fix" them, the
             published numbers depend on them.
"""
from __future__ import annotations

import gpytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.optim.fit import fit_gpytorch_mll_torch
from gpytorch.constraints import GreaterThan
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch.nn import init
from torch.optim.lr_scheduler import StepLR

DTYPE = torch.float64


class ProjectionLayer(nn.Module):
    """gollum/featurization/deep.py ProjectionLayer.

    `apply_dropout` is the paper/code switch: the shipped code creates the
    dropout module but forward() is fc1 -> elu only.
    """

    def __init__(self, input_dim, projection_dim=64, apply_dropout=False, p=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p)
        self.fc1 = nn.Linear(input_dim, projection_dim)
        self.fc1.bias.data.fill_(0.01)
        init.xavier_uniform_(self.fc1.weight)
        self.apply_dropout = apply_dropout

    def forward(self, x):
        x = self.fc1(x)
        if self.apply_dropout:          # paper Eq.10; OFF in the shipped code
            x = self.dropout(x)
        return F.elu(x)


class PrecomputedProjection(nn.Module):
    """PLLM variant: frozen LLM, so embeddings are precomputed once and only
    the projection is trainable. gollum's configs/pllm.yaml wires
    ProjectionLayer directly as the `finetuning_model`, and train.py:159
    refuses get_tokens for PLLM, confirming this path takes embeddings."""

    def __init__(self, input_dim, projection_dim=64, apply_proj_dropout=False):
        super().__init__()
        self.projection = ProjectionLayer(input_dim, projection_dim, apply_proj_dropout)
        self.projection.to(DTYPE)

    def forward(self, x):
        return self.projection(x)


def _kernel():
    return ScaleKernel(MaternKernel(nu=2.5, eps=1e-6), eps=1e-6)


def _apply_hypers(model, noise=1.0, lengthscale=1.0, outputscale=1.0, noise_constraint=1e-4):
    model.likelihood.noise_covar.register_constraint("raw_noise", GreaterThan(noise_constraint))
    hypers = {
        "likelihood.noise_covar.noise": torch.tensor(noise),
        "covar_module.base_kernel.lengthscale": torch.tensor(lengthscale),
        "covar_module.outputscale": torch.tensor(outputscale),
    }
    existing = {n for n, _ in model.named_parameters()}
    model.initialize(**{k: v for k, v in hypers.items() if k in existing})


class DeepGP(SingleTaskGP):
    """gollum.surrogate_models.gp.DeepGP — the LLM/projection lives INSIDE the
    kernel, and the GP marginal likelihood is the only loss."""

    def __init__(self, train_x, train_y, finetuning_model,
                 gp_lr=0.2, ft_lr=0.002, gp_step_lr=0.95, wd=1e-3, wd_llm=1e-3,
                 scale_embeddings=True, clip_mode="code", clip_norm=1.0,
                 step_limit=None):
        train_x = train_x.to(DTYPE)
        train_y = train_y.to(DTYPE)
        super().__init__(
            train_X=train_x, train_Y=train_y,
            likelihood=GaussianLikelihood(),
            covar_module=_kernel(), mean_module=ConstantMean(),
            outcome_transform=Standardize(train_y.shape[-1]),
            input_transform=None,
        )
        _apply_hypers(self)
        self.finetuning_model = finetuning_model
        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1.0, 1.0)
        self.scale_embeddings = scale_embeddings
        self.gp_lr, self.ft_lr, self.gp_step_lr = gp_lr, ft_lr, gp_step_lr
        self.wd, self.wd_llm = wd, wd_llm
        self.clip_mode, self.clip_norm = clip_mode, clip_norm
        self.step_limit = step_limit
        self._train_x_raw = train_x

    def forward(self, x):
        feats = self.finetuning_model(x)
        if self.scale_embeddings:
            feats = self.scale_to_bounds(feats)
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(feats), self.covar_module(feats)
        )

    def fit(self):
        self.train(); self.likelihood.train(); self.finetuning_model.train()
        mll = ExactMarginalLogLikelihood(self.likelihood, self)
        mll.train()

        optimizer = torch.optim.AdamW(
            [
                {"params": [p for p in self.finetuning_model.parameters() if p.requires_grad],
                 "lr": self.ft_lr, "weight_decay": self.wd_llm},
                {"params": list(self.covar_module.parameters())},
                {"params": list(self.mean_module.parameters())},
                {"params": list(self.likelihood.parameters())},
            ],
            lr=self.gp_lr, weight_decay=self.wd,
        )
        scheduler = StepLR(optimizer, step_size=1, gamma=self.gp_step_lr)

        # DISCREPANCY 2 -- gp.py calls this here, before any backward pass, so
        # it can never clip. Mode A replicates that; Mode B clips after backward.
        if self.clip_mode == "code":
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.clip_norm)

        self._n_steps = 0
        self._grad_seen = 0.0
        clip_after = self.clip_mode == "paper"
        clip_norm, params = self.clip_norm, self.parameters

        def gp_closure():
            nonlocal clip_after
            optimizer.zero_grad()
            output = self(self._train_x_raw)
            loss = -mll(output, self.train_targets.squeeze(-1))
            loss.backward()
            if clip_after:
                torch.nn.utils.clip_grad_norm_(params(), max_norm=clip_norm)
            self._n_steps += 1
            self._grad_seen += sum(
                float(p.grad.abs().sum()) for p in self.finetuning_model.parameters()
                if p.requires_grad and p.grad is not None
            )
            return loss, [p.grad for p in self.parameters() if p.requires_grad]

        kw = {"optimizer": optimizer, "scheduler": scheduler}
        if self.step_limit is not None:
            kw["step_limit"] = self.step_limit
        fit_gpytorch_mll(
            mll, closure=gp_closure, optimizer=fit_gpytorch_mll_torch,
            optimizer_kwargs=kw,
        )
        return {"n_steps": self._n_steps, "grad_mass_into_featurizer": self._grad_seen}
