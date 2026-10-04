"""Adapters from data we already have into the general search-space format.

The point of these is not convenience. If the general tool can be pointed at
the frozen benchmark matrices and reproduce the protocol, then the benchmark is
a special case of the tool rather than a separate code path -- which is the
only honest way to claim the new work is comparable with the published numbers.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .encode import DESCRIPTOR, Encoder
from .space import CATEGORICAL, Factor, SearchSpace


def space_from_matrix(X: np.ndarray, name: str = "candidate",
                      descriptor_names: Optional[Sequence[str]] = None,
                      space_name: Optional[str] = None
                      ) -> Tuple[SearchSpace, Encoder]:
    """A pre-featurised candidate matrix as a one-factor categorical space.

    Each row becomes a level; the row itself becomes that level's descriptor
    vector. Columns that are constant across every row are dropped here, once,
    because they cannot distinguish any candidate -- and dropping them at the
    adapter (where it is visible) is better than letting the encoder emit a
    column of zeros.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"expected a 2-D matrix, got shape {X.shape}")
    if not np.isfinite(X).all():
        n_bad = int((~np.isfinite(X)).sum())
        raise ValueError(
            f"descriptor matrix holds {n_bad} non-finite value(s). These are "
            f"refused rather than dropped or filled -- see the EDBO finding."
        )

    keep = X.max(axis=0) > X.min(axis=0)
    Xk = X[:, keep]
    if Xk.shape[1] == 0:
        raise ValueError("every descriptor column is constant; nothing to model")

    names = None
    if descriptor_names is not None:
        if len(descriptor_names) != X.shape[1]:
            raise ValueError(
                f"{len(descriptor_names)} descriptor names for {X.shape[1]} columns"
            )
        names = [descriptor_names[i] for i in np.flatnonzero(keep)]

    levels = list(range(len(X)))
    factor = Factor(name, CATEGORICAL, levels=levels,
                    descriptors={i: Xk[i].tolist() for i in levels},
                    descriptor_names=names)
    space = SearchSpace([factor], name=space_name)
    return space, Encoder(space, mode=DESCRIPTOR)


def lookup_oracle(y: np.ndarray, name: str = "candidate") -> Callable[[Dict], float]:
    """The ground-truth yield for a candidate, for benchmark runs."""
    y = np.asarray(y, dtype=float).reshape(-1)

    def f(point: Dict) -> float:
        return float(y[int(point[name])])

    return f


def dropped_columns(X: np.ndarray) -> int:
    X = np.asarray(X, dtype=float)
    return int((X.max(axis=0) <= X.min(axis=0)).sum())
