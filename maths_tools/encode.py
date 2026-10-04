"""Turning experiments into numbers, and numbers back into experiments.

Two encodings, because the two model paths want different things:

  "descriptor"  every column continuous in [0, 1]. Categorical factors become
                descriptor blocks, or one-hot if no descriptors were given.
                This is the enumerated path -- build the table, score the rows.

  "mixed"       numeric factors normalised to [0, 1]; each categorical factor
                occupies ONE column holding its level index. Those column
                indices are reported as `cat_dims`, which is what
                MixedSingleTaskGP and optimize_acqf_mixed expect.

The rule this module exists to enforce:

    Every normalisation constant is derived from the factor definition and
    fixed at construction. None of them depend on the candidate set, on which
    points have been observed, or on the order rows arrive in.

An Encoder can be serialised and re-loaded, so a space that gains a level later
can reuse the original constants and produce byte-identical columns for the
levels it already had.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .space import (
    CATEGORICAL,
    CONTINUOUS,
    INTEGER,
    LOG,
    ORDINAL,
    Factor,
    SearchSpace,
    SpaceError,
    _hashable,
)

DESCRIPTOR = "descriptor"
MIXED = "mixed"
AUTO = "auto"


class EncodingError(ValueError):
    pass


class _Block:
    """How one factor maps onto columns. Built once, then read-only."""

    __slots__ = ("factor", "start", "width", "kind", "names",
                 "offset", "scale", "level_keys")

    def __init__(self, factor, start, width, kind, names,
                 offset=None, scale=None, level_keys=None):
        self.factor = factor
        self.start = start
        self.width = width
        self.kind = kind            # numeric | ordinal | onehot | desc | catindex
        self.names = names
        self.offset = offset        # np array, subtracted before scaling
        self.scale = scale          # np array, divided after offset
        self.level_keys = level_keys

    @property
    def stop(self):
        return self.start + self.width


class Encoder:
    """Maps between experiments (dicts) and model input (float arrays)."""

    def __init__(self, space: SearchSpace, mode: str = AUTO,
                 _constants: Optional[dict] = None):
        self.space = space
        if mode == AUTO:
            mode = DESCRIPTOR if space.is_enumerable else MIXED
        if mode not in (DESCRIPTOR, MIXED):
            raise EncodingError(
                f"mode must be {DESCRIPTOR!r}, {MIXED!r} or {AUTO!r}, got {mode!r}"
            )
        self.mode = mode
        self._blocks: List[_Block] = []
        self._build(_constants or {})

    # -- construction -------------------------------------------------------

    def _build(self, pinned: dict):
        col = 0
        for f in self.space.factors:
            blk = self._block_for(f, col, pinned.get(f.name))
            self._blocks.append(blk)
            col = blk.stop
        self.n_columns = col

    def _block_for(self, f: Factor, start: int, pin: Optional[dict]) -> _Block:
        if f.type in (CONTINUOUS, INTEGER):
            low, high = f.bounds
            if f.transform == LOG:
                low, high = np.log(low), np.log(high)
            off = np.array([low], dtype=float)
            sc = np.array([high - low], dtype=float)
            if pin:
                off, sc = np.array(pin["offset"]), np.array(pin["scale"])
            return _Block(f, start, 1, "numeric", [f.name], off, sc)

        if f.type == ORDINAL and not f.has_descriptors:
            n = len(f.levels)
            off = np.array([0.0])
            sc = np.array([float(n - 1)])
            if pin:
                off, sc = np.array(pin["offset"]), np.array(pin["scale"])
            return _Block(f, start, 1, "ordinal", [f.name], off, sc,
                          level_keys=[_hashable(l) for l in f.levels])

        if self.mode == MIXED and f.type == CATEGORICAL:
            # One column holding the level index. Deliberately NOT normalised:
            # CategoricalKernel compares these for equality, not distance.
            return _Block(f, start, 1, "catindex", [f.name],
                          np.array([0.0]), np.array([1.0]),
                          level_keys=[_hashable(l) for l in f.levels])

        if f.has_descriptors:
            keys = [_hashable(l) for l in f.levels]
            mat = np.array([[float(v) for v in f.descriptors[l]]
                            for l in f.levels], dtype=float)
            lo = mat.min(axis=0)
            hi = mat.max(axis=0)
            span = hi - lo
            # A descriptor that is constant across the declared levels carries
            # no information. Scale 1.0 maps it to a constant 0.0 column rather
            # than dividing by zero. It is kept, not dropped, so the column
            # count is a function of the definition alone.
            span[span == 0] = 1.0
            if pin:
                lo, span = np.array(pin["offset"]), np.array(pin["scale"])
            names = (f.descriptor_names
                     or [f"{f.name}_d{i}" for i in range(mat.shape[1])])
            names = [f"{f.name}::{n}" for n in names]
            return _Block(f, start, mat.shape[1], "desc", names,
                          lo, span, level_keys=keys)

        # one-hot, in declared level order (never in first-seen order)
        keys = [_hashable(l) for l in f.levels]
        names = [f"{f.name}={l}" for l in f.levels]
        return _Block(f, start, len(f.levels), "onehot", names,
                      np.zeros(len(f.levels)), np.ones(len(f.levels)),
                      level_keys=keys)

    # -- introspection ------------------------------------------------------

    @property
    def column_names(self) -> List[str]:
        out = []
        for b in self._blocks:
            out.extend(b.names)
        return out

    @property
    def cat_dims(self) -> List[int]:
        """Column indices MixedSingleTaskGP should treat as categorical."""
        return [b.start for b in self._blocks if b.kind == "catindex"]

    @property
    def cont_dims(self) -> List[int]:
        cat = set(self.cat_dims)
        return [i for i in range(self.n_columns) if i not in cat]

    def bounds(self) -> np.ndarray:
        """2 x d array of (lower, upper) for acquisition optimisation."""
        lo = np.zeros(self.n_columns)
        hi = np.ones(self.n_columns)
        for b in self._blocks:
            if b.kind == "catindex":
                hi[b.start] = len(b.level_keys) - 1
        return np.vstack([lo, hi])

    def categorical_choices(self) -> Dict[int, List[int]]:
        """{column index: allowed integer values} for optimize_acqf_mixed."""
        return {b.start: list(range(len(b.level_keys)))
                for b in self._blocks if b.kind == "catindex"}

    # -- forward ------------------------------------------------------------

    def encode(self, points: Sequence[Dict[str, Any]],
               validate: bool = True) -> np.ndarray:
        if isinstance(points, dict):
            points = [points]
        X = np.zeros((len(points), self.n_columns), dtype=float)
        for i, p in enumerate(points):
            if validate:
                self.space.validate_point(p)
            for b in self._blocks:
                X[i, b.start:b.stop] = self._encode_block(b, p[b.factor.name])
        return X

    def _encode_block(self, b: _Block, value) -> np.ndarray:
        f = b.factor
        if b.kind == "numeric":
            v = float(value)
            if f.transform == LOG:
                v = np.log(v)
            return (np.array([v]) - b.offset) / b.scale

        idx = self._level_index(b, value)

        if b.kind == "ordinal":
            return (np.array([float(idx)]) - b.offset) / b.scale
        if b.kind == "catindex":
            return np.array([float(idx)])
        if b.kind == "onehot":
            row = np.zeros(b.width)
            row[idx] = 1.0
            return row
        if b.kind == "desc":
            raw = np.array([float(v) for v in f.descriptors[f.levels[idx]]])
            return (raw - b.offset) / b.scale
        raise EncodingError(f"unknown block kind {b.kind!r}")

    @staticmethod
    def _level_index(b: _Block, value) -> int:
        try:
            return b.level_keys.index(_hashable(value))
        except ValueError:
            raise EncodingError(
                f"{b.factor.name}: {value!r} is not a declared level"
            ) from None

    # -- inverse ------------------------------------------------------------

    def decode(self, X: np.ndarray) -> List[Dict[str, Any]]:
        """Numbers back to experiments.

        A recommendation is only useful as chemistry, so this is not optional
        plumbing -- it is how the tool answers.
        """
        X = np.atleast_2d(np.asarray(X, dtype=float))
        out = []
        for row in X:
            point = {}
            for b in self._blocks:
                point[b.factor.name] = self._decode_block(b, row[b.start:b.stop])
            out.append(point)
        return out

    def _decode_block(self, b: _Block, chunk: np.ndarray):
        f = b.factor
        if b.kind == "numeric":
            v = float(chunk[0]) * float(b.scale[0]) + float(b.offset[0])
            if f.transform == LOG:
                v = float(np.exp(v))
            low, high = f.bounds
            v = min(max(v, low), high)
            return int(round(v)) if f.type == INTEGER else v

        if b.kind == "ordinal":
            idx = int(round(float(chunk[0]) * float(b.scale[0]) + float(b.offset[0])))
            idx = min(max(idx, 0), len(f.levels) - 1)
            return f.levels[idx]

        if b.kind == "catindex":
            idx = int(round(float(chunk[0])))
            idx = min(max(idx, 0), len(f.levels) - 1)
            return f.levels[idx]

        if b.kind == "onehot":
            return f.levels[int(np.argmax(chunk))]

        if b.kind == "desc":
            # Nearest declared level in encoded space. Exact for any point the
            # encoder produced; sensible for a relaxation that landed between.
            table = np.array([
                (np.array([float(v) for v in f.descriptors[l]]) - b.offset) / b.scale
                for l in f.levels
            ])
            return f.levels[int(np.argmin(((table - chunk) ** 2).sum(axis=1)))]

        raise EncodingError(f"unknown block kind {b.kind!r}")

    # -- serialisation ------------------------------------------------------

    def constants(self) -> dict:
        return {b.factor.name: {"offset": b.offset.tolist(),
                                "scale": b.scale.tolist()}
                for b in self._blocks}

    def to_dict(self) -> dict:
        return {"mode": self.mode,
                "space": self.space.to_dict(),
                "constants": self.constants()}

    @classmethod
    def from_dict(cls, d: dict) -> "Encoder":
        return cls(SearchSpace.from_dict(d["space"]),
                   mode=d["mode"], _constants=d.get("constants"))

    def __repr__(self):
        return (f"Encoder(mode={self.mode!r}, columns={self.n_columns}, "
                f"cat_dims={self.cat_dims})")
