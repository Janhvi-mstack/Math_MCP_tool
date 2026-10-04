"""Search-space definition.

The format is the part we cannot change later, so it is defined here on its own
with no dependency on torch, rdkit or any model code.

Design rule that the rest of the package depends on:

    Every number needed to encode a factor comes from the factor's own
    definition -- its bounds, or its declared level set. Nothing is ever
    derived from the candidate set or from observed data.

That rule is why adding a candidate to a space cannot change the encoding of
the candidates already in it. EDBO does the opposite (it decorrelates each
component against the current members), which is why its feature set shifts
when the space changes.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from itertools import product
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

CONTINUOUS = "continuous"
INTEGER = "integer"
ORDINAL = "ordinal"
CATEGORICAL = "categorical"

NUMERIC_TYPES = (CONTINUOUS, INTEGER)
LEVEL_TYPES = (ORDINAL, CATEGORICAL)
ALL_TYPES = NUMERIC_TYPES + LEVEL_TYPES

LINEAR = "linear"
LOG = "log"

# A space bigger than this is treated as not-enumerable even if every factor is
# discrete. Scoring a table this size is slower than optimising the continuous
# relaxation, so the dispatch should send it down the continuous path.
ENUMERATION_LIMIT = 500_000


class SpaceError(ValueError):
    """Raised when a space or factor definition is not self-consistent."""


@dataclass
class Factor:
    """One thing the chemist can vary.

    Parameters
    ----------
    name : str
        Unique within a space. Used as the key in every recommendation.
    type : str
        One of continuous, integer, ordinal, categorical.
    bounds : (low, high), numeric types only
        Inclusive. `high` must be strictly greater than `low`.
    levels : list, level types only
        The allowed values, in order. For `ordinal` the order is meaningful and
        is what gets encoded; for `categorical` it only fixes column order.
    descriptors : {level: [floats]}, optional, level types only
        Numbers describing each level -- DFT, fingerprints, whatever. Must
        cover every level and all vectors must be the same length. When
        present these replace one-hot encoding for that factor.
    descriptor_names : list of str, optional
        Column names for the descriptor vectors. Length must match.
    transform : "linear" | "log"
        Numeric types only. `log` is for ranges spanning orders of magnitude;
        it requires a strictly positive lower bound.
    unit : str, optional
        Carried through for display. Never used in the maths.
    """

    name: str
    type: str
    bounds: Optional[Tuple[float, float]] = None
    levels: Optional[List[Any]] = None
    descriptors: Optional[Dict[Any, Sequence[float]]] = None
    descriptor_names: Optional[List[str]] = None
    transform: str = LINEAR
    unit: Optional[str] = None

    def __post_init__(self):
        self._validate()

    # -- validation ---------------------------------------------------------

    def _validate(self):
        if not isinstance(self.name, str) or not self.name.strip():
            raise SpaceError("factor name must be a non-empty string")
        if self.type not in ALL_TYPES:
            raise SpaceError(
                f"{self.name}: type must be one of {ALL_TYPES}, got {self.type!r}"
            )
        if self.transform not in (LINEAR, LOG):
            raise SpaceError(
                f"{self.name}: transform must be {LINEAR!r} or {LOG!r}, "
                f"got {self.transform!r}"
            )

        if self.type in NUMERIC_TYPES:
            self._validate_numeric()
        else:
            self._validate_levels()

    def _validate_numeric(self):
        if self.bounds is None:
            raise SpaceError(f"{self.name}: {self.type} factor requires bounds")
        if self.levels is not None:
            raise SpaceError(f"{self.name}: {self.type} factor must not have levels")
        if self.descriptors is not None:
            raise SpaceError(
                f"{self.name}: descriptors apply to ordinal/categorical factors only"
            )
        if len(self.bounds) != 2:
            raise SpaceError(f"{self.name}: bounds must be (low, high)")

        low, high = (float(b) for b in self.bounds)
        if not (math.isfinite(low) and math.isfinite(high)):
            raise SpaceError(f"{self.name}: bounds must be finite")
        if high <= low:
            raise SpaceError(f"{self.name}: bounds high must exceed low, got {self.bounds}")
        self.bounds = (low, high)

        if self.transform == LOG and low <= 0:
            raise SpaceError(
                f"{self.name}: log transform requires a strictly positive lower "
                f"bound, got {low}"
            )
        if self.type == INTEGER:
            if low != int(low) or high != int(high):
                raise SpaceError(f"{self.name}: integer bounds must be whole numbers")
            if int(high) - int(low) < 1:
                raise SpaceError(f"{self.name}: integer factor needs at least 2 values")

    def _validate_levels(self):
        if self.levels is None:
            raise SpaceError(f"{self.name}: {self.type} factor requires levels")
        if self.bounds is not None:
            raise SpaceError(f"{self.name}: {self.type} factor must not have bounds")
        if self.transform != LINEAR:
            raise SpaceError(f"{self.name}: transform applies to numeric factors only")
        if len(self.levels) < 2:
            raise SpaceError(f"{self.name}: needs at least 2 levels")
        if len(set(map(_hashable, self.levels))) != len(self.levels):
            raise SpaceError(f"{self.name}: levels must be unique")

        if self.descriptors is not None:
            self._validate_descriptors()

    def _validate_descriptors(self):
        missing = [l for l in self.levels if _hashable(l) not in
                   {_hashable(k) for k in self.descriptors}]
        if missing:
            raise SpaceError(
                f"{self.name}: descriptors missing for level(s) {missing!r}. "
                f"Partial descriptor coverage is refused rather than filled with "
                f"NaN -- supply every level or none."
            )
        extra = [k for k in self.descriptors
                 if _hashable(k) not in {_hashable(l) for l in self.levels}]
        if extra:
            raise SpaceError(
                f"{self.name}: descriptors given for unknown level(s) {extra!r}"
            )

        widths = {len(v) for v in self.descriptors.values()}
        if len(widths) != 1:
            raise SpaceError(
                f"{self.name}: descriptor vectors have differing lengths {sorted(widths)}"
            )
        width = widths.pop()
        if width == 0:
            raise SpaceError(f"{self.name}: descriptor vectors are empty")

        for level, vec in self.descriptors.items():
            for j, v in enumerate(vec):
                if not isinstance(v, (int, float)) or not math.isfinite(float(v)):
                    raise SpaceError(
                        f"{self.name}: descriptor for level {level!r} has a "
                        f"non-finite value at position {j} ({v!r}). NaN "
                        f"descriptors are refused, not dropped."
                    )

        if self.descriptor_names is not None and len(self.descriptor_names) != width:
            raise SpaceError(
                f"{self.name}: {len(self.descriptor_names)} descriptor_names for "
                f"{width} descriptor columns"
            )

    # -- properties ---------------------------------------------------------

    @property
    def is_numeric(self) -> bool:
        return self.type in NUMERIC_TYPES

    @property
    def has_descriptors(self) -> bool:
        return self.descriptors is not None

    @property
    def descriptor_width(self) -> int:
        if not self.has_descriptors:
            return 0
        return len(next(iter(self.descriptors.values())))

    @property
    def n_levels(self) -> Optional[int]:
        """Number of distinct settings, or None if uncountable."""
        if self.type == CONTINUOUS:
            return None
        if self.type == INTEGER:
            low, high = self.bounds
            return int(high) - int(low) + 1
        return len(self.levels)

    def values(self) -> List[Any]:
        """Every allowed setting. Raises for continuous factors."""
        if self.type == CONTINUOUS:
            raise SpaceError(f"{self.name}: a continuous factor cannot be listed")
        if self.type == INTEGER:
            low, high = self.bounds
            return list(range(int(low), int(high) + 1))
        return list(self.levels)

    # -- serialisation ------------------------------------------------------

    def to_dict(self) -> dict:
        out = {"name": self.name, "type": self.type}
        if self.bounds is not None:
            out["bounds"] = list(self.bounds)
        if self.levels is not None:
            out["levels"] = list(self.levels)
        if self.descriptors is not None:
            out["descriptors"] = {str(k): list(map(float, v))
                                  for k, v in self.descriptors.items()}
        if self.descriptor_names is not None:
            out["descriptor_names"] = list(self.descriptor_names)
        if self.transform != LINEAR:
            out["transform"] = self.transform
        if self.unit is not None:
            out["unit"] = self.unit
        return out

    @classmethod
    def from_dict(cls, d: dict) -> "Factor":
        d = dict(d)
        bounds = d.pop("bounds", None)
        return cls(
            name=d.pop("name"),
            type=d.pop("type"),
            bounds=tuple(bounds) if bounds is not None else None,
            levels=d.pop("levels", None),
            descriptors=d.pop("descriptors", None),
            descriptor_names=d.pop("descriptor_names", None),
            transform=d.pop("transform", LINEAR),
            unit=d.pop("unit", None),
        )


@dataclass
class Constraint:
    """A linear inequality over numeric factors: sum(w_i * x_i) <= rhs.

    Only numeric factors can appear. Evaluated on the factor's own units, not
    on the encoded scale, so it reads the way a chemist would write it:

        Constraint({"cat_loading": 1.0, "additive_loading": 1.0}, 10.0)
        -> cat_loading + additive_loading <= 10 mol%
    """

    weights: Dict[str, float]
    rhs: float
    name: Optional[str] = None

    def __post_init__(self):
        if not self.weights:
            raise SpaceError("constraint needs at least one factor weight")
        self.weights = {k: float(v) for k, v in self.weights.items()}
        self.rhs = float(self.rhs)

    def satisfied_by(self, point: Dict[str, Any], tol: float = 1e-9) -> bool:
        total = sum(w * float(point[n]) for n, w in self.weights.items())
        return total <= self.rhs + tol

    def to_dict(self) -> dict:
        out = {"weights": dict(self.weights), "rhs": self.rhs}
        if self.name:
            out["name"] = self.name
        return out

    @classmethod
    def from_dict(cls, d: dict) -> "Constraint":
        return cls(weights=d["weights"], rhs=d["rhs"], name=d.get("name"))


@dataclass
class SearchSpace:
    """The set of experiments a tool is allowed to recommend."""

    factors: List[Factor]
    constraints: List[Constraint] = field(default_factory=list)
    name: Optional[str] = None

    def __post_init__(self):
        if not self.factors:
            raise SpaceError("a search space needs at least one factor")
        names = [f.name for f in self.factors]
        dupes = {n for n in names if names.count(n) > 1}
        if dupes:
            raise SpaceError(f"duplicate factor names: {sorted(dupes)}")

        by_name = {f.name: f for f in self.factors}
        for c in self.constraints:
            for n in c.weights:
                if n not in by_name:
                    raise SpaceError(f"constraint references unknown factor {n!r}")
                if not by_name[n].is_numeric:
                    raise SpaceError(
                        f"constraint references {n!r}, which is "
                        f"{by_name[n].type}. Constraints apply to numeric "
                        f"factors only."
                    )

    def __len__(self) -> int:
        return len(self.factors)

    def __getitem__(self, name: str) -> Factor:
        for f in self.factors:
            if f.name == name:
                return f
        raise KeyError(name)

    @property
    def names(self) -> List[str]:
        return [f.name for f in self.factors]

    @property
    def has_continuous(self) -> bool:
        return any(f.type == CONTINUOUS for f in self.factors)

    @property
    def cardinality(self) -> Optional[int]:
        """Total number of distinct experiments, or None if infinite."""
        if self.has_continuous:
            return None
        total = 1
        for f in self.factors:
            total *= f.n_levels
        return total

    @property
    def is_enumerable(self) -> bool:
        """The single check the whole package dispatches on.

        True  -> build the table and score every row (what we do today).
        False -> no table exists; the acquisition function has to be optimised.
        """
        card = self.cardinality
        return card is not None and card <= ENUMERATION_LIMIT

    def enumerate(self, apply_constraints: bool = True) -> List[Dict[str, Any]]:
        """Every allowed experiment, as a list of dicts.

        Raises rather than truncating if the space is not enumerable -- a
        silently truncated candidate set is the kind of defect that produces
        plausible numbers and never raises.
        """
        if self.has_continuous:
            raise SpaceError(
                f"space has continuous factor(s) "
                f"{[f.name for f in self.factors if f.type == CONTINUOUS]}; "
                f"it cannot be enumerated"
            )
        card = self.cardinality
        if card > ENUMERATION_LIMIT:
            raise SpaceError(
                f"space has {card:,} combinations, above the enumeration limit "
                f"of {ENUMERATION_LIMIT:,}. Use the continuous path instead."
            )

        rows = []
        for combo in product(*[f.values() for f in self.factors]):
            point = dict(zip(self.names, combo))
            if apply_constraints and not self.satisfies(point):
                continue
            rows.append(point)
        return rows

    def satisfies(self, point: Dict[str, Any]) -> bool:
        return all(c.satisfied_by(point) for c in self.constraints)

    def validate_point(self, point: Dict[str, Any]) -> None:
        """Raise if `point` is not a legal experiment in this space."""
        missing = set(self.names) - set(point)
        if missing:
            raise SpaceError(f"point is missing factor(s) {sorted(missing)}")
        extra = set(point) - set(self.names)
        if extra:
            raise SpaceError(f"point has unknown factor(s) {sorted(extra)}")

        for f in self.factors:
            v = point[f.name]
            if f.is_numeric:
                fv = float(v)
                low, high = f.bounds
                if not (low - 1e-9 <= fv <= high + 1e-9):
                    raise SpaceError(
                        f"{f.name}={v} is outside bounds {f.bounds}"
                    )
                if f.type == INTEGER and abs(fv - round(fv)) > 1e-9:
                    raise SpaceError(f"{f.name}={v} is not an integer")
            else:
                allowed = {_hashable(l) for l in f.levels}
                if _hashable(v) not in allowed:
                    raise SpaceError(
                        f"{f.name}={v!r} is not one of the declared levels"
                    )
        if not self.satisfies(point):
            raise SpaceError(f"point violates a constraint: {point}")

    # -- serialisation ------------------------------------------------------

    def to_dict(self) -> dict:
        out = {"factors": [f.to_dict() for f in self.factors]}
        if self.constraints:
            out["constraints"] = [c.to_dict() for c in self.constraints]
        if self.name:
            out["name"] = self.name
        return out

    @classmethod
    def from_dict(cls, d: dict) -> "SearchSpace":
        return cls(
            factors=[Factor.from_dict(f) for f in d["factors"]],
            constraints=[Constraint.from_dict(c) for c in d.get("constraints", [])],
            name=d.get("name"),
        )

    def to_json(self, **kw) -> str:
        return json.dumps(self.to_dict(), **kw)

    @classmethod
    def from_json(cls, s: str) -> "SearchSpace":
        return cls.from_dict(json.loads(s))

    def summary(self) -> str:
        lines = [f"SearchSpace({self.name or 'unnamed'})"]
        for f in self.factors:
            if f.type == CONTINUOUS:
                d = f"{f.bounds[0]:g} to {f.bounds[1]:g}"
                if f.transform == LOG:
                    d += " (log)"
            elif f.type == INTEGER:
                d = f"{int(f.bounds[0])}..{int(f.bounds[1])}"
            else:
                d = f"{len(f.levels)} levels"
                if f.has_descriptors:
                    d += f", {f.descriptor_width} descriptors"
            lines.append(f"  {f.name:<20} {f.type:<12} {d}")
        for c in self.constraints:
            terms = " + ".join(f"{w:g}*{n}" for n, w in c.weights.items())
            lines.append(f"  constraint: {terms} <= {c.rhs:g}")
        card = self.cardinality
        lines.append(
            f"  cardinality: {'infinite' if card is None else format(card, ',')}"
            f"   enumerable: {self.is_enumerable}"
        )
        return "\n".join(lines)


def _hashable(v):
    """Levels are usually strings or numbers, but tolerate lists."""
    if isinstance(v, (list, tuple)):
        return tuple(v)
    return v
