"""Two maths tools for reaction optimisation.

Tool 1 -- the first experiments, before any data exists:
    initial_design(space, n, method="auto")          no prior
    prior_initial_design(space, n, prior, ...)       starting from what we know

Tool 2 -- what to run next, once results are coming back:
    surrogate = build_surrogate(encoder); surrogate.fit(X, y)
    next_recommendation(space, encoder, surrogate, n=1)

Or let Campaign hold the state and call both for you.

Everything works on any search space: continuous, integer, ordinal,
categorical, and any mix. The single check `space.is_enumerable` decides
whether a table gets scored or an acquisition surface gets climbed.
"""

from .space import (
    ALL_TYPES, CATEGORICAL, CONTINUOUS, INTEGER, LINEAR, LOG, ORDINAL,
    Constraint, Factor, SearchSpace, SpaceError,
)
from .encode import AUTO, DESCRIPTOR, MIXED, Encoder, EncodingError
from .design import (
    KMEDOIDS, MAXMIN, METHODS, RANDOM, SOBOL, Design, initial_design,
)
from .priors import (
    CategoricalPrior, CompositePrior, GaussianPrior, Prior, PriorError,
    UniformPrior, pibo_exponent, pibo_weights, prior_initial_design,
    rank_agreement, rgpe_weight,
)
from .surrogate import (
    DEEP, PLAIN, DeepKernelGP, MixedGP, PlainGP, Surrogate, SurrogateError,
    build_surrogate,
)
from .acquisition import (
    ACQUISITIONS, EI, QEI, QNEI, TS, AcquisitionError, Recommendation,
    next_recommendation,
)
from .campaign import Campaign, CampaignError, Observation, Skipped
from .adapters import lookup_oracle, space_from_matrix

__all__ = [
    # space
    "ALL_TYPES", "CATEGORICAL", "CONTINUOUS", "INTEGER", "LINEAR", "LOG",
    "ORDINAL", "Constraint", "Factor", "SearchSpace", "SpaceError",
    # encoding
    "AUTO", "DESCRIPTOR", "MIXED", "Encoder", "EncodingError",
    # tool 1
    "KMEDOIDS", "MAXMIN", "METHODS", "RANDOM", "SOBOL", "Design",
    "initial_design",
    # tool 1 with priors
    "CategoricalPrior", "CompositePrior", "GaussianPrior", "Prior",
    "PriorError", "UniformPrior", "pibo_exponent", "pibo_weights",
    "prior_initial_design", "rank_agreement", "rgpe_weight",
    # tool 2
    "DEEP", "PLAIN", "DeepKernelGP", "MixedGP", "PlainGP", "Surrogate",
    "SurrogateError", "build_surrogate",
    "ACQUISITIONS", "EI", "QEI", "QNEI", "TS", "AcquisitionError",
    "Recommendation", "next_recommendation",
    # state
    "Campaign", "CampaignError", "Observation", "Skipped",
    # adapters
    "lookup_oracle", "space_from_matrix",
]

__version__ = "0.1.0"
