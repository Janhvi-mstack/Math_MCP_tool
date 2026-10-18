"""Explicit index mappings for the three BEACON document types.

Why these are written down
--------------------------
AOSS dynamic-maps a field on first write and guesses its type from the first
value it sees. Two of those guesses would silently corrupt this data:

* A float epoch timestamp such as ``1760000000.42`` is guessed as ``float``,
  not ``date``, so no range query over it ever works. Every timestamp here is
  therefore declared ``date`` and written as integer epoch milliseconds.
* A yield of ``0`` in the first document mapped is guessed as ``long``, after
  which ``41.7`` is coerced to ``41``. Yields are declared ``double``.

Every index is ``dynamic: "strict"``. A field this module does not declare is
*rejected* rather than guessed, which turns a silent schema drift into a loud
write failure -- the preferred failure mode for this project.

The arbitrary-content problem
-----------------------------
A search space carries user-chosen factor names and descriptor dictionaries,
so its JSON has no fixed shape. Indexing that would cause a mapping explosion
and, eventually, a type collision between two campaigns that happened to use
the same factor name for different kinds of value. Each index therefore has a
``payload`` object declared ``enabled: false``: it round-trips verbatim in
``_source``, is never parsed, and can never collide. The indexed fields above
it are a small, controlled projection extracted by the store -- the query
surface, not the record.

The query surface
-----------------
``reaction_id``, ``user_id``, ``campaign_id`` and ``outcome`` are keyword
fields on every index where they are meaningful, so an agent can filter by
reaction, by user, by campaign and by outcome without touching ``payload``.
"""
from __future__ import annotations

from typing import Dict

#: Accepts integer epoch milliseconds and ISO-8601, in that order.
DATE_FORMAT = "epoch_millis||strict_date_optional_time"

_KEYWORD = {"type": "keyword"}
_DATE = {"type": "date", "format": DATE_FORMAT}
#: Verbatim, never-indexed home for the authoritative document.
_PAYLOAD = {"type": "object", "enabled": False}

#: Logical document type -> unsuffixed index name.
KINDS = ("space", "campaign", "observation")

_BASE_NAMES = {
    "space": "search-spaces",
    "campaign": "campaigns",
    "observation": "observations",
}

#: Bumped when a mapping changes incompatibly. AOSS cannot re-map a live
#: index in place, so a breaking change means a new index and a reindex.
SCHEMA_VERSION = "v1"


def index_name(kind: str, prefix: str = "beacon") -> str:
    """Return the full index name for a logical document kind.

    Parameters
    ----------
    kind : str
        One of :data:`KINDS`.
    prefix : str, optional
        Index-name prefix, from ``BEACON_AOSS_INDEX_PREFIX``.

    Returns
    -------
    str
        e.g. ``beacon-campaigns-v1``.

    Raises
    ------
    KeyError
        If ``kind`` is not a known document kind.

    Examples
    --------
    >>> index_name("campaign")
    'beacon-campaigns-v1'
    >>> index_name("observation", prefix="beacon-stg")
    'beacon-stg-observations-v1'
    """
    if kind not in _BASE_NAMES:
        raise KeyError(
            f"unknown document kind {kind!r}; expected one of {KINDS}")
    return f"{prefix}-{_BASE_NAMES[kind]}-{SCHEMA_VERSION}"


def all_index_names(prefix: str = "beacon") -> Dict[str, str]:
    """Return ``{kind: index_name}`` for every kind.

    Examples
    --------
    >>> sorted(all_index_names().values())
    ['beacon-campaigns-v1', 'beacon-observations-v1', 'beacon-search-spaces-v1']
    """
    return {k: index_name(k, prefix) for k in KINDS}


SPACE_MAPPING = {
    "mappings": {
        "dynamic": "strict",
        "properties": {
            "doc_id": _KEYWORD,
            "space_id": _KEYWORD,
            "schema_version": _KEYWORD,
            # Query surface -------------------------------------------------
            "reaction_id": _KEYWORD,
            "user_id": _KEYWORD,
            "name": {"type": "text",
                     "fields": {"raw": {"type": "keyword", "ignore_above": 256}}},
            "description": {"type": "text"},
            # Shape ---------------------------------------------------------
            "n_factors": {"type": "integer"},
            "cardinality": {"type": "long"},
            "is_enumerable": {"type": "boolean"},
            "factor_names": _KEYWORD,
            "factors": {
                "type": "nested",
                "properties": {
                    "name": _KEYWORD,
                    "kind": _KEYWORD,
                    "unit": _KEYWORD,
                    "n_levels": {"type": "integer"},
                    "low": {"type": "double"},
                    "high": {"type": "double"},
                    "has_descriptors": {"type": "boolean"},
                },
            },
            "needs_review": {"type": "boolean"},
            "created_at": _DATE,
            "updated_at": _DATE,
            "payload": _PAYLOAD,
        },
    }
}

CAMPAIGN_MAPPING = {
    "mappings": {
        "dynamic": "strict",
        "properties": {
            "doc_id": _KEYWORD,
            "campaign_id": _KEYWORD,
            "space_id": _KEYWORD,
            "schema_version": _KEYWORD,
            # Query surface -------------------------------------------------
            "reaction_id": _KEYWORD,
            "user_id": _KEYWORD,
            "status": _KEYWORD,
            # Progress ------------------------------------------------------
            "n_observations": {"type": "integer"},
            "n_skipped": {"type": "integer"},
            "n_pending": {"type": "integer"},
            "n_deferred": {"type": "integer"},
            "best_y": {"type": "double"},
            "config": {
                "properties": {
                    "init_method": _KEYWORD,
                    "acquisition": _KEYWORD,
                    "surrogate": _KEYWORD,
                    "seed": {"type": "integer"},
                },
            },
            # The experiments this campaign has suggested, and what became of
            # each. Nested so a query can ask for a campaign holding a
            # suggestion that is still outstanding.
            "suggested": {
                "type": "nested",
                "properties": {
                    "point_key": _KEYWORD,
                    "state": _KEYWORD,
                    "suggested_at": _DATE,
                },
            },
            "created_at": _DATE,
            "updated_at": _DATE,
            "payload": _PAYLOAD,
        },
    }
}

OBSERVATION_MAPPING = {
    "mappings": {
        "dynamic": "strict",
        "properties": {
            "doc_id": _KEYWORD,
            "campaign_id": _KEYWORD,
            "space_id": _KEYWORD,
            "schema_version": _KEYWORD,
            # Query surface -------------------------------------------------
            "reaction_id": _KEYWORD,
            "user_id": _KEYWORD,
            # observed | skipped | deferred_then_observed
            "outcome": _KEYWORD,
            "point_key": _KEYWORD,
            "experiment_index": {"type": "integer"},
            # Null for a skip: an experiment that was not run has no value,
            # and must never be stored as a zero.
            "y": {"type": "double"},
            "skip_reason": {"type": "text",
                            "fields": {"raw": {"type": "keyword",
                                               "ignore_above": 256}}},
            "was_deferred": {"type": "boolean"},
            "recorded_at": _DATE,
            "updated_at": _DATE,
            "payload": _PAYLOAD,
        },
    }
}

MAPPINGS = {
    "space": SPACE_MAPPING,
    "campaign": CAMPAIGN_MAPPING,
    "observation": OBSERVATION_MAPPING,
}


def mapping_for(kind: str) -> dict:
    """Return the index-creation body for a document kind.

    Only ``mappings`` is included. OpenSearch Serverless manages shards and
    replicas itself and rejects a creation body that tries to set them.

    Parameters
    ----------
    kind : str
        One of :data:`KINDS`.

    Returns
    -------
    dict
        Body for ``PUT /{index}``.

    Examples
    --------
    >>> mapping_for("observation")["mappings"]["properties"]["y"]["type"]
    'double'
    >>> mapping_for("campaign")["mappings"]["dynamic"]
    'strict'
    """
    if kind not in MAPPINGS:
        raise KeyError(
            f"unknown document kind {kind!r}; expected one of {KINDS}")
    return MAPPINGS[kind]
