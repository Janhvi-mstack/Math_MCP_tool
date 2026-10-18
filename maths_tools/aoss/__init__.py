"""AWS OpenSearch Serverless backend for the campaign store.

:class:`~maths_tools.aoss.store.AOSSStore` implements the same three methods as
:class:`maths_tools.store.LocalJSONStore` -- ``put``, ``get``, ``list`` -- so it
drops in behind the eight tool functions without any of them changing.

This is **AOSS, not Elasticsearch and not a managed OpenSearch domain**, and
three differences bite:

* every request is SigV4-signed with service name ``aoss``; signing as ``es``
  returns a bare 403 with no explanation
* there are no ``_cluster/*`` admin APIs, and ``_refresh`` does not exist, so
  index visibility cannot be forced
* IAM permissions alone are not enough -- the collection needs a data access
  policy granting the calling principal ``aoss:APIAccessAll``

Nothing here depends on boto3 or requests: signing is implemented against the
standard library in :mod:`maths_tools.aoss.signing`, so the tool adds no
dependency by gaining a backend.

See the project README for the index names, the mappings and the measured
near-real-time behaviour.
"""
from __future__ import annotations

from .client import (
    AOSSAccessDenied, AOSSClient, AOSSError, AOSSNotFound, AOSSRequestError,
)
from .config import AOSSConfig, ConfigError, available, load
from .mappings import (
    SCHEMA_VERSION, all_index_names, index_name, mapping_for,
)
from .store import AOSSStore, canonical_point_key

__all__ = [
    "AOSSAccessDenied", "AOSSClient", "AOSSError", "AOSSNotFound",
    "AOSSRequestError",
    "AOSSConfig", "ConfigError", "available", "load",
    "SCHEMA_VERSION", "all_index_names", "index_name", "mapping_for",
    "AOSSStore", "canonical_point_key",
]
