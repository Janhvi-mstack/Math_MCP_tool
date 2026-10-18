"""AOSS-backed implementation of the BEACON campaign store.

Implements the same three methods as ``maths_tools.store.LocalJSONStore`` --
:meth:`AOSSStore.put`, :meth:`AOSSStore.get`, :meth:`AOSSStore.list` -- so it
drops in behind the eight tool functions without any of them changing.

Three indices, not one
----------------------
See :mod:`mappings` for the field definitions. The split is by document kind
rather than a single index with a ``doc_type`` discriminator, for three
reasons:

1. The three documents have genuinely different field types. One index means
   one mapping that is the union of all three, with every field optional, and
   the union is exactly where a type collision becomes possible -- the failure
   this whole module exists to prevent.
2. ``_id`` is unique per index, not per ``doc_type``. With one index a space
   and a campaign that happened to share a key would overwrite each other, so
   ids would need composite prefixes. With three, ``key`` maps straight onto
   ``_id`` and the existing ``(kind, key)`` interface carries over unchanged.
3. Cross-type queries are not lost: AOSS searches comma-separated indices in
   one request, so "everything for campaign X" stays a single call.

What the store writes beyond the record
---------------------------------------
``put("campaign", ...)`` writes the campaign document *and* fans its
observations and skips out into the observations index, one document each,
with a deterministic id. That projection is what makes "which experiments for
this reaction were skipped, and why" a query rather than a scan: ``outcome``
is only a meaningful field at the level of a single experiment.

The campaign document stays authoritative. It is read back whole by id on the
resume path, so resume never depends on the projection being visible yet --
which matters, because it often is not. See :meth:`get`.
"""
from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Dict, List, Optional

from ..store import CampaignStore, StoreError
from . import mappings
from .client import AOSSClient, AOSSNotFound
from .config import AOSSConfig

# StoreError is deliberately the local store's class, not a parallel one. The
# tool functions catch `maths_tools.store.StoreError` to turn a missing
# campaign into a readable message; a separate exception type here would sail
# past that handler and surface as a stack trace instead.


#: Seconds to keep retrying a get that misses a key this process did not
#: write. The measured fresh-process visibility lag was 3.4-7.5 s over five
#: trials, so this covers it with headroom rather than sitting on the edge.
DEFAULT_MISS_GRACE = 15.0


def canonical_point_key(point: Dict[str, Any]) -> str:
    """Return a stable string key for an experiment's conditions.

    Mirrors ``maths_tools.mcp_server._point_key``: keys sorted, floats rounded
    to 12 decimal places so that re-serialisation noise cannot produce two
    different keys for the same experiment.

    Parameters
    ----------
    point : dict
        Factor name to level.

    Returns
    -------
    str
        Canonical JSON rendering of the sorted, rounded pairs.

    Examples
    --------
    >>> canonical_point_key({"b": 2, "a": 1.0000000000001})
    '[["a", 1.0], ["b", 2]]'
    >>> canonical_point_key({"a": 1.0, "b": 2}) == canonical_point_key({"b": 2, "a": 1.0})
    True
    """
    pairs = sorted((k, round(v, 12) if isinstance(v, float) else v)
                   for k, v in point.items())
    return json.dumps([list(p) for p in pairs], sort_keys=True, default=str)


def _point_hash(point: Dict[str, Any]) -> str:
    """Return a short, URL-safe digest of an experiment's conditions."""
    return hashlib.sha1(
        canonical_point_key(point).encode("utf-8")).hexdigest()[:16]


def _to_millis(value: Any) -> Optional[int]:
    """Convert a float epoch-seconds timestamp to integer epoch milliseconds.

    The mapping declares these fields ``date`` with an ``epoch_millis``
    format. Writing float seconds into them would either be rejected or,
    worse, silently read as a date in 1970.

    Parameters
    ----------
    value : object
        Epoch seconds as int or float, or ``None``.

    Returns
    -------
    int or None
        Epoch milliseconds, or ``None`` when ``value`` is not a number.

    Examples
    --------
    >>> _to_millis(1760000000.42)
    1760000000420
    >>> _to_millis(None) is None
    True
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(round(float(value) * 1000))


def _safe_id(key: str) -> str:
    """Validate a document id.

    Ids go into the request path and are part of the SigV4 canonical request,
    so anything needing escaping is rejected outright rather than encoded.

    Raises
    ------
    StoreError
        If the key is empty or contains a character outside
        ``[A-Za-z0-9._-]``.

    Examples
    --------
    >>> _safe_id("camp_ab12")
    'camp_ab12'
    >>> _safe_id("a/b")
    Traceback (most recent call last):
    StoreError: unsafe document id: 'a/b'
    """
    if not key or not all(c.isalnum() or c in "._-" for c in key):
        raise StoreError(f"unsafe document id: {key!r}")
    return key


class AOSSStore(CampaignStore):
    """Campaign store backed by an OpenSearch Serverless collection.

    Parameters
    ----------
    cfg : AOSSConfig
        Connection settings.
    client : AOSSClient, optional
        Pre-built client. Constructed from ``cfg`` when omitted.
    read_after_write_timeout : float, optional
        Seconds to keep retrying a :meth:`get` that misses a key this process
        wrote recently (default 30.0). See the note on near-real-time lag.
    cache_writes : bool, optional
        Serve :meth:`get` from a write-through cache of confirmed writes
        (default True). See the note on near-real-time lag.
    miss_grace : float, optional
        Seconds to keep retrying a :meth:`get` that misses a key this process
        did *not* write (default :data:`DEFAULT_MISS_GRACE`). Covers the
        window in which a document written by another process is durable but
        not yet visible. Zero disables it.
    search_page_size : int, optional
        Page size for :meth:`list` (default 500).

    Notes
    -----
    **Near-real-time lag.** This was measured against the live collection
    rather than assumed, and the result is worse than the classic OpenSearch
    behaviour:

    * ``POST /{index}/_refresh`` does **not exist** on serverless (404), so
      visibility cannot be forced.
    * ``GET /{index}/_doc/{id}`` is **not realtime** either. On classic
      OpenSearch a get-by-id reads through the translog and sees a write
      immediately; on AOSS the observed lag was 2.5-6.1 s (median 5.1 s over
      five trials). A naive read-after-write therefore returns "not found"
      for a document that was durably written seconds earlier.

    Two things make the resume path correct in spite of that:

    1. A **write-through cache** of confirmed writes. ``put`` populates it
       only after AOSS has acknowledged the write, so the cache can never
       report a document that is not durable. A read of a key this process
       just wrote is served from it and is instant.
    2. A **bounded retry** for everything else -- notably a different worker,
       or the same campaign after a process restart, where there is no cache
       to serve from. ``get`` retries with backoff for up to
       ``read_after_write_timeout`` seconds, but only for a key this process
       wrote; an ordinary miss still fails fast.

    The campaign document remains the authoritative record and is always read
    whole by id, so resume never depends on the observations projection
    having become searchable.
    """

    #: Most recent confirmed writes retained for read-your-writes. Small: the
    #: cache exists to cover the seconds-long visibility gap, not to be a
    #: durable store.
    _CACHE_LIMIT = 256

    def __init__(self, cfg: AOSSConfig, client: Optional[AOSSClient] = None,
                 read_after_write_timeout: float = 30.0,
                 search_page_size: int = 500,
                 cache_writes: bool = True,
                 miss_grace: float = DEFAULT_MISS_GRACE) -> None:
        self.cfg = cfg
        self.client = client or AOSSClient(cfg)
        self.read_after_write_timeout = read_after_write_timeout
        self.miss_grace = miss_grace
        self.search_page_size = search_page_size
        self.cache_writes = cache_writes
        self._indices = mappings.all_index_names(cfg.index_prefix)
        #: (kind, key) -> monotonic time of the most recent local write.
        self._recent: Dict[tuple, float] = {}
        #: (kind, key) -> the record as confirmed written.
        self._cache: Dict[tuple, dict] = {}

    # -- indices ------------------------------------------------------------

    def index_for(self, kind: str) -> str:
        """Return the index name backing a document kind."""
        try:
            return self._indices[kind]
        except KeyError:
            raise StoreError(
                f"unknown kind {kind!r}; expected one of "
                f"{tuple(self._indices)}") from None

    def ensure_indices(self) -> Dict[str, str]:
        """Create every index with its explicit mapping, if absent.

        Idempotent, and must be called before the first write: AOSS creates an
        index implicitly on first write and infers the mapping from that first
        document, which is precisely the guess this module exists to avoid.

        Returns
        -------
        dict
            ``{kind: "created" | "exists"}``.
        """
        result = {}
        for kind, index in self._indices.items():
            if self.client.index_exists(index):
                result[kind] = "exists"
                continue
            self.client.create_index(index, mappings.mapping_for(kind))
            result[kind] = "created"
        return result

    # -- the three methods --------------------------------------------------

    def put(self, kind: str, key: str, obj: dict) -> None:
        """Write an object under a deterministic id.

        Uses ``PUT /{index}/_doc/{id}`` so a repeated write replaces the
        document rather than adding a second copy under a generated id.

        For ``kind="campaign"`` this also projects the campaign's observations
        and skips into the observations index.

        Parameters
        ----------
        kind : str
            ``"space"``, ``"campaign"`` or ``"observation"``.
        key : str
            Document id, e.g. a ``space_id`` or ``campaign_id``.
        obj : dict
            The object to store. Round-trips verbatim via ``payload``.
        """
        index = self.index_for(kind)
        doc_id = _safe_id(key)
        record = dict(obj)
        record["_updated"] = time.time()

        doc = self._project(kind, doc_id, record)
        self.client.put_doc(index, doc_id, doc)
        # Only after AOSS has acknowledged the write: the cache must never be
        # able to report a document that is not durable.
        self._recent[(kind, key)] = time.monotonic()
        if self.cache_writes:
            if len(self._cache) >= self._CACHE_LIMIT:
                self._cache.pop(next(iter(self._cache)), None)
            self._cache[(kind, key)] = json.loads(
                json.dumps(record, default=str))

        if kind == "campaign":
            self._fan_out_observations(doc_id, record)

    def get(self, kind: str, key: str) -> dict:
        """Read an object back by id.

        Fetched by document id, never by search. Three paths, in order:

        1. The **write-through cache**, when this process wrote the key. It
           is populated only after AOSS acknowledged the write, so a hit can
           never report a document that is not durable.
        2. A **read-after-write retry** of up to
           ``read_after_write_timeout`` seconds for a key this process wrote
           but is not holding in cache.
        3. A **miss grace** of up to ``miss_grace`` seconds for every other
           miss, because a key this process never wrote may still have been
           written seconds ago by another one. See Notes.

        Raises
        ------
        StoreError
            If the object does not exist, or is still invisible after the
            grace period.

        Notes
        -----
        The miss grace exists because of a measurement, not a theory. A
        document written by one process and read by a *fresh* one -- the
        resume case, and the multi-worker case -- missed on the first
        get-by-id in **4 of 5 trials**, becoming visible only 3.4-7.5 s after
        the write. The fresh process has no cache and no record that anything
        was written, so without this it would report "not found" for a
        campaign that exists.

        The cost is that a genuinely absent key takes ``miss_grace`` seconds
        to be reported absent. That is the right trade: a slow "no such
        campaign" is an annoyance, while a fast one for a campaign that does
        exist tells a chemist their data is gone. Set ``miss_grace=0`` to
        turn it off.
        """
        index = self.index_for(kind)
        doc_id = _safe_id(key)

        cached = self._cache.get((kind, key)) if self.cache_writes else None
        if cached is not None:
            return json.loads(json.dumps(cached, default=str))

        source = self.client.get_doc(index, doc_id)
        if source is None:
            written = self._recent.get((kind, key))
            if written is not None:
                window = self.read_after_write_timeout
                started = written
            else:
                window = self.miss_grace
                started = time.monotonic()
            deadline = started + window
            delay = 0.2
            while time.monotonic() < deadline and source is None:
                time.sleep(delay)
                delay = min(delay * 1.6, 2.0)
                source = self.client.get_doc(index, doc_id)
        if source is None:
            raise StoreError(f"{kind} {key!r} not found")
        return source.get("payload", {})

    def list(self, kind: str) -> List[str]:
        """Return every document id for a kind, sorted.

        There is no ``_cat`` on serverless, so this is a paginated
        ``_search`` using ``search_after`` over the ``doc_id`` keyword field.
        AOSS has no scroll API, which is why ``search_after`` is used rather
        than a scrolling cursor.

        Because this is a search, it is subject to near-real-time lag: a
        document written moments ago may not appear yet. :meth:`get` is the
        read-your-writes path; :meth:`list` is not.
        """
        index = self.index_for(kind)
        ids: List[str] = []
        after: Optional[list] = None
        while True:
            body = {
                "size": self.search_page_size,
                "query": {"match_all": {}},
                "sort": [{"doc_id": "asc"}],
                "_source": ["doc_id"],
            }
            if after is not None:
                body["search_after"] = after
            try:
                res = self.client.search(index, body)
            except AOSSNotFound:
                return []
            hits = res.get("hits", {}).get("hits", [])
            if not hits:
                break
            for hit in hits:
                value = (hit.get("_source") or {}).get("doc_id") or hit.get("_id")
                if value:
                    ids.append(value)
            if len(hits) < self.search_page_size:
                break
            after = hits[-1].get("sort")
            if not after:
                break
        return sorted(ids)

    def exists(self, kind: str, key: str) -> bool:
        """Return whether an object is present, right now.

        Deliberately does **not** wait out the miss grace that :meth:`get`
        applies. ``get`` waits because the caller wants the document and a
        wrong "missing" is expensive; ``exists`` is asking a question about
        this instant, and an answer that takes fifteen seconds to say "no" is
        not useful. A cache hit still counts as present.
        """
        if self.cache_writes and (kind, key) in self._cache:
            return True
        return self.client.get_doc(
            self.index_for(kind), _safe_id(key)) is not None

    def delete(self, kind: str, key: str) -> bool:
        """Delete an object by id. Returns False when already absent.

        Evicts the cache entry first, so a delete can never be masked by a
        stale read-your-writes hit.
        """
        self._cache.pop((kind, key), None)
        self._recent.pop((kind, key), None)
        return self.client.delete_doc(self.index_for(kind), _safe_id(key))

    # -- visibility ---------------------------------------------------------

    def wait_until_searchable(self, kind: str, key: str,
                              timeout: float = 60.0) -> Optional[float]:
        """Block until a document is visible to *search*, and report the lag.

        Used to measure near-real-time lag rather than assume a value for it.

        Parameters
        ----------
        kind, key : str
            The document to wait for.
        timeout : float, optional
            Seconds to wait before giving up (default 60).

        Returns
        -------
        float or None
            Observed lag in seconds, or ``None`` if it never became visible.
        """
        index = self.index_for(kind)
        started = time.monotonic()
        delay = 0.2
        while time.monotonic() - started < timeout:
            body = {"size": 1, "query": {"term": {"doc_id": key}},
                    "_source": ["doc_id"]}
            try:
                res = self.client.search(index, body)
            except AOSSNotFound:
                res = {}
            if res.get("hits", {}).get("hits"):
                return time.monotonic() - started
            time.sleep(delay)
            delay = min(delay * 1.5, 2.0)
        return None

    # -- projection ---------------------------------------------------------

    def _project(self, kind: str, doc_id: str, record: dict) -> dict:
        """Build the indexed document for a record.

        The authoritative object goes into ``payload`` untouched; everything
        above it is the extracted query surface.
        """
        builder = {"space": self._project_space,
                   "campaign": self._project_campaign,
                   "observation": self._project_observation}[kind]
        doc = builder(doc_id, record)
        doc["doc_id"] = doc_id
        doc["schema_version"] = mappings.SCHEMA_VERSION
        doc["updated_at"] = _to_millis(record.get("_updated")) or _to_millis(
            time.time())
        doc["payload"] = record
        return {k: v for k, v in doc.items() if v is not None}

    @staticmethod
    def _meta(record: dict) -> Dict[str, Optional[str]]:
        """Pull optional ``reaction_id`` / ``user_id`` from a record.

        Neither is set by the eight tool functions today. The fields are
        declared and populated-if-present so that the query surface is ready
        the moment the tool starts supplying them, without a reindex.
        """
        meta = record.get("metadata") or {}
        return {
            "reaction_id": record.get("reaction_id") or meta.get("reaction_id"),
            "user_id": record.get("user_id") or meta.get("user_id"),
        }

    def _project_space(self, doc_id: str, record: dict) -> dict:
        space = record.get("space") or {}
        factors = space.get("factors") or []
        projected = []
        for f in factors:
            if not isinstance(f, dict):
                continue
            levels = f.get("levels")
            projected.append({
                "name": f.get("name"),
                "kind": f.get("kind") or f.get("type"),
                "unit": f.get("unit"),
                "n_levels": len(levels) if isinstance(levels, list) else None,
                "low": f.get("low"),
                "high": f.get("high"),
                "has_descriptors": bool(f.get("descriptors")),
            })
        norm = record.get("normalisation") or {}
        doc = {
            "space_id": record.get("space_id") or doc_id,
            "name": space.get("name") or record.get("name"),
            "description": space.get("description") or record.get("description"),
            "n_factors": len(factors) or None,
            "cardinality": record.get("cardinality") or space.get("cardinality"),
            "is_enumerable": record.get("is_enumerable"),
            "factor_names": [f.get("name") for f in projected if f.get("name")]
                            or None,
            "factors": [{k: v for k, v in f.items() if v is not None}
                        for f in projected] or None,
            "needs_review": norm.get("needs_review"),
            "created_at": _to_millis(record.get("created")),
        }
        doc.update(self._meta(record))
        return doc

    def _project_campaign(self, doc_id: str, record: dict) -> dict:
        observations = record.get("observations") or []
        skipped = record.get("skipped") or []
        pending = record.get("pending") or []
        deferred = record.get("deferral_history") or []
        ys = [o.get("y") for o in observations
              if isinstance(o.get("y"), (int, float))]
        cfg = record.get("config") or {}

        suggested = []
        for point in pending:
            if isinstance(point, dict):
                suggested.append({"point_key": _point_hash(point),
                                  "state": "pending"})
        for entry in skipped:
            point = entry.get("point") if isinstance(entry, dict) else None
            if isinstance(point, dict):
                suggested.append({"point_key": _point_hash(point),
                                  "state": "skipped",
                                  "suggested_at": _to_millis(entry.get("ts"))})
        for obs in observations:
            point = obs.get("point") if isinstance(obs, dict) else None
            if isinstance(point, dict):
                suggested.append({
                    "point_key": _point_hash(point), "state": "observed",
                    "suggested_at": _to_millis(obs.get("recorded"))})

        doc = {
            "campaign_id": record.get("campaign_id") or doc_id,
            "space_id": record.get("space_id"),
            "status": "complete" if record.get("complete") else "active",
            "n_observations": len(observations),
            "n_skipped": len(skipped),
            "n_pending": len(pending),
            "n_deferred": len(deferred),
            "best_y": max(ys) if ys else None,
            "config": {k: v for k, v in {
                "init_method": cfg.get("init_method"),
                "acquisition": cfg.get("acquisition"),
                "surrogate": cfg.get("surrogate"),
                "seed": cfg.get("seed"),
            }.items() if v is not None} or None,
            "suggested": [{k: v for k, v in s.items() if v is not None}
                          for s in suggested] or None,
            "created_at": _to_millis(record.get("created")),
        }
        doc.update(self._meta(record))
        return doc

    def _project_observation(self, doc_id: str, record: dict) -> dict:
        point = record.get("point") or {}
        doc = {
            "campaign_id": record.get("campaign_id"),
            "space_id": record.get("space_id"),
            "outcome": record.get("outcome"),
            "point_key": _point_hash(point) if point else None,
            "experiment_index": record.get("experiment_index"),
            "y": record.get("y"),
            "skip_reason": record.get("skip_reason") or record.get("reason"),
            "was_deferred": record.get("was_deferred"),
            "recorded_at": _to_millis(record.get("recorded")
                                      or record.get("ts")),
        }
        doc.update(self._meta(record))
        return doc

    # -- fan-out ------------------------------------------------------------

    def _fan_out_observations(self, campaign_id: str, record: dict) -> None:
        """Project a campaign's observations and skips into their own index.

        One document per experiment, with a deterministic id derived from the
        campaign and the conditions, so re-writing a campaign updates its
        projected documents in place instead of duplicating them.

        A skip is written with ``y`` absent, never zero: an experiment that
        was not run has no value, and recording one would be a fabricated
        measurement.
        """
        index = self.index_for("observation")
        meta = self._meta(record)
        space_id = record.get("space_id")
        deferred_keys = {
            _point_hash(d["point"]) for d in (record.get("deferral_history") or [])
            if isinstance(d, dict) and isinstance(d.get("point"), dict)}

        docs = []
        for i, obs in enumerate(record.get("observations") or []):
            point = obs.get("point") or {}
            if not point:
                continue
            key = _point_hash(point)
            docs.append((f"{campaign_id}.{key}", {
                "campaign_id": campaign_id, "space_id": space_id,
                "point": point, "y": obs.get("y"),
                "experiment_index": i,
                "outcome": ("deferred_then_observed" if key in deferred_keys
                            else "observed"),
                "was_deferred": key in deferred_keys,
                "recorded": obs.get("recorded"), **meta}))

        for entry in record.get("skipped") or []:
            point = entry.get("point") or {}
            if not point:
                continue
            key = _point_hash(point)
            docs.append((f"{campaign_id}.{key}", {
                "campaign_id": campaign_id, "space_id": space_id,
                "point": point, "outcome": "skipped",
                "skip_reason": entry.get("reason"),
                "was_deferred": False,
                "recorded": entry.get("ts"), **meta}))

        if not docs:
            return

        lines = []
        for doc_id, body in docs:
            projected = self._project("observation", doc_id, body)
            lines.append(json.dumps(
                {"index": {"_index": index, "_id": doc_id}}).encode("utf-8"))
            lines.append(json.dumps(projected, default=str).encode("utf-8"))
        payload = b"\n".join(lines) + b"\n"

        res = self.client.bulk(payload)
        if res.get("errors"):
            failed = [item for item in res.get("items", [])
                      if (item.get("index") or {}).get("error")]
            first = (failed[0].get("index", {}).get("error")
                     if failed else "unknown")
            raise StoreError(
                f"observation fan-out failed for {len(failed)}/{len(docs)} "
                f"documents; first error: {first}")
