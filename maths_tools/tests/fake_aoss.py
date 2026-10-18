"""An in-memory stand-in for an AOSS collection, used by the offline tests.

It is deliberately strict rather than permissive, because a lenient fake would
let exactly the bugs this project cares about pass offline and fail in the
cloud. Specifically it enforces:

* ``dynamic: "strict"`` -- a field the mapping does not declare is rejected
  with the same ``strict_dynamic_mapping_exception`` the real service returns,
  so a projection that invents a field fails in a unit test.
* Declared field types -- a float written into a ``date`` field is rejected,
  which is what catches a timestamp that was not converted to epoch
  milliseconds.
* Near-real-time lag -- a written document is retrievable by id immediately
  but invisible to ``_search`` until ``searchable_after`` seconds have
  elapsed, so the read-your-writes path is exercised rather than assumed.

It also refuses ``_cat`` and ``_cluster``, which do not exist on serverless.
"""
from __future__ import annotations

import json
import time
from typing import Dict, Mapping, Optional
from urllib.parse import unquote, urlsplit


class FakeAOSS:
    """A minimal, strict, in-memory OpenSearch Serverless substitute.

    Parameters
    ----------
    searchable_after : float, optional
        Seconds a document takes to become visible to search (default 0.0).
    get_by_id_lag : float, optional
        Seconds a document takes to become retrievable *by id* (default 0.0).
        Serverless makes no documented realtime-GET guarantee, so the tests
        set this to prove the retry path works if the guarantee does not hold.
    deny_all : bool, optional
        When True every request returns 403, simulating a missing data access
        policy (default False).
    """

    def __init__(self, searchable_after: float = 0.0,
                 get_by_id_lag: float = 0.0,
                 deny_all: bool = False) -> None:
        self.searchable_after = searchable_after
        self.get_by_id_lag = get_by_id_lag
        self.deny_all = deny_all
        #: index -> mapping body
        self.indices: Dict[str, dict] = {}
        #: index -> {doc_id: (source, written_at)}
        self.docs: Dict[str, Dict[str, tuple]] = {}
        #: every (method, path) seen, for assertions about API usage
        self.calls: list = []

    # -- transport ----------------------------------------------------------

    def transport(self, method: str, url: str, headers: Mapping[str, str],
                  body: bytes, timeout: float) -> tuple:
        """Transport callable matching :data:`client.Transport`."""
        path = unquote(urlsplit(url).path)
        query = urlsplit(url).query
        self.calls.append((method, path))

        if self.deny_all:
            return 403, json.dumps({"error": {
                "type": "security_exception",
                "reason": "no permissions for [indices:data/write/index]"}})

        if "/_cat" in path or "/_cluster" in path:
            return 404, json.dumps({"error": {
                "type": "index_not_found_exception",
                "reason": "no such index [_cat]"}})

        if not str(headers.get("Authorization", "")).startswith(
                "AWS4-HMAC-SHA256 "):
            return 403, json.dumps({"error": "unsigned request"})
        if "/aoss/aws4_request" not in str(headers.get("Authorization", "")):
            return 403, json.dumps({"error": {
                "type": "security_exception",
                "reason": "credential scope service must be aoss"}})

        parts = [p for p in path.split("/") if p]
        try:
            return self._route(method, parts, body, query)
        except _Rejected as exc:
            return exc.status, json.dumps(exc.payload)

    # -- routing ------------------------------------------------------------

    def _route(self, method: str, parts: list, body: bytes,
               query: str) -> tuple:
        if parts and parts[0] == "_bulk":
            return self._bulk(body)

        if not parts:
            return 200, json.dumps({"version": {"number": "2.x"}})

        index = parts[0]

        if len(parts) == 1:
            if method == "HEAD":
                return (200, "") if index in self.indices else (404, "")
            if method == "PUT":
                if index in self.indices:
                    return 400, json.dumps({"error": {
                        "type": "resource_already_exists_exception",
                        "reason": f"index [{index}] already exists"}})
                self.indices[index] = json.loads(body.decode())
                self.docs.setdefault(index, {})
                return 200, json.dumps({"acknowledged": True})
            if method == "DELETE":
                self.indices.pop(index, None)
                self.docs.pop(index, None)
                return 200, json.dumps({"acknowledged": True})

        if len(parts) == 2 and parts[1] == "_mapping":
            self._require_index(index)
            return 200, json.dumps({index: self.indices[index]})

        if len(parts) == 2 and parts[1] == "_search":
            self._require_index(index)
            return self._search(index, json.loads(body.decode() or "{}"))

        if len(parts) == 3 and parts[1] == "_doc":
            doc_id = parts[2]
            if method == "PUT":
                self._require_index(index)
                source = json.loads(body.decode())
                self._validate(index, source)
                self.docs[index][doc_id] = (source, time.monotonic())
                return 200, json.dumps({"_id": doc_id, "result": "created"})
            if method == "GET":
                self._require_index(index)
                entry = self.docs[index].get(doc_id)
                if entry is None:
                    return 404, json.dumps({"_id": doc_id, "found": False})
                source, written = entry
                if time.monotonic() - written < self.get_by_id_lag:
                    return 404, json.dumps({"_id": doc_id, "found": False})
                return 200, json.dumps(
                    {"_id": doc_id, "found": True, "_source": source})
            if method == "DELETE":
                self._require_index(index)
                if self.docs[index].pop(doc_id, None) is None:
                    return 404, json.dumps({"result": "not_found"})
                return 200, json.dumps({"result": "deleted"})

        return 400, json.dumps({"error": f"unhandled {method} /{'/'.join(parts)}"})

    def _require_index(self, index: str) -> None:
        if index not in self.indices:
            raise _Rejected(404, {"error": {
                "type": "index_not_found_exception",
                "reason": f"no such index [{index}]"}})

    # -- mapping enforcement -------------------------------------------------

    def _validate(self, index: str, source: dict) -> None:
        """Enforce strict dynamic mapping and declared field types."""
        mapping = self.indices[index].get("mappings", {})
        props = mapping.get("properties", {})
        strict = mapping.get("dynamic") == "strict"
        self._check_object(source, props, strict, path="")

    def _check_object(self, obj: dict, props: dict, strict: bool,
                      path: str) -> None:
        for name, value in obj.items():
            spec = props.get(name)
            where = f"{path}{name}"
            if spec is None:
                if strict:
                    raise _Rejected(400, {"error": {
                        "type": "strict_dynamic_mapping_exception",
                        "reason": f"mapping set to strict, dynamic introduction "
                                  f"of [{where}] within [_doc] is not allowed"}})
                continue
            if spec.get("enabled") is False:
                continue                      # payload: stored, never parsed
            kind = spec.get("type")
            if kind == "nested" or (kind is None and "properties" in spec):
                sub = spec.get("properties", {})
                rows = value if isinstance(value, list) else [value]
                for row in rows:
                    if isinstance(row, dict):
                        self._check_object(row, sub, strict, f"{where}.")
                continue
            self._check_scalar(where, value, kind)

    @staticmethod
    def _check_scalar(where: str, value, kind: Optional[str]) -> None:
        values = value if isinstance(value, list) else [value]
        for item in values:
            if item is None:
                continue
            if kind == "date":
                # epoch_millis accepts an integer; a float epoch-seconds value
                # is the mistake this check exists to catch.
                if isinstance(item, bool) or not isinstance(item, (int, str)):
                    raise _Rejected(400, {"error": {
                        "type": "mapper_parsing_exception",
                        "reason": f"failed to parse field [{where}] of type "
                                  f"[date]: got {type(item).__name__}"}})
            elif kind in ("integer", "long"):
                if isinstance(item, bool) or not isinstance(item, int):
                    raise _Rejected(400, {"error": {
                        "type": "mapper_parsing_exception",
                        "reason": f"failed to parse field [{where}] of type "
                                  f"[{kind}]: got {type(item).__name__}"}})
            elif kind == "double":
                if isinstance(item, bool) or not isinstance(item, (int, float)):
                    raise _Rejected(400, {"error": {
                        "type": "mapper_parsing_exception",
                        "reason": f"failed to parse field [{where}] of type "
                                  f"[double]: got {type(item).__name__}"}})
            elif kind == "boolean":
                if not isinstance(item, bool):
                    raise _Rejected(400, {"error": {
                        "type": "mapper_parsing_exception",
                        "reason": f"failed to parse field [{where}] of type "
                                  f"[boolean]"}})
            elif kind in ("keyword", "text"):
                if not isinstance(item, str):
                    raise _Rejected(400, {"error": {
                        "type": "mapper_parsing_exception",
                        "reason": f"failed to parse field [{where}] of type "
                                  f"[{kind}]: got {type(item).__name__}"}})

    # -- search --------------------------------------------------------------

    def _search(self, index: str, body: dict) -> tuple:
        now = time.monotonic()
        rows = [(doc_id, src) for doc_id, (src, written) in
                self.docs[index].items()
                if now - written >= self.searchable_after]

        query = body.get("query") or {"match_all": {}}
        if "term" in query:
            field, want = next(iter(query["term"].items()))
            rows = [r for r in rows if r[1].get(field) == want]
        elif "match" in query:
            field, want = next(iter(query["match"].items()))
            rows = [r for r in rows
                    if str(want).lower() in str(r[1].get(field, "")).lower()]

        sort = body.get("sort") or []
        if sort:
            field = next(iter(sort[0]))
            rows.sort(key=lambda r: str(r[1].get(field) or r[0]))

        after = body.get("search_after")
        if after:
            rows = [r for r in rows if str(r[1].get("doc_id") or r[0]) > str(after[0])]

        size = int(body.get("size", 10))
        page = rows[:size]
        hits = [{"_id": doc_id, "_source": src,
                 "sort": [str(src.get("doc_id") or doc_id)]}
                for doc_id, src in page]
        return 200, json.dumps({
            "hits": {"total": {"value": len(rows)}, "hits": hits}})

    # -- bulk ----------------------------------------------------------------

    def _bulk(self, body: bytes) -> tuple:
        lines = [l for l in body.decode().split("\n") if l.strip()]
        items, errors = [], False
        pending = None
        for line in lines:
            doc = json.loads(line)
            if pending is None and "index" in doc and isinstance(
                    doc.get("index"), dict) and "_index" in doc["index"]:
                pending = doc["index"]
                continue
            if pending is not None:
                index, doc_id = pending["_index"], pending["_id"]
                pending = None
                try:
                    self._require_index(index)
                    self._validate(index, doc)
                except _Rejected as exc:
                    errors = True
                    items.append({"index": {"_id": doc_id, "status": exc.status,
                                            "error": exc.payload["error"]}})
                    continue
                self.docs[index][doc_id] = (doc, time.monotonic())
                items.append({"index": {"_id": doc_id, "status": 200,
                                        "result": "created"}})
        return 200, json.dumps({"errors": errors, "items": items})


class _Rejected(Exception):
    """Internal: a routed request that must return a non-2xx response."""

    def __init__(self, status: int, payload: dict) -> None:
        super().__init__(str(payload))
        self.status = status
        self.payload = payload
