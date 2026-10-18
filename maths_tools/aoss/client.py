"""Minimal signed HTTP client for an OpenSearch Serverless collection.

Standard library only -- no AWS SDK, no ``opensearch-py``. The surface is the
handful of REST calls this backend needs, each SigV4-signed with the ``aoss``
service name.

What serverless does not have
-----------------------------
``_cat/*`` and ``_cluster/*`` do not exist on AOSS. Anything that would
normally be answered by ``_cat/indices`` is answered here by a search, and
listing documents is a paginated ``_search`` with ``search_after`` -- there is
no scroll API either.

Errors
------
A ``403`` is ambiguous at the HTTP layer: it is returned both for a
mis-signed request and for a correctly-signed request from a principal with no
data access policy. :class:`AOSSAccessDenied` carries the response body so the
caller can tell the two apart, because the remedy is different and only one of
them can be fixed in code.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Mapping, Optional

from .config import SERVICE_NAME, AOSSConfig
from .signing import sign_request


class AOSSError(RuntimeError):
    """Base class for every error raised by this client."""


class AOSSAccessDenied(AOSSError):
    """A ``403`` from the collection endpoint.

    Either the request was signed with the wrong service name, or the signing
    principal is absent from the collection's data access policy. The latter
    cannot be fixed in code -- it needs an ``aoss:APIAccessAll`` grant.
    """

    def __init__(self, message: str, body: str = "") -> None:
        super().__init__(message)
        self.body = body


class AOSSNotFound(AOSSError):
    """A ``404`` -- missing document or missing index."""


class AOSSConflict(AOSSError):
    """A ``409`` -- optimistic-concurrency conflict on a versioned write."""


class AOSSRequestError(AOSSError):
    """Any other non-2xx response. Carries ``status`` and ``body``."""

    def __init__(self, message: str, status: int = 0, body: str = "") -> None:
        super().__init__(message)
        self.status = status
        self.body = body


#: A transport takes (method, url, headers, body, timeout) and returns
#: (status, body_text). Injectable so tests exercise the client offline.
Transport = Callable[[str, str, Mapping[str, str], bytes, float], tuple]


def _urllib_transport(method: str, url: str, headers: Mapping[str, str],
                      body: bytes, timeout: float) -> tuple:
    """Default transport, backed by :mod:`urllib.request`."""
    req = urllib.request.Request(url, data=body or None, method=method)
    for name, value in headers.items():
        req.add_header(name, value)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", "replace")


class AOSSClient:
    """Signed REST client for one AOSS collection.

    Parameters
    ----------
    cfg : AOSSConfig
        Resolved connection settings.
    timeout : float, optional
        Per-request timeout in seconds (default 30).
    max_retries : int, optional
        Retries for throttling, transient 5xx responses and network-level
        failures such as a read timeout (default 3). A ``403`` is never
        retried -- it is a policy problem, not a blip.
    transport : callable, optional
        Override the HTTP transport. Used by tests.

    Examples
    --------
    >>> from maths_tools.aoss.config import AOSSConfig
    >>> calls = []
    >>> def fake(method, url, headers, body, timeout):
    ...     calls.append((method, url))
    ...     return 200, '{"ok": true}'
    >>> cfg = AOSSConfig("https://h", "h", "us-east-1", "AK", "sk", None)
    >>> AOSSClient(cfg, transport=fake).request("GET", "/x")
    {'ok': True}
    >>> calls
    [('GET', 'https://h/x')]
    """

    def __init__(self, cfg: AOSSConfig, timeout: float = 30.0,
                 max_retries: int = 3,
                 transport: Optional[Transport] = None) -> None:
        self.cfg = cfg
        self.timeout = timeout
        self.max_retries = max_retries
        self._transport = transport or _urllib_transport

    # -- core ---------------------------------------------------------------

    def request(self, method: str, path: str, *,
                body: Any = None,
                query: Optional[Mapping[str, object]] = None,
                ok_status: tuple = (200, 201)) -> Any:
        """Sign and send one request, returning the decoded JSON body.

        Parameters
        ----------
        method : str
            HTTP method.
        path : str
            Path beginning with ``/``.
        body : object, optional
            JSON-serialisable body, or ``bytes`` to send verbatim (used for
            the newline-delimited ``_bulk`` payload).
        query : mapping, optional
            Query parameters.
        ok_status : tuple of int, optional
            Statuses treated as success.

        Returns
        -------
        object
            Decoded JSON, or ``{"raw": text}`` when the body is not JSON.

        Raises
        ------
        AOSSAccessDenied
            On ``403``.
        AOSSNotFound
            On ``404``.
        AOSSConflict
            On ``409``.
        AOSSRequestError
            On any other non-success status, or when retries are exhausted.
        """
        if isinstance(body, bytes):
            payload = body
            content_type = "application/x-ndjson"
        elif body is None:
            payload = b""
            content_type = None
        else:
            payload = json.dumps(body, default=str).encode("utf-8")
            content_type = "application/json"

        headers = {"content-type": content_type} if content_type else {}
        url = self.cfg.endpoint + path
        if query:
            from .signing import canonical_query
            url = f"{url}?{canonical_query(query)}"

        last_error = None
        for attempt in range(self.max_retries + 1):
            # Network-level failures (read timeout, connection reset, DNS
            # blip) arrive as exceptions rather than status codes, so without
            # this they would bypass the retry loop entirely and fail a
            # campaign on a single transient stall. Treated like a 5xx.
            signed = sign_request(
                method, self.cfg.host, path,
                region=self.cfg.region, service=SERVICE_NAME,
                access_key=self.cfg.access_key,
                secret_key=self.cfg.secret_key,
                session_token=self.cfg.session_token,
                query=query, headers=headers, body=payload)

            try:
                status, text = self._transport(
                    method, url, signed, payload, self.timeout)
            except OSError as exc:        # socket.timeout, URLError, reset
                last_error = AOSSRequestError(
                    f"network error on {method} {path}: "
                    f"{type(exc).__name__}: {exc}", status=0)
                if attempt < self.max_retries:
                    time.sleep(min(2.0 ** attempt * 0.5, 8.0))
                    continue
                raise last_error from exc

            if status in ok_status:
                if not text:
                    return {}
                try:
                    return json.loads(text)
                except ValueError:
                    return {"raw": text}

            if status == 403:
                raise AOSSAccessDenied(
                    "403 from AOSS. The request was signed with service name "
                    f"{SERVICE_NAME!r}. If the signature is correct this means "
                    "the signing principal is not granted aoss:APIAccessAll by "
                    "the collection's data access policy.", body=text)
            if status == 404:
                raise AOSSNotFound(f"404 for {method} {path}: {text[:400]}")
            if status == 409:
                raise AOSSConflict(f"409 for {method} {path}: {text[:400]}")

            # 429 and 5xx are worth another go; anything else is not.
            if status == 429 or 500 <= status < 600:
                last_error = AOSSRequestError(
                    f"{status} for {method} {path}: {text[:400]}",
                    status=status, body=text)
                if attempt < self.max_retries:
                    time.sleep(min(2.0 ** attempt * 0.5, 8.0))
                    continue
                raise last_error

            raise AOSSRequestError(
                f"{status} for {method} {path}: {text[:400]}",
                status=status, body=text)

        raise last_error or AOSSRequestError("request failed")  # pragma: no cover

    # -- index management ---------------------------------------------------

    def index_exists(self, index: str) -> bool:
        """Return whether ``index`` exists.

        Uses ``HEAD /{index}``; ``_cat/indices`` does not exist on serverless.
        """
        try:
            self.request("HEAD", f"/{index}", ok_status=(200,))
            return True
        except AOSSNotFound:
            return False

    def create_index(self, index: str, body: dict) -> dict:
        """Create ``index`` with an explicit mapping.

        Returns ``{"acknowledged": True, "already_existed": True}`` when the
        index is already present, so provisioning is idempotent.
        """
        try:
            return self.request("PUT", f"/{index}", body=body)
        except AOSSRequestError as exc:
            if "resource_already_exists" in (exc.body or ""):
                return {"acknowledged": True, "already_existed": True}
            raise

    def get_mapping(self, index: str) -> dict:
        """Return the live mapping for ``index``."""
        return self.request("GET", f"/{index}/_mapping")

    def delete_index(self, index: str) -> dict:
        """Delete ``index``. Used only by tests and teardown."""
        return self.request("DELETE", f"/{index}")

    # -- documents ----------------------------------------------------------

    def put_doc(self, index: str, doc_id: str, doc: dict) -> dict:
        """Upsert a document under a caller-chosen id.

        Uses ``PUT /{index}/_doc/{id}`` so the id is deterministic and a
        repeated write replaces rather than duplicates. ``POST`` without an id
        would mint a fresh ``_id`` on every call and silently accumulate copies.
        """
        return self.request("PUT", f"/{index}/_doc/{doc_id}", body=doc)

    def get_doc(self, index: str, doc_id: str) -> Optional[dict]:
        """Fetch a document by id, or ``None`` when absent.

        Returns
        -------
        dict or None
            The ``_source`` of the document.
        """
        try:
            res = self.request("GET", f"/{index}/_doc/{doc_id}",
                               ok_status=(200,))
        except AOSSNotFound:
            return None
        if not res.get("found", True):
            return None
        return res.get("_source")

    def delete_doc(self, index: str, doc_id: str) -> bool:
        """Delete a document by id. Returns False when it was already absent."""
        try:
            self.request("DELETE", f"/{index}/_doc/{doc_id}", ok_status=(200,))
            return True
        except AOSSNotFound:
            return False

    def search(self, index: str, body: dict) -> dict:
        """Run a search against ``index``."""
        return self.request("POST", f"/{index}/_search", body=body,
                            ok_status=(200,))

    def bulk(self, lines: bytes) -> dict:
        """Send a newline-delimited ``_bulk`` payload."""
        return self.request("POST", "/_bulk", body=lines, ok_status=(200,))
