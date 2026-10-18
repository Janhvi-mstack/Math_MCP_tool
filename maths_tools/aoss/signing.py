"""AWS Signature Version 4 signing for OpenSearch Serverless.

Hand-rolled against the published specification so the backend needs no AWS
SDK. The implementation is verified in :mod:`tests.test_signing` against the
canonical ``aws-sig-v4-test-suite`` vectors rather than merely asserted to be
correct.

The one thing to get right
--------------------------
The credential scope service name must be ``aoss``. OpenSearch *Serverless*
and the provisioned OpenSearch *Service* are different services with different
scopes; signing a serverless request as ``es`` is accepted by the signer,
rejected by the endpoint, and returns a bare ``403`` whose body says nothing
about why. :data:`config.SERVICE_NAME` is therefore a constant, not a knob.

AOSS additionally requires the payload hash to be present as the
``x-amz-content-sha256`` header and included in the signed set.
"""
from __future__ import annotations

import hashlib
import hmac
from datetime import datetime, timezone
from typing import Dict, Mapping, Optional, Tuple
from urllib.parse import quote

ALGORITHM = "AWS4-HMAC-SHA256"

#: Headers excluded from the signed set. ``authorization`` is the output;
#: the others are routinely rewritten by proxies between signer and endpoint.
_UNSIGNED = frozenset({"authorization", "user-agent", "expect",
                       "content-length", "connection", "accept-encoding"})

#: SHA-256 of the empty string -- the payload hash for a body-less request.
EMPTY_PAYLOAD_SHA256 = hashlib.sha256(b"").hexdigest()


def _sha256_hex(data: bytes) -> str:
    """Return the lowercase hex SHA-256 digest of ``data``."""
    return hashlib.sha256(data).hexdigest()


def _hmac(key: bytes, msg: str) -> bytes:
    """Return ``HMAC-SHA256(key, msg)`` with ``msg`` encoded as UTF-8."""
    return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()


def canonical_uri(path: str) -> str:
    """Normalise and percent-encode a request path for the canonical request.

    Each segment is encoded with ``quote``; the separators are preserved. An
    empty path becomes ``/``.

    Parameters
    ----------
    path : str
        Raw request path, e.g. ``/beacon-campaigns-v1/_doc/camp_1``.

    Returns
    -------
    str
        The canonical, percent-encoded path.

    Examples
    --------
    >>> canonical_uri("")
    '/'
    >>> canonical_uri("/a b/c")
    '/a%20b/c'
    """
    if not path:
        return "/"
    return "/".join(quote(seg, safe="-_.~") for seg in path.split("/"))


def canonical_query(query: Optional[Mapping[str, object]]) -> str:
    """Build the canonical query string: sorted, encoded ``k=v`` pairs.

    Parameters
    ----------
    query : mapping or None
        Query parameters. ``None`` and ``{}`` both yield ``""``.

    Returns
    -------
    str
        Canonical query string.

    Examples
    --------
    >>> canonical_query({"b": 2, "a": "x y"})
    'a=x%20y&b=2'
    """
    if not query:
        return ""
    items = []
    for key in sorted(query):
        value = query[key]
        text = ("true" if value is True else
                "false" if value is False else str(value))
        items.append(f"{quote(str(key), safe='-_.~')}={quote(text, safe='-_.~')}")
    return "&".join(items)


def _canonical_headers(headers: Mapping[str, str]) -> Tuple[str, str]:
    """Return ``(canonical_headers, signed_headers)`` for the signed set.

    Header names are lowercased, values stripped and internal runs of spaces
    collapsed, and the result sorted by name.
    """
    cleaned = {}
    for name, value in headers.items():
        low = name.lower().strip()
        if low in _UNSIGNED:
            continue
        cleaned[low] = " ".join(str(value).split())
    names = sorted(cleaned)
    blob = "".join(f"{n}:{cleaned[n]}\n" for n in names)
    return blob, ";".join(names)


def signing_key(secret_key: str, date_stamp: str, region: str,
                service: str) -> bytes:
    """Derive the SigV4 signing key.

    Parameters
    ----------
    secret_key : str
        AWS secret access key.
    date_stamp : str
        UTC date as ``YYYYMMDD``.
    region : str
        AWS region.
    service : str
        Service name -- ``aoss`` for OpenSearch Serverless.

    Returns
    -------
    bytes
        The derived signing key.
    """
    k_date = _hmac(("AWS4" + secret_key).encode("utf-8"), date_stamp)
    k_region = _hmac(k_date, region)
    k_service = _hmac(k_region, service)
    return _hmac(k_service, "aws4_request")


def sign_request(method: str, host: str, path: str, *, region: str,
                 service: str, access_key: str, secret_key: str,
                 session_token: Optional[str] = None,
                 query: Optional[Mapping[str, object]] = None,
                 headers: Optional[Mapping[str, str]] = None,
                 body: bytes = b"",
                 now: Optional[datetime] = None,
                 include_content_sha256: bool = True) -> Dict[str, str]:
    """Sign a request and return the headers to send.

    Parameters
    ----------
    method : str
        HTTP method, e.g. ``"PUT"``.
    host : str
        Bare hostname; also sent as the ``host`` header.
    path : str
        Request path beginning with ``/``.
    region : str
        AWS region for the credential scope.
    service : str
        Service name for the credential scope. Pass ``"aoss"``.
    access_key, secret_key : str
        AWS credentials.
    session_token : str, optional
        Session token for temporary credentials. When given it is sent and
        signed as ``x-amz-security-token``.
    query : mapping, optional
        Query parameters.
    headers : mapping, optional
        Extra headers to send and sign, e.g. ``content-type``.
    body : bytes, optional
        Request body. Defaults to empty.
    now : datetime, optional
        Signing time; defaults to the current UTC time. Tests pass a fixed
        value to make signatures reproducible.
    include_content_sha256 : bool, optional
        Whether to send and sign ``x-amz-content-sha256`` (default True).
        AOSS requires it, so production always leaves this on. It is turned
        off only by :mod:`tests.test_signing`, which reproduces the official
        ``aws-sig-v4-test-suite`` vectors -- those predate the header, so
        including it would change the signed header set and make the published
        signatures unmatchable.

    Returns
    -------
    dict
        Complete headers including ``Authorization``, ready to send.

    Examples
    --------
    >>> from datetime import datetime, timezone
    >>> h = sign_request("GET", "h.us-east-1.aoss.amazonaws.com", "/",
    ...                  region="us-east-1", service="aoss",
    ...                  access_key="AKIDEXAMPLE", secret_key="secret",
    ...                  now=datetime(2015, 8, 30, 12, 36, tzinfo=timezone.utc))
    >>> h["Authorization"].startswith("AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/20150830/us-east-1/aoss/aws4_request")
    True
    """
    stamp = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    amz_date = stamp.strftime("%Y%m%dT%H%M%SZ")
    date_stamp = stamp.strftime("%Y%m%d")
    payload_hash = _sha256_hex(body or b"")

    send = {str(k): str(v) for k, v in (headers or {}).items()}
    send["host"] = host
    send["x-amz-date"] = amz_date
    # AOSS requires the payload hash as a header, not only inside the
    # canonical request.
    if include_content_sha256:
        send["x-amz-content-sha256"] = payload_hash
    if session_token:
        send["x-amz-security-token"] = session_token

    canon_headers, signed_headers = _canonical_headers(send)
    canonical_request = "\n".join([
        method.upper(),
        canonical_uri(path),
        canonical_query(query),
        canon_headers,
        signed_headers,
        payload_hash,
    ])

    scope = f"{date_stamp}/{region}/{service}/aws4_request"
    string_to_sign = "\n".join([
        ALGORITHM, amz_date, scope,
        _sha256_hex(canonical_request.encode("utf-8")),
    ])

    signature = hmac.new(
        signing_key(secret_key, date_stamp, region, service),
        string_to_sign.encode("utf-8"), hashlib.sha256).hexdigest()

    send["Authorization"] = (
        f"{ALGORITHM} Credential={access_key}/{scope}, "
        f"SignedHeaders={signed_headers}, Signature={signature}")
    return send
