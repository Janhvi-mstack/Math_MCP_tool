"""Configuration for the AOSS storage backend.

Every value is read from the process environment. Nothing is defaulted to a
real endpoint, nothing is written to a file, and no value in this module is
ever placed in a log line -- :func:`describe` exists so that diagnostics can
report *shape* without reporting *content*.

The expected environment is a small, fixed set of names. A deployment supplies
them the way the rest of the platform does (Vault -> ArgoCD -> container env);
local development supplies them by sourcing an untracked file.

Environment
-----------
AOSS_ENDPOINT
    Collection endpoint, e.g. ``https://abc123.us-east-1.aoss.amazonaws.com``.
    Scheme is optional; it is normalised to ``https``.
AOSS_REGION / AWS_DEFAULT_REGION
    AWS region of the collection, e.g. ``us-east-1``. ``AOSS_REGION`` wins;
    ``AWS_DEFAULT_REGION`` is accepted because that is the name the platform
    credential files already use. Inferred from the endpoint host when both
    are unset.
AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
    Long-lived credentials for the principal named in the collection's data
    access policy.
AWS_SESSION_TOKEN
    Optional. Required when the credentials are temporary (STS / assumed role).
BEACON_AOSS_INDEX_PREFIX
    Optional index-name prefix (default ``beacon``). Lets staging and
    production share one collection without sharing documents.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Optional

#: Service name used in the SigV4 credential scope. OpenSearch *Serverless* is
#: ``aoss``; the provisioned OpenSearch Service is ``es``. Signing a serverless
#: request as ``es`` produces a bare 403 with no explanatory body, so this
#: constant is deliberately not configurable.
SERVICE_NAME = "aoss"

DEFAULT_INDEX_PREFIX = "beacon"

#: Region embedded in a collection endpoint. AOSS serves two host forms and
#: the region sits in a different position in each, so both are matched:
#:   <id>.aoss[-fips].<region>.on.aws        (current serverless form)
#:   <id>.<region>.aoss.amazonaws.com        (legacy form)
_HOST_REGION_PATTERNS = (
    re.compile(r"^[0-9a-z]+\.aoss(?:-fips)?\.([a-z0-9-]+)\.on\.aws$", re.I),
    re.compile(r"^[0-9a-z]+\.([a-z0-9-]+)\.aoss(?:-fips)?\.amazonaws\.com$", re.I),
)


def region_from_host(host: str) -> Optional[str]:
    """Extract the AWS region from an AOSS collection hostname.

    Parameters
    ----------
    host : str
        Bare hostname.

    Returns
    -------
    str or None
        The region, or ``None`` when the host matches no known form.

    Examples
    --------
    >>> region_from_host("abc123.aoss.us-east-1.on.aws")
    'us-east-1'
    >>> region_from_host("abc123.aoss-fips.eu-west-2.on.aws")
    'eu-west-2'
    >>> region_from_host("abc123.us-east-1.aoss.amazonaws.com")
    'us-east-1'
    >>> region_from_host("example.com") is None
    True
    """
    for pattern in _HOST_REGION_PATTERNS:
        m = pattern.match(host)
        if m:
            return m.group(1)
    return None


class ConfigError(RuntimeError):
    """Raised when required configuration is absent or self-inconsistent."""


@dataclass(frozen=True)
class AOSSConfig:
    """Resolved AOSS connection settings.

    Attributes
    ----------
    endpoint : str
        Collection endpoint including scheme, with no trailing slash.
    host : str
        Bare hostname, used for the SigV4 ``host`` header.
    region : str
        AWS region used in the credential scope.
    access_key : str
        AWS access key id.
    secret_key : str
        AWS secret access key.
    session_token : str or None
        Session token when credentials are temporary, otherwise ``None``.
    index_prefix : str
        Prefix applied to every index name this backend owns.
    """

    endpoint: str
    host: str
    region: str
    access_key: str
    secret_key: str
    session_token: Optional[str]
    index_prefix: str = DEFAULT_INDEX_PREFIX

    def describe(self) -> dict:
        """Return a log-safe description of this configuration.

        No secret material is included. The access key id is reduced to its
        length, and the secret and session token to booleans, so that a
        diagnostic can confirm *that* credentials were supplied without
        disclosing them.

        Returns
        -------
        dict
            Keys ``host``, ``region``, ``index_prefix``, ``access_key_len``,
            ``has_secret`` and ``has_session_token``.

        Examples
        --------
        >>> c = AOSSConfig("https://h", "h", "us-east-1", "AK", "sk", None)
        >>> c.describe()["has_secret"]
        True
        >>> "sk" in str(c.describe())
        False
        """
        return {
            "host": self.host,
            "region": self.region,
            "index_prefix": self.index_prefix,
            "access_key_len": len(self.access_key),
            "has_secret": bool(self.secret_key),
            "has_session_token": bool(self.session_token),
        }

    def __repr__(self) -> str:  # pragma: no cover - trivial
        """Redacted repr, so an accidental print or traceback leaks nothing."""
        return (f"AOSSConfig(host={self.host!r}, region={self.region!r}, "
                f"index_prefix={self.index_prefix!r}, credentials=<redacted>)")

    __str__ = __repr__


def _normalise_endpoint(raw: str) -> tuple:
    """Split a raw endpoint into ``(endpoint, host)``.

    Parameters
    ----------
    raw : str
        Endpoint with or without a scheme, with or without a trailing slash.

    Returns
    -------
    tuple of (str, str)
        The ``https://`` endpoint with no trailing slash, and the bare host.

    Examples
    --------
    >>> _normalise_endpoint("abc.us-east-1.aoss.amazonaws.com/")
    ('https://abc.us-east-1.aoss.amazonaws.com', 'abc.us-east-1.aoss.amazonaws.com')
    """
    text = raw.strip().rstrip("/")
    if not text:
        raise ConfigError("AOSS_ENDPOINT is empty")
    host = text.split("://", 1)[1] if "://" in text else text
    host = host.split("/", 1)[0]
    if not host:
        raise ConfigError(f"AOSS_ENDPOINT has no host: {raw!r}")
    return f"https://{host}", host


def load(env: Optional[dict] = None) -> AOSSConfig:
    """Build an :class:`AOSSConfig` from the environment.

    Parameters
    ----------
    env : dict, optional
        Mapping to read instead of :data:`os.environ`. Used by tests.

    Returns
    -------
    AOSSConfig
        Resolved configuration.

    Raises
    ------
    ConfigError
        If any required variable is missing, or the region can be neither read
        nor inferred. The message names the missing variables and never echoes
        a value.

    Examples
    --------
    >>> env = {"AOSS_ENDPOINT": "https://abc123.aoss.us-east-1.on.aws",
    ...        "AWS_ACCESS_KEY_ID": "AKIA", "AWS_SECRET_ACCESS_KEY": "s"}
    >>> load(env).region
    'us-east-1'
    >>> load({**env, "AOSS_ENDPOINT": "https://h.example.com",
    ...       "AWS_DEFAULT_REGION": "eu-west-1"}).region
    'eu-west-1'
    """
    src = os.environ if env is None else env

    missing = [k for k in ("AOSS_ENDPOINT", "AWS_ACCESS_KEY_ID",
                           "AWS_SECRET_ACCESS_KEY")
               if not (src.get(k) or "").strip()]
    if missing:
        raise ConfigError(
            "missing required environment variable(s): " + ", ".join(missing) +
            ". Source the credentials file into the environment before use; "
            "this module never reads credentials from disk itself.")

    endpoint, host = _normalise_endpoint(src["AOSS_ENDPOINT"])

    region = ((src.get("AOSS_REGION") or "").strip()
              or (src.get("AWS_DEFAULT_REGION") or "").strip()
              or (src.get("AWS_REGION") or "").strip())
    if not region:
        region = region_from_host(host)
        if not region:
            raise ConfigError(
                "region is unset and cannot be inferred from the endpoint host. "
                "Set AOSS_REGION or AWS_DEFAULT_REGION explicitly.")

    token = (src.get("AWS_SESSION_TOKEN") or "").strip() or None
    prefix = (src.get("BEACON_AOSS_INDEX_PREFIX") or DEFAULT_INDEX_PREFIX).strip()
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", prefix):
        raise ConfigError(
            f"BEACON_AOSS_INDEX_PREFIX {prefix!r} is not a legal index-name prefix "
            "(lowercase letters, digits, '-' and '_'; must not start with '-' or '_')")

    return AOSSConfig(endpoint=endpoint, host=host, region=region,
                      access_key=src["AWS_ACCESS_KEY_ID"].strip(),
                      secret_key=src["AWS_SECRET_ACCESS_KEY"].strip(),
                      session_token=token, index_prefix=prefix)


def available(env: Optional[dict] = None) -> bool:
    """Return True when a complete configuration is present.

    Lets tests and scripts skip live work cleanly instead of failing.

    Examples
    --------
    >>> available({})
    False
    """
    try:
        load(env)
        return True
    except ConfigError:
        return False
