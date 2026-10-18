"""SigV4 verification against published AWS vectors.

These are external checks, not self-consistency checks: the expected values
come from AWS's own documentation and the ``aws-sig-v4-test-suite``, so a
mistake in :mod:`signing` cannot agree with them by construction.
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datetime import datetime, timezone

from maths_tools.aoss.config import SERVICE_NAME  # noqa: E402
from maths_tools.aoss.signing import (canonical_query, canonical_uri,  # noqa: E402
                     sign_request, signing_key)

# AWS "Examples of how to derive a signing key" (Signature Version 4 docs).
_DERIVE_SECRET = "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY"
_DERIVE_EXPECTED = ("f4780e2d9f65fa895f9c67b32ce1baf0"
                    "b0d8a43505a000a1a9e090d414db404d")

# aws-sig-v4-test-suite / get-vanilla
_SUITE_KEY = "AKIDEXAMPLE"
_SUITE_SECRET = "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY"
_SUITE_WHEN = datetime(2015, 8, 30, 12, 36, 0, tzinfo=timezone.utc)
_GET_VANILLA_SIG = ("5fa00fa31553b73ebf1942676e86291e"
                    "8372ff2a2260956d9b8aae1d763fbf31")


def test_signing_key_matches_aws_documented_derivation():
    """The HMAC chain reproduces AWS's published signing key."""
    got = signing_key(_DERIVE_SECRET, "20120215", "us-east-1", "iam")
    assert got.hex() == _DERIVE_EXPECTED, got.hex()


def test_get_vanilla_matches_official_test_suite():
    """A full signature matches the canonical get-vanilla vector."""
    headers = sign_request(
        "GET", "example.amazonaws.com", "/",
        region="us-east-1", service="service",
        access_key=_SUITE_KEY, secret_key=_SUITE_SECRET,
        now=_SUITE_WHEN, include_content_sha256=False)
    auth = headers["Authorization"]
    assert f"Signature={_GET_VANILLA_SIG}" in auth, auth
    assert "SignedHeaders=host;x-amz-date," in auth, auth


def test_service_name_is_aoss_not_es():
    """The credential scope says aoss. Signing as es returns a bare 403."""
    assert SERVICE_NAME == "aoss"
    auth = sign_request(
        "GET", "c.aoss.us-east-1.on.aws", "/i/_doc/x",
        region="us-east-1", service=SERVICE_NAME,
        access_key="AKIA", secret_key="s", now=_SUITE_WHEN)["Authorization"]
    assert "/20150830/us-east-1/aoss/aws4_request" in auth, auth
    assert "/es/aws4_request" not in auth


def test_content_sha256_is_signed_for_aoss():
    """AOSS requires x-amz-content-sha256, present and in the signed set."""
    body = b'{"a":1}'
    headers = sign_request(
        "PUT", "c.aoss.us-east-1.on.aws", "/i/_doc/x",
        region="us-east-1", service=SERVICE_NAME,
        access_key="AKIA", secret_key="s", body=body, now=_SUITE_WHEN)
    import hashlib
    assert headers["x-amz-content-sha256"] == hashlib.sha256(body).hexdigest()
    assert "x-amz-content-sha256" in headers["Authorization"]


def test_session_token_is_signed_when_present():
    """Temporary credentials sign and send x-amz-security-token."""
    headers = sign_request(
        "GET", "c.aoss.us-east-1.on.aws", "/",
        region="us-east-1", service=SERVICE_NAME,
        access_key="AKIA", secret_key="s", session_token="tok",
        now=_SUITE_WHEN)
    assert headers["x-amz-security-token"] == "tok"
    assert "x-amz-security-token" in headers["Authorization"]


def test_body_changes_the_signature():
    """A different payload must produce a different signature."""
    common = dict(region="us-east-1", service=SERVICE_NAME,
                  access_key="AKIA", secret_key="s", now=_SUITE_WHEN)
    a = sign_request("PUT", "h", "/i/_doc/x", body=b'{"a":1}', **common)
    b = sign_request("PUT", "h", "/i/_doc/x", body=b'{"a":2}', **common)
    assert a["Authorization"] != b["Authorization"]


def test_canonical_query_is_sorted_and_encoded():
    """SigV4 signs the sorted canonical query string."""
    assert canonical_query({"b": 2, "a": "x y"}) == "a=x%20y&b=2"
    assert canonical_query(None) == ""
    assert canonical_query({"size": 500, "flag": True}) == "flag=true&size=500"


def test_canonical_uri_encodes_segments_but_keeps_separators():
    assert canonical_uri("/beacon-campaigns-v1/_doc/camp_1") == \
        "/beacon-campaigns-v1/_doc/camp_1"
    assert canonical_uri("") == "/"


TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]

if __name__ == "__main__":
    passed, failed = 0, []
    for fn in TESTS:
        try:
            fn()
            passed += 1
        except Exception as exc:                                 # noqa: BLE001
            failed.append((fn.__name__, exc))
    print("PASS %d" % passed)
    for name, exc in failed:
        print("FAIL %s" % name)
        print("     %s: %s" % (type(exc).__name__, exc))
    print("FAILED %d" % len(failed))
    sys.exit(1 if failed else 0)
