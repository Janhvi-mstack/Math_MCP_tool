"""Unit tests for the index mappings.

These assert the specific type choices that prevent AOSS from guessing wrong,
so a future edit that reverts one of them fails here rather than in the cloud.
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from maths_tools.aoss import mappings  # noqa: E402


def test_three_distinct_index_names():
    names = mappings.all_index_names()
    assert len(set(names.values())) == 3, names
    assert sorted(names.values()) == ["beacon-campaigns-v1",
                                      "beacon-observations-v1",
                                      "beacon-search-spaces-v1"]


def test_prefix_separates_environments():
    """Staging and production can share a collection without sharing docs."""
    a = set(mappings.all_index_names("beacon").values())
    b = set(mappings.all_index_names("beacon-stg").values())
    assert not (a & b), a & b


def test_every_index_is_strict():
    """Strict means AOSS REFUSES an undeclared field instead of guessing it."""
    for kind in mappings.KINDS:
        m = mappings.mapping_for(kind)["mappings"]
        assert m["dynamic"] == "strict", kind


def test_yield_is_double_not_long():
    """A first document with y=0 would make long the guess, truncating 41.7."""
    props = mappings.OBSERVATION_MAPPING["mappings"]["properties"]
    assert props["y"]["type"] == "double"
    assert mappings.CAMPAIGN_MAPPING["mappings"]["properties"]["best_y"]["type"] \
        == "double"


def test_timestamps_are_dates_with_epoch_millis():
    """A float epoch second would otherwise be guessed as a float."""
    for kind in mappings.KINDS:
        props = mappings.mapping_for(kind)["mappings"]["properties"]
        for field, spec in props.items():
            if field.endswith("_at"):
                assert spec["type"] == "date", (kind, field)
                assert "epoch_millis" in spec["format"], (kind, field)


def test_payload_is_stored_but_never_indexed():
    """User-chosen factor names must not reach the mapping."""
    for kind in mappings.KINDS:
        payload = mappings.mapping_for(kind)["mappings"]["properties"]["payload"]
        assert payload["type"] == "object"
        assert payload["enabled"] is False, kind


def test_the_agent_query_surface_is_keyword_typed():
    """by reaction, by user, by campaign, by outcome."""
    obs = mappings.OBSERVATION_MAPPING["mappings"]["properties"]
    for field in ("reaction_id", "user_id", "campaign_id", "outcome"):
        assert obs[field]["type"] == "keyword", field
    camp = mappings.CAMPAIGN_MAPPING["mappings"]["properties"]
    for field in ("reaction_id", "user_id", "campaign_id", "space_id"):
        assert camp[field]["type"] == "keyword", field
    space = mappings.SPACE_MAPPING["mappings"]["properties"]
    for field in ("reaction_id", "user_id", "space_id"):
        assert space[field]["type"] == "keyword", field


def test_doc_id_is_sortable_for_pagination():
    """list() sorts on doc_id via search_after; _id is not sortable."""
    for kind in mappings.KINDS:
        props = mappings.mapping_for(kind)["mappings"]["properties"]
        assert props["doc_id"]["type"] == "keyword", kind


def test_creation_body_sets_no_shard_or_replica_settings():
    """Serverless manages these and rejects a body that tries to set them."""
    for kind in mappings.KINDS:
        body = mappings.mapping_for(kind)
        assert set(body) == {"mappings"}, (kind, set(body))


def test_unknown_kind_is_rejected():
    for fn in (mappings.mapping_for, mappings.index_name):
        try:
            fn("nonsense")
        except KeyError:
            continue
        raise AssertionError(f"{fn.__name__} accepted an unknown kind")


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
