"""Where campaign state lives.

The capability this exists for: "if you stop your BO at some sixth experiment
-- now, can you run from the seventh?"

Design decision worth stating, because it differs from persisting a pickled
model. A campaign here is completely described by

    (search space, observations, skipped, config, seed)

and `next_recommendation` refits the surrogate from scratch on every call.
There is no hidden model state. So the JSON record IS the model: reload it and
experiment 7 continues exactly as if nothing had stopped.

That is deliberately more robust than storing a fitted torch object, which
breaks across library upgrades and makes unpickling a security question in a
product. Nothing stops a cache being added later for speed; the JSON stays
authoritative.

The backend is pluggable. LocalJSONStore is the default so the endpoint runs
standalone with no infrastructure; S3, Azure Blob or Elasticsearch slot in
behind the same three methods.
"""
from __future__ import annotations

import json
import os
import threading
import time
import uuid
from typing import Any, Dict, List, Optional

DEFAULT_ROOT = os.environ.get(
    "BEACON_STORE_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "store_data"))


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


class StoreError(RuntimeError):
    pass


class CampaignStore:
    """Three methods. Implement these to swap in any backend."""

    def put(self, kind: str, key: str, obj: dict) -> None:
        raise NotImplementedError

    def get(self, kind: str, key: str) -> dict:
        raise NotImplementedError

    def list(self, kind: str) -> List[str]:
        raise NotImplementedError

    def exists(self, kind: str, key: str) -> bool:
        try:
            self.get(kind, key)
            return True
        except StoreError:
            return False


class LocalJSONStore(CampaignStore):
    """One JSON file per object, written atomically."""

    def __init__(self, root: str = DEFAULT_ROOT):
        self.root = root
        self._lock = threading.Lock()
        os.makedirs(root, exist_ok=True)

    def _path(self, kind: str, key: str) -> str:
        if not key or "/" in key or ".." in key:
            raise StoreError(f"unsafe key: {key!r}")
        d = os.path.join(self.root, kind)
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, f"{key}.json")

    def put(self, kind: str, key: str, obj: dict) -> None:
        p = self._path(kind, key)
        tmp = p + ".tmp"
        obj = dict(obj)
        obj["_updated"] = time.time()
        with self._lock:
            with open(tmp, "w") as fh:
                json.dump(obj, fh, indent=2, default=str)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, p)          # atomic: a crash cannot leave a torn file

    def get(self, kind: str, key: str) -> dict:
        p = self._path(kind, key)
        if not os.path.exists(p):
            raise StoreError(f"{kind} {key!r} not found")
        with open(p) as fh:
            return json.load(fh)

    def list(self, kind: str) -> List[str]:
        d = os.path.join(self.root, kind)
        if not os.path.isdir(d):
            return []
        return sorted(f[:-5] for f in os.listdir(d) if f.endswith(".json"))


class MemoryStore(CampaignStore):
    """For tests. Nothing survives the process."""

    def __init__(self):
        self._d: Dict[str, Dict[str, dict]] = {}

    def put(self, kind, key, obj):
        self._d.setdefault(kind, {})[key] = json.loads(json.dumps(obj, default=str))

    def get(self, kind, key):
        try:
            return json.loads(json.dumps(self._d[kind][key]))
        except KeyError:
            raise StoreError(f"{kind} {key!r} not found") from None

    def list(self, kind):
        return sorted(self._d.get(kind, {}))


_default: Optional[CampaignStore] = None


def default_store() -> CampaignStore:
    global _default
    if _default is None:
        _default = LocalJSONStore()
    return _default


def set_default_store(s: CampaignStore) -> None:
    global _default
    _default = s
