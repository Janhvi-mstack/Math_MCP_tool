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

The backend is pluggable behind those three methods. Two ship:

    AOSSStore         AWS OpenSearch Serverless -- see maths_tools/aoss/
    LocalJSONStore    one file per object, no infrastructure at all

`select_backend` picks between them from the environment and says which one it
picked. Local JSON remains fully supported, not a degraded mode: it is what
runs when nobody has credentials, and it is what the tests use.
"""
from __future__ import annotations

import json
import os
import sys
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


# --------------------------------------------------------------- selection

AOSS = "aoss"
LOCAL = "local-json"

#: Required for the AOSS backend. All present -> AOSS. None present -> local
#: JSON. Some but not all is a misconfiguration and raises, because that is
#: precisely how a production deployment ends up writing to a local file.
_AOSS_REQUIRED = ("AOSS_ENDPOINT", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY")

#: Set to "aoss" or "local" to override the automatic choice. An explicit
#: override is fine; a silent fallback is not.
_BACKEND_OVERRIDE = "BEACON_STORE_BACKEND"


def choose_backend(env: Optional[dict] = None) -> str:
    """Decide which backend the environment selects, without building it.

    Parameters
    ----------
    env : dict, optional
        Mapping to read instead of :data:`os.environ`. Used by tests.

    Returns
    -------
    str
        Either :data:`AOSS` or :data:`LOCAL`.

    Raises
    ------
    StoreError
        If ``BEACON_STORE_BACKEND`` holds an unrecognised value, or if the
        AOSS variables are present but incomplete.

    Examples
    --------
    >>> choose_backend({})
    'local-json'
    >>> choose_backend({"AOSS_ENDPOINT": "https://h", "AWS_ACCESS_KEY_ID": "a",
    ...                 "AWS_SECRET_ACCESS_KEY": "s"})
    'aoss'
    >>> choose_backend({"BEACON_STORE_BACKEND": "local",
    ...                 "AOSS_ENDPOINT": "https://h", "AWS_ACCESS_KEY_ID": "a",
    ...                 "AWS_SECRET_ACCESS_KEY": "s"})
    'local-json'
    """
    src = os.environ if env is None else env

    override = (src.get(_BACKEND_OVERRIDE) or "").strip().lower()
    if override:
        if override in (AOSS, "opensearch"):
            return AOSS
        if override in (LOCAL, "local", "json"):
            return LOCAL
        raise StoreError(
            f"{_BACKEND_OVERRIDE}={override!r} is not recognised; "
            f"use {AOSS!r} or 'local'")

    present = [k for k in _AOSS_REQUIRED if (src.get(k) or "").strip()]
    if not present:
        return LOCAL
    if len(present) == len(_AOSS_REQUIRED):
        return AOSS

    missing = [k for k in _AOSS_REQUIRED if k not in present]
    raise StoreError(
        "AOSS is partially configured: " + ", ".join(present) + " set, " +
        ", ".join(missing) + " missing. Refusing to fall back to local JSON "
        "silently -- set the missing variable(s), or set "
        f"{_BACKEND_OVERRIDE}=local to use local JSON deliberately.")


def _log_backend(line: str) -> None:
    """Announce the selected backend on stderr.

    stderr, never stdout: the MCP server speaks stdio JSON-RPC and a stray
    line on stdout corrupts the protocol stream.
    """
    print(f"[maths_tools] store backend: {line}", file=sys.stderr, flush=True)


def select_backend(env: Optional[dict] = None,
                   announce: bool = True) -> CampaignStore:
    """Build the store the environment selects, and say which one it is.

    AOSS when its variables are present, local JSON otherwise. The choice is
    logged either way; a backend that changes without anyone noticing is the
    failure this function exists to prevent.

    Parameters
    ----------
    env : dict, optional
        Mapping to read instead of :data:`os.environ`.
    announce : bool, optional
        Write the one-line selection notice to stderr (default True).

    Returns
    -------
    CampaignStore
        An :class:`~maths_tools.aoss.store.AOSSStore` or a
        :class:`LocalJSONStore`.

    Raises
    ------
    StoreError
        If the configuration is incomplete, or if AOSS is selected and its
        indices cannot be reached. It does **not** fall back to local JSON on
        an AOSS failure -- see Notes.

    Notes
    -----
    Selecting AOSS creates the three indices with their explicit mappings if
    they are absent, before anything is written. AOSS infers a mapping from
    the first document it receives and guesses types wrong, so the indices
    have to exist first.

    That makes startup fail when the collection is unreachable, which is
    deliberate. The alternatives are worse: falling back to local JSON hides a
    broken deployment, and writing anyway lets AOSS invent the mapping.

    The log line carries the region and the index names. It never carries the
    endpoint host, the access key, or any secret.

    ``BEACON_AOSS_CACHE_WRITES=false`` turns off the write-through cache.
    **Any deployment running more than one process must set it**, because the
    cache is per-process and is never invalidated by another process's write.

    ``BEACON_AOSS_MISS_GRACE`` overrides how long a ``get`` keeps retrying a
    key this process did not write. It exists so the behaviour can be turned
    off and the cross-process failure reproduced deliberately, as well as
    tuned; see the README. Setting it to 0 restores fail-fast and WILL produce
    spurious "not found" results under multiple workers.
    """
    kind = choose_backend(env)

    if kind == AOSS:
        from .aoss import AOSSStore, ConfigError
        from .aoss import load as load_aoss_config
        from .aoss.store import DEFAULT_MISS_GRACE
        src = os.environ if env is None else env
        raw = (src.get("BEACON_AOSS_MISS_GRACE") or "").strip()
        try:
            grace = float(raw) if raw else DEFAULT_MISS_GRACE
        except ValueError:
            raise StoreError(
                "BEACON_AOSS_MISS_GRACE=%r is not a number of seconds" % raw)

        # The write-through cache is sound only while ONE process writes a
        # given key. It is a per-process cache with no cross-process
        # invalidation, so under several workers each one keeps serving its
        # own version of a campaign indefinitely: measured on a four-worker
        # container, one worker reported 2 observations, another reported 0,
        # and neither ever converged. Any multi-process deployment must set
        # this false.
        flag = (src.get("BEACON_AOSS_CACHE_WRITES") or "").strip().lower()
        if flag in ("0", "false", "no", "off"):
            cache_writes = False
        elif flag in ("", "1", "true", "yes", "on"):
            cache_writes = True
        else:
            raise StoreError(
                "BEACON_AOSS_CACHE_WRITES=%r is not a boolean" % flag)

        try:
            cfg = load_aoss_config(env)
            store = AOSSStore(cfg, miss_grace=grace,
                              cache_writes=cache_writes)
            state = store.ensure_indices()
        except ConfigError as exc:
            raise StoreError(f"AOSS configuration is unusable: {exc}") from exc
        if announce:
            _log_backend(
                "aoss (AWS OpenSearch Serverless)  region=%s  cache=%s  "
                "indices=%s" % (
                    cfg.region, "on" if cache_writes else "off",
                    ", ".join("%s[%s]" % (name, state.get(k, "?"))
                              for k, name in sorted(
                                  store._indices.items(), key=lambda kv: kv[1]))))
        return store

    store = LocalJSONStore()
    if announce:
        _log_backend(
            "local JSON (standalone; no AOSS variables in the environment)  "
            "root=%s" % store.root)
    return store


_default: Optional[CampaignStore] = None


def default_store() -> CampaignStore:
    global _default
    if _default is None:
        _default = select_backend()
    return _default


def set_default_store(s: CampaignStore) -> None:
    global _default
    _default = s
