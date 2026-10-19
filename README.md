# BEACON BO tool

One MCP endpoint exposing Bayesian optimisation over any search space —
continuous, integer, ordinal, categorical, or any mix, with constraints.

**`MCP_TOOLS.md` is the contract.** It documents every function, its input
schema, what it returns, and how it fails.

## Install

```bash
pip install -r requirements.txt
```

Developed against torch 1.13.1 / botorch 0.8.5 / gpytorch 1.10, Python 3.9.

The AOSS storage backend adds **no** dependency: SigV4 is implemented against
the standard library, so there is no boto3 and no requests. A test enforces
that rather than trusting it.

## Run

### Standalone, on local JSON — no AWS, no credentials

```bash
python maths_tools/mcp_server.py        # stdio JSON-RPC
```

That is the whole thing. With no AOSS variables in the environment the tool
stores campaigns as one JSON file per object under `BEACON_STORE_DIR`
(default `maths_tools/store_data/`) and needs no network, no API key and no
external service. It is a fully supported mode, not a degraded one — it is
what the tests use.

### Against AOSS

```bash
set -a; . ./.env; set +a                # or however you load the environment
python maths_tools/mcp_server.py
```

## Which backend you got

**AOSS when its variables are present, local JSON when they are not.** The
server says which, on stderr, before it serves the first request:

```
[maths_tools] store backend: aoss (AWS OpenSearch Serverless)  region=us-east-1  indices=beacon-campaigns-v1[exists], ...
[maths_tools] store backend: local JSON (standalone; no AOSS variables in the environment)  root=/…/store_data
```

stderr, never stdout — stdout carries JSON-RPC frames and one stray line
corrupts the stream. The line names the region and the indices; it never
carries the endpoint host, the access key or any secret.

Two rules make the choice hard to get wrong by accident:

- **A partial AOSS configuration refuses to start.** Endpoint set but secret
  missing is a misconfiguration, not a reason to fall back. Falling back
  silently is how a deployment ends up writing to a container-local file that
  nobody looks at until the data is wanted.
- **AOSS failures do not fall back either.** If the collection is unreachable,
  or the indices cannot be created, startup fails. Local JSON is never chosen
  as a consolation prize.

Set `BEACON_STORE_BACKEND=local` to force local JSON deliberately with
credentials present, or `=aoss` to force the other way.

## Environment

**Never commit these, never echo them, never put them in a log line.** The
config module reads them from the environment and nowhere else — it does not
read credentials from disk itself. `.gitignore` covers `.env`.

### Required for AOSS

| variable | where it comes from |
|---|---|
| `AOSS_ENDPOINT` | the collection's **OpenSearch endpoint** — AWS console → OpenSearch Service → Serverless → Collections → *your collection*. `https://…` |
| `AWS_ACCESS_KEY_ID` | the IAM user or role the collection's data access policy names |
| `AWS_SECRET_ACCESS_KEY` | issued with the access key id |

### Optional

| variable | effect |
|---|---|
| `AWS_SESSION_TOKEN` | required **only** for temporary credentials (STS, an assumed role, SSO). Omit for a long-lived IAM user key. |
| `AWS_DEFAULT_REGION` / `AWS_REGION` / `AOSS_REGION` | the signing region. Inferred from the endpoint host when that host carries it, so usually unnecessary. First one set wins, in that order. |
| `BEACON_AOSS_INDEX_PREFIX` | prefix for this deployment's indices (default `beacon`). Staging and production can share a collection without sharing documents. |
| `BEACON_AOSS_MISS_GRACE` | seconds a `get` keeps retrying a key this process did not write (default 15). See below. **Setting it to 0 restores fail-fast and will produce spurious "not found" results under multiple workers** — it exists for tuning, and for reproducing that failure deliberately. |
| `BEACON_STORE_BACKEND` | `aoss` or `local`, to override the automatic choice |
| `BEACON_STORE_DIR` | where local JSON writes (default `maths_tools/store_data/`) |
| `BEACON_REFERENCE_SRC` | path to the reference protocol source; enables the benchmark-target guard in `metrics.py` |
| `BEACON_REFERENCE_KERNEL` | path to the reference kernel source; enables the source-level drift check |
| `BEACON_REFERENCE_PLATES` | directory of reference plate data; enables the real-data metric check |

### IAM is not enough

The collection needs a **data access policy** granting the calling principal
`aoss:APIAccessAll`. IAM permissions alone produce a bare `403` with no
explanation of what is missing. If writes 403 with correct `aoss` signing,
that policy is what to look at — the error raised names it.

## Check it works

```bash
PYTHONPATH=. python maths_tools/tests/run_all.py
```

**258 tests, no credentials and no network required.** The AOSS suites drive a
strict in-memory stand-in that enforces `dynamic: strict`, declared field
types and visibility lag, so the cloud-specific failures are reproduced
offline and cost nothing.

| suite | what it covers |
|---|---|
| `test_mcp.py` | the real server over stdio, including stop-and-resume |
| `test_backend.py` | which backend gets selected, and that the choice is announced |
| `test_aoss_signing.py` | SigV4 against AWS's **published** vectors, not self-consistency |
| `test_aoss_mappings.py` | the specific type choices that stop AOSS guessing |
| `test_aoss_store.py` | the store contract, parity with local JSON, the lag paths |
| `test_metrics.py` | the benchmark target definition |
| `test_vendored.py` | the vendored deep kernel against drift |

Live checks against a real collection, when you have credentials:

```bash
python scripts/probe.py                            # connectivity and auth
python scripts/provision.py                        # create the indices
python scripts/verify_live.py                      # write / read / search / delete
python scripts/verify_campaign.py --expect aoss    # a full campaign, end to end
python scripts/verify_campaign.py --expect local   # the same, standalone
python scripts/verify_resume.py  --expect aoss     # resume across SIGKILL
```

## Storage layout on AOSS

**Three indices, not one with a `doc_type` field.**

| index | one document per | id |
|---|---|---|
| `beacon-search-spaces-v1` | search space — factors, levels, constraints, the normalisation report | `space_id` |
| `beacon-campaigns-v1` | campaign — observations, skips, pending suggestions, config, seed. The authoritative record; this is what resume reads. | `campaign_id` |
| `beacon-observations-v1` | **single experiment**, observed or skipped. A projection of the campaign, one document per experiment, so "which experiments for this reaction were skipped, and why" is a query rather than a scan. | `{campaign_id}.{hash(point)}` |

`BEACON_AOSS_INDEX_PREFIX` replaces `beacon` throughout.

Why three and not one:

1. The three documents have genuinely different field types. One index means
   one mapping that is the union of all three with every field optional, and
   the union is exactly where a type collision becomes possible — the failure
   the explicit mappings exist to prevent.
2. `_id` is unique per index, not per `doc_type`. With one index, a space and
   a campaign that happened to share a key would overwrite each other, so ids
   would need composite prefixes. With three, the key maps straight onto `_id`
   and the existing `(kind, key)` interface carries over unchanged.
3. Nothing is lost — AOSS searches comma-separated indices in one request, so
   "everything for campaign X" is still a single call.

Every mapping is `dynamic: "strict"`: a field the mapping does not declare is
**refused**, not guessed. Yields are `double` (a first document with `y=0`
would otherwise make `long` the guess, after which `41.7` truncates to `41`),
timestamps are `date` with `epoch_millis`, and free-form payloads are
`{"type": "object", "enabled": false}` so user-chosen factor names never reach
the mapping.

## This is AOSS, not Elasticsearch

Three differences, all measured against the live collection rather than
assumed. They are why the backend looks more defensive than a sample script.

**`POST /{index}/_refresh` returns 404.** It does not exist on serverless.
Visibility cannot be forced, so every path below has to tolerate the lag
rather than waiting it out with a refresh.

**`GET /{index}/_doc/{id}` is not realtime.** On classic OpenSearch a
get-by-id reads through the translog and sees a write immediately. On AOSS it
does not: **2.5–6.1 s, median 5.1 s** over five trials. A naive
read-after-write returns "not found" for a document written seconds earlier
and already durable.

**`_cluster/*` is absent** (404), and so is `GET /` — a 404 there is normal,
not an auth failure. `_cat/indices` does in fact respond, but `_aliases` is
what this code uses, and `list` is a paginated `search_after` query because
there is no `_cat` to rely on and no scroll API.

### Why there is a cache *and* a retry

Those two facts mean a correct read-after-write needs both halves, because
they cover different processes:

| situation | what covers it |
|---|---|
| reading a key **this process** just wrote | the **write-through cache** — populated only *after* AOSS acknowledges the write, so a hit can never report a document that is not durable. Instant. |
| reading a key this process wrote but is no longer caching | **read-after-write retry**, up to 30 s |
| reading a key this process **never wrote** — a restarted server, or another worker | **miss grace**, up to 15 s |

The third row is the one that is easy to miss, and it was found by measuring
rather than reasoning. A *fresh* process reading a campaign another process
had just written **missed on the first get-by-id in 4 of 5 trials**, with the
document becoming visible only **3.4–7.5 s** after the write. That process has
no cache and no record that anything was written, so it cannot know to wait —
without the grace it reports "campaign not found" for a campaign that exists.

This matters beyond restarts. Under multiple workers with no sticky routing,
call N+1 routinely lands on a process that did not serve call N, so the
cross-process case is not an edge case — it is most calls.

The cost is that a genuinely absent key takes 15 s to be reported absent. That
is the right trade: a slow "no such campaign" is an annoyance, a fast one for
a campaign that does exist tells a chemist their data is gone.

**`exists()` deliberately skips the grace.** It asks a question about this
instant, and an answer that takes fifteen seconds to say "no" is not useful.
Only `get()` is worth waiting for.

Both are switchable. From the environment, `BEACON_AOSS_MISS_GRACE=0`
restores fail-fast; in code, `AOSSStore(cfg, miss_grace=0.0,
read_after_write_timeout=0.0)` does the same and `cache_writes=False` turns
off the cache.

## Shape of a session

```
create_search_space      ->  space_id
zero_shot_recommendation ->  first experiments + campaign_id
record_observations      ->  evidence back in
skip_recommendation      ->  a suggestion nobody ran
next_recommendation      ->  what to run next
get_campaign             ->  full state; reload this to resume
```

Every call also works statelessly: pass the space and the observations inline
and nothing is stored.

## Layout

| file | |
|---|---|
| `mcp_server.py` | the endpoint — 8 tools |
| `space.py` | search-space definition, validation, constraints |
| `normalise.py` | canonicalises chemical levels, reports what it changed |
| `encode.py` | search space to model input and back |
| `design.py` | zero-shot: random, k-medoids, MaxMin, Sobol |
| `surrogate.py` | deep-kernel GP, mixed GP, plain GP |
| `deep_kernel.py` | the deep-kernel model, vendored verbatim — see note below |
| `acquisition.py` | EI, qEI, qNEI, Thompson; enumerated and continuous paths |
| `campaign.py` | run state, the skip rule |
| `priors.py` | πBO prior injection with RGPE trust-weighting (v2) |
| `metrics.py` | the benchmark target, with a guard against using the wrong one |
| `store.py` | the three-method store interface, backend selection, local JSON |
| `aoss/` | the AOSS backend — config, SigV4, client, mappings, store |
| `adapters.py` | helpers for pre-featurised matrices |

## Two things worth reading before integrating

**Skipped experiments stay in the search space.** A recommendation nobody ran is
dropped from the model — not recorded as zero, not filled with a guess — but the
conditions remain eligible and can be recommended again. `MCP_TOOLS.md` §4.

**Resume is exact.** A campaign is `(space, observations, skipped, config,
seed)`, and the model is refitted from that evidence on every call. There is
no fitted-model state to restore, so stopping at experiment 6 and continuing
at 7 needs nothing but the campaign id — on either backend.

## A note on `deep_kernel.py`

That file is a **verbatim copy** of the reference implementation the published
benchmark numbers were produced with. Two consequences, both deliberate:

- **The docstrings reference the upstream project by name.** That is
  attribution for third-party code and should stay.
- **The text must match the original character for character.**
  `tests/test_vendored.py` diffs it against the reference when
  `BEACON_REFERENCE_KERNEL` points at one, and compares the fitted posterior
  either way.

So please do not tidy, reformat or re-word anything in that file. A cosmetic
cleanup will fail the drift guard, and if the guard is then disabled the copy
can diverge from the code behind the numbers without anyone noticing.
