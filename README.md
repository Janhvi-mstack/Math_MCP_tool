# BEACON BO tool

One MCP endpoint exposing Bayesian optimisation over any search space —
continuous, integer, ordinal, categorical, or any mix, with constraints.

**`MCP_TOOLS.md` is the contract.** It documents every function, its input
schema, what it returns, and how it fails.

## Install

```bash
pip install torch botorch gpytorch scikit-learn numpy scipy
pip install rdkit            # for search-space normalisation
```

Developed against torch 1.13.1 / botorch 0.8.5 / gpytorch 1.10.

## Run

```bash
python maths_tools/mcp_server.py        # stdio JSON-RPC
```

Self-contained: no network, no API keys, no external services. Campaign state
goes to `BEACON_STORE_DIR` (default `maths_tools/store_data/`).

## Check it works

```bash
PYTHONPATH=. python maths_tools/tests/run_all.py
```

148 tests. `test_mcp.py` drives the real server over stdio, including the
stop-and-resume case. `test_metrics.py` guards the benchmark target definition
and `test_vendored.py` guards the vendored deep kernel against drift.

## Shape of a session

```
create_search_space      ->  space_id
zero_shot_recommendation ->  first experiments + campaign_id
record_observations      ->  evidence back in
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
| `store.py` | campaign persistence; local JSON, pluggable |
| `adapters.py` | helpers for pre-featurised matrices |

## Two things worth reading before integrating

**Skipped experiments stay in the search space.** A recommendation nobody ran is
dropped from the model — not recorded as zero, not filled with a guess — but the
conditions remain eligible and can be recommended again. `MCP_TOOLS.md` §4.

**Resume is exact.** A campaign is `(space, observations, skipped, config,
seed)` in JSON, and the model is refitted from that evidence on every call.
There is no fitted-model state to restore, so stopping at experiment 6 and
continuing at 7 needs nothing but the campaign id.

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

## Optional environment variables

| variable | effect |
|---|---|
| `BEACON_STORE_DIR` | where campaign state is written (default `maths_tools/store_data/`) |
| `BEACON_REFERENCE_SRC` | path to the reference protocol source; enables the benchmark-target guard in `metrics.py` |
| `BEACON_REFERENCE_KERNEL` | path to the reference kernel source; enables the source-level drift check |
| `BEACON_REFERENCE_PLATES` | directory of reference plate data; enables the real-data metric check |

None are required. Unset, the corresponding checks report that they were
skipped rather than failing.
