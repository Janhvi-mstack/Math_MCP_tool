# The BO tool — MCP endpoint

One MCP server. Both halves of the problem take the same search space, so the
zero-shot recommendation and the next recommendation are two functions of one
tool, not two tools.

```bash
python maths_tools/mcp_server.py        # stdio JSON-RPC
```

No network access, no API keys, no external services. Everything runs locally.

---

## The eight functions

| function | what it does | asked for as |
|---|---|---|
| `create_search_space` | define the space once, get a `space_id`, see what normalisation did | **1. initiate search space** |
| `zero_shot_recommendation` | the opening experiments, before any data exists | **3. zero-shot recommendation** |
| `next_recommendation` | what to run next, fitted on the evidence | **4. next recommendation** |
| `get_campaign` | full state — this is how a stopped campaign resumes | **5. persist the BO and reload** |
| `record_observations` | evidence comes back | needed by 4 |
| `skip_recommendation` | a suggestion nobody ran | needed by 4 |
| `get_search_space` | read a stored space back | convenience |
| `list_campaigns` | what campaigns exist | convenience |

Eight functions rather than five because the five needed three supporting calls;
nothing from the original specification was dropped except one, which is deferred:

> **2. initiate knowledge graph — NOT in v1.5, deferred to v2.** The machinery
> exists in `maths_tools.priors` (πBO prior injection with RGPE trust-weighting)
> and will surface as `attach_prior`. It is held back deliberately: our
> knowledge graph currently ranks the genuinely best ligand **third of four**
> across all five plates, correlation **−0.60** with truth. Shipping it as a
> default would make the search worse than no prior at all. RGPE detects and
> switches off a prior that ranks backwards, and that is the mechanism the v2
> function will carry.

---

## 1. `create_search_space`

Takes the factors, returns a `space_id`. **Normalises first and tells you what
it did** — nothing is resolved by guessing.

```json
{
  "factors": [
    {"name": "ligand", "type": "categorical",
     "levels": ["CC(C)c1cc(C(C)C)...", "..."],
     "descriptors": {"CC(C)c1cc...": [1.2, 0.8, 3.4]}},
    {"name": "base", "type": "categorical", "levels": ["CCN(CC)CC", "CN(C)C=O"]},
    {"name": "temperature", "type": "continuous", "bounds": [20, 120], "unit": "C"},
    {"name": "loading", "type": "ordinal", "levels": [1, 2, 5, 10], "unit": "mol%"},
    {"name": "conc", "type": "continuous", "bounds": [0.01, 1.0],
     "transform": "log", "unit": "M"}
  ],
  "constraints": [{"weights": {"loading": 1.0}, "rhs": 10.0}]
}
```

### Factor types

| type | use for | needs |
|---|---|---|
| `continuous` | temperature, time, a real range | `bounds: [low, high]` |
| `integer` | whole-number counts | `bounds: [low, high]` |
| `ordinal` | 1, 2, 5, 10 mol% — a list where the **order means something** | `levels` |
| `categorical` | which ligand, which solvent — no order | `levels` |

Two fields that are easy to get wrong:

**`ordinal` vs `categorical`.** Catalyst loading at 1/2/5/10 mol% is a list, but
2 sits between 1 and 5. As `categorical` the model must rediscover that from
data. As `continuous` the tool will propose 3.7 mol%, which is not an option.

**`transform: "log"`.** For anything spanning decades. On a linear axis
0.01–1 M puts almost the whole range above 0.1 M and the bottom decade is never
explored.

### Normalisation report

Every chemical level is canonicalised with RDKit — deterministic, offline, no
LLM call and no network resolver. The response always contains:

```json
"normalisation": {
  "backend": "rdkit",
  "n_canonicalised": 1,
  "canonicalised": [{"factor": "ligand", "from": "C1=CC=CC=C1", "to": "c1ccccc1"}],
  "unresolved": [{"factor": "base", "value": "triethylamine",
                  "reason": "could not be parsed as SMILES ... passed through
                             verbatim and is NOT matched against other levels"}],
  "collisions": [],
  "needs_review": true
}
```

- **`canonicalised`** — rewritten to a canonical form. Descriptors move with the level.
- **`unresolved`** — a name, or malformed SMILES. **Passed through unchanged and
  flagged.** There is no offline name→structure source, and guessing is not done.

> **Why there is no IUPAC name resolution.** Resolving "triethylamine" to a
> structure requires either a name-to-structure service (the NIH/CACTUS resolver
> EDBO calls) or a language model. Both were ruled out: a network resolver is
> non-deterministic, unavailable offline, and silently returns different answers
> over time; an LLM call cannot be audited. A search space that resolves
> differently on two runs is not a search space. So names are flagged for a
> human rather than guessed at. If a deterministic local dictionary is added
> later it slots in here and the report gains a `resolved_from_name` section.
- **`collisions`** — two levels that denote the **same** structure. This is
  **refused**, because the space would silently be smaller than the caller thinks.

### Returns

`space_id`, the normalisation report, `cardinality` (null if any factor is
continuous), and `enumerable`.

---

## 2. `zero_shot_recommendation`

The opening experiments. No model exists yet; the job is to spread them out so
whatever gets fitted afterwards has something informative to learn from.

```json
{"space_id": "space_ab12...", "n": 5, "method": "random", "seed": 0}
```

Returns the picks and a `campaign_id` to carry forward.

| method | what it does |
|---|---|
| `random` | **default.** Uniform over the space |
| `kmedoids` | clusters the candidates, returns one real member of each cluster |
| `maxmin` | each pick is furthest from those already chosen |
| `sobol` | low-discrepancy, evenly spread. **Required when any factor is continuous** |

### What the benchmark says about this choice

`random` is the default for v1.5. On our own data it is not the best choice:

| method | beat random on | effect |
|---|---|---|
| **k-medoids** | **10 of 10** plate/surrogate combinations | 1–2.5 experiments better |
| Sobol | 7 of 10 | a wash |
| MaxMin | 2 of 10 | loses on 4 of 5 plates, by up to 19 experiments on one |

5 reactions × 50 seeds × 2 surrogates, 4,500 runs. Changing the default is a
one-parameter change.

---

## 3. `record_observations`

```json
{"campaign_id": "camp_...",
 "observations": [{"ligand": "...", "base": "...", "temperature": 95,
                   "loading": 2, "conc": 0.1, "yield": 73.4}]}
```

A `null` or `NaN` yield is **refused**, with the error pointing at
`skip_recommendation`. An experiment that was not run must not arrive carrying a
placeholder.

Returns per-observation `rejected` entries with reasons rather than failing the
whole batch.

---

## 4. `skip_recommendation`

A recommended experiment that nobody ran — no material, no time, or the chemist
judged it not worth doing.

```json
{"campaign_id": "camp_...", "picks": [{...}], "reason": "no material"}
```

**It is dropped from the training data entirely.** Not recorded as zero, not
filled with a predicted value, not included with inflated uncertainty. The model
ends up in exactly the state it would have been in had the experiment never been
proposed.

| approach | what goes wrong |
|---|---|
| record as 0% | the model learns that region is bad — we have no evidence of that |
| fill in a guess | the model treats its own prediction as a measurement |
| include, high uncertainty | still shifts the model, just more quietly |
| **leave it out** | **the model is exactly as if it was never proposed** |

### The conditions stay in the search space — this is the intended behaviour

A skip does **not** remove anything from the search space. Those conditions stay
eligible and **will** be recommended again when the maths says they are the best
thing to try.

That is the point, not a side effect. Because the model is unchanged by a skip,
the very next call will often propose the same conditions — correctly, since no
new information arrived, so the best next experiment has not changed. A tool
that quietly blacklisted a skipped condition would be throwing away the best
candidate on the strength of a scheduling problem.

`exclude_skipped: true` exists for the narrow case where the conditions are
genuinely unavailable — no material, instrument down — and proposing them again
wastes the chemist's time.

> **The flag is per call, never stored.** It affects only the request it is
> passed on. The next call without it sees the full space again, so a skipped
> condition can never become permanently blacklisted by accident.

If a deferred experiment is later actually run, `record_observations` moves it
out of the skipped list into the observations and keeps the deferral in
`deferral_history` for audit.

---

## 5. `next_recommendation`

**Stateful** — the normal case:

```json
{"campaign_id": "camp_...", "n": 1}
```

**Stateless** — nothing is stored:

```json
{"space": { ...the full space... },
 "observed": [{"ligand": "...", "...": "...", "yield": 73.4}, ...],
 "n": 1}
```

Both produce identical recommendations for identical inputs.

| acquisition | use |
|---|---|
| `ei` | **default.** Expected improvement |
| `qei` | sampled equivalent, for batches |
| `qnei` | when the same conditions can give different yields |
| `ts` | Thompson sampling, faster on very large spaces |

Expected Improvement does not ask which candidate is predicted highest. It asks
which has the best chance of beating the best result so far, using the
uncertainty — so a candidate predicted at 55% with wide uncertainty can beat one
predicted at 68% with high confidence, when the best so far is 70%.

Returns `picks`, `predicted_mean`, `predicted_sd`, `best_observed`,
`fit_failures`, and `notes` recording anything that was capped or sampled.

It also returns `n_candidates`, `n_excluded` and `excluded_skipped`, so a
caller can confirm from the response alone how many conditions were in play and
how many were held back. With the default `exclude_skipped: false`,
`n_excluded` counts only the experiments already run — skipped conditions are
not among them.

### Benchmarked choices

| decision | default | evidence |
|---|---|---|
| acquisition | `ei` | beats qEI by 1.23 and Thompson by 2.57 experiments |
| surrogate with descriptors | deep kernel | −2.02 experiments vs plain GP, CI [−3.09, −0.96] |
| batch size | caller's choice | every difference under 1 experiment |

---

## 6. `get_campaign` — resume

```json
{"campaign_id": "camp_..."}
```

Returns every observation, every skip, the best-so-far trace, and
`deferral_history`.

**Stop at experiment 6, restart the process, continue at 7.** A campaign is
completely described by `(space, observations, skipped, config, seed)`, and the
surrogate is refitted from that evidence on **every** call. There is no hidden
model state, so there is nothing else to restore.

This is deliberately different from persisting a fitted model object. Pickled
torch models break across library upgrades and make unpickling a security
question in a product. The JSON record is authoritative; a fitted-model cache
can be added later purely for speed.

---

## Storage

Local JSON by default, under `BEACON_STORE_DIR` (defaults to
`maths_tools/store_data/`). Writes are atomic — a crash cannot leave a torn
file.

The backend is pluggable: implement `put`, `get`, `list` on
`maths_tools.store.CampaignStore` and call `set_default_store()`. S3, Azure Blob
or Elasticsearch slot in behind the same three methods with no change to the
tools.

---

## Descriptors, and what happens without them

A categorical factor may carry `descriptors` — DFT values, fingerprints, any
numbers describing each option. With them, the deep kernel performs as
benchmarked. **Without them the factor is one-hot encoded**, which carries no
chemical similarity: the model cannot know two bulky phosphines behave alike.

This is a silent downgrade in quality, not an error, so it is worth deciding
deliberately rather than discovering later.

---

## What is validated, and what is not

**Benchmarked on real data with complete ground truth.** The metric is N95 —
the number of experiments needed before the best result so far reaches the
target. Lower is better. Protocol: `n_init=5`, `batch=1`, `budget=50`, 50 seeds.

> **The target is `0.95 × (best yield on the plate)`**, which is what the
> reference baselines use. It is *not* the 95th percentile — that is a much
> easier target (40 qualifying candidates per plate instead of 7–16) and
> produces N95 values roughly 6 experiments lower. `maths_tools/metrics.py`
> defines the target and verifies it against the reference protocol's source
> text on every benchmark run.

### Buchwald–Hartwig, 5 plates, 50 seeds

| plate | this tool | EDBO + trainable |
|---|---|---|
| bhR1 | 15.54 | 15.98 |
| bhR2 | 17.58 | 18.62 |
| bhR3 | 19.30 | 19.38 |
| bhR4 | **11.30** | 14.40 |
| bhR5 | **14.72** | 19.16 |
| **mean** | **15.69** | **17.51** |

About 1.8 experiments better on average, ahead on all five plates and by a
clear margin on two.

**Where that 1.8 comes from.** Each row below changes one thing from the row
above it, all five plates, 50 seeds:

| configuration | mean N95 | change |
|---|---|---|
| reference baseline, published | 17.51 | — |
| this tool, baseline configuration | 18.95 | +1.44 — run-to-run drift, not a real difference |
| dropout off, `clip_mode="code"` | 17.55 | **−1.40** |
| k-medoids instead of random initialisation | **15.69** | **−1.86** |

So the gain is two roughly equal parts: the projection settings and the
initialisation method. The +1.44 in the second row is the drift discussed
above — re-running the reference code itself moves it by a comparable amount —
so the honest reading is that this tool is about **1.8 experiments better, give
or take a point of run-to-run noise**.

### Harness validation

Configured to match the reference baseline exactly — same surrogate, same
projection dropout, same gradient clipping, same random initial design — this
tool reproduces the baseline:

| | mean N95 |
|---|---|
| reference baseline, published | 17.51 |
| this tool, baseline configuration | 18.95 |

The 1.4 difference is within run-to-run drift. Re-running the reference code
itself gives 17.72 on bhR1 where its published figure is 15.98 — Gaussian
process fitting is sensitive to the BLAS build, so stored numbers cannot be
diffed against fresh runs. **Any comparison between a stored figure and a new
one carries roughly ±1.5 experiments of drift**, and should be read with that
in mind.

### Direct arylation — a 5-factor mixed space

1,728 conditions: 4 bases × 12 ligands × 4 solvents × 3 concentrations × 3
temperatures. Run with **one-hot encoding only, no descriptors** — the case a
caller gets when they supply no `descriptors` field.

| | |
|---|---|
| mean N95, 20 seeds | **30.85** |
| median | 35 |
| reached target | **14 of 20** |
| target | 95.00 — only **10 of 1,728** conditions qualify |

This is a hard target: ten acceptable answers in 1,728, with a budget of 50.
Six runs in twenty never found one.

> **Not comparable to the published 19.04 for this dataset.** That baseline uses
> 291 DFT descriptors; this run used 22 one-hot columns. The gap measures the
> value of descriptors, not the quality of the optimiser. It is reported here
> as evidence the tool handles a five-factor mixed space end to end, and as a
> concrete illustration of the one-hot fallback cost described above.

A second run declared concentration and temperature as **continuous ranges**
rather than fixed levels completes the full 50 experiments and proposes
off-grid values that are snapped to the nearest measured point. That exercises
the continuous path on real data, but snapping does part of the work, so it is
a correctness check and not a performance number.

### Method choices, all measured

4,500 runs across 5 plates × 50 seeds × 2 surrogates:

| decision | default | evidence |
|---|---|---|
| initial design | k-medoids available, `random` shipped | k-medoids better than random on 10 of 10 plate/surrogate combinations |
| acquisition | `ei` | beats qEI by 1.23 and Thompson by 2.57 experiments |
| surrogate with descriptors | deep kernel | 2.02 experiments better than a plain GP, CI [0.96, 3.09] |
| batch size | caller's choice | every difference under 1 experiment |

These are relative comparisons within one protocol, so they are unaffected by
the target definition.

**Self-contained.** The deep-kernel model is vendored in
`maths_tools/deep_kernel.py` as a verbatim copy of the code that produced the
benchmark numbers, so the tool has no dependency on any benchmark directory.
`tests/test_vendored.py` guards the copy against drift.

**Implemented and unit-tested, but NOT benchmarked against continuous ground
truth:** the continuous path. On a test problem whose optimum is 80 °C it
returns 80.6 °C, and it runs end to end on direct arylation — but that dataset
has only 3 measured temperatures, so a proposal is snapped to the nearest
measured point. We have no dataset with a genuinely continuous response to
validate against.

Treat continuous-factor results as sound engineering, not as carrying the same
evidential weight as the categorical numbers.

---

## Errors

Bad input returns a readable message, never a stack trace:

```
ERROR search space is not valid: temperature: log transform requires a
strictly positive lower bound, got 0
```

`next_recommendation` needs at least 2 observations; below that it tells you to
use `zero_shot_recommendation`.

---

## Tests

```bash
python maths_tools/tests/run_all.py      # everything
python maths_tools/tests/test_mcp.py     # drives the real server over stdio
```
