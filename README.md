# decision-model-sssom-experiment

SSSOM mappings from System-One-style decision models (Kev, JevK5, Hopper), evaluated against curated
SNOMED CT -> Mondo mappings. (The CLI and local directory keep the historical name `kevmap` /
`kev-mapping-experiment`.)

Can a **System-One decision model** (Kev, an open Jev-like model) re-create curated
clinical-terminology → OBO mappings, with confidence scores you could act on? And does that ability
scale with model size in a way the experiment can actually see?

First target: **SNOMED CT disorders → Mondo**, because Mondo already curates ~9k `skos:exactMatch`
rows to SNOMED, which is the gold standard here. Output is a real SSSOM file per model.

## What Kev is, and why the pipeline looks like this

Kev (Jared Palmer, Apache-2.0; 0.8B / 4B / 9B / 27B, Qwen3.5-Base + LoRA + pointer head) serves
TypeSafe's `/v1/systemone` contract: one *state* document plus any number of typed questions
(`choice` over named options, `noul` yes/no, `score` ordinal), answered in **one prefill pass, no
generation**, with calibrated probabilities. It cannot propose a Mondo ID; it can only decide among
options it is shown. So the pipeline is **retrieve, then decide**:

1. **Retrieve** k=8 Mondo candidates for each SNOMED label (BM25 over Mondo labels + exact
   synonyms, rapidfuzz rerank). Retrieval recall@k is reported on its own so a retrieval miss is
   never charged to the model.
2. **Decide** with one Kev request per source concept. The state holds the SNOMED concept
   (preferred term, synonyms, text definition, parent concepts) and every candidate (label,
   synonyms, definition). Questions:
   - `match`: *Which candidate denotes the same disease as the source? (or `none`)* → `object_id`,
     and its probability is the SSSOM `confidence`.
   - `rel_<X>` for each candidate: *same / narrower / broader / related / unrelated* → the
     `predicate_id` (`same`→exactMatch, source-`narrower`→broadMatch, `broader`→narrowMatch,
     `related`→relatedMatch), plus an independent per-candidate p(same).

## How the evaluation distinguishes model quality

Accuracy on easy cases saturates fast and would hide the size effect, so items are built in tiers
with known correct answers and scored separately:

| tier | construction | what it measures |
|---|---|---|
| **easy** | gold is in the lexical top-k; candidates are whatever retrieval returned | can it pick the obvious answer |
| **hard** | gold is in the top-k, but the weakest lexical candidates are replaced by the gold's **Mondo parent, children and siblings** | can it tell a disease from its near relatives (where small models should fail) |
| **none-narrower** | SNOMED disorder with **no** Mondo mapping whose is_a **parent is mapped**; the parent's Mondo term is forced into the candidates | does it abstain instead of over-mapping to the broader term, and does it call the relation `narrower` |
| **gold-removed** | an ordinary mapped item with the gold deleted from the candidate list | the cleanest abstention test: `none` is the only right answer |
| **retrieval-miss** | gold exists but the retriever did not surface it | does it say `none` when the truth is absent |

Metrics per tier and overall (`results/<model>/metrics.json`, `results/comparison.md`):

- **acc** on `match` (argmax vs the gold option, or `none` where that's correct)
- **brier** and **ece** of the reported probabilities — is the confidence honest
- **coverage@5**: the share of decisions you could auto-accept, taking them in order of confidence,
  before the error rate exceeds 5%. This is the number that matters for a production mapping
  pipeline and the one System-One models are built to optimise
- **rel_acc**: predicate correct where the truth is known: gold → `same`; injected Mondo parent or the
  mapped parent → `narrower`; injected child → `broader`; injected sibling → no-map (`related` or
  `unrelated`: neither yields a mapping row). Reported per truth class (`rel_acc_by_truth`) with the raw
  5-way confusion (`rel_confusion`), so exact/broad/narrow/no-map are each visible
- **fp_same**: share of non-gold candidates the model nonetheless calls `same` (over-mapping)

Two runs are compared with Kev's own `kev.compare` (paired bootstrap CIs) when we want error bars.

## Why the result can be trusted

- **Gold is external and curated.** Mondo's SCTID mappings are maintained by Mondo curators, not
  derived from anything in this pipeline. Only Mondo terms that are current (non-obsolete in the
  2026-05-05 release) and SNOMED ids with exactly one Mondo exact match are used.
- **The model never sees the answer.** Candidate option names are letters; the gold sits among
  lexically retrieved or ontology-neighbour distractors and is not marked in any way.
  Candidate order is retrieval rank for lexical hits; Mondo-neighbour injections take the slots
  of the weakest lexical hits (known confound: position bias; Kev's `/v1/systemone/permute`
  endpoint exists to test it).
- **Retrieval and decision are scored separately.** recall@k is printed at build time; the
  `retrieval-miss` tier isolates the cases where the truth was absent.
- **Negatives are real, not synthetic.** `none-narrower` items are actual SNOMED disorders Mondo
  hasn't mapped, with the tempting broader term present. The caveat is that "Mondo hasn't mapped it"
  is not proof that no exact Mondo term exists; a model that "wrongly" picks a candidate there may
  be surfacing a missing curation, so those disagreements are worth reading, not just counting.
- **Same items, same seed, every model.** `data/items.jsonl` is built once (`--seed 7`) and
  reused across sizes; only the served checkpoint changes. Raw responses are kept so metrics can be
  recomputed without re-running the model.
- **Confidence is evaluated, not assumed.** Brier, ECE and coverage@5 test whether the probabilities
  mean what they claim, which is the whole point of preferring a System-One model over sampling
  a chat model.

## Other System-One models

The runner only needs a `/v1/systemone` endpoint, so any Jev-compatible model slots in as another
`--model` name. From the [JevBench leaderboard](https://benchmarkheaven.com/jev-models) (v1.4.1):

| model | base | license | runs here? | JevBench calibration | note |
|---|---|---|---|---|---|
| Kev 0.8B/4B/9B | Qwen3.5-Base + LoRA + pointer head | Apache-2.0 | yes (MLX) | 39.6 (4B) | shared-prefix: all questions in one pass |
| [JevK5](https://github.com/allebee/jevk5) 4B / 2B | Qwen3.5-4B + LoRA, letter readout | Apache-2.0 | yes (MPS/CPU/GGUF) | 74.5 | #2 overall, #1 open; <=16 options per pass, one pass per question |
| [Hopper](https://github.com/hopit-ai/hopper) | Qwen3.5-4B + LoRA, letter readout | code Apache-2.0, weights research/demo only | CUDA only -> LongLeaf | 79.1 | eval only; not for published mappings |
| Jev 1.13.0 | closed | TypeSafe API | no accounts open | 76.3 | the reference |

## Results so far (2026-09-24, Kev sizes on LongLeaf L40S; Hopper/JevK5 pending)

Original set (`items`, retrieval-rank order, n=1,500) and the shuffled / 50%-lexically-ambiguous set
(`items-v2`, n=1,700, adds the gold-removed tier). `lexical` = take the retriever's top-1 (never abstains).

| set | model | acc | lexical | ECE | cov@5 | cov@10 | prec@conf>=.9 (n) | gold-removed | none-narrower | retr-miss | rel_acc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| items | kev-0.8b | 0.736 | 0.692 | 0.213 | 0.061 | 0.382 | - (0) | - | 0.260 | 0.447 | 0.715 |
| items | kev-4b | 0.877 | 0.692 | 0.046 | 0.551 | 0.909 | 0.950 (814) | - | 0.573 | 0.682 | 0.954 |
| items | kev-9b | 0.876 | 0.692 | 0.057 | 0.628 | 0.922 | 0.943 (1003) | - | 0.577 | 0.682 | 0.956 |
| items-v2 | kev-0.8b | 0.644 | 0.423 | 0.179 | 0.029 | 0.065 | - (0) | 0.550 | 0.357 | 0.467 | 0.527 |
| items-v2 | kev-4b | 0.804 | 0.423 | 0.077 | 0.126 | 0.472 | 0.900 (781) | 0.760 | 0.523 | 0.687 | 0.871 |
| items-v2 | kev-9b | 0.825 | 0.423 | 0.115 | 0.009 | 0.429 | 0.895 (839) | 0.820 | 0.543 | 0.731 | 0.888 |

Findings:

- **The 0.8B -> 4B step is the one that matters.** 9B - 4B: -0.001 [-0.013, +0.012] on `items`;
  +0.021 [+0.009, +0.034] on `items-v2`, entirely from abstention (gold-removed +0.06, retrieval-miss
  +0.04), at 1.7x the latency.
- **The first `items` set mostly measured the retriever**: the gold sat at option A 93% of the time and
  lexical top-1 alone scored 0.93 on the mapped tiers. Shuffling and oversampling lexically ambiguous
  sources (`items-v2`) fixed that: lexical drops to 0.42, Kev-4B holds 0.80 (0.81 on the ambiguous
  half vs 0.29 for string matching).
- **Abstention is where models separate**: with the gold deleted, 0.8B says `none` 55%, 4B 76%, 9B 82%.
  Offered the broader Mondo term for an unmapped SNOMED child, 4B/9B abstain ~52-58% but call the
  relation `narrower` 85-91% of the time, i.e. they produce the `skos:broadMatch` row Mondo lacks.
- **Predicates (4B, items-v2)**: same 0.94, narrower 0.85, no-map 0.87 (siblings are called `unrelated`,
  not `related`; fine for mapping), broader 0.45 (children are the weak class). 0.8B's predicate answers
  are mostly a default (`broader` for 64% of `narrower` truths).
- **Wider candidate lists cost accuracy and calibration** (4B: 0.80 at k=8, 0.69 at k=64, 0.63 at
  k=250; ECE 0.08 -> 0.30; gold-removed abstention 0.76 -> 0.34), while staying above lexical (0.42).
  The wide sets are labels-only, so the k=8 labels-only control (`k8lab`) separates "more options" from
  "less context".
- **MLX (Mac) and CUDA (L40S) agree**: 0.8B 99.1% same top choice (mean |dp| 0.006), 4B 99.8%
  (mean |dp| 0.003, max 0.055).

## Layout

```
justfile              install / kev-install / serve SIZE / build / run MODEL / eval / sssom MODEL
src/kevmap/
  mondo.py            Mondo term table from the cached semsql sqlite (~/.data/oaklib/mondo.db)
  snomed.py           SNOMED disorder table straight from RF2 (~/Monarch/bdc/snomed/…)
  gold.py             Mondo's SCTID rows from mondo.sssom.tsv (medic-ingest/data)
  retrieve.py         BM25 + rapidfuzz candidate retrieval
  items.py            tiered eval items -> data/items.jsonl
  questions.py        item -> /v1/systemone request; relation -> SSSOM predicate
  run.py              resumable runner -> results/<model>/responses.jsonl
  evaluate.py         metrics.json, per_item.tsv, results/comparison.md
  sssom_out.py        results/<model>/mappings.sssom.tsv
kev/                  gitignored clone of github.com/jaredpalmer/kev with its own uv env
```

## Running it

```
just install && just kev-install
just build                     # ~1 min; prints tier counts and retrieval recall@k
just serve 0.8b                # separate terminal; first run downloads Qwen3.5-0.8B-Base
just run kev-0.8b              # resumable; Ctrl-C and re-run to continue
just serve 4b  && just run kev-4b
just serve 9b  && just run kev-9b    # ~22 GB resident: make sure omlx has nothing loaded
just eval                      # results/comparison.md
just sssom kev-4b              # results/kev-4b/mappings.sssom.tsv
```

Sizes and this machine (M1 Max, 32 GB): 0.8B and 4B (~9 GB bf16) are comfortable; 9B (~19 GB
weights, ~22 GB with batching buffers) works only with nothing else large resident; 27B has no Mac
path and would run on LongLeaf (`a100-gpu`).

**Memory finding (2026-09-24).** Plain `kev.serve` on the MLX backend grows without bound: the
0.8B server reached 25 GB after a dozen requests and pushed the machine into swap, with per-item
latency going from 0.7 s to 20 s. MLX caches Metal buffers by shape and nothing in Kev releases
them; every request has a new sequence length. `scripts/serve_kev.py` (what `just serve` runs)
caps MLX's buffer cache (`KEVMAP_MLX_CACHE_GB`, default 0.5) and clears it after every batch; the
0.8B server then sits flat at 2.1 GB. `kevmap run` also polls the server's RSS and aborts above
`KEVMAP_MAX_SERVER_GB` (default 18). Runs use one worker: the Metal path serves one request at a
time anyway.

## Cluster leg (LongLeaf)

`cluster/sync.sh` pushes code + item sets to `/work/users/k/s/kschaper/kev-mapping-experiment`;
`cluster/setup.sbatch` (CPU, `general`) builds the runner, Hopper, JevK5 and Kev CUDA envs and
pre-downloads weights to `/work/.hf`; `cluster/run_models.sbatch` (`l40-gpu`, 1x L40S 48 GB) serves
each model on the node in turn and runs every item set; `cluster/sync.sh --pull` brings
`results/<model>/<items>/responses.jsonl` back for `kevmap eval`. All Kev sizes, Hopper and JevK5 run
there; the Mac MLX run of kev-4b on `items` is kept as the MLX-vs-CUDA parity check.

## Licensing: what this repo does and does not contain

SNOMED CT is licensed content (SNOMED International Affiliate License, via the NLM UMLS license in the
US). This repo therefore contains **code and scores only**; it never redistributes SNOMED text:

- `data/` (term tables, item sets, training records: labels, synonyms, definitions, hierarchy) and
  `runs/` (adapters fine-tuned on that text) are gitignored and stay on the machine that holds the
  RF2 release. Reproducing the experiment needs your own SNOMED CT download (`paths.py`).
- Emitted SSSOM files carry `SCTID` identifiers with an empty `subject_label`, exactly as Mondo's own
  SCTID mappings do; Mondo labels are OBO (CC-BY 4.0) and stay.
- `results/<model>/<set>/per_item.tsv` and `metrics.json` reference items by SCTID only.
- Whether an adapter fine-tuned on SNOMED labels may be published is unresolved; treat the checkpoints
  as local until SNOMED International / NLM say otherwise. An ICD-10-CM-trained adapter (CMS, public
  domain) would carry no such restriction, which is one more reason to run that transfer next.
- Hopper's weights are research/demo-only (RACE terms); JevK5 and Kev are Apache-2.0.

## Inputs and versions

| input | where | version |
|---|---|---|
| Mondo SSSOM (gold) | `../medic-ingest/data/mondo.sssom.tsv` | Mondo release as downloaded by medic-ingest |
| Mondo terms | `~/.data/oaklib/mondo.db` (semsql) | obo:mondo/releases/2026-05-05 |
| SNOMED CT | `~/Monarch/bdc/snomed/…US1000124_20260301…/Snapshot` | US Edition 2026-03-01 |

Open item: pin gold SSSOM and ontology to the same Mondo GitHub release (`mondo.sssom.tsv` and
`mondo.db` are both release artefacts) so no gold row points at a term that moved.

## Phase 2: fine-tuning (started 2026-09-24 22:28, overnight on the Mac)

`kevmap train-data` builds `data/train.jsonl`: 4,000 labelled records in Kev's training format (an
`items-v2`-style item plus `label`s: the gold option or `none` for `match`, and the known relation for
each candidate whose relation is known), from sources **disjoint from every eval set** (3,469 sources
excluded; the builder asserts no overlap). `scripts/overnight_train.sh` fine-tunes Kev-0.8B from
`--init_from jaredpalmer/kev-0.8b` (`--max_state 1536`, since the default 384-token state limit drops
our ~1,100-token states), then serves the checkpoint and scores `items-v2`, `k8lab`, `items` as model
`kev-0.8b-snomed`. MPS has no DeltaNet kernels, so training runs at ~17 s/record: the overnight run is
1,500 records x 1 epoch. Question: does a mapping-specific 0.8B beat the stock 4B? 4B fine-tuning
(peak 24.6 GB) goes to the L40S (`cluster/`), not the Mac.

### The 4B class: picking vs abstaining (`items-v2`, 2026-09-25)

Same Qwen3.5-4B base, four training recipes, plus the Mac-fine-tuned 0.8B. Every model runs the identical
items; Hopper and JevK5 take one question per request (their servers' rule) and cannot run the wide sets.

| model | acc | gold present | gold absent | none P / R | ECE | cov@10 | rel_acc | ms/item |
|---|---|---|---|---|---|---|---|---|
| kev-4b | 0.804 | 0.927 | 0.640 | 0.96 / 0.64 | 0.077 | 0.472 | 0.871 | 111 |
| kev-9b | **0.825** | **0.935** | 0.678 | 0.96 / 0.68 | 0.115 | 0.429 | **0.888** | 192 |
| jevk5 (4B) | 0.779 | 0.846 | 0.689 | 0.83 / 0.69 | **0.069** | 0.477 | 0.801 | 655 |
| hopper (4B) | 0.754 | 0.920 | 0.532 | 0.96 / 0.53 | 0.073 | 0.135 | 0.840 | (n/a) |
| kev-0.8b-snomed | **0.825** | 0.775 | **0.891** | 0.78 / 0.89 | 0.103 | **0.724** | 0.865 | 600 (MLX) |

Reading: Kev-4B/9B are the balanced choice. Hopper is the strongest picker (ties Kev on the original set
at 0.978 with the best calibration there, ECE 0.031) but rarely abstains. JevK5 abstains more but picks
worse and is ~6x slower. The mapping-fine-tuned 0.8B is an abstainer that has not learned to pick.
Hopper's weights are research/demo-only, so it is a reference point, not a production option.

### Phase 2 result: 4B fine-tune (L40S, 2026-09-25, 3.2 h, 1.46 s/record)

| set | model | acc | gold present | gold absent | none P / R | ECE | cov@10 | prec@>=.9 (n) | rel_acc |
|---|---|---|---|---|---|---|---|---|---|
| items-v2 | **kev-4b-snomed-full** | **0.894** | 0.891 | **0.897** | 0.91 / 0.90 | **0.073** | **0.972** | 0.914 (**1559**) | **0.925** |
| items-v2 | kev-0.8b-snomed-full | 0.872 | 0.866 | 0.879 | 0.88 / 0.88 | 0.087 | 0.896 | 0.903 (1487) | 0.912 |
| items-v2 | kev-4b (stock) | 0.804 | **0.927** | 0.640 | 0.96 / 0.64 | 0.077 | 0.472 | 0.900 (781) | 0.871 |
| items | **kev-4b-snomed-full** | **0.933** | 0.954 | **0.873** | 0.89 / 0.87 | **0.041** | **1.000** | 0.949 (**1411**) | **0.970** |
| items | kev-4b (stock) | 0.877 | **0.973** | 0.597 | 0.94 / 0.60 | 0.046 | 0.909 | 0.950 (814) | 0.954 |
| k8lab | kev-4b-snomed-full | 0.780 | 0.629 | **0.947** | 0.72 / 0.95 | 0.109 | 0.402 | 0.861 (368) | 0.854 |
| k8lab | kev-4b (stock) | 0.790 | **0.822** | 0.754 | 0.89 / 0.75 | 0.127 | 0.472 | 0.952 (42) | 0.783 |

Fine-tuned 4B - fine-tuned 0.8B: +0.022 [+0.012, +0.032] (`items-v2`), +0.019 [+0.010, +0.028] (`items`),
-0.025 [-0.053, +0.000] (`k8lab`). Fine-tuned 4B - stock 4B: +0.089 [+0.071, +0.107] on `items-v2`.

**Conclusions (2026-09-25).**

1. Fine-tuning on ~4k in-domain decisions is worth more than any size step: fine-tuned 0.8B > stock 9B,
   fine-tuned 4B > everything, on the definition-bearing sets. On `items-v2` the fine-tuned 4B would
   auto-accept 92% of decisions at 91% precision (coverage@10 = 0.97).
2. The 4B-over-0.8B margin after fine-tuning is small (+2 pts) for ~5x the latency; the 0.8B is the
   practical engine unless the last points matter.
3. Both fine-tunes trade a little picking for a lot of abstention: gold-present accuracy 0.927 -> 0.891
   (4B). `--replay` (mix in Kev's own training data) is the trainer's tool for this and is untried.
4. Both fine-tunes collapse on **labels-only** inputs (`k8lab`: 4B picking 0.822 -> 0.629), because every
   training state carried synonyms/definitions. Next training set should mix labels-only states.
5. Everything here is in-distribution (train and eval both from Mondo's SNOMED mappings, disjoint
   sources). ICD-10-CM -> Mondo, with no ICD training data, is the transfer test to run next.

### Phase 2 result (full-set 0.8B fine-tune on the L40S, 2026-09-25)

`kev-0.8b-snomed-full`: Kev-0.8B from `--init_from jaredpalmer/kev-0.8b`, all 4,000 training records,
2 epochs, 70 min on one L40S (0.52 s/record vs 21 s on the Mac). **Best model on every set**:

| set | model | acc | gold present | gold absent | none P / R | ECE | cov@10 | prec@>=.9 (n) | rel_acc |
|---|---|---|---|---|---|---|---|---|---|
| items-v2 | kev-0.8b-snomed-full | **0.872** | 0.866 | **0.879** | 0.88 / 0.88 | 0.087 | **0.896** | 0.903 (**1487**) | **0.912** |
| items-v2 | kev-9b (best stock) | 0.825 | **0.935** | 0.678 | 0.96 / 0.68 | 0.115 | 0.429 | 0.895 (839) | 0.888 |
| items-v2 | kev-4b | 0.804 | 0.927 | 0.640 | 0.96 / 0.64 | 0.077 | 0.472 | 0.900 (781) | 0.871 |
| items | kev-0.8b-snomed-full | **0.914** | 0.935 | **0.855** | 0.84 / 0.86 | 0.055 | **1.000** | 0.941 (**1380**) | **0.953** |
| items | kev-4b | 0.877 | **0.973** | 0.597 | 0.94 / 0.60 | 0.046 | 0.909 | 0.950 (814) | 0.954 |
| k8lab | kev-0.8b-snomed-full | **0.805** | 0.711 | **0.909** | 0.76 / 0.91 | 0.122 | **0.655** | 0.868 (469) | **0.853** |
| k8lab | kev-4b | 0.790 | **0.822** | 0.754 | 0.89 / 0.75 | 0.127 | 0.472 | 0.952 (42) | 0.783 |

+0.047 [+0.035, +0.059] over the Mac's 1,491 x 1-epoch fine-tune on `items-v2`, and unlike that run it
improved picking as well as abstention. Caveats: pure picking still trails stock 4B (0.866 vs 0.927);
the labels-only set transfers worse (training states carried definitions); train and eval sources
are disjoint but both come from Mondo's SNOMED mappings, so this is in-distribution: transfer to
ICD-10-CM / LOINC is the next test. At ~25 ms/item on a GPU (and ~0.7 s on the Mac via MLX) it is the
practical candidate for a mapping engine. The 4B fine-tune (same recipe) is the pending comparison.

### Phase 2 result (Mac fine-tune, 2026-09-25)

`kev-0.8b-snomed` (1,491 records, 1 epoch, 8.7 h on MPS) on `items-v2`, next to the stock sizes:

| model | acc | acc, gold present | acc, gold absent | none rate (truth 0.43) | none P / R | ECE | cov@10 | rel_acc |
|---|---|---|---|---|---|---|---|---|
| kev-0.8b | 0.644 | 0.793 | 0.444 | 0.23 | 0.84 / 0.44 | 0.179 | 0.065 | 0.527 |
| **kev-0.8b-snomed** | **0.825** | 0.775 | **0.891** | 0.49 | 0.78 / 0.89 | 0.103 | **0.724** | 0.865 |
| kev-4b | 0.804 | **0.927** | 0.640 | 0.29 | **0.96** / 0.64 | 0.077 | 0.472 | 0.871 |
| kev-9b | 0.825 | 0.935 | 0.678 | 0.30 | 0.96 / 0.68 | 0.115 | 0.429 | 0.888 |
| jevk5 | 0.779 | 0.846 | 0.689 | 0.36 | 0.83 / 0.69 | 0.069 | 0.477 | 0.801 |

+0.181 [+0.157, +0.205] over stock 0.8B, matching stock 9B in aggregate, but the gain is **abstention
and predicates, not discrimination**: given the gold is present it still picks it only 0.775 of the
time (4B: 0.927); it answers `none` 49% of the time at 0.78 precision, so about a fifth of its
abstentions drop a mappable term; predicates reach near-4B level (same 0.91, narrower 0.85, no-map
0.92) except `broader` (0.16, children are rare in training). Part of the aggregate gain is the eval's
43% none-rate rewarding a model that abstains readily. Follow-ups: the 4B fine-tune on the L40S
(`cluster/finetune.sbatch`, full 4,000 records x 2 epochs) asks whether abstention can be added to
a model that already discriminates; `acc_gold_present`, `none_precision`, `none_recall` are now in
every metrics table so this cannot hide again.

## Later phases (not built)
- **ICD-10-CM → Mondo** as the fully redistributable twin of this experiment (2.1k gold rows,
  CMS tabular list free).
- **Option-order robustness** via `/v1/systemone/permute`.
- A plain-LLM baseline over the same items (deliberately out of scope for the first pass).
