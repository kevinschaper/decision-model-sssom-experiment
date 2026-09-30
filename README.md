# decision-model-sssom-experiment

Decision models answer a typed question about a document in one forward pass and return a
probability for every option; they generate no text. This experiment asks whether such models can
reproduce curated **SNOMED CT → Mondo** mappings as SSSOM, with confidence scores a curator could act
on. Seven models answer identical requests: **Jev** (TypeSafe's hosted reference), **Kev** 0.8B / 4B /
9B (open, stock and fine-tuned on eval-disjoint SNOMED→Mondo decisions), **JevK5** 4B / 2B, and
**Hopper** 4B. The gold standard is Mondo's own ~9k `skos:exactMatch` rows to SNOMED.

## Summary

| model | training | acc | gold present | gold absent | none P / R | ECE | cov@10 | rel_acc |
|---|---|---|---|---|---|---|---|---|
| Jev (TypeSafe, hosted) | zero-shot | 0.881 | 0.930 | 0.814 | 0.95 / 0.81 | 0.066 | 0.939 | 0.901 |
| Kev-4B fine-tuned | SNOMED-tuned | 0.894 | 0.891 | 0.897 | 0.91 / 0.90 | 0.073 | 0.972 | 0.925 |
| Kev-0.8B fine-tuned | SNOMED-tuned | 0.872 | 0.866 | 0.879 | 0.88 / 0.88 | 0.087 | 0.896 | 0.912 |
| Kev-9B | zero-shot | 0.825 | 0.935 | 0.678 | 0.96 / 0.68 | 0.115 | 0.429 | 0.888 |
| Kev-4B | zero-shot | 0.804 | 0.927 | 0.640 | 0.96 / 0.64 | 0.077 | 0.472 | 0.871 |
| JevK5 (4B) | zero-shot | 0.779 | 0.846 | 0.689 | 0.83 / 0.69 | 0.069 | 0.477 | 0.801 |
| Hopper (4B) | zero-shot | 0.754 | 0.920 | 0.532 | 0.96 / 0.53 | 0.073 | 0.135 | 0.840 |
| JevK5-2B | zero-shot | 0.673 | 0.717 | 0.613 | 0.70 / 0.61 | 0.152 | 0.112 | 0.342 |
| Kev-0.8B | zero-shot | 0.644 | 0.793 | 0.444 | 0.84 / 0.44 | 0.179 | 0.065 | 0.527 |
| lexical top-1 (retriever alone) | none | 0.423 | 0.714 | 0.000 | never abstains | | | |

*Hard evaluation set (`items-v2`): candidate order shuffled, half the sources lexically ambiguous,
n = 1,700. `gold present`: picks the curated match when it is offered. `gold absent`: answers `none`
when it should. `cov@10`: the share of decisions a curator could auto-accept, taken in confidence
order, before the error rate passes 10%. `rel_acc`: predicate accuracy (exact / broad / narrow /
no-map).*

1. **Four thousand in-domain decisions beat model size.** Kev‑0.8B, fine-tuned on eval-disjoint
   SNOMED→Mondo records, outscores stock Kev‑9B on every set; Kev‑4B, fine-tuned the same way, becomes
   the best model in-distribution.
2. **Jev is the strongest zero-shot model and the most robust.** It picks as well as stock Kev and
   also abstains (none recall 0.81 at 0.95 precision); it trails the fine-tuned 4B by one point
   in-distribution (+0.013 [+0.001, +0.025]) and leads it by six points on labels-only inputs.
3. **Abstention separates the models.** Zero-shot models pick when they should abstain; the
   SNOMED-tuned models abstain when they should pick. Coverage at a 10% error budget ranges from
   0.07 (stock 0.8B) to 0.97 (fine-tuned 4B).
4. **The retriever alone solves the easy cases.** With candidates in retrieval order, lexical top‑1
   scores 0.93 on mapped items, so the shuffled, ambiguous set is the evaluation that matters.
5. **Most shared misses are gold errors.** Where ≥ 6 of 7 models agree against Mondo (230 items, 98 at
   p ≥ 0.9), Mondo usually maps a general SNOMED concept to a numbered genetic subtype or a sibling.
   `results/consensus-disagreements.tsv` lists them as a curation queue.

Cost: Jev scored every set for $0.53; the open models ran on one L40S GPU (Kev at 25–190 ms per
item). Details follow; reproduction and licensing are at the end. **SNOMED CT text never appears in
this repository.**

## Method

### Retrieve, then decide

A decision model cannot propose a Mondo term; it can only choose among terms it is shown. Each SNOMED
disorder therefore gets **eight Mondo candidates** from a lexical retriever (BM25 over labels and
exact synonyms, rapidfuzz rerank) and then **one request** whose state holds the SNOMED concept
(preferred term, synonyms, definition, parents) and every candidate (label, synonyms, definition).
The request asks:

- `match`: which candidate denotes the same disease, or `none`. The chosen option becomes the SSSOM
  `object_id`; its probability becomes `confidence`.
- `rel_<X>`, one per candidate: same / narrower / broader / related / unrelated. The relation of the
  chosen candidate becomes the predicate (`exactMatch`, `broadMatch`, `narrowMatch`, `relatedMatch`).

Retrieval recall@8 (0.93) is reported on its own, so a retrieval miss is never charged to the model.
All models are called through the same `/v1/systemone` contract; Hopper and JevK5 accept one question
per request, so the runner splits the request for them.

### Evaluation design

Accuracy on easy items saturates and hides the differences between models, so the items are built
in tiers with known answers and scored separately:

| tier | construction | what it tests |
|---|---|---|
| easy | gold among ordinary lexical candidates | picking the obvious answer |
| hard | the gold's Mondo parent, children and siblings replace the weakest lexical candidates | telling a disease from its relatives |
| gold-removed | a mapped item with the gold deleted | abstaining when the answer is absent |
| none-narrower | an unmapped SNOMED child whose parent is mapped; the parent's Mondo term is offered | abstaining from the broader term, and calling the relation `narrower` |
| retrieval-miss | the retriever did not surface the gold | abstaining when the truth is absent |

Three controls guard against measuring the wrong thing. **Candidate order is shuffled** (`items-v2`,
`k8lab`, `wide*`); in retrieval order the gold sat at option A 93% of the time. **Half the sources
are lexically ambiguous** (the retriever's top‑1 is wrong, or its top two nearly tie). **Labels-only
and wide-list variants** (`k8lab` at k = 8; `wide64`, `wide250`) separate "more options" from "less
context".

Metrics: accuracy overall and per tier; accuracy with the gold present and with it absent; precision
and recall of `none`; Brier score and expected calibration error; coverage at 5% and 10% error
budgets; precision of the ≥ 0.9-confidence slice; predicate accuracy, per truth class, with the raw
confusion. The lexical top‑1 baseline (which never abstains) appears in every table. Model pairs are
compared with a paired bootstrap over items (`kevmap compare`).

The gold is Mondo's own curation, restricted to current terms and to SNOMED ids with exactly one
exact match. The model sees no identifier and no mark on the gold. The fine-tunes train on sources
absent from every evaluation set; the builder asserts the disjointness.

## Results

### Size, stock Kev

On the hard set, accuracy rises from 0.644 (0.8B) to 0.804 (4B) and 0.825 (9B); calibration improves
with the first step only (ECE 0.18 → 0.08 → 0.12). The 9B's advantage over 4B is small and confined
to abstention (+0.021 [+0.009, +0.034]; gold-removed +0.06), for 1.7× the latency. Neither abstains
well: with the gold absent they answer `none` 64–68% of the time.

### Width of the candidate list

Widening the list costs accuracy, calibration and abstention. Kev‑4B falls from 0.804 (k = 8) to
0.685 (k = 64) and 0.632 (k = 250); its ECE rises from 0.08 to 0.30; its gold-removed abstention
falls from 0.76 to 0.34. The labels-only k = 8 control scores 0.790, so the loss comes from the
option count, not the missing definitions. Jev holds 0.80 at k = 64 and 0.79 at k = 250 (ECE 0.06,
0.07), leading stock 9B by +0.120 [+0.083, +0.157] at k = 250.

### Same base, four recipes

Kev‑4B, JevK5 and Hopper share the Qwen3.5‑4B base and differ in training. They land on a
picking-versus-abstaining trade-off: Hopper picks best (0.92 with the gold present) and abstains
least (0.53); JevK5 abstains more (0.69) and picks worse (0.85); stock Kev‑4B sits between. Jev does
both (0.93 / 0.81) and is the best calibrated (ECE 0.066). JevK5 and Hopper run one question per
pass and cost 6× Kev's latency; Hopper's weights are research-only.

### Fine-tuning

Kev's trainer takes the evaluation request plus a `label` per question and starts from the released
adapter (`--init_from`). Training records mirror `items-v2` (4,000 records; gold option or `none`
for `match`, the known relation for each candidate) from sources absent from every evaluation set.

| model | acc | gold present | gold absent | none P / R | ECE | cov@10 | rel_acc |
|---|---|---|---|---|---|---|---|
| Kev‑4B fine-tuned | **0.894** | 0.891 | **0.897** | 0.91 / 0.90 | 0.073 | **0.972** | **0.925** |
| Kev‑0.8B fine-tuned | 0.872 | 0.866 | 0.879 | 0.88 / 0.88 | 0.087 | 0.896 | 0.912 |
| Kev‑4B stock | 0.804 | **0.927** | 0.640 | 0.96 / 0.64 | 0.077 | 0.472 | 0.871 |
| Kev‑0.8B stock | 0.644 | 0.793 | 0.444 | 0.84 / 0.44 | 0.179 | 0.065 | 0.527 |

Fine-tuning taught abstention and predicates more than discrimination: the fine-tuned 4B abstains
correctly 0.90 of the time (stock 0.64) but picks the present gold 0.891 (stock 0.927). Fine-tuned 4B
beats fine-tuned 0.8B by +0.022 [+0.012, +0.032] at five times the latency. Both fine-tunes collapse
on labels-only inputs (`k8lab`: 4B picking 0.822 → 0.629) because every training state carried
definitions; the next training set should mix labels-only states, and Kev's `--replay` option can
retain general skill. On the hard set the fine-tuned 4B gets the predicate right for same 0.95, narrower 0.92 and no-map
0.95 of candidates, but broader only 0.34. `broader` (the source is a parent of the candidate) is the
weakest class for every model; Jev scores highest on it, at 0.60.

### Jev

Jev answers the unchanged requests at ~2.8k input tokens per item, so all five sets cost $0.53. It is
the best model on every set the fine-tunes were not trained for:

| set | Jev | best open model | |
|---|---|---|---|
| `items-v2` | 0.881 | 0.894 | Kev‑4B fine-tuned |
| `items` | 0.921 | 0.933 | Kev‑4B fine-tuned |
| `k8lab` (labels only) | **0.840** | 0.805 | Kev‑0.8B fine-tuned |
| `wide64` | **0.802** | 0.692 | Kev‑9B |
| `wide250` | **0.790** | 0.670 | Kev‑9B |

Fine-tuned 4B minus Jev: +0.013 [+0.001, +0.025] on `items-v2`, −0.060 [−0.097, −0.027] on `k8lab`.

### Where the models miss

On the hard set 108 items defeat all seven models and 1,073 defeat none; miss sets overlap by recipe
(the two fine-tunes agree with each other, the stock Kevs with Hopper, Jev between). Reading the
shared misses shows that Mondo, not the models, is usually wrong: SCTID:34000006 is mapped to
*inflammatory bowel disease 1* while every model chooses *Crohn disease*; SCTID:118601006 to
*lymphoma, non‑Hodgkin, familial* (*non‑Hodgkin lymphoma*); SCTID:127004000 to *lacrimal gland
cancer* (*lacrimal gland neoplasm*); SCTID:406506008 to *ADHD, inattentive type* (*ADHD*). On the
none-narrower tier, `none` means only that Mondo has no mapping; in 96 of those items ≥ 6 of 7 models
agree on the same candidate, which is probably the missing mapping. The genuine shared failures are
35 items where every model abstains, typically because the SNOMED term names a protein by an older
alias.

Two consequences follow. Gold errors understate every accuracy above, though the comparisons hold
because all models face the same gold. And unanimous disagreement among independently trained
decision models makes a cheap curation queue: `results/consensus-disagreements.tsv` holds the 230
cases.

### Next

- ICD‑10‑CM → Mondo with no ICD training data: the transfer test, with Jev as the zero-shot
  reference and an adapter that carries no SNOMED licensing question.
- Retrain with labels-only states in the mix and `--replay`, to close the format-shift gap.
- Option-order robustness through `/v1/systemone/permute`.

## Reproducing

```
just install && just kev-install        # runner env; Kev checkout with its MLX serving env
just build                              # term tables, gold, retrieval, tiered items -> data/
just serve 4b                           # separate terminal (scripts/serve_kev.py caps MLX's cache)
just run kev-4b && just eval            # results/<model>/items/, results/comparison-items.md
just sssom kev-4b                       # results/kev-4b/items/mappings.sssom.tsv
```

`kevmap build --name <set> --shuffle --ambiguous-share 0.5 [--labels-only --match-only --k N]` builds
the variants; `kevmap run --items <set>` scores them; `kevmap train-data` writes the fine-tuning
records; `kevmap compare A B` gives the paired bootstrap. `scripts/run_jev.sh` scores Jev
(`JEV_API_KEY` from the shell, read at run time). `cluster/` holds the SLURM scripts that built the
CUDA environments, ran every open model and fine-tuned Kev on an L40S; `cluster/sync.sh --pull`
brings results back as `results/<model>@ll/`.

Serving Kev on Apple Silicon needs `scripts/serve_kev.py`: plain `kev.serve` on MLX never releases
Metal buffers and grows without bound. Kev's trainer needs `--max_state 1536`; its 384-token default
silently drops these records.

Layout: `src/kevmap/` (`mondo.py`, `snomed.py`, `gold.py`, `retrieve.py`, `items.py`, `questions.py`,
`run.py`, `evaluate.py`, `sssom_out.py`, `train_data.py`, `cli.py`), `tests/`, `scripts/`, `cluster/`,
`results/<model>/<set>/` (`metrics.json`, `per_item.tsv`, `server.json`, `mappings.sssom.tsv`).

## Inputs

| input | source | version |
|---|---|---|
| gold: Mondo's SCTID mappings | `mondo.sssom.tsv` (Mondo release artefact) | as downloaded by medic-ingest |
| Mondo terms | semsql `mondo.db` | obo:mondo/releases/2026-05-05 |
| SNOMED CT | RF2 snapshot, your own licensed copy (`paths.py`) | US Edition 2026-03-01 |

Open item: pin the gold SSSOM and `mondo.db` to the same Mondo release.

## Licensing

Code is MIT. SNOMED CT is licensed content (SNOMED International Affiliate License, via the NLM UMLS
license in the US), so this repository never redistributes SNOMED text:

- `data/` (term tables, item sets, training records) and `runs/` (adapters fine-tuned on that text)
  are gitignored; reproducing needs your own SNOMED CT download.
- SSSOM output carries `SCTID` identifiers with an empty `subject_label`, as Mondo's own SCTID rows do;
  Mondo labels (CC‑BY 4.0) stay.
- `per_item.tsv`, `metrics.json` and `consensus-disagreements.tsv` reference items by SCTID only.
- Whether an adapter fine-tuned on SNOMED labels may be published is unresolved; the checkpoints stay
  local. Hopper's weights are research/demo-only; Kev and JevK5 are Apache‑2.0.
