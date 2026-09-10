# Downstream eval: first working run

Date: 2026-09-10. Checkpoint: sep07 base, `t0-rep1-1736798/checkpoint-11036`
(46.3B tokens, NorMuon 7e-3, mlm 30%, geglu, 399.6M).

## What the two benchmarks are

Both are **zero-shot**: no probe is trained, no fine-tuning. They read the MLM
head directly.

- **PGYM** (ProteinGym). Wet-lab experiments measured how mutations change a
  protein's fitness. The model scores every mutant; the metric is Spearman
  (`scc`) between its ranking and the measured one. Scored with
  **masked marginals**: mask the mutated position, compare log-prob of mutant
  vs wild-type residue. One forward per position.
- **PBC zero-shot contact**. The **categorical Jacobian**: substitute every
  position with all 20 amino acids, watch how logits move elsewhere. Positions
  that move each other are predicted contacts, scored against real structures
  as precision@L and AUC. Costs 20 x L forward rows per protein.

## Numbers (development mode, so subsampled, not final)

PGYM vs ESM C 300M through the same pipeline:

| task | ours scc | ESM C scc | ours ndcg | ESM C ndcg |
|---|---|---|---|---|
| total (n=86) | 0.341 | 0.412 | 0.712 | 0.738 |
| nonvirus (n=74) | 0.376 | 0.443 | 0.750 | 0.773 |
| virus (n=12) | 0.123 | 0.224 | 0.480 | 0.516 |

PBC zero-shot contact, same n as the reference on every dataset (38/38/71), so
no protein was skipped on either side:

| dataset | metric | ours | ESM C |
|---|---|---|---|
| casp14 | local P@L | 0.457 | 0.522 |
| casp14 | long P@L | 0.143 | 0.287 |
| casp15 | local P@L | 0.485 | 0.527 |
| casp15 | long P@L | 0.170 | 0.314 |
| selected_protein | local P@L | 0.546 | 0.622 |
| selected_protein | long P@L | 0.395 | 0.587 |

Read: the model has clearly learned protein structure, and the deficit is
**concentrated in long-range contacts**, where it is roughly half of ESM C,
while local contacts are within 10-15%. PGYM sits about 0.07 scc below.

Do not read the PGYM gap as a token-budget gap alone. This checkpoint trained
at `mlm_probability: 0.30`, so it saw ~150 masked tokens per 512-token
sequence; masked marginals masks exactly **one** position per forward. ESM C
does not pay that distribution shift to the same degree. The gap is consistent
with fewer tokens **and** a higher training masking rate, and Tier 1c is what
separates them: if 15% wins there, some of this closes for reasons unrelated
to val loss.

The contact numbers do not carry that confound. The categorical Jacobian feeds
**unmasked** mutant sequences, so it never depends on the training masking
rate. Contacts are therefore the cleaner training-budget signal of the two,
and the more defensible number for the paper.

Both frameworks together took **31.5 min** on one GH200 (login node).

## Setup, for reproduction

- biotrainer **PR #192** (`peymanvahidi/biotrainer`, `fix/v2-migration-findings`),
  pinned at commit `ed33f6a`, editable from
  `sep07_abl/pkgs/biotrainer-pr192`, installed `--no-deps`.
  Released biotrainer 2.0.0 cannot run the contact framework at all: it dies
  with `TypeError: Got unsupported ScalarType BFloat16`.
- `zero_shot_batch_size: 256`, `zero_shot_method: masked_marginals`,
  `development_mode: true`.
- Config `sep07_abl/eval/eval-t0rep1-dev.yaml`, datasets and results under
  `sep07_abl/eval/`.
- nanoPLM commit `7fa8d16` for two eval fixes that matter for the numbers:
  the token budget was capping a configured batch of 256 down to 64, and the
  categorical Jacobian had no context-length guard, so it would have scored
  proteins the reference embedder skips at lengths the model never trained on.

## Open

- These are **development-mode** numbers. The reference ESM C report is also
  development mode, so the comparison is like-for-like, but neither is a
  reportable final number. A `development_mode: false` run is still owed.
- Only the base checkpoint has been evaluated. Whether downstream scores can
  separate ablation arms at this token budget is untested; the val-loss
  threshold (2 sigma_seed = 0.0019) has no downstream equivalent yet.
