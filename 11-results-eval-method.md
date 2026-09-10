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

## Making eval fast enough to run on every arm

The first working run took 31.5 min per checkpoint in **development mode**.
The full benchmark is much bigger: PGYM 86 -> 217 assays, contacts 147 ->
1621 proteins. At the original speed that is ~5.5 h per checkpoint, and 52
arms would have cost ~280 GPU-hours, about three training runs.

Three things were wrong, in increasing order of how much they cost:

1. **The configured batch size was not the batch size.** A 65,536-token
   budget capped a 1024-token scoring window at 64 rows, so a requested 256
   silently ran as 64. Budget raised to 262,144.
2. **256 underfeeds the GPU anyway.** A typical contact protein is ~300
   residues, so 256 rows is only 76k tokens per forward. Raising the request
   to 1024 (the budget then caps the effective batch at 865) is 20% faster.
   Above that it saturates.
3. **Eval ran the model in eager mode.** A profile of the eval forward showed
   only **27.7% of GPU time in matmuls**, despite GEMMs of
   262144 x 1024 x 2688 that should run near peak. The rest was unfused
   elementwise work: 20.1% `mul` (RoPE and gating), 13.3% `copy_`, 13.0%
   `add`, 8.6% `gelu`, 9.9% `neg`+`cat` for rotate_half. Training runs this
   model compiled; eval did not, and paid for it on every one of the 20*L
   forwards the Jacobian needs per protein.

`torch.compile(dynamic=True)` took the forward from **185k to 375k tok/s**
(148 -> 300 TFLOP/s), which is finally in line with training. Dynamic shapes
are required, not optional: every protein is a different length. Three unseen
shapes measured after a 12 s warmup all ran at full speed, so the shape
variation costs nothing after the first few seconds.

**Validation.** Compiled vs eager on the same checkpoint, same dev subset, all
54 report metrics: PGYM total scc differed by 0.0001, contact P@L by <=0.0012,
and the largest deviation anywhere was 0.0025 on a metric whose bootstrap CI
is +-0.08. That is bf16 reordering noise. End to end 1891 s -> 1091 s.

Net: ~2.7 h per checkpoint on the full benchmark, ~143 GPU-hours for 52 arms.

## Running it across the series

52 arms have loadable checkpoints (t0b batch sweep, t0 seeds and repeats, t1a
LR, t1b optimizer knobs). All 52 have a byte-identical `model_config.yaml`,
which is what makes one compiled graph serve the whole sweep.

`slurm/sbatch_eval.sh` runs them as a job array: one node per array element,
one eval process per GPU, each task taking every Nth line of the worklist.
Notes that are load-bearing:

- Slurm 25.05.9 binds a single GPU per task, which the training launcher has
  to work around by unsetting `CUDA_VISIBLE_DEVICES`. Here it is exactly what
  is wanted, so it is left alone.
- The inductor cache is shared across tasks on purpose: same architecture, so
  the first compile serves all 52.
- The benchmark task lists are materialised once before submitting
  (`eval/tools/warm_datasets.py`). Without that, 52 tasks race to preprocess
  the same `dataset_dir`.

Eval loss is joined to the downstream scores from each run's Slurm log at the
nearest eval step at or before the checkpoint step, plus a common-step column.
The series is wall-clock matched, so arms stop at different steps and their
end-of-run losses are not comparable to each other; the downstream scores have
no equivalent correction, so the step has to be read alongside them.

## Setup, for reproduction

- biotrainer **PR #192** (`peymanvahidi/biotrainer`, `fix/v2-migration-findings`),
  pinned at commit `ed33f6a`, editable from
  `sep07_abl/pkgs/biotrainer-pr192`, installed `--no-deps`.
  Released biotrainer 2.0.0 cannot run the contact framework at all: it dies
  with `TypeError: Got unsupported ScalarType BFloat16`.
- `zero_shot_batch_size: 1024` (256 for the first run), `zero_shot_method:
  masked_marginals`, `torch_compile: true`.
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

## What downstream can and cannot decide (PGYM, all 52 arms)

PGYM finished for all 52 arms before the contact phase, which is enough to
answer the question the series actually needs answered: does a downstream
benchmark resolve our decisions, or only echo the loss?

**It tracks loss closely.** Over the 26 arms that share a batch size and token
budget (t0, t1a, t1b), Spearman(eval loss, PGYM scc) = **-0.873**, and a
regression gives d(scc)/d(loss) = **-0.443**. Across the wider t0b range
(loss 2.20 to 2.51) it stays monotone. So the benchmark is measuring
something real and not noise.

**It confirms the series' biggest decision.** NorMuon at its tuned 7e-3 vs
AdamW at its tuned 5.6e-4: loss 2.2038 vs 2.2350 (+0.0312 for NorMuon), PGYM
scc 0.3542 vs 0.3380 (+0.0162 for NorMuon). That is 3.4x the downstream
threshold. The optimizer choice is now supported by two independent measures.

**But its resolution is coarse.** The six base replicates give the downstream
noise floor:

| | sigma(scc) | 2 sigma |
|---|---|---|
| 3 same-seed repeats | 0.0018 | 0.0037 |
| 3 different seeds | 0.0024 | 0.0047 |
| all 6 | 0.0020 | 0.0040 |

Replicates of the *same configuration* span 0.3492-0.3544. Combining
2 sigma = 0.0047 with the slope, **a loss gap of 0.0106 is needed before PGYM
can resolve it at 2 sigma**. For comparison, 2 sigma on eval loss is 0.0019.

So PGYM is roughly **5x blunter than val loss**. Every Tier 1 arm, winners and
rejects alike, lands inside the base replicate band:

| arm | loss | scc | vs replicate band |
|---|---|---|---|
| tuned LR winner (7e-3) | 2.2038 | 0.3542 | inside |
| wd winner (1e-5) | 2.2014 | 0.3540 | inside |
| cautious (rejected on loss) | 2.2077 | 0.3528 | inside |
| nesterov (rejected on loss) | 2.2068 | 0.3493 | inside |

**How to use this.** Downstream scores confirm large decisions and cannot
adjudicate small ones. Concretely:

- The Tier 1c 20%-vs-25% masking question is ~0.001 in loss. PGYM **cannot**
  settle it, and asking it to would be reading noise. Val loss at equal steps
  remains the decision rule there.
- The Tier 1c 100/0/0 question is ~0.15 in loss, 14x what PGYM needs. That one
  it will settle decisively, which matters because eval loss is the measure
  that is biased against those arms.
- Reporting downstream numbers for an arm is still worth doing; treating a
  0.002 downstream difference as a result is not.

**Per-arm numbers for all 52 arms are in [12-results-eval-table.md](12-results-eval-table.md).**

## The contact metrics resolve much better than PGYM

All 52 arms finished both frameworks. Measuring every metric the same way -
2 sigma over the six base replicates, divided by the metric's slope against
eval loss - gives the loss gap each one needs before it can call a winner:

| metric | 2 sigma (replicates) | slope | loss gap needed |
|---|---|---|---|
| pgym scc | 0.0040 | -0.443 | 0.0091 |
| casp14 local P@L | 0.0040 | -0.221 | 0.0179 |
| casp15 local P@L | 0.0037 | -0.333 | 0.0113 |
| selected_protein local P@L | 0.0064 | -0.477 | 0.0134 |
| **selected_protein long P@L** | 0.0048 | **-1.972** | **0.0024** |
| **selected_protein long AUC** | 0.0052 | **-2.277** | **0.0023** |

Eval loss itself is 0.0019, so **long-range contact prediction is almost as
sharp a discriminator as val loss**, and roughly 4x sharper than PGYM. It is
not that it is less noisy - the noise is comparable - it is that it responds
4.5x more steeply. This corrects the earlier reading, taken from PGYM alone,
that downstream is uniformly ~5x blunter than loss. **Long-range contact P@L
is the downstream number to report.**

Re-testing the series' decisions on it, against a base replicate band of
0.3891-0.3959:

| decision | d(loss) | d(long P@L) | x threshold | verdict |
|---|---|---|---|---|
| NorMuon 7e-3 vs AdamW 5.6e-4 | +0.0312 | +0.0810 | 17.0 | separated |
| NorMuon wd 1e-5 vs wd 0.1 | +0.0483 | +0.0205 | 4.3 | separated |
| beta2 0.95 vs 0.9 | +0.0031 | +0.0041 | 0.9 | tie |
| cautious off vs on | +0.0036 | +0.0035 | 0.7 | tie |
| nesterov off vs on | +0.0027 | +0.0068 | 1.4 | separated (marginal) |

Every verdict agrees with the one val loss gave. Two independent measures, same
answers, including the two rejections. That is the strongest statement the
series can make about its own method.

## An arm where loss and structure disagree

`t0b-normuon-b1M-1721969` has eval loss 2.3367 and long P@L of **0.0135** -
essentially no long-range contact signal at all - while
`t0b-normuon-b1M-lo-1723131`, at a *better but comparable* loss of 2.3114,
reaches 0.2622, and `t0b-normuon-b4M-mid-1723134` at an almost identical loss
of 2.3369 reaches 0.1813. Its PGYM score is unremarkable (0.2718 vs 0.2940),
so only the long-range structure collapsed.

The two linear-decay arms behave similarly (0.0384 and 0.0531 at losses 2.46
and 2.51). The common thread among the collapsed arms is a higher learning
rate or a decay shape that ended mid-schedule, but this is three points and a
hypothesis, not a result. It is worth chasing because a case where val loss
cannot see a downstream collapse is exactly the failure mode a loss-only
ablation series is blind to.
