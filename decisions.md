# Decisions

Date order. Each one says what we chose and why.

## Throw away the jul30 ablations

Started over on 2026-09-07.

Why: arms were compared against a spine that kept changing, and two of its
rungs were regressions. Nothing had a measured noise floor, so most of the
"wins" could have been noise. Not fixable after the fact.

## The paper's baseline is stock ModernBERT

Not our best-performing config.

Why: a paper needs a baseline other people recognise. Every claim is a delta
from stock ModernBERT.

## Wall-clock matched, not token matched

Each run gets 6 h of stable training on 4 nodes. About 96 GPU-hours per run.
The 30 min LR decay is a separate campaign, run later off the saved
checkpoints rather than chained behind each job.

Why: we care about loss per GPU-hour, not per token. If a change makes the
model slower, it gets fewer tokens in its 6 h and pays for the slowdown
itself. That is the honest comparison.

## Base shape: h1024 / L32 / 16 heads / 2688 MLP = 399.6M

Kept ESM C's proportions (aspect ratio 32, 8/3 MLP) but moved the width to
1024.

Why: we measured the ladder on 4 nodes.

| shape | aspect | params | step | MFU |
|---|---|---|---|---|
| h896 / L28 | 32.0 | 273.8M | 418.8 ms | 42.72% |
| h896 / L34 | 26.4 | 332.3M | 510.9 ms | 42.49% |
| h960 / L30 | 32.0 | 332.8M | 470.5 ms | 46.02% |
| h1024 / L27 | 37.9 | 337.3M | 427.4 ms | 51.24% |
| h1024 / L32 | 32.0 | 399.6M | 511.0 ms | 50.79% |
| h1152 / L36 | 32.0 | 574.8M | 771.5 ms | 48.01% |

Two things fell out of this:

1. Aspect ratio is nearly free. At the same width, going from aspect 37.9 to
   32 costs 0.45 MFU points.
2. Width is not free. 1024 divides the 128-wide Hopper GEMM tile and 960 does
   not. That gap is about 5 points.

So of three goals (power-of-two width, ESM C aspect 32, ESMC-300M size) we can
have any two. We picked the first two. The cost is real and goes in the paper:
the model is 399.6M instead of 332.8M, and it sees about 16% fewer tokens in
6 h than h1024/L27 would. Every arm pays this equally, so it does not bias
anything inside the series.

## Large model: h1152 / L36 = 574.8M

Same proportions. Needs `micro_batch_seqs: 64`; at 128 it OOMs (peak 97,268 of
97,280 MiB).

## UniRef50 only

Why: about 5.2 epochs of repetition at this budget. We considered adding
corpora and decided the extra variable was not worth it for an architecture
study.

## bf16 everywhere, one fp8 pair at the end

Why: fp8 is a separate question. Running it under every arm would confound the
architecture results. One bf16-vs-fp8 pair on the final recipe answers it.

## Logging and profiling

`logging_steps: 20`, `eval_steps: 500`, layerwise metrics every 20, profiler
traces on for every run.

Why: traces are cheap and we have already had two runs where the log alone
would have hidden the problem. Eval is 4.0 s a time, so every 500 steps costs
about 1.5% of a run rather than 3% at every 250. Eval loss is a secondary
signal now, so the cadence does not need to be tight.

## num_workers: 4, written out explicitly

Why: `"auto"` resolves from world size. At 4 GPUs it gives 16 workers per rank,
at 16 GPUs it gives 64, which is 256 processes per node and a host-RAM OOM.

## Coordinate-descent inductor tuning on

`TORCHINDUCTOR_COORDINATE_DESCENT_TUNING=1`.

Why: about +2.3 MFU points, and the cost is compile time only, which sits
outside the wall-clock budget.

## Biotrainer decides, not eval loss

The deciding metric is biotrainer PBC and PGym, run later on the stored
checkpoints. Biotrainer is not ready, so the ablations run now and get scored
afterwards.

Why it matters for planning: every checkpoint has to survive, which is on the
order of a terabyte across the series, and arms cannot be ranked until
biotrainer lands.

## Eval masking pinned and made deterministic

Eval used to inherit the whole training masking recipe, and redrew its masks
on every call.

Why: a model trained at 15% masking was also scored at 15% while the base was
scored at 30%. Different tasks, incomparable losses, so any arm touching the
masking recipe was uninterpretable. And redrawn masks meant every eval number
carried mask noise. Both fixed. Eval is now pinned at 15% token masking with
80/10/10 and a fixed seed, for every arm.

## TE fused RoPE: built, measured, reverted

Why: it works and it is about 2x faster at the kernel level, but the step time
is identical. See `findings.md`. Kept on branch `perf/te-fused-rope`, not
merged.
