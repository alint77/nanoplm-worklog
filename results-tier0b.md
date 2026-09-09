# Tier 0b: global batch size

Token-matched probe, 10.07B tokens per run, h1024/L32 399.6M, 4 nodes.
Not wall-clock matched: the point is to isolate the batch effect.

## Results

| optimizer | batch | grad_accum | LR | rule | eval @ 10.07B |
|---|---|---|---|---|---|
| AdamW | 1.05M | 1 | 1e-4 | anchor | 2.3819 |
| AdamW | 2.10M | 2 | 1.41e-4 | sqrt(B) | 2.3822 |
| AdamW | 2.10M | 2 | 2e-4 | linear | **2.3716** |
| AdamW | 4.19M | 4 | 2e-4 | sqrt(B) | 2.3980 |
| AdamW | 4.19M | 4 | 4e-4 | linear | 2.3828 |
| NorMuon | 1.05M | 1 | 1e-2 | anchor | **2.3367** |
| NorMuon | 2.10M | 2 | 1.41e-2 | sqrt(B) | 2.3516 |
| NorMuon | 2.10M | 2 | 2e-2 | linear | 2.4582 |
| NorMuon | 4.19M | 4 | 2e-2 | sqrt(B) | 2.3882 |
| NorMuon | 4.19M | 4 | 4e-2 | linear | 2.5116 |

Best per batch:

| batch | AdamW | NorMuon | NorMuon advantage |
|---|---|---|---|
| 1M | 2.3819 | 2.3367 | -0.045 |
| 2M | 2.3716 | 2.3516 | -0.020 |
| 4M | 2.3828 | 2.3882 | +0.005 |

## What is solid

**NorMuon beats AdamW, but the margin shrinks with batch and is gone at 4M.**
-0.045 at 1M, -0.020 at 2M, +0.005 at 4M. If that survives the noise floor it is
a genuinely interesting result and not one we would have found by testing the
optimizer at a single batch.

**AdamW is nearly flat across the range.** Best-per-batch spread is 0.011, and
1M vs 4M differ by 0.0009. For AdamW, 4.2M would be free.

**The two optimizers want different LR scaling.** AdamW prefers linear at both
larger batches; NorMuon prefers sqrt(B) at both, and 4e-2 (linear at 4M) is
badly overcooked at 2.5116. A single scaling rule would have handicapped one of
them.

**NorMuon is far more LR-sensitive than AdamW.** At 2M, 1.41e-2 to 2e-2 costs
0.107. AdamW's entire spread over every batch and LR is 0.026. Tier 1 needs a
denser LR grid for NorMuon than 2x spacing.

## The confound, and why the first two rounds were not conclusive

For NorMuon the winning LR was, at every batch, the LOWEST one tested. Both
scaling rules raise LR with batch, so "bigger batch is worse" and "lower LR is
better" were entangled.

Round 2 (low-LR fill-ins) confirmed that and reversed the reading:

| NorMuon | LR | eval |
|---|---|---|
| 1M | 1e-2 | 2.3367 |
| 1M | 7e-3 | **2.3114** |
| 2M | 1e-2 | 2.3209 |
| 2M | 1.41e-2 | 2.3516 |
| 2M | 2e-2 | 2.4582 |
| 4M | 1e-2 | 2.3212 |
| 4M | 1.41e-2 | 2.3369 |
| 4M | 2e-2 | 2.3882 |
| 4M | 4e-2 | 2.5116 |

Two things this establishes:

- At a FIXED LR of 1e-2 the batch axis is flat: 1M 2.3367, 2M 2.3209,
  4M 2.3212. 2M and 4M tie and both beat 1M.
- NorMuon's optimum does NOT scale with batch the way either rule predicts.
  1e-2 wins at 2M and 4M alike, and both sqrt(B) and linear overshot at 4M.

**But the minimum is still not bracketed.** At every batch the best LR is the
lowest one tried, and each new low point beats the last: 2e-2 to 1.41e-2 to
1e-2 to 7e-3. So whichever batch happens to be probed lowest looks best, and
the apparent batch ranking has now flipped twice for that reason alone. Round 1
made big batches look bad; round 2 made 1M look best, purely because 1M was the
only batch given a 7e-3 point.

Nothing about the batch axis can be concluded until each batch's LR curve has a
real interior minimum, meaning a point where going lower makes the loss worse.

## Round 3: bracket the minimum (running)

Descending LR at the two extremes, 1M and 4M, at 5e-3, 3.5e-3 and 2.5e-3
(jobs 1723615-1723620). 2M is dropped: it interpolates.

Decision rule, fixed now: compare each batch at its own bracketed minimum. If
4M's minimum is within noise of 1M's, take 4.2M for the 4x scale-out headroom.

## Method note

Three successive "results" moved because each round added a single LR point
instead of bracketing the optimum. A best-over-LR comparison is only meaningful
once every arm's curve has turned around. This applies directly to Tier 1: with
NorMuon losing 0.107 to a 1.4x LR change, a grid that does not bracket the
minimum will rank optimizers by grid placement rather than by merit.
