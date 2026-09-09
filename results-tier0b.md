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

## Round 3/4: minima bracketed (13 runs, two forks)

Two sessions of the same fork submitted overlapping grids 34 s apart. Three
runs were exact duplicates; those are kept because same-config repeats measure
sigma_repeat, which Tier 0 was going to spend runs on.

NorMuon, eval loss at 10.07B tokens:

| LR | 1M | 2M | 4M |
|---|---|---|---|
| 2.5e-3 | pending | - | 2.3474 |
| 3.5e-3 | 2.3106 | - | pending |
| 5e-3 | **2.3086** | **2.3118** | 2.3248 / 2.3250 |
| 7e-3 | 2.3114 | **2.3118** | **2.3200** |
| 1e-2 | 2.3367 | 2.3209 | 2.3212 |
| 1.41e-2 | - | 2.3516 | 2.3369 |
| 2e-2 | - | 2.4582 | 2.3882 |

**sigma_repeat = 0.0002**, from the 4M/5e-3 duplicate pair (2.3248 vs 2.3250).
Run-to-run noise from kernel non-determinism is negligible at this horizon.
sigma_seed (different data order) is still unmeasured and will be larger.

**Minima are bracketed at 1M and 4M.** Loss rises on both sides: at 1M,
3.5e-3 (2.3106) > 5e-3 (2.3086) < 7e-3 (2.3114); at 4M, 5e-3 (2.3248) >
7e-3 (2.3200) < 1e-2 (2.3212) < 1.41e-2. 2M has 5e-3 and 7e-3 tied and 1e-2
worse; its low side is untested but both neighbours turn up below 5e-3.

## Findings

**Best per batch: 1M 2.3086, 2M 2.3118, 4M 2.3200.** Monotone, and the
1M-to-4M gap of 0.0114 is 57x sigma_repeat, so it is real signal.

**The optimal LR barely moves with batch**: 5e-3 at 1M and 2M, 7e-3 at 4M.
That is a factor 1.4 for a 4x batch change, roughly B^0.24. Neither sqrt(B) nor
linear describes it, and both overshot badly at 4M (2e-2 cost 0.068 against
7e-3, 4e-2 cost 0.19).

**NorMuon's LR curve is asymmetric.** The bottom is flat: 0.003 across 3.5e-3
to 7e-3 at 1M. Above the optimum it is punishing: 1e-2 costs 0.028 and 2e-2
costs 0.15. So Tier 1's grid should sit at or below the optimum, not straddle
it from above.

**AdamW comparison.** AdamW's best anywhere was 2.3716 (2M, 2e-4). NorMuon's
worst bracketed point beats it. NorMuon wins by 0.05 to 0.06 at every batch,
and the earlier reading that "NorMuon's advantage vanishes at 4M" was purely an
LR artifact.

## The batch decision, and its caveat

A real tradeoff rather than a free lunch:

- 1M: best loss, caps the world size at 16 GPUs / 4 nodes
- 4M: costs 0.0114 eval loss, allows 64 GPUs at full local batch

**Caveat that limits how far this generalises:** the probe is at 10.07B tokens
and a real ablation run is ~44B. Larger batches typically catch up over a longer
horizon, since the per-step disadvantage shrinks as the number of steps grows.
So 0.0114 is an upper bound on the cost at the real horizon, not a measurement
of it. Confirming would take a pair of 6 h runs at 1M and 4M.

DECISION NEEDED: accept 0.0114 (upper bound) for 4x scale-out headroom, or keep
1M and the 4-node ceiling.
