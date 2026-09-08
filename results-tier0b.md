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

## The confound, and the fill-in runs

For NorMuon the winning LR at every batch is the LOWEST one tested: 1e-2 at 1M,
1.41e-2 at 2M, 2e-2 at 4M. Both scaling rules raise LR with batch, so "bigger
batch is worse" and "lower LR is better" are entangled. The larger batches may
simply have run further above their optimum.

Four fill-in runs on the low side (1723131-1723134): NorMuon 1M at 7e-3, 2M at
1e-2, 4M at 1e-2, 4M at 1.41e-2. If 4M improves materially at a lower LR, the
batch penalty is smaller than it looks and 4.2M is back on the table.

## Not yet decided

Nothing is decided until the fill-ins land and Tier 0 gives a noise floor. The
1M-to-2M NorMuon gap is 0.015 and could be noise; the 1M-to-4M gap is 0.052 and
probably is not.
