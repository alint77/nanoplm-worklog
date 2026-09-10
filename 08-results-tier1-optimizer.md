# Tier 1: the optimizer

Base: h1024/L32 399.6M, 4.19M tokens/step (ga 4), 6 h stable on 4 nodes,
~10,500-11,750 steps, ~44-49B tokens.

## Wave 1: learning rate (10 runs, COMPLETE)

Two columns because node placement makes them differ. See 04-findings.md: step
time varied 1826-1998 ms across runs that differ ONLY in LR, so the wall-clock
stop hands different token budgets to identical configurations. Same-compute
arms must be read in the equal-step column.

| AdamW | steps | wall-clock | equal-step (10500) |
|---|---|---|---|
| **5.6e-4** | 10500 | **2.2350** | **2.2350** |
| 8e-4 | 10500 | 2.2365 | 2.2365 |
| 4e-4 | 11000 | 2.2374 | 2.2413 |
| 2.8e-4 | 10500 | 2.2489 | 2.2489 |
| 2e-4 | 11000 | 2.2566 | 2.2607 |

| NorMuon | steps | wall-clock | equal-step (10500) |
|---|---|---|---|
| **7e-3** | 11000 | 2.2038 | **2.2061** |
| 5e-3 | 11500 | **2.2011** | 2.2068 |
| 3.5e-3 | 10500 | 2.2111 | 2.2111 |
| 1e-2 | 10500 | 2.2116 | 2.2116 |
| 1.41e-2 | 10500 | 2.2623 | 2.2623 |

### Findings

**NorMuon beats AdamW by 0.029** (2.2061 vs 2.2350), each at its own tuned LR.
That is ~70x sigma_repeat. It is the largest effect the series has produced so
far and it decides the optimizer.

**Both optima are interior**, so neither grid needs extending. AdamW rises on
both sides of 5.6e-4; NorMuon rises on both sides of 7e-3.

**AdamW's optimum, 5.6e-4, sits beside ESM C's published 5e-4** for their 300M
at the same 4.2M batch. An independent check that our setup is not somewhere
strange.

**NorMuon's optimum, 7e-3, reproduces Tier 0b's measurement at 4.19M**, made at
a quarter of the tokens. 5e-3 and 7e-3 are 0.0007 apart at equal steps, which
is under 2x sigma_repeat, so they are effectively tied; 7e-3 is taken because
two independent measurements agree on it.

**Node placement reordered the NorMuon grid.** On wall clock 5e-3 looked best,
having drawn nodes 8% faster and run 500 extra steps. At equal steps 7e-3 wins.
The ordering of the AdamW grid was unaffected.

## Wave 2: weight decay, beta2, and the nanoplm deviations (10 runs, COMPLETE)

At the Wave 1 optima. Controls are the Wave 1 winners, so nothing was re-run.
Both columns given, because the rule from 04-findings.md applies: same-compute arms
(weight decay, beta2) are judged at equal steps, compute-changing arms
(cautious decay, which is ~5% slower) at the wall-clock stop.

**NorMuon**, control `t1a-normuon-lr7e-3`: wall-clock 2.2038 (11000 steps),
equal-step 2.2095.

| arm | steps | wall-clock | equal-step | delta (equal) |
|---|---|---|---|---|
| **wd 1e-5** | 10500 | **2.2014** | **2.2045** | **-0.0050** |
| cautious=true | 10000 | 2.2077 | 2.2077 | -0.0018 |
| beta2 0.9 | 10500 | 2.2072 | 2.2102 | +0.0007 |
| nesterov=true | 10500 | 2.2068 | 2.2102 | +0.0007 |
| beta2 0.98 | 10500 | 2.2085 | 2.2114 | +0.0019 |
| wd 0.1 | 10500 | 2.2497 | 2.2514 | +0.0419 |

**AdamW**, control `t1a-adamw-lr5.6e-4`: wall-clock 2.2350, equal-step 2.2387.

| arm | steps | wall-clock | equal-step | delta (equal) |
|---|---|---|---|---|
| beta2 0.95 | 10500 | 2.2353 | 2.2390 | +0.0003 |
| wd 0.01 | 11000 | 2.2317 | 2.2398 | +0.0011 |
| wd 0.1 | 10500 | 2.2380 | 2.2421 | +0.0034 |
| beta2 0.999 | 10500 | 2.2440 | 2.2485 | +0.0098 |

### Findings

**One real winner: NorMuon weight decay 1e-5 instead of dion's 0.01, worth
-0.0050.** That is 12x sigma_repeat and it wins in both columns. The only arm in
the wave that clearly beats its control.

**Weight decay matters asymmetrically for NorMuon.** 1e-5 gains 0.0050, 0.1
loses 0.0419. An 8x span around dion's 0.01 moves the loss by 0.047, so this is
not a knob to leave at a library default.

**Both nanoplm deviations are ties. CORRECTED: cautious decay does NOT cost
throughput.** `nesterov=true` is +0.0007. `cautious_wd=true` is -0.0018 at
equal steps, and equal steps is the correct comparison for it because it does
not change compute.

An earlier version of this file claimed cautious was ~5% slower and therefore
lost on wall clock. That was wrong. Its run was the only one on jpbo-013 and
the trace shows the slowdown is not the arm:

| | cautious (jpbo-013) | wd1e-5 (jpbo-001) |
|---|---|---|
| GEMM launches | 11,861 | 11,861 |
| GEMM us/launch | 680.5 | 676.6 (ratio 1.006) |
| `multi_tensor_apply` | 75.2 ms | 75.7 ms |
| compute union | 13,286 ms | 13,436 ms |
| exposed NCCL/step | 174.2 ms | 89.8 ms |
| step time | 2091 ms | 1981 ms |

`multi_tensor_apply` holds the optimizer's elementwise work and is identical, so
cautious decay adds nothing measurable. Compute union is 1% LOWER. The 110 ms
step difference is the 84 ms/step of extra exposed comms on that node group.

Verdict: cautious decay is a tie, arguably a marginal win, and stays off only
because -0.0018 does not clear the noise floor. It is not slower.

**AdamW's stock ModernBERT settings are already optimal among those tested.**
wd 1e-5 and beta2 0.98 beat every alternative. Nothing to change in the paper's
baseline, which is a convenient result: the baseline is not being handicapped.

**beta2 is close to flat for NorMuon** (0.9 and 0.98 both within 0.002 of 0.95)
and clearly matters for AdamW on the high side (0.999 costs 0.0098).

### Pending: sigma_seed

Every delta except NorMuon wd (-0.0050) and wd 0.1 (+0.0419) is under 0.004.
sigma_repeat is 0.0004 but sigma_seed is unmeasured and will be larger, so the
small deltas are not yet distinguishable from noise. Tier 0 measures it next.

The wd 1e-5 result should be adopted only after Tier 0 confirms 0.0050 clears
2 x sigma_seed. It is written here as a candidate, not yet folded into the base.
