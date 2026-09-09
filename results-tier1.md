# Tier 1: the optimizer

Base: h1024/L32 399.6M, 4.19M tokens/step (ga 4), 6 h stable on 4 nodes,
~10,500-11,750 steps, ~44-49B tokens.

## Wave 1: learning rate (10 runs, COMPLETE)

Two columns because node placement makes them differ. See findings.md: step
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

## Wave 2: weight decay, beta2, and the two nanoplm deviations (10 runs, RUNNING)

At the Wave 1 optima. Controls are the Wave 1 winners themselves
(t1a-adamw-lr5.6e-4 at 2.2350, t1a-normuon-lr7e-3 at 2.2061), so no control
runs are repeated.

| optimizer | knob | values | base |
|---|---|---|---|
| AdamW @ 5.6e-4 | `adam_weight_decay` | 0.01, 0.1 | 1e-5 |
| AdamW @ 5.6e-4 | `adam_beta2` | 0.95, 0.999 | 0.98 |
| NorMuon @ 7e-3 | `muon_weight_decay` | 1e-5, 0.1 | 0.01 |
| NorMuon @ 7e-3 | `muon_beta2` | 0.9, 0.98 | 0.95 |
| NorMuon @ 7e-3 | `muon_cautious_weight_decay` | true | false |
| NorMuon @ 7e-3 | `muon_nesterov` | true | false |

The last two test the nanoplm defaults that deviate from dion's. nanoplm ships
cautious decay and nesterov ON, dion ships them OFF, and nothing documents the
deviation as deliberate. Either they win, which vindicates nanoplm, or they do
not and the base keeps dion's defaults with evidence behind it.

AdamW is tuned alongside NorMuon even though it has already lost, because it is
the paper's baseline and a poorly tuned baseline would weaken the NorMuon
claim rather than strengthen it.
