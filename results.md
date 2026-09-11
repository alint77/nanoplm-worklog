# Results

Everything measured, in the order it was settled. Arm names on fscratch keep
the old tier prefixes, given per section so the docs and the filesystem agree.

Contents: [batch size](#global-batch-size) ·
[optimizer](#optimizer) · [MLM objective](#the-mlm-objective) ·
[LR re-check](#the-lr-re-check-at-20-masking) ·
[downstream](#downstream-across-67-arms) · [FA3 fork](#flashattention-3-fork-ab)

Per-arm downstream scores for all 67 arms are in
[eval-table.md](eval-table.md).

---

## Global batch size

Arms `t0b-*`. 26 runs, token-matched at 10.07B tokens each, h1024/L32 399.6M, 4 nodes. Not
wall-clock matched: the point was to isolate the batch effect. Settled before
the optimizer, because both optimizers' LR optima move with batch size.

First round, one LR rule per cell:

| optimizer | batch | grad_accum | LR | rule | eval @ 10.07B |
|---|---|---|---|---|---|
| AdamW | 1.05M | 1 | 1e-4 | anchor | 2.3819 |
| AdamW | 2.10M | 2 | 1.41e-4 | sqrt(B) | 2.3822 |
| AdamW | 2.10M | 2 | 2e-4 | linear | **2.3716** |
| AdamW | 4.19M | 4 | 2e-4 | sqrt(B) | 2.3980 |
| AdamW | 4.19M | 4 | 4e-4 | linear | 2.3828 |
| NorMuon | 1.05M | 1 | 1e-2 | anchor | 2.3367 |
| NorMuon | 2.10M | 2 | 1.41e-2 | sqrt(B) | 2.3516 |
| NorMuon | 2.10M | 2 | 2e-2 | linear | 2.4582 |
| NorMuon | 4.19M | 4 | 2e-2 | sqrt(B) | 2.3882 |
| NorMuon | 4.19M | 4 | 4e-2 | linear | 2.5116 |

**Why that round decided nothing.** For NorMuon the winning LR was, at every
batch, the lowest one tested. Both scaling rules raise LR with batch, so
"bigger batch is worse" and "lower LR is better" were entangled. Filling in
lower LRs reversed the reading twice: round 1 made big batches look bad, round 2
made 1M look best purely because 1M was the only batch given a 7e-3 point.
Nothing about the batch axis is conclusive until every batch has an interior
minimum, meaning a point where going lower makes the loss worse.

Rounds 3 and 4 got there. NorMuon, eval loss at 10.07B tokens (two sessions of
the same fork submitted overlapping grids 34 s apart, so three cells are
duplicate pairs; those are kept, since same-config repeats measure
sigma_repeat):

| LR | 1M | 2M | 4M |
|---|---|---|---|
| 2.5e-3 | 2.3169 | - | 2.3474 |
| 3.5e-3 | 2.3100 / 2.3106 | - | 2.3337 |
| 5e-3 | **2.3086 / 2.3089** | **2.3118** | 2.3248 / 2.3250 |
| 7e-3 | 2.3114 | **2.3118** | **2.3200** |
| 1e-2 | 2.3367 | 2.3209 | 2.3212 |
| 1.41e-2 | - | 2.3516 | 2.3369 |
| 2e-2 | - | 2.4582 | 2.3882 |

1M and 4M are bracketed on both sides. 2M has 5e-3 and 7e-3 tied with 1e-2
worse, and its low side is untested, but both neighbours turn up below 5e-3.

**Best per batch: 1M 2.3086, 2M 2.3118, 4M 2.3200.** Monotone, and the 1M-to-4M
gap of 0.0114 is ~29x sigma_repeat, so it is signal. Against AdamW's best
anywhere of 2.3716, NorMuon's worst bracketed point still wins; the optimizer
gap is 0.05 to 0.06 at every batch. The first round said otherwise, with the
margin shrinking from -0.045 at 1M to -0.020 at 2M and reversing to +0.005 at
4M; that was purely an LR artifact of running NorMuon above its optimum at the
larger batches.

Three things about LR came out of the same grid and set up Tier 1:

- **The optimum barely moves with batch**: 5e-3 at 1M and 2M, 7e-3 at 4M. A
  factor 1.4 across a 4x batch change, roughly B^0.24. Neither sqrt(B) nor
  linear describes it, and both overshot at 4M (2e-2 cost 0.068 against 7e-3,
  4e-2 cost 0.19). So Tier 1's grids are anchored on measured optima, not a
  rule.
- **NorMuon's curve is asymmetric**: flat below the optimum (0.003 across
  3.5e-3 to 7e-3 at 1M), punishing above (1e-2 costs 0.028, 2e-2 costs 0.15).
  A grid should sit at or below the optimum, not straddle it from above.
- **NorMuon is far more LR-sensitive than AdamW.** At 2M, 1.41e-2 to 2e-2 costs
  0.107, while AdamW's entire spread over every batch and LR is 0.026. AdamW is
  nearly flat: best-per-batch spread 0.011, and 1M vs 4M differ by 0.0009, so
  4.19M would be free for AdamW.

**Chose 4.19M**, at a cost of 0.0114. Reasoning in
[decisions.md](decisions.md#global-batch-size-419m-tokens).

---

## Optimizer

Arms `t1a-*` (learning rate) and `t1b-*` (knobs). Base: h1024/L32 399.6M, 4.19M tokens/step (ga 4), 6 h stable on 4 nodes,
~10,500-11,750 steps, ~44-49B tokens. NorMuon at dion's defaults with
spectral_norm scaling, weight decay matched at 1e-5 on both optimizers.

Two loss columns throughout, because node placement moved step time
1826-1998 ms across runs differing **only** in LR, so the wall-clock stop hands
different token budgets to identical configurations. Same-compute arms are read
in the equal-step column.

### Learning rate (10 runs)

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

**NorMuon beats AdamW by 0.0289** (2.2061 vs 2.2350), each at its own tuned LR.
That is 15x the 2-sigma threshold and the largest effect the series has
produced.

Both optima are interior, so neither grid needs extending. AdamW's 5.6e-4 sits
beside ESM C's published 5e-4 for their 300M at the same 4.2M batch, which is a
useful independent check that our setup is not somewhere strange. NorMuon's
7e-3 reproduces the Tier 0b measurement made at a quarter of the tokens; 5e-3
is 0.0007 behind at equal steps, effectively tied, and 7e-3 is taken because two
independent measurements agree on it.

**Node placement reordered this grid.** On wall clock 5e-3 looked best, having
drawn nodes 8% faster and run 500 extra steps, worth about 0.004 at the late-run
slope. At equal steps 7e-3 wins. The AdamW ordering was unaffected.

### Weight decay, beta2, and the nanoplm deviations (10 runs)

At the winning LRs. Controls are the LR winners, so nothing was re-run.

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

Verdicts against the pre-registered threshold of 0.0019 at equal steps:

| candidate | delta | verdict |
|---|---|---|
| NorMuon over AdamW | -0.0289 | **WINS**, 15x |
| NorMuon wd 1e-5 (vs dion's 0.01) | -0.0050 | **WINS**, 2.6x |
| cautious_wd = true | -0.0018 | no measured effect |
| nesterov = true | +0.0007 | no measured effect |
| NorMuon beta2 0.9 / 0.98 | +0.0007 / +0.0019 | no measured effect |
| AdamW beta2 0.95 | +0.0003 | no measured effect |
| AdamW wd 0.01 | +0.0011 | no measured effect |

**Weight decay matters asymmetrically for NorMuon**: 1e-5 gains 0.0050, 0.1
loses 0.0419. An 8x span around dion's 0.01 moves the loss by 0.047, so this is
not a knob to leave at a library default. beta2 is close to flat for NorMuon
(0.9 and 0.98 both within 0.002 of 0.95) and matters for AdamW on the high side
(0.999 costs 0.0098).

**Both nanoplm deviations are ties, and cautious decay is not slower.** An
earlier reading had it ~5% slower and therefore losing on wall clock. Wrong: its
run was the only one on jpbo-013, and the trace says the slowdown is the node.

| | cautious (jpbo-013) | wd1e-5 (jpbo-001) |
|---|---|---|
| GEMM launches | 11,861 | 11,861 |
| GEMM us/launch | 680.5 | 676.6 (ratio 1.006) |
| `multi_tensor_apply` | 75.2 ms | 75.7 ms |
| compute union | 13,286 ms | 13,436 ms |
| exposed NCCL/step | 174.2 ms | 89.8 ms |
| step time | 2091 ms | 1981 ms |

`multi_tensor_apply` holds the optimizer's elementwise work and is identical;
compute union is 1% *lower*. The 110 ms step difference is the 84 ms/step of
extra exposed comms on that node group. So cautious decay is a tie, arguably a
marginal win, and stays off only because -0.0018 does not clear the floor.

**AdamW's stock ModernBERT settings are already the best of those tested** (wd
1e-5, beta2 0.98), which is convenient: the paper's baseline is not handicapped.

**Storage aside.** NorMuon checkpoints are 3.0 GB against AdamW's 4.5 GB,
because its optimizer state is smaller, so choosing NorMuon takes about 30% off
the series storage bill.

---

## The MLM objective

Arms `t1c-*`. 15 runs. Ordered before the architecture tiers deliberately: masking rate
changes the task, and a task change moves what the architecture should be.

A 5 x 3 factorial rather than one-at-a-time, because rate and corruption split
plausibly interact: at 15% masking, a 10% random share is a larger fraction of a
smaller signal. All arms at base LR 7e-3, NorMuon, 4.19M tokens/step, 6 h on
4 nodes. Eval masking pinned at 15% / 80-10-10 / seed 20260908 for every arm,
verified in the generated configs. Loss at step 10000, the largest step all 15
reached:

| rate | 80/10/10 | 90/5/5 | 100/0/0 |
|---|---|---|---|
| 15% | 2.2050 | 2.2040 | 2.3523 |
| **20%** | **2.1975** | **2.1977** | 2.3496 |
| **25%** | **2.1982** | **2.1984** | 2.3519 |
| 30% (old base) | 2.2051 | 2.2050 | 2.3586 |
| 40% | 2.2311 | 2.2311 | 2.3883 |

**The optimum is 20-25%, and 15% is not it.** The four cells
{20, 25} x {80/10/10, 90/5/5} span 2.1975-2.1984, a four-way statistical tie.
Both neighbours are cleanly separated: 15% is +0.0065 to +0.0075 and 30% is
+0.0075, each ~4x threshold; 40% is +0.034. A parabola through 15/20/25/30 puts
the minimum near 22-23%.

**The split does not matter.** 80/10/10 vs 90/5/5 differs by at most 0.0010 (at
15%) and by <=0.0002 everywhere else. Five independent ties is an answer: BERT's
10%-random / 10%-keep detail buys nothing here, as long as it is not removed
entirely.

**100/0/0 is +0.15 across the board, and this table cannot judge it.** The
pinned eval feeds 10% random tokens, and a model trained with pure masking has
never seen a random substitution: it copies the corrupted token and eats a large
loss on exactly those positions. That is the eval task penalising a distribution
mismatch, not necessarily worse representations. Deferred to downstream, which
has now answered: see below.

**Base moves from 30% to 20% masking, 80/10/10 kept.** Worth 0.0076 against the
old base, 4x threshold. The LR was re-checked at the new rate and 7e-3 still
wins, so the base is 20% at 7e-3 (see below).

### Downstream reverses the loss on pure masking

All 15 arms scored on the full benchmark. These runs are wall-clock matched over
steps 10323-11051, and 700 steps is worth 0.0088 of long P@L, which is bigger
than the effects here, so every score is corrected to step 10500 using the
1.262e-05 per-step slope measured on the six base replicates. That correction
also tightens the replicate band to 0.3875-0.3900. Threshold 0.0024.

long P@L, corrected:

| rate | 80/10/10 | 90/5/5 | 100/0/0 |
|---|---|---|---|
| 15% | **0.4149** | 0.4107 | 0.4142 |
| 20% | 0.4074 | 0.4109 | **0.4201** |
| 25% | 0.4015 | 0.4063 | 0.4002 |
| 30% (old base) | 0.3849 | 0.3908 | 0.3848 |
| 40% | 0.3423 | 0.3379 | 0.3232 |

**100/0/0 is exonerated, and on three of the four bands it wins outright.** Its
+0.15 eval loss was the masking artifact and nothing else. Taking 100/0/0 minus
80/10/10 at each rate, step-corrected per band:

| rate | local | short | medium | long |
|---|---|---|---|---|
| 15% | +0.0251 | +0.0147 | +0.0108 | -0.0006 |
| 20% | +0.0187 | +0.0125 | +0.0121 | +0.0126 |
| 25% | +0.0171 | +0.0075 | +0.0088 | -0.0013 |
| 30% | +0.0115 | +0.0082 | +0.0113 | -0.0001 |
| 40% | +0.0078 | +0.0036 | +0.0004 | -0.0191 |

Sixteen of twenty positive, and on local, short and medium it is positive at
every single rate. Per-band replicate bands make it starker: on local P@L every
100/0/0 arm at 15-30% lands **above** the base band (0.5527-0.5607) while every
80/10/10 arm lands inside or below it. The long band is the patchy one, positive
only at 20%.

The effect is also not dose-dependent: 90/5/5 tracks 80/10/10 rather than
sitting between it and 100/0/0, so this reads as a threshold effect of having
*any* non-mask corruption, not of how much.

Pure masking only hurts at 40%, and only on long contacts (-0.0191, 8x
threshold). That is the interaction the factorial was built to catch: at a high
rate, corrupting a fifth of the input with no real tokens anywhere costs
something.

**The rate axis disagrees with the loss, mildly.** Loss put the optimum at
20-25% with 15% clearly worse (+0.0065). Contacts put it at 15-20%: the top five
cells are (20, 100/0/0) 0.4201, (15, 80/10/10) 0.4149, (15, 100/0/0) 0.4142,
(20, 90/5/5) 0.4109, (15, 90/5/5) 0.4107, and 25% is already 0.4002-0.4063.
30% sits at or below the base replicate band, and 40% collapses on both measures.
So both metrics agree the answer is in 15-25% and disagree about where inside
it, which is why 20% is the choice: it is top-group on contacts and top-group on
loss.

**PGYM sees the rate and not the split, but it is too blunt to count as
disagreement.** It ranks rates the same way (15-25% 0.3635-0.3744, 30%
0.3535-0.3566, 40% 0.3307-0.3360) and calls every split comparison a tie. Push
the long P@L gain through the two slopes to see why that is uninformative:
0.0127 / 1.958 = 0.0065 in loss-equivalent, times 0.438 = 0.0028 in scc, against
a PGYM threshold of 0.0040. **PGYM could not have detected this effect even if
it is entirely real.** Inconclusive, not dissenting.

**What this changes, and the one thing that holds it up.** By the pre-registered
rule this is an adopt: (20%, 100/0/0) clears the deciding metric by 5.8x, and
the band profile makes it a consistent effect rather than one lucky cell. The
base still keeps 80/10/10 for now, on a single specific worry.

The 10% random-token share explicitly trains the model to be *robust* to
substitutions, and the categorical Jacobian *is* a substitution-sensitivity
measurement: it perturbs a residue and reads how far the logits move. So a
scheme that removes substitution training could score higher on this benchmark
by making the model twitchier rather than by giving it better structure. The
band pattern fits that reading uncomfortably well, since the gain is largest
near the diagonal (+0.025 local) and fades with separation (-0.001 long at most
rates). It is not a clean fit either: P@L is rank-based within a band, so a
uniform sensitivity change should cancel, and the artifact story needs the
inflation to be separation-dependent.

Every zero-shot number we have goes through the model's own output sensitivity,
so none of them can settle this. What would: a probe trained on frozen features,
which reads the representation rather than the sensitivity. Biotrainer ships one
(`autoeval_supervised_contact.py`, a logistic regression in the AMPLIFY style),
but it reads `embedder.compute_attention_map(sequence)` and nanoPLM runs FA3,
which never materialises an attention matrix, so it needs an eager-attention
path in the eval wrapper first. `pbc_supervised` and `flip` are both in
nanoPLM's `REMOVED_FRAMEWORKS` today.

So the decision is deferred rather than made, and [plan.md](plan.md#next-settle-the-masking-split)
carries both halves: the confirmation seeds, and the independent measurement.

This tier is also the clearest case for the eval-masking fix. Before
2026-09-08 the eval collator inherited `mlm_probability` from training and
reseeded every call, so it would have scored the 40% arm on a 40%-masked eval
and the 15% arm on a 15%-masked one, then reported that 15% wins by a mile: an
artefact of task difficulty, pointing exactly where our prior already did.

---

## The LR re-check at 20% masking

Arms `t1d-*`. 4 runs, and the first use of the fixed-step rule: `max_steps:
5000` instead of a 6 h wall-clock budget, at the new 20% / 80-10-10 masking,
NorMuon, 4.19M tokens/step, 4 nodes each. Masking changes the task, and 7e-3 had
been tuned at 30%, so it had to be re-confirmed before the base moved.

| LR | 2000 | 3000 | 4000 | 5000 | vs best |
|---|---|---|---|---|---|
| 3.5e-3 | 2.3512 | 2.3098 | 2.2829 | 2.2623 | +0.0094 |
| 5e-3 | 2.3416 | 2.3015 | 2.2758 | 2.2557 | +0.0028 |
| **7e-3** | **2.3362** | **2.2976** | **2.2723** | **2.2529** | best |
| 1e-2 | 2.3379 | 2.3013 | 2.2767 | 2.2579 | +0.0050 |

**7e-3 holds.** An interior minimum with both neighbours worse, and the third
independent confirmation of it: Tier 0b at 10B tokens, Tier 1a at 44B, and now
at a different masking rate. The masking change did not move the optimum, so no
grid extension is needed and the base moves to 20% at 7e-3.

Two things worth recording about the method, since this was the first fixed-step
tier.

**Every run ended on the same step**, so there is no equal-step correction, no
step spread to leak into a downstream score, and the final checkpoint is a
common-step checkpoint by construction. The four jobs also started together and
finished together despite landing on four different node groups, which is
exactly what wall-clock matching could not give us.

**The best-to-second gap peaks mid-run**, as the Tier 1a curves predicted:
0.0017 at step 2000, 0.0037 at 3000, 0.0035 at 4000, 0.0028 at 5000. A longer
sweep would have resolved this ranking slightly *worse*, not better.

**Second place changes hands**, which is the constant-LR bias showing up
directly: 1e-2 is second through step 3000 and 5e-3 is second from 3500 on. A
short sweep flatters the higher LR. It did not change the winner here, but it is
the reason the final long run still owes a re-check of the top two with decay
attached.

One honest caveat: the 0.0019 threshold was measured on 44B-token runs, and the
noise floor at 5000 steps is not separately measured. 7e-3 over 5e-3 is 0.0028,
so 1.5x a threshold borrowed from a different horizon. The direction agrees with
two earlier independent measurements, which is what makes it safe to adopt.

---

## Downstream across 67 arms

Full benchmark, every arm with a loadable checkpoint (t0b, t0, t1a, t1b, t1c).
They share a byte-identical `model_config.yaml`, which is what lets one compiled
graph serve the whole sweep. Per-arm numbers: [eval-table.md](eval-table.md).

Both benchmarks are zero-shot, no probe and no fine-tuning, reading the MLM head
directly.

- **PGYM** (ProteinGym): wet-lab experiments measured how mutations change
  fitness; the model scores every mutant and the metric is Spearman (`scc`)
  against the measured ranking. Masked marginals, one forward per position.
- **PBC zero-shot contact**: the categorical Jacobian. Substitute every position
  with all 20 amino acids and watch how logits move elsewhere; positions that
  move each other are predicted contacts, scored against real structures as
  precision@L and AUC. Costs 20 x L forward rows per protein.

### Where the model stands

Development mode (subsampled, and the ESM C 300M reference report is also
development mode, so it is like-for-like), base checkpoint
`t0-rep1-1736798/checkpoint-11036` at 46.3B tokens:

| task | ours scc | ESM C scc | ours ndcg | ESM C ndcg |
|---|---|---|---|---|
| PGYM total (n=86) | 0.341 | 0.412 | 0.712 | 0.738 |
| nonvirus (n=74) | 0.376 | 0.443 | 0.750 | 0.773 |
| virus (n=12) | 0.123 | 0.224 | 0.480 | 0.516 |

| dataset | metric | ours | ESM C |
|---|---|---|---|
| casp14 | local P@L | 0.457 | 0.522 |
| casp14 | long P@L | 0.143 | 0.287 |
| casp15 | local P@L | 0.485 | 0.527 |
| casp15 | long P@L | 0.170 | 0.314 |
| selected_protein | local P@L | 0.546 | 0.622 |
| selected_protein | long P@L | 0.395 | 0.587 |

Same n as the reference on every contact dataset (38/38/71), so no protein was
skipped on either side. The model has clearly learned structure, and the deficit
is **concentrated in long-range contacts**, roughly half of ESM C, while local
contacts are within 10-15%. PGYM sits about 0.07 scc below.

Do not read the PGYM gap as a token-budget gap alone: this checkpoint trained at
30% masking, so it saw ~150 masked tokens per 512-token sequence, while masked
marginals masks exactly **one**. The contact numbers carry no such confound,
because the Jacobian feeds unmasked sequences, so they are the cleaner
training-budget signal and the more defensible number for the paper.

### Does downstream agree with the loss

**Yes, and on the same verdicts.** PGYM tracks loss closely: over the 26 arms
that share a batch size and token budget, Spearman(loss, scc) = **-0.873**, and
it stays monotone across the wider t0b range (loss 2.20 to 2.51).

Re-testing the series' decisions on long P@L, against a base replicate band of
0.3891-0.3959 and its 0.0024 loss-gap resolution:

| decision | d(loss) | d(long P@L) | x threshold | verdict |
|---|---|---|---|---|
| NorMuon 7e-3 vs AdamW 5.6e-4 | +0.0312 | +0.0810 | 17.0 | separated |
| NorMuon wd 1e-5 vs wd 0.1 | +0.0483 | +0.0205 | 4.3 | separated |
| beta2 0.95 vs 0.9 | +0.0031 | +0.0041 | 0.9 | tie |
| cautious off vs on | +0.0036 | +0.0035 | 0.7 | tie |
| nesterov off vs on | +0.0027 | +0.0068 | 1.4 | separated (marginal) |

Every verdict agrees with the one val loss gave, including the two rejections.
Two independent measures, same answers, which is the strongest statement the
series can make about its own method. On PGYM the optimizer decision shows up as
0.3542 vs 0.3380 (+0.0162), 3.4x that metric's threshold, while every Tier 1 knob lands
inside the base replicate band (0.3492-0.3544): winners and rejects alike.

### One arm where loss and structure disagree

`t0b-normuon-b1M-1721969` has eval loss 2.3367 and long P@L of **0.0135**,
essentially no long-range signal at all, while `t0b-normuon-b1M-lo-1723131` at a
better but comparable 2.3114 reaches 0.2622 and `t0b-normuon-b4M-mid-1723134` at
an almost identical 2.3369 reaches 0.1813. Its PGYM score is unremarkable
(0.2718 vs 0.2940), so only the long-range structure collapsed. The two
linear-decay arms look similar: 0.0384 and 0.0531 at losses 2.46 and 2.51.

Common thread is a higher learning rate or a decay that ended mid-schedule.
Three points and a hypothesis, not a result, but worth chasing: val loss being
blind to a downstream collapse is exactly what a loss-only series cannot catch.

---

## FlashAttention-3 fork A/B

Independent single-node check of `alint77/flash-attention`, branch
`gh200-persistent-bwd` (`fa_gh200/fa_push`, head `af86f68`), against its own
merge-base. Same nanoPLM config, same seed, same data order, only the FA3
kernels differ. 250 steps each, one node, 4 GPUs, strictly sequential and
interleaved `stock -> fork -> forkflag` x 2 repeats. Reproduction notes in
[env.md](env.md#flashattention-3-ab-builds).

| arm | step time | vs stock | repeat spread |
|---|---|---|---|
| stock (merge-base `0251105`, = upstream main) | 1936.6 ms | - | 1.12 ms |
| fork (default build) | 1882.9 ms | **+2.85%** | 0.08 ms |
| fork + `FLASH_ATTENTION_SHORT_SEQ_TILES=TRUE` | 1877.9 ms | **+3.12%** | 0.34 ms |

The measurement is far tighter than the effect: within-run scatter is +-3 ms and
the same arm reproduces across repeats to 0.06-0.79 ms sigma (<=0.04%) against a
54-59 ms effect, roughly 50x the noise floor. The two repeats agree to 0.06 and
0.08 percentage points.

**Loss is unaffected, and the control is what establishes it.** Max abs delta
loss over 50 shared steps: fork vs stock 0.0013, forkflag vs stock 0.0010,
**stock vs stock 0.0017**. The same-arm delta is larger than either cross-arm
delta, so the differences are the pipeline's own nondeterminism. Final loss at
step 250 is 2.6089 / 2.6089 / 2.6088 across all six runs. Gradients match at the
kernel level too: on the shapes nanoPLM runs (350 packed sequences, 64,287
tokens, D=64, bf16), max abs difference against stock is 7.8e-3 to 1.6e-2 on
output, dq, dk and dv for both a full-attention and a sliding-window layer, and
one bf16 ulp at those magnitudes is 8.0e-3, so every difference is 1-2 ulps.

**Why our number is 2.4x the fork's own.** Their README reports +1.21% default
and +1.51% with the flag, on a 4-GPU 16-layer ModernBERT with 10 sliding-window
layers. Ours is 32 layers with 21 sliding-window (`attn_layer_pattern: null`),
so roughly twice as much of the changed path per step. Consistent with the
numbers, untested here. The opt-in forward flag adds only +0.27 percentage
points over the default build and the README notes it regresses long sequences,
so the default build is the sensible choice.

**The 1-node number should transfer.** The A/B puts 65,536 tokens per GPU per
micro-step at grad_accum 4, identical per-GPU work to the 4-node runs. Stock
reads 1936.6 ms here and `t0-rep1` on four nodes reads 1936-1945 ms, which
confirms exposed inter-node comms costs only a few percent in this
configuration. Attention is not a smaller share of a 4-node step.

**Not adopted mid-series.** Reasoning in
[decisions.md](decisions.md#measured-and-not-adopted).
