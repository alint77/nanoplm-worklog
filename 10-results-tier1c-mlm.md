# Tier 1c: the MLM objective (15 runs, COMPLETE)

Ordered before the architecture tiers deliberately: masking rate changes the
task, and a task change moves what the architecture should be.

**Design.** A 5 x 3 factorial, not one-at-a-time: rate {15, 20, 25, 30, 40}%
crossed with mask/random/keep split {80/10/10, 90/5/5, 100/0/0}. Crossed
because the two plausibly interact - at 15% masking, a 10% random share is a
larger fraction of a smaller signal. All arms at base LR 7e-3, NorMuon, 4.19M
tokens/step, 6 h on 4 nodes.

**Eval masking is pinned** at 15% / 80-10-10 / seed 20260908 for every arm,
independent of the training masking. Without that decoupling these numbers
would be meaningless: each arm would be scored on its own easier or harder
task. Verified in the generated configs, not just intended.

## Result, at step 10000 (largest step all 15 reached)

| rate | 80/10/10 | 90/5/5 | 100/0/0 |
|---|---|---|---|
| 15% | 2.2050 | 2.2040 | 2.3523 |
| **20%** | **2.1975** | **2.1977** | 2.3496 |
| **25%** | **2.1982** | **2.1984** | 2.3519 |
| 30% (old base) | 2.2051 | 2.2050 | 2.3586 |
| 40% | 2.2311 | 2.2311 | 2.3883 |

Threshold is 2 sigma_seed = 0.0019.

**The optimum is 20-25%, and 15% is not it.** The four cells
{20, 25} x {80/10/10, 90/5/5} span 2.1975-2.1984, a range of 0.0009, i.e. a
four-way statistical tie. Both neighbours are cleanly separated: 15% is
+0.0065 to +0.0075 and 30% is +0.0075, each ~4x the threshold. 40% is +0.034.
A parabola through 15/20/25/30 puts the true minimum near 22-23%.

**The split does not matter.** 80/10/10 vs 90/5/5 differs by at most 0.0010
(at 15%) and by <=0.0002 at every other rate. Five independent ties is an
answer, not a coincidence: the 10%-random / 10%-keep detail that BERT
introduced buys nothing here, as long as it is not removed entirely.

**100/0/0 is a different story, and this table cannot judge it.** It is
+0.15 across the board, which looks catastrophic, but the pinned eval feeds
10% random tokens and a model trained with pure masking has never seen a
random substitution: it copies the corrupted token and eats a large loss on
exactly those positions. That is the eval task penalising a distribution
mismatch, not necessarily worse representations. Deferred to downstream, where
PGYM masked-marginals masks a single position with the mask token and no
random tokens - which if anything favours these arms. The gap is 14x what PGYM
can resolve, so it will answer cleanly.

## Decision

**Base moves from 30% to 20% masking, 80/10/10 kept.** Worth 0.0076 against
the old base, 4x the threshold. 80/10/10 over 90/5/5 is a coin flip on the
evidence; keeping it costs nothing and stays with the ModernBERT default,
which is one less deviation to defend.

Still owed before adoption: re-check the winner at LR 5e-3 and 1e-2. Masking
changes the task, and 7e-3 was tuned at 30%. Two runs.

## Aside: this is why the eval-masking fix mattered

Before 2026-09-08 the eval collator inherited `mlm_probability` from training
and reseeded every call. On this grid it would have scored the 40% arm on a
40%-masked eval and the 15% arm on a 15%-masked one, then reported that 15%
wins by a mile - an artefact of task difficulty, pointing exactly where our
prior already did.