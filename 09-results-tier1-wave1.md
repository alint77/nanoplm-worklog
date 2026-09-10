# Tier 1 Wave 1: optimizer and learning rate

10 runs, 6 h stable phase each, 4 nodes, 4.19M tokens/step, ~44B tokens.
h1024/L32 399.6M, dion defaults, spectral_norm scaling, weight decay matched at
1e-5 on both optimizers.

## Result

Compared at **equal steps** (10,500 = 44.0B tokens), because these arms have
identical compute per step and their throughput differences are node noise, not
the arm. Wall-clock numbers shown alongside.

| optimizer | LR | equal-step | wall-clock | steps reached |
|---|---|---|---|---|
| NorMuon | **7e-3** | **2.2061** | 2.2038 | 11,000 |
| NorMuon | 5e-3 | 2.2068 | 2.2011 | 11,500 |
| NorMuon | 3.5e-3 | 2.2111 | 2.2111 | 10,500 |
| NorMuon | 1e-2 | 2.2116 | 2.2116 | 10,500 |
| NorMuon | 1.41e-2 | 2.2623 | 2.2623 | 10,500 |
| AdamW | **5.6e-4** | **2.2350** | 2.2350 | 10,500 |
| AdamW | 8e-4 | 2.2365 | 2.2365 | 10,500 |
| AdamW | 4e-4 | 2.2413 | 2.2374 | 11,000 |
| AdamW | 2.8e-4 | 2.2489 | 2.2489 | 10,500 |
| AdamW | 2e-4 | 2.2607 | 2.2566 | 11,000 |

**NorMuon at 7e-3 wins by 0.0289 over AdamW at 5.6e-4.**

## What is settled

**Optimizer: NorMuon.** It beats AdamW by 0.029 here and beat it by 0.05-0.06
at every batch size in Tier 0b. Both phases agree.

**NorMuon LR: 7e-3.** Confirmed three times independently: Tier 0b at 10B
tokens, the 9,000-step check, and the final 44B comparison. Both neighbours are
worse, so it is an interior minimum, and the curve is shallow near it (7e-3 and
5e-3 differ by 0.0007 at equal steps).

**AdamW LR: 5.6e-4**, an interior minimum with 8e-4 second at +0.0015. Close to
ESM C's published 5e-4 for their 300M, which is a useful independent check on
our setup even though we are not adopting AdamW.

## Why equal-step and not wall-clock

At the wall-clock stop NorMuon 5e-3 looks best (2.2011). It is not: it landed
on faster nodes, ran 11,500 steps against 7e-3's 11,000, and those 500 extra
steps are worth about 0.004 at the late-run slope. At equal steps 7e-3 wins.

Node placement moved step time from 1826 to 1998 ms across ten runs that differ
only in learning rate, a 9.4% spread with no architectural cause. That is worth
more loss than the gaps between adjacent LR points, so it can and did reorder
them. See 04-findings.md.

## Storage note

NorMuon checkpoints are 3.0 GB against AdamW's 4.5 GB, because its optimizer
state is smaller. Choosing NorMuon takes about 30% off the series storage bill.

## Next

Wave 2: weight decay and beta2 at LR 7e-3, plus one run each for
`cautious_wd=true` and `nesterov=true` so the two undocumented nanoplm
deviations from dion get tested rather than inherited. 14 runs, blocked on
nothing now.

Then Tier 0's noise floor on NorMuon at 7e-3, which finally gives sigma_seed and
makes the pre-registered 2-sigma threshold usable. Note it must be measured at
equal steps too, or it will absorb the 9% node-speed spread.
