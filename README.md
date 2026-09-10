# nanoPLM worklog

We are building a 400M protein language model and writing down every choice we
make, with the measurement that justified it. This repo is the paper's
notebook. The runs themselves live on fscratch:
`/e/fscratch/profound/naeimitabiei1/sep07_abl/`.

## The plan

Start from stock ModernBERT and change one thing at a time, cheapest and most
load-bearing first: batch size, then the optimizer, then the MLM objective,
then architecture, then a greedy stack of whatever won. Each arm is 4 nodes for
a few hours. Nothing is adopted on a hunch: an arm has to beat the base by more
than the noise floor we measured up front.

Where we are: batch, optimizer and masking are settled. Architecture is next.

## Where to look

**Start here**
- `decisions.md` - what we chose, and the number that decided it
- `ablation-ladder.md` - the plan and what is left to run

**Before you trust a result**
- `prereg.md` - the rules, fixed before arms run: noise floors, win thresholds,
  and which arms are matched by step vs by wall-clock
- `findings.md` - things that cost us a run to learn. Read this before
  debugging anything.

**The numbers**
- `results-tier0.md`, `results-tier0b.md` - shape and batch size
- `results-tier1.md`, `results-tier1-wave1.md` - the optimizer
- `results-tier1c.md` - the MLM objective
- `results-eval.md` - downstream benchmarks: what they can and cannot decide
- `results-eval-table.md` - every arm's downstream scores in one table

**Infrastructure**
- `env.md` - pinned versions, directory layout, launcher gotchas
- `results-fa3-fork.md` - a FlashAttention fork worth +2.85%, and why we have
  not switched to it yet

## Two things that will save you time

Compare arms at **equal steps**, not equal wall-clock. Node speed varies enough
to reorder them otherwise, and that is not a small effect.

**Val loss is not the only measure, and not always the right one.** Long-range
contact prediction disagrees with it in places, and eval loss can be biased by
the masking scheme it is measured under. `results-eval.md` has the details.
