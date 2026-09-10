# nanoPLM worklog

We are building a 400M protein language model and writing down every choice we
make, with the measurement that justified it. This repo is the paper's notebook.
The runs live on fscratch: `/e/fscratch/profound/naeimitabiei1/sep07_abl/`.

## The plan

Start from stock ModernBERT and change one thing at a time, cheapest and most
load-bearing first: batch size, then the optimizer, then the MLM objective, then
architecture, then a greedy stack of whatever won. Each arm is 4 nodes for a few
hours. Nothing is adopted on a hunch: an arm has to beat the base by more than
the noise floor we measured up front.

Batch, optimizer and masking are settled. Architecture is next.

## Where to look

- [decisions.md](decisions.md) - what we chose, and the number that decided it
- [method.md](method.md) - the rules, fixed before the arms ran: noise floors,
  win thresholds, what counts as a win
- [results.md](results.md) - everything measured, by topic
- [eval-table.md](eval-table.md) - downstream scores for all 52 arms
- [plan.md](plan.md) - what is left to run
- [findings.md](findings.md) - things that cost us a run to learn. Read this
  before debugging anything.
- [env.md](env.md) - pinned versions, layout, launcher and eval gotchas

## Two things that will save you time

Compare arms at **equal steps** unless they differ in compute per step. Node
speed varies ~9% and that is enough to reorder them.

**Val loss is not the only measure.** Long-range contact prediction is about as
sharp and disagrees with it in places, and eval loss can be biased by the
masking scheme it is measured under.
