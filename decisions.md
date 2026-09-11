# Decisions

What we chose, and the number that decided it. Measurements live in
[results.md](results.md); the rules for reading them are in
[method.md](method.md).

## The recipe so far

| | choice | decided by |
|---|---|---|
| shape | h1024 / L32 / 16 heads / 2688 MLP = 399.6M | MFU ladder, below |
| global batch | 4.19M tokens (grad_accum 4 at 16 GPUs) | Tier 0b, below |
| optimizer | NorMuon, `adjust_lr: spectral_norm` | -0.0289 over tuned AdamW |
| muon LR | 7e-3 | interior minimum, three times |
| muon weight decay | 1e-5 (dion ships 0.01) | -0.0050 |
| muon beta2 / cautious / nesterov | 0.95 / off / off | all ties, library defaults kept |
| MLM | 20% masking, 80/10/10 split, token | -0.0076 over the old 30% (split under review) |
| eval masking | pinned 15%, 80/10/10, fixed seed | see below |
| precision | bf16, one fp8 pair at the end | below |
| corpus | UniRef50 only | below |

Open: everything architectural. That is what the remaining tiers are for
([plan.md](plan.md)).

## The series

**Threw away the jul30 ablations** on 2026-09-07 and started over. Arms were
compared against a spine that kept changing, two of its rungs were regressions,
and nothing had a measured noise floor, so most of the wins could have been
noise. Not fixable after the fact.

**The paper's baseline is stock ModernBERT**, not our best config. A paper needs
a baseline other people recognise, so every claim is a delta from stock.

**Runs are matched by wall clock when they change compute per step and by step
count when they do not.** We care about loss per GPU-hour, not per token: if a
change makes the model slower it gets fewer tokens in its 6 h and pays for the
slowdown itself. That only applies when arms actually differ in cost, which is
the whole rule in [method.md](method.md#matching-fixed-step-or-wall-clock).

## The model

**Base shape: h1024 / L32 = 399.6M.** ESM C's proportions (aspect ratio 32, 8/3
MLP) at width 1024. Measured on 4 nodes:

| shape | aspect | params | step | MFU |
|---|---|---|---|---|
| h896 / L28 | 32.0 | 273.8M | 418.8 ms | 42.72% |
| h896 / L34 | 26.4 | 332.3M | 510.9 ms | 42.49% |
| h960 / L30 | 32.0 | 332.8M | 470.5 ms | 46.02% |
| h1024 / L27 | 37.9 | 337.3M | 427.4 ms | 51.24% |
| h1024 / L32 | 32.0 | 399.6M | 511.0 ms | 50.79% |
| h1152 / L36 | 32.0 | 574.8M | 771.5 ms | 48.01% |

Aspect ratio is nearly free: at the same width, going from 37.9 to 32 costs 0.45
MFU points. Width is not: 1024 divides the 128-wide Hopper GEMM tile and 960
does not, worth about 5 points. Of power-of-two width, ESM C's aspect 32, and
ESM C-300M's size we could have any two; we took the first two. The cost, which
goes in the paper: 399.6M instead of 332.8M, and ~16% fewer tokens in 6 h than
h1024/L27 would give. Every arm pays it equally.

**Large model: h1152 / L36 = 574.8M**, same proportions. Needs
`micro_batch_seqs: 64`; at 128 it OOMs (peak 97,268 of 97,280 MiB).

**Untied embeddings.** ModernBERT ties the input embedding to the output head to
save parameters on LLM-sized vocabularies. At vocab 32 the embedding is
32 x 1024, so tying saves 32,768 parameters out of 399.6M, or 0.008%: the
rationale does not transfer to a protein model. The untied head is a (32, 1024)
matrix and routes to the AdamW group rather than Newton-Schulz, because
`_is_embedding_or_unembedding_param` catches `decoder.weight`, so it is safe
under NorMuon. The Tier 2 arm flips accordingly and tests *tied* embeddings, in
case the coupling helps for some other reason.

## Data and objective

**UniRef50 only.** About 5.2 epochs of repetition at this budget. We considered
adding corpora and decided the extra variable was not worth it for an
architecture study.

**MLM at 20% masking, 80/10/10**, moved from 30% after the Tier 1c factorial
([results](results.md#the-mlm-objective)). 80/10/10 over 90/5/5 is a coin flip
on the evidence; keeping it stays with the ModernBERT default, which is one less
deviation to defend.

**The split is not finished.** Downstream exonerated 100/0/0, whose +0.15 eval
loss turned out to be entirely the eval-masking artifact, and (20%, 100/0/0)
leads the whole factorial on long-range contacts by 5x threshold. One seed, and
PGYM calls it a tie, so the base keeps 80/10/10 until four confirmation runs say
otherwise ([plan.md](plan.md#next-confirm-the-masking-split-4-runs)).

**Eval masking pinned and deterministic.** Eval used to inherit the whole
training masking recipe and redraw its masks on every call. So an arm trained at
15% was also *scored* at 15% while the base was scored at 30%: different tasks,
incomparable losses, and any arm touching the masking recipe was uninterpretable.
Redrawn masks put noise in every number too. Both fixed in `4e3bbfc`: eval is
pinned at 15% token masking, 80/10/10, fixed seed, for every arm.

## The optimizer

**Settled first**, because everything after it is a single-factor arm off the
winner, and NorMuon looked much better than AdamW in the jul30 work. Only AdamW
and NorMuon were in it; Muon, NorDion2 and stable_adamw were dropped. Result:
NorMuon 7e-3, weight decay 1e-5
([results](results.md#optimizer)).

**Weight decay was matched across the optimizer arms before running them.** The
dataclass gives NorMuon 0.01 with cautious decay while our base ran AdamW at
1e-5 plain, so left alone the optimizer comparison would also have been a
weight-decay comparison. Both set to 1e-5, cautious off.

**`adjust_lr: spectral_norm`, not rms_norm.** NorMuon orthogonalizes the update
then rescales it by a shape-dependent factor, and dion's docstring says what
each choice is for: `spectral_norm` for LR transfer across model scale
(`lr * sqrt(fan_out / fan_in)`), `rms_norm` for LR compatibility with AdamW
(`lr * 0.2 * sqrt(max(fan_out, fan_in))`). On our matrices at h1024:

| matrix | shape | rms_norm | spectral |
|---|---|---|---|
| QKV q-block | (1024, 1024) | 6.400 | 1.000 |
| Wo | (1024, 1024) | 6.400 | 1.000 |
| MLP up/gate | (5376, 1024) | 14.664 | 2.291 |
| MLP down | (1024, 2688) | 10.369 | 0.617 |

Three reasons. We sweep the NorMuon LR ourselves, so AdamW compatibility, the
only thing rms_norm buys, is worthless to us. We scale h1024 to h1152 later:
under rms_norm every effective LR drifts +6.1% and has to be re-tuned, under
spectral_norm it moves -0.8% to 0.0%, which is the entire point of that scaling.
And rms_norm uses `max(fan_out, fan_in)`, so it cannot tell a tall matrix from a
wide one: the MLP down-projection gets 1.6x the square-matrix LR under rms_norm
and 0.62x under spectral, a 2.6x difference in relative weighting, and under GQA
a k-block `(512, 1024)` gets exactly the same LR as a full `(1024, 1024)` block.
For a study whose whole job is varying matrix shapes, that is not a principled
choice. dion defaults to spectral_norm for every optimizer it ships; nanoplm
overrides it to rms_norm to preserve older behaviour, and that older behaviour
is the jul30 series we discarded.

**The sweep is centred at 5e-3, not 1e-3.** rms_norm multiplies the
square-matrix LR by 6.4 and spectral_norm by 1.0, so an LR tuned at 1e-3 under
rms_norm is roughly 6.4e-3 under spectral. Centred anywhere lower and the sweep
would cover the wrong decade and NorMuon would "lose" for no reason.

## Global batch size: 4.19M tokens

Best achievable loss per batch, each at its own bracketed LR minimum, measured
token-matched at 10.07B tokens ([results](results.md#global-batch-size)):

| batch | best LR | eval @ 10.07B |
|---|---|---|
| 1.05M | 5e-3 | 2.3086 |
| 2.10M | 5e-3 / 7e-3 | 2.3118 |
| 4.19M | 7e-3 | 2.3200 |

**Chose 4.19M despite it being 0.0114 worse**, for four reasons in order of
weight.

1. **The 0.0114 is an upper bound, not the cost at our horizon.** It was
   measured at 10.07B tokens; a real ablation run is ~44B, and a large batch's
   per-step disadvantage shrinks as the number of steps grows. Confirming it
   exactly would take a pair of 6 h runs, which is not worth 200 GPU-hours.
2. **Scale-out headroom.** Global batch caps the world size that can run at full
   local batch. At `micro_batch_seqs: 128` and 512-token sequences each GPU
   takes 65,536 tokens, so 1M caps us at 16 GPUs and 4M allows 64. Below that
   ceiling the local batch shrinks, GEMMs get smaller and utilisation drops. The
   ablations run on 4 nodes either way; this is about the model we eventually
   train.
3. **It biases no comparison.** Every arm runs at the same batch, so it is a
   constant offset, not a differential effect.
4. **It matches ESM C's 4.2M** (stage 1, 512 context, from the ESM Cambrian
   blog), which keeps our numbers comparable to the obvious reference point.

Explicitly **not** a reason: comms. Exposed (non-overlapped) NCCL is 5.2 ms of a
511 ms step, 1.0%, so gradient accumulation saves under 1% of throughput.

Consequences for the writeup:

- `warmup_steps` drops from 1000 to 250, holding warmup at 1.05B **tokens**
  rather than at a step count. Keeping 1000 steps would have quadrupled warmup
  to 4.2B tokens, 9.5% of a 6 h run instead of 2.4%.
- grad_accum becomes 4 at 16 GPUs. A 6 h run is ~10,568 optimizer steps and
  ~44.3B tokens, against ~42,270 steps for the same tokens at 1M.
- The batch was chosen at 400M parameters. The 600M transfer runs have a
  different critical batch and the writeup must say so.

## Runtime

**bf16 everywhere, one fp8 pair at the end.** fp8 is a separate question;
running it under every arm would confound the architecture results. One
bf16-vs-fp8 pair on the final recipe answers it.

**`logging_steps: 20`, `eval_steps: 500`, layerwise metrics every 20, profiler
traces on for every run.** Traces are cheap and we have already had two runs
where the log alone would have hidden the problem. Eval is 4.0 s a time, so
every 500 steps costs about 1.5% of a run rather than 3% at every 250, and eval
loss is a secondary signal now.

**`num_workers: 4`, written out explicitly.** `"auto"` resolves from world size:
16 workers per rank at 4 GPUs, 64 at 16 GPUs, which is 256 processes per node
and a host-RAM OOM.

**`TORCHINDUCTOR_COORDINATE_DESCENT_TUNING=1`.** About +2.3 MFU points, paid in
compile time, which sits outside the wall-clock budget.

**Biotrainer decides, not eval loss.** Consequence for planning: every
checkpoint has to survive, on the order of a terabyte across the series, and
arms cannot be ranked until the downstream runs land.

## Ground rules for the architecture tiers

**MoE is compared at matched *active* parameters**, not matched total, so the
comparison is at equal compute per token, which is what a wall-clock-matched
study measures. Concretely `top_k x expert_intermediate = 2688`, giving every
cell the dense base's 8,257,536 active MLP parameters per layer. Total parameters
vary between cells; that is expected and reported. The control is dense
*swiglu*, not the geglu base, because `use_moe=true` refuses geglu, which is why
tier 2b runs after 2a.

**mHC-lite is tested at both placements.** DSv4 and GLM 5.3 Flash apply it per
sublayer, nanoplm defaults to per layer boundary, and
`mhc_lite_wrapping_level` already accepts both, so rather than pick we run both.

## Measured and not adopted

**TE fused RoPE.** Works, about 2x faster at the kernel level, and the step time
is identical. Kept on branch `perf/te-fused-rope`, not merged. See
[findings.md](findings.md#te-fused-rope-is-faster-and-buys-nothing).

**FlashAttention-3 fork: not yet.** +2.85% step time, free and loss-neutral
([results](results.md#flashattention-3-fork-ab)), but the sep07 series is
wall-clock matched: a 2.85% faster step means 2.85% more tokens in 6 h, so arms
on the fork would not be comparable to t0, t0b, t1a, t1b or t1c, which all ran
on stock. Comparability is worth more than 2.85%. Adopt at the next clean
boundary (the 600M transfer, the final long runs, or a new series), and re-pin
`env/PINS.txt` with the FA3 commit beside it, because the token budget per run
changes.
