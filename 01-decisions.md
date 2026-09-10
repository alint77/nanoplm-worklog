# Decisions

Date order. Each one says what we chose and why.

## Throw away the jul30 ablations

Started over on 2026-09-07.

Why: arms were compared against a spine that kept changing, and two of its
rungs were regressions. Nothing had a measured noise floor, so most of the
"wins" could have been noise. Not fixable after the fact.

## The paper's baseline is stock ModernBERT

Not our best-performing config.

Why: a paper needs a baseline other people recognise. Every claim is a delta
from stock ModernBERT.

## Wall-clock matched, not token matched

Each run gets 6 h of stable training on 4 nodes. About 96 GPU-hours per run.
The 30 min LR decay is a separate campaign, run later off the saved
checkpoints rather than chained behind each job.

Why: we care about loss per GPU-hour, not per token. If a change makes the
model slower, it gets fewer tokens in its 6 h and pays for the slowdown
itself. That is the honest comparison.

## Base shape: h1024 / L32 / 16 heads / 2688 MLP = 399.6M

Kept ESM C's proportions (aspect ratio 32, 8/3 MLP) but moved the width to
1024.

Why: we measured the ladder on 4 nodes.

| shape | aspect | params | step | MFU |
|---|---|---|---|---|
| h896 / L28 | 32.0 | 273.8M | 418.8 ms | 42.72% |
| h896 / L34 | 26.4 | 332.3M | 510.9 ms | 42.49% |
| h960 / L30 | 32.0 | 332.8M | 470.5 ms | 46.02% |
| h1024 / L27 | 37.9 | 337.3M | 427.4 ms | 51.24% |
| h1024 / L32 | 32.0 | 399.6M | 511.0 ms | 50.79% |
| h1152 / L36 | 32.0 | 574.8M | 771.5 ms | 48.01% |

Two things fell out of this:

1. Aspect ratio is nearly free. At the same width, going from aspect 37.9 to
   32 costs 0.45 MFU points.
2. Width is not free. 1024 divides the 128-wide Hopper GEMM tile and 960 does
   not. That gap is about 5 points.

Three goals - power-of-two width, ESM C's aspect ratio 32, ESM C-300M's size -
and we can have any two. We took the first two. The cost, which goes in the
paper: 399.6M instead of 332.8M, and ~16% fewer tokens in 6 h than h1024/L27
would give. Every arm pays it equally, so nothing inside the series is biased.

## Large model: h1152 / L36 = 574.8M

Same proportions. Needs `micro_batch_seqs: 64`; at 128 it OOMs (peak 97,268 of
97,280 MiB).

## UniRef50 only

Why: about 5.2 epochs of repetition at this budget. We considered adding
corpora and decided the extra variable was not worth it for an architecture
study.

## bf16 everywhere, one fp8 pair at the end

Why: fp8 is a separate question. Running it under every arm would confound the
architecture results. One bf16-vs-fp8 pair on the final recipe answers it.

## Logging and profiling

`logging_steps: 20`, `eval_steps: 500`, layerwise metrics every 20, profiler
traces on for every run.

Why: traces are cheap and we have already had two runs where the log alone
would have hidden the problem. Eval is 4.0 s a time, so every 500 steps costs
about 1.5% of a run rather than 3% at every 250. Eval loss is a secondary
signal now, so the cadence does not need to be tight.

## num_workers: 4, written out explicitly

Why: `"auto"` resolves from world size. At 4 GPUs it gives 16 workers per rank,
at 16 GPUs it gives 64, which is 256 processes per node and a host-RAM OOM.

## Coordinate-descent inductor tuning on

`TORCHINDUCTOR_COORDINATE_DESCENT_TUNING=1`.

Why: about +2.3 MFU points, and the cost is compile time only, which sits
outside the wall-clock budget.

## Biotrainer decides, not eval loss

The deciding metric is biotrainer PBC and PGym, run later on the stored
checkpoints. Biotrainer is not ready, so the ablations run now and get scored
afterwards.

Why it matters for planning: every checkpoint has to survive, which is on the
order of a terabyte across the series, and arms cannot be ranked until
biotrainer lands.

## Eval masking pinned and made deterministic

Eval used to inherit the whole training masking recipe, and redrew its masks
on every call.

Why: an arm trained at 15% masking was also *scored* at 15%, while the base
was scored at 30% - different tasks, incomparable losses, so any arm touching
the masking recipe was uninterpretable. Redrawn masks also put noise in every
eval number. Both fixed: eval is pinned at 15% token masking with 80/10/10 and
a fixed seed, for every arm.

## Untied embeddings in the base

ModernBERT ties the input embedding to the output head. We do not.

Why: tying exists to save parameters on LLM-sized vocabularies. At vocab 32 the
embedding matrix is 32 x 1024, so tying saves 32,768 parameters out of 399.6M,
which is 0.008%. The rationale does not transfer to a protein model.

The untied head is a (32, 1024) matrix and routes to the AdamW group, not into
Newton-Schulz, because `_is_embedding_or_unembedding_param` catches
`decoder.weight`. So this is safe under NorMuon.

The Tier 2 arm flips accordingly: it now tests *tied* embeddings, in case the
coupling helps for reasons other than parameter count.

## Settle the optimizer first, and use spectral_norm scaling

NorMuon looked much better than AdamW in the jul30 work, so the optimizer
phase moves to the front. Everything after it is a single-factor arm off
whichever optimizer wins.

Only AdamW and NorMuon are in it. Muon, NorDion2 and stable_adamw are dropped.

**The parameterization matters more than it looks.** NorMuon orthogonalizes the
update, then rescales it by a factor that depends on the matrix shape. There
are two choices, and dion's docstring says what each is for:

- `spectral_norm`: "for learning rate transfer across model scale"
- `rms_norm`: "for learning rate compatibility with Adam/AdamW"

The formulas:

- `rms_norm` = `lr * 0.2 * sqrt(max(fan_out, fan_in))`
- `spectral_norm` = `lr * sqrt(fan_out / fan_in)`

On our actual matrices at h1024:

| matrix | shape | rms_norm | spectral |
|---|---|---|---|
| QKV q-block | (1024, 1024) | 6.400 | 1.000 |
| Wo | (1024, 1024) | 6.400 | 1.000 |
| MLP up/gate | (5376, 1024) | 14.664 | 2.291 |
| MLP down | (1024, 2688) | 10.369 | 0.617 |

We chose **spectral_norm**, for three reasons.

1. We sweep the NorMuon LR ourselves, so "compatibility with AdamW's LR" buys
   us nothing. That is the only thing rms_norm is for.
2. We scale h1024 to h1152 later. Under rms_norm every effective LR drifts
   +6.1% on that step and has to be re-tuned. Under spectral_norm it moves
   -0.8% to 0.0%, which is the entire point of that scaling.
3. rms_norm uses `max(fan_out, fan_in)`, so it cannot tell a tall matrix from
   a wide one. The MLP down-projection gets 1.6x the square-matrix LR under
   rms_norm and 0.62x under spectral: a 2.6x difference in relative weighting.
   Under GQA the k-block `(512, 1024)` gets exactly the same LR as a full
   `(1024, 1024)` block under rms_norm, and 0.707x under spectral. For a study
   whose whole job is varying matrix shapes, rms_norm is not a principled
   choice.

dion defaults to spectral_norm for every optimizer it ships. nanoplm overrides
that to rms_norm to preserve older behaviour, and that older behaviour is the
jul30 series we discarded. We set it explicitly in our configs rather than
relying on either default.

**LR anchor.** rms_norm multiplies the square-matrix LR by 6.4 and
spectral_norm by 1.0. So an LR tuned at 1e-3 under rms_norm corresponds to
roughly 6.4e-3 under spectral for square matrices. The spectral sweep is
centred at 5e-3, not 1e-3, or it would sweep the wrong decade and NorMuon
would "lose" for no reason.

## Weight decay matched across the optimizer arms

The dataclass gives NorMuon `muon_weight_decay: 0.01` with cautious decay,
while our base runs AdamW at `1e-5` plain. Left alone, the optimizer
comparison would also have been a weight-decay comparison. Both are now 1e-5,
cautious off.

## Global batch size: 4.19M tokens (DECIDED)

Settled before the optimizer, because both optimizers' LR optima move with
batch size, so an LR sweep at one batch does not transfer to another.

**Measured, 26 runs, token-matched at 10.07B tokens each** (Tier 0b; not
wall-clock matched, so the batch effect is isolated). Best achievable eval loss
per batch, each at its own bracketed LR minimum:

| batch | best LR | eval @ 10.07B |
|---|---|---|
| 1.05M | 5e-3 | 2.3086 |
| 2.10M | 5e-3 / 7e-3 | 2.3118 |
| 4.19M | 7e-3 | 2.3200 |

**Chose 4.19M despite it being 0.0114 worse.** Four reasons, in order of
weight:

1. **The 0.0114 is an upper bound, not the cost at our horizon.** It was
   measured at 10.07B tokens; a real ablation run is ~44B. A large batch's
   per-step disadvantage shrinks as the number of optimizer steps grows, so the
   gap at 44B is smaller than 0.0114 and possibly zero. Confirming it exactly
   would take a pair of 6 h runs, which we judged not worth 200 GPU-hours.
2. **Scale-out headroom.** Global batch caps the world size that can run at
   full local batch. At `micro_batch_seqs` 128 and 512-token sequences each GPU
   takes 65,536 tokens, so 1M caps us at 16 GPUs (4 nodes) and 4M allows 64
   GPUs (16 nodes). Below that ceiling the local batch shrinks, GEMMs get
   smaller and utilisation drops. The ablations themselves run on 4 nodes
   either way; this is about the model we eventually train.
3. **It biases no comparison.** Every arm runs at the same batch, so the
   0.0114 is a constant offset on all of them, not a differential effect. What
   it buys is that the ablation results describe the batch we would actually
   train at.
4. **It matches ESM C's 4.2M**, verified from the ESM Cambrian blog (stage 1,
   512 context), which keeps our numbers comparable to the obvious reference
   point.

Explicitly NOT a reason: comms. Exposed (non-overlapped) NCCL is 5.2 ms of a
511 ms step, 1.0%, so gradient accumulation saves under 1% of throughput.
Anyone reaching for a bigger batch to save collectives at this scale is
reaching for nothing.

Consequences recorded for the writeup:

- `warmup_steps` drops from 1000 to 250, holding warmup at 1.05B TOKENS rather
  than at a step count. Keeping 1000 steps would have quadrupled warmup to 4.2B
  tokens, 9.5% of a 6 h run instead of 2.4%.
- grad_accum becomes 4 at 16 GPUs. A 6 h run is ~10,568 optimizer steps and
  ~44.3B tokens, against ~42,270 steps for the same tokens at 1M.
- The batch was chosen at 400M parameters. The 600M transfer runs in Tier 5
  have a different critical batch, and the writeup must say so.

## Learning rate: measured, not assumed

Tier 0b also produced the LR curves, and three things came out of them that
change how Tier 1 is run:

**The optimum barely moves with batch.** 5e-3 at 1M, 5e-3 at 2M, 7e-3 at 4M: a
factor 1.4 across a 4x batch change, roughly B^0.24. Neither sqrt(B) nor linear
scaling describes it, and both overshot badly at 4M (sqrt(B) put us at 2e-2,
costing 0.068; linear at 4e-2, costing 0.19). Our Tier 1 grids are therefore
anchored on measured optima, not on a scaling rule.

**NorMuon's LR curve is asymmetric.** The bottom is flat, 0.003 across 3.5e-3
to 7e-3 at 1M, but above the optimum it is punishing: 1e-2 costs 0.028 and 2e-2
costs 0.15. So a grid should sit at and below the optimum rather than straddle
it from above. An earlier plan anchored at 1e-2 with 2x spacing would have
ranked NorMuon by grid placement rather than merit.

**NorMuon beats AdamW at every batch by 0.05 to 0.06.** AdamW's best anywhere
was 2.3716. NorMuon's worst bracketed point beats it. An earlier reading that
the advantage vanished at 4M was purely an artifact of NorMuon being run above
its optimum there.

## Noise floor: sigma_repeat ~ 0.0004

Three same-config, same-seed pairs (an accident of two agent sessions
submitting overlapping grids): 2.3086/2.3089, 2.3100/2.3106, 2.3248/2.3250.

This measures kernel non-determinism only. **sigma_seed, from different data
order, is unmeasured and will be larger**, and it is sigma_seed that the
pre-registered 2-sigma win threshold refers to. Tier 0 still owes us that.

## MoE is compared at matched active parameters

Not matched total parameters.

Why: at equal active parameters the comparison is at equal compute per token,
which is what a wall-clock-matched study measures. Total parameters vary
between cells and that is expected and reported.

Concretely `top_k x expert_intermediate = 2688`, so every cell has the dense
base's 8,257,536 active MLP parameters per layer.

The control is dense *swiglu*, not the geglu base, because `use_moe=true`
refuses geglu. That is why tier 2b runs after 2a.

## mHC-lite is tested at both placements

DSv4 and GLM 5.3 Flash apply it per sublayer; nanoplm defaults to per layer
boundary. Rather than pick, both run.

No code change needed: `mhc_lite_wrapping_level` already accepts
`layer | sublayers`.

## TE fused RoPE: built, measured, reverted

Why: it works and it is about 2x faster at the kernel level, but the step time
is identical. See `04-findings.md`. Kept on branch `perf/te-fused-rope`, not
merged.
