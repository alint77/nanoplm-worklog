# FlashAttention-3 fork A/B: +2.85% step time, loss-neutral

Independent single-node check of `alint77/flash-attention`, branch
`gh200-persistent-bwd` (`fa_gh200/fa_push`, head `af86f68`), against its own
merge-base. Run 2026-09-10.

## Result

Three arms, same nanoPLM config, same seed, same data order; only the FA3
kernels differ. 250 steps each, one node, 4 GPUs, arms strictly sequential and
interleaved `stock -> fork -> forkflag` x 2 repeats.

| arm | step time | vs stock | repeat spread |
|---|---|---|---|
| stock (merge-base `0251105`, = upstream main) | 1936.6 ms | - | 1.12 ms |
| fork (default build) | 1882.9 ms | **+2.85%** | 0.08 ms |
| fork + `FLASH_ATTENTION_SHORT_SEQ_TILES=TRUE` | 1877.9 ms | **+3.12%** | 0.34 ms |

The measurement is far tighter than the effect: within-run scatter is +-3 ms
and the same arm reproduces across repeats to **0.06-0.79 ms sigma (<=0.04%)**,
against a 54-59 ms effect. Roughly 50x the noise floor. The two repeats agree
to 0.06 and 0.08 percentage points.

**Loss is unaffected**, and the control is what establishes it:

| comparison | max abs delta loss over 50 shared steps |
|---|---|
| fork vs stock | 0.0013 |
| forkflag vs stock | 0.0010 |
| **stock vs stock** (identical kernels, same seed) | **0.0017** |

The same-arm delta is *larger* than either cross-arm delta, so the differences
are the pipeline's own run-to-run nondeterminism, not the kernels. Final loss
at step 250 is 2.6089 / 2.6089 / 2.6088 across all six runs: identical to four
decimals, which is exactly what the fork claims.

**Gradients match at the kernel level too.** On the shapes nanoPLM runs (350
packed sequences, 64,287 tokens, D=64, bf16), max abs difference against stock
is 7.8e-3 to 1.6e-2 on output, dq, dk and dv, for both a full-attention and a
sliding-window layer. One bf16 ulp at those magnitudes is 8.0e-3, so every
difference is 1-2 ulps of rounding.

## Why our number is 2.4x the fork's own

The fork's README reports **+1.21%** default and +1.51% with the flag, on a
4-GPU 16-layer ModernBERT with 10 sliding-window layers. We measure +2.85% and
+3.12%. Our model is **32 layers with 21 sliding-window** (`attn_layer_pattern:
null` = full every third layer), so there is roughly twice as much of the
changed path per step. That is the obvious explanation and it is consistent,
but it was not tested here.

The opt-in forward flag adds only **+0.27 percentage points** over the default
build. The README notes it regresses long sequences; at our 512-token cap that
does not bite, but the marginal gain is small enough that the default build is
the sensible choice.

## Does the 1-node number transfer to our 4-node runs?

Yes, and this was checked rather than assumed. The A/B config puts 65,536
tokens per GPU per micro-step with grad_accum 4, which is *identical per-GPU
work* to the 4-node series runs. Stock reads 1936.6 ms here; `t0-rep1` on four
nodes reads 1936-1945 ms. Nearly the same number, which independently confirms
the trace finding that exposed inter-node comms is only a few percent in this
configuration. Attention is therefore not a smaller share of a 4-node step, and
the speedup should carry over.

## Recommendation: do not swap mid-series

The gain is real, free, and loss-neutral, but the sep07 series is **wall-clock
matched**. A 2.85% faster step means a 6-hour run sees 2.85% more tokens, so
arms run on the fork would not be comparable to tier 0, 1a, 1b or 1c, which all
ran on stock. Comparability across the series is worth more than 2.85%.

Adopt at the next clean boundary: the 600M transfer run, the final long runs, or
a new series. Whoever does should re-pin `env/PINS.txt` and note the FA3 commit
beside it, because the token budget per run changes.

## Reproducing

- Builds: `fa_gh200/abx/{stock,fork,forkflag}`, built by
  `abx/build_min.sbatch`. Only the corner nanoPLM calls is compiled (varlen,
  bf16, headdim 64, MHA, sliding-window, fwd+bwd); the full 451-instantiation
  matrix takes ~8 hours, this takes 4 minutes. `LOCAL` must stay enabled.
- Two build traps: the fork calls `git submodule update` unconditionally in
  `setup.py` where the merge-base guards it, so compute nodes need
  `module load git`; and `csrc/cutlass` must be populated beforehand because
  compute nodes have no outbound network.
- A/B: `abx/ab.sbatch`, analysis `abx/analyse_ab.py`. Kernel parity:
  `abx/parity.py`.
- Each arm is selected with `PYTHONPATH=<tree>/hopper`, which overrides the
  installed `flash_attn_3`. **Verify the three `_C.abi3.so` md5s differ**
  (`3008de034885` / `d880c883f129` / `84db0afd17c4`): a PYTHONPATH that fails
  to override would make every arm identical and the A/B a silent no-op.
