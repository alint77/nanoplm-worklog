# Tier 0: the noise floor

6 runs, NorMuon 7e-3 at the decided base (h1024/L32 399.6M, 4.19M tokens/step,
6 h on 4 nodes). All six passed the trace gate before being allowed to run
(exposed/compute 0.5% to 5.5%, threshold 8%), with the 8 known-bad nodes
excluded.

| run | seed | steps | wall-clock | equal-step (10500) |
|---|---|---|---|---|
| t0-rep1 | 42 | 11000 | 2.2041 | 2.2065 |
| t0-rep2 | 42 | 10500 | 2.2064 | 2.2064 |
| t0-rep3 | 42 | 10500 | 2.2063 | 2.2063 |
| t0-seed43 | 43 | 10500 | 2.2077 | 2.2077 |
| t0-seed44 | 44 | 10500 | 2.2058 | 2.2058 |
| t0-seed45 | 45 | 10500 | 2.2078 | 2.2078 |

## The numbers

| quantity | equal-step | wall-clock |
|---|---|---|
| sigma_repeat (3 identical runs) | **0.00010** | 0.00130 |
| sigma_seed (4 seeds) | **0.00097** | 0.00176 |
| **2 x sigma_seed** | **0.0019** | 0.0035 |

**sigma_repeat at equal steps is 0.00010.** Three byte-identical configs at the
same seed land within 0.0002 of each other over 44B tokens. Kernel
non-determinism is essentially nil at this horizon.

**The wall-clock column is 13x noisier for the SAME runs** (0.00130 vs
0.00010). Nothing differs between those three runs except which nodes they drew
and therefore how many steps they finished (11000 vs 10500 vs 10500). That is
the equal-step rule justified with a direct measurement rather than an argument:
comparing identical configs at the wall-clock stop manufactures 13x the noise.

**sigma_seed is 0.00097**, about 10x sigma_repeat. So run-to-run variation is
dominated by data order, not by kernels, which is the expected and healthy
ordering.

## Verdicts on the pending Tier 1 candidates

Threshold is the pre-registered 2 x sigma_seed = 0.0019, equal-step.

| candidate | delta | verdict |
|---|---|---|
| NorMuon over AdamW | -0.0289 | **WINS**, 15x the threshold |
| NorMuon wd 1e-5 (vs dion's 0.01) | -0.0050 | **WINS**, 2.6x the threshold |
| cautious_wd = true | -0.0018 | no measured effect |
| nesterov = true | +0.0007 | no measured effect |
| beta2 0.9 / 0.98 (NorMuon) | +0.0007 / +0.0019 | no measured effect |
| AdamW beta2 0.95 | +0.0003 | no measured effect |
| AdamW wd 0.01 | +0.0011 | no measured effect |

## Caveat on the sample size

sigma_seed comes from n=4 and sigma_repeat from n=3. A standard deviation from
n=4 is itself uncertain: the 95% interval on sigma runs roughly from 0.55x to
2.9x the estimate, so 2 x sigma_seed could plausibly be anywhere from 0.0011 to
0.0056.

That does not threaten the two wins. NorMuon over AdamW (-0.0289) clears even
the pessimistic end by 5x. NorMuon wd 1e-5 (-0.0050) clears the point estimate
by 2.6x and sits at about the pessimistic end, so it is adopted but flagged: if
it ever matters at the margin, it is worth two more seeds.

Nothing else in Tier 1 is distinguishable from noise at any reading of sigma.
