# Model implementation

The training/inference implementation follows the manuscript's finite
observation model, Method and Algorithm 1. It was written from those
descriptions; historical trained weights are not supplied. The packaged CSV
results and their figure/table programs remain a separate reproduction path.

## Mathematical map

| Manuscript operation | Implementation |
|---|---|
| Retained evidence, joint whitening, fixed finest-grid likelihood | `evidence.Evidence` |
| IR/SR kernel on log-relaxation cells | `evidence.nmr_kernel` |
| Nested partitions and additive reference masses | `partitions.py` |
| Exact CIR corruption and analytic mass-weighted denoising target | `cir.corrupt`, `cir.denoising_target` |
| Shared boundary-aware flux `r = q + m h` | `model.SharedFlux` |
| Paired fine/coarse states and denoising on both | `training.Trainer.batch`, `losses` |
| Coarse regression to a stop-gradient fine EMA teacher | `training.Trainer.step` |
| Weighted conditional allocation and continuous inverse CDF | `refinement.py` |
| Frozen-coefficient nonnegative reverse update | `cir.positive_reverse_step` |
| Prescribed coarse-to-fine inference to time zero | `sampling.posterior_draws` |
| Lognormal reference posterior | `reference.py` |
| Correlated-noise held-out prediction | `evidence.predictive_distribution` |

One model sees the complete current partition. Each cell carries its current
mass, log-coordinate boundaries and additive reference mass. Self-attention
couples cells; cross-attention conditions them on all retained observations.
The observation encoder receives jointly whitened `(y, K)` with **all
finest-grid kernel columns**. A coarse state never introduces a replacement
coarse-center likelihood. Observation IDs represent physical acquisitions:
exact copies can be deduplicated, whereas independent repeats retain distinct
IDs even when their numerical values happen to agree.

The two training resolutions share one corrupted fine state: the coarse state
is its sum. Each resolution has its own analytic CIR denoising target. The
aggregation target sums the frozen teacher's fine flux, and regresses the
online coarse flux to it. It is not imposed as a pointwise identity between
fine and coarse online predictions. The first aggregation update initializes
the teacher from the current online model **before** taking the optimizer
step; subsequent EMA updates happen **after** the optimizer step.

During a split, every allocation candidate contains both proposed children
and all other current cells. The next split uses the updated full state. Only
the split preserves a parent sum; total amplitude remains random through
reverse evolution. The final reference-cell ordering is restored explicitly.

## Implementation choices not numerically fixed by the paper

`configs/model.json` makes these choices explicit:

| Quantity | Default |
|---|---|
| Token width / attention heads / interaction blocks | 64 / 4 / 2 |
| Evidence features | `asinh` of whitened signal and finest-grid kernel row, plus `log1p` of retained observation count |
| Cell features | `log1p(m/a)`, `log(a)`, four normalized bounds, time |
| Time encoding | Raw normalized time and four sine/cosine frequency pairs |
| Feed-forward activation and expansion | SiLU, four times token width |
| Residual output | `20 tanh(head)`; initialized near stationary residual `-2` |
| CIR constants | `kappa = sigma = 1` |
| CIR reference masses | Finest log-cell areas, normalized to total mass 1 |
| Terminal / minimum training time | 5 / 0.01 |
| Time distribution / positive loss weight | Uniform / constant 1 |
| Denoising reduction | Mean over examples of squared cell-wise L2 norms; average the two resolutions |
| Minibatch construction | Uniformly select one training evidence case, then sample its reference draws |
| Reference sampler | Elliptical slice sampling in Gaussian log-density coordinates |
| Kernel construction helper | Four-point Gauss–Legendre cell averaging along each log coordinate |

The paper's optimizer, warm-up, EMA, draw budgets and validation rule are read
from `configs/training.json`; its inference schedule comes from
`configs/protocol.json`. The six-trial tuning grid and fold roles are recorded
there, but a `train` invocation runs **one specified fold and seed**. Separate
invocations and explicitly prepared manifests are needed for other folds,
seeds or tuning candidates. Baseline networks are not implemented here.

The observation-count feature lets the normalized cross-attention distinguish
an acquisition from independently repeated identical rows, which carry extra
likelihood information. Physical copies removed by ID do not increase this
count.

The reference gamma law initializes the reverse sampler at finite terminal
time; it is an approximation to the conditional terminal law. It does not
replace the lognormal prior used for posterior reference sampling. The Python
sampling API also accepts explicitly supplied coarsest terminal draws.

## Input format

Download the original field/laboratory measurements from the
[Zenodo record](https://zenodo.org/records/14947500) or
[complete ZIP (286.5 MB)](https://zenodo.org/api/records/14947500/files/Mashhadi%20et%20al%20(T1-T2%20Data).zip/content).
Extract under `data/raw/` and place prepared case files and manifests under
`inputs/`. [The data guide](../data/README.md) gives the B10 paths inside the
archive. The training/reference command help also displays these links.

Provide one `.npz` file per evidence set. NumPy object arrays/pickle are not
used. All cell arrays flatten the fixed grid in row-major order: T1 first, T2
second. Acquisition times for `nmr_kernel` are **seconds**; edge coordinates
are **log10(seconds)**. Supply the calibrated kernel and covariance in the
same signal units as `y`; numerical normalization is not inferred from test
values.

| NPZ key | Shape / meaning |
|---|---|
| `y` | `[O]` signed context observations |
| `K` | `[O,J]` fixed reference-grid response |
| `covariance` | `[O,O]` context noise covariance |
| `log_t1_edges`, `log_t2_edges` | Strictly increasing reference-grid edges |
| `retained_mask` (optional) | Boolean `[O]`; remove missing rows before whitening |
| `observation_ids` (optional) | String `[O]`; shared IDs only for exact copies |
| `reference_draws` (training) | `[draw,J]` or `[chain,draw,J]` posterior **mass** samples conditioned only on retained context |
| `reference_chain_draws` (reference output) | Full `[chain,draw,J]` samples retained for diagnostics, separate from the training budget |
| `reference_selected_indices` (reference output) | `[training draw,2]` source chain/draw indices of the selected training subset |
| `reference_retained_indices` (optional) | Retained original indices used by the reference sampler |
| `y_holdout` (validation/evaluation) | `[H]`; never sent to the model or reference sampler |
| `heldout_train_ids` (validation) | String or integer `[H]` identifying each target's recovery train; required for per-train RMSE |
| `K_holdout` | `[H,J]` |
| `covariance_hh` | `[H,H]` |
| `covariance_ho` (optional) | `[H,O]` in original observation order; omitted means independent context/held-out noise |

Held-out observations are supplied separately from context. When cross-noise
covariance is present, prediction uses the conditional Gaussian mean and
Schur-complement covariance. Spectral intervals use posterior mass draws;
signal-predictive intervals additionally include conditional observation noise.

Example manifest (paths relative to the manifest):

```json
{
  "cases": [
    {"record_id": "train_01", "site_id": "site_A", "well_id": "well_A", "mask_id": "0", "role": "train",
     "scale": 10.0, "path": "train_01_with_references.npz"},
    {"record_id": "val_01", "site_id": "site_B", "well_id": "well_B", "mask_id": "0", "role": "validation",
     "scale": 10.0, "path": "val_01.npz"},
    {"record_id": "test_01", "site_id": "site_C", "well_id": "well_C", "mask_id": "0", "role": "test",
     "scale": 10.0, "path": "test_01.npz"}
  ]
}
```

Keep wells/specimens and shared sites together when preparing folds. The
loader rejects a site assigned to multiple roles. Training reads only train
and validation cases, with test and recalibration records excluded. Distinct
`mask_id` values allow multiple evidence masks for the same physical record;
`(site_id, well_id, record_id, mask_id)` must be unique. Use the specimen ID as
`well_id` for laboratory data. Validation requires this grouping metadata and
does not silently collapse all targets into one recovery train.

Validation computes normalized RMSE **separately for each recovery train**, then
equally averages trains, masks within each depth record, records within each
well, wells within each site, and sites. Strictly smaller scores replace
`best.pt`, preserving the earliest checkpoint on an exact tie. Its weights are
the **online** network.

The acquisition CSVs included for manuscript plotting are selected echoes.
Full field training requires the original acquisition inputs, specified
finest-grid edges, calibrated kernel/noise settings and fold manifests. The
packaged spectrum/score summaries are not a substitute for posterior training
draws. The `demo` command writes a complete, small example of the NPZ schema.

## Running the model

Install `requirements-model.txt` after selecting the desired PyTorch CPU/CUDA
build. Python 3.12, PyTorch 2.11, SciPy 1.17 and NumPy 2.3 were used locally.

```bash
# A newly sampled four-cell example: reference sampling, 200 training updates,
# validation selection, 16 posterior draws and marginal summaries.
python main.py demo --out outputs/model_demo

# Short pipeline check (also run by CI):
python main.py demo --steps 8 --out outputs/quick_demo

# Reference sampling from retained evidence under the lognormal prior:
python main.py reference --case inputs/train_01.npz \
  --out inputs/train_01_with_references.npz --chains 4 --draws 512 \
  --draws-per-chain 512 --burnin 1000

# One fold/seed with the paper's training and refinement settings:
python main.py train --manifest inputs/fold_01.json --out runs/fold01_seed1001 --seed 1001

# Stop at an absolute update and resume without changing the learning-rate curve:
python main.py train --manifest inputs/fold_01.json --out runs/interrupted --stop-after 2000
python main.py train --manifest inputs/fold_01.json --out runs/interrupted \
  --resume runs/interrupted/latest.pt

# Infer using the validation-selected online model:
python main.py infer --checkpoint runs/fold01_seed1001/best.pt \
  --case inputs/test_01.npz --out outputs/test_01_posterior.npz --draws 512
```

For the small example, training configurations and a compatible schedule are
written to `outputs/model_demo`. To use those manually, pass `--model-config`,
`--training-config` and `--schedule`; the paper's default 4→8→16 grids are not
compatible with a 2×2 example grid. Every schedule ends at the input reference
grid, doubles both dimensions between stages, and has one decreasing split
time per stage. Binary boundaries are inherited from the supplied finest
log-grid; each parent spans an even number of reference cells per split axis.
Do not infer the historical nonuniform edges from the domain endpoints alone.

`reference` retains all chains in `reference_chain_draws` and reports
rank-normalized R-hat and bulk/tail ESS for total and specified regional
amplitudes. `--draws` specifies the **total training budget per evidence set**
(512), whereas `--draws-per-chain` specifies the diagnostic chain length
(default 512 on each of four chains). A fixed, seed-controlled, chain-balanced
subset of 512 is selected with one draw from each temporal stratum; its source
indices are saved. The trainer checks the configured reference budget instead
of training on every diagnostic draw. The `reference` command exits with code 2 if the
paper's thresholds (`R-hat < 1.01`, both ESS `> 400`) are unmet; sampled chains
are retained. Regions must align with reference cell boundaries. Increase the
sampling budget as needed; a requested draw count alone does not establish
convergence. `demo` uses short reference runs solely to exercise the software.

`history.csv` records actual losses and validation scores. `latest.pt` contains
online and teacher weights, optimizer state, update count, configurations,
best score **and selected online weights**, data roles and NumPy/PyTorch RNG states.
Thus resuming into a new run directory preserves the previous selection even
if no subsequent checkpoint improves it. `best.pt` is an inference-only
snapshot; continue optimization from `latest.pt`. Same-environment CPU
continuation is tested against an uninterrupted next update. Resume assumes
the underlying input files are unchanged; a seed does not ensure identical
trajectories across devices or software versions.

Checkpoint selection records its metric definition. Checkpoints selected by
the earlier flattened-echo RMSE remain readable for inference, but cannot carry
that old score into a resumed run using the corrected hierarchical metric.

## Numerical choices and checks

The main sampler uses float32 neural evaluation and float64 CIR transitions,
allocation and summaries. Conditional allocation integrates the beta endpoint
weights with weighted quadrature. It interpolates the bounded neural
potential on successively refined Chebyshev grids (up to 513 nodes by default).
The potential approximation tolerance (`5e-5`) is separate from the paper's
weighted integration tolerance (`1e-10`). Inverse CDFs are solved in beta
probability coordinates, not sampled from a discrete allocation grid. Very
small beta shapes can place a child below floating-point resolution. Failed
potential convergence or quadrature is reported rather than silently replaced
by a uniform split. This direct implementation favors explicit scientific
operations; no claim is made that its runtime matches the historical CSVs.

Command defaults read `configs/runtime.json`: 8 PyTorch intra-operation
threads, 1 inter-operation thread, and at most 32 state rows per neural
sampling call. `--threads`, `--interop-threads`, and `--sample-microbatch`
provide explicit overrides. The evidence is encoded once per evidence set;
reverse updates and allocation-candidate evaluations reuse that encoding.
The random-number order is independent of neural microbatch boundaries,
although neural floating-point roundoff can vary with batch size.

The tests cover analytic CIR scores, transition moments and aggregation,
positive steps including zero frozen coefficient, singular beta endpoints,
known tilted allocation laws, complete-state conditioning, parent mass
preservation, evidence whitening/deduplication, correlated predictive noise,
teacher stop-gradient and update order, fixed-batch learning, checkpoint
continuation, and posterior summaries. A CUDA forward–loss–backward–update and
reload test runs when CUDA is available. These checks and the small example
do not constitute the paper's full multi-site/five-seed evaluation.

## Numerical references

- Murray, Adams and MacKay, [Elliptical slice sampling](https://arxiv.org/abs/1001.0175).
- SciPy: [noncentral chi-square](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ncx2.html)
  and [weighted quadrature](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.quad.html).
- ArviZ: [R-hat](https://python.arviz.org/en/v0.22.0/api/generated/arviz.rhat.html)
  and [effective sample size](https://python.arviz.org/en/v0.22.0/api/generated/arviz.ess.html).
