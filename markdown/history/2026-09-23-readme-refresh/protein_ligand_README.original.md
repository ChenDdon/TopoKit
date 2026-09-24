# Protein–ligand prediction: one selected application

## Final topology ML selection — September 23, 2026

The final **FS-AN topology GBDT** averages predictions from seeds **0, 1 and 2**.
Each model uses 10,000 trees, learning_rate=0.005, depth7, min_samples_split=2,
subsample=0.4 and max_features=sqrt, with its own training-only StandardScaler.
All **12 models/scalers and six ensemble evaluations** are delivered and verified.
The default predictor is the **general-v2020R1 ensemble**; matching refined-set
ensembles are retained for the three historical benchmarks.

Feature selection used the **previous parameters**. Historical ablations retain
their original parameters, predictions and scores; the fixed FS-AN tensors and
memberships are unchanged. The new hyperparameters and three-seed consensus
produce the **final ML results**.

[Final parameters and model bundles](../../../datasets/protein_ligand_prediction/final/topology_gbdt/README.md) ·
[Final results](../../../datasets/protein_ligand_prediction/final/topology_gbdt/results/FINAL_REPORT.md) ·
[RMSE](../../../datasets/protein_ligand_prediction/final/topology_gbdt/results/RMSE.md) · [PCC](../../../datasets/protein_ligand_prediction/final/topology_gbdt/results/PCC.md).

## Final TopoFormer architecture — September 23, 2026

The user selected **A03** as the protocol's final DL architecture: width 768,
12 encoder layers, 12 attention heads, feed-forward width 3072 and 86,071,297
parameters. [Final profile, models and scalers](../../../datasets/protein_ligand_prediction/final/topoformer_a03/README.md)
provide the canonical entry point. The existing full predictors used
102/104/92 epochs; all three seed bundles and their prediction ensemble are retained.

[Unified DL ablation record](../../../datasets/protein_ligand_prediction/experiments/protocol_ablation/studies/ST-08/README.md)
organizes 20 candidate settings, 34 unique validation fits, six full fits and
24 CASF rows, with separate RMSE/PCC tables. Both original studies remain
archived in place. A03 wins the validation comparison and CASF-2007, while the
six-layer control retains better RMSE/PCC on CASF-2013/2016. Fewer epochs does
not imply less wall time; the explicit user selection fixes capacity, not a
claim of universal superiority.

[Active A03 tuning study](../../../datasets/protein_ligand_prediction/experiments/topoformer_a03_tuning_20260923/README.md)
started on both Cornell RTX3080 GPUs on September23 at17:58UTC. The fixed
16,648/1,850 split tests batch16/32/64 × actual LR3e-5/1e-4/2e-4, followed by
seed confirmation and weight-decay comparisons, up to25 new fits. Both maximum
epochs and cosine horizon are200, with10-epoch warmup and validation patience50.
The controller also runs a common-validation training-fraction diagnostic and
three full18498-row refits before CASF evaluation. All42 preflight tests and nine
GPU profiles passed. Optimizer selection is pending; the delivered A03 models,
feature strategy, GBDT and sequence selections retain their existing provenance.

## Final sequence configuration — September 23, 2026

The user selected **ESM-2 + CPZ, FS-AU NMI-inspired sqrt GBDT**, as the final
sequence ML configuration. [Final parameters, models, scalers and results](../../../datasets/protein_ligand_prediction/final/sequence_esm2_cpz/README.md)
provide the canonical saved profile: 10,000 trees, learning_rate=0.005,
max_depth=7, min_samples_split=2, subsample=0.4, max_features='sqrt',
random_state=None, with a training-only embedded StandardScaler.
All four model/scaler bundles, 19,066 × 1,792 embeddings and six evaluations are
preserved; the general-v2020R1 bundle is the primary reuse entry point.

[Previous-parameter FS-AW results and settings](../../../datasets/protein_ligand_prediction/experiments/sequence_esm2_cpz_previous_20260923/ARCHIVE.md) are archived in place,
with all existing artifacts retained. This explicit user selection supersedes
the historical comparison status below. Frozen scientific records and the
separate FS-AN topology selection remain unchanged. Monitoring is retired.

## Completed: previous GBDT settings on retained CPZ (FS-AW)

[FS-AW final report](../../../datasets/protein_ligand_prediction/experiments/sequence_esm2_cpz_previous_20260923/results/FINAL_REPORT.md) records four complete 10,000-tree models/scalers
and six evaluations, independently delivered locally at 15:33 UTC on September23.
The same 19,066 × 1,792 CPZ feature matrix was reused byte-for-byte. All resolved
parameters match the four original FS-AQ previous-setting model receipts.

The NMI-inspired settings give lower RMSE/MAE and higher PCC on all three
general-set tests and refined-2016. Previous settings improve all three metrics
on refined-2007. Refined-2013 is mixed: previous settings improve PCC, while
NMI-inspired settings improve RMSE and MAE. Overall NMI-inspired settings have
lower RMSE/MAE in five of six rows and higher PCC in four of six rows.

Separate [RMSE](../../../datasets/protein_ligand_prediction/experiments/sequence_esm2_cpz_previous_20260923/results/RMSE.md), [PCC](../../../datasets/protein_ligand_prediction/experiments/sequence_esm2_cpz_previous_20260923/results/PCC.md) and [MAE](../../../datasets/protein_ligand_prediction/experiments/sequence_esm2_cpz_previous_20260923/results/MAE.md) tables include all six
refined/general evaluations and signed differences.

This compares complete configurations, including learning_rate .005→.002,
min_samples_split 2→5, subsample .4→.8 and random_state None→0. Both use
10,000 trees, depth7, sqrt and identical features/memberships with training-only
scaling. One fit per training set does not establish significance or isolate
individual effects. RMSE/MAE use logKa/pK, not kcal/mol. FS-AU remains retained;
no automatic replacement or CASF selection. FS-AV stays archived with model
assets intentionally absent. Historical source and delivery receipts are preserved.

Delivery and documentation are complete; the existing monitor is retired.

## Current retention — September 23, 2026

The user retained **FS-AU, ESM-2 + CPZ**, for future sequence-model use: all
19,066 embeddings, four trained pipelines with embedded scalers and standalone
scaler arrays, results, encoder assets and required parent caches are preserved.
[CPZ reuse guide](../../../datasets/protein_ligand_prediction/experiments/sequence_esm2_cpz_sqrt_20260923/REUSE.md). **FS-AV, ESM-2 + ChEMBL27-only**, is archived with
settings, results and predictions retained; its four model bundles and standalone
scalers were removed from both Mac and Cornell (139,448,346 bytes per machine).
[Archive and pruning records](../../../datasets/protein_ligand_prediction/experiments/sequence_esm2_chembl27_sqrt_20260923/ARCHIVE.md). Earlier completion descriptions record
the successful delivery before pruning. Frozen scientific receipts remain unchanged;
ARCHIVE.json records current availability. Earlier FS-AQ/FS-AR and selected
FS-AN are preserved. FS-AW is also delivered and documented; monitoring is retired.


## Completed sequence revision: sqrt GBDT (September 23, 2026)

Both user-requested ESM-2 studies are complete and independently verified locally
as of 13:40 UTC: **FS-AU (CPZ)** and **FS-AV (ChEMBL27)**. Each delivered all
19,066 × 1,792 float32 feature rows (protein 1,280 + ligand 512), four full
10,000-tree model/scaler bundles and six CASF evaluations. All eight fits ran
concurrently on Cornell CPU; no embeddings or memberships were changed.

GBDT uses depth 7, learning rate 0.005, min_samples_split=2, subsample=0.4,
and max_features='sqrt' (resolved parameter 42). Remaining sklearn 1.9.0 defaults
include random_state=None; actual pre-fit RNG states are saved. Scaling uses
training data only. No early stopping, holdout, ensemble, tuning or target scaling.

[CPZ final report](../../../datasets/protein_ligand_prediction/experiments/sequence_esm2_cpz_sqrt_20260923/results/FINAL_REPORT.md) · [ChEMBL27 final report](../../../datasets/protein_ligand_prediction/experiments/sequence_esm2_chembl27_sqrt_20260923/results/FINAL_REPORT.md)

General-v2020R1 model results:

**RMSE (logKa/pK; lower is better)**

| Training → test | ESM-2 + CPZ (FS-AU) | ESM-2 + ChEMBL27 (FS-AV) |
|---|---:|---:|
| general_v2020R1 → casf_2007 | 1.401586 | 1.442955 |
| general_v2020R1 → casf_2013 | 1.389762 | 1.435526 |
| general_v2020R1 → casf_2016 | 1.198154 | 1.252878 |

**PCC (higher is better)**

| Training → test | ESM-2 + CPZ (FS-AU) | ESM-2 + ChEMBL27 (FS-AV) |
|---|---:|---:|
| general_v2020R1 → casf_2007 | 0.826199 | 0.813440 |
| general_v2020R1 → casf_2013 | 0.805119 | 0.789548 |
| general_v2020R1 → casf_2016 | 0.851202 | 0.837198 |

CPZ has lower RMSE and higher PCC in all six matched evaluations, including
the three refined models in the full reports. These are single stochastic
fits, so this does not establish significance or isolate every source of variation.
Both reports contain separate RMSE/PCC comparisons with FS-AQ, FS-AR and FS-AN;
FS-AU also labels the old FS-AS refined results explicitly as partial.

All model/feature hashes, pooling/cache provenance, memberships, labels,
training-only scaler arrays and metrics passed local verification without
unpickling incompatible remote models locally. Old max_features=None FS-AS
is preserved as partial; FS-AT was canceled while queued and must not restart.
Completed FS-AR/FS-AQ and selected FS-AN remain unchanged. The historical NMI
max_features attribution is not independently established. Monitoring has
completed its delivery scope; historical receipts remain intact.

**FS-AN** is the selected feature strategy within TopoKit's sixth module,
`workflows`. The installed API is `topokit.workflows.protein_ligand_prediction`.
It fixes the completed **15 Å protein crop / alpha radii 0.1–5.0 Å** recipe,
55 channels, ten summaries and 27,500 features. GBDT is the trained downstream
model; its fitted StandardScaler stays inside its pipeline.

[Feature selection receipt](REPRESENTATIVE_STRATEGY.json) preserves the original
recipe `hpcc-alpha15-cb4760366e92bd2e`. [Model selection](MODEL_SELECTION.json)
identifies the **general-v2020R1 three-seed ensemble** as the default
final predictor. Its trained companion bundle is
[`pretrained/representative_gbdt`](../../../datasets/protein_ligand_prediction/pretrained/representative_gbdt/).
Weights and scientific datasets stay outside the wheel; the reusable feature
encoder, model profile and guarded prediction API are installed in the package.
Final refined-set ensembles are saved alongside the general ensemble; historical
single-model ablation scores remain in their original study folders.

All ablation variants, old schedulers and historical feature engines are organized
outside TopoKit in the [protocol study catalogue](../../../datasets/protein_ligand_prediction/experiments/protocol_ablation/README.md).
It gives every complete strategy an FS letter ID and each study an ST number;
old m/r labels remain aliases. **FS-Y** is the historical r4/PCC leader and
**FS-AJ** the preceding 20 Å native reference. FS-AN was selected for its smaller
field and similar results, not because it wins every benchmark. All 19,066 features,
four benchmark fits and six evaluations have been delivered and verified locally.
[Selected results](../../../datasets/protein_ligand_prediction/results/representative_alpha15/RESULTS.md).

The [optional supervised TopoFormer module](dl/README.md) is a downstream application
using these same features. Its hyperparameter ablation lives in
[ST-08](../../../datasets/protein_ligand_prediction/experiments/protocol_ablation/studies/ST-08/README.md);
the general-v2020R1 study has completed23 fits and12 CASF evaluations. The selected
six-layer model has three audited, locally delivered predictor bundles with
their fitted scalers. [DL results](../../../datasets/protein_ligand_prediction/experiments/topoformer_general_v2020R1_20260922/reports/FINAL_REPORT.md)
report individual seeds, seed mean/SD and the prediction ensemble separately.

The [article-informed follow-up](../../../datasets/protein_ligand_prediction/experiments/topoformer_article_followup_20260922/reports/FINAL_REPORT.md)
is complete: 14 screening/confirmation fits, three full refits and 12 CASF
evaluation rows, with all models/scalers and predictions verified locally.
Validation selected A03: width 768, 12 layers, 12 heads and feed-forward width
3072; mean validation RMSE/PCC are 1.171782/0.772468. Full refits used
102/104/92 epochs on all 18,498 general-v2020R1 rows. Its prediction ensemble
scores RMSE/PCC 1.351085/0.831628, 1.450729/0.771897 and 1.206813/0.840541 on
CASF-2007/2013/2016. It improves CASF-2007, while the earlier six-layer control
retains lower RMSE and higher PCC on CASF-2013/2016. Both model sets remain
available; no workflow default is changed by this mixed comparison.

## Feature definition

1. **Inputs and protein field.** Read full-protein PDB `ATOM` records for
   C/N/O/S and the explicit ligand MOL2 atoms. Keep protein atoms with minimum
   distance **≤15 Å** to any supported explicit ligand atom. Supported ligand
   elements are C/N/O/S/P/F/Cl/Br/I/H. Explicit H is retained in both the
   feature categories and crop anchors; missing H is not inferred. Existing
   input bytes, including the provenance-pinned `6djc` normalized MOL2, are
   preserved unchanged on disk. After cropping, inspect the combined selected
   protein-then-ligand list and keep the **first atom at each exactly equal
   coordinate**. Keep its element and component labels; do not average or lift
   edges back to removed atoms. Deduplicate across element labels and both
   components **before creating channels**. Thus a protein/ligand coordinate
   collision keeps the protein row. Within a component, reader order wins.
   Distinct nearby coordinates are never rounded or merged. Crop anchors use
   the original ligand sites, so coordinate multiplicity cannot alter the
   distance test. Each JSON record lists the removed and retained atom rows,
   labels and coordinates, along with input/unique counts. Original coordinates
   of retained atoms determine alpha births.
2. **Categories.** Protein groups are `{null,C,N,O,S}`; ligand groups are
   `{null,C,N,O,S,P,F,Cl,Br,I,H}`. Their protein-major Cartesian product gives
   **55 channels**: 40 mixed channels, 4 protein-only, 10 ligand-only and one
   intentional null/null zero channel. Protein-only means the selected
   element in the cropped field, not the entire protein. `null` omits that
   side; it does not mean `all`.
3. **Geometry.** Construct alpha geometry separately on each selected cloud,
   using the public `native` backend, engine **1.1.1**. TopoKit computes
   circumspheres, empty-ball decisions and coface propagation; SciPy supplies
   candidate triangulation. Rational checks repair ill-conditioned cases;
   bounded small-cloud retriangulation is available without jitter or GUDHI.
   This is floating-point geometry with rational repair, not a universal
   exact-predicate guarantee. Process full coface births before taking the 1-skeleton. In mixed
   channels build on the selected protein/ligand union, then retain **only
   cross-component edges**. In null-side channels retain the selected
   component's internal edges. Delaunay support with an edge-length cutoff
   is not this alpha filtration. There is no r1 complete-connection override.
4. **Filtration.** Observe exactly **50 alpha radii `0.1,0.2,…,5.0 Å`**,
   including 5.0. These are physical radii, not squared values or diameters.
   The equivalent diameter grid is `0.2,0.4,…,10.0 Å`. Include an edge
   when its squared alpha birth is **≤ radius²**; do not divide by two again. These births are not
   Laplacian weights: every admitted edge has unit weight and is oriented
   once by row order. Keep isolated selected vertices.
5. **Spectrum.** Compute the complete ordinary unweighted Hyperdigraph L0
   spectrum at each scale. Build alpha once per channel and use the public
   incremental `L0Sweep`; insert each edge once and reuse summaries when the
   active edge set has not changed. These are successive ordinary L0
   spectra, not a two-scale persistent Laplacian. No L1/L2 features are used.
6. **Encoding.** Let `λ+` contain eigenvalues `>1e-10`, `μ=mean(λ+)`, and
   `MAD=mean(abs(λ+−μ))`. Zero eigenvalues satisfy `abs(λ)≤1e-10`.
   Store the following ten statistics in this exact order:

   | Index | Statistic | Scope |
   |---:|---|---|
   | 0 | Matrix size | Full spectrum, including isolated vertices |
   | 1 | Number of zero eigenvalues | Full spectrum |
   | 2 | Mean | Positive eigenvalues |
   | 3 | Minimum / mean | Positive eigenvalues; r5 is not applied |
   | 4 | MAD / mean | Positive mean absolute deviation, not standard deviation |
   | 5 | Maximum / mean | Positive eigenvalues; r5 is not applied |
   | 6 | 25th percentile | Positive eigenvalues; linear interpolation |
   | 7 | Median | Positive eigenvalues; linear interpolation |
   | 8 | 75th percentile | Positive eigenvalues; linear interpolation |
   | 9 | Spectral entropy | `−Σp ln(p)`, `p=λ+/Σλ+`; natural logs, no `log(k)` division |

   Empty positive spectra give zero positive-only summaries. A valid cloud
   with no edges still retains its matrix size and zero count. An absent
   required non-null element partner makes its entire channel zero.
   Null/null is always zero. Geometry failures are recorded and block any
   model requiring those samples; they are never encoded as missing-channel zeros.

The stored tensor is C-order **float32 `(10,50,55)`**, computed in float64.
Flattening yields **27,500 features**, with channels varying fastest. Presence
and atom-count diagnostics are stored in JSON, not appended to model inputs.
The null/null channel contributes 500 fixed zero coordinates. Ligand H and
normalized min/max remain included. The only changes from the previous
20 Å reference are the 15 Å crop and the alpha-radius grid shifted to 0.1–5.0 Å;
the explicit parameters in the selection receipt define this version.

## Dataset names, standard counts and exclusions

**2020 means the updated official PDBbind v2020R1 release in this workflow.**
It is explicitly named `general_v2020R1` throughout. Its local official
general index has **19,037 IDs**. Remove the **539** IDs appearing in the
three-CASF union to obtain **18,498 training samples**. The CASF union has
540 IDs because `1xd1` is retained as an archive-external CASF fallback.
This is the user-selected v2020R1 protocol, not the original 18,904-row
TopoFormer v2020 training membership.

| Model | Training membership | Train | Evaluation | Test |
|---|---|---:|---|---:|
| `refined_2007` | v2007 refined minus CASF-2007 | 1,105 | CASF-2007 | 195 |
| `refined_2013` | v2013 refined minus CASF-2013 | 2,764 | CASF-2013 | 195 |
| `refined_2016` | v2016 refined minus CASF-2016 | 3,772 | CASF-2016 | 285 |
| `general_v2020R1` | v2020R1 general minus union of all three CASF sets | 18,498 | CASF-2007 / 2013 / 2016 separately | 195 / 195 / 285 |

Refined memberships are complete: 1,300−195, 2,959−195 and 4,057−285.
CASF-2016 is the 285-complex benchmark, not the distinct 290-complex PDBbind
core definition. Refined models remove only their matching CASF edition;
the v2020R1 model removes all three. No feature-availability intersection or
silent sample exclusion is permitted. The union of all seven manifests is
**19,066 unique complexes**, each with exactly one shared tensor.

`labels/final/` preserves the previously audited NMI/TopoFormer point targets
and row order. The selected v2020R1 release defines general-set membership and
updated structure provenance; using it does **not** silently replace the
already audited affinity targets with a different label source. Historical
refined memberships reuse updated v2020R1 structures plus the 28 audited
historical recoveries. This is not a claim of byte-identical original
2007/2013/2016 release structures. All selected protein/MOL2 hashes are checked.

## Fixed machine learning

Fit `Pipeline(StandardScaler, GradientBoostingRegressor)` on all declared
training rows, with unscaled pK targets. The scaler sees training rows only;
each test set reuses that fitted scaler inside `predict`.

| Setting | Value |
|---|---|
| Trees | 10,000 |
| Learning rate | 0.002 |
| Maximum depth | 7 |
| Features sampled at each split | `sqrt` (165 of 27,500) |
| Minimum samples to split | 5 |
| Subsample fraction | 0.8 |
| Random seed | 0 |
| Early stopping | Disabled (`n_iter_no_change=None`) |
| Inner validation / tuning | None; strategy and parameters are fixed |

These are the refined-v2/ablation settings for **all four models**, including
v2020R1. They differ from the archived early general-v2 demonstration's
learning rate 0.01. Four independent serial GBDT fits can run in parallel;
the one v2020R1 fit is reused for its three test sets. Reports contain six
evaluation rows with separate RMSE, PCC and MAE. CASF-2016 was used in the
earlier exploratory strategy selection, so its reused score is not an
independent confirmation or a statistical significance claim.

## Installed feature and prediction API

Install TopoKit from its package directory (`python -m pip install -e .`).
Feature generation requires NumPy/SciPy and no GUDHI. For one complex:

```python
import numpy as np
from topokit.workflows.protein_ligand_prediction import featurize, schema

tensor = featurize("protein.pdb", "ligand.mol2")  # float32, C-order (10,50,55)
np.save("complex_alpha15.npy", tensor)
print(schema()["recipe_id"])
```

For diagnostics use `read_selected_atoms(...)` followed by `compute(...)`,
which returns the float64 tensor, channel-presence mask and atom counts.
`implementation_receipt()` identifies the installed source separately from the
unchanged reference schema; its historical engine hashes always refer to the
original frozen source, not to a renamed package file.

For prediction on the existing audited feature store, activate the pinned
[model runtime](ml/environment.yml), install TopoKit there, then use paths from
the workspace root:

```python
from topokit.workflows.protein_ligand_prediction import predict

root = "datasets/protein_ligand_prediction/"
predict(
    root + "pretrained/representative_gbdt",
    root + "features/representative_alpha15",
    root + "labels/final/test_casf_2016.csv",
    "predictions_casf2016.csv",
)
```

`predict` accepts a trusted model bundle, a feature store containing its schema,
per-sample records/tensors, and a manifest with unique `pdb_id` values. It checks
runtime, hashes and shapes, and scales raw inputs exactly once. It refuses failed,
missing or incompatible features and an existing output file. Saving a standalone
NPY with `featurize` is not by itself an audited feature store; callers must also
supply the schema and provenance records required by the prediction loader.

`representative_strategy.py` and `ml/predict.py` are thin compatibility/CLI wrappers
around the installed implementation. There are no alternate feature switches.
No package code imports a dataset study or a historical workflow.

Exact historical reproduction uses the frozen source/environment in
[ST-07](../../../datasets/protein_ligand_prediction/experiments/protocol_ablation/studies/ST-07/README.md),
not the current package file hashes. Old Cornell/MSU/AWS launchers are archived
with the study and are not part of this demonstration. No new jobs were submitted.

## Additional sequence modality (FS-AQ)

[ESM-2 + ChEMBL27 workflow](sequence/README.md) produces 1,792 frozen language-model features and applies the same fixed GBDT settings to the four training manifests and six CASF tests. All 19,066 feature rows, four trained model/scaler bundles and six evaluations are complete and independently verified locally; see the [final report](../../../datasets/protein_ligand_prediction/experiments/sequence_esm2_chembl27_20260922/results/FINAL_REPORT.md). Weights, input preparation and results are external dataset assets. FS-AN retains its identity as the representative topology strategy and has lower RMSE and higher PCC in all six matching comparisons.

The user-requested [FS-AR ablation](../../../datasets/protein_ligand_prediction/experiments/sequence_esm1b_chembl27_nmi_20260922/README.md)
uses ESM-1b with the same ligand embeddings and NMI GBDT settings: 10,000 trees,
depth 7, learning rate 0.005, min_samples_split=2, subsample=0.4; all other
estimator parameters use installed defaults. All four single 10,000-tree fits
and six evaluations are complete, with no ensemble. All 19,066 features,
model/scaler bundles and predictions passed independent local delivery on
September 23 at 10:34 UTC. General-v2020R1 → CASF-2016 gives RMSE 1.251483
(logKa/pK) and PCC 0.836777; [all results and comparisons](../../../datasets/protein_ligand_prediction/experiments/sequence_esm1b_chembl27_nmi_20260922/results/FINAL_REPORT.md)
remain in the external experiment. Both encoder and GBDT differ from FS-AQ;
these single fits do not isolate their effects. The installed defaults and
selected FS-AN topology strategy remain unchanged.

## Historical ESM-2 + ChEMBL27 queue (FS-AT; superseded)

[FS-AT](../../../datasets/protein_ligand_prediction/experiments/sequence_esm2_chembl27_nmi_20260923/README.md) prepared the byte-identical FS-AQ feature matrix,
then waited for FS-AS delivery. It was canceled while queued on September 23
and produced no production fits. FS-AV completed the requested sqrt revision
using those same ESM-2/ChEMBL27 features. Original frozen sources and queue
receipts remain available. Installed APIs and selected FS-AN are unchanged.

## Historical ESM-2 + CPZ all-feature run (FS-AS; superseded)

The user-confirmed FS-AS study uses `esm2_t33_650M_UR50D` protein embeddings
and the supplied `chembl27_pubchem_zinc_512` ligand checkpoint. All 8,830 ESM-2
windows were verified and reused; 15,364 new CPZ BOS ligand vectors generated
on the Mac. The complete 19,066 × 1,792 float32 feature matrix passed its audit
on September 23 at 02:12 UTC. At the user's sqrt revision, this max_features=None
run was stopped at 13:04 UTC: three refined models/evaluations were completed
and archived with checked hashes; the general fit stopped at 7,401 trees and
is not complete. FS-AU reuses these audited CPZ features and is fully delivered.
Historical frozen sources, receipts and partial results remain preserved.

[Study strategy and receipts](../../../datasets/protein_ligand_prediction/experiments/sequence_esm2_cpz_nmi_20260923/README.md). All checkpoint assets, study adapters,
models and results live outside the installed package.
