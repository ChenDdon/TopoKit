# Sequence-based protein–ligand prediction

## Final sequence configuration — September 23, 2026

The user selected **ESM-2 + CPZ, FS-AU NMI-inspired sqrt GBDT**, as the final
sequence ML configuration. [Final parameters, models, scalers and results](../../../../datasets/protein_ligand_prediction/final/sequence_esm2_cpz/README.md)
provide the canonical saved profile: 10,000 trees, learning_rate=0.005,
max_depth=7, min_samples_split=2, subsample=0.4, max_features='sqrt',
random_state=None, with a training-only embedded StandardScaler.
All four model/scaler bundles, 19,066 × 1,792 embeddings and six evaluations are
preserved; the general-v2020R1 bundle is the primary reuse entry point.

[Previous-parameter FS-AW results and settings](../../../../datasets/protein_ligand_prediction/experiments/sequence_esm2_cpz_previous_20260923/ARCHIVE.md) are archived in place,
with all existing artifacts retained. This explicit user selection supersedes
the historical comparison status below. Frozen scientific records and the
separate FS-AN topology selection remain unchanged. Monitoring is retired.

## Completed: previous GBDT settings on retained CPZ (FS-AW)

[FS-AW final report](../../../../datasets/protein_ligand_prediction/experiments/sequence_esm2_cpz_previous_20260923/results/FINAL_REPORT.md) records four complete 10,000-tree models/scalers
and six evaluations, independently delivered locally at 15:33 UTC on September23.
The same 19,066 × 1,792 CPZ feature matrix was reused byte-for-byte. All resolved
parameters match the four original FS-AQ previous-setting model receipts.

The NMI-inspired settings give lower RMSE/MAE and higher PCC on all three
general-set tests and refined-2016. Previous settings improve all three metrics
on refined-2007. Refined-2013 is mixed: previous settings improve PCC, while
NMI-inspired settings improve RMSE and MAE. Overall NMI-inspired settings have
lower RMSE/MAE in five of six rows and higher PCC in four of six rows.

Separate [RMSE](../../../../datasets/protein_ligand_prediction/experiments/sequence_esm2_cpz_previous_20260923/results/RMSE.md), [PCC](../../../../datasets/protein_ligand_prediction/experiments/sequence_esm2_cpz_previous_20260923/results/PCC.md) and [MAE](../../../../datasets/protein_ligand_prediction/experiments/sequence_esm2_cpz_previous_20260923/results/MAE.md) tables include all six
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
[CPZ reuse guide](../../../../datasets/protein_ligand_prediction/experiments/sequence_esm2_cpz_sqrt_20260923/REUSE.md). **FS-AV, ESM-2 + ChEMBL27-only**, is archived with
settings, results and predictions retained; its four model bundles and standalone
scalers were removed from both Mac and Cornell (139,448,346 bytes per machine).
[Archive and pruning records](../../../../datasets/protein_ligand_prediction/experiments/sequence_esm2_chembl27_sqrt_20260923/ARCHIVE.md). Earlier completion descriptions record
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

[CPZ final report](../../../../datasets/protein_ligand_prediction/experiments/sequence_esm2_cpz_sqrt_20260923/results/FINAL_REPORT.md) · [ChEMBL27 final report](../../../../datasets/protein_ligand_prediction/experiments/sequence_esm2_chembl27_sqrt_20260923/results/FINAL_REPORT.md)

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

This second modality uses frozen protein and ligand language models followed
by the same GBDT settings as the topology application. It belongs to TopoKit's
existing **workflow** module. It does not replace FS-AN or its trained model.
The new feature strategy is **FS-AQ** (`sequence-esm2-t33-chembl27-bos-v1`).
No pretraining, encoder fine-tuning, or fusion with topology predictions is
performed. All weights, structures, study scripts and results remain outside
the installed package, under `datasets/protein_ligand_prediction/`.

## Fixed representation

| Component | Setting | Features |
|---|---|---:|
| Protein | ESM-2 `facebook/esm2_t33_650M_UR50D`, 33 layers, final hidden states | 1,280 |
| Ligand | WeilabMSU ChEMBL27 RoBERTa, supplied 512-dimensional checkpoint, BOS pooling | 512 |
| Complex | Protein vector followed by ligand vector, float32 | **1,792** |

Protein embeddings use full protein chains, without a distance crop. For the
benchmark, chains with ATOM records in the supplied protein PDB use complete
SEQRES sequences. SEQRES chains absent from that protein's ATOM records and
pure nucleic-acid chains are explicitly recorded and excluded. If SEQRES is
absent, first-model ATOM residues provide the observed sequence (4as6).
Standard amino acids and explicit MODRES parents are mapped to one-letter
codes; the preparation script lists common modified-residue mappings, and
other residues become X with an audit record.

Each chain is divided into consecutive, nonoverlapping windows of at most
1,022 residues. ESM runs in evaluation mode and FP32. Its mean excludes BOS,
EOS and padding. Window and chain means are weighted by residue count;
repeated chains retain their multiplicity. This is a declared long-protein
adaptation, not a claim of full-chain attention beyond the context window.
Identical windows are cached; cached outputs never depend on labels.

Ligand inputs are canonical isomeric SMILES from the corresponding official
v2020 SDF, with supplied historical MOL2 fallback when the SDF is unavailable.
The downloaded ChEMBL model has a 256-token context including BOS and EOS.
The benchmark explicitly uses its first-254-content-token convention and
records **589 truncated ligands**. A lossless lexer maps unsupported element
symbols as whole tokens to the checkpoint's existing UNK; **61 complexes**
contain such tokens. For 19,005 complexes the encoded token IDs exactly match
the downloaded tokenizer. The original regex can silently drop characters
in unsupported elements, which this interface rejects or records explicitly.

Four source graphs (1lvk, 1rle, 3vjs, 3vjt) fail ordinary RDKit valence checks.
Their supplied graphs are serialized using an explicit exception to the
properties check; atom/bond counts and canonical roundtrip are checked. These
are representation limitations, not chemically repaired structures. Their
diagnostics, unknown residues, truncation and source hashes are retained in
the input audit. No complex or label is dropped or imputed.

## Use the encoders

```bash
pip install -e '.[sequence]'
```

The ChEMBL checkpoint and ESM snapshot have pinned SHA256 identities.
The package does not download models when imported or run.

```python
from topokit.workflows.protein_ligand_prediction.sequence import (
    SequenceEncoder, fit_gbdt, predict_gbdt,
)

encoder = SequenceEncoder(
    esm_dir="/path/to/pretrained/sequence/esm2_t33_650M_UR50D",
    chembl_dir="/path/to/pretrained/sequence/PretrainModels/chembl27_512",
    device="mps",  # Apple Silicon; also supports cpu or cuda:0
)
features, receipt = encoder.encode(
    protein_chains=["MKTAYIAKQRQISFVKSHFSRQ"],
    smiles="CC(=O)Oc1ccccc1C(=O)O",
)
assert features.shape == (1792,)

# For a ligand requiring the benchmark's explicit context/UNK policies:
# encoder.encode(chains, smiles, allow_truncation=True, allow_unknown=True)

# X_train contains only training members; y_train is unscaled pK.
# pipeline = fit_gbdt(X_train, y_train)
# y_pred = pipeline.predict(X_test)  # fitted scaler is embedded
# y_pred = predict_gbdt(features, "/path/to/trusted/sequence/model/bundle")
```

The reusable module accepts sequences and prepared SMILES. The dataset study's
`prepare_inputs.py` performs the source-specific PDB/SDF/MOL2 preparation;
`run_study.py` caches embeddings, builds the matrix and runs the benchmarks.

## Benchmark and GBDT settings

| Training set | Train count | CASF tests |
|---|---:|---|
| Refined 2007 | 1,105 | 2007 (195) |
| Refined 2013 | 2,764 | 2013 (195) |
| Refined 2016 | 3,772 | 2016 (285) |
| Official general v2020R1 | 18,498 | 2007 (195), 2013 (195), 2016 (285) |

The exact existing manifests are copied and hashed. Refined training excludes
its matching test; general training excludes the union of all three tests.
These are PDB-ID splits, not sequence- or scaffold-disjoint splits. Overlap
with the encoders' unsupervised pretraining corpora has not been audited.

Each pipeline contains a training-only StandardScaler and GBDT with 10,000
estimators, learning rate 0.002, depth 7, max_features=sqrt,
min_samples_split=5, subsample=0.8, seed=0. There is no early stopping,
validation holdout, tuning, or target scaling. Production fits use the
existing Cornell scikit-learn 1.9.0 environment, matching the topology study.
Saved bundles include the pipeline, portable scaler arrays and hash receipts.

The [completed study](../../../../datasets/protein_ligand_prediction/experiments/sequence_esm2_chembl27_20260922/README.md)
contains all 19,066 feature rows, four full model/scaler bundles and six
independently verified evaluations. The [final report](../../../../datasets/protein_ligand_prediction/experiments/sequence_esm2_chembl27_20260922/results/FINAL_REPORT.md)
links separate RMSE and PCC comparisons. General-v2020R1 training gives
CASF-2016 RMSE **1.310083** and PCC **0.825491**. The existing FS-AN topology
baseline has lower RMSE and higher PCC in all six matching comparisons;
it remains the representative strategy.

Locally delivered bundles are under
`datasets/protein_ligand_prediction/pretrained/sequence_gbdt/<training_set>/`,
with `pipeline.joblib`, `scaler.npz` and `RECEIPT.json`. Use one of
`refined_2007`, `refined_2013`, `refined_2016` or `general_v2020R1` as the
training-set directory. Pass that directory to `predict_gbdt`; it applies
the embedded scaler to raw 1,792-dimensional embeddings. Do not standardize
the input twice. Loading these bundles requires the recorded scikit-learn
**1.9.0** environment; the prediction helper checks this before loading.

## Sources and attribution

- [ESM](https://github.com/facebookresearch/esm) and the official
  [ESM-2 checkpoint](https://huggingface.co/facebook/esm2_t33_650M_UR50D), revision
  `08e4846e537177426273712802403f7ba8261b6c`.
- [WeilabMSU/PretrainModels](https://github.com/WeilabMSU/PretrainModels), local
  repository commit `454393fb57bfbf12982745449dc3b6752b00d04d`, supplied by the
  user. Its README declares MIT. The compact `_chembl.py` loader is copied
  from the supplied `bt_fps/molecular_roberta.py` implementation, with source
  attribution; the checkpoint is unchanged. The parent source/copy receipts
  are under `pretrained/sequence/`. No Fairseq installation is required.

### Completed execution

ESM generated all 8,830 protein windows on the user's local Mac on September 22, using validated FP32 Apple Metal acceleration. All 15,364 lambda-1 CPU ligand embeddings were reused. Cornell completed the four GBDT fits, two at a time, and final local delivery passed at 18:15 UTC. Original numerical files and their source lock remain intact; separate migration/controller receipts document scheduling. The canceled remote GPU waiter was not resumed.

### ESM-1b / NMI-parameter experiment (FS-AR)

The [new isolated experiment](../../../../datasets/protein_ligand_prediction/experiments/sequence_esm1b_chembl27_nmi_20260922/README.md)
implements the user's subsequent request to replace ESM-2 with first-generation
ESM-1b (`esm1b_t33_650M_UR50S`) and use the five published NMI GBDT overrides.
Both ESM checkpoints have width 1,280; the concatenated width stays 1,792.
ChEMBL27 ligand arrays and all training/test memberships are reused unchanged.

It uses one fit per training set, with 10,000 trees, depth 7, learning rate
0.005, minimum split size 2 and row subsample 0.4. Other estimator parameters
retain Cornell scikit-learn 1.9.0 defaults, including all input features per
split, no early stopping and `random_state=None`; the actual RNG state is
saved for replay. Training-only scaling is retained. Mac MPS inference passed
CPU/padding checks. Cornell completed all four 10,000-tree fits, and all
19,066 features, model/scaler bundles and six evaluations passed independent
local delivery at 10:34 UTC on September 23. The general-v2020R1 model gives
CASF-2016 RMSE 1.251483 (logKa/pK) and PCC 0.836777. See the
[final report and separate RMSE/PCC comparisons](../../../../datasets/protein_ligand_prediction/experiments/sequence_esm1b_chembl27_nmi_20260922/results/FINAL_REPORT.md).
Both encoder and GBDT changed relative to FS-AQ, so their effects are not
isolated by these single stochastic fits.

This experiment has a distinct recipe and model identity. Its adapter and
scripts live outside the installed package; the convenience functions above
still refer to the completed FS-AQ recipe. It is not an exact NMI reproduction:
the historical ESM checkpoint is unresolved, the NMI ligand model was CPZ,
and its reported evaluation used a multi-model consensus.

## Historical ESM-2 + ChEMBL27 queue (FS-AT; superseded)

[FS-AT](../../../../datasets/protein_ligand_prediction/experiments/sequence_esm2_chembl27_nmi_20260923/README.md) prepared the byte-identical FS-AQ feature matrix,
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

[Study strategy and receipts](../../../../datasets/protein_ligand_prediction/experiments/sequence_esm2_cpz_nmi_20260923/README.md). All checkpoint assets, study adapters,
models and results live outside the installed package.

Use the study's `cpz_adapter.py` for CPZ embeddings. The convenience functions
earlier in this README retain the FS-AQ identity; do not use them to label or
load FS-AS models. The active CPZ vocabulary is audited separately.
