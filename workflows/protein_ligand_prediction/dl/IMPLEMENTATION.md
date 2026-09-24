# Supervised TopoFormer implementation contract

Implemented September 22, 2026. The optional installed API is
`TopoFormerConfig`, `build_model`, `save_bundle` and `load_predictor` under
`topokit.workflows.topoformer`. Model/config, numerical token/position encoding
and portable prediction are reusable; dataset selection, experiment orchestration,
normalizer fitting and training remain in the external
[general-v2020R1 study](../../../../datasets/protein_ligand_prediction/experiments/topoformer_general_v2020R1_20260922/README.md).

## Model semantics

`config.py` is a validated dataclass; `encoding.py` provides dependency-light
NumPy reference functions; `_model.py` lazily imports PyTorch. The protocol's
final A03 config is width 768/12 pre-norm blocks/12 heads/FF width 3072/GELU,
loaded explicitly from `configs/final_a03.json`. The generic constructor retains
width 256/four blocks/eight heads for historical baseline compatibility. Blocks initialize
independently. Linear layers use normal(std=.02), zero biases; the token
projection uses Xavier uniform; CLS uses normal(std=.02); LayerNorm weights 1,
biases 0 and epsilon 1e-12. There is no masked decoder or pretraining operation.

The input is normalized `(B,10,50,55)`. The model permutes to `(B,50,10,55)`
then forms `(B,50,550)` before its biased linear projection. Its trainable CLS
is prepended. Frozen positions follow the pinned original TopoFormer
`get_2d_sincos_pos_embed(d,50,1,add_cls_token=True)` convention: width-first
meshgrid, width then height sine/cosine halves, frequencies with base10000,
float32 and an all-zero CLS row. Patch indices 0–49 are not alpha radii.
The width must be divisible by four and by the number of heads.

Each block has pre-LayerNorm attention, a residual connection, pre-LayerNorm
FF/GELU/output and another residual. PyTorch scaled-dot-product attention
provides the attention calculation; there is no causal mask. Attention
probability dropout and hidden/output dropout are0.1. No extra dropout lies
between GELU and the second FF linear layer. Final LayerNorm precedes
CLS → Linear(d,d) → tanh → Linear(d,1). There is no mean pooling or extra
position/pooling/dropout ablation. The upstream zero-mask helper's incidental
random token permutation is unnecessary here and is not reproduced.

## Training and refitting

The application uses batch 32, AdamW, MSE, gradient clipping 1, ten warmup epochs,
cosine decay and cap 500. Validation RMSE chooses the saved best checkpoint;
patience 50 resets only after an improvement exceeding1e-4. The full refit
reinitializes weights and fits a new scaler on all training rows, trains to
that seed's selected epoch, and preserves the original500-epoch LR horizon.
There is no CASF monitoring, target scaling or early stopping in the refit.
Float32/TF32-disabled execution is fixed across candidates. See PLAN.json for
all registered ablations and seed/selection rules.

`study.py` saves model, AdamW state, CPU/CUDA RNG, counters and history after
each epoch. The shuffle is a reproducible function of seed and epoch. OS locks
prevent concurrent controllers or duplicate fits. A failed child stops later
stages. Resume checks identities and completed output hashes; it never silently
drops complexes or substitutes zeros for feature failures.

## Bundle and inference

`bundle.py` saves state_dict weights, JSON config/provenance, non-pickled scaler
arrays and the exact feature schema. BUNDLE.json authenticates their hashes.
Torch loading uses `weights_only=True`; optimizer resume state remains separate.
Inference validates the feature-schema digest supplied by the producer, tensor
shape and finite values, standardizes raw features once and predicts unscaled pK.
Constant coordinates follow StandardScaler's scale 1 convention. Existing GBDT
scalers are never reused for the DL validation fits. Model weights remain
separate data assets and are never embedded in a package wheel.

## Verified acceptance

The GPU host passed 32 model/architecture/trainer tests, including explicit token
ordering, independent position formulas at widths128/256/512, independent block
initialization,3,366,913 baseline parameters, tiny synthetic overfit, normalizer
isolation, exact checkpoint round trips and checksum/schema rejection. An
interruption after epoch 2 resumed to bitwise-identical selected weights and
validation predictions relative to an uninterrupted four-epoch synthetic fit.
Another 60 Laplacian checks cover the unchanged higher-dimensional routes.
All 12 candidates and a second-device width 512 smoke test passed at batch 32.
All 19,066 transferred record/tensor pairs matched the baseline inventory.

The built wheel passed file-content checks and an isolated installed import
with torch/sklearn/GUDHI blocked. These are implementation/preflight results,
not completed CASF performance claims. Evidence is in the study's `receipts/`.
Reference [source conventions](TOPOFORMER_REFERENCE.json) and
[original TopoFormer](https://github.com/WeilabMSU/TopoFormer/tree/a63a8383be3cd9938a498d8a676840b531b3eb32).
