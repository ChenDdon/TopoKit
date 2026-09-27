# Supervised TopoFormer implementation contract

The optional installed API exposes `TopoFormerConfig`, `build_model`,
`save_bundle` and `load_predictor` under `topokit.workflows.topoformer`.
Model/configuration, numerical token and position encoding, and portable
prediction are reusable. Dataset selection, splitting, normalizer fitting and
training are caller responsibilities. The [model card](MODEL_CARD.md) describes
scope and evaluation limitations; [RECIPE.json](RECIPE.json) contains the
selected model and training settings.

## Model semantics

`config.py` is a validated dataclass; `encoding.py` supplies dependency-light
NumPy reference functions; `_model.py` imports PyTorch lazily. The explicit
[A03 config](configs/final_a03.json) defines the selected architecture. The
generic constructor uses the [small baseline](configs/baseline.json).
Blocks initialize independently. Linear layers use normal(std=.02) weights and
zero biases; the token projection uses Xavier uniform; CLS uses normal(std=.02).
LayerNorm uses unit weights, zero biases and epsilon 1e-12. There is no masked
decoder or pretraining operation.

The input is normalized `(B,10,50,55)`. The model permutes to `(B,50,10,55)`
then forms `(B,50,550)` before its biased linear projection. Its trainable CLS
is prepended. Frozen positions follow the pinned TopoFormer
`get_2d_sincos_pos_embed(d,50,1,add_cls_token=True)` convention: width-first
meshgrid, width then height sine/cosine halves, frequencies with base 10000,
float32 and an all-zero CLS row. Patch indices 0–49 are not alpha radii.
The width must be divisible by four and by the number of heads.

Each block has pre-LayerNorm attention, a residual connection, pre-LayerNorm
feed-forward/GELU/output and another residual. PyTorch scaled-dot-product
attention provides the attention calculation without a causal mask. No extra
dropout lies between GELU and the second feed-forward linear layer. Final
LayerNorm precedes CLS → Linear(d,d) → tanh → Linear(d,1). Pooling uses CLS,
not a mean over tokens. The encoder preserves input token order.

## Scaling and feature identity

Fit StandardScaler on active training rows only, independently for each of the
27,500 flattened coordinates. Selection and final full-data fitting require
their own appropriate scaler fit; a GBDT scaler must not be reused for DL
validation. Targets remain unscaled pK. The exact recipe, tensor axes and
canonical schema remain part of the model contract, including the null/null
channel's fixed zero coordinates.

## Bundle and inference

`bundle.py` saves state_dict weights, JSON config/provenance, non-pickled scaler
arrays and the exact feature schema. `BUNDLE.json` authenticates their hashes.
Torch loading uses `weights_only=True`. Optimizer state is separate from a
prediction bundle. Inference validates the feature-schema digest supplied by
the producer, tensor shape and finite values, standardizes raw features once,
and predicts unscaled pK. Constant coordinates follow StandardScaler's scale-1
convention. Fitted weights are external assets and never embedded in a wheel.

See [pinned source conventions](TOPOFORMER_REFERENCE.json) for the reference
commit, source checksums, positional encoding and pooling definitions. Source
convention compatibility does not imply compatibility with every published
checkpoint or configuration.
