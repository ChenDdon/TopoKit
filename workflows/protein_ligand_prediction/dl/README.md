# Protocol TopoFormer application — final A03 architecture

**A03 is the final protocol DL architecture**, selected by the user on September
23, 2026: width 768, 12 encoder layers, 12 attention heads, feed-forward width
3072 and **86,071,297 parameters**. The reusable optional implementation remains
`topokit.workflows.topoformer`. It trains from random initialization, with no
pretraining or decoder, on the fixed FS-AN alpha15 features.

- [Final profile and delivered models](../../../../datasets/protein_ligand_prediction/final/topoformer_a03/README.md).
- [Workflow architecture config](configs/final_a03.json) and [selection receipt](MODEL_SELECTION.json).
- [Unified DL ablation record](../../../../datasets/protein_ligand_prediction/experiments/protocol_ablation/studies/ST-08/README.md): 20 configurations, 34 unique validation fits, six full fits, 24 CASF rows; separate RMSE/PCC tables.
- [Active A03 tuning study](../../../../datasets/protein_ligand_prediction/experiments/topoformer_a03_tuning_20260923/README.md): running on two GPUs; fixed90:10 split, batch16/32/64, LR3e-5/1e-4/2e-4, staged weight decay;200-epoch cap and cosine horizon, up to25 fits. Existing delivered predictors remain the reference.

The three existing full predictors use **102/104/92 epochs** at seeds 0/1/2,
trained on all **18,498** general-v2020R1 rows excluding every CASF test ID.
Each bundle contains its fitted training-only scaler; the reported ensemble
averages the three seed predictions. A03 led mean validation RMSE/PCC at
1.171782/0.772468 on the fixed 14,799/3,699 split. It improves CASF-2007; the
preserved six-layer control remains better on CASF-2013/2016. Recorded A03
full fits took 83–96 minutes/seed versus 42–75 for the control, so fewer epochs
does not establish lower wall time or universal superiority.

Both completed studies are archived in place under ST-08. Generic
`TopoFormerConfig()` and `configs/baseline.json` retain the original four-layer
baseline for compatibility; **the protocol uses `configs/final_a03.json` or
loads the saved A03 bundle**. No historical receipt or checkpoint is rewritten.

## Feature and model contract

Recipe `hpcc-alpha15-cb4760366e92bd2e` is unchanged: float32 `(10,50,55)`, 15 Å
crop, 50 alpha radii 0.1–5.0 Å, 55 null/element channels including ligand H,
normalized min/max/MAD and ten spectral summaries. Fit StandardScaler on active
training rows only, independently for each of 27,500 flattened coordinates.
Reshape to `(batch,10,50,55)`, permute to `(batch,50,10,55)`, then flatten the last
two axes into **50 tokens × 550 values**. A direct reshape to `(50,550)` is wrong.

The final A03 profile has width 768, 12 independently initialized pre-norm blocks,
12 heads and FF multiplier 4: **86,071,297 trainable parameters**. Fixed positions
use the original width-first 2D sin/cos convention on a 50×1 grid, with a zero CLS
position. The final normalized CLS state passes through Linear(d,d), tanh and
Linear(d,1). Hidden/output and attention-probability dropout are 0.1; batch 32.
LayerNorm epsilon is 1e-12. GELU has no extra intermediate dropout. Positions,
pooling and dropout remain fixed. Batch 32 was fixed in the completed studies;
the proposed next stage explicitly considers batches 16/32/64.

## Installed API

Install the appropriate PyTorch build for your device, then `pip install '.[dl]'`
from the package directory. PyTorch and scikit-learn remain optional; plain
TopoKit imports do not load either library. No datasets or weights ship in wheels.

```python
import json
from pathlib import Path
from topokit.workflows.topoformer import build_model, load_predictor

# For a new supervised A03 fit, use the explicit protocol architecture:
config_path = Path('/path/to/topokit/workflows/protein_ligand_prediction/dl/configs/final_a03.json')
model = build_model(json.loads(config_path.read_text()))  # random initialization

# For direct prediction, load an existing full A03 model and its saved scaler:
predictor = load_predictor(
    '/path/to/datasets/protein_ligand_prediction/final/topoformer_a03/models/seed_0/bundle',
    device='cpu',
)
predicted_pk = predictor.predict(
    raw_features,  # (N,10,50,55), before standardization
    schema_sha256='34dde88aff5d7deab170347299f10b4153d2cdf3408a06d4f681b685490c0fa2',
)
```

Bundles contain `weights.pt` (state_dict; loaded with weights_only=True),
`config.json`, non-pickled `scaler.npz`, the exact `feature_schema.json` and a
checksummed `BUNDLE.json` with training provenance. Normalization occurs once
inside the predictor. Checksum, schema, shape and finite-value checks reject
incompatible input; the producer must supply its actual feature-schema digest.

Dataset-specific loading, splitting, ablation selection, two-GPU orchestration
and evaluation live in the external study's `scripts/` directory. The reusable
module contains only model/configuration, token/position encoding and portable
prediction. See [implementation contract](IMPLEMENTATION.md),
[baseline template](configs/baseline.json), and [pinned source conventions](TOPOFORMER_REFERENCE.json).

This is an independent compact implementation inspired by
[TopoFormer at commit a63a838](https://github.com/WeilabMSU/TopoFormer/tree/a63a8383be3cd9938a498d8a676840b531b3eb32).
It follows the specified position and pooling conventions, but does not claim to
reproduce the published pretrained ensemble or its scores. Training results are
reported only when the corresponding audit and prediction receipts exist.
