# Optional TopoFormer affinity prediction

The optional `topokit.workflows.topoformer` module predicts affinity from existing
Topo features using a compatible trained model bundle. The public package
provides the architecture and portable inference API; weights, fitted scalers,
training datasets and training controllers remain external.

The [model card](MODEL_CARD.md) describes the selected A03 architecture,
input contract and evaluation limitations. Exact architecture and training
settings are in [RECIPE.json](RECIPE.json), with the validation-selection
configuration in [OPTIMIZER_SELECTION.json](OPTIMIZER_SELECTION.json).
Neither model training nor pretrained weights are needed for
[Topo feature extraction](../README.md).

## Load a trusted model

Install the appropriate PyTorch build for your device, then run
`python -m pip install '.[dl]'` from the repository root. PyTorch and
scikit-learn are optional; ordinary TopoKit imports do not load either library.

```python
from topokit.workflows.topoformer import load_predictor

predictor = load_predictor(
    '/path/to/trusted/topoformer/seed_0/bundle',
    device='cpu',
)
predicted_pk = predictor.predict(
    raw_features,  # finite (N, 10, 50, 55), before standardization
    schema_sha256='34dde88aff5d7deab170347299f10b4153d2cdf3408a06d4f681b685490c0fa2',
)
```

`raw_features` must come from a completed compatible Topo feature store; supply
its actual schema digest. This example loads one member. For an ensemble,
load every member specified by its recipe and combine predictions using the
recorded weights. Each member applies its own saved scaler once.

Bundles contain `weights.pt`, `config.json`, non-pickled `scaler.npz`, the exact
`feature_schema.json`, and a checksummed `BUNDLE.json`. The loader rejects
checksum, schema, shape and finite-value mismatches. Model files must come from
a trusted source and match the bundle's recorded configuration.

`build_model` creates randomly initialized weights; it is not a fitted
predictor. The generic configuration is a small baseline. Use the explicit
[architecture config](configs/final_a03.json) when constructing an A03 model.
See the [implementation contract](IMPLEMENTATION.md) for tensor ordering and
bundle semantics.
