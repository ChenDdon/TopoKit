# TopoFormer model card

## Purpose and availability

TopoFormer is an optional supervised pK regressor over the predefined Topo
protein–ligand features. The installed API implements the architecture and
bundle inference; fitted weights, scalers and training controllers are not
bundled. The [usage guide](README.md) demonstrates a trusted external bundle.

This is an independent compact implementation inspired by
[TopoFormer at commit a63a838](https://github.com/WeilabMSU/TopoFormer/tree/a63a8383be3cd9938a498d8a676840b531b3eb32).
It follows the pinned position and pooling conventions, without claiming to
reproduce that project's published pretrained ensemble or reported scores.

## Inputs and architecture

The input is the raw float32 `(10, 50, 55)` Topo tensor for recipe
`hpcc-alpha15-cb4760366e92bd2e`. The canonical schema SHA-256 is
`34dde88aff5d7deab170347299f10b4153d2cdf3408a06d4f681b685490c0fa2`.
Each model standardizes the 27,500 flattened coordinates with its own
training-fitted scaler. It reshapes to `(B,10,50,55)`, permutes to
`(B,50,10,55)`, and creates 50 tokens of 550 values. A direct reshape of the
raw tensor to `(50,550)` changes the feature order and is incorrect.

The selected A03 architecture has width 768, 12 pre-norm encoder blocks,
12 attention heads, a feed-forward width of 3,072 and 86,071,297 parameters.
It uses fixed width-first 2D sinusoidal positions on a 50 × 1 grid, a zero CLS
position and CLS → Linear → tanh → Linear regression pooling. It has no
pretraining or masked decoder. [RECIPE.json](RECIPE.json) specifies all model
and training settings; [IMPLEMENTATION.md](IMPLEMENTATION.md) defines numerical
semantics. The generic API configuration is a separate small baseline.

## Training and evaluation scope

The general-set selection uses a fixed 16,648/1,850 training/validation split;
the selected ensemble consists of three fresh fits on all 18,498 training
complexes with every CASF test ID excluded. Each scaler uses training rows only.
[OPTIMIZER_SELECTION.json](OPTIMIZER_SELECTION.json) records validation selection;
[RECIPE.json](RECIPE.json) records full-fit settings and per-seed epochs.
The architecture-only [reference profile](MODEL_SELECTION.json) retains hashes
of the original A03 reference models. Those hashes identify a different training
configuration and must not be treated as the selected optimizer recipe's models.

Refined-set models use independent fresh fits on 1,105, 2,764 and 3,772 rows for
the 2007, 2013 and 2016 memberships, respectively. Each uses its own
training-only scaler and the general-set selected per-seed epoch counts, with
no refined validation search. Evaluation is restricted to each model's matching
year because other-year core memberships can overlap the training data.

| Matching core | Test count | Ensemble RMSE | Ensemble PCC |
| --- | ---: | ---: | ---: |
| CASF-2007 | 195 | 1.476146 | 0.783543 |
| CASF-2013 | 195 | 1.471495 | 0.765443 |
| CASF-2016 | 285 | 1.272761 | 0.815260 |

These scores describe arithmetic means of three prediction vectors; RMSE is
in pK units. They do not average member metrics. General and refined cohorts
must remain separate, as must the distinct validation cohorts of 3,699 and
1,850 rows. Diagnostics on the selection validation IDs are not independent
test evidence. CASF is a reused benchmark and does not select seeds or epochs.
These results do not establish universal superiority, statistical significance
or expected accuracy on an unseen target. Model weights and per-sample
predictions remain external assets, so a public clone alone cannot reproduce
the reported benchmark scores.
