# ESM-2 + CPZ ensemble evaluation

These reported results evaluate the external general-set model described in the
[model card](MODEL_CARD.md) and [protocol](ENSEMBLE_PROTOCOL.md). The ensemble
uses the arithmetic mean of three seed predictions. All member and ensemble
metrics, including prediction-vector checksums, are recorded at full precision
in [ENSEMBLE_METRICS.json](ENSEMBLE_METRICS.json). Model identities are in
[MODEL_SELECTION.json](MODEL_SELECTION.json). Models and benchmark data remain
external; these scores are not reproduced by the public extraction examples.

| Member | Test | RMSE | PCC | MAE |
| --- | --- | ---: | ---: | ---: |
| general_seed_0 | casf_2007 | 1.404233 | 0.824291 | 1.089683 |
| general_seed_0 | casf_2013 | 1.400785 | 0.800302 | 1.114886 |
| general_seed_0 | casf_2016 | 1.202998 | 0.850276 | 0.935407 |
| general_seed_1 | casf_2007 | 1.414540 | 0.820248 | 1.093713 |
| general_seed_1 | casf_2013 | 1.405256 | 0.797174 | 1.122677 |
| general_seed_1 | casf_2016 | 1.207716 | 0.847664 | 0.938932 |
| general_seed_2 | casf_2007 | 1.388987 | 0.830104 | 1.077219 |
| general_seed_2 | casf_2013 | 1.407818 | 0.796835 | 1.120899 |
| general_seed_2 | casf_2016 | 1.202461 | 0.849240 | 0.935531 |
| general_ensemble_3seed | casf_2007 | 1.400593 | 0.825812 | 1.086143 |
| general_ensemble_3seed | casf_2013 | 1.402958 | 0.799008 | 1.118615 |
| general_ensemble_3seed | casf_2016 | 1.202560 | 0.849905 | 0.934954 |

## Single-model comparison

| Test | Single RMSE | Ensemble RMSE | Single PCC | Ensemble PCC | Ensemble MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| casf_2007 | 1.401586 | 1.400593 | 0.826199 | 0.825812 | 1.086143 |
| casf_2013 | 1.389762 | 1.402958 | 0.805119 | 0.799008 | 1.118615 |
| casf_2016 | 1.198154 | 1.202560 | 0.851202 | 0.849905 | 0.934954 |

Only CASF-2007 RMSE improves slightly; all other metric/test comparisons favor
the single-model reference. No seed or weight was selected using these tests.

## Seed variation

| Test | Seed RMSE mean ± SD | Seed PCC mean ± SD | Seed MAE mean ± SD |
| --- | ---: | ---: | ---: |
| casf_2007 | 1.402587 ± 0.012856 | 0.824881 ± 0.004954 | 1.086872 ± 0.008599 |
| casf_2013 | 1.404619 ± 0.003560 | 0.798104 ± 0.001912 | 1.119487 ± 0.004083 |
| casf_2016 | 1.204392 ± 0.002892 | 0.849060 ± 0.001315 | 0.936623 ± 0.002000 |

SD is the sample standard deviation across the three seed metrics; it is not
a confidence interval or a measure of ensemble uncertainty. Ensemble metrics
come from averaged predictions, not averaged member metrics. RMSE/MAE use
logKa/pK. The three-member ensemble follows the fixed equal-weight averaging
rule and does not establish improved generalization or statistical significance.
