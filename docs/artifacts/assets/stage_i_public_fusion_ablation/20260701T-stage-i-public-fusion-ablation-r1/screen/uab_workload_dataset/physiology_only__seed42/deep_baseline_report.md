# Stage I Deep Baseline - uab_workload_dataset - chronaris_public_fusion_physiology_only

- profile: `window_v2`
- artifact root: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_public_fusion_ablation/20260701T-stage-i-public-fusion-ablation-r1/screen/uab_workload_dataset/physiology_only__seed42`
- prepared root: `/tmp/chronaris_stage_i_public_fusion_refresh/20260701T-stage-i-public-fusion-refresh-r1/uab_workload_dataset`

## Objective

| group | macro-F1 | balanced accuracy | samples | folds |
| --- | ---: | ---: | ---: | ---: |
| `n_back` | 0.329048 | 0.334123 | 3764 | 2 |
| `heat_the_chair` | 0.381882 | 0.503636 | 638 | 2 |

## Subjective

| group | RMSE | MAE | R2 | Spearman | samples | folds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `n_back` | 5.915880 | 5.238205 | 0.054966 | 0.320055 | 3764 | 2 |
| `heat_the_chair` | 2.093045 | 1.790801 | -0.098037 | -0.774159 | 638 | 2 |

