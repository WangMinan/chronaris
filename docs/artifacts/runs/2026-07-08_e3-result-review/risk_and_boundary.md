# E3 Risk And Boundary Review

## Input Representation Boundary

MulT and ContiFormer use `T2_response_lovo_seed17_pooled_embedding`: held-out pooled embeddings from models trained on the T2 regression task. Chronaris and naive_time_sync use fusion streams reconstructed from existing feature values. This means the four-method E3 table is useful for structural diagnosis, but it is not a perfectly identical representation-family export across all methods.

## Metric Sensitivity Boundary

The current Dingxin E3 run has short streams: 36 windows per stream and only three streams per method. ClaSP and STUMPY run, but CLaP is unavailable for all 12 streams. Metrics tied at zero or derived from fallback state sequences are not fit for main-text claims.

## Model Diagnosis

The current result does not show Chronaris structural advantage. Possible causes include insufficient temporal smoothness, unstable fusion pooling, residual/gate behavior that creates fragmented representation geometry, or a Chronaris export family that is less comparable to T2-trained deep baseline embeddings.

## Data Scale Boundary

One sortie has two views and one sortie has a single view in the current E3 sample. Cross-view metrics are therefore completed for only one sortie per method and unavailable for the single-view sortie.

## Writing Boundary

E3 should not be described as expert truth or as a main proof of Chronaris superiority. If cited, write it as a structure diagnostic that motivates controlled candidate optimization.

## Prohibited Follow-up

Do not change labels, splits, E3 m-grid, tolerance, PCA policy, metric weights, or delete unfavorable rows. Do not overwrite the current E3 validation run or thesis protocol snapshot.
