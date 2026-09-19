# PRG Edge Error Sweep (SIM)

SYNTHETIC / SIM ONLY -- ray-cast depth on staged SDF scenes, no
real hardware, no DA-V2 (blocked, see BINARY_MODEL_DIAGNOSIS.md H1).
Ground truth is SDFGeometryRelationSource (static-pose heuristic),
NOT physics-verified gazebo_contact -- see caveat in this script's
docstring and BINARY_MODEL_DIAGNOSIS.md H2. Not citable as C1/C1-ext/
C4/C4-ext evidence. Reproduce with:
`python3 diagnosis/scripts/sweep_prg_edge_error_SIM.py`

Total adjacent pairs swept across all 6 worlds: 43

## prg_edge_error distribution per ground-truth relation

| gt_relation | n | mean | std | min | max |
|---|---|---|---|---|---|
| Contact | 18 | 0.1944 | 0.2437 | 0.0000 | 0.5000 |
| Separate | 9 | 0.1111 | 0.2079 | 0.0000 | 0.5000 |
| Support | 16 | 0.2813 | 0.2480 | 0.0000 | 0.5000 |

## Does high prg_edge_error co-occur with a wrong relation prediction?

- Binary (touching vs. Separate) wrong-prediction rate: 0.791 (34/43)
- AUROC(prg_edge_error, binary_wrong) = 0.6193
- Point-biserial correlation(prg_edge_error, binary_wrong) = 0.2048
- 3-way (Contact/Support/Separate) wrong-prediction rate: 0.791 (34/43)
- AUROC(prg_edge_error, three_way_wrong) = 0.6193
- Point-biserial correlation(prg_edge_error, three_way_wrong) = 0.2048

## prg_edge_error value range check

- min=0.0000, max=0.5000, all in [0,1]: True
- constant? unique values: 4

## Interpretation

Given BINARY_MODEL_DIAGNOSIS.md's H3 finding that the frozen binary classifier is currently near-degenerate (predicts 'Separate' on almost every pair regardless of ground truth), any AUROC signal above 0.5 here is expected to come largely from compute_prg_edge_error()'s geometric_term (a raw depth-gap check against 'Separate' predictions), essentially independent of the classifier's own confidence -- NOT evidence that the two-signal (classifier confidence + geometric consistency) design is working as intended. This should be re-run once the classifier itself is fixed (see BINARY_MODEL_DIAGNOSIS.md recommendation) before drawing any conclusion about prg_edge_error's real diagnostic value.
