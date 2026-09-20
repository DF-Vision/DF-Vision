# retrain/data/relation_dataset_v2_SIM.csv -- Manifest

Generated: 2026-09-20 (autonomous retrain session)
Generator: retrain/scripts/r2_generate_dataset.py
Generator git commit (at generation time): c42b01712d1a4442f7b2d0625ccc4803d111e67d
Random seed: RANDOM_STATE=20260920
SHA256 of relation_dataset_v2_SIM.csv: 49b0609669dbbae445d5adbade43b2a73cce773fd9a6001d28df4a2155f63e2b
Row count: 330
Viewpoints used per world: [1, 2, 3, 4, 5, 6] (of 6 available)
Noise levels sampled from: [0.0, 1.0, 2.0, 3.0, 5.0] (uniform random choice per (world,viewpoint), seeded)

Label counts per world:
```
relation                Contact  Separate  Support
world_id                                          
world1_baseline               6        17        6
world2_dense_clutter          6        20       11
world3_thin_objects           6        13       21
world4_support_scene         11        22       36
world5_occlusion_scene        8        63       17
world6_dense_mixed            4        22       41
```

Overall label counts:
```
relation
Separate    157
Support     132
Contact      41
```

SIM / SYNTHETIC. Depth source: ray-cast renderer (relation_classifier/scripts/validate_live_depth_feature_source.ray_cast_scene) + retrain/scripts/noise_model.py (documented, disclosed approximation of real sensor noise -- see that file's docstring). Labels: physics-verified gazebo_contact (touching vs not) + SDF-geometry-derived Support/Contact split for touching pairs. Not real hardware data. Not citable as C1/C1-ext/C4/C4-ext evidence.

Features produced by the REAL, unmodified LiveDepthFeatureSource.compute() -- train/serve parity by construction (rule 7).

world6_dense_mixed rows ARE included in this file (generating them is not 'peeking' -- using them for model selection would be; see retrain/GATES.md for the held-out-test discipline).
