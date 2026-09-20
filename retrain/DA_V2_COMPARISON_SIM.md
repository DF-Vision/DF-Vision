# DA-V2 Secondary Feature Comparison — BLOCKED

Attempted per R2's instruction ("if a DA-V2 checkpoint exists, ALSO
generate DA-V2-depth features"). `retrain/BASELINE.md` confirmed a
checkpoint exists (`/home/manasi/bps-project/checkpoints/depth_anything_v2_vits.pth`)
and that direct (non-ROS) Python inference should be possible via
`models/depth_anything_v2/dpt.py`, which IS committed in this repo.

## What actually happened

```
$ python3 retrain/scripts/r2b_generate_da_v2_features.py
...
File "/home/manasi/DF-Vision/models/depth_anything_v2/dinov2.py", line 20, in <module>
    from .dinov2_layers import Mlp, PatchEmbed, SwiGLUFFNFused, MemEffAttention, NestedTensorBlock as Block
ModuleNotFoundError: No module named 'models.depth_anything_v2.dinov2_layers'
```

`models/depth_anything_v2/dinov2.py` (committed) imports from a
`dinov2_layers` submodule that was **never committed to this repo** —
`ls models/depth_anything_v2/` shows only `dinov2.py`, `dpt.py`,
`__init__.py`. A machine-wide search confirms the missing submodule
exists in two sibling, out-of-repo locations:
```
/home/manasi/bps-project/models/depth_anything_v2/dinov2_layers
/home/manasi/Depth-Anything-V2/depth_anything_v2/dinov2_layers
/home/manasi/Depth-Anything-V2/metric_depth/depth_anything_v2/dinov2_layers
```

## Why not fixed here

Per rule 10 (max ~3 attempts per problem, then log and move on): the fix
would mean copying files from outside this git repository into
`models/depth_anything_v2/` — a shared/vendored dependency directory, not
something this session's scope (relation classifier retraining) should
be unilaterally altering, and copying code from another project
directory into this one without checking its license/version/provenance
match is exactly the kind of "reach outside the repo and patch it in"
action this project's stated discipline warns against doing casually.
This is a real, precisely-diagnosed, one-line-cause blocker — not a dead
end — logged here for a human (or a future session explicitly scoped to
fix it) to resolve: either vendor `dinov2_layers/` into this repo
properly (with review of which of the two sibling copies is the "right"
one, since there are two, possibly-different candidates), or point
`sys.path` at one of the sibling repos directly if that's an acceptable
long-term arrangement.

## What this means for the retrain decision

The DA-V2 comparison was a secondary, nice-to-have data point (per R2's
own "ALSO generate" phrasing) — its absence does not change
`retrain/GATES_RESULTS.md`'s verdict, which already rests on 5 of 7 gates
failing using the ray-cast depth source alone. This blocker means the
"does real depth change the picture" question (flagged as UNVERIFIED in
`MORNING_HANDOFF_3.md`) remains genuinely open, to be resolved either by
fixing this import gap or by the real-hardware capture protocol
(`retrain/REAL_CAPTURE_PROTOCOL.md`), whichever comes first.
