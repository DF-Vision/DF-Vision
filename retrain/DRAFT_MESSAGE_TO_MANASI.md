# Draft message to Manasi (NOT sent — for your review/editing)

Hi Manasi — retrain attempt update.

I actually retrained the touching/Support/Contact classifier this time,
properly: real physics-verified contact labels (not the heuristic),
production feature-extraction code end to end (no separate training-time
code path), gates written down and committed before I ran anything.

Short version: it doesn't clear the gates, and I'm not flipping anything.
5 of 7 pre-committed checks fail — calibration is bad (confidence doesn't
track correctness), the "Contact" class (two things touching side by
side, not stacked) is barely learnable across every model type I tried,
and it loses to a dumb single-feature-threshold baseline on
cross-validation. Interestingly it did fine on one held-out test scene
(82% balanced accuracy) despite the CV average being much worse (67%) —
so it's not uniformly bad, it's unstable across scenes, which if
anything is a worse property for something feeding a safety gate.

`relation_model_version` stays "legacy" — I built the switch (config file
is separate from `use_stub`, which I never touched), but there's no
version worth switching to yet.

What's actually useful from tonight: the dataset provenance problem from
before is fixed for good — the new training data is committed with a
manifest, seeds, hashes, all of it. And now there's a real answer for why
retraining alone won't fix this: it looks like the 8 features themselves
don't carry enough signal for the Contact class specifically, not a bad
training run. Next step, if you want it, is either better/more features
or a real depth-camera capture test — I wrote a protocol for the second
one (`retrain/REAL_CAPTURE_PROTOCOL.md`), needs you and the camera, ~30-60
min.

One more thing: I tried to also test this against real DA-V2 depth (found
the checkpoint on this machine, that part works) but hit a missing-file
import error in `models/depth_anything_v2/` — a `dinov2_layers`
submodule that exists in your other project folders but was never copied
into this repo. Small, precisely-diagnosed, didn't fix it myself since it
meant reaching outside this repo to patch in files. Details in
`retrain/DA_V2_COMPARISON_SIM.md` if you want to close that gap.

Full numbers in `retrain/GATES_RESULTS.md` and `MORNING_HANDOFF_3.md`.
