# Draft message to Manasi (NOT sent — for your review/editing)

Hi Manasi — one more update on the PRG side, and this one needs a
decision from you rather than just an FYI.

Two things this session:

**First, a genuinely useful diagnostic**: I computed relation features
straight from the true SDF geometry — no camera, no depth map, nothing
lossy — to check whether the Contact-recall problem from the retrain
was about bad labels or bad features. With true geometry, a plain
logistic regression gets 65% recall on Contact, versus 0-27% for
everything depth-based. So it looks like the features are the bottleneck,
not the labels — which is actually good news, it means more/better
features or more data could plausibly fix this, rather than needing a
different way of defining the classes.

**Second, I tried building `prg_edge_error` a completely different way**
— no classifier at all, just geometric signals straight off the depth
map (how uncertain is the gap estimate, how sharp is the boundary). I
pre-registered the pass/fail bars before running anything. All three
versions failed decisively — not even close, AUROC at or below chance.
I found the specific reason: without knowing which direction the actual
neighboring object is, any ring sampled around just the target's own box
mostly picks up background, not the thing it's touching. Same failure
mode as the boundary-sharpness bug from before, just recurring in a new
design.

So: that's four sessions now — the original classifier, the retrain, the
oracle check, and this — and none of them have produced a
`prg_edge_error` that actually works. I don't think a fifth autonomous
attempt is the right move without you weighing in. The honest options are
basically: keep going (maybe with a fixed version of the geometric
approach, or real depth once I can get DA-V2 running), or reframe this
part of the paper as an audited negative result — which your own
pre-registration already treats as legitimate for the other calibration
contribution, so it wouldn't be unprecedented.

Nothing changed in production — `use_stub` stays true,
`prg_edge_error_mode` stays "classifier", both new config files default
to the old behavior. Full numbers in `explore/E1_FINDINGS.md`,
`explore/E3_FINDINGS.md`, and `MORNING_HANDOFF_4.md`. I also wrote up a
protocol for measuring real gaps with calipers if you want to actually
test any of this on hardware — `explore/REAL_CAPTURE_PROTOCOL_V2.md`,
about an hour, arm off, camera only.
