# Draft message to Manasi (NOT sent — for your review/editing)

Hi Manasi — quick update on the relation-classifier diagnosis.

The overnight session's "confirmed root cause" for the touching-model bug
doesn't hold up. I went back and checked it properly: the thing I called
a training-vs-live code mismatch (the boundary feature counting
background pixels) turned out to be identical code on both sides, not a
mismatch at all. The one real mismatch I could find (a ring-shape
difference) is small and, when I tested it against real physics-verified
contact data — which I also got working for the first time, running
Krupa's `extract_gazebo_contacts.py` script that had never actually been
executed — it changed nothing. The model still calls everything
"Separate," 43 times out of 43.

So: no fix, and I didn't ship one, because it wouldn't have done
anything. What I do have is a real, working way to generate physics-
verified ground truth now (that's genuinely new and useful), and a
three-step proposal for what to try next — relabel with the real contact
data first, check whether the same background-domination shows up on
real Gazebo depth (not just my ray-cast test) before touching the model
again, then only retrain after that. Full writeup in
`verification/RETRAIN_OR_RELABEL_PROPOSAL.md` and `MORNING_HANDOFF_2.md`.

Nothing touched — models, config, PREREG all byte-identical to before.
`use_stub` stays `true` until you say otherwise. Let me know which of the
three steps you want to greenlight, or if you'd rather look at this
yourself first.
