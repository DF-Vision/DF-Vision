# Closure Criteria — pre-committed, per task

Written BEFORE that task's implementation, in commit order (verified by
`git log` in C10). Never loosened after seeing results. This file is
append-only within a session: each task gets its own dated section below,
committed alone before any of that task's implementation commits.

---

## C2 — Live pipeline `--replay` mode (2026-09-21)

**Scope**: `perception-calibration-gap/real_camera/capture_and_gap.py`
(the only real-camera trial entry point in the repo — confirmed via
`grep -rl "def run_one_trial\|def main" perception-calibration-gap/real_camera/`).

### Acceptance criteria (fixed before implementation)

1. **`--replay <dir>` flag exists** on `capture_and_gap.py`'s CLI and runs
   the exact same per-trial code path as live capture (ArUco pose ->
   `occlusion_estimator` -> `tsdf_fusion` -> `gap_computer` ->
   `prg_client` -> `logger`) with frames read from disk instead of the
   Orbbec SDK. No camera, no `pyorbbecsdk` import required to run replay
   mode (verified: replay mode importable/runnable in an environment
   where `pyorbbecsdk` is absent or unmocked).
2. **Camera source is config-driven**: a `camera_config.json` (or
   equivalent) declares `source: "live" | "replay"` and SDK stream
   parameters; no resolution/format/topic value is hardcoded inside
   `capture_and_gap.py` itself for the parts that vary by source. A
   documented expected-topic list is present (even though this repo's
   live path uses the pyorbbecsdk `Pipeline`/`Config` API directly, not
   ROS topics — the config explicitly states this and lists the ROS
   topic names to verify IF a ROS-based Orbbec driver is substituted
   later, satisfying the task's "Orbbec Gemini 2 topic/SDK names in
   config" requirement without inventing a ROS dependency this repo
   doesn't have).
3. **Depth units**: exactly ONE function performs the on-disk-mm-PNG ->
   in-memory-meters conversion, used by both the replay reader and
   (where applicable) any future recorder. A test asserts a synthetic
   millimeter-valued uint16 PNG converts to the correct meter values. A
   guard function rejects (raises, with the array's median value in the
   message) any depth array whose median exceeds 20.0 (a value that
   cannot be a real meters-scale depth for this project's benchtop
   scenes, and is the task's own stated example threshold) — this guard
   is a separate, directly-tested unit, not folded silently into the
   conversion function.
4. **Fixture set**: `perception-calibration-gap/real_camera/generate_replay_fixture.py`
   produces a SYNTHETIC (filename-stamped `SYNTHETIC`), deterministic
   (fixed seed, documented) set of on-disk frames in the SAME layout
   `capture_and_gap.py`'s replay reader consumes — i.e., the reader and
   generator share one documented layout spec, not two independently
   guessed ones.
5. **Schema-correct output**: one `--replay <fixture_dir>` invocation
   over the fixture set produces `TrialRecord`-shaped JSONL rows (all
   `REQUIRED_FIELDS` present) via the existing `TrialLogger`.
6. **Bit-for-bit determinism**: two separate replay runs over the same
   fixture directory produce identical output, field-for-field, EXCEPT
   `timestamp_utc` (which is wall-clock by design, per `logger.py`'s own
   `field(default_factory=...)` — excluding it from the comparison is not
   a loosening of "deterministic," it is the pre-existing, unrelated
   behavior of every trial record in this codebase).
7. **Corrupt/missing input handled without crashing**: a fixture
   trial with a missing depth file, a corrupt (truncated/unreadable) PNG,
   an all-zero mask, and an implausible (all-`NaN`, or median > 20m)
   depth array are each handled by: (a) a logged warning identifying the
   trial and the specific failure, (b) a row-level flag in the emitted
   `TrialRecord.notes` field (not a silent `None`), (c) the replay run
   continuing to the next trial rather than raising out of `main()`. A
   test constructs one fixture directory containing exactly one bad
   trial of each kind among otherwise-good trials and asserts: all good
   trials are logged correctly, all bad trials are logged with a
   `notes` flag (not silently dropped, not crashing the run), and the
   process exits 0.
8. **Per-stage timing logged**: each trial's per-stage wall-clock time
   (capture/load, marker pose, TSDF, Gap, PRG, logger append) is
   captured in a new additive `TrialRecord.stage_timings_ms: Optional[dict]`
   field (default `None`, existing `REQUIRED_FIELDS` and every existing
   caller/test unaffected — same discipline as every prior additive
   `TrialRecord` extension in this codebase) and printed per trial.
9. **`masks_are_bbox_placeholder` verified at every call site**: a test
   enumerates every call to `integration.prg_client.get_prg_edge_error`
   (or its wrapped variants) anywhere in the repo (via `grep`/AST, not a
   hand-maintained list that can silently go stale) and asserts each one
   passes `masks_are_bbox_placeholder=True` given that neither the live
   nor replay path in this repo has a real segmentation model — the
   masks are always marker-corner-bbox-derived.
10. **Legacy behavior unchanged**: the pre-existing live-capture path
    (`main()` with no `--replay` flag) is behaviorally unchanged —
    verified by a diff-level read of the refactored `run_one_trial`
    showing the live branch's argument values match what the
    pre-refactor code passed, and by keeping the live path's own
    `input()`-per-trial prompts and 8-trial structure exactly as before.

None of the above is loosened after implementation. If any criterion
can't be met within the 3-attempt budget (rule 6), it is logged as a
blocker in `project_closure/LOG.md`'s C2 entry, not silently dropped.

### Addendum, same day, written after first implementation run (logged, not silent)

Criterion 6 ("identical... EXCEPT `timestamp_utc`") and criterion 8
(per-trial wall-clock stage timing must be captured) are in tension:
`stage_timings_ms` is, by criterion 8's own requirement, wall-clock
measured and therefore cannot be bit-identical across runs, exactly like
`timestamp_utc` already wasn't. This was an oversight in how criterion 6
was scoped, not a substantive result being loosened -- the underlying
scientific/schema content (every field except the two wall-clock-derived
ones) is still required to be, and was verified to be, bit-for-bit
identical. **Determinism check is therefore: identical field-for-field
EXCEPT `timestamp_utc` and `stage_timings_ms`.** This is a scope
clarification of an internal inconsistency between two criteria written
in the same commit, not a pass/fail bar being relaxed after a failing
result -- no criterion's actual pass/fail THRESHOLD changed.
