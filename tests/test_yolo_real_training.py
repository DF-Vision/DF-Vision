"""
tests/test_yolo_real_training.py

C3(d): an ACTUAL model.train() call (not mocked) on the grouped-split
synthetic dataset, confirming the training loop mechanics work end to
end. Separated from test_yolo_pipeline.py because this is slow (real
ultralytics training, even at 2 epochs/CPU/tiny images, takes real wall-
clock time) -- CI/fast-loop runs can skip this file, full verification
runs should not.

THIS IS A CODE CHECK, NOT A DETECTOR-QUALITY RESULT: the dataset is
solid-color synthetic boxes (see train_yolo_scaffold.py's own docstring),
never cited as evidence the trained model detects anything real.
"""
import sys
import time
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "perception-calibration-gap" / "yolo"))

import train_yolo_scaffold as tys

try:
    from ultralytics import YOLO
    _ULTRALYTICS_AVAILABLE = True
except Exception as _e:
    _ULTRALYTICS_AVAILABLE = False
    _ULTRALYTICS_IMPORT_ERROR = str(_e)


pytestmark = pytest.mark.skipif(
    not _ULTRALYTICS_AVAILABLE,
    reason="ultralytics not importable in this environment" if not _ULTRALYTICS_AVAILABLE else "",
)


@pytest.fixture(scope="module")
def trained_run(tmp_path_factory):
    """Runs ONE real 2-epoch training, shared across this file's
    assertions (re-running per-test would multiply the wall-clock cost
    for no additional verification value)."""
    ds_dir = tmp_path_factory.mktemp("c3_real_training_SYNTHETIC")
    data_yaml, session_of_image, train_sessions, val_sessions = tys.generate_synthetic_yolo_dataset_grouped(
        str(ds_dir / "dataset"), n_sessions=12, images_per_session=4, seed=20260921,
    )
    model = YOLO("yolov8n.pt")
    print("\nSYNTHETIC / CODE CHECK ONLY -- not a detector-quality result.")
    t0 = time.time()
    results = model.train(
        data=data_yaml, epochs=2, imgsz=320, batch=4, seed=42, verbose=False,
        project=str(ds_dir / "runs"), name="c3_real_training_SYNTHETIC",
        exist_ok=True, plots=False, device="cpu",
    )
    elapsed = time.time() - t0
    print(f"Training wall-clock: {elapsed:.1f}s (SYNTHETIC, code check only)")
    return model, results, elapsed


def test_training_completes_and_saves_checkpoints(trained_run):
    model, results, elapsed = trained_run
    save_dir = Path(results.save_dir)
    last_pt = save_dir / "weights" / "last.pt"
    best_pt = save_dir / "weights" / "best.pt"
    assert last_pt.exists(), f"expected checkpoint at {last_pt}"
    assert best_pt.exists(), f"expected checkpoint at {best_pt}"


def test_checkpoint_reloads_and_predicts(trained_run):
    """Reload the saved checkpoint fresh (not the in-memory model object)
    and confirm inference runs without raising -- proves the save/load
    round trip works, not just that the in-process object still works."""
    model, results, elapsed = trained_run
    save_dir = Path(results.save_dir)
    last_pt = save_dir / "weights" / "last.pt"

    reloaded = YOLO(str(last_pt))
    # predict on one of the dataset's own val images
    val_images = list((Path(results.save_dir).parent.parent).glob("**/images/val/*.jpg"))
    if not val_images:
        # results.save_dir's dataset copy path can vary by ultralytics version;
        # fall back to generating one fresh image to predict on.
        import numpy as np
        from PIL import Image
        img_path = save_dir / "predict_input_SYNTHETIC.jpg"
        Image.fromarray((np.random.default_rng(0).integers(0, 255, (320, 320, 3))).astype("uint8")).save(img_path)
        val_images = [img_path]

    pred_results = reloaded.predict(source=str(val_images[0]), verbose=False)
    assert pred_results is not None
    assert len(pred_results) >= 1


def test_training_loss_available_and_finite(trained_run):
    """Weak, honest baseline claim: loss values were recorded for both
    epochs and are finite (not NaN/inf) -- NOT a claim of convergence or
    detector quality over just 2 epochs on synthetic solid-color boxes."""
    model, results, elapsed = trained_run
    import csv
    results_csv = Path(results.save_dir) / "results.csv"
    assert results_csv.exists(), f"expected {results_csv} from training"
    with open(results_csv) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2, f"expected 2 epochs of logged results, got {len(rows)}"
    loss_cols = [c for c in rows[0].keys() if "loss" in c.lower()]
    assert loss_cols, "expected at least one loss column in results.csv"
    for row in rows:
        for col in loss_cols:
            val = float(row[col])
            assert val == val and abs(val) != float("inf"), f"non-finite loss in {col}: {val}"


def test_train_loss_decreases_epoch1_to_epoch2(trained_run):
    """Stronger claim, confirmed against this session's own real run
    (project_closure/LOG.md's C3 entry has the exact numbers): all three
    train-loss components (box/cls/dfl) decreased from epoch 1 to epoch 2
    in the run that produced this repo's committed C3 evidence. Still
    NOT a convergence or detector-quality claim -- 2 epochs on tiny
    synthetic solid-color boxes -- only that the optimizer step is
    actually reducing the loss it's computing, i.e. the training loop
    mechanics work, not just run without crashing."""
    model, results, elapsed = trained_run
    import csv
    results_csv = Path(results.save_dir) / "results.csv"
    with open(results_csv) as f:
        rows = list(csv.DictReader(f))
    for col in ("train/box_loss", "train/cls_loss", "train/dfl_loss"):
        v1, v2 = float(rows[0][col]), float(rows[1][col])
        assert v2 < v1, f"{col} did not decrease: epoch1={v1}, epoch2={v2}"
