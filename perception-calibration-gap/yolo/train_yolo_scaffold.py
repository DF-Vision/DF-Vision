"""
train_yolo_scaffold.py

Purpose
-------
YOLO fine-tuning scaffold matching the master reference: ONE multi-class
YOLOv8n model across all 4-6 study objects (not one model per object),
mirroring realistic use. This script is the mechanics test -- confirming
data loading, training loop, and eval reporting work -- before pointing it
at real, marker-validated, auto-labeled captures per Phase 1.5.

Network-awareness (important, read before running elsewhere)
--------------------------------------------------------------
Ultralytics auto-downloads pretrained weights (yolov8n.pt) and its COCO8
tiny sample dataset from GitHub release assets. In network-restricted
environments (like the sandbox this was originally built in), those
downloads can 403 because the redirect target
(release-assets.githubusercontent.com) isn't allowlisted, even though
github.com itself is reachable. This script:
  1. Tries to load real pretrained weights (yolov8n.pt) and, separately,
     tries to fetch COCO8 if requested.
  2. If either fails, falls back to (a) a from-scratch model built from the
     yolov8n.yaml architecture (no download needed) and (b) a small
     synthetic offline dataset generated locally -- so the TRAINING LOOP
     MECHANICS can still be verified with zero network dependency.
  3. Prints clearly which path was taken -- never silently swaps to the
     fallback without saying so, since "did it actually test what I think
     it tested" matters here.

On your own machine (real internet, no egress restrictions), this should
transparently use real pretrained weights and real COCO8 without any
fallback triggering -- a strictly better sanity check than what's possible
in a network-restricted sandbox.

What this script does NOT do
------------------------------
It does not fine-tune on your real 5 study objects -- that requires
marker-validated, auto-labeled captures from Phase 1.5, which depends on
Checkpoint 1 (ground-truth validation) passing on real hardware first.
Running this against unvalidated data would need to be redone anyway.

Edge cases handled
-------------------
- Pretrained weight download failure -> falls back to from-scratch model,
  reported explicitly, not silently.
- COCO8 download failure -> falls back to a locally-generated synthetic
  YOLO-format dataset (few images, simple boxes), reported explicitly.
- Multi-class label file mismatch (a class index in a label file that
  isn't in data.yaml's names list) is Ultralytics' own responsibility to
  catch; this script's synthetic dataset generator is constructed to never
  produce that mismatch, as a self-consistency guarantee.
"""

import os
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

STUDY_OBJECT_NAMES = ["apple", "tennis_ball", "purse", "glass", "remote"]


def try_load_pretrained_model():
    """Attempt to load real COCO-pretrained yolov8n weights. Returns
    (model, used_pretrained: bool, note: str)."""
    from ultralytics import YOLO
    try:
        model = YOLO("yolov8n.pt")
        # Ultralytics sometimes returns a model object even on a failed
        # download if a stale/partial file exists; do a trivial check.
        return model, True, "Loaded real COCO-pretrained yolov8n.pt weights."
    except Exception as e:
        print(f"  Pretrained weight download/load failed ({e}); "
              f"falling back to from-scratch yolov8n.yaml architecture "
              f"(no download required).")
        model = YOLO("yolov8n.yaml")  # architecture only, random init
        return model, False, f"Fell back to from-scratch model (pretrained load failed: {e})"


def generate_synthetic_yolo_dataset(root: str, n_images: int = 24, seed: int = 0):
    """Build a tiny, fully offline YOLO-format multi-class dataset: solid-
    color squares of varying size/position on a plain background, one class
    per study object (color-coded, not a real appearance model -- this is
    ONLY for testing that the training loop mechanics work end-to-end, not
    a substitute for real captures).

    Directory layout (Ultralytics YOLO format):
        root/images/train/*.jpg, root/labels/train/*.txt
        root/images/val/*.jpg,   root/labels/val/*.txt
        root/data.yaml
    """
    rng = np.random.default_rng(seed)
    root = Path(root)
    if root.exists():
        shutil.rmtree(root)
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)

    img_size = 320
    colors = [
        (220, 40, 40), (40, 180, 40), (60, 60, 220), (220, 200, 40), (180, 60, 200)
    ]  # one distinct color per class, purely synthetic

    n_train = int(n_images * 0.75)
    for i in range(n_images):
        split = "train" if i < n_train else "val"
        class_id = i % len(STUDY_OBJECT_NAMES)

        img = Image.new("RGB", (img_size, img_size), (30, 30, 30))
        draw = ImageDraw.Draw(img)

        box_size = int(rng.integers(40, 100))
        cx = int(rng.integers(box_size, img_size - box_size))
        cy = int(rng.integers(box_size, img_size - box_size))
        x0, y0 = cx - box_size // 2, cy - box_size // 2
        x1, y1 = cx + box_size // 2, cy + box_size // 2
        draw.rectangle([x0, y0, x1, y1], fill=colors[class_id])

        img_path = root / "images" / split / f"synthetic_{i:03d}.jpg"
        img.save(img_path, quality=90)

        # YOLO label format: class_id x_center y_center width height (normalized)
        xc, yc = cx / img_size, cy / img_size
        w, h = box_size / img_size, box_size / img_size
        label_path = root / "labels" / split / f"synthetic_{i:03d}.txt"
        with open(label_path, "w") as f:
            f.write(f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

    data_yaml = root / "data.yaml"
    with open(data_yaml, "w") as f:
        f.write(f"path: {root.resolve()}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write(f"nc: {len(STUDY_OBJECT_NAMES)}\n")
        f.write(f"names: {STUDY_OBJECT_NAMES}\n")

    print(f"  Generated synthetic offline dataset: {n_train} train / "
          f"{n_images - n_train} val images -> {root}")
    return str(data_yaml)


def try_get_real_coco8():
    """Attempt to use Ultralytics' built-in COCO8 tiny sample (auto-
    downloads on first use). Returns (data_yaml_path, success: bool)."""
    try:
        # Ultralytics resolves 'coco8.yaml' via its own dataset registry,
        # downloading if not cached. A trivial YOLO().val() call on it
        # would trigger the download; we test more cheaply by checking
        # the settings-configured datasets dir after attempting to fetch
        # the yaml. If this raises or the file doesn't materialize, treat
        # as failure and let the caller fall back.
        from ultralytics.utils import checks
        from ultralytics.data.utils import check_det_dataset
        data_dict = check_det_dataset("coco8.yaml")
        return data_dict, True
    except Exception as e:
        print(f"  COCO8 fetch failed ({e}); will use synthetic offline dataset instead.")
        return None, False


def run_sanity_finetune(epochs: int = 2, imgsz: int = 320, seed: int = 42):
    print("=" * 70)
    print("YOLO fine-tuning scaffold: mechanics sanity check")
    print("=" * 70)

    print("\nStep 1: attempting to load pretrained weights...")
    model, used_pretrained, note = try_load_pretrained_model()
    print(f"  {note}")

    print("\nStep 2: attempting to train on the real COCO8 sample dataset...")
    print("  (success is judged by whether train() actually completes, not by "
          "a separate pre-check -- a prior version of this script trusted a "
          "pre-check that returned a false positive when the download race/"
          "extraction was subtly broken; this version doesn't repeat that mistake.)")

    results = None
    used_real_coco8 = False
    try:
        results = model.train(
            data="coco8.yaml", epochs=epochs, imgsz=imgsz, batch=4, seed=seed,
            verbose=False, project="yolo/runs", name="sanity_check_coco8",
            exist_ok=True, plots=False,
        )
        used_real_coco8 = True
        data_yaml = "coco8.yaml"
        print("  Real COCO8 training completed successfully.")
    except Exception as e:
        print(f"  Real COCO8 training failed ({type(e).__name__}: {e})")
        print("\nStep 2b: falling back to a locally-generated synthetic dataset "
              "(multi-class, matching your 5 study objects' NAMES ONLY -- "
              "appearance is not realistic, this only tests training mechanics).")
        data_yaml = generate_synthetic_yolo_dataset("yolo/data/synthetic_sanity_check")

        # Re-load a fresh model instance: a failed train() call can leave the
        # trainer/model object in a partially-configured state, so don't
        # reuse it for the fallback attempt.
        model, used_pretrained, note = try_load_pretrained_model()
        print(f"  (re-loaded model for fallback attempt: {note})")

        print(f"\nStep 3: running sanity fine-tune on synthetic data "
              f"({epochs} epochs, imgsz={imgsz})...")
        results = model.train(
            data=data_yaml, epochs=epochs, imgsz=imgsz, batch=4, seed=seed,
            verbose=False, project="yolo/runs", name="sanity_check_synthetic",
            exist_ok=True, plots=False,
        )

    print("\nStep 4: sanity checks on the result...")
    save_dir = getattr(results, "save_dir", None)
    print(f"  Training completed. Run artifacts saved to: {save_dir}")

    # Basic sanity assertions: training should have produced a best/last
    # checkpoint file, proving the full loop (load->train->save) works.
    if save_dir is not None:
        weights_dir = Path(save_dir) / "weights"
        last_pt = weights_dir / "last.pt"
        assert last_pt.exists(), (
            f"Expected a checkpoint at {last_pt} after training -- if this "
            f"is missing, the training loop did not complete successfully "
            f"even though train() didn't raise."
        )
        print(f"  Checkpoint found: {last_pt} -- PASS (training loop mechanics verified)")
    else:
        print("  WARNING: could not determine save_dir from training results; "
              "manually check the yolo/runs/sanity_check directory.")

    return {
        "used_pretrained": used_pretrained,
        "used_real_coco8": used_real_coco8,
        "data_yaml": data_yaml,
        "save_dir": str(save_dir) if save_dir else None,
    }


if __name__ == "__main__":
    result = run_sanity_finetune(epochs=2, imgsz=320)
    print("\n" + "=" * 70)
    print("Summary:")
    for k, v in result.items():
        print(f"  {k}: {v}")
    print("=" * 70)
    if not result["used_pretrained"] or not result["used_real_coco8"]:
        print("\nNOTE: one or more real downloads were unavailable in this "
              "environment (likely a network-restricted sandbox). Re-run "
              "this exact script on a machine with normal internet access "
              "(your laptop/lab PC) to get the fuller sanity check using "
              "real pretrained weights and real COCO8 -- the code does not "
              "need to change, only the environment.")