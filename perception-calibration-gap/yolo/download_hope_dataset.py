"""
download_hope_dataset.py

*** RUN THIS ON YOUR OWN MACHINE, NOT IN A NETWORK-RESTRICTED SANDBOX ***

HOPE is distributed via Google Drive through the official repo's setup.py
(https://github.com/swtyree/hope-dataset), using the `gdown` package. This
cannot run in an environment without general internet/Google Drive access.

Usage (on your machine):
    pip install gdown
    git clone https://github.com/swtyree/hope-dataset.git
    cd hope-dataset
    python setup.py                     # downloads validation+test images + meshes by default
    # or, to only grab what you actually need for the object-swap plan:
    python setup.py --hope-image --meshes-eval   # smaller: ~50MB valid + 33MB low-res meshes

This script (download_hope_dataset.py) is a thin wrapper that does the same
clone+setup, plus prints a reminder of exactly which HOPE objects your team
proposed swapping in -- so you don't have to cross-reference two documents
while your download is running.
"""

import subprocess
import sys
from pathlib import Path

# Keep this in sync with HOPE_INTEGRATION_NOTES.md's swap proposal.
PROPOSED_SWAP = {
    "tennis_ball": "Tomato Sauce (can) -- cylindrical, easy-detect, HOPE grocery-can class",
    "purse": "Mac & Cheese (box) -- rigid rectangular box, HOPE grocery-shelf-box class",
}
KEPT_UNCHANGED = ["apple", "glass", "remote"]

REPO_URL = "https://github.com/swtyree/hope-dataset.git"
CLONE_DIR = Path("hope-dataset")


def main():
    print("=" * 70)
    print("HOPE dataset download (run this on your own machine)")
    print("=" * 70)
    print("\nProposed object swap (per HOPE_INTEGRATION_NOTES.md -- confirm with "
          "your mentor before treating this as final):")
    for old, new in PROPOSED_SWAP.items():
        print(f"  {old}  ->  {new}")
    print(f"Kept unchanged: {KEPT_UNCHANGED}")
    print("\nThis is a PROPOSAL, not a decision made for you -- edit "
          "PROPOSED_SWAP above (and the matching section in "
          "HOPE_INTEGRATION_NOTES.md) once you and your mentor finalize it.\n")

    try:
        import gdown  # noqa: F401
    except ImportError:
        print("gdown not installed. Run: pip install gdown")
        sys.exit(1)

    if not CLONE_DIR.exists():
        print(f"Cloning {REPO_URL} ...")
        subprocess.run(["git", "clone", REPO_URL, str(CLONE_DIR)], check=True)
    else:
        print(f"{CLONE_DIR} already exists, skipping clone.")

    print("\nRunning setup.py to download HOPE-Image validation+test sets and "
          "low/high-res meshes (this can take a while depending on your "
          "connection -- hope_image/valid is ~50MB, hope_image/test ~179MB, "
          "meshes ~33MB+98MB; skip hope_video/ (~2.9GB) unless you actually "
          "need the video sequences).")
    subprocess.run([sys.executable, "setup.py"], cwd=str(CLONE_DIR), check=True)

    print("\nDone. Check hope-dataset/ for the downloaded hope_image/, "
          "meshes/ directories. Cross-reference the object names in "
          "hope_image/*/annotations against PROPOSED_SWAP above to find "
          "which image files/frames actually contain your swapped-in objects "
          "-- HOPE's images are cluttered multi-object scenes, not one-object-"
          "per-image, so you'll need to filter/crop rather than use them as-is.")


if __name__ == "__main__":
    main()