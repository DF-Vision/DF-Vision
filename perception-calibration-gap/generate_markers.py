"""
Marker generator — produces printable ArUco marker PNGs sized per object.

Uses a single dictionary (DICT_5X5_50) across the whole study, per the hardware
requirements doc: pick ONE dictionary and never mix.

Output: one PNG per object/role, sized so it can be printed at 100% scale and
come out at the correct physical dimension (uses DPI math, not pixel guessing).
"""

import cv2
import cv2.aruco as aruco
import os

OUTPUT_DIR = "generated_markers"
DICTIONARY = aruco.getPredefinedDictionary(aruco.DICT_5X5_50)
PRINT_DPI = 300  # standard printer resolution; adjust to match your actual printer

# marker_id must be unique per role across the whole study -- never reuse an id
# for two different objects/occluder roles, or detection becomes ambiguous.
MARKERS = [
    {"marker_id": 0, "name": "apple",       "size_cm": 3.5},
    {"marker_id": 1, "name": "tennis_ball",  "size_cm": 3.5},
    {"marker_id": 2, "name": "purse",        "size_cm": 5.5},
    {"marker_id": 3, "name": "glass",        "size_cm": 4.5},  # mount on rigid tab, not on the glass itself
    {"marker_id": 4, "name": "stapler",       "size_cm": 4.5},
    {"marker_id": 5, "name": "occluder_A",   "size_cm": 5.0},  # marker on the occluder panel itself
    {"marker_id": 6, "name": "occluder_B",   "size_cm": 5.0},  # second marker for orientation, if needed
    {"marker_id": 7, "name": "world_origin", "size_cm": 6.0},  # optional fixed reference marker
]


def cm_to_pixels(size_cm: float, dpi: int) -> int:
    inches = size_cm / 2.54
    return int(round(inches * dpi))


def generate_marker(marker_id: int, size_cm: float, dpi: int = PRINT_DPI):
    side_px = cm_to_pixels(size_cm, dpi)
    img = aruco.generateImageMarker(DICTIONARY, marker_id, side_px)
    return img, side_px


def add_print_border_and_label(img, name: str, size_cm: float, dpi: int, border_frac: float = 0.15):
    """Add a white border + text label so you know what you're printing and can
    verify the physical size with a ruler after printing (a real reviewer-proof
    sanity check: measure the printed marker, confirm it matches size_cm)."""
    h, w = img.shape[:2]
    border = int(w * border_frac)
    label_h = int(dpi * 0.35)  # room for a text label below the marker

    canvas = 255 * (1 + 0 * cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))  # white canvas, same channel depth trick
    canvas_h = h + 2 * border + label_h
    canvas_w = w + 2 * border
    out = 255 * (
        __import__("numpy").ones((canvas_h, canvas_w, 3), dtype="uint8")
    )
    out[border:border + h, border:border + w, 0] = img
    out[border:border + h, border:border + w, 1] = img
    out[border:border + h, border:border + w, 2] = img

    label = f"{name}  id={marker_id_lookup(name)}  size={size_cm}cm  dict=DICT_5X5_50"
    cv2.putText(
        out, label, (border, canvas_h - int(label_h * 0.3)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA,
    )
    return out


_id_lookup_cache = {m["name"]: m["marker_id"] for m in MARKERS}


def marker_id_lookup(name: str) -> int:
    return _id_lookup_cache[name]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    manifest_lines = ["name,marker_id,size_cm,dpi,dictionary"]

    for m in MARKERS:
        img, side_px = generate_marker(m["marker_id"], m["size_cm"])
        labeled = add_print_border_and_label(img, m["name"], m["size_cm"], PRINT_DPI)

        out_path = os.path.join(OUTPUT_DIR, f"{m['name']}_id{m['marker_id']}.png")
        cv2.imwrite(out_path, labeled)
        print(f"Wrote {out_path}  ({m['size_cm']}cm @ {PRINT_DPI}dpi -> {side_px}px marker)")

        manifest_lines.append(f"{m['name']},{m['marker_id']},{m['size_cm']},{PRINT_DPI},DICT_5X5_50")

    manifest_path = os.path.join(OUTPUT_DIR, "manifest.csv")
    with open(manifest_path, "w") as f:
        f.write("\n".join(manifest_lines))
    print(f"\nWrote manifest: {manifest_path}")
    print("\nIMPORTANT: when printing, use 'Actual Size' / 100% scale, NOT 'Fit to page'.")
    print("After printing, measure each marker with a ruler and confirm it matches size_cm")
    print("in the manifest -- a mismatch here silently corrupts every occlusion calculation downstream.")


if __name__ == "__main__":
    main()