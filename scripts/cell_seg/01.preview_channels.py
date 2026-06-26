"""Per-channel preview of the confocal stack for channel identification.

Writes a max-Z-projection grid and a mid-Z-slice grid (all 4 channels), and prints
per-channel intensity stats, so C0=DAPI can be confirmed before segmentation.
See plan/cell_seg/01.nuclei_segmentation_plan.md. Run with:
    conda run --no-capture-output -n img python -u scripts/cell_seg/01.preview_channels.py
"""
import os

import numpy as np
import tifffile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage import exposure

# ---- File paths (input/output) ----
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INPUT_FILE   = os.path.join(PROJECT_ROOT, 'links', 'cell_seg', 'cell_segmentation_Saumya.tif')
OUT_FIG_DIR  = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'cell_seg')
OUT_MAXPROJ  = os.path.join(OUT_FIG_DIR, '01.channels_maxproj.png')
OUT_MIDSLICE = os.path.join(OUT_FIG_DIR, '01.channels_midslice.png')

# ---- Config ----
DISP_LO = 1.0
DISP_HI = 99.8


def stretch(img2d):
    lo, hi = np.percentile(img2d, [DISP_LO, DISP_HI])
    return exposure.rescale_intensity(img2d.astype(np.float32),
                                      in_range=(lo, max(hi, lo + 1e-6)), out_range=(0, 1))


def grid(images, titles, path, suptitle):
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    for ax, im, t in zip(np.atleast_1d(axes).ravel(), images, titles):
        ax.imshow(stretch(im), cmap="gray")
        ax.set_title(t, fontsize=10)
        ax.axis("off")
    fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Input stack not found: {INPUT_FILE}")
    os.makedirs(OUT_FIG_DIR, exist_ok=True)

    stack = tifffile.imread(INPUT_FILE)  # ZCYX
    if stack.ndim != 4:
        raise ValueError(f"Expected 4D ZCYX stack, got shape {stack.shape}")
    Z, C, Y, X = stack.shape
    print(f"Loaded {os.path.basename(INPUT_FILE)}: ZCYX={stack.shape}, dtype={stack.dtype}")

    print(f"{'chan':>4} {'mean':>9} {'max':>7} {'nonzero%':>9}  pctile 50/99/99.9")
    for c in range(C):
        ch = stack[:, c]
        p = [int(np.percentile(ch, q)) for q in (50, 99, 99.9)]
        print(f"C{c:>3} {ch.mean():9.2f} {ch.max():7d} {100 * (ch > 0).mean():8.1f}%  {p}")

    zmid = Z // 2
    maxproj = [stack[:, c].max(axis=0) for c in range(C)]
    midsl = [stack[zmid, c] for c in range(C)]
    titles = [f"C{c}" for c in range(C)]
    grid(maxproj, titles, OUT_MAXPROJ, "Max-Z projection per channel")
    grid(midsl, titles, OUT_MIDSLICE, f"Mid-Z slice (z={zmid}) per channel")
    print(f"Wrote {OUT_MAXPROJ}")
    print(f"Wrote {OUT_MIDSLICE}")


if __name__ == "__main__":
    main()
