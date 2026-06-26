"""Cellpose-SAM nuclei segmentation (DAPI / C0) -- enhanced 2D + Z-stitch.

Same 2D-per-slice + Z-stitch pipeline as 02.segment_nuclei_cellpose.py, plus
per-slice preprocessing to recover dim/blurry deep nuclei and Cellpose test-time
augmentation:
  - white tophat background subtraction (flatten low-frequency haze)
  - per-Z percentile normalization
  - CLAHE local contrast equalization (make every slice locally high-contrast)
  - eval(augment=True)
Each step is a config toggle so it can be A/B-tested against 02's output.
See plan/cell_seg/01.nuclei_segmentation_plan.md. Run with:
    conda run --no-capture-output -n img python -u scripts/cell_seg/02d.segment_nuclei_cellpose_enhanced.py
"""
import os
import time

import numpy as np
import tifffile
from skimage import segmentation, exposure, filters
from skimage.morphology import white_tophat, disk
from cellpose import models

# ---- File paths (input/output) ----
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INPUT_FILE   = os.path.join(PROJECT_ROOT, 'links', 'cell_seg', 'cell_segmentation_Saumya.tif')
OUT_RES_DIR  = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'cell_seg')
OUT_LABELS   = os.path.join(OUT_RES_DIR, '02d.nuclei_labels_cellpose_enhanced.tif')

# ---- Config ----
NUC_CHANNEL        = 0          # C0 = DAPI
# preprocessing (per Z-slice)
USE_TOPHAT         = True       # white-tophat background subtraction
TOPHAT_RADIUS      = 25         # disk radius (px); > nucleus radius -> flatten background only
GAUSSIAN_SIGMA     = 1.0        # light denoise after tophat (0 to disable); curbs deep-slice speckle
PNORM_LO, PNORM_HI = 1.0, 99.8  # per-Z percentile normalization
USE_CLAHE          = True       # local contrast equalization
CLAHE_CLIP         = 0.005      # higher = stronger local contrast (lower -> less noise amplification)
CLAHE_KERNEL       = 64         # px; larger than a nucleus -> equalize across nuclei, not within
# cellpose
DIAMETER           = None       # None -> cpsam diam_mean (~30 px); else value in px
FLOW_THRESHOLD     = 0.4
CELLPROB_THRESHOLD = -1.0       # lower = recover dimmer/deeper nuclei
STITCH_THRESHOLD   = 0.4        # IoU to link masks across Z-slices
MIN_SIZE           = 500        # drop masks smaller than this (voxels); real nuclei are >2000
BATCH_SIZE         = 16
AUGMENT            = True       # test-time augmentation (steadier on hard slices)


def preprocess(vol):
    """Per-slice: (optional) tophat -> percentile normalize -> (optional) CLAHE. -> float [0,1]."""
    out = np.empty(vol.shape, dtype=np.float32)
    footprint = disk(TOPHAT_RADIUS) if USE_TOPHAT else None
    for z in range(vol.shape[0]):
        img = vol[z].astype(np.float32)
        if USE_TOPHAT:
            img = white_tophat(img, footprint=footprint)
        if GAUSSIAN_SIGMA > 0:
            img = filters.gaussian(img, sigma=GAUSSIAN_SIGMA)
        lo, hi = np.percentile(img, [PNORM_LO, PNORM_HI])
        if hi <= lo:
            out[z] = 0.0
            continue
        img = np.clip(exposure.rescale_intensity(img, in_range=(lo, hi), out_range=(0.0, 1.0)), 0.0, 1.0)
        if USE_CLAHE:
            img = exposure.equalize_adapthist(img, kernel_size=CLAHE_KERNEL, clip_limit=CLAHE_CLIP)
        out[z] = img
    return out


def main():
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Input stack not found: {INPUT_FILE}")
    os.makedirs(OUT_RES_DIR, exist_ok=True)

    stack = tifffile.imread(INPUT_FILE)  # ZCYX
    if stack.ndim != 4:
        raise ValueError(f"Expected 4D ZCYX stack, got shape {stack.shape}")
    vol = stack[:, NUC_CHANNEL].astype(np.float32)  # (Z, Y, X)
    print(f"Loaded C{NUC_CHANNEL} volume {vol.shape} from {os.path.basename(INPUT_FILE)}")

    vol = preprocess(vol)
    print(f"Preprocess: tophat={USE_TOPHAT}(r={TOPHAT_RADIUS}), clahe={USE_CLAHE}(clip={CLAHE_CLIP})")

    model = models.CellposeModel(gpu=True)
    print(f"CellposeModel ready (gpu={model.gpu}); 2D+stitch, diameter={DIAMETER}, "
          f"cellprob={CELLPROB_THRESHOLD}, stitch={STITCH_THRESHOLD}, augment={AUGMENT}")

    t0 = time.time()
    masks, flows, styles = model.eval(
        vol,
        z_axis=0,
        channel_axis=None,
        normalize=False,  # we already normalized per-slice
        diameter=DIAMETER,
        flow_threshold=FLOW_THRESHOLD,
        cellprob_threshold=CELLPROB_THRESHOLD,
        do_3D=False,
        stitch_threshold=STITCH_THRESHOLD,
        min_size=MIN_SIZE,
        batch_size=BATCH_SIZE,
        augment=AUGMENT,
    )
    dt = time.time() - t0

    labels, _, _ = segmentation.relabel_sequential(masks.astype(np.int32))
    n = int(labels.max())
    dtype = np.uint16 if n < 65535 else np.uint32
    tifffile.imwrite(OUT_LABELS, labels.astype(dtype), imagej=True)
    print(f"enhanced 2D+stitch -> {n} nuclei in {dt:.1f}s. Wrote {OUT_LABELS}")


if __name__ == "__main__":
    main()
