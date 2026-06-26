"""Cellpose-SAM nuclei segmentation (DAPI / C0) -- 2D per-slice + Z-stitch.

Cellpose v4 (cpsam): segments each Z-slice in 2D then links across Z by mask IoU
(do_3D=False, stitch_threshold>0). No anisotropy needed, matching the unknown
Z-step. For the true-3D variant see 02c.segment_nuclei_cellpose_3d.py.
See plan/cell_seg/01.nuclei_segmentation_plan.md. Run with:
    conda run --no-capture-output -n img python -u scripts/cell_seg/02.segment_nuclei_cellpose.py
"""
import os
import time

import numpy as np
import tifffile
from skimage import segmentation, exposure
from cellpose import models

# ---- File paths (input/output) ----
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INPUT_FILE   = os.path.join(PROJECT_ROOT, 'links', 'cell_seg', 'cell_segmentation_Saumya.tif')
OUT_RES_DIR  = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'cell_seg')
OUT_LABELS   = os.path.join(OUT_RES_DIR, '02.nuclei_labels_cellpose.tif')

# ---- Config ----
NUC_CHANNEL        = 0          # C0 = DAPI
PER_Z_NORM         = True       # rescale each Z-slice (1-99.8 pctile) to counter depth dimming
DIAMETER           = None       # None -> cpsam diam_mean (~30 px); else value in px
FLOW_THRESHOLD     = 0.4
CELLPROB_THRESHOLD = -1.0       # lower = recover dimmer/deeper nuclei
STITCH_THRESHOLD   = 0.4        # IoU to link masks across Z-slices
MIN_SIZE           = 50         # drop masks smaller than this (voxels)
BATCH_SIZE         = 16         # GPU has headroom
PNORM_LO, PNORM_HI = 1.0, 99.8  # per-Z normalization percentiles


def normalize_per_slice(vol):
    """Rescale each Z-slice to its own [PNORM_LO, PNORM_HI] percentile range -> float [0,1]."""
    out = np.empty(vol.shape, dtype=np.float32)
    for z in range(vol.shape[0]):
        lo, hi = np.percentile(vol[z], [PNORM_LO, PNORM_HI])
        out[z] = 0.0 if hi <= lo else exposure.rescale_intensity(
            vol[z], in_range=(lo, hi), out_range=(0.0, 1.0))
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

    if PER_Z_NORM:
        vol = normalize_per_slice(vol)
        print("Applied per-Z percentile normalization")

    model = models.CellposeModel(gpu=True)
    print(f"CellposeModel ready (gpu={model.gpu}); 2D+stitch, diameter={DIAMETER}, "
          f"cellprob={CELLPROB_THRESHOLD}, stitch={STITCH_THRESHOLD}, per_z_norm={PER_Z_NORM}")

    t0 = time.time()
    masks, flows, styles = model.eval(
        vol,
        z_axis=0,
        channel_axis=None,
        normalize=not PER_Z_NORM,  # avoid double-normalizing if we pre-normalized per-Z
        diameter=DIAMETER,
        flow_threshold=FLOW_THRESHOLD,
        cellprob_threshold=CELLPROB_THRESHOLD,
        do_3D=False,
        stitch_threshold=STITCH_THRESHOLD,
        min_size=MIN_SIZE,
        batch_size=BATCH_SIZE,
    )
    dt = time.time() - t0

    labels, _, _ = segmentation.relabel_sequential(masks.astype(np.int32))
    n = int(labels.max())
    dtype = np.uint16 if n < 65535 else np.uint32
    tifffile.imwrite(OUT_LABELS, labels.astype(dtype), imagej=True)
    print(f"2D+stitch -> {n} nuclei in {dt:.1f}s. Wrote {OUT_LABELS}")


if __name__ == "__main__":
    main()
