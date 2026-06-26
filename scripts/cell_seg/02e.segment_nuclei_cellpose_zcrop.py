"""Cellpose-SAM nuclei segmentation (DAPI / C0) -- best procedure on Z-cropped stack.

Reruns the baseline 02 procedure (per-Z percentile normalize, cpsam 2D + Z-stitch,
cellprob=-1.0, stitch=0.4) but on only the well-resolved slices, dropping the deep
low-contrast tail (z >= Z_END) that segments poorly. Per-slice contrast (p99/median)
peaks at z~18-24 and falls to ~3.1 over z~27-38 -> drop those. The label volume is
padded back to full Z (dropped slices = 0) so 03.measure_and_qc.py works unchanged.
See plan/cell_seg/01.nuclei_segmentation_plan.md. Run with:
    conda run --no-capture-output -n img python -u scripts/cell_seg/02e.segment_nuclei_cellpose_zcrop.py
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
OUT_LABELS   = os.path.join(OUT_RES_DIR, '02e.nuclei_labels_cellpose_zcrop.tif')

# ---- Config ----
NUC_CHANNEL        = 0          # C0 = DAPI
Z_START            = 0          # keep slices [Z_START, Z_END)
Z_END              = 27         # exclusive; drops deep low-contrast tail z>=27 (12 of 39)
PER_Z_NORM         = True       # per-Z percentile normalization (baseline 02 procedure)
DIAMETER           = None       # None -> cpsam diam_mean (~30 px)
FLOW_THRESHOLD     = 0.4
CELLPROB_THRESHOLD = -1.0
STITCH_THRESHOLD   = 0.4        # IoU to link masks across Z-slices
MIN_SIZE           = 50         # drop masks smaller than this (voxels)
BATCH_SIZE         = 16
PNORM_LO, PNORM_HI = 1.0, 99.8


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
    Z = stack.shape[0]
    if not (0 <= Z_START < Z_END <= Z):
        raise ValueError(f"Bad Z range [{Z_START},{Z_END}) for stack with Z={Z}")
    full = stack[:, NUC_CHANNEL].astype(np.float32)  # (Z, Y, X)
    vol = full[Z_START:Z_END]
    print(f"Loaded C{NUC_CHANNEL} volume {full.shape}; keeping z[{Z_START}:{Z_END}] -> {vol.shape} "
          f"(dropped {Z - (Z_END - Z_START)} deep slices)")

    if PER_Z_NORM:
        vol = normalize_per_slice(vol)
        print("Applied per-Z percentile normalization")

    model = models.CellposeModel(gpu=True)
    print(f"CellposeModel ready (gpu={model.gpu}); 2D+stitch, diameter={DIAMETER}, "
          f"cellprob={CELLPROB_THRESHOLD}, stitch={STITCH_THRESHOLD}")

    t0 = time.time()
    masks, flows, styles = model.eval(
        vol,
        z_axis=0,
        channel_axis=None,
        normalize=not PER_Z_NORM,
        diameter=DIAMETER,
        flow_threshold=FLOW_THRESHOLD,
        cellprob_threshold=CELLPROB_THRESHOLD,
        do_3D=False,
        stitch_threshold=STITCH_THRESHOLD,
        min_size=MIN_SIZE,
        batch_size=BATCH_SIZE,
    )
    dt = time.time() - t0

    masks, _, _ = segmentation.relabel_sequential(masks.astype(np.int32))
    # pad back to full Z so downstream measure/QC matches the raw stack
    labels = np.zeros(full.shape, dtype=np.int32)
    labels[Z_START:Z_END] = masks
    n = int(labels.max())
    dtype = np.uint16 if n < 65535 else np.uint32
    tifffile.imwrite(OUT_LABELS, labels.astype(dtype), imagej=True)
    print(f"zcrop 2D+stitch -> {n} nuclei in {dt:.1f}s. Wrote {OUT_LABELS}")


if __name__ == "__main__":
    main()
