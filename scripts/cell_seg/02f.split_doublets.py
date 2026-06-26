"""Split fused doublet/cluster nuclei in a label volume via intensity-guided watershed.

Post-processing for the Cellpose masks: oversized objects (fused nuclei) are split
with a marker-controlled watershed *within each mask*. Seeds come from peaks of the
heavily-smoothed DAPI intensity (~one peak per real nucleus, sub-nuclear texture
smoothed away), so splits fall in the dim valleys between nuclei rather than along
arbitrary geometric cuts. The watershed runs on the negative smoothed intensity;
sliver parts are merged into their largest neighbour. Default input is the Z-crop
result (02e). See report/cell_seg/02.zcrop_deep_slice_results.md. Run with:
    conda run --no-capture-output -n img python -u scripts/cell_seg/02f.split_doublets.py [labels.tif]
"""
import os
import sys

import numpy as np
import tifffile
from scipy import ndimage as ndi
from skimage import segmentation, filters, feature, exposure

# ---- File paths (input/output) ----
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INPUT_FILE   = os.path.join(PROJECT_ROOT, 'links', 'cell_seg', 'cell_segmentation_Saumya.tif')
OUT_RES_DIR  = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'cell_seg')
LABELS_IN    = os.path.join(OUT_RES_DIR, '02e.nuclei_labels_cellpose_zcrop.tif')
OUT_LABELS   = os.path.join(OUT_RES_DIR, '02f.nuclei_labels_split.tif')

# ---- Config ----
NUC_CHANNEL        = 0
VOXEL_XY_UM        = 0.0575
VOXEL_Z_UM         = 0.160
ANISOTROPY_Z       = VOXEL_Z_UM / VOXEL_XY_UM         # 2.78
VOXEL_UM3          = VOXEL_XY_UM * VOXEL_XY_UM * VOXEL_Z_UM
SPLIT_ABOVE_UM3    = 18.0     # only attempt to split masks bigger than this (~2x a typical nucleus)
MIN_PART_UM3       = 2.0      # parts smaller than this are merged into their largest neighbour
INTENSITY_SMOOTH_XY = 6.0     # gaussian sigma (XY) on DAPI -> ~one peak per nucleus
MIN_DISTANCE       = 15       # peak_local_max separation (XY px) -- ~nucleus diameter
PNORM_LO, PNORM_HI = 1.0, 99.8

SPLIT_ABOVE_VOX = int(SPLIT_ABOVE_UM3 / VOXEL_UM3)
MIN_PART_VOX    = int(MIN_PART_UM3 / VOXEL_UM3)


def normalize_per_slice(vol):
    out = np.empty(vol.shape, dtype=np.float32)
    for z in range(vol.shape[0]):
        lo, hi = np.percentile(vol[z], [PNORM_LO, PNORM_HI])
        out[z] = 0.0 if hi <= lo else exposure.rescale_intensity(
            vol[z], in_range=(lo, hi), out_range=(0.0, 1.0))
    return out


def merge_slivers(ws):
    """Reassign parts smaller than MIN_PART_VOX to their largest adjacent part."""
    sizes = np.bincount(ws.ravel())
    sizes[0] = 0
    small = [l for l in np.where((sizes > 0) & (sizes < MIN_PART_VOX))[0]]
    for l in small:
        region = ws == l
        border = ndi.binary_dilation(region) & ~region
        neigh = ws[border]
        neigh = neigh[(neigh != 0) & (neigh != l)]
        if neigh.size:
            ws[region] = np.bincount(neigh).argmax()
    return ws


def try_split(mask, smoothed):
    """Split a single binary object using intensity peaks; return labelled split or None."""
    coords = feature.peak_local_max(smoothed, min_distance=MIN_DISTANCE,
                                    labels=mask, exclude_border=False)
    if len(coords) < 2:
        return None
    peaks = np.zeros(mask.shape, dtype=bool)
    peaks[tuple(coords.T)] = True
    markers, n = ndi.label(peaks)
    if n < 2:
        return None
    ws = segmentation.watershed(-smoothed, markers, mask=mask)
    ws = merge_slivers(ws)
    if len([l for l in np.unique(ws) if l != 0]) < 2:
        return None
    return ws


def main():
    labels_in = sys.argv[1] if len(sys.argv) > 1 else LABELS_IN
    if not os.path.exists(labels_in):
        raise FileNotFoundError(f"Label volume not found: {labels_in}")
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Input stack not found: {INPUT_FILE}")
    os.makedirs(OUT_RES_DIR, exist_ok=True)

    labels = tifffile.imread(labels_in).astype(np.int32)
    dapi = normalize_per_slice(tifffile.imread(INPUT_FILE)[:, NUC_CHANNEL].astype(np.float32))
    sz = INTENSITY_SMOOTH_XY / ANISOTROPY_Z  # equal physical smoothing in Z
    smoothed = filters.gaussian(dapi, sigma=(sz, INTENSITY_SMOOTH_XY, INTENSITY_SMOOTH_XY))

    out = labels.copy()
    next_id = int(labels.max()) + 1
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    big = list(np.where(sizes >= SPLIT_ABOVE_VOX)[0])
    print(f"Loaded {labels_in}: {int(labels.max())} objects; "
          f"{len(big)} above {SPLIT_ABOVE_UM3} µm³ ({SPLIT_ABOVE_VOX} vox) -> split-candidates")

    n_split = 0
    for lbl in big:
        mask = labels == lbl
        ws = try_split(mask, smoothed)
        if ws is None:
            continue
        parts = [p for p in np.unique(ws) if p != 0]
        for i, p in enumerate(parts):
            out[ws == p] = lbl if i == 0 else next_id
            if i > 0:
                next_id += 1
        n_split += 1
        print(f"  label {lbl} ({sizes[lbl]} vox, {sizes[lbl]*VOXEL_UM3:.1f} µm³) -> {len(parts)} parts")

    out, _, _ = segmentation.relabel_sequential(out)
    n = int(out.max())
    dtype = np.uint16 if n < 65535 else np.uint32
    tifffile.imwrite(OUT_LABELS, out.astype(dtype), imagej=True)
    print(f"Split {n_split} object(s): {int(labels.max())} -> {n} nuclei. Wrote {OUT_LABELS}")


if __name__ == "__main__":
    main()
