"""Watershed baseline for 3D nuclei segmentation (DAPI / C0).

Distance-transform marker-controlled watershed. See
plan/cell_seg/02.watershed_baseline_plan.md. Run with:
    conda run --no-capture-output -n img python -u scripts/cell_seg/02b.segment_watershed.py
"""
import os

import numpy as np
import tifffile
from scipy import ndimage as ndi
from skimage import exposure, filters, morphology, segmentation, feature

# ---- File paths (input/output) ----
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INPUT_FILE   = os.path.join(PROJECT_ROOT, 'links', 'cell_seg', 'cell_segmentation_Saumya.tif')
OUT_RES_DIR  = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'cell_seg')
OUT_LABELS   = os.path.join(OUT_RES_DIR, '02b.nuclei_labels_watershed.tif')

# ---- Config ----
NUC_CHANNEL  = 0          # C0 = DAPI
MODE         = "3d"       # "3d" | "slice_stitch"
ANISOTROPY_Z = 3.0        # estimated Z:XY voxel ratio (Z-step unrecorded)
SIGMA_XY     = 1.5
SIGMA_Z      = 1.0
PNORM_LO     = 1.0        # per-slice percentile normalization
PNORM_HI     = 99.8
THRESHOLD    = "otsu"     # "otsu" | "local"
LOCAL_BLOCK  = 51         # block size for threshold_local (odd)
MIN_DISTANCE = 10         # peak_local_max separation (px)
DIST_SMOOTH  = 2.0        # gaussian sigma (XY) on the distance map before seeding (curbs over-split)
SEED_MODE    = "peak"     # "peak" | "hmaxima"
H_MAXIMA     = 2.0
MIN_VOXELS   = 50
MAX_VOXELS   = 50000
STITCH_IOU   = 0.3        # slice_stitch mode only


def normalize_per_slice(vol):
    """Rescale each Z-slice to its own [PNORM_LO, PNORM_HI] percentile range -> float [0,1]."""
    out = np.empty(vol.shape, dtype=np.float32)
    for z in range(vol.shape[0]):
        sl = vol[z]
        lo, hi = np.percentile(sl, [PNORM_LO, PNORM_HI])
        if hi <= lo:
            out[z] = 0.0
        else:
            out[z] = exposure.rescale_intensity(sl, in_range=(lo, hi), out_range=(0.0, 1.0))
    return out


def foreground_mask(smoothed):
    """Binary foreground from a smoothed (float) image of any dimensionality."""
    if THRESHOLD == "otsu":
        thr = filters.threshold_otsu(smoothed)
        return smoothed > thr
    elif THRESHOLD == "local":
        # threshold_local works per-2D-plane; apply slice-wise if 3D
        if smoothed.ndim == 3:
            fg = np.zeros(smoothed.shape, dtype=bool)
            for z in range(smoothed.shape[0]):
                t = filters.threshold_local(smoothed[z], block_size=LOCAL_BLOCK)
                fg[z] = smoothed[z] > t
            return fg
        t = filters.threshold_local(smoothed, block_size=LOCAL_BLOCK)
        return smoothed > t
    raise ValueError(f"Unknown THRESHOLD={THRESHOLD!r}; expected 'otsu' or 'local'")


def clean_mask(fg):
    """Fill holes and drop tiny connected components."""
    fg = ndi.binary_fill_holes(fg)
    fg = morphology.remove_small_objects(fg, min_size=MIN_VOXELS)
    return fg


def make_markers(distance, fg, smooth_xy):
    """Seed markers from a smoothed copy of the distance map (curbs over-segmentation)."""
    seed_dist = filters.gaussian(distance, sigma=(0.0, smooth_xy, smooth_xy)) if distance.ndim == 3 \
        else filters.gaussian(distance, sigma=(smooth_xy, smooth_xy))
    if SEED_MODE == "peak":
        coords = feature.peak_local_max(
            seed_dist, min_distance=MIN_DISTANCE, labels=fg, exclude_border=False
        )
        peaks = np.zeros(distance.shape, dtype=bool)
        if coords.size:
            peaks[tuple(coords.T)] = True
    elif SEED_MODE == "hmaxima":
        peaks = morphology.h_maxima(seed_dist, H_MAXIMA) > 0
        peaks &= fg
    else:
        raise ValueError(f"Unknown SEED_MODE={SEED_MODE!r}; expected 'peak' or 'hmaxima'")
    markers, n = ndi.label(peaks)
    if n == 0:
        raise RuntimeError("No seed markers found; loosen MIN_DISTANCE / SEED_MODE or check input.")
    return markers


def size_filter(labels):
    """Remove objects outside [MIN_VOXELS, MAX_VOXELS] and relabel sequentially."""
    counts = np.bincount(labels.ravel())
    counts[0] = 0  # background
    keep = (counts >= MIN_VOXELS) & (counts <= MAX_VOXELS)
    drop = np.where(~keep)[0]
    if drop.size:
        labels[np.isin(labels, drop)] = 0
    labels, _, _ = segmentation.relabel_sequential(labels)
    return labels


def watershed_3d(vol):
    """Full-3D distance-transform watershed with anisotropic Z sampling."""
    smoothed = filters.gaussian(vol, sigma=(SIGMA_Z, SIGMA_XY, SIGMA_XY))
    fg = clean_mask(foreground_mask(smoothed))
    if not fg.any():
        raise RuntimeError("Empty foreground after thresholding; adjust THRESHOLD/normalization.")
    distance = ndi.distance_transform_edt(fg, sampling=(ANISOTROPY_Z, 1.0, 1.0))
    markers = make_markers(distance, fg, DIST_SMOOTH)
    labels = segmentation.watershed(-distance, markers, mask=fg)
    return size_filter(labels)


def watershed_2d_slice(plane):
    """2D distance-transform watershed on a single normalized Z-slice."""
    smoothed = filters.gaussian(plane, sigma=(SIGMA_XY, SIGMA_XY))
    fg = clean_mask(foreground_mask(smoothed))
    if not fg.any():
        return np.zeros(plane.shape, dtype=np.int32)
    distance = ndi.distance_transform_edt(fg)
    seed_dist = filters.gaussian(distance, sigma=(DIST_SMOOTH, DIST_SMOOTH))
    coords = feature.peak_local_max(
        seed_dist, min_distance=MIN_DISTANCE, labels=fg, exclude_border=False
    )
    peaks = np.zeros(distance.shape, dtype=bool)
    if coords.size:
        peaks[tuple(coords.T)] = True
    markers, n = ndi.label(peaks)
    if n == 0:
        return np.zeros(plane.shape, dtype=np.int32)
    return segmentation.watershed(-distance, markers, mask=fg)


def stitch_slices(vol):
    """Per-slice 2D watershed, then link labels across Z by IoU > STITCH_IOU."""
    Z = vol.shape[0]
    out = np.zeros(vol.shape, dtype=np.int32)
    next_id = 1
    prev = None
    for z in range(Z):
        cur = watershed_2d_slice(vol[z])
        relabeled = np.zeros(cur.shape, dtype=np.int32)
        for lbl in np.unique(cur):
            if lbl == 0:
                continue
            cur_mask = cur == lbl
            assigned = 0
            if prev is not None:
                overlap = prev[cur_mask]
                overlap = overlap[overlap > 0]
                if overlap.size:
                    cand = np.bincount(overlap).argmax()
                    inter = int((prev == cand)[cur_mask].sum())
                    union = int((prev == cand).sum() + cur_mask.sum() - inter)
                    if union > 0 and inter / union > STITCH_IOU:
                        assigned = cand
            if assigned == 0:
                assigned = next_id
                next_id += 1
            relabeled[cur_mask] = assigned
        out[z] = relabeled
        prev = relabeled
    return size_filter(out)


def main():
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Input stack not found: {INPUT_FILE}")
    os.makedirs(OUT_RES_DIR, exist_ok=True)

    stack = tifffile.imread(INPUT_FILE)  # ZCYX
    if stack.ndim != 4:
        raise ValueError(f"Expected 4D ZCYX stack, got shape {stack.shape}")
    vol = stack[:, NUC_CHANNEL].astype(np.float32)  # (Z, Y, X)
    print(f"Loaded C{NUC_CHANNEL} volume {vol.shape} from {os.path.basename(INPUT_FILE)}")

    norm = normalize_per_slice(vol)

    if MODE == "3d":
        labels = watershed_3d(norm)
    elif MODE == "slice_stitch":
        labels = stitch_slices(norm)
    else:
        raise ValueError(f"Unknown MODE={MODE!r}; expected '3d' or 'slice_stitch'")

    n = int(labels.max())
    dtype = np.uint16 if n < 65535 else np.uint32
    tifffile.imwrite(OUT_LABELS, labels.astype(dtype), imagej=True)
    print(f"MODE={MODE} -> {n} nuclei. Wrote {OUT_LABELS}")


if __name__ == "__main__":
    main()
