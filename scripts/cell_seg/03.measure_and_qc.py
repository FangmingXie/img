"""Measure + QC for a nuclei label volume (generic over any label TIFF).

Produces a per-cell CSV, a count summary, and QC overlay PNGs. Works for both the
watershed (02b) and Cellpose (02) masks -- set LABELS_FILE / TAG below, or pass a
label TIFF path as the first CLI arg. See plan/cell_seg/02.watershed_baseline_plan.md.
Run with:
    conda run --no-capture-output -n img python -u scripts/cell_seg/03.measure_and_qc.py [labels.tif]
"""
import os
import sys

import numpy as np
import pandas as pd
import tifffile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage import measure, segmentation, exposure

# ---- File paths (input/output) ----
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INPUT_FILE   = os.path.join(PROJECT_ROOT, 'links', 'cell_seg', 'cell_segmentation_Saumya.tif')
OUT_RES_DIR  = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'cell_seg')
OUT_FIG_DIR  = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'cell_seg')
LABELS_FILE  = os.path.join(OUT_RES_DIR, '02b.nuclei_labels_watershed.tif')

# ---- Config ----
TAG          = "watershed"   # output suffix; overridden from the label filename if passed via CLI
NUC_CHANNEL  = 0             # channel used as the QC background (DAPI)
N_QC_SLICES  = 6            # number of evenly spaced Z-slices in the slice montage
DISP_LO      = 1.0          # display normalization percentiles
DISP_HI      = 99.8
VOXEL_XY_UM  = 0.0575       # 57.5 nm/px (XY)
VOXEL_Z_UM   = 0.160        # 160 nm/step (Z)
VOXEL_UM3    = VOXEL_XY_UM * VOXEL_XY_UM * VOXEL_Z_UM  # µm³ per voxel


def to_display_rgb(gray, boundaries=None):
    """Percentile-stretch a 2D gray image to RGB uint8; paint boundaries red if given."""
    lo, hi = np.percentile(gray, [DISP_LO, DISP_HI])
    g = exposure.rescale_intensity(gray.astype(np.float32), in_range=(lo, max(hi, lo + 1e-6)),
                                   out_range=(0, 255)).astype(np.uint8)
    rgb = np.stack([g, g, g], axis=-1)
    if boundaries is not None:
        rgb[boundaries] = [255, 0, 0]
    return rgb


def measure_table(labels, stack):
    """Per-object regionprops with mean intensity in every channel."""
    intensity = np.moveaxis(stack, 1, -1)  # ZCYX -> ZYXC for multichannel regionprops
    props = measure.regionprops_table(
        labels, intensity_image=intensity,
        properties=("label", "centroid", "area", "intensity_mean"),
    )
    df = pd.DataFrame(props)
    rename = {"centroid-0": "centroid_z", "centroid-1": "centroid_y", "centroid-2": "centroid_x"}
    n_ch = stack.shape[1]
    for c in range(n_ch):
        rename[f"intensity_mean-{c}"] = f"mean_C{c}"
    df = df.rename(columns=rename)
    df = df.rename(columns={"area": "volume_voxels"})
    df["volume_um3"] = df["volume_voxels"] * VOXEL_UM3
    # physical centroids (µm): z uses Z spacing, y/x use XY spacing
    df["centroid_z_um"] = df["centroid_z"] * VOXEL_Z_UM
    df["centroid_y_um"] = df["centroid_y"] * VOXEL_XY_UM
    df["centroid_x_um"] = df["centroid_x"] * VOXEL_XY_UM
    return df


def write_summary(df, path, mode_label):
    vol = df["volume_voxels"].to_numpy()
    um3 = df["volume_um3"].to_numpy()
    lines = [
        f"Label source: {mode_label}",
        f"Voxel size: XY={VOXEL_XY_UM} µm, Z={VOXEL_Z_UM} µm ({VOXEL_UM3:.3e} µm³/voxel)",
        f"Nuclei count: {len(df)}",
        "",
        "Volume (voxels):",
        f"  min/median/mean/max: {vol.min():.0f} / {np.median(vol):.0f} / {vol.mean():.1f} / {vol.max():.0f}",
        f"  pctiles 10/25/75/90: " + " / ".join(f"{np.percentile(vol, q):.0f}" for q in (10, 25, 75, 90)),
        "",
        "Volume (µm³):",
        f"  min/median/mean/max: {um3.min():.2f} / {np.median(um3):.2f} / {um3.mean():.2f} / {um3.max():.2f}",
        f"  pctiles 10/25/75/90: " + " / ".join(f"{np.percentile(um3, q):.2f}" for q in (10, 25, 75, 90)),
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def qc_slice_montage(c0, labels, path):
    """Grid of evenly spaced Z-slices with mask boundaries over the DAPI channel."""
    Z = c0.shape[0]
    zs = np.linspace(0, Z - 1, min(N_QC_SLICES, Z)).round().astype(int)
    ncol = 3
    nrow = int(np.ceil(len(zs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 4 * nrow))
    for ax, z in zip(np.atleast_1d(axes).ravel(), zs):
        b = segmentation.find_boundaries(labels[z], mode="outer")
        ax.imshow(to_display_rgb(c0[z], b))
        ax.set_title(f"z={z}  (n={len(np.unique(labels[z])) - 1})", fontsize=9)
        ax.axis("off")
    for ax in np.atleast_1d(axes).ravel()[len(zs):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def qc_maxproj(c0, labels, path):
    """Max-Z projection of DAPI with boundaries of the max-projected labels."""
    proj = c0.max(axis=0)
    lab_proj = labels.max(axis=0)
    b = segmentation.find_boundaries(lab_proj, mode="outer")
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(to_display_rgb(proj, b))
    ax.set_title("C0 max-Z projection + mask boundaries", fontsize=10)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    labels_file = sys.argv[1] if len(sys.argv) > 1 else LABELS_FILE
    tag = TAG
    if len(sys.argv) > 1:
        stem = os.path.splitext(os.path.basename(labels_file))[0]
        tag = stem.split("_")[-1]  # e.g. 02.nuclei_labels_cellpose -> 'cellpose'
    if not os.path.exists(labels_file):
        raise FileNotFoundError(f"Label volume not found: {labels_file}")
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Input stack not found: {INPUT_FILE}")
    os.makedirs(OUT_RES_DIR, exist_ok=True)
    os.makedirs(OUT_FIG_DIR, exist_ok=True)

    stack = tifffile.imread(INPUT_FILE)        # ZCYX
    labels = tifffile.imread(labels_file)      # ZYX
    if labels.shape != stack.shape[0:1] + stack.shape[2:]:
        raise ValueError(f"Label shape {labels.shape} does not match stack ZYX {stack.shape[0:1] + stack.shape[2:]}")
    c0 = stack[:, NUC_CHANNEL]
    n = int(labels.max())
    print(f"Loaded labels {labels.shape} ({n} objects) for tag={tag!r}")

    df = measure_table(labels, stack)
    csv_path = os.path.join(OUT_RES_DIR, f"03.nuclei_measurements_{tag}.csv")
    df.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path} ({len(df)} rows)")
    if len(df) != n:
        print(f"  NOTE: regionprops rows ({len(df)}) != max label ({n}); labels may be non-contiguous.")

    write_summary(df, os.path.join(OUT_RES_DIR, f"03.count_summary_{tag}.txt"), labels_file)

    qc_slice_montage(c0, labels, os.path.join(OUT_FIG_DIR, f"03.qc_slices_{tag}.png"))
    qc_maxproj(c0, labels, os.path.join(OUT_FIG_DIR, f"03.qc_maxproj_{tag}.png"))
    print(f"Wrote QC overlays to {OUT_FIG_DIR}")


if __name__ == "__main__":
    main()
