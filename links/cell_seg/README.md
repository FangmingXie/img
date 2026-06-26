# cell_seg data

## Files

- `cell_segmentation_Saumya.tif` → symlink to
  `/home/qlyu/mydata/data/jainlab26_img/cell_segmentation_Saumya.tif` (21 MB)

## Derived image information

Read from the TIFF/ImageJ metadata and pixel data (via `tifffile`).

| Property | Value |
|---|---|
| Dimensions (ZCYX) | **39 × 4 × 262 × 266** |
| Data type | `uint16` |
| Format | ImageJ hyperstack (composite), big-endian |
| Pages | 156 (= 39 Z × 4 C) |
| Unit | micron |
| **XY voxel size** | **0.0575 µm/px (57.5 nm)** — confirmed by user; matches the embedded resolution tag |
| **Z voxel size** | **0.160 µm/step (160 nm)** — confirmed by user (not in file metadata) |
| Anisotropy (Z:XY) | **2.78** |
| Voxel volume | **5.29×10⁻⁴ µm³** |
| Display range (min/max) | 0 / 125 |

### Acquisition (from embedded `Info` metadata)
- Source: `9B08_slp1_RNAiv20_24h_slp1_beatiia_5-28-26_an1_lobe1 - Deconvolved 20 iterations, Type Blind.nd2 (series 1)`
- Microscope: Nikon AX confocal (`Nikon_Confocal_Ax`)
- Original `.nd2` size: SizeX=1024, SizeY=1024, SizeZ=39, SizeC=4, SizeT=1
  (XY downsampled to ~262×266 in this TIFF)
- Deconvolved: 20 iterations, Blind
- *Drosophila* sample; markers referenced: slp1, beat-IIa; region "lobe1"
- Fluorophore components referenced: DAPI, AF488, AF568, AF647

### Per-channel intensity stats (whole stack)

| Channel | mean | max | coverage | likely identity |
|---|---|---|---|---|
| C0 | 78.0 | 382 | dense, uniform (100% nonzero) | **DAPI / nuclei** |
| C1 | 115.9 | 680 | dense | broad marker (cytoplasm/neuropil?) |
| C2 | 7.4 | 336 | sparse | specific antibody |
| C3 | 1.6 | 648 | 51.5% nonzero | sparse specific marker (beat-IIa?) |

C0 is the segmentation target for nuclei (to be visually confirmed via per-channel
max-projection previews). See `plan/cell_seg/01.nuclei_segmentation_plan.md`.

> Note: channel→marker assignments above are inferred from intensity signatures and
> the embedded fluorophore list; confirm against the acquisition record.
