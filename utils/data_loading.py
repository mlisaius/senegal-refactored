from __future__ import annotations

import logging

import numpy as np
import numpy.typing as npt
import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Sentinel-2 and Sentinel-1 normalization constants (defined once here)
# ---------------------------------------------------------------------------

# Per-band mean reflectance values for Sentinel-2 (10 optical bands),
# computed across the Senegal training region.  Used to zero-centre inputs.
S2_BAND_MEAN = np.array(
    [
        1711.0938,
        1308.8511,
        1546.4543,
        3010.1293,
        3106.5083,
        2068.3044,
        2685.0845,
        2931.5889,
        2514.6928,
        1899.4922,
    ],
    dtype=np.float32,
)

# Per-band standard deviation for Sentinel-2 (same 10 bands).
# Dividing by std puts each band on a roughly unit-variance scale.
S2_BAND_STD = np.array(
    [
        1926.1026,
        1862.9751,
        1803.1792,
        1741.7837,
        1677.4543,
        1888.7862,
        1736.3090,
        1715.8104,
        1514.5199,
        1398.4779,
    ],
    dtype=np.float32,
)

# Mean and std for the two Sentinel-1 SAR backscatter bands (VV, VH).
S1_BAND_MEAN = np.array([5484.0407, 3003.7812], dtype=np.float32)
S1_BAND_STD = np.array([1871.2334, 1726.0670], dtype=np.float32)


# ---------------------------------------------------------------------------
# Private helper
# ---------------------------------------------------------------------------


def _load_npy(path: str, description: str, mmap_mode: str | None = None) -> npt.NDArray:
    """Load a .npy file with a descriptive FileNotFoundError on failure."""
    try:
        # mmap_mode="r" avoids loading the whole array into RAM — useful for
        # large feature cubes that are sliced spatially before use.
        return np.load(path, mmap_mode=mmap_mode)
    except FileNotFoundError:
        # Re-raise with a human-readable message pointing to config.yaml.
        raise FileNotFoundError(
            f"Could not find {description} at: {path}\n"
            "Check the paths section in config.yaml."
        )


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def load_config(path: str) -> dict:
    """Load YAML config and resolve {year} tokens in path strings."""
    with open(path) as f:
        cfg = yaml.safe_load(f)

    year = cfg["year"]
    paths = cfg["paths"]

    # Replace every "{year}" placeholder in path strings with the actual year
    # value so callers never have to do string formatting themselves.
    for k, v in paths.items():
        if isinstance(v, str):
            paths[k] = v.format(year=year)

    cfg["paths"] = paths
    return cfg


def validate_config(cfg: dict) -> None:
    # Allowed values for each config key — fail fast before any I/O happens.
    valid_approaches = {"raw", "tessera", "stm", "alphaearth", "specmat"}
    valid_classifications = {"landcover", "maincrop"}
    valid_sampling = {"bypercentage", "bypercentage_count", "bycount"}
    valid_models = {"RandomForest", "LogisticRegression", "XGBOOST", "SVM", "MLP"}

    approach = cfg["approach"]
    if approach not in valid_approaches:
        raise ValueError(
            f"Unknown approach '{approach}'. Choose from: {valid_approaches}"
        )

    classification = cfg["classification"]
    if classification not in valid_classifications:
        raise ValueError(f"Unknown classification '{classification}'.")

    sampling = cfg["training"]["sampling"]
    if sampling not in valid_sampling:
        raise ValueError(f"Unknown sampling '{sampling}'.")

    # cfg["models"] is a list — validate every entry.
    for m in cfg["models"]:
        if m not in valid_models:
            raise ValueError(f"Unknown model '{m}'. Choose from: {valid_models}")


# ---------------------------------------------------------------------------
# Label loading
# ---------------------------------------------------------------------------


def load_labels(cfg: dict) -> tuple[npt.NDArray, npt.NDArray]:
    """Return (labels, field_ids) as squeezed int64/float arrays."""
    # Cast labels to int64 so class codes can be used as array indices.
    labels = _load_npy(cfg["paths"]["label"], "label raster").astype(np.int64).squeeze()
    # field_ids maps every pixel to its parent field polygon ID.
    field_ids = _load_npy(cfg["paths"]["field_ids"], "field ID raster").squeeze()

    # Sanity-check: both rasters must cover exactly the same spatial extent.
    assert (
        labels.shape == field_ids.shape
    ), f"Shape mismatch: labels {labels.shape} vs field_ids {field_ids.shape}"

    return labels, field_ids


# ---------------------------------------------------------------------------
# Per-approach feature loaders  (signature: cfg, h0, h1, w0, w1 -> ndarray)
# Each returns a 2-D array of shape (h*w, n_features).
# ---------------------------------------------------------------------------


def load_features_raw(cfg: dict, h0: int, h1: int, w0: int, w1: int) -> npt.NDArray:
    """
    Load raw S2 + SAR features for a spatial chunk.

    Parameters
    ----------
    cfg : dict
        Pipeline config with ``paths`` keys for ``s2_bands``, ``s2_mask``, ``sar_bands``.
    h0, h1 : int
        Row slice boundaries of the spatial chunk [h0, h1).
    w0, w1 : int
        Column slice boundaries of the spatial chunk [w0, w1).

    Returns
    -------
    npt.NDArray
        2-D array of shape ``(h*w, n_features)`` with normalised S2 + SAR bands.
    """
    paths = cfg["paths"]

    # --- Sentinel-2 ---
    # Array on disk: (time_steps, H, W, n_bands).  Slice spatially before
    # loading into memory to keep RAM usage proportional to chunk size.
    s2 = _load_npy(paths["s2_bands"], "S2 bands", mmap_mode="r")[:, h0:h1, w0:w1, :]
    # Z-score normalise using pre-computed dataset statistics.
    s2 = (s2 - S2_BAND_MEAN) / S2_BAND_STD

    # Cloud mask: 1 = clear, 0 = cloudy.  Squeeze the last singleton band dim
    # so it broadcasts against the (time, h, w, bands) feature array.
    s2_mask = _load_npy(paths["s2_mask"], "S2 cloud mask")[:, h0:h1, w0:w1].squeeze(
        axis=-1
    )
    s2_mask = s2_mask[..., np.newaxis]  # re-add band dim for broadcasting
    s2 = s2 * s2_mask  # zero out cloudy observations

    # Flatten time × band dimensions → each pixel gets one long feature vector.
    time_steps, h, w, s2_b = s2.shape
    s2_chunk = s2.transpose(1, 2, 0, 3).reshape(-1, time_steps * s2_b)

    # --- Sentinel-1 (merged SAR file) ---
    sar = _load_npy(paths["sar_bands"], "SAR bands")[:, h0:h1, w0:w1]
    # Drop time steps where every pixel is zero (fill / no-data acquisitions).
    valid_time_mask = np.any(sar != 0, axis=(1, 2, 3))
    sar = sar[valid_time_mask]
    sar = (sar - S1_BAND_MEAN) / S1_BAND_STD

    ts_sar, h, w, b_sar = sar.shape
    sar_chunk = sar.transpose(1, 2, 0, 3).reshape(-1, ts_sar * b_sar)

    # Concatenate S2 and SAR feature vectors along the band axis.
    return np.concatenate((s2_chunk, sar_chunk), axis=1)


def load_features_tessera(cfg: dict, h0: int, h1: int, w0: int, w1: int) -> npt.NDArray:
    """
    Load and dequantize Tessera int8 representations for a spatial chunk.

    Parameters
    ----------
    cfg : dict
        Pipeline config with ``paths`` keys for ``tessera_reps`` and ``tessera_scales``.
    h0, h1 : int
        Row slice boundaries of the spatial chunk [h0, h1).
    w0, w1 : int
        Column slice boundaries of the spatial chunk [w0, w1).

    Returns
    -------
    npt.NDArray
        2-D array of shape ``(h*w, 128)`` with dequantized float32 representations.
    """
    paths = cfg["paths"]

    # Tessera stores representations as int8 with a per-pixel scale factor to
    # save disk space.  We must dequantize: float_value = int8_value * scale.
    # Shape: (128, H, W), dtype int8
    rep = _load_npy(paths["tessera_reps"], "tessera representations", mmap_mode="r")
    # scales shape: (H, W) — one scalar per pixel.
    scales = np.squeeze(_load_npy(paths["tessera_scales"], "tessera scales"))

    rep_chunk = rep[:, h0:h1, w0:w1].astype(np.float32)  # (128, h, w)
    scales_chunk = scales[h0:h1, w0:w1][np.newaxis, :, :]  # (1, h, w) for broadcast
    rep_chunk = rep_chunk * scales_chunk  # (128, h, w) dequantized

    # Unused after dequantization — kept for clarity.
    h = h1 - h0
    w = w1 - w0
    # Reshape to (h*w, 128) so every pixel is one row.
    return rep_chunk.transpose(1, 2, 0).reshape(-1, 128)


def load_features_stm(cfg: dict, h0: int, h1: int, w0: int, w1: int) -> npt.NDArray:
    """
    Load and concatenate 6 STM group chunks for a spatial chunk.

    Parameters
    ----------
    cfg : dict
        Pipeline config with ``paths`` keys for ``stm_group0`` … ``stm_group5``.
    h0, h1 : int
        Row slice boundaries of the spatial chunk [h0, h1).
    w0, w1 : int
        Column slice boundaries of the spatial chunk [w0, w1).

    Returns
    -------
    npt.NDArray
        2-D array of shape ``(h*w, total_stm_bands)`` with concatenated STM features.
    """
    paths = cfg["paths"]
    chunks = []

    # STM features are stored in 6 separate files (band groups) to keep
    # individual files at a manageable size.  Load and concat along band axis.
    for i in range(6):
        arr = _load_npy(paths[f"stm_group{i}"], f"STM group {i}", mmap_mode="r")
        chunks.append(arr[h0:h1, w0:w1, :])  # spatial slice only

    # Concatenate along band dimension (axis=2) → (h, w, total_bands).
    bands_all = np.concatenate(chunks, axis=2)
    h, w, b = bands_all.shape
    return bands_all.reshape(-1, b)  # flatten spatial dims → (h*w, bands)


def load_features_alphaearth(
    cfg: dict, h0: int, h1: int, w0: int, w1: int
) -> npt.NDArray:
    """
    Load AlphaEarth EFM features for a spatial chunk.

    Parameters
    ----------
    cfg : dict
        Pipeline config with ``paths`` key for ``alphaearth_efm``.
    h0, h1 : int
        Row slice boundaries of the spatial chunk [h0, h1).
    w0, w1 : int
        Column slice boundaries of the spatial chunk [w0, w1).

    Returns
    -------
    npt.NDArray
        2-D array of shape ``(h*w, n_efm_bands)`` with AlphaEarth EFM features.
    """
    paths = cfg["paths"]
    # EFM array layout: (H, W, n_bands) — already in spatial-last order.
    efm = _load_npy(paths["alphaearth_efm"], "AlphaEarth EFM", mmap_mode="r")
    # Apply spatial crop [:, :3863, :] to align with label raster if needed
    efm_chunk = efm[h0:h1, w0:w1, :]
    h, w, b = efm_chunk.shape
    return efm_chunk.reshape(-1, b)  # (h*w, n_bands)


def load_features_specmat(cfg: dict, h0: int, h1: int, w0: int, w1: int) -> npt.NDArray:
    """
    Load NDVI monthly data for a spatial chunk (spectral matching approach).

    Parameters
    ----------
    cfg : dict
        Pipeline config with ``paths`` key for ``specmat_ndvi``.
    h0, h1 : int
        Row slice boundaries of the spatial chunk [h0, h1).
    w0, w1 : int
        Column slice boundaries of the spatial chunk [w0, w1).

    Returns
    -------
    npt.NDArray
        2-D array of shape ``(h*w, timesteps)`` with monthly NDVI values.
    """
    paths = cfg["paths"]
    # Shape: (timesteps, H, W) — time-first layout on disk.
    ndvi = _load_npy(paths["specmat_ndvi"], "NDVI monthly", mmap_mode="r")
    # Transpose to (h, w, timesteps) so we can reshape to (h*w, timesteps).
    chunk = ndvi[:, h0:h1, w0:w1].transpose(1, 2, 0)  # (h, w, timesteps)
    h, w, b = chunk.shape
    return chunk.reshape(-1, b)


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

# Maps the string value of cfg["approach"] to the corresponding loader function.
# classify.py uses this to select the right loader without any if/elif chains.
FEATURE_LOADERS = {
    "raw": load_features_raw,
    "tessera": load_features_tessera,
    "stm": load_features_stm,
    "alphaearth": load_features_alphaearth,
    "specmat": load_features_specmat,
}


# ---------------------------------------------------------------------------
# Pixel CSV export (merged from export_reps.py + export_reps_alphaearth.py)
# ---------------------------------------------------------------------------


def export_pixel_csv(cfg: dict, approach: str) -> None:
    """Export labelled pixels with their feature vectors to a CSV file."""
    year = cfg["year"]
    paths = cfg["paths"]

    # Load field-ID raster — non-zero pixels belong to labelled field polygons.
    labels = _load_npy(paths["field_ids"], "field ID raster").squeeze()

    if approach == "tessera":
        rep = _load_npy(paths["tessera_reps"], "tessera representations")
        scales = np.squeeze(_load_npy(paths["tessera_scales"], "tessera scales"))
        scales_expanded = scales[np.newaxis, :, :]  # (1, H, W) for broadcasting
        values = rep.astype(np.float32) * scales_expanded  # (128, H, W) dequantized
        n_bands = 128
        out_path = f"{paths['output_root']}/pixelcsvs/{year}_pixels_with_labels.csv"

    elif approach == "alphaearth":
        # Shape on disk: (H, W, 64) — transpose to (64, H, W) for consistent handling
        raw = _load_npy(paths["alphaearth_efm"], "AlphaEarth EFM")
        values = raw.transpose(2, 0, 1)  # → (64, H, W)
        n_bands = values.shape[0]
        out_path = f"{paths['output_root']}/pixelcsvs/{year}_efm_with_labels.csv"

    else:
        raise ValueError(f"export_pixel_csv not supported for approach: '{approach}'")

    # Guard against accidental spatial misalignment between feature and label arrays.
    assert (
        values.shape[1:] == labels.shape
    ), f"Mismatch: values spatial {values.shape[1:]} vs labels {labels.shape}"

    # Flatten both arrays so we can filter rows by label.
    labels_flat = labels.flatten()
    values_flat = values.reshape(n_bands, -1).T  # (num_pixels, n_bands)

    # Keep only labelled (non-background) pixels.
    mask = labels_flat != 0
    labels_sel = labels_flat[mask]
    values_sel = values_flat[mask]

    # Build DataFrame with one column per band and a leading "label" column.
    df = pd.DataFrame(values_sel, columns=[f"value_{i+1}" for i in range(n_bands)])
    df.insert(0, "label", labels_sel)
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows to {out_path}")
