"""Time window utilities for PAT model data processing.

This module provides a canonical implementation for slicing time series data
into fixed-length windows, particularly for the PAT model's 7-day requirement.
"""

# removed - breaks FastAPI

import logging
from typing import Any

import numpy as np
import numpy.typing as npt

from clarity.core.constants import MINUTES_PER_WEEK

logger = logging.getLogger(__name__)

# Re-export the constant for convenience
WEEK_MINUTES = MINUTES_PER_WEEK

# Maximum dimensions for array validation
MAX_DIMENSIONS = 2


def slice_to_weeks(
    arr: npt.NDArray[np.floating[Any]] | npt.NDArray[np.integer[Any]],
    minutes_per_week: int = WEEK_MINUTES,
    keep: str = "latest",
) -> list[npt.NDArray[np.floating[Any]] | npt.NDArray[np.integer[Any]]]:
    """Split array into consecutive 7-day windows.

    This is the canonical implementation for preparing data for PAT model inference.
    It ensures consistent behavior across all data paths (actigraphy, step data, etc.).

    Parameters
    ----------
    arr : numpy array
        Input time series data (shape: NxC or N,)
    minutes_per_week : int, optional
        Number of minutes in a week (default: 10,080)
    keep : {"latest", "all"}
        - "latest": return only the most-recent week (common for real-time PAT)
        - "all": return every full week chunk in chronological order

    Returns:
    -------
    list of numpy arrays
        List containing the week chunks. Empty list if input is empty.

    Examples:
    --------
    >>> # Get latest week from 2 weeks of data
    >>> data = np.arange(20160)  # 2 weeks
    >>> chunks = slice_to_weeks(data, keep="latest")
    >>> len(chunks)
    1
    >>> chunks[0].shape
    (10080,)

    >>> # Get all weeks from 3 weeks of data
    >>> data = np.arange(30240)  # 3 weeks
    >>> chunks = slice_to_weeks(data, keep="all")
    >>> len(chunks)
    3
    """
    # Input validation
    max_dimensions = 2
    if arr.ndim > max_dimensions:
        msg = f"Expected 1-D or 2-D array, got {arr.ndim}-D"
        raise ValueError(msg)

    if arr.size == 0:
        logger.warning("Empty array provided to slice_to_weeks")
        return []

    # For 2D arrays, ensure we're slicing along time axis (axis 0)
    n_features = arr.shape[1] if arr.ndim == MAX_DIMENSIONS else None

    # Get the time dimension
    n_samples = arr.shape[0]

    if n_samples < minutes_per_week:
        logger.info(
            "Input has %d samples, less than one week (%d). Returning empty list.",
            n_samples,
            minutes_per_week,
        )
        return []

    # Calculate how many full weeks we have
    n_full_weeks = n_samples // minutes_per_week
    remainder = n_samples % minutes_per_week

    if remainder > 0:
        logger.info(
            "Trimming %d samples from the beginning to get full weeks", remainder
        )

    # Trim from the beginning to ensure we have complete weeks
    # This keeps the most recent data intact
    start_idx = n_samples - (n_full_weeks * minutes_per_week)
    trimmed_arr = arr[start_idx:]

    # Split into week chunks
    if n_full_weeks == 0:
        chunks = []
    # Reshape to (n_weeks, minutes_per_week, ...) then convert to list
    elif n_features is not None:
        reshaped = trimmed_arr.reshape(n_full_weeks, minutes_per_week, n_features)
        chunks = [reshaped[i] for i in range(n_full_weeks)]
    else:
        reshaped = trimmed_arr.reshape(n_full_weeks, minutes_per_week)
        chunks = [reshaped[i] for i in range(n_full_weeks)]

    # Return based on keep parameter
    if keep == "latest" and chunks:
        logger.debug("Returning latest week from %d available weeks", len(chunks))
        return chunks[-1:]
    if keep == "all":
        logger.debug("Returning all %d week chunks", len(chunks))
        return chunks
    if keep not in {"latest", "all"}:
        msg = f"Invalid keep parameter: {keep}. Must be 'latest' or 'all'"
        raise ValueError(msg)
    return chunks


def pad_to_week(
    arr: npt.NDArray[np.floating[Any]] | npt.NDArray[np.integer[Any]],
    minutes_per_week: int = WEEK_MINUTES,
    pad_value: float = 0.0,
    pad_side: str = "left",
) -> npt.NDArray[np.float32]:
    """Pad array to exactly one week length.

    Parameters
    ----------
    arr : numpy array
        Input array (must be <= minutes_per_week in length)
    minutes_per_week : int, optional
        Target length (default: 10,080)
    pad_value : float, optional
        Value to use for padding (default: 0.0)
    pad_side : {"left", "right"}
        Side to pad on (default: "left" for older data).
        "left" padding preserves the most recent data at the end,
        consistent with the latest-week-wins semantics

    Returns:
    -------
    numpy array
        Padded array of shape (minutes_per_week,) or (minutes_per_week, n_features)

    Raises:
    ------
    ValueError
        If input array is longer than minutes_per_week
    """
    if arr.shape[0] > minutes_per_week:
        msg = (
            f"Input array has {arr.shape[0]} samples, "
            f"exceeds target length {minutes_per_week}"
        )
        raise ValueError(msg)

    if arr.shape[0] == minutes_per_week:
        return np.ascontiguousarray(arr, dtype=np.float32)

    # Calculate padding needed
    pad_length = minutes_per_week - arr.shape[0]

    # Handle multi-dimensional arrays
    if arr.ndim == MAX_DIMENSIONS:
        pad_shape: tuple[int, ...] = (pad_length, arr.shape[1])
    else:
        pad_shape = (pad_length,)

    # Create padding array
    padding = np.full(pad_shape, pad_value, dtype=np.float32)

    # Apply padding
    if pad_side == "left":
        result = np.concatenate([padding, arr], axis=0)
    else:
        result = np.concatenate([arr, padding], axis=0)

    typed_result: npt.NDArray[np.float32] = np.ascontiguousarray(
        result, dtype=np.float32
    )
    return typed_result


def prepare_for_pat_inference(
    arr: npt.NDArray[np.floating[Any]] | npt.NDArray[np.integer[Any]],
    target_length: int = WEEK_MINUTES,
) -> npt.NDArray[np.float32]:
    """Prepare array for PAT model inference with exactly target_length samples.

    This is the main entry point for PAT data preparation. It handles:
    - Arrays shorter than target_length: pads with zeros
    - Arrays equal to target_length: returns as-is
    - Arrays longer than target_length: takes the most recent target_length samples

    Parameters
    ----------
    arr : numpy array
        Input time series data
    target_length : int, optional
        Required length for PAT model (default: 10,080)

    Returns:
    -------
    numpy array
        Array of shape (target_length,) ready for PAT inference
    """
    n_samples = arr.shape[0]

    if n_samples == target_length:
        return arr.astype(np.float32)
    if n_samples < target_length:
        # Pad shorter sequences
        logger.debug("Padding array from %d to %d samples", n_samples, target_length)
        return pad_to_week(arr, target_length, pad_value=0.0, pad_side="left")
    # Take most recent samples for longer sequences
    logger.debug(
        "Truncating array from %d to most recent %d samples",
        n_samples,
        target_length,
    )
    # Use slice_to_weeks with "latest" to get the most recent week
    chunks = slice_to_weeks(arr, minutes_per_week=target_length, keep="latest")
    if chunks:
        return chunks[0].astype(np.float32)
    # Fallback: should not happen, but handle edge case
    logger.warning("Unexpected: no chunks returned, using zeros")
    return np.zeros(target_length, dtype=np.float32)
