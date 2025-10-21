import numpy as np

def transition_check(y_pred, i, window=10):
    """
    Returns True if a 0→1 transition occurs at index i or
    within ±`window` frames around it.

    Parameters
    ----------
    y_pred : list or np.ndarray
        Sequence of predicted values (0s and 1s).
    i : int
        Current frame index to check.
    window : int
        Number of frames to look backward and forward (default: 10).

    Returns
    -------
    bool
        True if a 0→1 transition occurs at or near index i.
    """
    y_pred = np.asarray(y_pred)

    # Handle edge cases
    if len(y_pred) < 2 or i <= 0 or i >= len(y_pred):
        return False

    # Define window safely
    start = max(1, i - window)
    end = min(len(y_pred) - 1, i + window)

    # Check for any 0→1 transition within ±window frames
    for j in range(start, end + 1):
        if y_pred[j-1] == 0 and y_pred[j] == 1:
            return True

    return False

def has_one_nearby(arr, i, window=10):
    """
    Returns True if there is a 1 within ±`window` frames of index i.

    Parameters
    ----------
    arr : list or np.ndarray
        Sequence of 0s and 1s.
    i : int
        Current frame index to check.
    window : int
        Number of frames to look backward and forward (default: 10).

    Returns
    -------
    bool
        True if there is at least one 1 in the surrounding window.
    """
    arr = np.asarray(arr)

    # Check index validity
    if i < 0 or i >= len(arr):
        return False

    # Define safe window bounds
    start = max(0, i - window)
    end = min(len(arr), i + window + 1)

    # Check for any 1 in the window
    return np.any(arr[start:end] == 1)