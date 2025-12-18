import numpy as np

def _to_float_rgb(image):
    """Convert image to float32 RGB in [0, 1], shape (H, W, 3)"""
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError("Input image must be a 3D array with 3 channels (H, W, 3).")
    if np.issubdtype(arr.dtype, np.integer):
        info = np.iinfo(arr.dtype)
        arr = arr.astype(np.float32) / float(info.max)
    else:
        arr = arr.astype(np.float32, copy=False)

    return np.clip(arr, 0.0, 1.0)
