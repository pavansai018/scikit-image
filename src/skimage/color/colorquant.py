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

def median_cut_quantize(image, n_colors=256, max_pixels=None, random_state=0):
    rgb = _to_float_rgb(image)
    H, W, _ = rgb.shape
    pixels = rgb.reshape(-1, 3)
    if max_pixels is not None and pixels.shape[0] > max_pixels:
        rng = np.random.default_rng(random_state)
        indices = rng.choice(pixels.shape[0], size=max_pixels, replace=False)
        pixels = pixels[indices]
    else:
        pixels = pixels

    boxes = [np.arange(pixels.shape[0], dtype=np.int32)]

    def box_range(idxs):
        pts = pixels[idxs]
        mn = pts.min(axis=0)
        mx = pts.max(axis=0)
        return mx - mn
    while len(boxes) < n_colors:
        # Pick splittable box with largest spread
        best_i = None
        best_spread = -1.0
        best_channel = None

        for i, idxs in enumerate(boxes):
            if idxs.size < 2:
                continue
            spread = box_range(idxs)
            smax = float(spread.max())
            if smax > best_spread:
                best_spread = smax
                best_i = i
                best_channel = int(np.argmax(spread))
        if best_i is None:
            break  # no more splittable boxes

        idxs = boxes.pop(best_i)
        pts = pixels[idxs]

        # Sort by the channel with largest range
        order = np.argsort(pts[:, best_channel], kind="mergesort")
        idxs_sorted = idxs[order]

        mid = idxs_sorted.size // 2
        left = idxs_sorted[:mid]
        right = idxs_sorted[mid:]

        if left.size == 0 or right.size == 0:
            # can't split meaningfully
            boxes.append(idxs)
            break

        boxes.append(left)
        boxes.append(right)

    # Build palette: mean color per box
    palette = np.zeros((len(boxes), 3), dtype=np.float32)
    for i, idxs in enumerate(boxes):
        palette[i] = pixels[idxs].mean(axis=0)
    # Map ALL pixels in the original image to nearest palette color
    # (chunked to avoid huge memory use)
    pal = palette.astype(np.float32, copy=False)
    flat = pixels.astype(np.float32, copy=False)
    out = np.empty(flat.shape[0], dtype=np.int32)

    chunk = 200_000
    for start in range(0, flat.shape[0], chunk):
        end = min(start + chunk, flat.shape[0])
        x = flat[start:end]  # (C,3)
        d2 = (
            (x[:, None, 0] - pal[None, :, 0]) ** 2
            + (x[:, None, 1] - pal[None, :, 1]) ** 2
            + (x[:, None, 2] - pal[None, :, 2]) ** 2
        )
        out[start:end] = np.argmin(d2, axis=1).astype(np.int32)

    X = out.reshape(H, W)
    return X, palette