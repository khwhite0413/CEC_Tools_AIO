import cv2
import numpy as np


def detect_watermark(video_path, roi_mask, num_samples=30, sensitivity=50):
    """
    Detect semi-transparent watermark within ROI.

    Works by finding pixels that have a consistent brightness "offset"
    compared to their local neighborhood across multiple frames.
    Watermarks add a constant overlay, so they shift pixel values
    consistently regardless of the underlying content.

    Args:
        video_path: path to video
        roi_mask: rough mask (H, W) uint8, 255 = region of interest
        num_samples: frames to sample
        sensitivity: 0-100, higher = detect more (looser)

    Returns:
        precise_mask: (H, W) uint8
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    if total_frames < 3:
        cap.release()
        return roi_mask

    roi_binary = roi_mask > 127
    if not np.any(roi_binary):
        cap.release()
        return roi_mask

    # Sample frames evenly
    indices = np.linspace(0, total_frames - 1, min(num_samples, total_frames), dtype=int)
    samples = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            samples.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32))
    cap.release()

    if len(samples) < 3:
        return roi_mask

    stack = np.stack(samples, axis=0)  # (N, H, W)

    # --- Key insight: watermark creates a consistent local contrast pattern ---
    # For each frame, compute pixel value minus local mean (removes background)
    # The watermark residual should be consistent across frames
    residuals = []
    for frame in stack:
        local_mean = cv2.blur(frame, (31, 31))
        residual = frame - local_mean
        residuals.append(residual)

    residual_stack = np.stack(residuals, axis=0)  # (N, H, W)

    # Watermark pixels: consistent residual (low std of residuals)
    residual_std = np.std(residual_stack, axis=0)
    # Background pixels: varying residual (high std)

    # Also check: watermark has a non-zero mean residual (it shifts brightness)
    residual_mean = np.abs(np.mean(residual_stack, axis=0))

    # Score: high mean residual + low residual std = watermark
    # Normalize both within ROI
    roi_vals_std = residual_std[roi_binary]
    roi_vals_mean = residual_mean[roi_binary]

    if len(roi_vals_std) == 0:
        return roi_mask

    # Stability score (low std = more stable = good)
    std_min, std_max = roi_vals_std.min(), roi_vals_std.max()
    if std_max > std_min:
        stability = 1.0 - (residual_std - std_min) / (std_max - std_min)
    else:
        stability = np.ones_like(residual_std)

    # Prominence score (high mean residual = visible watermark)
    mean_max = roi_vals_mean.max()
    if mean_max > 0:
        prominence = residual_mean / mean_max
    else:
        prominence = np.zeros_like(residual_mean)

    # Combined score
    combined = stability * 0.4 + prominence * 0.6
    combined[~roi_binary] = 0

    # Threshold based on sensitivity (0-100)
    # sensitivity 50 = median threshold, 100 = include almost everything
    roi_scores = combined[roi_binary]
    threshold = np.percentile(roi_scores, 100 - sensitivity)
    precise_mask = np.zeros((h, w), dtype=np.uint8)
    precise_mask[(combined > threshold) & roi_binary] = 255

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    precise_mask = cv2.morphologyEx(precise_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    precise_mask = cv2.morphologyEx(precise_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Dilate slightly to cover edges
    precise_mask = cv2.dilate(precise_mask, kernel, iterations=2)

    return precise_mask
