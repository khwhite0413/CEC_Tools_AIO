import cv2
import numpy as np


def dilate_mask(mask, iterations=4):
    """Dilate binary mask to cover edges better."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    return cv2.dilate(mask, kernel, iterations=iterations)


def repeat_mask_for_batch(mask, count):
    """Repeat a single mask for multiple frames."""
    return [mask.copy() for _ in range(count)]


def mask_to_binary(mask):
    """Ensure mask is binary (0 or 255)."""
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return binary


def resize_mask(mask, width, height):
    """Resize mask to target dimensions."""
    return cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
