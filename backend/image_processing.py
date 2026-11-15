
import cv2
import numpy as np


def validate_image(image):
    """Validate the uploaded image."""
    if image is None or not isinstance(image, np.ndarray):
        raise ValueError("Invalid image.")

    if image.size == 0:
        raise ValueError("Image is empty.")

    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    return image


def histogram_equalization(image):
    """Apply histogram equalization to improve global contrast."""
    image = validate_image(image)

    if len(image.shape) == 2:
        return cv2.equalizeHist(image)

    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    l = cv2.equalizeHist(l)

    enhanced = cv2.merge((l, a, b))
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)


def clahe_enhancement(image, clip_limit=2.0, tile_grid_size=8):
    """Apply Contrast Limited Adaptive Histogram Equalization."""
    image = validate_image(image)

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=(tile_grid_size, tile_grid_size)
    )

    if len(image.shape) == 2:
        return clahe.apply(image)

    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    enhanced_l = clahe.apply(l)

    enhanced = cv2.merge((enhanced_l, a, b))
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)


def contrast_stretching(image, lower_percentile=2, upper_percentile=98):
    """Stretch intensity values using percentile clipping."""
    image = validate_image(image)

    if not (0 <= lower_percentile < upper_percentile <= 100):
        raise ValueError("Invalid percentile range.")

    if len(image.shape) == 2:
        channels = [image]
    else:
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        channels = list(cv2.split(lab))

    l = channels[0].astype(np.float32)

    low = np.percentile(l, lower_percentile)
    high = np.percentile(l, upper_percentile)

    if high <= low:
        return image.copy()

    stretched = (l - low) * (255.0 / (high - low))
    stretched = np.clip(stretched, 0, 255).astype(np.uint8)

    if len(image.shape) == 2:
        return stretched

    enhanced = cv2.merge((stretched, channels[1], channels[2]))
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)


def gamma_correction(image, gamma=1.0):
    """Adjust image brightness using gamma correction."""
    image = validate_image(image)

    if gamma <= 0:
        raise ValueError("Gamma must be greater than zero.")

    lookup_table = np.array([
        ((i / 255.0) ** gamma) * 255
        for i in range(256)
    ]).clip(0, 255).astype(np.uint8)

    return cv2.LUT(image, lookup_table)


def automatic_enhancement(image):
    """Select an enhancement method based on image statistics."""
    image = validate_image(image)

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    mean_intensity = np.mean(gray)
    contrast = np.std(gray)

    if mean_intensity < 80:
        return gamma_correction(image, gamma=0.7), "Gamma Correction"

    elif contrast < 45:
        return clahe_enhancement(image), "CLAHE"

    else:
        return contrast_stretching(image), "Contrast Stretching"


# def enhance_image(image, method="CLAHE", **kwargs):
#     """Main function to select and execute an enhancement algorithm."""

#     methods = {
#         "Histogram Equalization": histogram_equalization,
#         "CLAHE": clahe_enhancement,
#         "Contrast Stretching": contrast_stretching,
#         "Gamma Correction": gamma_correction,
#     }

#     if method == "Automatic":
#         return automatic_enhancement(image)

#     if method not in methods:
#         raise ValueError(f"Unknown enhancement method: {method}")

#     return methods[method](image, **kwargs)
def enhance_image(image, method="CLAHE", **kwargs):
    methods = {
        "Histogram Equalization": histogram_equalization,
        "CLAHE": clahe_enhancement,
        "Contrast Stretching": contrast_stretching,
        "Gamma Correction": gamma_correction,
    }

    if method == "Automatic":
        return automatic_enhancement(image)

    if method not in methods:
        raise ValueError(f"Unknown enhancement method: {method}")

    enhanced = methods[method](image, **kwargs)
    return enhanced, method