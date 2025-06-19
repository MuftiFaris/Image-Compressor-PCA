import numpy as np
from sklearn.decomposition import PCA as SKPCA
import cv2

def compress_image_pca(image_array: np.ndarray, percent: int) -> np.ndarray:
    """
    Compress an image using PCA on each channel separately.
    percent: 0-100, percentage of components relative to min(height, width)
    """
    h, w = image_array.shape[:2]
    max_components = min(h, w)
    n_components = max(1, int(percent / 100.0 * max_components))

    compressed_channels = []
    for ch in cv2.split(image_array):
        pca = SKPCA(n_components=n_components)
        transformed = pca.fit_transform(ch)
        reconstructed = pca.inverse_transform(transformed)
        reconstructed = np.clip(reconstructed, 0, 255)
        compressed_channels.append(reconstructed.astype(np.uint8))

    return cv2.merge(compressed_channels)