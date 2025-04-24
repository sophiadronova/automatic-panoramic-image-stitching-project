import cv2
import numpy as np

def build_laplacian_pyramid(image, levels):
    gaussian_pyr = [image.astype(np.float32)]
    for _ in range(levels):
        image = cv2.pyrDown(image)
        gaussian_pyr.append(image.astype(np.float32))

    laplacian_pyr = []
    for i in range(len(gaussian_pyr) - 1):
        GE = cv2.pyrUp(gaussian_pyr[i + 1])
        GE = cv2.resize(GE, gaussian_pyr[i].shape[1::-1])
        L = gaussian_pyr[i] - GE
        laplacian_pyr.append(L)
    laplacian_pyr.append(gaussian_pyr[-1])
    return laplacian_pyr


def generate_weight_map(shape):
    h, w = shape[:2]
    y = np.linspace(0, 1, h)
    x = np.linspace(0, 1, w)
    wx = 1 - np.abs(x - 0.5) * 2
    wy = 1 - np.abs(y - 0.5) * 2
    weight = np.outer(wy, wx)
    weight = np.clip(weight, 0, 1)
    return np.dstack([weight]*3)  # 3 channels for RGB

def compute_max_weight_maps(weight_maps):
    stacked = np.stack(weight_maps)   # Shape: (n_images, H, W, 3)
    gray_weights = np.mean(stacked, axis=3)  # Convert to grayscale weights
    max_indices = np.argmax(gray_weights, axis=0)

    max_weight_maps = []
    for i in range(len(weight_maps)):
        mask = (max_indices == i).astype(np.float32)
        mask = np.dstack([mask]*3)
        max_weight_maps.append(mask)
    return max_weight_maps

def blur_pyramid_weights(max_weight_maps, levels):
    blurred_weights = []
    for weight in max_weight_maps:
        gp = [weight]
        for _ in range(levels):
            weight = cv2.pyrDown(weight)
            gp.append(weight)
        blurred_weights.append(gp)
    return blurred_weights

def blend_pyramids_with_weights(laplacian_pyramids, blurred_weights):
    num_levels = len(laplacian_pyramids[0])
    blended_pyramid = []

    for level in range(num_levels):
        numerator = np.zeros_like(laplacian_pyramids[0][level])
        denominator = np.zeros_like(laplacian_pyramids[0][level][:,:,0])

        for i in range(len(laplacian_pyramids)):
            L = laplacian_pyramids[i][level]
            W = blurred_weights[i][level]

            numerator += L * W
            denominator += W[:,:,0]  # Use one channel for normalization

        denominator = np.maximum(denominator, 1e-6)
        blended_level = numerator / denominator[:,:,None]
        blended_pyramid.append(blended_level)

    return blended_pyramid

def reconstruct_from_pyramid(pyramid):
    img = pyramid[-1]
    for i in range(len(pyramid) - 2, -1, -1):
        img = cv2.pyrUp(img)
        img = cv2.resize(img, pyramid[i].shape[1::-1])
        img += pyramid[i]
    return np.clip(img, 0, 255).astype(np.uint8)

def WeightPyramidBlending(images):
    w, h, _ = images[0].shape
    levels = int(np.floor(np.log2(min(w, h))))

    weight_maps = [generate_weight_map(image.shape) for image in images]
    max_weight_maps = compute_max_weight_maps(weight_maps)
    blurred_weights = blur_pyramid_weights(max_weight_maps, levels)
    laplacian_pyramids = [build_laplacian_pyramid(image, levels) for image in images]
    blended_pyramid = blend_pyramids_with_weights(laplacian_pyramids, blurred_weights)
    blended_image = reconstruct_from_pyramid(blended_pyramid)
    return blended_image    