import cv2
import numpy as np

def build_laplacian_pyramid(image, levels):
    gaussian_pyr = [image.astype(np.float32)]
    for _ in range(levels):
        image = cv2.pyrDown(image)
        gaussian_pyr.append(image.astype(np.float32))

    laplacian_pyr = []
    for i in range(len(gaussian_pyr) - 1):
        GE = cv2.pyrUp(gaussian_pyr[i + 1], dstsize=(gaussian_pyr[i].shape[1], gaussian_pyr[i].shape[0]))
        L = gaussian_pyr[i] - GE
        laplacian_pyr.append(L)
    laplacian_pyr.append(gaussian_pyr[-1])
    return laplacian_pyr

def generate_weight_map(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = (gray > 0).astype(np.uint8)
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    dist = dist / (np.max(dist) + 1e-6)
    return np.dstack([dist] * 3)

def compute_max_weight_maps(weight_maps):
    stacked = np.stack(weight_maps)  
    gray_weights = np.mean(stacked, axis=3) 
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
        gaussian_pyr = [weight]
        for _ in range(levels):
            weight = cv2.GaussianBlur(weight, (5, 5), 0)
            weight = cv2.pyrDown(weight)
            gaussian_pyr.append(weight)
        blurred_weights.append(gaussian_pyr)
    return blurred_weights

def reconstruct_from_pyramid(pyramid):
    image = pyramid[-1]
    for i in range(len(pyramid) - 2, -1, -1):
        image = cv2.pyrUp(image)
        image = cv2.resize(image, pyramid[i].shape[1::-1])
        image += pyramid[i]
    return np.clip(image, 0, 255).astype(np.uint8)

def create_total_mask(images):
    total_mask = np.zeros(images[0].shape[:2], dtype=np.uint8) 
    for image in images:
        mask = np.any(image > 0, axis=2).astype(np.uint8)  
        total_mask = np.maximum(total_mask, mask)    
    total_mask = np.dstack([total_mask]*3)       
    return total_mask

def generate_valid_masks(images):
    masks = []
    for image in images:
        valid = np.any(image > 0, axis=2).astype(np.float32)
        masks.append(np.dstack([valid]*3)) 
    return masks

def build_gaussian_pyramid(mask, levels):
    gaussian_pyr = [mask.astype(np.float32)]
    for _ in range(levels):
        mask = cv2.pyrDown(mask)
        gaussian_pyr.append(mask)
    return gaussian_pyr

def blend_pyramids_with_weights(images, laplacian_pyramids, blurred_weights, valid_pyramids):
    num_levels = len(laplacian_pyramids[0])
    blended_pyramid = []

    for level in range(num_levels):
        num = np.zeros_like(laplacian_pyramids[0][level])
        denom = np.zeros_like(num[:, :, 0])

        for i in range(len(laplacian_pyramids)):
            L = laplacian_pyramids[i][level]
            W = blurred_weights[i][level]
            M = valid_pyramids[i][level]

            W_clean = W * M  # Cleaned weight
            num += L * W_clean
            denom += W_clean[:, :, 0]

        denom = np.maximum(denom, 1e-6)
        blended_level = num / denom[:, :, None]
        blended_pyramid.append(blended_level)

    return blended_pyramid

def WeightPyramidBlending(images):
    w, h, _ = images[0].shape
    levels = min(int(np.floor(np.log2(min(w, h)))), 6)

    weight_maps = [generate_weight_map(image) for image in images]
    valid_masks = generate_valid_masks(images)
    masked_weights = [w * m for w, m in zip(weight_maps, valid_masks)]

    max_weight_maps = compute_max_weight_maps(masked_weights)
    blurred_weights = blur_pyramid_weights(max_weight_maps, levels)
    laplacian_pyramids = [build_laplacian_pyramid(img, levels) for img in images]
    valid_pyramids = [build_gaussian_pyramid(mask, levels) for mask in valid_masks]

    blended_pyramid = blend_pyramids_with_weights(images, laplacian_pyramids, blurred_weights, valid_pyramids)
    blended_image = reconstruct_from_pyramid(blended_pyramid)
    mask = create_total_mask(images)
    return (blended_image * mask)   
