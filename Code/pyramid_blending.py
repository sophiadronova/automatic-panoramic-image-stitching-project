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

def blend_multiple_pyramids(pyramids):
    blended = []
    num_levels = len(pyramids[0])

    for level in range(num_levels):
        sum_level = np.zeros_like(pyramids[0][level])
        count = np.zeros_like(pyramids[0][level][:,:,0])

        for p in pyramids:
            img_level = p[level]
            mask = (cv2.cvtColor(np.abs(img_level).astype(np.uint8), cv2.COLOR_BGR2GRAY) > 0)
            mask = mask.astype(np.float32)

            count += mask
            sum_level += img_level * mask[:,:,None]  # Apply mask per channel

        # Avoid division by zero
        count[count == 0] = 1
        blended_level = sum_level / count[:,:,None]
        blended.append(blended_level)
    return blended

def reconstruct_from_pyramid(pyramid):
    img = pyramid[-1]
    for i in range(len(pyramid) - 2, -1, -1):
        img = cv2.pyrUp(img)
        img = cv2.resize(img, pyramid[i].shape[1::-1])
        img += pyramid[i]
    return np.clip(img, 0, 255).astype(np.uint8)

def PyramidBlending(images):
    pyramids = []
    w, h, _ = images[0].shape
    levels = int(np.floor(np.log2(min(w, h))))
    for i, image in enumerate(images):
        laplacian_pyr = build_laplacian_pyramid(image, levels)
        pyramids.append(laplacian_pyr)
    blended_laplacian_pyr = blend_multiple_pyramids(pyramids)
    blended_image = reconstruct_from_pyramid(blended_laplacian_pyr)
    return blended_image