import cv2
import glob
import numpy as np
from collections import namedtuple
import networkx as nx
from weight_pyramid_blending import WeightPyramidBlending
from pyramid_blending import PyramidBlending

inputDir = '../Images/'
outputDir = '../MB_Blend/'

Feature = namedtuple('Feature', ['keypoints', 'descriptors'])
HWeight = namedtuple('HWeight', ['H', 'weight'])

def ReadImages(dir):
    image_paths = glob.glob(dir)
    images = [cv2.imread(path) for path in image_paths]
    return images

def Cv2SIFT(images):
    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Store keypoints and descriptors
    features = {}  # filename -> (keypoints, descriptors)
    for index in range(len(images)):
        image = images[index]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect SIFT keypoints and compute descriptors
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        features[index] = Feature(keypoints = keypoints, descriptors = descriptors)
        print(f"[INFO] Processed {index} image - {len(keypoints)} keypoints found")
        print(f"Descriptor shape: {descriptors.shape}")
    return features

def FLANNMatcher(images, features):
    # Set up FLANN matcher
    index_params = dict(algorithm=1, trees=5)       # 1 = KDTree for SIFT
    search_params = dict(checks=50)                 # Number of times the tree(s) are recursively traversed
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    GM_dict = {}
    for i in range(len(images)):
        for j in range(i + 1, len(images)):
            des_i = features[i].descriptors
            des_j = features[j].descriptors

            # KNN matching (k=2)
            matches = flann.knnMatch(des_i, des_j, k=2)

            # Lowe's ratio test
            good_matches = []
            for m, n in matches:
                if m.distance < 0.6 * n.distance:
                    good_matches.append(m)

            if len(good_matches) > 15:
                print(f"Good matches found for pair: ({i}, {j}) = {len(good_matches)}")
                GM_dict[(i, j)] = good_matches
    return GM_dict

def ComputeHomography(GM_dict, features):
    H_dict = {}
    for (i, j), good_matches in GM_dict.items():
        kp_i = features[i].keypoints
        kp_j = features[j].keypoints
        if len(good_matches) > 4:
            src_pts = np.float32([kp_i[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_j[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            H_dict[(i, j)] = HWeight(H = H, weight = mask.sum())
    return H_dict

def CreateGraph(H_dict):
    G = nx.Graph()
    for (i, j), data in H_dict.items():
        G.add_edge(i, j, weight = data.weight, homography = data.H, from_node = i, to_node = j)
    return G

def ComputeCumulativeHomographies(H_dict):
    G = CreateGraph(H_dict)
    MST = nx.minimum_spanning_tree(G, weight = 'weight')
    degrees = dict(MST.degree()) # select node with highest degree (most connections)
    ref_img_idx = max(degrees, key=degrees.get) # reference image index
    print(f"Reference image (highest degree): {ref_img_idx}")
    paths = nx.single_source_shortest_path(MST, ref_img_idx) # get paths from the reference image to every other image

    cumulative_H = {}
    for target, path in paths.items():
        if target == ref_img_idx:
            cumulative_H[target] = np.eye(3) # identity matrix
        path = path[::-1]
        H_total = np.eye(3)
        for i in range(len(path) - 1):
            curr = path[i]
            next = path[i + 1]

            edge_data = MST.get_edge_data(curr, next)
            H = edge_data['homography']

            if edge_data['from_node'] != curr:
                H_use = np.linalg.inv(H)
            else:
                H_use = H
            H_total = H_use @ H_total

        cumulative_H[target] = H_total
    return cumulative_H

def ComputeExtentAndTranslation(cumulative_H, images):
    all_points = []
    for i, H in cumulative_H.items():
        image = images[i]
        h, w, _ = image.shape

        points = np.array([[0, 0], [0, h], [w, 0], [w, h]], dtype='float32').reshape(-1, 1, 2)
        transformed_points = cv2.perspectiveTransform(points, H).reshape(-1, 2)
        all_points.append(transformed_points)

    all_points = np.vstack(all_points)

    min_x, min_y = np.floor(all_points.min(axis=0)).astype(int)
    max_x, max_y = np.ceil(all_points.max(axis=0)).astype(int)

    if min_x < 0 or min_y < 0:
        translation = np.array([
            [1, 0, -min_x],
            [0, 1, -min_y],
            [0, 0,     1]
        ], dtype=np.float32)
    else:
        translation = np.eye(3, dtype=np.float32)

    width = max_x - min_x
    height = max_y - min_y
    
    return (width, height, translation)

def ComputeWarpedImage(w, h, translation, images, cumulative_H):
    warped = []
    for i, H in cumulative_H.items():
        image = images[i]
        H_corrected = translation @ H
        warped_image = cv2.warpPerspective(image, H_corrected, (w, h), borderValue=(0,0,0))
        warped.append(warped_image)
    return warped

def main():
    subDir = 'c'
    images = ReadImages(inputDir + subDir + '/*')
    #Cv2Stitcher(images, outputDir + "panorama.png")
    features = Cv2SIFT(images)

    GM_dict = FLANNMatcher(images, features) # good matches dicitonary for each pair of images
    H_dict = ComputeHomography(GM_dict, features)

    cumulative_H = ComputeCumulativeHomographies(H_dict)
    w, h, translation = ComputeExtentAndTranslation(cumulative_H, images)

    warped = ComputeWarpedImage(w, h, translation, images, cumulative_H)
    blended_image = WeightPyramidBlending(warped)

    cv2.imwrite(outputDir + "warped_" + subDir + '.jpg', blended_image)


if __name__ == "__main__":
    main()
