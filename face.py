'''
Notes:
1. All of your implementation should be in this file. This is the ONLY .py file you need to edit & submit. 
2. Please Read the instructions and do not modify the input and output formats of function detect_faces() and cluster_faces().
3. If you want to show an image for debugging, please use show_image() function in helper.py.
4. Please do NOT save any intermediate files in your final submission.
'''


import torch

import face_recognition

from typing import Dict, List
from utils import show_image

'''
Please do NOT add any imports. The allowed libraries are already imported for you.
'''

def detect_faces(img: torch.Tensor) -> List[List[float]]:
    """
    Args:
        img : input image is a torch.Tensor represent an input image of shape H x W x 3.
            H is the height of the image, W is the width of the image. 3 is the [R, G, B] channel (NOT [B, G, R]!).

    Returns:
        detection_results: a python nested list. 
            Each element is the detected bounding boxes of the faces (may be more than one faces in one image).
            The format of detected bounding boxes a python list of float with length of 4. It should be formed as 
            [topleft-x, topleft-y, box-width, box-height] in pixels.
    """
    """
    Torch info: All intermediate data structures should use torch data structures or objects. 
    Numpy and cv2 are not allowed, except for face recognition API where the API returns plain python Lists, convert them to torch.Tensor.
    
    """
    detection_results: List[List[float]] = []

    ##### YOUR IMPLEMENTATION STARTS HERE #####
    img_np = _prepare_image(img)
    boxes = face_recognition.face_locations(img_np)

    for (top, right, bottom, left) in boxes:
        x = float(left)
        y = float(top)
        w = float(right - left)
        h = float(bottom - top)
        detection_results.append([x, y, w, h])

    return detection_results



def cluster_faces(imgs: Dict[str, torch.Tensor], K: int) -> List[List[str]]:
    """
    Args:
        imgs : input images. It is a python dictionary
            The keys of the dictionary are image names (without path).
            Each value of the dictionary is a torch.Tensor represent an input image of shape H x W x 3.
            H is the height of the image, W is the width of the image. 3 is the [R, G, B] channel (NOT [B, G, R]!).
        K: Number of clusters.
    Returns:
        cluster_results: a python list where each elemnts is a python list.
            Each element of the list a still a python list that represents a cluster.
            The elements of cluster list are python strings, which are image filenames (without path).
            Note that, the final filename should be from the input "imgs". Please do not change the filenames.
    """
    """
    Torch info: All intermediate data structures should use torch data structures or objects. 
    Numpy and cv2 are not allowed, except for face recognition API where the API returns plain python Lists, convert them to torch.Tensor.
    
    """
    cluster_results: List[List[str]] = [[] for _ in range(K)] # Please make sure your output follows this data format.
        
    ##### YOUR IMPLEMENTATION STARTS HERE #####
    names = []
    features = []

    for name, img in imgs.items():
        img_np = _prepare_image(img)

        encodings = face_recognition.face_encodings(img_np)

        if len(encodings) == 0:
            continue

        feat = torch.tensor(encodings[0], dtype=torch.float32)
        names.append(name)
        features.append(feat)

    if len(features) == 0:
        return cluster_results

    X = torch.stack(features)
    N = X.shape[0]

    best_labels = None
    best_inertia = float('inf')

    for seed in range(15):
        torch.manual_seed(seed * 7 + 3)
        perm = torch.randperm(N)
        centroids = X[perm[:K]].clone()

        prev_labels = torch.full((N,), -1, dtype=torch.long)

        for _ in range(200):
            dists = torch.cdist(X, centroids)
            labels = dists.argmin(dim=1)
            if torch.equal(labels, prev_labels):
                break
            prev_labels = labels.clone()
            for k in range(K):
                members = X[labels == k]
                if members.shape[0] > 0:
                    centroids[k] = members.mean(dim=0)

        inertia = torch.cdist(X, centroids).gather(1, labels.unsqueeze(1)).squeeze().sum().item()
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.clone()

    for i, name in enumerate(names):
        cluster_results[int(best_labels[i])].append(name)

    return cluster_results


'''
If your implementation requires multiple functions. Please implement all the functions you design under here.
But remember the above 2 functions are the only functions that will be called by task1.py and task2.py.
'''

# TODO: Your functions. (if needed)

def _prepare_image(img: torch.Tensor):
    img = img.permute(1, 2, 0)  # (C, H, W) â†’ (H, W, C)
    img = img[..., [2, 1, 0]]   # swap channels
    return img.contiguous().numpy()