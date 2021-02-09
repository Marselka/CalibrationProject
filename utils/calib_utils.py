import cv2 as cv
import numpy as np


def detect_keypoints(images, pattern_size, edge_length=1.0):
    """
    :param edge_length: The length of the edge of a single quad in meters
    """
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    num_points = pattern_size[0] * pattern_size[1]

    # Points in the board's coordinate frame
    scene_points = np.zeros((num_points, 3), np.float32)
    scene_points[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * edge_length

    results = {}

    for key, img in images.items():
        if img.shape[-1] == 3:
            gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        elif img.shape[-1] == 4:
            gray_img = cv.cvtColor(img, cv.COLOR_BGRA2GRAY)

        else:
            raise NotImplementedError

        success, kp = cv.findChessboardCorners(gray_img, pattern_size, None)

        if success:
            loc_kp = cv.cornerSubPix(gray_img, kp, (11, 11), (-1, -1), criteria)

            results[key] = (scene_points, loc_kp)

    return results


def to_lists(results):
    keys = results.keys()

    scene_points = [results[k][0] for k in keys]
    loc_kp = [results[k][1] for k in keys]

    return scene_points, loc_kp


def check_neigh_consistency(results, pattern_size, const_thresh=3.0):
    """
    The function expects a list of images with moderate (or small) latency between adjacent
    items by default. It's needed in order to find movement in the sequence and eliminate corresponding images
    as it will cause problems when calculating [R,t] between stereo cameras.
    :param const_thresh: The maximum displacement between corresponding keypoints on neighbouring images
    """
    num_points = pattern_size[0] * pattern_size[1]

    results = dict(results)

    keys = list(results.keys())

    prev_loc_kp = results[keys[0]][1]

    for k in keys[1:]:
        curr_loc_kp = results[k][1]
        diff_norm = np.linalg.norm((curr_loc_kp - prev_loc_kp).reshape(num_points, 2), axis=-1).mean()

        prev_loc_kp = curr_loc_kp

        if diff_norm > const_thresh:
            del results[k]

    return results


def filter_by_orientation(results, pattern_size):
    orient = {}
    majority_orient = 0

    for key, (_, lkp) in results.items():
        is_left = (lkp[0] - lkp[pattern_size[0]])[0, 0] < 0

        orient[key] = is_left
        majority_orient += float(is_left)

    if majority_orient >= (len(results.keys()) // 2):
        majority_orient = True

    else:
        majority_orient = False

    f_results = {}

    for key, r in results.items():

        if orient[key] == majority_orient:
            f_results[key] = r

    return f_results


def check_stereo_orientation(results0, results1, pattern_size, orient_thresh):
    """
    The function assumes that cam0 and cam1 image pairs are aligned within a certain small latency (e.g. 25 ms).
    """
    keys = results0.keys()

    num_points = pattern_size[0] * pattern_size[1]

    orient_scene_points = []
    orient_loc_kp0 = []
    orient_loc_kp1 = []

    for k in keys:
        r0, r1 = results0.get(k), results1.get(k)

        if r0 is not None and r1 is not None:
            # Check that the orientation of detected keypoints is similar by measuring distance variance
            diff_norm = np.linalg.norm((r0[1] - r1[1]).reshape(num_points, 2), axis=-1)

            if (abs(diff_norm - diff_norm.mean())).mean() <= orient_thresh:
                orient_scene_points.append(r0[0])
                orient_loc_kp0.append(r0[1])
                orient_loc_kp1.append(r1[1])

    return orient_scene_points, orient_loc_kp0, orient_loc_kp1
