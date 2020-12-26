import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

"""
Functions for reading the data
"""


def get_cam_images(folder_path, period=None, file_name_list=None, return_file_name=False):
    """
    :param folder_path: path to the folder with images
    :param period: the rate of considering images; used to reduce the final number of images
    :param file_name_list: the list of file names to consider; used to provide manually selected images
    """
    file_names = sorted(os.listdir(folder_path), key=lambda s: s.lower())

    images = []

    if return_file_name:
        chosen_file_names = []

    if file_name_list is None:
        for i, fname in enumerate(file_names):
            if period is not None and i % period != 0:
                continue

            fpath = os.path.join(folder_path, fname)

            if fpath.lower().endswith('jpg') or \
                    fpath.lower().endswith('jpeg') or \
                    fpath.lower().endswith('png'):

                img = cv.imread(fpath)
                images.append(img)

                if return_file_name:
                    chosen_file_names.append(fname)

        if return_file_name:
            return images, chosen_file_names

        else:
            return images

    else:
        for fname in file_name_list:
            fpath = os.path.join(folder_path, fname)

            img = cv.imread(fpath)
            images.append(img)

        return images


"""
Functions for writing the data and providing visualizations
"""


def output_calib_results(intrinsics, dist_coeff, shape, images, idx, cam_id=None):
    print("Intrinsics:")
    print(intrinsics)

    print("Distortion coefficients:")
    print(dist_coeff)

    undist_intrinsics, _ = cv.getOptimalNewCameraMatrix(intrinsics, dist_coeff, shape, 1, shape)

    print("Undistored intrinsics: ")
    print(undist_intrinsics)

    undist_img = cv.undistort(images[idx], intrinsics, dist_coeff, None, undist_intrinsics)

    plt.figure(figsize=(9, 9))
    plt.imshow(undist_img)

    if cam_id is not None:
        calib_fpath = 'calib_%s' % cam_id
        calib = {'intrinsics': intrinsics,
                 'dist_coeff': dist_coeff,
                 'undist_intrinsics': undist_intrinsics}

        np.save(calib_fpath, calib)

        print('Saved calibration results as %s.npy' % calib_fpath)


def output_stereo_calib_results(R, T, E, F, cam0_images, cam1_images, results0, results1, pattern_size, idx,
                                stereo_id=None):
    print("R:")
    print(R)

    print("T:")
    print(T)

    print("E:")
    print(E)

    print("F:")
    print(F)

    num_points = pattern_size[0] * pattern_size[1]

    loc_kp0 = results0.get(idx)[1]
    loc_kp1 = results1.get(idx)[1]

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    if loc_kp0 is not None and loc_kp1 is not None:
        loc_kp0 = loc_kp0.reshape(num_points, 1, 2)
        loc_kp1 = loc_kp1.reshape(num_points, 1, 2)

        ep_lines0 = cv.computeCorrespondEpilines(loc_kp1, 2, F).reshape(-1, 3)
        ep_lines1 = cv.computeCorrespondEpilines(loc_kp0, 1, F).reshape(-1, 3)

        ep_img0 = draw_ep_lines(np.copy(cam0_images[idx]), ep_lines0, loc_kp0)
        det_img1 = cv.drawChessboardCorners(np.copy(cam1_images[idx]), pattern_size, loc_kp1, True)

        det_img0 = cv.drawChessboardCorners(np.copy(cam0_images[idx]), pattern_size, loc_kp0, True)
        ep_img1 = draw_ep_lines(np.copy(cam1_images[idx]), ep_lines1, loc_kp1)

        axes[0][0].imshow(ep_img0)
        axes[0][1].imshow(det_img1)

        axes[1][0].imshow(det_img0)
        axes[1][1].imshow(ep_img1)

    else:
        print("The pair doesn't have matching detections detections")

        axes[0][0].imshow(cam0_images[idx])
        axes[0][1].imshow(cam0_images[idx])

        axes[1][0].imshow(cam1_images[idx])
        axes[1][1].imshow(cam1_images[idx])

    if stereo_id is not None:
        calib_fpath = 'calib_%s_extrinsics' % stereo_id
        calib = {'R': R,
                 'T': T,
                 'E': E,
                 'F': F}

        np.save(calib_fpath, calib)

        print('Saved calibration results as %s.npy' % calib_fpath)


"""
Functions for visualization only
"""


def draw_detections(images, results, pattern_size, idx):
    loc_kp = results.get(idx)[1]

    if loc_kp is not None:
        det_img = cv.drawChessboardCorners(np.copy(images[idx]), pattern_size, loc_kp, True)

        plt.figure(figsize=(9, 9))
        plt.imshow(det_img)

    else:
        print("No detections")


def draw_stereo_pair(cam0_images, cam1_images, idx):
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    axes[0].imshow(cam0_images[idx])
    axes[1].imshow(cam1_images[idx])


def draw_stereo_pair_detections(cam0_images, cam1_images, results0, results1, pattern_size, idx):
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    loc_kp0 = results0.get(idx)[1]
    loc_kp1 = results1.get(idx)[1]

    if loc_kp0 is not None and loc_kp1 is not None:
        det_img0 = cv.drawChessboardCorners(np.copy(cam0_images[idx]), pattern_size, loc_kp0, True)
        det_img1 = cv.drawChessboardCorners(np.copy(cam1_images[idx]), pattern_size, loc_kp1, True)

        axes[0].imshow(det_img0)
        axes[1].imshow(det_img1)

    else:
        print("The pair doesn't have matching detections detections")

        axes[0].imshow(cam0_images[idx])
        axes[1].imshow(cam1_images[idx])


def draw_ep_lines(img, ep_line, loc_kp):
    c = img.shape[1]

    for l, lk in zip(ep_line, loc_kp):
        color = tuple(np.random.randint(0, 255, 3).tolist())

        x0, y0 = map(int, [0, -l[2] / l[1]])
        x1, y1 = map(int, [c, -(l[2] + l[0] * c) / l[1]])

        img = cv.line(img, (x0, y0), (x1, y1), color, 1, lineType=cv.LINE_AA)
        img = cv.circle(img, tuple(lk[0]), 5, color, -1)

    return img