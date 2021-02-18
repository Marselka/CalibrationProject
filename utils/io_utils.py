import os
import pcl
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

"""
Functions for reading the data
"""


def get_images(folder_path, start=None, period=None, file_name_list=None):
    """
    :param folder_path: path to the folder with images
    :param period: the rate of considering images; used to reduce the final number of images
    :param file_name_list: the list of file names to consider; used to provide manually selected images
    """
    return get_data(load_image, folder_path, start, period, file_name_list)


def get_pointclouds(folder_path, start=None, period=None, file_name_list=None):
    """
    :param folder_path: path to the folder with pointclouds
    :param start: pointcloud to start from
    :param period: the rate of considering pointclouds
    :param file_name_list: the list of file names to consider
    """
    return get_data(load_pcd, folder_path, start, period, file_name_list)


def get_depth(folder_path, start=None, period=None, file_name_list=None):
    """
    :param folder_path: path to the folder with depths
    :param start: depth to start from
    :param period: the rate of considering depths
    :param file_name_list: the list of file names to consider
    """
    return get_data(load_depth, folder_path, start, period, file_name_list)


def get_data(data_loader, folder_path, start, period, file_name_list):
    """
    :param data_loader: function for loading data
    :param folder_path: folder to load data from
    :param start: the position from which to start reading data
    :param period: period to consider each i-th data sample
    :param file_name_list: list of files to load
    """
    file_names = sorted(os.listdir(folder_path), key=lambda s: s.lower())

    data = {}

    if file_name_list is None:
        for i, fname in enumerate(file_names):
            if start is not None and i < start:
                continue

            if period is not None and i % period != 0:
                continue

            fpath = os.path.join(folder_path, fname)

            datai = data_loader(fpath)

            if datai is not None:
                data[fname] = datai

    else:
        for fname in file_name_list:
            fpath = os.path.join(folder_path, fname)

            datai = data_loader(fpath)

            if datai is not None:
                data[fname] = datai

    return data


def pointcloudify_depths(depths, intrinsics, dist_coeff, undistort=True):
    pointclouds = {}

    shape = list(depths.values())[0].shape[::-1]

    # Calculate undistorted intrinsics
    if undistort:
        undist_intrinsics, _ = cv.getOptimalNewCameraMatrix(intrinsics, dist_coeff, shape, 1, shape)
        inv_undist_intrinsics = np.linalg.inv(undist_intrinsics)

    else:
        inv_undist_intrinsics = np.linalg.inv(intrinsics)

    for key, depthi in depths.items():
        # Undistort depth
        if undistort:
            # undist_depthi = cv.undistort(depthi, intrinsics, dist_coeff, None, undist_intrinsics)
            map_x, map_y = cv.initUndistortRectifyMap(intrinsics, dist_coeff, None, undist_intrinsics, shape, cv.CV_32FC1)
            undist_depthi = cv.remap(depthi, map_x, map_y, cv.INTER_NEAREST)

        # Generate x,y grid for H x W image
        grid_x, grid_y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))
        grid = np.concatenate([np.expand_dims(grid_x, -1),
                               np.expand_dims(grid_y, -1)], axis=-1)

        grid = np.concatenate([grid, np.ones((shape[1], shape[0], 1))], axis=-1)

        # To normalized image coordinates
        local_grid = inv_undist_intrinsics @ grid.reshape(-1, 3).transpose()  # 3 x H * W

        # Raise by undistorted depth value from image plane to local camera space
        if undistort:
            local_grid = local_grid.transpose() * np.expand_dims(undist_depthi.reshape(-1), axis=-1)

        else:
            local_grid = local_grid.transpose() * np.expand_dims(depthi.reshape(-1), axis=-1)

        pointclouds[key] = local_grid.astype(np.float32)

    return pointclouds


"""
Functions for writing the data and providing visualizations
"""


def output_calib_results(intrinsics, dist_coeff, shape, images, idx, calib_name=None):
    print("Intrinsics:")
    print(intrinsics)

    print("Distortion coefficients:")
    print(dist_coeff)

    undist_intrinsics, _ = cv.getOptimalNewCameraMatrix(intrinsics, dist_coeff, shape, 1, shape)

    print("Undistored intrinsics: ")
    print(undist_intrinsics)

    undist_img = cv.undistort(images[list(images.keys())[idx]], intrinsics, dist_coeff, None, undist_intrinsics)

    plt.figure(figsize=(9, 9))
    plt.imshow(undist_img)

    if calib_name is not None:
        calib_fpath = 'calib_output/%s_intrinsics' % calib_name
        calib = {'intrinsics': intrinsics,
                 'dist_coeff': dist_coeff,
                 'undist_intrinsics': undist_intrinsics}

        np.save(calib_fpath, calib)

        print('Saved calibration results as %s.npy' % calib_fpath)


def output_stereo_calib_results(R, T, E, F, images1, images2, results1, results2, pattern_size, idx, pair_dict,
                                calib_name=None):
    print("R:")
    print(R)

    print("T:")
    print(T)

    print("E:")
    print(E)

    print("F:")
    print(F)

    num_points = pattern_size[0] * pattern_size[1]

    key1 = list(images1.keys())[idx]
    key2 = pair_dict[key1]

    res1 = results1.get(key1)
    res2 = results2.get(key2)

    if res1 is not None and res2 is not None:
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))

        loc_kp1 = res1[1].reshape(num_points, 1, 2)
        loc_kp2 = res2[1].reshape(num_points, 1, 2)

        ep_lines0 = cv.computeCorrespondEpilines(loc_kp2, 2, F).reshape(-1, 3)
        ep_lines1 = cv.computeCorrespondEpilines(loc_kp1, 1, F).reshape(-1, 3)

        ep_img0 = draw_ep_lines(np.copy(images1[key1]), ep_lines0, loc_kp1)
        det_img1 = cv.drawChessboardCorners(np.copy(images2[key2]), pattern_size, loc_kp2, True)

        det_img0 = cv.drawChessboardCorners(np.copy(images1[key1]), pattern_size, loc_kp1, True)
        ep_img1 = draw_ep_lines(np.copy(images2[key2]), ep_lines1, loc_kp2)

        axes[0][0].imshow(ep_img0)
        axes[0][1].imshow(det_img1)

        axes[1][0].imshow(det_img0)
        axes[1][1].imshow(ep_img1)

    else:
        print("The pair doesn't have matching detections")

    if calib_name is not None:
        calib_fpath = 'calib_output/%s_extrinsics' % calib_name
        calib = {'R': R,
                 'T': T,
                 'E': E,
                 'F': F}

        np.save(calib_fpath, calib)

        print('Saved calibration results as %s.npy' % calib_fpath)


"""
Functions for visualization only
"""


def draw_detections(images, results, pattern_size, idx, normalize=False):
    key = list(results.keys())[idx]
    res = results.get(key)

    if res is not None:
        det_img = np.copy(images[key]).astype(np.float32)

        if normalize:
            det_img = det_img / det_img.reshape(-1, 3).max(axis=0).reshape(1, 1, 3)

        det_img = cv.drawChessboardCorners(det_img, pattern_size, res[1], True)

        plt.figure(figsize=(9, 9))
        plt.imshow(det_img)

    else:
        print("No detections")


def draw_stereo_pair(images1, images2, idx, pair_dict):
    key1 = list(images1.keys())[idx]
    key2 = pair_dict[key1]

    img1 = images1.get(key1)
    img2 = images2.get(key2)

    if img1 is not None and img2 is not None:
        fig, axes = plt.subplots(1, 2, figsize=(18, 7))

        axes[0].imshow(img1)
        axes[1].imshow(img2)

    else:
        print("Stereo pair doesn't exist")


def draw_stereo_pair_detections(images1, images2, results1, results2, pattern_size, idx, pair_dict):
    key1 = list(images1.keys())[idx]
    key2 = pair_dict[key1]

    res1 = results1.get(key1)
    res2 = results2.get(key2)

    if res1 is not None and res2 is not None:
        fig, axes = plt.subplots(1, 2, figsize=(18, 7))

        det_img1 = cv.drawChessboardCorners(np.copy(images1.get(key1)), pattern_size, res1[1], True)
        det_img2 = cv.drawChessboardCorners(np.copy(images2.get(key2)), pattern_size, res2[1], True)

        axes[0].imshow(det_img1)
        axes[1].imshow(det_img2)

    else:
        print("The pair doesn't have matching detections")


"""
Support utils
"""


def load_image(file_path):
    if file_path.lower().endswith('jpg') or \
            file_path.lower().endswith('jpeg') or \
            file_path.lower().endswith('png'):
        img = cv.imread(file_path)

    else:
        return None

    return img


def load_pcd(file_path):
    if file_path.lower().endswith('pcd'):
        pcd = pcl.load(file_path).to_array()

    else:
        return None

    nan_mask = np.isnan(pcd).prod(axis=-1) == 1
    pcd = pcd[~nan_mask, :]

    return pcd


def load_depth(file_path):
    if file_path.lower().endswith('npy'):
        pcd = np.load(file_path)

    else:
        return None

    return pcd


def draw_ep_lines(img, ep_line, loc_kp):
    c = img.shape[1]

    for l, lk in zip(ep_line, loc_kp):
        color = tuple(np.random.randint(0, 255, 3).tolist())

        x0, y0 = map(int, [0, -l[2] / l[1]])
        x1, y1 = map(int, [c, -(l[2] + l[0] * c) / l[1]])

        img = cv.line(img, (x0, y0), (x1, y1), color, 1, lineType=cv.LINE_AA)
        img = cv.circle(img, tuple(lk[0]), 5, color, -1)

    return img
