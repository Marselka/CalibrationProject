import os
import re
import json
import open3d
import matplotlib.pyplot as plt
import numpy as np

from utils.common_utils import find_nearest_mutual


def get_matching_pairs(folder_path1, folder_path2,
                       file_filter='\d{5}.*\.(png|jpg|jpeg)$',
                       start=0, period=1):
    file_names1, file_names2 = sorted(filter_files(os.listdir(folder_path1), file_filter), key=str.lower), \
                               sorted(filter_files(os.listdir(folder_path2), file_filter), key=str.lower)

    timestamps1, timestamps2 = np.array([int(fn.split('.')[0]) for fn in file_names1]), \
                               np.array([int(fn.split('.')[0]) for fn in file_names2])

    timestamps_str1, timestamps_str2 = np.array([fn.split('.')[0] for fn in file_names1]), \
                                       np.array([fn.split('.')[0] for fn in file_names2])

    mnn_mask, nn_indices1 = find_nearest_mutual(timestamps1, timestamps2)

    timestamps_str1 = timestamps_str1[mnn_mask]
    timestamps_str2 = timestamps_str2[nn_indices1][mnn_mask]

    return timestamps_str1[start::period], timestamps_str2[start::period]


def get_nearest_files(timestamps_str1, folder_path2,
                      file_filter='\d{5}.*\.(png|jpg|jpeg)$'):
    file_names2 = sorted(filter_files(os.listdir(folder_path2), file_filter), key=str.lower)

    timestamps1 = np.array([int(ts) for ts in timestamps_str1])
    timestamps2 = np.array([int(fn.split('.')[0]) for fn in file_names2])

    timestamps_str2 = np.array([fn.split('.')[0] for fn in file_names2])

    mnn_mask, nn_indices1 = find_nearest_mutual(timestamps1, timestamps2)

    timestamps_str1 = timestamps_str1[mnn_mask]
    timestamps_str2 = timestamps_str2[nn_indices1][mnn_mask]

    return timestamps_str1, timestamps_str2


def get_matching_in_range(timestamp, folder_path1, step=1, k=5, file_filter='\d{5}.*\.(png|jpg|jpeg)$'):
    file_names1 = sorted(filter_files(os.listdir(folder_path1), file_filter), key=str.lower)
    timestamps1 = np.array([int(fn.split('.')[0]) for fn in file_names1])

    idx = np.where(timestamps1 == int(timestamp))[0].item()

    closest_timestamps = []

    for i in range(idx - step * k, idx + step * k, step):
        closest_timestamps.append(timestamps1[i])

    return np.array(closest_timestamps).astype(np.str)


"""
Functions for reading the data
"""


def get_images(folder_path, start=None, period=1, file_name_list=None, is_inverted=False):
    """
    :param folder_path: path to the folder with images
    :param period: the rate of considering images; used to reduce the final number of images
    :param file_name_list: the list of file names to consider; used to provide manually selected images
    """
    return get_data(load_image, folder_path, start, period, file_name_list)


def get_pointclouds(folder_path, start=None, period=1, file_name_list=None):
    """
    :param folder_path: path to the folder with pointclouds
    :param start: pointcloud to start from
    :param period: the rate of considering pointclouds
    :param file_name_list: the list of file names to consider
    """
    return get_data(load_pcd, folder_path, start, period, file_name_list)


def get_depths(folder_path, start=None, period=1, file_name_list=None, is_inverted=False):
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
        for i, fname in enumerate(file_name_list):
            if period is not None and i % period != 0:
                continue

            fpath = os.path.join(folder_path, fname)

            datai = data_loader(fpath)

            if datai is not None:
                data[fname] = datai

    return data


def get_azure_parameters(params_file):
    az_param = load_json(params_file)

    az_cp = az_param["color_camera"]
    az_dp = az_param["depth_camera"]

    az_cep = az_cp["extrinsics"]

    az_cip = az_cp["intrinsics"]["parameters"]["parameters_as_dict"]
    az_dip = az_dp["intrinsics"]["parameters"]["parameters_as_dict"]

    az_img_extrinsics = np.block([[np.array(az_cep['rotation']).reshape(3, 3),
                                   np.array(az_cep['translation_in_meters']).reshape(3, 1)],
                                  [0, 0, 0, 1]])

    az_img_intrinsics = np.array([[az_cip['fx'], 0, az_cip['cx']],
                                  [0, az_cip['fy'], az_cip['cy']],
                                  [0, 0, 1]])
    az_depth_intrinsics = np.array([[az_dip['fx'], 0, az_dip['cx']],
                                    [0, az_dip['fy'], az_dip['cy']],
                                    [0, 0, 1]])
    az_img_dist_coeffs = np.array([az_cip['k1'], az_cip['k2'], az_cip['p1'], az_cip['p2'],
                                   az_cip['k3'], az_cip['k4'], az_cip['k5'], az_cip['k6']])
    az_depth_dist_coeffs = np.array([az_dip['k1'], az_dip['k2'], az_dip['p1'], az_dip['p2'],
                                     az_dip['k3'], az_dip['k4'], az_dip['k5'], az_dip['k6']])

    az_img_size = (az_cp["resolution_width"], az_cp["resolution_height"])
    az_depth_size = (az_dp["resolution_width"], az_dp["resolution_height"])

    return az_img_extrinsics, \
           az_img_intrinsics, az_depth_intrinsics, \
           az_img_dist_coeffs, az_depth_dist_coeffs,\
           az_img_size, az_depth_size


def get_charuco_parameters(params_file):
    char_param = load_json(params_file)

    s_p = char_param["shapes"][0]

    return s_p["squares_x"], s_p["squares_y"], s_p["square_length"], s_p["marker_length"]


"""
Functions for saving the data
"""


def save_intrinsics_calib(calib_name, intrinsics, dist_coeff, undist_intrinsics):
    calib_fpath = 'calib_output/%s_intrinsics' % calib_name

    calib = {'intrinsics': intrinsics,
             'dist_coeff': dist_coeff,
             'undist_intrinsics': undist_intrinsics}

    np.save(calib_fpath, calib)

    print('Saved calibration results as %s.npy' % calib_fpath)


def save_extrinsics_calib(calib_name, T):
    calib_fpath = 'calib_output/%s_extrinsics' % calib_name
    calib = {'T': T}

    np.save(calib_fpath, calib)
    print('Saved calibration results as %s.npy' % calib_fpath)


"""
Support utils
"""


def load_image(file_path):
    if file_path.lower().endswith('jpg') or \
            file_path.lower().endswith('jpeg') or \
            file_path.lower().endswith('png'):

        img = plt.imread(file_path)

        if len(img.shape) == 3 and img.dtype == np.float32 and img.max() <= 1:
            img = (img * 255).astype(np.uint8)

        # Infra-red image
        if len(img.shape) == 2 and img.dtype == np.float32 and img.max() <= 1:
            img = (img / img.max() * 255).astype(np.uint8)

    else:
        return None

    return img


def load_pcd(file_path):
    if file_path.lower().endswith('pcd'):
        pcd = np.asarray(open3d.io.read_point_cloud(file_path).points).astype(np.float32)

    else:
        return None

    return pcd


def load_depth(file_path):
    if file_path.lower().endswith('npy'):
        depth = np.load(file_path)

    elif file_path.lower().endswith('png'):
        depth = (plt.imread(file_path) * 65536).astype(np.int) / 1000
    else:
        return None

    return depth


def load_json(file_path):
    with open(file_path) as f:
        data = json.load(f)

        return data


def filter_files(file_names, file_filter):
    return [fn for fn in file_names if re.match(file_filter, fn)]

