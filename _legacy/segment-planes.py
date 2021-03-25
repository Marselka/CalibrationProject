import os
import sys

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import numpy as np

# from utils.io_utils import get_depth, pointcloudify_depths
from _legacy.segment_utils import filter_pointclouds, segment_pointclouds
from utils.io_utils import get_pointclouds

pnp_results = np.load('../calib_output/pnp_results.npy', allow_pickle=True).item()
pnp_keys = [i for i in list(pnp_results.keys()) if 'frame' not in i]

azure_pcd_folder_path = "/home/konstantin/datasets/bandeja-sequence/2021-02-23-21-18-22/_azure_points2"
azure_pcd_names = sorted(os.listdir(azure_pcd_folder_path), key=str.lower)[:-2]

pnp_timestamps = np.array([int(i) for i in pnp_keys])
azure_pcd_timestamps = np.array([int(i.split('.')[0]) for i in azure_pcd_names])

diff = np.abs(pnp_timestamps.reshape(-1, 1) - azure_pcd_timestamps.reshape(1, -1))
nn_indices = np.argmin(diff, axis=-1)

sel_azure_pcd_names = [azure_pcd_names[i] for i in nn_indices]

# depths = get_depth(azure_depth_folder_path, file_name_list=sel_azure_depth_names)

# calib_intrinsics = np.load("/home/konstantin/personal/CalibrationProject/calib_output/azure_intrinsics.npy", allow_pickle=True).item()

# intrinsics = calib_intrinsics['intrinsics']
# dist_coeff = calib_intrinsics['dist_coeff']

# pointclouds = pointcloudify_depths(depths, intrinsics, dist_coeff)

azure_pcd = get_pointclouds(azure_pcd_folder_path, file_name_list=sel_azure_pcd_names)

f_pointclouds = filter_pointclouds(azure_pcd, (-1.5, -1.5, 0.0), (1.5, 0.5, 1.8))
s_pointclouds = segment_pointclouds(f_pointclouds)

r_pointclouds = {}

for i, j, in zip(pnp_keys, sel_azure_pcd_names):
    if j in s_pointclouds:
        r_pointclouds[i] = s_pointclouds[j]

    else:
        del pnp_results[i]

np.save('../calib_output/r_pnp_results.npy', pnp_results)
np.save('../calib_output/r_pcd.npy', r_pointclouds)