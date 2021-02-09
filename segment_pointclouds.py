import numpy as np

from utils.io_utils import get_pointclouds
from utils.segment_utils import filter_pointclouds, segment_pointclouds

data_path = "/home/konstantin/datasets/bandeja-sequence/2021_02_01_2/2021-02-01-16-17-42/_velodyne_velodyne_points"

pointclouds = get_pointclouds(data_path, period=4)

fpointclouds = filter_pointclouds(pointclouds, (0.0, -1.0, 0.0), (2.5, 1.0, 3.0))
s_pointclouds = segment_pointclouds(fpointclouds)

np.save('calib_output/check.npy', s_pointclouds)