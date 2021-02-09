import pcl


def filter_pointclouds(pointclouds, lbound, ubound):
    fpointclouds = {}

    lx, ly, lz = lbound
    ux, uy, uz = ubound

    for key, pcd in pointclouds.items():
        mask = (pcd[:, 0] < ux) * (pcd[:, 0] > lx) * \
               (pcd[:, 1] < uy) * (pcd[:, 1] > ly) * \
               (pcd[:, 2] < uz) * (pcd[:, 2] > lz)

        fpointclouds[key] = pcd[mask, :]

    return fpointclouds


def segment_pointclouds(pointclouds, ksearch=50, distance_threshold=0.01, normal_distance_weight=0.01, max_iterations=100):
    s_pointclouds = {}

    for key, pcd in pointclouds.items():
        pcl_pcd = pcl.PointCloud(pcd)

        seg = pcl_pcd.make_segmenter_normals(ksearch)
        seg.set_optimize_coefficients(True)
        seg.set_model_type(pcl.SACMODEL_NORMAL_PLANE)
        seg.set_method_type(pcl.SAC_RANSAC)
        seg.set_distance_threshold(distance_threshold)
        seg.set_normal_distance_weight(normal_distance_weight)
        seg.set_max_iterations(max_iterations)
        indices, coefficients = seg.segment()

        if len(indices) != 0:
            s_pointclouds[key] = pcd[indices, :]

    return s_pointclouds

