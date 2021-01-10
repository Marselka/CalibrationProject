import pcl


def filter_pointclouds(pointclouds, lbound, ubound):
    filtered_pcd = []

    for pcd in pointclouds:
        mask = (pcd[:, 2] < ubound) * (pcd[:, 2] > lbound)
        filtered_pcd.append(pcd[mask, :])

    return filtered_pcd


def segment_pointclouds(pointclouds, ksearch=50, distance_threshold=0.01, normal_distance_weight=0.01, max_iterations=100):
    segmented_pcd = []

    for pcd in pointclouds:
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
            segmented_pcd.append(pcd[indices, :])

    return segmented_pcd
