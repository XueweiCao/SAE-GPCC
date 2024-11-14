import open3d as o3d
import numpy as np


def get_projection(normal, vector):
    dot_product = np.dot(normal, vector)
    norm = np.linalg.norm(normal)
    project = dot_product / norm
    return project


def get_norm(vector):
    norm = np.linalg.norm(vector)
    return norm


def get_RMSE(ref_pcd, res_pcd, mode):
    ref_points = np.asarray(ref_pcd.points)
    res_points = np.asarray(res_pcd.points)

    ref_tree = o3d.geometry.KDTreeFlann(ref_pcd)
    ref_pcd.estimate_normals(
        search_param = o3d.geometry.KDTreeSearchParamKNN(knn=5)
    )
    MeanSquareError = []
    for point in res_points:
        k = 1
        [k, idx, _] = ref_tree.search_knn_vector_3d(point, k)
        ref_point = ref_points[idx[0]]
        normal = ref_pcd.normals[idx[0]]
        vector = point - ref_point
        if mode == 'c2p':
            distance = get_projection(normal, vector)
        elif mode == 'c2c':
            distance = get_norm(vector)
        else:
            print('Mode input error.\n')
        MeanSquareError.append(distance ** 2)

    RootMeanSquareError = np.sqrt(np.mean(MeanSquareError))
    return RootMeanSquareError


def get_distance(point1, point2):
    d2 = []
    for i in range(len(point1)):
        d2.append((point1[i] - point2[i]) ** 2)
    distance = np.sqrt(np.sum(d2))
    return distance


def get_resolution(pcd):
    points = np.asarray(pcd.points)
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)

    resolution = []
    for point in points:
        k = 2
        [k, idx, _] = pcd_tree.search_knn_vector_3d(point, k)
        near_point = points[idx[1]]
        distance = get_distance(near_point, point)
        resolution.append(distance)

    intrinsic_res = np.max(resolution)
    return intrinsic_res


def get_PSNR(ref_pcd, res_pcd):
    mode = 'c2c'
    c2c_RMSE1 = get_RMSE(ref_pcd, res_pcd, mode)
    c2c_RMSE2 = get_RMSE(res_pcd, ref_pcd, mode)
    c2c_RMSE = np.max([c2c_RMSE1, c2c_RMSE2])

    mode = 'c2p'
    c2p_RMSE1 = get_RMSE(ref_pcd, res_pcd, mode)
    c2p_RMSE2 = get_RMSE(res_pcd, ref_pcd, mode)
    c2p_RMSE = np.max([c2p_RMSE1, c2p_RMSE2])

    resolution = get_resolution(ref_pcd)
    c2c_PSNR = 20 * np.log10(resolution / c2c_RMSE)
    c2p_PSNR = 20 * np.log10(resolution / c2p_RMSE)
    
    return c2c_PSNR, c2p_PSNR


def get_BPP(encoded, npoint):
    tot_size = 0
    for seq in encoded:
        element_size = seq.element_size()
        tot_size += element_size * seq.nelement()

    BitPerPoint = (tot_size * 8) / npoint
    return BitPerPoint
