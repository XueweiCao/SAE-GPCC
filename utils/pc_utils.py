import numpy as np
import open3d as o3d


def load_pcd(pcd_path):
    pcd = o3d.io.read_point_cloud(pcd_path)
    points = np.asarray(pcd.points)
    return pcd, points


def save_pcd(points, pcd_path):
    points = np.asarray(points)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(pcd_path, pcd)
    return pcd
