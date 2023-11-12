import open3d as o3d


pcd = o3d.io.read_point_cloud("/home/praveenvnktsh/alfred-autonomy/src/perception/uatu/outputs/tsdf.ply")
o3d.visualization.draw_geometries([pcd])