import open3d as o3d
import numpy as np
import cv2
import json 

volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length= 0.01,
            sdf_trunc=0.04,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

for i in range(0, 23):
    colorimg = cv2.imread(f'images/rgb/{str(i).zfill(6)}.png')
    depthimg = cv2.imread(f'images/depth/{str(i).zfill(6)}.tif', cv2.IMREAD_UNCHANGED)
    
    with open(f'images/detections/{str(i).zfill(6)}.json', 'r') as f:
        detections = json.load(f)
        boxes = np.array(detections['boxes'])
        classes = np.array(detections['classes'])
        bottles = classes == 39
        boxes = boxes[bottles]
        if len(boxes) == 0:
            depthimg = np.zeros_like(depthimg)
        for box in boxes:
            depthmask = np.zeros_like(depthimg)
            depthmask[int(box[1]):int(box[3]), int(box[0]):int(box[2])] = 1
            depthimg = depthimg * depthmask
            print(depthimg.shape)
    with open(f'images/pose/{str(i).zfill(6)}.json', 'r') as f:
        data = json.load(f)

    extrinsics = data['transform']
    color = o3d.geometry.Image(colorimg.astype(np.uint8))
    depthimg = depthimg.astype(np.uint16)
    depthimg[depthimg < 150] = 0
    depthimg[depthimg > 1200] = 0
    depth = o3d.geometry.Image(depthimg)

    
    extrinsics = np.concatenate((extrinsics, np.array([[0, 0, 0, 1]])), axis=0)
    intrinsics = data['intrinsics']
    
    weight = 1
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color, depth, depth_scale=1000, convert_rgb_to_intensity=False)
    
    cam = o3d.camera.PinholeCameraIntrinsic()
    cam.intrinsic_matrix = intrinsics
    cam.width = colorimg.shape[1]
    cam.height = colorimg.shape[0]
    volume.integrate(
        rgbd,
        cam,
        extrinsics
    )
mesh = volume.extract_point_cloud()
# mesh = volume.extract_triangle_mesh()
# o3d.io.write_point_cloud("/home/praveenvnktsh/alfred-autonomy/src/manipulation/scripts/sim/pcd.ply", mesh)
o3d.visualization.draw_geometries([mesh])