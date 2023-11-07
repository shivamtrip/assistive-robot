import numpy as np
import open3d as o3d 
import cv2
pcd = o3d.io.read_point_cloud("/home/praveenvnktsh/alfred-autonomy/src/manipulation/scripts/sim/mesh.ply")
points = np.asarray(pcd.points)
x_proj = points[:,0]
y_proj = points[:,1]

x_min = np.min(x_proj)
x_max = np.max(x_proj)
y_min = np.min(y_proj)
y_max = np.max(y_proj)

print(np.mean(x_proj), np.mean(y_proj))

source = np.array([-0.4, 0.6])
target = np.array([0.7, 0.3])
theta = 0
theta = theta * np.pi/180
# target = source + 1.5 * np.array([np.cos(theta), np.sin(theta)])

resolution = 0.05

x_grid = np.arange(x_min, x_max, resolution)
y_grid = np.arange(y_min, y_max, resolution)

img = np.zeros((len(x_grid), len(y_grid)))
print(img.shape)
sorted_points = points[np.argsort(points[:,2])]

def convert_point(x, y):
    x_ind = int((x - x_min)/resolution)
    y_ind = int((y - y_min)/resolution)
    return x_ind, y_ind
    

for point in sorted_points:
    x, y, z = point
    # x_ind = int((x - x_min)/resolution)
    # y_ind = int((y - y_min)/resolution)
    x_ind, y_ind = convert_point(x, y)
    img[x_ind, y_ind] = z

radius_reachable = int(0.85 / resolution)
max_dist_traverse = int(1.3 / resolution)


img -= np.min(img)
img /= np.max(img)
binary = (img <= 0.1) * 255
img *= 255



safety_dist = cv2.distanceTransform(binary.astype(np.uint8), cv2.DIST_L2, 5)



img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2BGR)


def viz(img, name = 'img'):
    timg = img.copy()
    timg = timg.astype(np.float32)
    timg /= timg.max()
    timg *= 255
    timg = timg.astype(np.uint8)
    timg = cv2.resize(timg, (0, 0), fx = 5, fy = 5)
    cv2.imshow(name, timg)
    cv2.waitKey(1)
        

indices = (safety_dist > 0.5) & (safety_dist < 0.6)

reachability_map = np.zeros_like(img[:, :, 0], dtype = np.uint8)
source_reachability_map = np.zeros_like(reachability_map)

cv2.circle(reachability_map, convert_point(target[0], target[1])[::-1], radius_reachable, 255, -1)
cv2.circle(source_reachability_map, convert_point(source[0], source[1])[::-1], max_dist_traverse, 255, -1)

# reachability_map = ((reachability_map & binary) * 255).astype(np.uint8)

target_reachable_dist = cv2.distanceTransform(reachability_map, cv2.DIST_L2, 5)
source_reachabile_dist = cv2.distanceTransform(source_reachability_map, cv2.DIST_L2, 5)

dist = target_reachable_dist * source_reachabile_dist * safety_dist
dist -= np.min(dist)
dist /= np.max(dist)
dist[dist < 0.6] = 0

# pick the point with the highest distance

base_loc = np.unravel_index(np.argmax(dist), dist.shape)
base_loc = np.array(base_loc) * resolution + np.array([x_min, y_min])



img[:, :, 2][dist != 0] = dist[dist!=0] * 255
img = cv2.circle(img, convert_point(source[0], source[1])[::-1], 2, (0, 255, 0), -1)
img = cv2.circle(img, convert_point(target[0], target[1])[::-1], radius_reachable, (255, 0, 0), 1)
img = cv2.circle(img, convert_point(target[0], target[1])[::-1], 2, (255, 0, 0), -1)
img = cv2.circle(img, convert_point(source[0], source[1])[::-1], max_dist_traverse, (0, 255, 0), 1)



viz(dist, 'reachability_dist')
viz(binary, 'dist_btransform')
viz(img, 'img')
viz(safety_dist, 'dist_transform')
cv2.waitKey(0)