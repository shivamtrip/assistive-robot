import numpy as np
import open3d as o3d 
import cv2
pcd = o3d.io.read_point_cloud("/home/praveenvnktsh/alfred-autonomy/src/manipulation/scripts/sim/pcd.ply")
points = np.asarray(pcd.points)
x_proj = points[:,0]
y_proj = points[:,1]

x_min = np.min(x_proj)
x_max = np.max(x_proj)
y_min = np.min(y_proj)
y_max = np.max(y_proj)

print(np.mean(x_proj), np.mean(y_proj))

source = np.array([0.0, 0.0])
target = np.array([0.8, -0.1])
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

characteristic_dist = 0.15 / resolution
target_reachable_dist = cv2.distanceTransform(reachability_map, cv2.DIST_L2, 5)
source_reachabile_dist = cv2.distanceTransform(source_reachability_map, cv2.DIST_L2, 5)
print(np.max(target_reachable_dist), np.max(source_reachabile_dist), characteristic_dist)
# target_reachable_dist[target_reachable_dist < characteristic_dist] = 0
# source_reachabile_dist[source_reachabile_dist < characteristic_dist] = 0
safety_dist[safety_dist < characteristic_dist] = 0

dist = target_reachable_dist * source_reachabile_dist * safety_dist
dist -= np.min(dist)
dist /= np.max(dist)
dist[dist < 0.8] = 0

# pick the point with the highest distance

base_loc = np.unravel_index(np.argmax(dist), dist.shape)
base_theta = np.arctan2(-(target[0] - base_loc[0]), (target[1] - base_loc[1]))
# we want base_dir to be perpendicular to the line joining base_loc and target
away_point = base_loc + np.array([np.cos(base_theta), np.sin(base_theta)]) * 20
away_point = away_point.astype(np.int32)



img[:, :, 2][dist != 0] = dist[dist!=0] * 255

img = cv2.circle(img, (base_loc[0], base_loc[1])[::-1], 1, (0, 255, 255), -1)
cv2.line(img, (base_loc[0], base_loc[1])[::-1],(away_point[0], away_point[1])[::-1], (0, 255, 255), 1)

img = cv2.circle(img, convert_point(source[0], source[1])[::-1], 2, (0, 255, 0), -1)
img = cv2.circle(img, convert_point(target[0], target[1])[::-1], radius_reachable, (255, 0, 0), 1)
img = cv2.circle(img, convert_point(target[0], target[1])[::-1], 2, (255, 0, 0), -1)
img = cv2.circle(img, convert_point(source[0], source[1])[::-1], max_dist_traverse, (0, 255, 0), 1)



viz(dist, 'reachability_dist')
viz(source_reachabile_dist, 'source_reachabile_dist')
viz(target_reachable_dist, 'target_reachable_dist')
viz(img, 'img')
viz(safety_dist, 'dist_transform')
cv2.waitKey(0)