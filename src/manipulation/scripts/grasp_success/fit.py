import numpy as np
import json
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pickle
import glob
import os
file_list = glob.glob('/home/hello-robot/ws/src/manipulation/scripts/grasp_success/newest_data/*.json')

X = []  # input features
y = []  # target variable (outcome)

# "wrist_extension", 
# "joint_lift", 
# "joint_mobile_base_translation",
# "joint_head_pan",
# "joint_head_tilt",
# "joint_wrist_yaw",
# "joint_gripper_finger_left",
# "joint_arm_l3",
# "joint_arm_l2",
# "joint_arm_l1",
# "joint_arm_l0",
# "gripper_aperture",
# "joint_gripper_finger_right

features = [
    ('joint_lift', 'effort'),
    ('joint_gripper_finger_right', 'effort'),
    ('joint_gripper_finger_left', 'effort'),
    ('joint_gripper_finger_left', 'position'),
    ('joint_gripper_finger_right', 'position'),
    ('gripper_aperture', 'position') 
]
mapping = {
    'effort' : 1,
    'position' : 2,
    'velocity' : 3,
}
print(file_list)
interval = 5
joint_names = [joint[0] for joint in features] # ['joint_lift', 'joint_gripper_finger_right', 'joint_gripper_finger_left','joint_gripper_finger_left','joint_gripper_finger_right', 'gripper_aperture']
feature_names = [mapping[joint[1]] for joint in features] # [1, 1, 1, 2, 2, 2]
for file in file_list:
    with open(file, 'r') as f:
        data = f.readlines()[0]
        X.append([])
        d = json.loads(data) # name, effort, position, velocity
        for k, dd in enumerate(d):
            # if k == 0 or k == len(d) - 1:
            if k % interval == 0:
                for i, name in enumerate(dd[0]):
                    if name in joint_names:
                        feature_idx = feature_names[joint_names.index(name)]
                        X[-1].append(dd[feature_idx][i])
        if 'success' in os.path.basename(file):
            y.append(1)
        else:
            y.append(0)

X = np.array(X)
y = np.array(y)
print(X.shape, y.shape)
print('Positive examples = ', sum(y == 1))
print('Negative examples = ', sum(y == 0))

scaler = StandardScaler()
X = scaler.fit_transform(X)
y = np.array(y)
clf = LogisticRegression(random_state=0, solver = 'newton-cholesky').fit(X, y)

print(clf.score(X, y))
with open('/home/hello-robot/ws/src/manipulation/scripts/grasp_success/model_new_data.pkl','wb') as f:
    pickle.dump({
        'feature_names' : feature_names,
        'joint_names' : joint_names,
        'scaler' : scaler,
        'model' : clf,
        'interval' : interval
    }, f)