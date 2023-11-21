import numpy as np
import ikpy.urdf.utils
import matplotlib.pyplot as plt
import networkx as nx
iktuturdf_path = '/home/praveenvnktsh/alfred-autonomy/src/common/alfred_core/config/stretch_ik.urdf'
tree = ikpy.urdf.utils.get_urdf_tree(iktuturdf_path, "base_link")[0]
# display.display_png(tree)
# nx.draw_networkx(tree)
# plt.show()
# print(tree)
import ikpy.chain
chain = ikpy.chain.Chain.from_urdf_file(iktuturdf_path)
print(chain)

def get_current_configuration(tool):
    def bound_range(name, value):
        names = [l.name for l in chain.links]
        index = names.index(name)
        bounds = chain.links[index].bounds
        return min(max(value, bounds[0]), bounds[1])

    if tool == 'tool_stretch_gripper':
        q_base = 0.0
        q_lift = bound_range('joint_lift', robot.lift.status['pos'])
        q_arml = bound_range('joint_arm_l0', robot.arm.status['pos'] / 4.0)
        q_yaw = bound_range('joint_wrist_yaw', robot.end_of_arm.status['wrist_yaw']['pos'])
        return [0.0, q_base, 0.0, q_lift, 0.0, q_arml, q_arml, q_arml, q_arml, q_yaw, 0.0, 0.0]
    elif tool == 'tool_stretch_dex_wrist':
        q_base = 0.0
        q_lift = bound_range('joint_lift', robot.lift.status['pos'])
        q_arml = bound_range('joint_arm_l0', robot.arm.status['pos'] / 4.0)
        q_yaw = bound_range('joint_wrist_yaw', robot.end_of_arm.status['wrist_yaw']['pos'])
        q_pitch = bound_range('joint_wrist_pitch', robot.end_of_arm.status['wrist_pitch']['pos'])
        q_roll = bound_range('joint_wrist_roll', robot.end_of_arm.status['wrist_roll']['pos'])
        return [0.0, q_base, 0.0, q_lift, 0.0, q_arml, q_arml, q_arml, q_arml, q_yaw, 0.0, q_pitch, q_roll, 0.0, 0.0]
    
target_point = [-0.043, -0.441, 0.654]


target_point = [-0.043, -0.441, 0.654]
target_orientation = ikpy.utils.geometry.rpy_matrix(0.0, 0.0, -np.pi/2)
pretarget_orientation = ikpy.utils.geometry.rpy_matrix(0.0, 0.0, 0.0)

q_init = get_current_configuration(tool='tool_stretch_gripper')
q_mid = chain.inverse_kinematics(target_point, pretarget_orientation, orientation_mode='all', initial_position=q_init)
q_soln = chain.inverse_kinematics(target_point, target_orientation, orientation_mode='all', initial_position=q_mid)
with np.printoptions(precision=3, suppress=True):
    print(q_soln)

