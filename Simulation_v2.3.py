import pybullet as p
import time
import pybullet_data
import numpy as np

# import torch
physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
p.setGravity(0,0,-10)
planeId = p.loadURDF("plane.urdf")
robotStartPos = [0,0,0.25]
robotStartOrientation = p.getQuaternionFromEuler([1.57,0,0])
robotId = p.loadURDF("Bobo_shoes_2.urdf",robotStartPos, robotStartOrientation)
#set the center of mass frame (loadURDF sets base link frame) startPos/Ornp.resetBasePositionAndOrientation(boxId, startPos, startOrientation)

#initializing parameters for motor movement
num_joints = 8
# Defining joint indices
joint_indices = np.arange(8) # Starting at 0
control_mode = p.POSITION_CONTROL # Initialize control of all joints to position control
freq = 0.1
a = 0.17
b = 0.25
amplitudes = [a,b,a,b,a,b,a,b]
freq = 0.05
for i in range (10000):
    target_positions = [0]*8
    for j in range(num_joints):
        if j == 0 or j == 3 or j == 5 or j == 6:
            target_positions[j] = amplitudes[j]*np.sin(i*freq)
        if j == 1 or j == 2 or j == 4 or j == 7:
            target_positions[j] = amplitudes[j]*np.cos(i*freq)
    p.setJointMotorControlArray(robotId, joint_indices, control_mode, target_positions)
    p.stepSimulation()
    time.sleep(1./240.)
cubePos, cubeOrn = p.getBasePositionAndOrientation(robotId)
print(cubePos,cubeOrn)
p.disconnect()