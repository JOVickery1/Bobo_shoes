"""
Program written by James Vickery with sections taken from PyBullet quickstart guide and Hod Lipson's tutorial

Current state of the program: 
Train a neural network on random data
NN takes in 8 floats (current joint angles) and returns 8 floats (next joint angles)
It gets trained on random data generated as numpy arrays

Then the framework for a reinforcement learning DQN is setup with a replay buffer
The replay buffer gets initialized to be full before the first (and only) episode
Then the 
"""

import pybullet as p
import time
import pybullet_data
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import random
from collections import deque, namedtuple

# Use CUDA GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define a namedtuple to represent an experience
Experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state', 'done'))

# Initializing a ReplayBuffer class for the DQN
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
        init_state = np.zeros(8) #initializing an all zero state to append to buffer as an experience
        init_action = np.zeros(8) # initializing all zero action to append to buffer as an experience
        init_reward = np.zeros(1) # initializing a zero reward  (I wonder if I'll need to change dimensions on this stuff)
        init_next_state = np.zeros(8) # initializing the next state to be 0's
        init_done = np.zeros(1) # initializing the "dones" not sure what this does yet
        # Filling the buffer with repeats of the above
        init_experience = Experience(init_state, init_action, init_reward, init_next_state, init_done)
        for i in range(capacity):
            self.memory.append(init_experience)

    def push(self, experience):
        """Add a new experience to memory."""
        self.memory.append(experience)

    def sample(self, batch_size):
        """Randomly sample a batch of experiences from memory."""
        batch = random.sample(self.memory, batch_size)
        
        states = torch.tensor(np.array([exp.state for exp in batch]), dtype=torch.float32, device=device)
        actions = torch.tensor(np.array([exp.action for exp in batch]), dtype=torch.int64, device=device) # This being an int64 may be an issue!
        rewards = torch.tensor(np.array([exp.reward for exp in batch]), dtype=torch.float32, device=device)
        next_states = torch.tensor(np.array([exp.next_state for exp in batch]), dtype=torch.float32, device=device)
        dones = torch.tensor(np.array([exp.done for exp in batch]), dtype=torch.uint8, device=device)
        
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)

class MyDataset(Dataset):
    def __init__(self, features, labels = None):
        self.labels = labels
        if not labels == None:
            self.X = features
            self.Y = labels
        else:
            self.X = features
        

    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        if not self.labels == None:
            return self.X[idx], self.Y[idx]
        else:
            return self.X[idx]
    
class Net(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 256)
        self.fc5 = nn.Linear(256, 8)

    def forward(self, x):
        x = F.relu(self.fc1(x)) 
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x

def train(model, train_loader, optimizer, criterion):
    model.train()
    # running_loss = 0.0
    for i, ndata in enumerate(train_loader, 0):
      inputs, labels = ndata

      # zero the parameter gradients
      optimizer.zero_grad()

      # forward + backward + optimize
      outputs = model(inputs)
      # For the DQN will need to change loss to be the loss for reinforcement learning somehow
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()

    """ # Add the current loss to the running loss
    #   running_loss += loss.item()

    # Average over the Epoch
    # running_loss = running_loss/len(train_loader)
    # return running_loss"""  

# modify for complexity as needed
def predict(model, inputs):
    outputs = model(inputs)
    # Maybe make it so that it outputs in degrees or something? 
    return outputs  

'''----------- Generating Synthetic Data------------'''
# Set the size of the synthetic data
n = 1000  # For example, you can set it to any desired value

# Generate randomized synthetic data
synthetic_data = np.random.uniform(-100, 100, size=(n, 8))
synthetic_labels = np.random.uniform(-100, 100, size=(n, 8))

# Making it a tensor
syn_data_tensor = torch.tensor(synthetic_data, dtype=torch.float32, device=device)
syn_labels_tensor = torch.tensor(synthetic_labels, dtype=torch.float32, device=device)

# Putting it into a Dataset
init_data_set = MyDataset(syn_data_tensor, syn_labels_tensor)

'''------------ Instantiating and defining some hyper parameters -----------'''
# Defining a batch_size
batch_size = 100

# Train loader
train_loader = torch.utils.data.DataLoader(init_data_set, batch_size=batch_size, shuffle = True)
    
bobo_model = Net(8)
bobo_model = bobo_model.cuda()

'''------------ Training the model -------------'''
epochs = 100
for epoch in range(epochs):
    train(bobo_model, train_loader=train_loader, criterion = nn.MSELoss(), optimizer = optim.Adam(bobo_model.parameters(), lr=0.001))
    if epoch % 50 == 0: print("Training!")

# Speeds up compute time from when I was just passing the network straight through without attempting to train
# bobo_model.eval()

'''------------ DQN/RL stuff --------------'''
# Initializing the target model (target Q-function)
bobo_model_target = Net(8)
# Making the target model a copy of the original model
bobo_model_target.load_state_dict(bobo_model.state_dict())

# Defining the capacity of the replay buffer (replay memory)
buffer_size = 100

# Defining the size of each mini_batch - 10 is arbitrary start point
mini_batch_size = 10

'''------------------START PYBULLET CODE------------------'''

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
# defining parameters for old manual way of walking
freq = 0.1
a = 0.17
b = 0.25
amplitudes = [a,b,a,b,a,b,a,b]
freq = 0.05

'''------------ Beginning of Episode Stuff -----------'''

# Eventually should add a surrounding for loop with 

# Setting replayBuffer to have a max capacity of buffer_size experiences
replayBuffer = ReplayBuffer(buffer_size)

# Setting up simulation duration
duration = 10000

for i in range (duration):
    # target_positions = [0]*8
    
    # Setting up epsilon-greedy algorithm
    epsilon = 0.5-i/(duration-1) # Starts at 0.5 and gradually decreases to -0.5 (halfway through for loop, guaranteed to always do what model outputs)
    """The above should likely decrease to zero with episodes rather than with steps
    but I haven't implemented episodes yet"""

    # Generate random number between 0 and 1 (half-open interval)
    random_num = np.random.random_sample() # Will be used for comparison with epsilon to determine random action or not

    # get the states of the joints THIS IS NOT THE STATE VARIABLE
    joint_states = p.getJointStates(robotId,joint_indices)

    """# Pull out only the joint angle values (joint positions in documentation)
    # input_states = [entry[0] for entry in joint_states] #THIS is the individual image state variable thing (x_t from slides)"""
    
    # Pull out only the joint angle values (joint positions in documentation)
    joint_angles = np.array([entry[0] for entry in joint_states]) #THIS is the individual image state variable thing (x_t from slides)
    # This is just so I don't have to change the downstream code
    input_states = joint_angles
    
    # Check input_states if they seem correct
    if i <= 50: print(f"-----{input_states}-----")

    # initialize tensor with input_states and send to compute device
    input_tensor = torch.tensor(input_states, dtype=torch.float32, device=device)
    # Commented out below line because it creates an issue with converting to numpy with batching above
    # joint_angles = torch.tensor(joint_angles, dtype=torch.float32, device=device)
    # print(input_tensor.shape)
    
    # This is where we get our action. May need to discretize the action space by converting to degrees and then back to rad?
    if random_num >= epsilon:
        # Get target positions from a forward pass of this model 
        target_positions = bobo_model(input_tensor)
    else:
        target_positions = torch.tensor(np.random.uniform(low=-1.57/2, high=1.57/2, size=(8)), device=device)  

    # This just checks to see what the values are
    if i <= 50: print(f'TARGET POS: {target_positions}')

    # Execute action from forward pass of model and step the simulation
    p.setJointMotorControlArray(robotId, joint_indices, control_mode, target_positions) # Make sure target_positions is in rad
    p.stepSimulation()

    base_state = p.getLinkState(robotId, 0, computeLinkVelocity = True)

    # Extract out the base CoM location - World location
    base_position = np.array(base_state[0])

    # Extract out the base CoM orientation (quaternion) - World orientation
    base_orientation = np.array(base_state[1])

    # Extract out base linear velocity - SEE DOCUMENTATION this says it's worldLink vs linkWorld
    base_vel = np.array(base_state[6]) # I don't get the implications of link vs world vs URDF frames

    # Observe the reward here:
    '''Figure out how to calculate the reward, probably use if statements based on the center of mass'''
    reward = np.array([0]) # Temporary value until I come up with good reward
    # reward = torch.tensor(reward, dtype=torch.float32, device=device) # might need to add a dimension with .view()
    # Getting the new joint state again NOT THE STATE VARIABLE
    new_joint_states = p.getJointStates(robotId,joint_indices)

    # Extracting only the joint angle values
    new_joint_angles = np.array([entry[0] for entry in new_joint_states])
    # Dangerous make sure data types line up -- ALSO ensure that dimensions all line up
    # new_joint_angles = torch.tensor(new_joint_angles, dtype=torch.float32, device=device)

    # done thing? I guess use done when episode is over
    # Maybe make done true if it tips over and then reset the simulation?
    done = np.array([0]) # something???? 
    # done = torch.tensor(done, dtype=torch.uint8, device=device) # make sure the .shape() of this works well here, might need to add a dimension

    # converst target_positions (random act or output of bobo_model) to a numpy array
    # Also converts from radians to degrees for angles, THIS IS SO THAT WHEN IT CONVERTS TO AN INT IT DOESN'T JUST ROUND TO 0
    current_action = target_positions.cpu().detach().numpy()
    current_action = current_action*(180/(np.pi)) # Need to deal with the fact that it goes to degrees in the replay

    if i==101: print(f'CURRENT ACTION: {current_action}')

    # Collect what we have in an instance of the Experience class (namedTuple)
    # latest_experience = Experience(joint_angles, target_positions, reward, new_joint_angles, done)
    # This is like the above line, but uses .numpy() version of target_positions defined above
    latest_experience = Experience(joint_angles, current_action, reward, new_joint_angles, done)

    if i == 55: print(latest_experience)
    
    # push to replayBuffer (replay memory in slides) standard insert removes first entry to make room
    replayBuffer.push(latest_experience)

    if i== 100: print(f'Replay Buffer: {replayBuffer.sample(mini_batch_size)}')

    # Create for loop to do loss accumulation
    # minibatch = replayBuffer.sample(mini_batch_size)
    # for each Experience in minibatch from Memory - I think QA = bobo_model and QT = bobo_model_target
    #   Calculate loss using QA(state,action) - [reward + gamma* max(QT(next_state, action))]
    #   Add to running gradients
    # Perform gradient descent step on bobo_model

    # Sleep for gui simulation viewing purposes only, delete if actually training
    time.sleep(1./240.)
cubePos, cubeOrn = p.getBasePositionAndOrientation(robotId)
print(cubePos,cubeOrn)
p.disconnect()