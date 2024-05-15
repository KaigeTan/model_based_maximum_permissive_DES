import numpy as np
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# %% define DQN network
class DeepQNetwork():
    def __init__(self,
                 n_actions,
                 n_features,
                 learning_rate = 0.005,
                 reward_decay = 0.95,
                 e_greedy = 0.9,
                 # iterations to update target network
                 replace_target_iteration = 300,
                 # memory size to store last experiences
                 memory_size = 1024,
                 # number of experiences to extract when update
                 batch_size = 128,
                 # initial epsilon value
                 epsilon_init = 0.1,
                 # decide whether change epsilon value with learning
                 epsilon_increment = False,
                 max_num_nextS = 10,
                 l1_node = 128,
                 l2_node = 128):
        
        # assign initial values
        self.n_actions = n_actions
        self.n_features = n_features
        self.learning_rate = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iteration = replace_target_iteration
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.max_num_nextS = max_num_nextS
        self.l1_node = l1_node
        self.l2_node = l2_node

        # build network
        self.eval_net = self._build_network()
        self.target_net = self._build_network()
        self.target_net.load_state_dict(self.eval_net.state_dict())  # initialize target net with eval net's weights
        self.optimizer = optim.RMSprop(self.eval_net.parameters(), lr=self.learning_rate)
        # self.optimizer = optim.Adam(self.eval_net.parameters())
        
        self.epsilon_increment = epsilon_increment
        if epsilon_increment == False:
            self.epsilon = self.epsilon_max
        else:
            self.epsilon = epsilon_init
            
        # total learning step
        self.learn_step_counter = 0
        # initialize training parameter
        self.learning_step_counter = 0
        # initialize memeory matrix
        # memory column: S, A, R, all_S_ -- S:(size: n_features), A & R: scalar
        # all_S_ -- all the next available states in the selected pattern
        # a memory: [S, A, R, all_S_]
        self.memory = np.zeros((self.memory_size, self.n_features*(max_num_nextS+1) + 2))

        self.cost_history = []
        

    # %% build the neural networks
    def _build_network(self):
        n_l1 = self.l1_node
        n_l2 = self.l2_node
        net = nn.Sequential(
            nn.Linear(self.n_features, self.l1_node),
            nn.ReLU(),
            nn.Linear(self.l1_node, self.l2_node),
            nn.ReLU(),
            nn.Linear(self.l2_node, self.n_actions))
        return net

    def forward(self, x):
        return self.eval_net(x)
    
    def replace_target_net(self):
        self.target_net.load_state_dict(self.eval_net.state_dict())


    # %% choose action, return the number index of the action, S -- type list
    def choose_action(self, S):
        # add a deminsion for PyTorch tensor
        S = torch.tensor(S, dtype=torch.float32).unsqueeze(0)
        if(np.random.uniform() > self.epsilon):
            action_index = np.random.randint(0, self.n_actions)
        else:
            # evaluate the current q_values of all actions based on state S
            with torch.no_grad():
                action_value = self.eval_net(S)
                action_index = torch.argmax(action_value).item()
        return action_index
    


    # %% store experience replay
    # TODO: check this function and comment on it
    def store_exp(self, S, A, R, all_S_):
        # at first call, add an attribute to count memory list
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        # all_S_: N X 7 list (N assigned with the number of 5, maximal of 5 different S_)
        all_S_list = [-1]*(self.max_num_nextS*self.n_features)   # initialize all_S_list with all -1
        all_S_1D = sum(all_S_, []) # flatten to 1D
        if len(all_S_1D) < self.max_num_nextS*self.n_features:
            all_S_list[: len(all_S_1D)] = all_S_1D
        else:
            print('Error: exceed max number of available events in a pattern!')
            sys.exit("Exiting due to unexpected condition")
        
        # stack SARS_ in: [S, A, R, S_], column number identical
        new_memory = np.hstack((S,[A, R], all_S_list))      # size: |N_state| + 1 (reward_dim) + 1 (action_dim) + |N_state| X num_state_
        memory_index = self.memory_counter % self.memory_size
        self.memory[memory_index, :] = new_memory
        self.memory_counter += 1



    # %% train the nn with batched S, A, R, S_
    def learn(self):
        # reset q target every replace_target_iteration interations
        if self.learn_step_counter % self.replace_target_iteration == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
            print('replace target network parameters at:',\
                  self.memory_counter, 'step\n')
        
        # sample random minibatch of transitions from memory
        if self.memory_counter > self.batch_size:
            sample_index = np.random.choice(self.memory_size, self.batch_size)
        else:
            # if the number of experiences is less than batch_size,
            # some of the experiences will be picked repetitively
            sample_index = np.random.choice(self.memory_counter, self.batch_size)
        batch_memory = self.memory[sample_index, :]
        
        # Convert batch_memory to PyTorch tensors
        b_s = torch.tensor(batch_memory[:, :self.n_features], dtype=torch.float32)
        b_a = torch.tensor(batch_memory[:, self.n_features], dtype=torch.long).unsqueeze(1)
        b_r = torch.tensor(batch_memory[:, self.n_features + 1], dtype=torch.float32).unsqueeze(1)
        # b_s_ = torch.tensor(batch_memory[:, -self.n_features:], dtype=torch.float32)

        # Get Q-evaluation value of the current state (Q(s)) from eval network
        train_q_eval = self.eval_net(b_s).gather(1, b_a)
        
        # get Q-target value of the next state Q(s', a') from target network
        #################### note: here q_next contains all possible S_ in a pattern ############################
        avail_S_num_ = []
        for i_sample in range(self.batch_size):
            avail_S_num_.append((batch_memory[i_sample].tolist().index(-1) - 2 - self.n_features)/self.n_features)  # calculate the number of avaiable next states in each sample
        train_q_target_ = torch.zeros(self.batch_size, dtype=torch.float32)
        
        for idx_act in range(int(max(avail_S_num_))):
            sample_idx = np.where(np.array(avail_S_num_) > idx_act)[0] # the index number of the S_ 
            vector_S_ = batch_memory[sample_idx, (idx_act+1)*self.n_features+2: (idx_act+2)*self.n_features+2]
            # this is the evaluation value of one S_ for an action in pattern, max(Q(S_, a_))
            vector_S_tensor = torch.tensor(vector_S_, dtype=torch.float32)
            vector_train_q_target_temp = self.target_net(vector_S_tensor).max(1)[0]
            train_q_target_[sample_idx] += vector_train_q_target_temp
        train_q_target_ = torch.divide(train_q_target_, torch.tensor(avail_S_num_, dtype=torch.float32))
        # replace the evaluation value to the target value R + gamma*max(Q(S'))
        q_target = b_r.squeeze(1) + self.gamma*train_q_target_ # already perform np.max before when calculating all Q(S_, A_) value
        
        # Calculate the loss based on batch samples
        loss = F.mse_loss(train_q_eval, q_target.unsqueeze(1))

        # Perform optimization
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.cost_history.append(loss.item())
        
        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

