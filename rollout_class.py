import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from MystepFun import AGV_StepFun, Train_StepFun
from util import AGV_norm_state
from util_Train import Train_norm_state
from RL_class import DeepQNetwork
import sys
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# %% define DQN network
class Rollout():
    def __init__(self, 
                 n_actions,
                 n_features,
                 check_pt_path,
                 plant_param,
                 case_name,
                 look_ahead_step = 3,
                 RO_nodes = 5,
                 RO_traces = 50,
                 RO_depth = 3,
                 RO_gamma = 0.95):
        
        # assign initial values
        self.n_actions = n_actions
        self.n_features = n_features
        self.plant_param = plant_param
        self.case_name = case_name
        # two steps look-ahead for the value forcast
        self.look_ahead_step = look_ahead_step
        self.RO_nodes = RO_nodes
        self.RO_traces = RO_traces
        self.RO_depth = RO_depth
        self.RO_gamma = RO_gamma

        # %% load the checkpoint and build the neural network
        self.DQN_agent = DeepQNetwork(n_actions, n_features)
        # Load the checkpoint
        checkpoint = torch.load(check_pt_path)
        # Load the state_dict into your model
        self.DQN_agent.eval_net.load_state_dict(checkpoint)
        # Set the model to evaluation mode
        self.DQN_agent.eval_net.eval()



    def check_action_AGV_rollout(self, S):
        # %% rollout test initialization
        Problem_state = []
        reach_states = []
        generated_states = []
        # reach_states_full is a 2x list, the first element is the state, the second is the state
        # from last step, which is stored in generated_states
        reach_states_full = []
        generated_states_full = []
        
        reach_states.append(S)
        S_full = [S, S]
        reach_states_full.append(S_full)
        
        while(len(reach_states) != 0):
            # iterate to the next state to test
            S = reach_states[0]
            S_full = reach_states_full[0]
            if S not in generated_states: # if S not tested
                _, S_norm = AGV_norm_state(S)
                if self.look_ahead_step == 2:
                    pattern_value, pattern_length = self.Q_value_eval(S)
                    _, _, pattern_index = self.pattern_index_select(pattern_value, S_norm, pattern_length)
                elif self.look_ahead_step == 3:
                    [pattern_value, pattern_index] = self.rollout_test(S_norm, S) # TODO: add rollout test
                else:
                    if not torch.is_tensor(S_norm):
                        S_norm = torch.tensor(S_norm, dtype=torch.float32)
                    pattern_value = self.DQN_agent.eval_net(S_norm)
                    pattern_index = torch.argmax(pattern_value).item()
                
                if pattern_index == None:
                    isDone_test = 1
                    all_S_ = []
                else:
                    [_, all_S_, _, isDone_test, _, _, _] = AGV_StepFun(S, pattern_index, self.plant_param, self.n_actions)
                if isDone_test == 1:
                    Problem_state.append(S)
                
                for all_s_ in all_S_: # iterate all next generated states  
                    if all_s_ not in reach_states: # if the newly generated state never appears before
                        reach_states.append(all_s_)
                        reach_states_full.append([all_s_, S])
                        
                generated_states.append(S)  #collect the verified traversed states
                generated_states_full.append(S_full)
            # remove S since it is traversed
            reach_states.remove(reach_states[0])
            reach_states_full.remove(reach_states_full[0])
            
            if len(generated_states)%100 == 0:
                print("verified states:", len(generated_states))
                print("to-test states:", len(reach_states))
                print("problem states:", len(Problem_state))
                print("**************************************")
                
        return generated_states_full, Problem_state



    """
    check R_t + gamma*Q(s_t+1, a_t+1)
    """
    def Q_value_eval(self, S):
        pattern_value = np.zeros(self.n_actions)
        pattern_length = np.zeros(self.n_actions)
        for pattern_ind in range(self.n_actions):
            if self.case_name == "AGV":
                [_, all_S_, R_t, isDone_test, _, _, _] = AGV_StepFun(S, pattern_ind, self.plant_param, self.n_actions)
            elif self.case_name == "Train":
                [_, all_S_, R_t, isDone_test, _, _, _, _] = Train_StepFun(S, pattern_ind, self.plant_param, 0)
            else:
                print("Error: case_name unexpected, double check the case name!\n")
                sys.exit("Exiting due to unexpected condition")
            if isDone_test:
                pattern_value[pattern_ind] = -1e5 # assign a big negative number, so that avoid blocking
                pattern_length[pattern_ind] = 0
            else:
                # evaluate the expected Q of S_t+1
                if self.case_name == "AGV":
                    S_norm_vec, _ = AGV_norm_state(all_S_)
                elif self.case_name == "Train":
                    S_norm_vec, _ = Train_norm_state(all_S_)
                if not torch.is_tensor(S_norm_vec):
                    S_norm_vec_tensor = torch.tensor(S_norm_vec, dtype=torch.float32)
                Q_s1 = self.DQN_agent.eval_net(S_norm_vec_tensor)
                max_values_rows, _ = torch.max(Q_s1, dim=1)
                Q_s1 = max_values_rows.tolist()
                Q_s1_exp = sum(Q_s1)/len(Q_s1) # mean but not max, since it has possiblity of stepping into any states
                # the expected value of Q is two step look ahead
                pattern_value[pattern_ind] = R_t + self.RO_gamma*Q_s1_exp
                pattern_length[pattern_ind] = len(Q_s1)
        return pattern_value, pattern_length
    


    def pattern_index_select(self, pattern_value, S_norm, pattern_length = None):
        # select pattern_index based on pattern_value
        max_value = np.max(pattern_value)
        max_value_indices = np.where(pattern_value == max_value)[0]
        # if there are multiple indexs with the same value, use the NN to decide one
        if len(max_value_indices) > 1:
            if not torch.is_tensor(S_norm):
                S_norm = torch.tensor(S_norm, dtype=torch.float32)
            pattern_value_NN = self.DQN_agent.eval_net(S_norm)
            pattern_index_NN = torch.argmax(pattern_value_NN).item()
            if any(max_indices_temp == pattern_index_NN for max_indices_temp in max_value_indices):
                pattern_index = pattern_index_NN
            else:
                if pattern_length is None:
                    pattern_index = random.choice(max_value_indices) # TODO: check if better solution
                else:
                    max_len = np.max(pattern_length[max_value_indices])
                    max_len_indices = np.where(pattern_length[max_value_indices] == max_len)[0]
                    if len(max_len_indices) == 1:
                        pattern_index = max_value_indices[max_len_indices][0] # select the pattern index which gives maximal permissive
                    else:
                        pattern_index = random.choice(max_value_indices) # TODO: check if better solution
        elif len(max_value_indices) == 0:
            print("Error: !\n")
            sys.exit("Exiting due to unexpected condition")
        else:
            pattern_index = max_value_indices[0]
        return max_value, max_value_indices, pattern_index
    


    """
    perform rollout based on given state
    """
    def rollout_test(self, S_norm, S):
        # use Q evaluation network to get guided indices of action selection
        # top N actions to evaluate in rollout
        if not torch.is_tensor(S_norm):
                S_norm = torch.tensor(S_norm, dtype=torch.float32)
        temp_value = self.DQN_agent.eval_net(S_norm)
        sorted_indices = torch.argsort(-temp_value)
        top_N_indices = sorted_indices[0][: self.RO_nodes]
        pattern_value = np.zeros(self.n_actions)
        pattern_count = np.zeros(self.n_actions)
        RO_nodes_num = self.RO_nodes
        # perform rollout, run monte-carlo traces for RO_traces times, depth is RO_depth.
        for i in range(self.RO_traces):
            if len(top_N_indices) > 0:
                try:
                    act_idx_ori = top_N_indices[np.mod(i, RO_nodes_num)].item()
                    act_idx = act_idx_ori
                    pattern_count[act_idx_ori] += 1
                    Q_eval_ro = 0
                    skip_outer_loop = False  # Flag to control the outer loop
                except Exception as e:
                    print("An error occurred in the main function:", e)
            else:
                # when top_N_indices is [], means all top nodes are blocking
                break
            for RO_i in range(self.RO_depth):
                if self.case_name == "AGV":
                    [RO_S_, _, R_t, isDone_test, _, _, _] = AGV_StepFun(S, act_idx, self.plant_param, self.n_actions)
                elif self.case_name == "Train":
                    [RO_S_, _, R_t, isDone_test, _, _, _, _] = Train_StepFun(S, act_idx, self.plant_param, 0)
                # if identified a blocking state, remove it from the candidiate list and assign a negative V value
                if isDone_test: # two ways to handle a MC trace blocking
                    # 1) directly remove the nodes,
                    # 2) assign a negative value to the MC trace
                    rollout_method = 1 # 1 for assign negative val, 2 for cut the path
                    if rollout_method == 1:
                        pattern_value[act_idx_ori] += -100*(self.RO_gamma**RO_i)
                    else:
                        pattern_value[act_idx_ori] = -np.inf
                        top_N_indices = np.delete(top_N_indices, np.where(top_N_indices == act_idx_ori))
                        RO_nodes_num -= 1
                    skip_outer_loop = True  # Set the flag to True
                    break
                Q_eval_ro += (self.RO_gamma**RO_i)*R_t
                if self.case_name == "AGV":
                    _, RO_S_norm_ = AGV_norm_state(RO_S_)
                elif self.case_name == "Train":
                    _, RO_S_norm_ = Train_norm_state(RO_S_)
                if not torch.is_tensor(RO_S_norm_):
                        RO_S_norm_ = torch.tensor(RO_S_norm_, dtype=torch.float32)
                pattern_value_ = self.DQN_agent.eval_net(RO_S_norm_)
                if not RO_i + 1 == self.RO_depth:
                    # act_idx = random.randint(0,10)
                    # rollout for policy improvement, try with using DQN policy but not ramdomized
                    act_idx = torch.argmax(pattern_value_).item()
            if skip_outer_loop:
                continue  # Skip to the next iteration of the outer loop                        
            Q_eval_ro += (self.RO_gamma**self.RO_depth)*(torch.max(pattern_value_).item())
            pattern_value[act_idx_ori] += Q_eval_ro
        pattern_value[top_N_indices] = pattern_value[top_N_indices]/pattern_count[top_N_indices]
        # select pattern_index based on pattern_value
        if len(top_N_indices) > 0:
            if self.case_name == "AGV":
                _, pattern_length = self.Q_value_eval(S)
            elif self.case_name == "Train":
                _, pattern_length = self.Q_value_eval(S)
            _, _, pattern_index = self.pattern_index_select(pattern_value, S_norm, pattern_length)
            # _, _, pattern_index = self.pattern_index_select(pattern_value, S_norm)
        else:
            pattern_index = None
            
        return pattern_value, pattern_index