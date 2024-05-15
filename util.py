# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 12:52:44 2022

@author: KaigeT
"""
import numpy as np
import matplotlib.pyplot as plt
#  solve the conflict
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


# %% normalize AGV states
def AGV_norm_state(S):
    max_state = np.array([3, 7, 3, 5, 3, 1, 255])
    S_norm_arr = np.array(S)/max_state
    S_norm = S_norm_arr.tolist()
    S_norm_arr = np.array(S_norm)
    S_norm_arr = S_norm_arr[np.newaxis, :]
    return S_norm, S_norm_arr

# %% check AGV next states
def AGV_Next(State, action, param):
    # params
    AGV_1 = param.AGV_1
    AGV_2 = param.AGV_2
    AGV_3 = param.AGV_3
    AGV_4 = param.AGV_4
    AGV_5 = param.AGV_5
    SUP_IPSR = param.SUP_IPSR
    SUP_ZWSR = param.SUP_ZWSR
    
    X1 = State[0]   #from 0 
    X2 = State[1]
    X3 = State[2]
    X4 = State[3] 
    X5 = State[4]
    X6 = State[5]
    X7 = State[6]
    
    X1_ = np.where(AGV_1[X1, :, action] == 1)
    if len(X1_[0]) == 0:
        X1_ = X1
    else:
        X1_ = X1_[0][0]
       
    X2_ = np.where(AGV_2[X2, :, action] == 1)
    if len(X2_[0]) == 0:
        X2_ = X2
    else:
        X2_ = X2_[0][0]
       
       
    X3_ = np.where(AGV_3[X3, :, action] == 1)
    
    if len(X3_[0]) == 0:
       X3_ = X3
    else:
       X3_ = X3_[0][0]
        
        
    X4_ = np.where(AGV_4[X4, :, action] == 1)
    if len(X4_[0]) == 0:
        X4_ = X4
    else:
        X4_ = X4_[0][0]
       
       
    X5_ = np.where(AGV_5[X5, :, action] == 1)
    if len(X5_[0]) == 0:
        X5_ = X5
    else:
        X5_ = X5_[0][0]
        
          
    X6_ = np.where(SUP_IPSR[X6, :, action] == 1)   
    X6_ = X6_[0][0]
    
    X7_ = np.where(SUP_ZWSR[X7, :, action] == 1)
    X7_ = X7_[0][0]
    
    State_ = [X1_, X2_, X3_, X4_, X5_, X6_, X7_]
    
    return(State_)
    
# %% check available events
def AGV_Enb(state, DFA):
    Events = [];
    M = np.where(DFA == 1)
    N = M[0]   #current state
    O = np.where(N==state)
    Q = M[2]
    for i in O:
        Events.append(Q[i])
    
    return(Events)

# %% check permit states
def AGV_Permit(obs, param):
        
    AGV_1 = param.AGV_1
    AGV_2 = param.AGV_2
    AGV_3 = param.AGV_3
    AGV_4 = param.AGV_4
    AGV_5 = param.AGV_5
    SUP_IPSR = param.SUP_IPSR
    SUP_ZWSR = param.SUP_ZWSR
    
    Enable_P1 = AGV_Enb(obs[0], AGV_1)   #define Enb function
    Enable_P2 = AGV_Enb(obs[1], AGV_2)
    Enable_P3 = AGV_Enb(obs[2], AGV_3)
    Enable_P4 = AGV_Enb(obs[3], AGV_4)
    Enable_P5 = AGV_Enb(obs[4], AGV_5)
    
    Enable_P = np.union1d(Enable_P1, Enable_P2)
    Enable_P = np.union1d(Enable_P, Enable_P3)
    Enable_P = np.union1d(Enable_P, Enable_P4)
    Enable_P = np.union1d(Enable_P, Enable_P5)
    
    
    Enable_B1SUP = AGV_Enb(obs[5], SUP_IPSR)
    Enable_B2SUP = AGV_Enb(obs[6], SUP_ZWSR)    
    Enable = np.intersect1d(Enable_B1SUP, Enable_B2SUP)
    
    Enable_P_S = np.intersect1d(Enable_P, Enable)
    
    return(Enable_P_S,Enable_P)


# %% plot functions
def plot_cost(cost_history):
    plt.figure()
    plt.plot(np.arange(len(cost_history)), cost_history)
    plt.ylabel('Cost')
    plt.xlabel('Training Steps')
    plt.show()
    plt.close()
    
def plot_reward(reward_his, moving_avg_len):
    plt.figure()
    avg_list = np.convolve(reward_his, np.ones(moving_avg_len), 'valid')/moving_avg_len
    plt.plot(np.arange(len(avg_list)), avg_list)
    plt.ylabel('Episode reward')
    plt.xlabel('Episode number')
    plt.show()
    plt.close()

def plot_epiEvent(good_event_history):
    plt.figure()
    plt.plot(np.arange(len(good_event_history)), good_event_history)
    plt.ylabel('Number of good events')
    plt.xlabel('Episode number')
    plt.show()
    plt.close()