# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 12:52:44 2022

@author: KaigeT
"""
import numpy as np

# %% normalize Train states
def Train_norm_state(S):
    max_state = np.array([16, 16, 16, 16, 16, 9, 9, 9, 4, 2, 2, 4, 1, 1, 1])       # maximal number of train model state
    S_norm_arr = np.array(S)/max_state
    S_norm = S_norm_arr.tolist()
    return S_norm

# %% define the set of initial states for Train case study
def Train_init_state():
    # set_S0 = [15*[0]]
    set_S0 = [15*[0],
              [0, 0, 0, 0, 0, 4, 4, 0, 0, 1, 0, 0, 0, 0, 5]] # ok
    
    return set_S0

# %% check Train next states
def Train_Next(State, action, param):
    X1 = State[0]   
    X2 = State[1]
    X3 = State[2]
    X4 = State[3] 
    X5 = State[4]
    X6 = State[5]
    X7 = State[6]
    X8 = State[7]  
    X9 = State[8]
    X10 = State[9]
    X11 = State[10] 
    X12 = State[11]
    X13 = State[12]
    X14 = State[13] 
    X15 = State[14]
    
    if action <= 16:       # 10 ~ 17 -> 9~16 Python
        X1_ = np.where(param.st1[X1, :, action] == 1)
        X1_ = X1_[0][0]
    else:
        X1_ = X1
    
    if action > 16 and action <= 26:
        X2_ = np.where(param.st2[X2, :, action] == 1)
        X2_ = X2_[0][0]
    else:
        X2_ = X2
       
    if action > 26 and action <= 36:
        X3_ = np.where(param.st3[X3, :, action] == 1)
        X3_ = X3_[0][0]
    else:
       X3_ = X3
        
    if action > 36 and action <= 46:  
        X4_ = np.where(param.st4[X4, :, action] == 1)
        X4_ = X4_[0][0]
    else:
        X4_ = X4
        
    X5_ = np.where(param.rs1[X5, :, action] == 1)   
    X5_ = X5_[0][0]
    
    X6_ = np.where(param.rs2[X6, :, action] == 1)   
    X6_ = X6_[0][0]
    
    X7_ = np.where(param.rs3[X7, :, action] == 1)   
    X7_ = X7_[0][0]
    
    X8_ = np.where(param.rs4[X8, :, action] == 1)   
    X8_ = X8_[0][0]
    
    X9_ = np.where(param.rt1[X9, :, action] == 1)   
    X9_ = X9_[0][0]
    
    X10_ = np.where(param.rt2[X10, :, action] == 1)   
    X10_ = X10_[0][0]
    
    X11_ = np.where(param.rt3[X11, :, action] == 1)   
    X11_ = X11_[0][0]
    
    X12_ = np.where(param.rt4[X12, :, action] == 1)   
    X12_ = X12_[0][0]
    
    X14_ = X14 + 1 if action == 15 else X14
    X13_ = X13 + 1 if action == 41 else X13
    if action in [10, 44]:
        X15_ = X15 + 1  
    elif action in [15, 41]:
        X15_ = X15 - 1 
    else:
        X15_ = X15
    
    State_ = [X1_, X2_, X3_, X4_, X5_, X6_, X7_, X8_, X9_, X10_, X11_, X12_, X13_, X14_, X15_]
    return(State_)
    
# %% check permit states
def Train_Permit(obs, param):    
    Enable_P1 = param.A1[0][obs[0]]
    Enable_P2 = param.A2[0][obs[1]]
    Enable_P3 = param.A3[0][obs[2]]
    Enable_P4 = param.A4[0][obs[3]]
    
    Enable_P = np.union1d(Enable_P1, Enable_P2)
    Enable_P = np.union1d(Enable_P, Enable_P3)
    Enable_P = np.union1d(Enable_P, Enable_P4)    # Plant permits
    
    Enable_P5 = param.A5[0][obs[4]]
    Enable_P6 = param.A6[0][obs[5]]
    Enable_P7 = param.A7[0][obs[6]]
    Enable_P8 = param.A8[0][obs[7]]
    Enable_P9 = param.A9[0][obs[8]]
    Enable_P10 = param.A10[0][obs[9]]
    Enable_P11 = param.A11[0][obs[10]]
    Enable_P12 = param.A12[0][obs[11]]  #modular supervisor permits
    
    Enable_S = np.intersect1d(Enable_P5, Enable_P6)
    Enable_S = np.intersect1d(Enable_S, Enable_P7)
    Enable_S = np.intersect1d(Enable_S, Enable_P8)
    Enable_S = np.intersect1d(Enable_S, Enable_P9)
    Enable_S = np.intersect1d(Enable_S, Enable_P10)
    Enable_S = np.intersect1d(Enable_S, Enable_P11)
    Enable_S = np.intersect1d(Enable_S, Enable_P12)
    
    Enable_P_S = np.intersect1d(Enable_S, Enable_P)
    
    return(Enable_P_S, Enable_P)

# %% check the state of the train case study, reached_state_set records the full set of the reached states
# we only care about elements which characterize the train states, the -2 and -3 element are discarded
class StateManager:
    _instance = None  # Class-level variable to store the instance
    # make sure that only one instance is created, o.w. in MystepFun.py will be initialized repeatedly
    def __new__(cls):
        if not cls._instance:
            cls._instance = super(StateManager, cls).__new__(cls)
            cls._instance.reached_state_set = set()
        return cls._instance
    

    def check_and_add_state(self, obs):
        # if the newly generated state never appears before (only check the sublist, without -2 and -3 element)
        sub_obs = tuple(obs[0:-3] + [obs[-1]])
        if sub_obs not in self.reached_state_set:
            self.reached_state_set.add(sub_obs)
            return 1 # return 1 implies a new state is explored, add it to the reward
        else:
            return 0 # o.w. not change reward
            
            