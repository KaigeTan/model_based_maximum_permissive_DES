import numpy as np
from random import choice
from util import AGV_Next, AGV_Permit
from util_Train import Train_Next, Train_Permit, StateManager
import os

# %% step function for AGV case study
def AGV_StepFun(obs, pattern_index, param):
        # Determine the available event set at the current state
        [pattern, Enable_P] = AGV_Permit(obs, param)        
        
        if not len(np.intersect1d(pattern, param.E_c)) == 0:
            if not pattern_index == 10:  # No event is disabled if action==10
                pattern = np.setdiff1d(pattern, param.E_c[pattern_index])
        # iterate to the next state, and calculate the running cost
        isDone = 0
        reward = 0.1
        IfAppearGoodEvent = 0
        stop_ind = 0
        all_S_ = []
        all_Enb_ = []
        # Next reachable states
        if len(pattern) != 0:
            # select action and step
            action = choice(pattern)        # random selection of action, the value of action: from E_c and E_u
            obs_ = AGV_Next(obs, action, param)         # iterate to the next state
            #[Enable_P_S_, N] = Permit(obs_, AGV_1, AGV_2, AGV_3, AGV_4, AGV_5, SUP_IPSR, SUP_ZWSR) # if the available event set for the next observation
            # iterate all available actions in the pattern and calculate the all_S_
            for i_action in pattern:
                S_temp_ = AGV_Next(obs, i_action, param)
                all_S_.append(S_temp_)
                [Enable_next_state, M] = AGV_Permit(S_temp_, param)
                all_Enb_.append(Enable_next_state)
            if any(len(vec) == 0 for vec in all_Enb_): # If any event in the selected pattern leads to a deadlock
                Enable_P_S_ = []                                   # an episode is terminated
                obs_ = obs
                stop_ind = 2
                action = -1
            else:
                Enable_P_S_ = [obs_]
        else:
            Enable_P_S_ = []
            obs_ = obs
            all_S_ = [obs]
            stop_ind = 1
            action = -1
            
        # if no possible actions in the next state/intersection is empty set, set current state as the next state and continue.
        if len(Enable_P_S_) == 0:
             isDone = 1
             reward = -100
        else:
            # only give reward if 32 in the pattern action
            if action == 31:
                IfAppearGoodEvent = 1
            
            # here we calculate the average value of all possible actions
            for i_action in pattern:
                i_reward = 0
                # first priority: 31; second priority: 16; third priority: 19， 27;
                if i_action == 31:
                    i_reward = 10
                elif action in [16]:
                    i_reward = 5
                elif action in [19, 27]:
                    i_reward = 1
                reward += i_reward
            reward /= len(pattern)

        return obs_, all_S_, reward, isDone, IfAppearGoodEvent, stop_ind, action
    
    
# %% step function for Train case study
def Train_StepFun(obs, pattern_index, param):
    state_set = StateManager()
    # Determine the available event set at the current state
    [pattern, Enable_P] = Train_Permit(obs, param)
    # # only select pattern from 0 - 16
    #the control pattern with the selected pattern index
    if not len(np.intersect1d(pattern, param.E_c)) == 0:
        if not pattern_index == 16:
            pattern = np.setdiff1d(pattern, param.E_c[pattern_index])
    
    # remove action that enter train if already > 4 trains in the system
    if 11 in pattern and obs[-1] > 4:
        pattern = np.setdiff1d(pattern, 11)
    if 45 in pattern and obs[-1] > 4:
        pattern = np.setdiff1d(pattern, 45)
    # iterate to the next state, and reward definition
    def reward_cal(x):
        return 2/(1+np.exp(-2*(x+1))) - 0.762
    # Calculate the running cost, if 32 is a possible event, give 50 reward
    isDone = 0
    reward = 0
    stop_ind = 0
    all_S_ = []
    all_Enb_ = []
    
    if len(pattern) != 0:                                          
        action = choice(pattern) - 1        # random selection of action, the value of action: from E_c and E_u
        obs_ = Train_Next(obs, action, param)    # iterate to the next state
        [Enable_P_S_, N] = Train_Permit(obs_, param) # if the available event set for the next observation
        # iterate all available actions in the pattern and calculate the all_S_
        for act_idx in pattern:
            act_idx = act_idx - 1        # pattern follows MATLAB naming, from 1; act_idx calls python array, from 0
            S_temp_ = Train_Next(obs, act_idx, param)
            all_S_.append(S_temp_)
            [Enable_next_state, M] = Train_Permit(S_temp_, param)
            all_Enb_.append(Enable_next_state)
            
        # if [] in all_S_:
        if any(len(vec) == 0 for vec in all_Enb_):
            Enable_P_S_ = []
            stop_ind = 2
            action = -1
    else:
        Enable_P_S_ = []
        all_S_ = [obs]
        obs_ = obs
        stop_ind = 1
        action = -1
    
    # if no possible actions in the next state/intersection is empty set, terminate the episode
    prob_state_set_file = os.getcwd() + '\\data\\Train\\train_prob_state_set.txt'
    prob_state_set = np.loadtxt(prob_state_set_file, dtype=int).tolist()
    
    
    def compare_prob_state(vec1, vec2):
        return vec1[:-3] + [vec1[-1]] == vec2[:-3] + [vec2[-1]] # exclude the last 3nd and last 2nd elements, since they represent the finishing number of trains
    prob_flag = any(compare_prob_state(test_vec1, test_vec2) for test_vec1 in prob_state_set for test_vec2 in all_S_) # check if any all_S_ appear iin prob_state_set
    
    if len(Enable_P_S_) == 0: # or prob_flag == 1: # if selected action leads to a potential S_ causing deadlock, or the iterated next states has prob_state
         isDone = 1
         reward = -30
    else:
        X13 = obs[-3]
        X14 = obs[-2]
        ratio_inf = 1.0
        # here we calculate the average value of all possible actions
        for i_action in pattern:
            i_reward = 0
            
            # 15(16) ---> state14, 41(42) ---> state13
            if i_action == 42:
                i_reward = ratio_inf*20*reward_cal(X14 - X13) # np.exp(0.7*(X14 - X13))*20
            elif i_action == 16:
                i_reward = 20*reward_cal(X13 - X14) # np.exp(0.7*(X13 - X14))*20
            elif i_action == 43:
                i_reward = ratio_inf*10*reward_cal(X14 - X13) # np.exp(0.7*(X14 - X13))*20
            elif i_action == 17:
                i_reward = 10*reward_cal(X13 - X14) # np.exp(0.7*(X13 - X14))*20
            elif i_action == 41:
                i_reward = ratio_inf*5*reward_cal(X14 - X13) # np.exp(0.7*(X14 - X13))*10
            elif i_action == 15:
                i_reward = 5*reward_cal(X13 - X14) # np.exp(0.7*(X13 - X14))*10
            elif i_action == 33:
                i_reward = ratio_inf*2.5*reward_cal(X14 - X13) # np.exp(0.7*(X14 - X13))*5
            elif i_action == 27:
                i_reward = 2.5*reward_cal(X13 - X14) # np.exp(0.7*(X13 - X14))*5
            elif i_action == 31:
                i_reward = ratio_inf*reward_cal(X14 - X13) # np.exp(0.7*(X14 - X13))*2.5
            elif i_action == 25:
                i_reward = reward_cal(X13 - X14) # np.exp(0.7*(X13 - X14))*2.5
            elif i_action == 23:
                i_reward = ratio_inf*0.5*reward_cal(X14 - X13) # np.exp(0.7*(X14 - X13))*1
            elif i_action == 37:
                i_reward = 0.5*reward_cal(X13 - X14) # np.exp(0.7*(X13 - X14))*1
            elif i_action == 21:
                i_reward = ratio_inf*0.25*reward_cal(X14 - X13) # np.exp(0.7*(X14 - X13))*0.5
            elif i_action == 35:
                i_reward = 0.25*reward_cal(X13 - X14) # np.exp(0.7*(X13 - X14))*0.5
            elif i_action == 13:
                i_reward = ratio_inf*0.1*reward_cal(X14 - X13) # np.exp(0.7*(X14 - X13))*0.25
            elif i_action == 47:
                i_reward = 0.1*reward_cal(X13 - X14) # np.exp(0.7*(X13 - X14))*0.25
            elif i_action == 11:
                i_reward = ratio_inf*0.05*reward_cal(X14 - X13) # np.exp(0.7*(X14 - X13))*0.1
            elif i_action == 45:
                i_reward = 0.05*reward_cal(X13 - X14) # np.exp(0.7*(X13 - X14))*0.1
            
            reward += i_reward
        reward /= len(pattern)
    
    # check if obs_ is explored before
    if_new_state = state_set.check_and_add_state(obs_) # if obs_ not appears before, if_new_state == 1, o.w. 0
    # for the exploration desire, add reward to the unexplored state
    reward += if_new_state
    
    IfAppearGoodEvent = 1 if action in [15, 41] else 0
    # if len(all_S_) >= 7:
    #     print('') # seems like 7 is a viable number for this case, no |S_| will exceed 7
    return obs_, all_S_, reward, isDone, IfAppearGoodEvent, stop_ind, action, len(state_set.reached_state_set)
    
    
    
    
    
    