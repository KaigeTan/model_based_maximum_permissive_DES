import scipy.io as sio
import numpy as np
from MystepFun import AGV_StepFun
import time
from param_class import AGV_param
from util import AGV_norm_state, plot_cost, plot_reward, plot_epiEvent
import random
from RL_class import DeepQNetwork
from rollout_class import Rollout
import os
from datetime import datetime
import torch
import matplotlib.pyplot as plt

# %% Load models
plant_param = AGV_param()

# %% hyperparameters definition
NUM_ACTION = 11 
NUM_OBS = 7
MEMORY_SIZE = np.exp2(13).astype(int)
BATCH_SIZE = np.exp2(11).astype(int)
NUM_EPISODE = 12000
# initial state
INIT_obs = [[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 3, 0, 0, 205], [0, 0, 0, 3, 3, 0, 149], [0, 0, 2, 0, 3, 0, 113]]
INIT_OBS = [0, 0, 0, 0, 0, 0, 0]

MAX_EPI_STEP = 200
RECORD_VAL = 200
STEP = 3
RO_NODES = 5
RO_TRACES = RO_NODES
RO_DEPTH = 2
RO_GAMMA = 0.7
random.seed(5) # 50
# train or not
Train = 0

# %% train
if Train:
    cwd = os.getcwd() + '\\' + datetime.today().strftime('%Y-%m-%d') + '\\AGV_torch'
    if not os.path.exists(cwd):
        os.makedirs(cwd)
    
    total_step = 0
    reward_history = []
    good_event_history = []
    episode_step_history = [0]
    max_epi_reward = -30

    # build network
    RL = DeepQNetwork(NUM_ACTION,
                      NUM_OBS,
                      learning_rate = 1e-3,
                      reward_decay = 0.98,
                      e_greedy = 0.95,
                      replace_target_iteration = 100,
                      memory_size = MEMORY_SIZE,
                      batch_size = BATCH_SIZE,
                      epsilon_increment = 5e-5, #Next step: increase the value of epsilon_increment
                      epsilon_init = 0.10, # from epsilon_increment = 1e-5 to 1e-4
                      max_num_nextS = 10,
                      l1_node = 128,
                      l2_node = 128)

    for num_episode in range(NUM_EPISODE):
        S = INIT_obs[random.randint(0, len(INIT_obs)-1)] # INIT_OBS
        init_S = S
        S_norm, _ = AGV_norm_state(S)
        episode_reward = 0
        episode_step = 0
        epi_good_event = 0
        epi_action_list = []
        if num_episode > 6000 and num_episode < 7500:
            RL.learning_rate -= 2e-8
        while True:         
            # initialize the Action
            A = RL.choose_action(S_norm)
            # take action and observe
            [S_, all_S_, R, isDone, IfAppear32, stop_ind, selected_action] = \
                AGV_StepFun(S, A, plant_param, NUM_ACTION)
            S_norm_, _ = AGV_norm_state(S_)
            all_S_norm_, _ = AGV_norm_state(all_S_)
            
            # store transition
            RL.store_exp(S_norm, A, R, all_S_norm_)
            # control the learning starting time and frequency
            if total_step > MEMORY_SIZE and (total_step % 10 == 0):
                RL.learn()
            # update states
            episode_reward += R
            episode_step += 1
            epi_good_event += IfAppear32
            S = S_
            S_norm = S_norm_
            if isDone or episode_step > MAX_EPI_STEP:
                if stop_ind == 1:
                    stop_reason = 'pattern is impossible'
                elif stop_ind == 2:
                    stop_reason = 'bad event in a pattern'
                elif episode_step > MAX_EPI_STEP:
                    stop_reason = 'reach 200 steps'
                else:
                    stop_reason = 'next state is empty'
                if max_epi_reward < episode_reward:
                    max_epi_reward = episode_reward
                print('episode:', num_episode, '\n', 
                      'init state:', init_S, '\n',
                      'episode reward:', episode_reward, '\n',
                      'episode step:', episode_step, '\n',
                      #'good event:', epi_good_event, '\n',
                      'epsilon value:', RL.epsilon, '\n',
                      'action list:', epi_action_list, '\n',
                      'learning rate:',  RL.learning_rate, '\n',
                      #'maximal running step:', np.max(episode_step_history), '\n',
                      'maximal episode reward:', max_epi_reward, '\n',
                      #'total good event:', np.sum(good_event_history), '\n',
                      stop_reason, '\n',
                      '*******************************************')
                reward_history.append(episode_reward)
                good_event_history.append(epi_good_event)
                episode_step_history.append(episode_step)
                
                # save checkpoint model, if a good model is received
                if episode_reward > RECORD_VAL:
                    save_path = cwd +'\\' + str(num_episode) + '_reward' + str(episode_reward) + 'step' + str(episode_step) + '.ckpt'
                    torch.save(RL.eval_net.state_dict(), save_path)
                break
            total_step += 1
            epi_action_list.append(selected_action)
    
    # %%draw cost curve
    # plot_cost(RL.cost_history)
    plot_reward(reward_history, 250)
    plot_epiEvent(good_event_history)

    save_path_reward_mat = cwd + '\\' + 'reward_his.mat'
    save_path_epiEvent_mat = cwd + '\\' + 'event_his.mat'
    
    sio.savemat(save_path_reward_mat, mdict={'reward': reward_history})
    sio.savemat(save_path_epiEvent_mat, mdict={'event': good_event_history})
    
else:
    # %% for single checkpoint test
    # file_path = os.getcwd() + "\\2024-04-22\\AGV_torch\\8660_reward210.27272727272768step201.ckpt"
    # file_path = os.getcwd() + "\\2024-04-29\\AGV_torch\\9617_reward206.27272727272765step201.ckpt"
    file_path = os.getcwd() + "\\2024-05-06\\AGV_torch\\11959_reward207.4393939393944step201.ckpt"

    S = [0, 0, 0, 0, 0, 0, 0]

    Rollout_test = Rollout(NUM_ACTION,
                           NUM_OBS,
                           file_path,
                           plant_param = plant_param,
                           case_name = "AGV",
                           look_ahead_step = STEP,
                           RO_nodes = RO_NODES,
                           RO_traces = RO_TRACES,
                           RO_depth = RO_DEPTH,
                           RO_gamma = RO_GAMMA)
    [generated_states_full, Problem_state] = Rollout_test.check_action_AGV_rollout(S)
    print('Tested states:', len(generated_states_full), ', Problem states:', len(Problem_state))
    print('The problem states are: ', Problem_state)