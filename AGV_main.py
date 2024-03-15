import scipy.io as sio
import numpy as np
from MystepFun import AGV_StepFun
import time
from param_class import AGV_param
from util import AGV_norm_state
import random
from RL_brain_class import DeepQNetwork
import tensorflow as tf
import os
from datetime import datetime

# %% Load models
plant_param = AGV_param()

# %% hyperparameters defination

NUM_ACTION = 11 
NUM_OBS = 7
MEMORY_SIZE = np.exp2(13).astype(int)
BATCH_SIZE = np.exp2(11).astype(int)
NUM_EPISODE = 10000
# initial state
INIT_obs = [[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 3, 0, 0, 205], [0, 0, 0, 3, 3, 0, 149], [0, 0, 2, 0, 3, 0, 113]]
INIT_OBS = [0, 0, 0, 0, 0, 0, 0]

MAX_EPI_STEP = 200
RECORD_VAL = 200
STEP = 1
RO_NODES = 4
RO_TRACES = 25
RO_DEPTH = 4
random.seed(5) # 50

# build network
tf.reset_default_graph()
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
                  output_graph = False,
                  max_num_nextS = 26,
                  l1_node = 128,
                  look_ahead_step = STEP)

saver = tf.compat.v1.train.Saver(max_to_keep=None)
cwd = os.getcwd() + '\\' + datetime.today().strftime('%Y-%m-%d') + '\\AGV'
if not os.path.exists(cwd):
    os.makedirs(cwd)

total_step = 0
reward_history = []
good_event_history = []
episode_step_history = [0]
max_epi_reward = -30

# train or not
Train = 0

# %% train
if Train:
    for num_episode in range(NUM_EPISODE):
        
        S = INIT_OBS # INIT_obs[random.randint(0, len(INIT_obs)-1)]
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
                    saver.save(RL.sess, save_path)
                break
            total_step += 1
            epi_action_list.append(selected_action)
    
    # %%draw cost curve
    RL.plot_cost()
    RL.plot_reward(reward_history, 250)
    RL.plot_epiEvent(good_event_history)
    save_path_reward_mat = cwd + '\\' + 'reward_his.mat'
    sio.savemat(save_path_reward_mat, mdict={'reward': reward_history})

else:
    # %% for single checkpoint test
    file_path = "C:\\Users\\kaiget\\OneDrive - KTH\\work\\MB_DQN_Junjun\\\disable_approach\\2024-03-13\\AGV\\9309_reward200.27272727272754step201.ckpt"
    # file_path = "C:\\Users\\kaiget\\OneDrive - KTH\\work\\MB_DQN_Junjun\\AGV_dis\\2023-12-19\\8521_reward253.42000000000033step201.ckpt" # fill in the target ckpt
    # file_path = "C:\\Users\\kaiget\\OneDrive - KTH\\work\\MB_DQN_Junjun\\AGV_dis\\2023-12-20\\9598_reward216.56666666666646step201.ckpt"  # 9990_reward210.81666666666632step201.ckpt
    
    tf.reset_default_graph()    
    S = [0, 0, 0, 0, 0, 0, 0]
    [generated_states_full, Problem_state] = RL.check_action_AGV_rollout(S, file_path, plant_param)
            