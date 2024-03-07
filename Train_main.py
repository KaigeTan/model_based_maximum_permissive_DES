import scipy.io as sio
import numpy as np
from MystepFun import Train_StepFun
from param_class import Train_param
from util_Train import Train_norm_state, Train_init_state
import random
from RL_brain_class import DeepQNetwork
import tensorflow as tf
import os
from datetime import datetime

# %% Load models
plant_param = Train_param()

# hyperparameters defination
NUM_ACTION = 17        
NUM_OBS = 15
MEMORY_SIZE = np.exp2(13).astype(int)
BATCH_SIZE = np.exp2(11).astype(int)
NUM_EPISODE = 120000
# INIT_OBS = 14*[0] # 0~13, 14 states, last 2 states are binary
INIT_obs = Train_init_state()
MAX_EPI_STEP = 1000
RETRAIN = 0
RECORD_VAL = 1000
STEP = 3
RO_NODES = 4
RO_TRACES = 100
RO_DEPTH = 4
random.seed(50)

# build network
tf.compat.v1.reset_default_graph()
RL = DeepQNetwork(NUM_ACTION, 
                  NUM_OBS,
                  learning_rate = 1e-3,
                  reward_decay = 0.98,
                  e_greedy = 0.95,
                  replace_target_iteration = 100,
                  memory_size = MEMORY_SIZE,
                  batch_size = BATCH_SIZE,
                  epsilon_increment = 0.8e-5,
                  epsilon_init = 0.10,
                  output_graph = False,
                  max_num_nextS = 7,
                  l1_node = 256,
                  look_ahead_step = STEP,
                  RO_nodes = RO_NODES,
                  RO_traces = RO_TRACES,
                  RO_depth = RO_DEPTH)
saver = tf.compat.v1.train.Saver(max_to_keep=None)

total_step = 1
reward_history = []
good_event_history = []
episode_step_history = [0]
max_reward_his = -50


# %% model training
IfTrain = 0
if IfTrain:
    if RETRAIN:
        # restore the ckpt file, the pretrained model is the initial point
        file_path = "C:\\Users\\kaiget\\OneDrive - KTH\\work\\MB_DQN_Junjun\\disable_approach\\2023-12-27\\Train\\79964_reward739.3585824436218step201.ckpt" # fill in the target ckpt
        meta_path = file_path + '.meta'
        tf.reset_default_graph()
        saver = tf.train.import_meta_graph(meta_path)
        saver.restore(RL.sess, file_path)
        
    # ckpt save dir.
    cwd = os.getcwd() + '\\' + datetime.today().strftime('%Y-%m-%d') + '\\Train'
    if not os.path.exists(cwd):
        os.makedirs(cwd)
    
    for num_episode in range(NUM_EPISODE):
        # in each episode, reset initial state and total reward
        ind_temp = random.randint(0, len(INIT_obs)-1)
        S = INIT_obs[ind_temp]
        init_S = S
        S_norm, _ = Train_norm_state(S)
        episode_reward = 0
        episode_step = 0
        epi_good_event = 0
        epi_action_list = []
        # if num_episode > 200000:
        #     RL.learning_rate -= 1.25e-8
        while True:
            # initialize the Action
            A = RL.choose_action_Train(S, plant_param, 1)
            # take action and observe
            [S_, all_S_, R, isDone, IfAppearGoodEvent, stop_ind, selected_action, reached_state_len] = \
                Train_StepFun(S, A, plant_param, IfTrain)
            # normalize the actual next state
            S_norm_, _ = Train_norm_state(S_)
            # normalize the all next states set
            all_S_norm_, _ = Train_norm_state(all_S_)
            
            # store transition
            RL.store_exp(S_norm, A, R, all_S_norm_)
            # control the learning starting time and frequency
            if total_step > MEMORY_SIZE and (total_step % 50 == 0):
                RL.learn()
            # update states
            episode_reward += R
            episode_step += 1
            epi_good_event += IfAppearGoodEvent
    
            S = S_
            S_norm = S_norm_
            if isDone or episode_step > MAX_EPI_STEP:
                if stop_ind == 1:
                    stop_reason = 'pattern is impossible'
                elif stop_ind == 2:
                    stop_reason = 'a deadlock state'
                elif episode_step > MAX_EPI_STEP:
                    stop_reason = 'reach max steps'
                else:
                    stop_reason = 'next state is empty'
                if max_reward_his < episode_reward:
                    max_reward_his = episode_reward
                print('episode:', num_episode, '\n',
                      'init state:', init_S, '\n',
                      'episode reward:', episode_reward, '\n',
                      'episode step:', episode_step, '\n',
                      'good event:', epi_good_event, '\n',
                      'epsilon value:', RL.epsilon, '\n',
                      #'action list:', epi_action_list, '\n',
                      'maximal running step:', np.max(episode_step_history), '\n',
                      # 'total good event:', np.sum(good_event_history), '\n',
                      'maximal episode reward:', max_reward_his, '\n',
                      # stop_reason, '\n',
                      'total state explored:', reached_state_len, '\n',
                      '*******************************************')
                reward_history.append(episode_reward)
                good_event_history.append(epi_good_event)
                episode_step_history.append(episode_step)
                
                # save checkpoint model, if a good model is received
                if episode_reward > RECORD_VAL:
                    save_path = cwd +'\\' + str(num_episode) + '_reward' + str(np.round(episode_reward)) + 'init_state_' + str(ind_temp) + '.ckpt'
                    saver.save(RL.sess, save_path)
                break
            total_step += 1
            epi_action_list.append(selected_action)
    
    # draw cost curve
    RL.plot_cost()
    RL.plot_reward(reward_history, 100)
    RL.plot_epiEvent(good_event_history)
    save_path_reward_mat = cwd + '\\' + 'reward_his.mat'
    sio.savemat(save_path_reward_mat, mdict={'reward':reward_history})
else:
    # %% for single checkpoint test
    # file_path = "C:\\Users\\kaiget\\OneDrive - KTH\\work\\MB_DQN_Junjun\\disable_approach\\2024-01-12\\Train\\119771_reward2065.0init_state_0.ckpt" # fill in the target ckpt
    file_path = "C:\\Users\\kaiget\\OneDrive - KTH\\work\\MB_DQN_Junjun\\disable_approach\\2023-12-27\\Train\\79964_reward739.3585824436218step201.ckpt" # fill in the target ckpt
    tf.reset_default_graph()    
    [generated_states_full, Problem_state, len_gen_states] = RL.check_action_Train(15*[0], file_path, plant_param)
    # 0115: check the previous actions
    # prob_state_set_path = os.getcwd() + '\\data\\Train\\train_prob_state_set.txt'
    # prob_state_set = np.loadtxt(prob_state_set_path, dtype=int).tolist()
    # [matching_state, Problem_state] = RL.check_previous_state(file_path, plant_param, prob_state_set)
    
    

            
            