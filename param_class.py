# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 10:00:06 2023

@author: kaiget
"""

import scipy.io as sio

class AGV_param:
    def __init__(self):
        # %% Load models
        load_data = sio.loadmat('.\\data\\AGV\\test3.mat')  #Get the model file from MATLAB 
        # load_dataB = sio.loadmat('B_2pattern_AGV.mat')  #Get the model file from MATLAB 
        load_dataB_new = sio.loadmat('.\\data\\AGV\\B_new.mat')
        self.AGV_1 = load_data['AGV_1'] 
        self.AGV_2 = load_data['AGV_2'] 
        self.AGV_3 = load_data['AGV_3'] 
        self.AGV_4 = load_data['AGV_4'] 
        self.AGV_5 = load_data['AGV_5'] 
        self.SUP_IPSR = load_data['SUP_IPSR']
        self.SUP_ZWSR = load_data['SUP_ZWSR']
        # B = load_dataB['B']   #0 ~ 55
        self.B_new = load_dataB_new['B_new']  # in the modified action sets, we use B_new
        self.E_c = range(0,20,2)  # 0 ~ 18, in MATLAB, it is 1~19, controllable events
        self.E_u = range(1,33,2)  # 1 ~ 31, in MATLAB, it is 2~32, uncontrollable events
        self.E = set(self.E_c).union(self.E_u)
        
        
class Train_param:
    def __init__(self):
        # %% Load models
        load_data = sio.loadmat('.\\data\\Train\\Train_model_0616.mat')  #Get the model file from MATLAB 
        self.st1 = load_data['st1']    # st1[i-1][j-1][k-1] <-- st1[i][j][k]
        self.st2 = load_data['st2']
        self.st3 = load_data['st3']
        self.st4 = load_data['st4']

        self.rs1 = load_data['rs1']
        self.rs2 = load_data['rs2']
        self.rs3 = load_data['rs3']
        self.rs4 = load_data['rs4']

        self.rt1 = load_data['rt1']
        self.rt2 = load_data['rt2']
        self.rt3 = load_data['rt3']
        self.rt4 = load_data['rt4']

        self.A1 = load_data['A1']
        self.A2 = load_data['A2']
        self.A3 = load_data['A3']
        self.A4 = load_data['A4']
        self.A5 = load_data['A5']
        self.A6 = load_data['A6']
        self.A7 = load_data['A7']
        self.A8 = load_data['A8']
        self.A9 = load_data['A9']
        self.A10 = load_data['A10']
        self.A11 = load_data['A11']
        self.A12 = load_data['A12']

        self.E_c = [11,13,15,17,21,23,25,27,31,33,35,37,41,43,45,47]
        self.E_u = [10,12,14,16,20,22,24,26,30,32,34,36,40,42,44,46]
        
        
        
        