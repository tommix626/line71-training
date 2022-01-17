#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 16:19:18 2021

@author: mendy
"""
import numpy as np
from Env import Env
from DrawPicture import gif_init,gif_update,plot_tsd
#from MRL_brain_bus import agent_PPO
#from MRL_policy_estimator import policy_estimator
import matplotlib.pyplot as plt
import pandas as pd
import numpy.ma as ma
import math
import matplotlib
import os
from utils.PPO_answer import Model
import pickle
from IPython.display import display, clear_output, Image
#matplotlib.rc("font",**{"family":"sans-serif","sans-serif":["Helvetica","Arial"],"size":14})
matplotlib.rc('pdf', fonttype=42, use14corefonts=True, compression=1)
matplotlib.rc('ps', useafm=True, usedistiller='none', fonttype=42)
matplotlib.rc("axes", unicode_minus=False, linewidth=1, labelsize='medium')
matplotlib.rc("axes.formatter", limits=[-7,7])
#matplotlib.rc('savefig', bbox='tight', format='pdf', frameon=False, pad_inches=0.05)
matplotlib.rc('lines', marker=None, markersize=4)
matplotlib.rc('text', usetex=False)
matplotlib.rc('xtick', direction='in')
matplotlib.rc('xtick.major', size=8)
matplotlib.rc('xtick.minor', size=2)
matplotlib.rc('ytick', direction='in')
matplotlib.rc('lines', linewidth=1)
matplotlib.rc('ytick.major', size=8)
matplotlib.rc('ytick.minor', size=2)
matplotlib.rcParams['lines.solid_capstyle'] = 'butt'
matplotlib.rcParams['lines.solid_joinstyle'] = 'bevel'
matplotlib.rc('mathtext', fontset='stixsans')
matplotlib.rc('legend', fontsize='small', frameon=False,
              handleheight=0.5, handlelength=1, handletextpad=0.1, numpoints=1)

from utils.hyperparameters import EnvConfig
#from utils.PPO import Model
from utils.hyperparameters import Config
import torch
import time

#######-------参数设置----#######
envConfig = EnvConfig()

Hold_Strategy = envConfig.Hold_Strategy  # 0 - no control; 1 - schedule-based 2 - forward headway, 3- two-way headway based # 4- Jiawei
#---存储结果"/result"---
root = os.getcwd()
result_dname = "result_test_71line_16car_"
if Hold_Strategy == 0:
    tmp = os.path.join(root,result_dname)
    result_path = os.path.join(tmp,"no_control")
elif Hold_Strategy == 1:
    tmp = os.path.join(root,result_dname)
    result_path = os.path.join(tmp,"schedule_based")
elif Hold_Strategy == 2:
    tmp = os.path.join(root,result_dname)
    result_path = os.path.join(tmp,"forward_headway")
elif Hold_Strategy == 3:
    tmp = os.path.join(root,result_dname)
    result_path = os.path.join(tmp,"two_way")
elif Hold_Strategy == 4:
    tmp = os.path.join(root,result_dname)
    result_path = os.path.join(tmp,"jiawei")
elif Hold_Strategy == 5:
    tmp = os.path.join(root,result_dname)
    result_path = os.path.join(tmp,"me")

fig_dname = "fig" 
fig_dir = os.path.join(result_path,fig_dname)
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)
para_dname = "para"
para_dir = os.path.join(result_path,"para")
if not os.path.exists(para_dir):
    os.makedirs(para_dir)
    
model_dname = "saved_agents"
model_dir = os.path.join(result_path,"saved_agents")
if not os.path.exists(model_dir):
    os.makedirs(model_dir)



####------rl setting ------
if Hold_Strategy == 4:
    config = Config()
    
    #模型
    model = Model(config)
    model.load_w(model_dir)
    reward_set = []
    reward_p1_set = []
    reward_p2_set = []
    actor_loss_set=[]
    v_loss_set = []

line = [[]for i in range(envConfig.Dir_Num)]
ax = []
buses_plot = []
pax_bar = []
rollout_num = 2



# ave_wait_list,std_wait_list,ave_travel_list,std_travel_list,ave_all_list,std_all_list = [],[],[],[],[],[]
for i_ep in range(envConfig.num_episodes):
    ave_reward_each_rollout = []
    ave_var_each_rollout = [] 
    ave_reward_p1_each_rollout = []
    ave_reward_p2_each_rollout = []
    for rolloutidx in range(rollout_num):  #FIXME what is rollout_num
        
        # env = Env(bus_stop_num=envConfig.Stop_Num,bus_num = envConfig.bus_num_each_dir,\
        #           stop_loc_list=envConfig.Stop_Loc_List,arr_rates = envConfig.Arr_Rates,bus_omega= envConfig.bus_omega,hold_strategy=Hold_Strategy,\
        #               pax_saturate=envConfig.pax_saturate,ran_travel=envConfig.ran_travel,mu_time = envConfig.mu_time,headway=envConfig.headway,\
        #                   alight_rate=envConfig.alight_rate,board_rate=envConfig.board_rate,warm_up_time = envConfig.warm_up_time,road_length=envConfig.road_length,envConfig=envConfig)
        env = Env(envConfig=envConfig)
        step_t = 0
        var = []
        start_time = time.time()
        while step_t <= envConfig.Sim_Horizon:
            # update environment
            
            s = env.update(step_t)
            
            if step_t > envConfig.warm_up_time:
                if Hold_Strategy == 4:
                    actions = [-1 for _ in range(envConfig.bus_num)]
                    actions = np.array(actions,dtype=float)
                    values = np.zeros(envConfig.bus_num,dtype=float)
                    local_obs = [[]for _ in range(envConfig.bus_num)]
                    exist_ctl = False
                    for i in range(envConfig.bus_num):
                        
                        if len(s[i][:])>0:
                            # print("s[i]=",s[i])
                            var.append(1/(1.+np.var(np.array(s[i]))))
                            exist_ctl = True
                            local_obs[i].extend([s[i][0],s[i][-1]]) #local_obs 只存前后车时距 s存的是对每个车来说所有其他车的时距
                            current_local_obs = torch.from_numpy(np.array(local_obs[i])).float().to(config.device)
                            current_obs = torch.from_numpy(np.array(s[i])).float().to(config.device)
                            with torch.no_grad():
                                a = model.get_action(current_local_obs)#执行动作
                                # v = model.get_value(current_obs)#获得模型
                            actions[i] = a.view(-1).cpu().numpy() #FIXME what is view & what num stands for what action
                            # values[i] = v.view(-1).cpu().numpy()
                            # if env.bus_list[i].catch == True:
                            #     actions[i] = 1.0
                            #     with torch.no_grad():
                            #         v = model.get_value(current_obs)
                            #     values[i] = v.view(-1).cpu().numpy()
                            # else:
                            #     with torch.no_grad():
                            #         a = model.get_action(current_local_obs)#执行动作
                            #         v = model.get_value(current_obs)#获得模型
                            #     actions[i] = a.view(-1).cpu().numpy()
                            #     values[i] = v.view(-1).cpu().numpy()
                            # print("bus id ,驻留：",i,actions[i],step_t)
                        
    
                    if exist_ctl == True:
                        # print(s)
                        #把动作，状态，奖励存进memory中
                        model.rollouts.insert(step_t,local_obs,s,actions,config.w1,config.w2) #FIXME what is model.rollouts : savedatas
                        env.control(actions)
            step_t += 1
        end_time = time.time()
        print("cost time:",end_time-start_time)
        if Hold_Strategy == 4:
            ave_reward = 0
            ave_reward_p1 = 0
            ave_reward_p2 = 0
            length = 0
            # f = plt.figure()
            
            for i in range(len(model.rollouts.rewards)):
                # print(i,len(model.rollouts.rewards[i]))
                temp_r = np.array(model.rollouts.rewards[i])
                temp_r1 = np.array(model.rollouts.rewards_p1[i])
                temp_r2 =np.array(model.rollouts.rewards_p2[i])
                length += temp_r.shape[0]
                ave_reward_p1 += temp_r1.sum()
                ave_reward_p2 += temp_r2.sum()
                ave_reward += temp_r.sum()
                # plt.plot(list(range(len(model.rollouts.actions[i]))),model.rollouts.actions[i])
            ave_reward /= length
            ave_reward_p1 /= length
            ave_reward_p2 /= length
            ave_var = sum(var)/len(var)
            print(' num_episodes:%d rollout index:%d  r:%g   realvar:%g' % (i_ep,rolloutidx, ave_reward,ave_var))
            ave_reward_each_rollout.append(ave_reward)
            ave_reward_p1_each_rollout.append(ave_reward_p1)
            ave_reward_p2_each_rollout.append(ave_reward_p2)
        env.save_traj(step_t)
        if (i_ep*rollout_num+rolloutidx) % 15== 0:
            plot_tsd(env.trajectory,envConfig.Sim_Horizon,env.bus_num,envConfig.Stop_Loc_List[0],fig_dir,i_ep)
            
        # ave_w,std_w,ave_tr,std_tr,ave_all,std_all = env.cal_ave_time()
        # ave_wait_list.append(ave_w)
        # std_wait_list.append(std_w)
        # ave_travel_list.append(ave_tr)
        # std_travel_list.append(std_tr)
        # ave_all_list.append(ave_all)
        # std_all_list.append(std_all)

        #仿真结束，更新网络
        if Hold_Strategy == 4:
            for i in range(envConfig.bus_num):
                for a_idx in range(len(model.rollouts.actions[i])-1):
                    start_t = model.rollouts.time[i][a_idx]
                    end_t = model.rollouts.time[i][a_idx+1]
                    #其他bus的駐留時間
                    for j in range(1,envConfig.bus_num):
                        for_ind = (i-j)%envConfig.bus_num
                        tmp = np.array(env.hold_record[for_ind][start_t:end_t])
                        hold_t = np.sum(tmp>0)/180.0
                        model.rollouts.observations_all[i][a_idx].append(hold_t)
                model.rollouts.observations_all[i].remove(model.rollouts.observations_all[i][-1])
                 # caculate value function 
                obs_tmp = torch.from_numpy(np.array(model.rollouts.observations_all[i])).float().to(config.device)
                with torch.no_grad():
                    v = model.get_value(obs_tmp)
                v_cpu = v.view(-1).cpu().numpy().tolist()
                # print(len(v_cpu))
                model.rollouts.value_preds[i].extend(v_cpu)
    
            model.rollouts.compute_returns(config.GAMMA)
            model.rollouts.after_epoch()
        
    if Hold_Strategy == 4:
        reward_set.append(sum(ave_reward_each_rollout)/rollout_num)
        reward_p1_set.append(sum(ave_reward_p1_each_rollout)/rollout_num)
        reward_p2_set.append(sum(ave_reward_p2_each_rollout)/rollout_num)
        value_loss, action_loss= model.update(model.rollouts) #update the network(training)
        model.rollouts.after_update()
        v_loss_set.append(value_loss)
        actor_loss_set.append(action_loss)
        model.save_w(model_dir)

# ave_wait = np.mean(ave_wait_list)
# std_wait = np.mean(std_wait_list)
# ave_travel = np.mean(ave_travel_list)
# std_travel = np.mean(std_travel_list)
# ave_alltime = np.mean(ave_all_list)
# std_alltime = np.mean(std_all_list)
# print("平均等待时间：",ave_wait)
# print("标准差：",std_wait)
    # print("平均乘车时间：",ave_travel)
    # print("标准差：",std_travel)
    # print("average journey time:",ave_alltime)
    # print("std:",std_alltime)

if Hold_Strategy == 4: # ploting
    #PLOT critic loss
    f = plt.figure()
    ax = plt.subplot(111)
    plt.xlabel('Training episode')
    plt.ylabel('Mean squared error')
    smoothing_window = 10
    v_loss_set_smoothed = pd.Series(v_loss_set).rolling(smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(v_loss_set, alpha=0.2)
    plt.plot(v_loss_set_smoothed, color='orange')
    plt.grid()
    plt.show()
    f.savefig(os.path.join(fig_dir,"critic.pdf"), bbox_inches='tight')
    #plot actor loss
    f = plt.figure()
    ax = plt.subplot(111)
    plt.xlabel('Training episode')
    plt.ylabel('Actor loss')
    smoothing_window = 10
    a_loss_set_smoothed = pd.Series(actor_loss_set).rolling(smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(actor_loss_set, alpha=0.2)
    plt.plot(a_loss_set_smoothed, color='orange')
    plt.grid()
    plt.show()
    f.savefig(os.path.join(fig_dir,"actor.pdf"), bbox_inches='tight')
    # plot reward
    f = plt.figure()
    ax = plt.subplot(111)
    ax.tick_params(length=4, width=0.5)
    plt.xlabel('Training episode' )
    plt.ylabel('Cumulative global reward' )
    smoothing_window = 10
    rewards_smoothed = pd.Series(reward_set).rolling(smoothing_window, min_periods=smoothing_window).mean()
    rewards_smoothed1 = pd.Series(reward_p1_set).rolling(smoothing_window, min_periods=smoothing_window).mean()
    rewards_smoothed2 = pd.Series(reward_p2_set).rolling(smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(reward_set, alpha=0.2)
    plt.plot(rewards_smoothed,color='orange',label='total reward')
    plt.plot(reward_p1_set, alpha=0.2)
    plt.plot(rewards_smoothed1, color='red',label='reward for holding penalty')
    plt.plot(reward_p2_set, alpha=0.2)
    plt.plot(rewards_smoothed2, color='green',label='reward for headway equalization')
    ax.legend(loc='best',  fancybox=True, shadow=False, ncol=1, prop={'size': 12})
    plt.show()
    f.savefig(os.path.join(fig_dir,"reward_train.pdf"), bbox_inches='tight')

org_file = os.path.join(root,"utils/hyperparameters.py")
command = "cp "+org_file+" "+result_path
os.system(command)