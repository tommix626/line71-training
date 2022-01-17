from torch.distributions.normal import Normal
import torch.nn.functional as F
#from .networks import MLPNetwork
from .policy import PolicyNetwork
from .value import ValueNetwork
from .misc import hard_update
from .Rollout import RolloutStorage
from torch.optim import Adam
import torch.nn as nn
import torch 
import numpy as np
import os
import torch.optim.lr_scheduler as lr_scheduler
class Model(object):
    def __init__(self,config=None):#定义智能体的策略网络，评价网络，网络优化器
        return
    def get_action(self,s):#输入状态，返回动作
        action = 1
        return action

    def get_value(self,s):#输入状态，返回v(s)
        return
        return value

    def save_loss(self, policy_loss, value_loss):#保存每个epoch的policy loss和value_loss
        return
    def evaluate_actions(self, obs, obs_all,actions):# 返回memory中所有的v(s)，采样策略的动作概率，和当前策略的动作概率
        return
        return values, old_action_log_probs, action_log_probs

    def compute_loss(self, observation_batch,obs_all_batch,action_batch,return_batch):#计算这个batch的记录所对应的loss，并返回
        """
        observation_batch --- list of states
        action_batch   ---- list of actions
        """
        return
        return action_loss, value_loss

    def update(self, rollout):#根据rollout更新actor和critic网络，返回平均每个epoch的actor loss和critic loss
        return
        return action_loss_epoch, value_loss_epoch

    def save_w(self,filepath):#保存当前模型
        return
    def load_w(self,filepath):#加载模型
        return
