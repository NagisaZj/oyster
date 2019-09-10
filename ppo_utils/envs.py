import numpy as np
import torch
import os
from rlkit.envs import ENVS
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.torch.networks import SnailEncoder, MlpEncoder
from rlkit.torch.sac.policies import  PEARLTanhGaussianPolicy
from rlkit.torch.sac.agent import PEARLAgent, ExpAgent

class Env():
    def create_env(self,model_path=None):
        self.env = NormalizedBoxEnv(ENVS['sparse-point-robot'](n_tasks=100, randomize_tasks=True))
        self.encoder = MlpEncoder(hidden_sizes=10,
                             input_size=5,
                             output_size=10, )
        self.policy = PEARLTanhGaussianPolicy(
        hidden_sizes=[300, 300, 300],
        obs_dim=2 + 10,
        latent_dim=10,
        action_dim=2,
    )
        self.agent  = PEARLAgent(
        5,
        self.encoder,
        self.policy,
            {'recurrent':False,'use_information_bottleneck':False,'sparse_rewards':1,'use_next_obs_in_context':False,'use_info_in_context':False}
    )
        self.num_task = 0

        if model_path is not None:
            self.encoder.load_state_dict(torch.load(os.path.join(model_path, 'context_encoder.pth')))
            self.policy.load_state_dict(torch.load(os.path.join(model_path, 'policy.pth')))

    def reset_task(self,task_num):
        self.env.reset_task(task_num)

    def forward(self,action,get_exploit_reward=False,context=None):
        next_o, r, d, env_info = self.env.step(action)
        exploit_reward = None
        if get_exploit_reward:
            paths = self.collect_path(context)
            exploit_reward = self.cal_rew(paths)
        return next_o, r, d, env_info, exploit_reward





