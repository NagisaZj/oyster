import numpy as np
import matplotlib.pyplot as plt
import json
import rlkit.torch.pytorch_util as ptu
import os
import joblib
from rlkit.envs.wrappers import NormalizedBoxEnv, AugmentedBoxObservationShapeEnv
from rlkit.envs.sparse_point_env import PointEnv_SMM
from rlkit.smm.smm_hook import SMMHook
from rlkit.smm.historical_policies_hook import HistoricalPoliciesHook
import pickle

def create_env(env_id, env_kwargs, num_skills=0):
    if env_id == 'PointEnv':
        env = NormalizedBoxEnv(PointEnv_SMM(**env_kwargs))
        training_env = NormalizedBoxEnv(PointEnv_SMM(**env_kwargs))
    else:
        raise NotImplementedError('Unrecognized environment:', env_id)

    # Append skill to observation vector.
    if num_skills > 0:
        env = AugmentedBoxObservationShapeEnv(env, num_skills)
        training_env = AugmentedBoxObservationShapeEnv(env, num_skills)

    return env, training_env

def overwrite_dict(old_dict, new_dict):
    """Recursively update old_dict (in-place) with values from new_dict."""
    for key, val in new_dict.items():
        if isinstance(val, dict):
            if key not in old_dict:
                old_dict[key] = val
            else:
                overwrite_dict(old_dict[key], val)
        else:
            old_dict[key] = val

def load_experiment(log_dir, variant_overwrite=dict()):
    """
    Loads environment and trained policy from file.
    """
    # Load variant.json.
    with open(os.path.join(log_dir, 'variant.json')) as json_file:
        variant = json.load(json_file)
    variant['log_dir'] = log_dir
    print("Read variants:")
    print(json.dumps(variant, indent=4, sort_keys=True))

    # Overwrite variants.
    overwrite_dict(variant, variant_overwrite)
    print('Overwrote variants:')
    print(json.dumps(variant, indent=4, sort_keys=True))

    # Load trained policy from file.
    if 'params_pkl' in variant:
        pkl_file = variant['params_pkl']
    else:
        pkl_file = 'params.pkl'
    ckpt_path = os.path.join(log_dir, pkl_file)
    print('Loading checkpoint:', ckpt_path)
    f = open(ckpt_path, 'rb')
    info = pickle.load(f)
    print(info)
    data = joblib.load(ckpt_path)
    print('Data:')
    print(data)

    # Create environment.
    num_skills = variant['smm_kwargs']['num_skills'] if variant['intrinsic_reward'] == 'smm' else 0
    env, training_env = create_env(variant['env_id'], variant['env_kwargs'], num_skills)
    print('env.action_space.low.shape:', env.action_space.low.shape)

    return env, training_env, data, variant

class hard_smm_point():
    def __init__(self):
        return


    def get_action(self,observation):
        if np.sqrt(observation[0]**2+observation[1]**2)<=0.95:
            action = np.array([-1,0],dtype=np.float32)
        elif observation[1]<0:
            action = -1 * observation
            action = np.clip(action,-1,1)
        else:
            theta = np.arccos(observation[0]/np.sqrt(observation[0]**2+observation[1]**2))
            action = np.array([np.cos(theta-np.pi/2),np.sin(theta-np.pi/2)],dtype=np.float32)*1

        return action,None



class trained_smm_point():
    def __init__(self,use_history,SMM_path,num_skills):
        self.use_history = use_history
        log_dir=SMM_path
        self.num_skills = num_skills
        from rlkit.torch.sac.sac import SoftActorCritic
        self.config = dict(
  env_kwargs=dict(
    goal_prior=[1.12871704, 0.46767739, 0.42], # Test-time object goal position
    sample_goal=False,
    shaped_rewards=['object_off_table', 'object_goal_indicator', 'object_gripper_indicator', 'action_penalty'],
    terminate_upon_success=False,
    terminate_upon_failure=False,
  ),
  test_goal=[1.12871704, 0.46767739, 0.42],
  algo_kwargs=dict(
    max_path_length=50,  # Maximum path length in the environment
    num_episodes=100,  # Number of test episodes
    reward_scale=100,  # Weight of the extrinsic reward relative to the SAC reward
    collection_mode='episodic',  # Each epoch is one episode
    num_updates_per_episode=0,  # Evaluate without additional training
  ),
  smm_kwargs=dict(
    update_p_z_prior_coeff=1,  # p(z) coeff for SMM posterior adaptation (higher value corresponds to more uniform p(z))

    # Turn off SMM reward.
    state_entropy_coeff=0,
    latent_entropy_coeff=0,
    latent_conditional_entropy_coeff=0,
    discriminator_lr=0,
  ),
)
        with open('/home/zj/Desktop/sample/smm/configs/test_no_ha_point.json') as f:
            exp_params = json.load(f)
        overwrite_dict(self.config, exp_params)

        ptu.set_gpu_mode(True)
        env, _, data, variant = load_experiment(log_dir, self.config)

        variant['historical_policies_kwargs']['num_historical_policies'] = 10 if self.use_history else 0
        self.policy = data['policy']

        vf = data['vf']
        qf = data['qf']
        self.algorithm = SoftActorCritic(
            env=env,
            training_env=env,  # can't clone box2d env cause of swig
            save_environment=False,  # can't save box2d env cause of swig
            policy=self.policy,
            qf=qf,
            vf=vf,
            **variant['algo_kwargs'],
        )
        self.policy.to('cuda')
        if variant['intrinsic_reward'] == 'smm':
            discriminator = data['discriminator']
            density_model = data['density_model']
            SMMHook(
                base_algorithm=self.algorithm,
                discriminator=discriminator,
                density_model=density_model,
                **variant['smm_kwargs'])

        # Overwrite algorithm for historical averaging.
        if variant['historical_policies_kwargs']['num_historical_policies'] > 0:
            HistoricalPoliciesHook(
                base_algorithm=self.algorithm,
                log_dir=log_dir,
                **variant['historical_policies_kwargs'],
            )

    def get_action(self,observation):
        ptu.set_gpu_mode(True)
        #for param in self.policy.parameters():
        #    print(param)
        aug_observation = np.hstack([observation,1])
        action = self.algorithm.policy.get_actions(aug_observation)
        return action,None



if __name__=="__main__":
    '''policy = hard_smm_point()
    state = np.zeros((2,),dtype=np.float32)
    state_rem = np.zeros((200,2),dtype=np.float32)
    plt.figure()
    for i in range(1000):
        #state_rem[i] = state
        action,_ = policy.get_action(state)
        state = state + action
        plt.scatter(state[0],state[1])
        print(state)

    plt.show()'''

    policy = trained_smm_point(True,'/home/zj/Desktop/sample/smm/out/PointEnv-1-0.1/sac-smm-1-rl1.0-sec10.0-lec1.0-lcec1.0_2019_07_08_20_20_29_0000--s-0',1)
    print(policy.get_action(np.array([0,0])))



