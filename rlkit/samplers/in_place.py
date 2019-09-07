import numpy as np

from rlkit.samplers.util import rollout,SMMrollout,seedrollout, exprollout, exprollout_split,exprollout_splitsimple,exprolloutsimple
from rlkit.torch.sac.policies import MakeDeterministic


class InPlacePathSampler(object):
    """
    A sampler that does not serialization for sampling. Instead, it just uses
    the current policy and environment as-is.

    WARNING: This will affect the environment! So
    ```
    sampler = InPlacePathSampler(env, ...)
    sampler.obtain_samples  # this has side-effects: env will change!
    ```
    """
    def __init__(self, env, policy, max_path_length):
        self.env = env
        self.policy = policy

        self.max_path_length = max_path_length

    def start_worker(self):
        pass

    def shutdown_worker(self):
        pass

    def obtain_samples(self, deterministic=False, max_samples=np.inf, max_trajs=np.inf, accum_context=True, resample=1):
        """
        Obtains samples in the environment until either we reach either max_samples transitions or
        num_traj trajectories.
        The resample argument specifies how often (in trajectories) the agent will resample it's context.
        """
        assert max_samples < np.inf or max_trajs < np.inf, "either max_samples or max_trajs must be finite"
        policy = MakeDeterministic(self.policy) if deterministic else self.policy
        paths = []
        n_steps_total = 0
        n_trajs = 0
        while n_steps_total < max_samples and n_trajs < max_trajs:
            path = rollout(
                self.env, policy, max_path_length=self.max_path_length, accum_context=accum_context)
            # save the latent context that generated this trajectory
            path['context'] = policy.z.detach().cpu().numpy()
            paths.append(path)
            n_steps_total += len(path['observations'])
            n_trajs += 1
            # don't we also want the option to resample z ever transition?
            if n_trajs % resample == 0:
                policy.sample_z()
        return paths, n_steps_total

class ExpInPlacePathSampler(object):
    """
    A sampler that does not serialization for sampling. Instead, it just uses
    the current policy and environment as-is.

    WARNING: This will affect the environment! So
    ```
    sampler = InPlacePathSampler(env, ...)
    sampler.obtain_samples  # this has side-effects: env will change!
    ```
    """
    def __init__(self, env, policy, max_path_length,encoder):
        self.env = env
        self.policy = policy
        self.encoder = encoder
        self.max_path_length = max_path_length

    def start_worker(self):
        pass

    def shutdown_worker(self):
        pass

    def obtain_samples(self, deterministic=False, max_samples=np.inf, max_trajs=np.inf, accum_context_for_agent=False, resample=1, context_agent = None,split=False):
        """
        Obtains samples in the environment until either we reach either max_samples transitions or
        num_traj trajectories.
        The resample argument specifies how often (in trajectories) the agent will resample it's context.
        """
        assert max_samples < np.inf or max_trajs < np.inf, "either max_samples or max_trajs must be finite"
        policy = MakeDeterministic(self.policy) if deterministic else self.policy
        paths = []
        n_steps_total = 0
        n_trajs = 0
        if not split:
            path = exprollout(
            self.env, policy, max_path_length=self.max_path_length, max_trajs=max_trajs, accum_context_for_agent=accum_context_for_agent, context_agent = context_agent)
            paths.append(path)
            n_steps_total += len(path['observations'])
            n_trajs += 1

            return paths, n_steps_total
        else:
            path = exprollout_split(
                self.env, policy, max_path_length=self.max_path_length, max_trajs=max_trajs,
                accum_context_for_agent=accum_context_for_agent, context_agent=context_agent)
            n_steps_total += self.max_path_length * max_trajs
            n_trajs += max_trajs

            return path, n_steps_total

class ExpInPlacePathSamplerSimple(object):
    """
    A sampler that does not serialization for sampling. Instead, it just uses
    the current policy and environment as-is.

    WARNING: This will affect the environment! So
    ```
    sampler = InPlacePathSampler(env, ...)
    sampler.obtain_samples  # this has side-effects: env will change!
    ```
    """
    def __init__(self, env, policy, max_path_length,encoder):
        self.env = env
        self.policy = policy
        self.encoder = encoder
        self.max_path_length = max_path_length

    def start_worker(self):
        pass

    def shutdown_worker(self):
        pass

    def obtain_samples(self, deterministic=False, max_samples=np.inf, max_trajs=np.inf, accum_context_for_agent=False, resample=1, context_agent = None,split=False):
        """
        Obtains samples in the environment until either we reach either max_samples transitions or
        num_traj trajectories.
        The resample argument specifies how often (in trajectories) the agent will resample it's context.
        """
        assert max_samples < np.inf or max_trajs < np.inf, "either max_samples or max_trajs must be finite"
        policy = MakeDeterministic(self.policy) if deterministic else self.policy
        paths = []
        n_steps_total = 0
        n_trajs = 0
        if not split:
            path = exprolloutsimple(
            self.env, policy, max_path_length=self.max_path_length, max_trajs=max_trajs, accum_context_for_agent=accum_context_for_agent, context_agent = context_agent)
            paths.append(path)
            n_steps_total += len(path['observations'])
            n_trajs += 1

            return paths, n_steps_total
        else:
            path = exprollout_splitsimple(
                self.env, policy, max_path_length=self.max_path_length, max_trajs=max_trajs,
                accum_context_for_agent=accum_context_for_agent, context_agent=context_agent)
            n_steps_total += self.max_path_length * max_trajs
            n_trajs += max_trajs

            return path, n_steps_total

class SeedInPlacePathSampler(object):
    """
    A sampler that does not serialization for sampling. Instead, it just uses
    the current policy and environment as-is.

    WARNING: This will affect the environment! So
    ```
    sampler = InPlacePathSampler(env, ...)
    sampler.obtain_samples  # this has side-effects: env will change!
    ```
    """
    def __init__(self, env, policy, max_path_length):
        self.env = env
        self.policy = policy

        self.max_path_length = max_path_length
        self.latent_dim = policy.latent_dim

    def start_worker(self):
        pass

    def shutdown_worker(self):
        pass

    def obtain_samples(self, deterministic=False, max_samples=np.inf, max_trajs=np.inf, accum_context=True, resample=1):
        """
        Obtains samples in the environment until either we reach either max_samples transitions or
        num_traj trajectories.
        The resample argument specifies how often (in trajectories) the agent will resample it's context.
        """
        assert max_samples < np.inf or max_trajs < np.inf, "either max_samples or max_trajs must be finite"
        policy = MakeDeterministic(self.policy) if deterministic else self.policy
        paths = []
        n_steps_total = 0
        n_trajs = 0
        while n_steps_total < max_samples and n_trajs < max_trajs:
            self.random_seed = np.random.randn(1,self.latent_dim)
            path = seedrollout(
                self.env, policy, max_path_length=self.max_path_length, accum_context=accum_context,random_seed=self.random_seed)
            # save the latent context that generated this trajectory
            path['context'] = policy.z.detach().cpu().numpy()
            paths.append(path)
            n_steps_total += len(path['observations'])
            n_trajs += 1
            # don't we also want the option to resample z ever transition?
            #if n_trajs % resample == 0:
            #    policy.sample_z()
        return paths, n_steps_total

class SMMInPlacePathSampler(object):
    """
    A sampler that does not serialization for sampling. Instead, it just uses
    the current policy and environment as-is.

    WARNING: This will affect the environment! So
    ```
    sampler = InPlacePathSampler(env, ...)
    sampler.obtain_samples  # this has side-effects: env will change!
    ```
    """
    def __init__(self, env, policy, max_samples, max_path_length):
        self.env = env
        self.policy = policy
        self.max_path_length = max_path_length
        self.max_samples = max_samples
        assert max_samples >= max_path_length, "Need max_samples >= max_path_length"

    def start_worker(self):
        pass

    def shutdown_worker(self):
        pass

    def start_new_rollout(self):
        pass

    def handle_rollout_ending(self):
        pass

    def obtain_samples(self):
        paths = []
        n_steps_total = 0
        while n_steps_total + self.max_path_length <= self.max_samples:
            self.start_new_rollout()
            path = SMMrollout(
                self.env, self.policy, max_path_length=self.max_path_length
            )
            self.handle_rollout_ending()
            paths.append(path)
            n_steps_total += len(path['observations'])
        return paths
