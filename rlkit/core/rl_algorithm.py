import abc
from collections import OrderedDict
import time

import gtimer as gt
import numpy as np

from rlkit.core import logger, eval_util
from rlkit.data_management.env_replay_buffer import MultiTaskReplayBuffer,EnvReplayBuffer
from rlkit.data_management.path_builder import PathBuilder
from rlkit.samplers.in_place import SMMInPlacePathSampler, InPlacePathSampler,SeedInPlacePathSampler, ExpInPlacePathSampler,ExpInPlacePathSamplerSimple
from rlkit.torch import pytorch_util as ptu
from rlkit.smm.smm_policy import hard_smm_point
from rlkit.smm.smm_sampler import SMMSampler
from rlkit.policies.base import ExplorationPolicy
import pickle
import torch

class MetaRLAlgorithm(metaclass=abc.ABCMeta):
    def __init__(
            self,
            env,
            agent,
            train_tasks,
            eval_tasks,
            meta_batch=64,
            num_iterations=100,
            num_train_steps_per_itr=1000,
            num_initial_steps=100,
            num_tasks_sample=100,
            num_steps_prior=100,
            num_steps_posterior=100,
            num_extra_rl_steps_posterior=100,
            num_evals=10,
            num_steps_per_eval=1000,
            batch_size=1024,
            embedding_batch_size=1024,
            embedding_mini_batch_size=1024,
            max_path_length=1000,
            discount=0.99,
            replay_buffer_size=1000000,
            reward_scale=1,
            num_exp_traj_eval=1,
            update_post_train=1,
            eval_deterministic=False,
            render=False,
            save_replay_buffer=False,
            save_algorithm=False,
            save_environment=False,
            render_eval_paths=False,
            dump_eval_paths=False,
            plotter=None,
            use_SMM=False,
            load_SMM =False,
            use_history=False,
            SMM_path=None,
            num_skills = 1,
            seed_sample=False,
            attention=False,
            snail=False
    ):
        """
        :param env: training env
        :param agent: agent that is conditioned on a latent variable z that rl_algorithm is responsible for feeding in
        :param train_tasks: list of tasks used for training
        :param eval_tasks: list of tasks used for eval

        see default experiment config file for descriptions of the rest of the arguments
        """
        self.env = env
        self.agent = agent
        self.exploration_agent = agent # Can potentially use a different policy purely for exploration rather than also solving tasks, currently not being used
        self.train_tasks = train_tasks
        self.eval_tasks = eval_tasks
        self.meta_batch = meta_batch
        self.num_iterations = num_iterations
        self.num_train_steps_per_itr = num_train_steps_per_itr
        self.num_initial_steps = num_initial_steps
        self.num_tasks_sample = num_tasks_sample
        self.num_steps_prior = num_steps_prior
        self.num_steps_posterior = num_steps_posterior
        self.num_extra_rl_steps_posterior = num_extra_rl_steps_posterior
        self.num_evals = num_evals
        self.num_steps_per_eval = num_steps_per_eval
        self.batch_size = batch_size
        self.embedding_batch_size = embedding_batch_size
        self.embedding_mini_batch_size = embedding_mini_batch_size
        self.max_path_length = max_path_length
        self.discount = discount
        self.replay_buffer_size = replay_buffer_size
        self.reward_scale = reward_scale
        self.update_post_train = update_post_train
        self.num_exp_traj_eval = num_exp_traj_eval
        self.eval_deterministic = eval_deterministic
        self.render = render
        self.save_replay_buffer = save_replay_buffer
        self.save_algorithm = save_algorithm
        self.save_environment = save_environment

        self.eval_statistics = None
        self.render_eval_paths = render_eval_paths
        self.dump_eval_paths = dump_eval_paths
        self.plotter = plotter
        self.use_SMM = use_SMM
        self.load_SMM = load_SMM
        self.use_history = use_history,
        self.SMM_path = SMM_path
        self.num_skills = num_skills
        self.seed_sample = seed_sample

        self.sampler = InPlacePathSampler(
            env=env,
            policy=agent,
            max_path_length=self.max_path_length,
        )

        if self.seed_sample:
            self.seedsampler = SeedInPlacePathSampler(
            env=env,
            policy=agent,
            max_path_length=self.max_path_length,
        )

        if self.use_SMM:
            self.smm_sampler = SMMSampler(
                env=env,
                max_path_length=max_path_length,
                agent = agent,
                load_SMM=self.load_SMM,
                use_history=self.use_history,
                SMM_path=self.SMM_path,
                num_skills = self.num_skills
        )

        # separate replay buffers for
        # - training RL update
        # - training encoder update
        self.replay_buffer = MultiTaskReplayBuffer(
                self.replay_buffer_size,
                env,
                self.train_tasks,
            )

        self.enc_replay_buffer = MultiTaskReplayBuffer(
                self.replay_buffer_size,
                env,
                self.train_tasks,
        )

        self._n_env_steps_total = 0
        self._n_train_steps_total = 0
        self._n_rollouts_total = 0
        self._do_train_time = 0
        self._epoch_start_time = None
        self._algo_start_time = None
        self._old_table_keys = None
        self._current_path_builder = PathBuilder()
        self._exploration_paths = []

    def make_exploration_policy(self, policy):
         return policy

    def make_eval_policy(self, policy):
        return policy

    def sample_task(self, is_eval=False):
        '''
        sample task randomly
        '''
        if is_eval:
            idx = np.random.randint(len(self.eval_tasks))
        else:
            idx = np.random.randint(len(self.train_tasks))
        return idx

    def train(self):
        '''
        meta-training loop
        '''
        self.pretrain()
        params = self.get_epoch_snapshot(-1)
        logger.save_itr_params(-1, params)
        gt.reset()
        gt.set_def_unique(False)
        self._current_path_builder = PathBuilder()

        # at each iteration, we first collect data from tasks, perform meta-updates, then try to evaluate
        for it_ in gt.timed_for(
                range(self.num_iterations),
                save_itrs=True,
        ):
            self._start_epoch(it_)
            self.training_mode(True)
            if it_ == 0:
                print('collecting initial pool of data for train and eval')
                # temp for evaluating
                for idx in self.train_tasks:
                    self.task_idx = idx
                    self.env.reset_task(idx)
                    if not self.use_SMM:
                        if not self.seed_sample:
                            self.collect_data(self.num_initial_steps, 1, np.inf)
                        else:
                            self.collect_data(self.num_initial_steps, 1, np.inf)
                            self.collect_data_seed(self.num_initial_steps, 1, np.inf,accumulate_context=False)
                    else:
                        self.collect_data_smm(self.num_initial_steps)
                        self.collect_data_policy(self.num_initial_steps, 1, np.inf)
            # Sample data from train tasks.
            for i in range(self.num_tasks_sample):
                idx = np.random.randint(len(self.train_tasks))
                self.task_idx = idx
                self.env.reset_task(idx)
                self.enc_replay_buffer.task_buffers[idx].clear()

                if not self.use_SMM:
                    if not self.seed_sample:
                        # collect some trajectories with z ~ prior
                        if self.num_steps_prior > 0:
                            self.collect_data(self.num_steps_prior, 1, np.inf)
                        # collect some trajectories with z ~ posterior
                        if self.num_steps_posterior > 0:
                            self.collect_data(self.num_steps_posterior, 1, self.update_post_train)
                        # even if encoder is trained only on samples from the prior, the policy needs to learn to handle z ~ posterior
                        if self.num_extra_rl_steps_posterior > 0:
                            self.collect_data(self.num_extra_rl_steps_posterior, 1, self.update_post_train,
                                              add_to_enc_buffer=False)
                    else:
                        if self.num_steps_prior > 0:
                            self.collect_data(self.num_steps_prior, 1, np.inf)
                            self.collect_data_seed(self.num_steps_prior, 1, np.inf,accumulate_context=False)
                        if self.num_steps_posterior > 0:
                            self.collect_data(self.num_steps_posterior, 1, self.update_post_train)
                            self.collect_data_seed(self.num_steps_posterior, 1, self.update_post_train)
                        if self.num_extra_rl_steps_posterior > 0:
                            self.collect_data(self.num_extra_rl_steps_posterior, 1, self.update_post_train,
                                              add_to_enc_buffer=False)
                            self.collect_data_seed(self.num_extra_rl_steps_posterior, 1, self.update_post_train,
                                              add_to_enc_buffer=False)
                else:
                    if self.num_steps_prior > 0:
                        self.collect_data_smm(self.num_steps_prior)
                        self.collect_data_policy(self.num_steps_prior, 1, np.inf)
                    # collect some trajectories with z ~ posterior
                    if self.num_steps_posterior > 0:
                        self.collect_data_smm(self.num_steps_posterior)
                        self.collect_data_policy(self.num_steps_posterior, 1, self.update_post_train)
                    # even if encoder is trained only on samples from the prior, the policy needs to learn to handle z ~ posterior
                    if self.num_extra_rl_steps_posterior > 0:
                        self.collect_data_policy(self.num_extra_rl_steps_posterior, 1, self.update_post_train)

            # Sample train tasks and compute gradient updates on parameters.
            for train_step in range(self.num_train_steps_per_itr):
                indices = np.random.choice(self.train_tasks, self.meta_batch)
                self._do_training(indices)
                self._n_train_steps_total += 1
            gt.stamp('train')

            self.training_mode(False)

            # eval
            self._try_to_eval(it_)
            gt.stamp('eval')

            self._end_epoch()

    def pretrain(self):
        """
        Do anything before the main training phase.
        """
        pass


    def collect_data_smm(self,num_samples):
        '''
        Notice that SMM data should only be available for the encoder
        :param num_samples: number of transitions to sample
        :return:
        '''
        num_transitions = 0
        while num_transitions < num_samples:
            paths, n_samples = self.smm_sampler.obtain_samples(max_samples=num_samples - num_transitions,
                                                           max_trajs=np.inf)
            num_transitions += n_samples

            self.enc_replay_buffer.add_paths(self.task_idx, paths)
        self._n_env_steps_total += num_transitions
        gt.stamp('smm sample')

    def collect_data_policy(self, num_samples, resample_z_rate, update_posterior_rate):
        '''
        get trajectories from current env in batch mode with given policy
        collect complete trajectories until the number of collected transitions >= num_samples

        :param agent: policy to rollout
        :param num_samples: total number of transitions to sample
        :param resample_z_rate: how often to resample latent context z (in units of trajectories)
        :param update_posterior_rate: how often to update q(z | c) from which z is sampled (in units of trajectories)
        :param add_to_enc_buffer: whether to add collected data to encoder replay buffer
        '''
        # start from the prior
        self.agent.clear_z()

        num_transitions = 0
        while num_transitions < num_samples:
            paths, n_samples = self.sampler.obtain_samples(max_samples=num_samples - num_transitions,
                                                                max_trajs=update_posterior_rate,
                                                                accum_context=False,
                                                                resample=resample_z_rate)
            num_transitions += n_samples
            self.replay_buffer.add_paths(self.task_idx, paths)
            if update_posterior_rate != np.inf:
                context = self.sample_context(self.task_idx)
                self.agent.infer_posterior(context)
        self._n_env_steps_total += num_transitions
        gt.stamp('policy sample')

    def collect_data(self, num_samples, resample_z_rate, update_posterior_rate, add_to_enc_buffer=True,add_to_policy_buffer=True):
        '''
        get trajectories from current env in batch mode with given policy
        collect complete trajectories until the number of collected transitions >= num_samples

        :param agent: policy to rollout
        :param num_samples: total number of transitions to sample
        :param resample_z_rate: how often to resample latent context z (in units of trajectories)
        :param update_posterior_rate: how often to update q(z | c) from which z is sampled (in units of trajectories)
        :param add_to_enc_buffer: whether to add collected data to encoder replay buffer
        '''
        # start from the prior
        self.agent.clear_z()

        num_transitions = 0
        while num_transitions < num_samples:
            paths, n_samples = self.sampler.obtain_samples(max_samples=num_samples - num_transitions,
                                                                max_trajs=update_posterior_rate,
                                                                accum_context=False,
                                                                resample=resample_z_rate)
            num_transitions += n_samples
            #for p in paths:
            #    print(p['actions'],p['rewards'])
            if add_to_policy_buffer:
                self.replay_buffer.add_paths(self.task_idx, paths)
            if add_to_enc_buffer:
                self.enc_replay_buffer.add_paths(self.task_idx, paths)
            if update_posterior_rate != np.inf:
                context = self.sample_context(self.task_idx)
                self.agent.infer_posterior(context)
        self._n_env_steps_total += num_transitions
        gt.stamp('sample')

    def collect_data_seed(self, num_samples, resample_z_rate, update_posterior_rate, add_to_enc_buffer=True,add_to_policy_buffer=True,accumulate_context=True):
        self.agent.clear_z()
        num_transitions = 0
        while num_transitions < num_samples:
            paths, n_samples = self.seedsampler.obtain_samples(max_samples=num_samples - num_transitions,
                                                                max_trajs=1,
                                                                accum_context=accumulate_context
                                                                )
            num_transitions += n_samples
            if add_to_policy_buffer:
                self.replay_buffer.add_paths(self.task_idx, paths)
            if add_to_enc_buffer:
                self.enc_replay_buffer.add_paths(self.task_idx, paths)
            #if update_posterior_rate != np.inf:
            #    context = self.prepare_context(self.task_idx)
            #    self.agent.infer_posterior(context)
        self._n_env_steps_total += num_transitions
        gt.stamp('sample')

    def _try_to_eval(self, epoch):
        logger.save_extra_data(self.get_extra_data_to_save(epoch))
        if self._can_evaluate():
            self.evaluate(epoch)

            params = self.get_epoch_snapshot(epoch)
            logger.save_itr_params(epoch, params)
            table_keys = logger.get_table_key_set()
            if self._old_table_keys is not None:
                assert table_keys == self._old_table_keys, (
                    "Table keys cannot change from iteration to iteration."
                )
            self._old_table_keys = table_keys

            logger.record_tabular(
                "Number of train steps total",
                self._n_train_steps_total,
            )
            logger.record_tabular(
                "Number of env steps total",
                self._n_env_steps_total,
            )
            logger.record_tabular(
                "Number of rollouts total",
                self._n_rollouts_total,
            )

            times_itrs = gt.get_times().stamps.itrs
            train_time = times_itrs['train'][-1]
            if not self.use_SMM:
                sample_time = times_itrs['sample'][-1]
            else:
                sample_time = times_itrs['policy sample'][-1]
            eval_time = times_itrs['eval'][-1] if epoch > 0 else 0
            epoch_time = train_time + sample_time + eval_time
            total_time = gt.get_times().total

            logger.record_tabular('Train Time (s)', train_time)
            logger.record_tabular('(Previous) Eval Time (s)', eval_time)
            logger.record_tabular('Sample Time (s)', sample_time)
            logger.record_tabular('Epoch Time (s)', epoch_time)
            logger.record_tabular('Total Train Time (s)', total_time)

            logger.record_tabular("Epoch", epoch)
            logger.dump_tabular(with_prefix=False, with_timestamp=False)
        else:
            logger.log("Skipping eval for now.")

    def _can_evaluate(self):
        """
        One annoying thing about the logger table is that the keys at each
        iteration need to be the exact same. So unless you can compute
        everything, skip evaluation.

        A common example for why you might want to skip evaluation is that at
        the beginning of training, you may not have enough data for a
        validation and training set.

        :return:
        """
        # eval collects its own context, so can eval any time
        return True

    def _can_train(self):
        return all([self.replay_buffer.num_steps_can_sample(idx) >= self.batch_size for idx in self.train_tasks])

    def _get_action_and_info(self, agent, observation):
        """
        Get an action to take in the environment.
        :param observation:
        :return:
        """
        agent.set_num_steps_total(self._n_env_steps_total)
        return agent.get_action(observation,)

    def _start_epoch(self, epoch):
        self._epoch_start_time = time.time()
        self._exploration_paths = []
        self._do_train_time = 0
        logger.push_prefix('Iteration #%d | ' % epoch)

    def _end_epoch(self):
        logger.log("Epoch Duration: {0}".format(
            time.time() - self._epoch_start_time
        ))
        logger.log("Started Training: {0}".format(self._can_train()))
        logger.pop_prefix()

    ##### Snapshotting utils #####
    def get_epoch_snapshot(self, epoch):
        data_to_save = dict(
            epoch=epoch,
            exploration_policy=self.exploration_policy,
        )
        if self.save_environment:
            data_to_save['env'] = self.training_env
        return data_to_save

    def get_extra_data_to_save(self, epoch):
        """
        Save things that shouldn't be saved every snapshot but rather
        overwritten every time.
        :param epoch:
        :return:
        """
        if self.render:
            self.training_env.render(close=True)
        data_to_save = dict(
            epoch=epoch,
        )
        if self.save_environment:
            data_to_save['env'] = self.training_env
        if self.save_replay_buffer:
            data_to_save['replay_buffer'] = self.replay_buffer
        if self.save_algorithm:
            data_to_save['algorithm'] = self
        return data_to_save

    def collect_paths(self, idx, epoch, run):
        self.task_idx = idx
        self.env.reset_task(idx)

        self.agent.clear_z()
        paths = []
        num_transitions = 0
        num_trajs = 0
        if self.use_SMM:
            if not self.load_SMM:
                path, num = self.smm_sampler.obtain_samples(max_samples=self.max_path_length, max_trajs=1,
                                                            accum_context=True)
                num_transitions += num
                self.agent.infer_posterior(self.agent.context)
                while num_transitions < self.num_steps_per_eval:
                    path, num = self.sampler.obtain_samples(deterministic=self.eval_deterministic,
                                                            max_samples=self.num_steps_per_eval - num_transitions,
                                                            max_trajs=1, accum_context=False)
                    paths += path
                    num_transitions += num
                    num_trajs += 1
                    if num_trajs >= self.num_exp_traj_eval:
                        self.agent.infer_posterior(self.agent.context)
            else:
                while num_transitions < self.num_steps_per_eval:
                    path, num = self.smm_sampler.obtain_samples(max_samples=self.max_path_length, max_trajs=1,
                                                                accum_context=True)
                    num_transitions += num
                    #paths+=path
                    num_trajs += 1
                    path, num = self.sampler.obtain_samples(deterministic=self.eval_deterministic,
                                                            max_samples=self.num_steps_per_eval - num_transitions,
                                                            max_trajs=1, accum_context=False)
                    paths += path
                    num_transitions += num
                    num_trajs += 1

                    self.agent.infer_posterior(self.agent.context)
        else:
            while num_transitions < self.num_steps_per_eval:
                if self.seed_sample:
                    path, num = self.seedsampler.obtain_samples(deterministic=self.eval_deterministic,
                                                        max_samples=self.num_steps_per_eval - num_transitions,
                                                        max_trajs=1, accum_context=True)
                else:
                    path, num = self.sampler.obtain_samples(deterministic=self.eval_deterministic,
                                                                max_samples=self.num_steps_per_eval - num_transitions,
                                                                max_trajs=1, accum_context=True)
                paths += path
                num_transitions += num
                num_trajs += 1
                if num_trajs >= self.num_exp_traj_eval:
                    self.agent.infer_posterior(self.agent.context)

        if self.sparse_rewards:
            for p in paths:
                sparse_rewards = np.stack(e['sparse_reward'] for e in p['env_infos']).reshape(-1, 1)
                p['rewards'] = sparse_rewards

        goal = self.env._goal
        for path in paths:
            path['goal'] = goal # goal
        if hasattr(self.env,"_pitfall"):
            pitfall = self.env._pitfall
            for path in paths:
                path['pitfall'] = pitfall

        # save the paths for visualization, only useful for point mass
        if self.dump_eval_paths:
            logger.save_extra_data(paths, path='eval_trajectories/task{}-epoch{}-run{}'.format(idx, epoch, run))

        return paths

    def _do_eval(self, indices, epoch):
        final_returns = []
        online_returns = []
        for idx in indices:
            all_rets = []
            for r in range(self.num_evals):
                paths = self.collect_paths(idx, epoch, r)
                all_rets.append([eval_util.get_average_returns([p]) for p in paths])
            final_returns.append(np.mean([np.mean(a) for a in all_rets]))
            # record online returns for the first n trajectories
            n = min([len(a) for a in all_rets])
            all_rets = [a[:n] for a in all_rets]
            all_rets = np.mean(np.stack(all_rets), axis=0) # avg return per nth rollout
            online_returns.append(all_rets)
        n = min([len(t) for t in online_returns])
        online_returns = [t[:n] for t in online_returns]
        return final_returns, online_returns

    def evaluate(self, epoch):
        if self.eval_statistics is None:
            self.eval_statistics = OrderedDict()

        ### sample trajectories from prior for debugging / visualization
        if self.dump_eval_paths:
            # 100 arbitrarily chosen for visualizations of point_robot trajectories
            # just want stochasticity of z, not the policy
            self.agent.clear_z()
            if not self.use_SMM:
                prior_paths, _ = self.sampler.obtain_samples(deterministic=self.eval_deterministic,
                                                             max_samples=self.max_path_length * 20,
                                                        accum_context=False,
                                                        resample=1)
            else:
                prior_paths, _ = self.smm_sampler.obtain_samples(
                                                             max_samples=self.max_path_length * 20,
                                                             )
            logger.save_extra_data(prior_paths, path='eval_trajectories/prior-epoch{}'.format(epoch))

        ### train tasks
        # eval on a subset of train tasks for speed
        indices = np.random.choice(self.train_tasks, len(self.eval_tasks))
        eval_util.dprint('evaluating on {} train tasks'.format(len(indices)))
        ### eval train tasks with posterior sampled from the training replay buffer
        train_returns = []
        for idx in indices:
            self.task_idx = idx
            self.env.reset_task(idx)
            paths = []
            for _ in range(self.num_steps_per_eval // self.max_path_length):
                context = self.sample_context(idx)
                self.agent.infer_posterior(context)
                p, _ = self.sampler.obtain_samples(deterministic=self.eval_deterministic, max_samples=self.max_path_length,
                                                        accum_context=False,
                                                        max_trajs=1,
                                                        resample=np.inf)
                paths += p
            #for p in paths:
            #    print(p['actions'],p['rewards'])
            if self.sparse_rewards:
                for p in paths:
                    sparse_rewards = np.stack(e['sparse_reward'] for e in p['env_infos']).reshape(-1, 1)
                    p['rewards'] = sparse_rewards

            train_returns.append(eval_util.get_average_returns(paths))
        train_returns = np.mean(train_returns)
        ### eval train tasks with on-policy data to match eval of test tasks
        train_final_returns, train_online_returns = self._do_eval(indices, epoch)
        eval_util.dprint('train online returns')
        eval_util.dprint(train_online_returns)

        ### test tasks
        eval_util.dprint('evaluating on {} test tasks'.format(len(self.eval_tasks)))
        test_final_returns, test_online_returns = self._do_eval(self.eval_tasks, epoch)
        eval_util.dprint('test online returns')
        eval_util.dprint(test_online_returns)

        # save the final posterior
        self.agent.log_diagnostics(self.eval_statistics)

        #if hasattr(self.env, "log_diagnostics"):
        #    self.env.log_diagnostics(paths, prefix=None)

        avg_train_return = np.mean(train_final_returns)
        avg_test_return = np.mean(test_final_returns)
        avg_train_online_return = np.mean(np.stack(train_online_returns), axis=0)
        avg_test_online_return = np.mean(np.stack(test_online_returns), axis=0)
        self.eval_statistics['AverageTrainReturn_all_train_tasks'] = train_returns
        self.eval_statistics['AverageReturn_all_train_tasks'] = avg_train_return
        self.eval_statistics['AverageReturn_all_test_tasks'] = avg_test_return
        logger.save_extra_data(avg_train_online_return, path='online-train-epoch{}'.format(epoch))
        logger.save_extra_data(avg_test_online_return, path='online-test-epoch{}'.format(epoch))

        for key, value in self.eval_statistics.items():
            logger.record_tabular(key, value)
        self.eval_statistics = None

        if self.render_eval_paths:
            self.env.render_paths(paths)

        if self.plotter:
            self.plotter.draw()

    @abc.abstractmethod
    def training_mode(self, mode):
        """
        Set training mode to `mode`.
        :param mode: If True, training will happen (e.g. set the dropout
        probabilities to not all ones).
        """
        pass

    @abc.abstractmethod
    def _do_training(self):
        """
        Perform some update, e.g. perform one gradient step.
        :return:
        """
        pass

class ExpAlgorithm(metaclass=abc.ABCMeta):
    def __init__(
            self,
            env,
            agent,
            agent_exp,
            train_tasks,
            eval_tasks,
            encoder,
            meta_batch=64,
            num_iterations=100,
            num_train_steps_per_itr=1000,
            num_initial_steps=100,
            num_tasks_sample=100,
            num_steps_prior=100,
            num_steps_posterior=100,
            num_extra_rl_steps_posterior=100,
            num_evals=10,
            num_steps_per_eval=1000,
            batch_size=1024,
            embedding_batch_size=1024,
            embedding_mini_batch_size=1024,
            max_path_length=1000,
            discount=0.99,
            replay_buffer_size=1000000,
            reward_scale=1,
            num_exp_traj_eval=1,
            update_post_train=1,
            eval_deterministic=True,
            render=False,
            save_replay_buffer=False,
            save_algorithm=False,
            save_environment=False,
            render_eval_paths=False,
            dump_eval_paths=False,
            plotter=None,
            use_SMM=False,
            load_SMM =False,
            use_history=False,
            SMM_path=None,
            num_skills = 1,
            seed_sample=False,
            snail=False,
            meta_episode_len=10,
            num_trajs = 2,
            num_trajs_eval=1
    ):
        """
        :param env: training env
        :param agent: agent that is conditioned on a latent variable z that rl_algorithm is responsible for feeding in
        :param train_tasks: list of tasks used for training
        :param eval_tasks: list of tasks used for eval

        see default experiment config file for descriptions of the rest of the arguments
        """
        self.env = env
        self.agent = agent
        self.exploration_agent = agent_exp # Can potentially use a different policy purely for exploration rather than also solving tasks, currently not being used
        self.context_encoder = encoder
        self.train_tasks = train_tasks
        self.eval_tasks = eval_tasks
        self.meta_batch = meta_batch
        self.num_iterations = num_iterations
        self.num_train_steps_per_itr = num_train_steps_per_itr
        self.num_initial_steps = num_initial_steps
        self.num_tasks_sample = num_tasks_sample
        self.num_steps_prior = num_steps_prior
        self.num_steps_posterior = num_steps_posterior
        self.num_extra_rl_steps_posterior = num_extra_rl_steps_posterior
        self.num_evals = num_evals
        self.num_steps_per_eval = num_steps_per_eval
        self.batch_size = batch_size
        self.embedding_batch_size = embedding_batch_size
        self.embedding_mini_batch_size = embedding_mini_batch_size
        self.max_path_length = max_path_length
        self.discount = discount
        self.replay_buffer_size = replay_buffer_size
        self.reward_scale = reward_scale
        self.update_post_train = update_post_train
        self.num_exp_traj_eval = num_exp_traj_eval
        self.eval_deterministic = eval_deterministic
        self.render = render
        self.save_replay_buffer = save_replay_buffer
        self.save_algorithm = save_algorithm
        self.save_environment = save_environment

        self.eval_statistics = None
        self.render_eval_paths = render_eval_paths
        self.dump_eval_paths = dump_eval_paths
        self.plotter = plotter
        self.use_SMM = use_SMM
        self.load_SMM = load_SMM
        self.use_history = use_history,
        self.SMM_path = SMM_path
        self.num_skills = num_skills
        self.seed_sample = seed_sample
        self.meta_episode_len = meta_episode_len
        self.num_trajs = num_trajs
        self.num_trajs_eval = num_trajs_eval

        self.sampler = InPlacePathSampler(
            env=env,
            policy=agent,
            max_path_length=self.max_path_length,
        )

        self.expsampler = ExpInPlacePathSampler(
            env=env,
            policy=self.exploration_agent,
            encoder=self.context_encoder,
            max_path_length=self.max_path_length,
        )


        # separate replay buffers for
        # - training RL update
        # - training encoder update
        self.replay_buffer = MultiTaskReplayBuffer(
                self.replay_buffer_size,
                env,
                self.train_tasks,
            )

        self.enc_replay_buffer = MultiTaskReplayBuffer(
            self.replay_buffer_size,
            env,
            self.train_tasks,
        )

        self._n_env_steps_total = 0
        self._n_train_steps_total = 0
        self._n_rollouts_total = 0
        self._do_train_time = 0
        self._epoch_start_time = None
        self._algo_start_time = None
        self._old_table_keys = None
        self._current_path_builder = PathBuilder()
        self._exploration_paths = []

    def make_exploration_policy(self, policy):
         return policy

    def make_eval_policy(self, policy):
        return policy

    def sample_task(self, is_eval=False):
        '''
        sample task randomly
        '''
        if is_eval:
            idx = np.random.randint(len(self.eval_tasks))
        else:
            idx = np.random.randint(len(self.train_tasks))
        return idx

    def train(self):
        '''
        meta-training loop
        '''
        self.pretrain()
        params = self.get_epoch_snapshot(-1)
        logger.save_itr_params(-1, params)
        gt.reset()
        gt.set_def_unique(False)
        self._current_path_builder = PathBuilder()

        # at each iteration, we first collect data from tasks, perform meta-updates, then try to evaluate
        for it_ in gt.timed_for(
                range(self.num_iterations),
                save_itrs=True,
        ):
            self._start_epoch(it_)
            self.training_mode(True)
            if it_ == 0:
                print('collecting initial pool of data for train and eval')
                # temp for evaluating
                for idx in self.train_tasks:
                    self.task_idx = idx
                    self.env.reset_task(idx)
                    for _ in range(self.num_trajs):
                        self.collect_data_exp(self.meta_episode_len)
                    self.collect_data(self.num_initial_steps, 1, np.inf,add_to_enc_buffer=True)
            # Sample data from train tasks.
            for i in range(self.num_tasks_sample):
                idx = np.random.randint(len(self.train_tasks))
                self.task_idx = idx
                self.env.reset_task(idx)
                if (it_+1)%5==0:
                    self.enc_replay_buffer.task_buffers[idx].clear()
                for _ in range(self.num_trajs):
                    self.collect_data_exp(self.meta_episode_len)
                if self.num_steps_prior > 0:
                    self.collect_data(self.num_steps_prior, 1, np.inf,add_to_enc_buffer=True)
                    # collect some trajectories with z ~ posterior
                if self.num_steps_posterior > 0:
                    self.collect_data(self.num_steps_posterior, 1, self.update_post_train,add_to_enc_buffer=True)
                    # even if encoder is trained only on samples from the prior, the policy needs to learn to handle z ~ posterior
                if self.num_extra_rl_steps_posterior > 0:
                    self.collect_data(self.num_extra_rl_steps_posterior, 1, self.update_post_train,)
            print('collect over')

            # Sample train tasks and compute gradient updates on parameters.
            for train_step in range(self.num_train_steps_per_itr):
                indices = np.random.choice(self.train_tasks, self.meta_batch)
                self._do_training(indices)
                self._n_train_steps_total += 1
            gt.stamp('train')

            self.training_mode(False)

            # eval
            self._try_to_eval(it_)
            gt.stamp('eval')

            self._end_epoch()

    def pretrain(self):
        """
        Do anything before the main training phase.
        """
        pass

    def sample_eval(self,indices, context):
        reward = torch.zeros(context.shape[0],1,1).cuda()
        rem = 0
        for indice in indices:
            self.env.reset_task(indice)
            context_i = context[rem,...]
            context_i = torch.unsqueeze(context_i,0)
            self.agent.clear_z()
            self.agent.infer_posterior(context_i)
            path,_ = self.sampler.obtain_samples(deterministic=self.eval_deterministic, max_samples=self.max_path_length*5,resample=1)
            reward[rem] = eval_util.get_average_returns(path)
            rem = rem + 1
        return reward



    def collect_data(self, num_samples, resample_z_rate, update_posterior_rate, add_to_enc_buffer=False):
        '''
        get trajectories from current env in batch mode with given policy
        collect complete trajectories until the number of collected transitions >= num_samples
        :param agent: policy to rollout
        :param num_samples: total number of transitions to sample
        :param resample_z_rate: how often to resample latent context z (in units of trajectories)
        :param update_posterior_rate: how often to update q(z | c) from which z is sampled (in units of trajectories)
        :param add_to_enc_buffer: whether to add collected data to encoder replay buffer
        '''
        # start from the prior
        self.agent.clear_z()

        num_transitions = 0
        while num_transitions < num_samples:
            paths, n_samples = self.sampler.obtain_samples(max_samples=num_samples - num_transitions,
                                                           max_trajs=update_posterior_rate,
                                                           accum_context=False,
                                                           resample=resample_z_rate)
            num_transitions += n_samples
            self.replay_buffer.add_paths(self.task_idx, paths)
            if add_to_enc_buffer:
                self.enc_replay_buffer.add_paths(self.task_idx, paths)
            if update_posterior_rate != np.inf:
                context, context_unbatched = self.sample_context(self.task_idx)
                self.agent.infer_posterior(context)
        self._n_env_steps_total += num_transitions
        gt.stamp('sample')

    def collect_data_exp(self, num_episodes):
        '''
        get trajectories from current env in batch mode with given policy
        collect complete trajectories until the number of collected transitions >= num_samples

        :param agent: policy to rollout
        :param num_samples: total number of transitions to sample
        :param resample_z_rate: how often to resample latent context z (in units of trajectories)
        :param update_posterior_rate: how often to update q(z | c) from which z is sampled (in units of trajectories)
        :param add_to_enc_buffer: whether to add collected data to encoder replay buffer
        '''
        # start from the prior


        paths, n_samples = self.expsampler.obtain_samples(max_trajs=num_episodes)


        self.enc_replay_buffer.add_paths(self.task_idx, paths)

        self._n_env_steps_total += n_samples
        gt.stamp('sample')


    def _try_to_eval(self, epoch):
        logger.save_extra_data(self.get_extra_data_to_save(epoch))
        if self._can_evaluate(epoch):
            self.evaluate(epoch,self.num_trajs)

            params = self.get_epoch_snapshot(epoch)
            logger.save_itr_params(epoch, params)
            table_keys = logger.get_table_key_set()
            if self._old_table_keys is not None:
                assert table_keys == self._old_table_keys, (
                    "Table keys cannot change from iteration to iteration."
                )
            self._old_table_keys = table_keys

            logger.record_tabular(
                "Number of train steps total",
                self._n_train_steps_total,
            )
            logger.record_tabular(
                "Number of env steps total",
                self._n_env_steps_total,
            )
            logger.record_tabular(
                "Number of rollouts total",
                self._n_rollouts_total,
            )

            times_itrs = gt.get_times().stamps.itrs
            train_time = times_itrs['train'][-1]
            if not self.use_SMM:
                sample_time = times_itrs['sample'][-1]
            else:
                sample_time = times_itrs['policy sample'][-1]
            eval_time = times_itrs['eval'][-1] if epoch > 0 else 0
            epoch_time = train_time + sample_time + eval_time
            total_time = gt.get_times().total

            logger.record_tabular('Train Time (s)', train_time)
            logger.record_tabular('(Previous) Eval Time (s)', eval_time)
            logger.record_tabular('Sample Time (s)', sample_time)
            logger.record_tabular('Epoch Time (s)', epoch_time)
            logger.record_tabular('Total Train Time (s)', total_time)

            logger.record_tabular("Epoch", epoch)
            logger.dump_tabular(with_prefix=False, with_timestamp=False)
        else:
            logger.log("Skipping eval for now.")

    def _can_evaluate(self,epoch):
        """
        One annoying thing about the logger table is that the keys at each
        iteration need to be the exact same. So unless you can compute
        everything, skip evaluation.

        A common example for why you might want to skip evaluation is that at
        the beginning of training, you may not have enough data for a
        validation and training set.

        :return:
        """
        # eval collects its own context, so can eval any time
        return True #if (epoch+1)%5==0 else False

    def _can_train(self):
        return all([self.replay_buffer.num_steps_can_sample(idx) >= self.batch_size for idx in self.train_tasks])

    def _get_action_and_info(self, agent, observation):
        """
        Get an action to take in the environment.
        :param observation:
        :return:
        """
        agent.set_num_steps_total(self._n_env_steps_total)
        return agent.get_action(observation,)

    def _start_epoch(self, epoch):
        self._epoch_start_time = time.time()
        self._exploration_paths = []
        self._do_train_time = 0
        logger.push_prefix('Iteration #%d | ' % epoch)

    def _end_epoch(self):
        logger.log("Epoch Duration: {0}".format(
            time.time() - self._epoch_start_time
        ))
        logger.log("Started Training: {0}".format(self._can_train()))
        logger.pop_prefix()

    ##### Snapshotting utils #####
    def get_epoch_snapshot(self, epoch):
        data_to_save = dict(
            epoch=epoch,
            exploration_policy=self.exploration_policy,
        )
        if self.save_environment:
            data_to_save['env'] = self.training_env
        return data_to_save

    def get_extra_data_to_save(self, epoch):
        """
        Save things that shouldn't be saved every snapshot but rather
        overwritten every time.
        :param epoch:
        :return:
        """
        if self.render:
            self.training_env.render(close=True)
        data_to_save = dict(
            epoch=epoch,
        )
        if self.save_environment:
            data_to_save['env'] = self.training_env
        if self.save_replay_buffer:
            data_to_save['replay_buffer'] = self.replay_buffer
        if self.save_algorithm:
            data_to_save['algorithm'] = self
        return data_to_save

    def collect_paths(self, idx, epoch, run):
        self.task_idx = idx
        self.env.reset_task(idx)

        self.agent.clear_z()
        paths = []
        num_transitions = 0
        num_trajs = 0

        path, num = self.expsampler.obtain_samples(deterministic=self.eval_deterministic,
                                                   max_trajs=self.num_exp_traj_eval, accum_context_for_agent=True, context_agent = self.agent,split=True)
        num_transitions += num
        num_trajs +=self.num_exp_traj_eval
        paths+=path

        while num_transitions < self.num_steps_per_eval:

            path, num = self.sampler.obtain_samples(deterministic=self.eval_deterministic, max_samples=self.num_steps_per_eval - num_transitions, max_trajs=1, accum_context=True)
            paths += path
            num_transitions += num
            num_trajs += 1
            self.agent.infer_posterior(self.agent.context)

        if self.sparse_rewards:
            for p in paths:
                sparse_rewards = np.stack(e['sparse_reward'] for e in p['env_infos']).reshape(-1, 1)
                p['rewards'] = sparse_rewards

        goal = self.env._goal
        for path in paths:
            path['goal'] = goal # goal

        # save the paths for visualization, only useful for point mass
        if self.dump_eval_paths:
            logger.save_extra_data(paths, path='eval_trajectories/task{}-epoch{}-run{}'.format(idx, epoch, run))

        return paths

    def _do_eval(self, indices, epoch):
        final_returns = []
        online_returns = []
        for idx in indices:
            all_rets = []
            for r in range(self.num_evals):
                paths = self.collect_paths(idx, epoch, r)
                all_rets.append([eval_util.get_average_returns([p]) for p in paths])
            final_returns.append(np.mean([np.mean(a) for a in all_rets]))
            # record online returns for the first n trajectories
            n = min([len(a) for a in all_rets])
            all_rets = [a[:n] for a in all_rets]
            all_rets = np.mean(np.stack(all_rets), axis=0) # avg return per nth rollout
            online_returns.append(all_rets)
        n = min([len(t) for t in online_returns])
        online_returns = [t[:n] for t in online_returns]
        return final_returns, online_returns

    def evaluate(self, epoch,num_trajs):
        if self.eval_statistics is None:
            self.eval_statistics = OrderedDict()

        ### sample trajectories from prior for debugging / visualization
        if self.dump_eval_paths:
            # 100 arbitrarily chosen for visualizations of point_robot trajectories
            # just want stochasticity of z, not the policy
            self.agent.clear_z()
            #if not self.use_SMM:
            prior_paths, _ = self.sampler.obtain_samples(deterministic=self.eval_deterministic,
                                                         max_samples=self.max_path_length * 20,
                                                         accum_context=False,
                                                         resample=1)
            #else:
            #    prior_paths, _ = self.smm_sampler.obtain_samples(
            #                                                 max_samples=self.max_path_length * 20,
            #                                                 )
            logger.save_extra_data(prior_paths, path='eval_trajectories/prior-epoch{}'.format(epoch))

        ### train tasks
        # eval on a subset of train tasks for speed
        indices = np.random.choice(self.train_tasks, len(self.eval_tasks))
        eval_util.dprint('evaluating on {} train tasks'.format(len(indices)))
        ### eval train tasks with posterior sampled from the training replay buffer
        train_returns = []
        for idx in indices:
            self.task_idx = idx
            self.env.reset_task(idx)
            paths = []
            for _ in range(self.num_steps_per_eval // self.max_path_length):
                context, context_unbatched = self.sample_context(idx)
                self.agent.infer_posterior(context)
                p, _ = self.sampler.obtain_samples(deterministic=self.eval_deterministic,
                                                   max_samples=self.max_path_length,
                                                   accum_context=False,
                                                   max_trajs=1,
                                                   resample=np.inf)
                paths += p

            if self.sparse_rewards:
                for p in paths:
                    sparse_rewards = np.stack(e['sparse_reward'] for e in p['env_infos']).reshape(-1, 1)
                    p['rewards'] = sparse_rewards

            train_returns.append(eval_util.get_average_returns(paths))
        train_returns = np.mean(train_returns)
        ### eval train tasks with on-policy data to match eval of test tasks
        train_final_returns, train_online_returns = self._do_eval(indices, epoch)
        eval_util.dprint('train online returns')
        eval_util.dprint(train_online_returns)

        ### test tasks
        eval_util.dprint('evaluating on {} test tasks'.format(len(self.eval_tasks)))
        test_final_returns, test_online_returns = self._do_eval(self.eval_tasks, epoch)
        eval_util.dprint('test online returns')
        eval_util.dprint(test_online_returns)

        # save the final posterior
        self.agent.log_diagnostics(self.eval_statistics)

        if hasattr(self.env, "log_diagnostics"):
            self.env.log_diagnostics(paths, prefix=None)

        avg_train_return = np.mean(train_final_returns)
        avg_test_return = np.mean(test_final_returns)
        avg_train_online_return = np.mean(np.stack(train_online_returns), axis=0)
        avg_test_online_return = np.mean(np.stack(test_online_returns), axis=0)
        self.eval_statistics['AverageTrainReturn_all_train_tasks'] = train_returns
        self.eval_statistics['AverageReturn_all_train_tasks'] = avg_train_return
        self.eval_statistics['AverageReturn_all_test_tasks'] = avg_test_return
        logger.save_extra_data(avg_train_online_return, path='online-train-epoch{}'.format(epoch))
        logger.save_extra_data(avg_test_online_return, path='online-test-epoch{}'.format(epoch))

        for key, value in self.eval_statistics.items():
            logger.record_tabular(key, value)
        self.eval_statistics = None

        if self.render_eval_paths:
            self.env.render_paths(paths)

        if self.plotter:
            self.plotter.draw()

    @abc.abstractmethod
    def training_mode(self, mode):
        """
        Set training mode to `mode`.
        :param mode: If True, training will happen (e.g. set the dropout
        probabilities to not all ones).
        """
        pass

    @abc.abstractmethod
    def _do_training(self):
        """
        Perform some update, e.g. perform one gradient step.
        :return:
        """
        pass

class ExpAlgorithmSimple(metaclass=abc.ABCMeta):
    def __init__(
            self,
            env,
            agent,
            agent_exp,
            train_tasks,
            eval_tasks,
            encoder,
            meta_batch=64,
            num_iterations=100,
            num_train_steps_per_itr=1000,
            num_initial_steps=100,
            num_tasks_sample=100,
            num_steps_prior=100,
            num_steps_posterior=100,
            num_extra_rl_steps_posterior=100,
            num_evals=10,
            num_steps_per_eval=1000,
            batch_size=1024,
            embedding_batch_size=1024,
            embedding_mini_batch_size=1024,
            max_path_length=1000,
            discount=0.99,
            replay_buffer_size=1000000,
            reward_scale=1,
            num_exp_traj_eval=1,
            update_post_train=1,
            eval_deterministic=True,
            render=False,
            save_replay_buffer=False,
            save_algorithm=False,
            save_environment=False,
            render_eval_paths=False,
            dump_eval_paths=False,
            plotter=None,
            use_SMM=False,
            load_SMM =False,
            use_history=False,
            SMM_path=None,
            num_skills = 1,
            seed_sample=False,
            snail=False,
            meta_episode_len=10,
            num_trajs = 2,
            num_trajs_eval=1
    ):
        """
        :param env: training env
        :param agent: agent that is conditioned on a latent variable z that rl_algorithm is responsible for feeding in
        :param train_tasks: list of tasks used for training
        :param eval_tasks: list of tasks used for eval

        see default experiment config file for descriptions of the rest of the arguments
        """
        self.env = env
        self.agent = agent
        self.exploration_agent = agent_exp # Can potentially use a different policy purely for exploration rather than also solving tasks, currently not being used
        self.context_encoder = encoder
        self.train_tasks = train_tasks
        self.eval_tasks = eval_tasks
        self.meta_batch = meta_batch
        self.num_iterations = num_iterations
        self.num_train_steps_per_itr = num_train_steps_per_itr
        self.num_initial_steps = num_initial_steps
        self.num_tasks_sample = num_tasks_sample
        self.num_steps_prior = num_steps_prior
        self.num_steps_posterior = num_steps_posterior
        self.num_extra_rl_steps_posterior = num_extra_rl_steps_posterior
        self.num_evals = num_evals
        self.num_steps_per_eval = num_steps_per_eval
        self.batch_size = batch_size
        self.embedding_batch_size = embedding_batch_size
        self.embedding_mini_batch_size = embedding_mini_batch_size
        self.max_path_length = max_path_length
        self.discount = discount
        self.replay_buffer_size = replay_buffer_size
        self.reward_scale = reward_scale
        self.update_post_train = update_post_train
        self.num_exp_traj_eval = num_exp_traj_eval
        self.eval_deterministic = eval_deterministic
        self.render = render
        self.save_replay_buffer = save_replay_buffer
        self.save_algorithm = save_algorithm
        self.save_environment = save_environment

        self.eval_statistics = None
        self.render_eval_paths = render_eval_paths
        self.dump_eval_paths = dump_eval_paths
        self.plotter = plotter
        self.use_SMM = use_SMM
        self.load_SMM = load_SMM
        self.use_history = use_history,
        self.SMM_path = SMM_path
        self.num_skills = num_skills
        self.seed_sample = seed_sample
        self.meta_episode_len = meta_episode_len
        self.num_trajs = num_trajs
        self.num_trajs_eval = num_trajs_eval

        self.sampler = InPlacePathSampler(
            env=env,
            policy=agent,
            max_path_length=self.max_path_length,
        )

        self.expsampler = ExpInPlacePathSamplerSimple(
            env=env,
            policy=self.exploration_agent,
            encoder=self.context_encoder,
            max_path_length=self.max_path_length,
        )

        self.seedsampler = SeedInPlacePathSampler(
            env=env,
            policy=agent,
            max_path_length=self.max_path_length,
        )
        # separate replay buffers for
        # - training RL update
        # - training encoder update
        self.replay_buffer = MultiTaskReplayBuffer(
                self.replay_buffer_size,
                env,
                self.train_tasks,
            )

        self.enc_replay_buffer = MultiTaskReplayBuffer(
            self.replay_buffer_size,
            env,
            self.train_tasks,
        )

        self._n_env_steps_total = 0
        self._n_train_steps_total = 0
        self._n_rollouts_total = 0
        self._do_train_time = 0
        self._epoch_start_time = None
        self._algo_start_time = None
        self._old_table_keys = None
        self._current_path_builder = PathBuilder()
        self._exploration_paths = []

    def make_exploration_policy(self, policy):
         return policy

    def make_eval_policy(self, policy):
        return policy

    def sample_task(self, is_eval=False):
        '''
        sample task randomly
        '''
        if is_eval:
            idx = np.random.randint(len(self.eval_tasks))
        else:
            idx = np.random.randint(len(self.train_tasks))
        return idx

    def train(self):
        '''
        meta-training loop
        '''
        self.pretrain()
        params = self.get_epoch_snapshot(-1)
        logger.save_itr_params(-1, params)
        gt.reset()
        gt.set_def_unique(False)
        self._current_path_builder = PathBuilder()

        # at each iteration, we first collect data from tasks, perform meta-updates, then try to evaluate
        for it_ in gt.timed_for(
                range(self.num_iterations),
                save_itrs=True,
        ):
            self._start_epoch(it_)
            self.training_mode(True)
            if it_ == 0:
                print('collecting initial pool of data for train and eval')
                # temp for evaluating
                for idx in self.train_tasks:
                    self.task_idx = idx
                    self.env.reset_task(idx)
                    for _ in range(self.num_trajs):
                        self.collect_data_exp(self.meta_episode_len)
                    if self.seed_sample:
                        self.collect_data_seed(self.num_initial_steps, 1, np.inf, add_to_enc_buffer=True)
                    else:
                        self.collect_data(self.num_initial_steps, 1, np.inf,add_to_enc_buffer=True)
            # Sample data from train tasks.
            for i in range(self.num_tasks_sample):
                idx = np.random.randint(len(self.train_tasks))
                self.task_idx = idx
                self.env.reset_task(idx)
                if (it_+1)%5==0:
                    self.enc_replay_buffer.task_buffers[idx].clear()
                for _ in range(self.num_trajs):
                    self.collect_data_exp(self.meta_episode_len)
                if self.num_steps_prior > 0:
                    if self.seed_sample:
                        self.collect_data_seed(self.num_steps_prior, 1, np.inf, add_to_enc_buffer=True)
                    else:
                        self.collect_data(self.num_steps_prior, 1, np.inf,add_to_enc_buffer=True)
                    # collect some trajectories with z ~ posterior
                if self.num_steps_posterior > 0:
                    if self.seed_sample:
                        self.collect_data_seed(self.num_steps_posterior, 1, self.update_post_train, add_to_enc_buffer=True)
                    else:
                        self.collect_data(self.num_steps_posterior, 1, self.update_post_train,add_to_enc_buffer=True)
                    # even if encoder is trained only on samples from the prior, the policy needs to learn to handle z ~ posterior
                if self.num_extra_rl_steps_posterior > 0:
                    if self.seed_sample:
                        self.collect_data_seed(self.num_extra_rl_steps_posterior, 1, self.update_post_train,)
                    else:
                        self.collect_data(self.num_extra_rl_steps_posterior, 1, self.update_post_train,)
            print('collect over')

            # Sample train tasks and compute gradient updates on parameters.
            for train_step in range(self.num_train_steps_per_itr):
                indices = np.random.choice(self.train_tasks, self.meta_batch)
                self._do_training(indices)
                self._n_train_steps_total += 1
            gt.stamp('train')

            self.training_mode(False)

            # eval
            self._try_to_eval(it_)
            gt.stamp('eval')

            self._end_epoch()

    def pretrain(self):
        """
        Do anything before the main training phase.
        """
        pass

    def sample_eval(self,indices, context):
        reward = torch.zeros(context.shape[0],1,1).cuda()
        rem = 0
        for indice in indices:
            self.env.reset_task(indice)
            context_i = context[rem,...]
            context_i = torch.unsqueeze(context_i,0)
            self.agent.clear_z()
            self.agent.infer_posterior(context_i)
            path,_ = self.sampler.obtain_samples(deterministic=self.eval_deterministic, max_samples=self.max_path_length*5,resample=1)
            reward[rem] = eval_util.get_average_returns(path)
            rem = rem + 1
        return reward



    def collect_data(self, num_samples, resample_z_rate, update_posterior_rate, add_to_enc_buffer=False):
        '''
        get trajectories from current env in batch mode with given policy
        collect complete trajectories until the number of collected transitions >= num_samples
        :param agent: policy to rollout
        :param num_samples: total number of transitions to sample
        :param resample_z_rate: how often to resample latent context z (in units of trajectories)
        :param update_posterior_rate: how often to update q(z | c) from which z is sampled (in units of trajectories)
        :param add_to_enc_buffer: whether to add collected data to encoder replay buffer
        '''
        # start from the prior
        self.agent.clear_z()

        num_transitions = 0
        while num_transitions < num_samples:
            paths, n_samples = self.sampler.obtain_samples(max_samples=num_samples - num_transitions,
                                                           max_trajs=update_posterior_rate,
                                                           accum_context=False,
                                                           resample=resample_z_rate)
            num_transitions += n_samples
            self.replay_buffer.add_paths(self.task_idx, paths)
            if add_to_enc_buffer:
                self.enc_replay_buffer.add_paths(self.task_idx, paths)
            if update_posterior_rate != np.inf:
                context, context_unbatched = self.sample_context(self.task_idx)
                self.agent.infer_posterior(context)
        self._n_env_steps_total += num_transitions
        gt.stamp('sample')

    def collect_data_exp(self, num_episodes):
        '''
        get trajectories from current env in batch mode with given policy
        collect complete trajectories until the number of collected transitions >= num_samples

        :param agent: policy to rollout
        :param num_samples: total number of transitions to sample
        :param resample_z_rate: how often to resample latent context z (in units of trajectories)
        :param update_posterior_rate: how often to update q(z | c) from which z is sampled (in units of trajectories)
        :param add_to_enc_buffer: whether to add collected data to encoder replay buffer
        '''
        # start from the prior


        paths, n_samples = self.expsampler.obtain_samples(max_trajs=num_episodes)


        self.enc_replay_buffer.add_paths(self.task_idx, paths)

        self._n_env_steps_total += n_samples
        gt.stamp('sample')

    def collect_data_seed(self, num_samples, resample_z_rate, update_posterior_rate, add_to_enc_buffer=True,add_to_policy_buffer=True,accumulate_context=True):
        self.agent.clear_z()
        num_transitions = 0
        while num_transitions < num_samples:
            paths, n_samples = self.seedsampler.obtain_samples(max_samples=num_samples - num_transitions,
                                                                max_trajs=1,
                                                                accum_context=accumulate_context
                                                                )
            num_transitions += n_samples
            if add_to_policy_buffer:
                self.replay_buffer.add_paths(self.task_idx, paths)
            if add_to_enc_buffer:
                self.enc_replay_buffer.add_paths(self.task_idx, paths)
            #if update_posterior_rate != np.inf:
            #    context = self.prepare_context(self.task_idx)
            #    self.agent.infer_posterior(context)
        self._n_env_steps_total += num_transitions
        gt.stamp('sample')
    def _try_to_eval(self, epoch):
        logger.save_extra_data(self.get_extra_data_to_save(epoch))
        if self._can_evaluate(epoch):
            self.evaluate(epoch,self.num_trajs)

            params = self.get_epoch_snapshot(epoch)
            logger.save_itr_params(epoch, params)
            table_keys = logger.get_table_key_set()
            if self._old_table_keys is not None:
                assert table_keys == self._old_table_keys, (
                    "Table keys cannot change from iteration to iteration."
                )
            self._old_table_keys = table_keys

            logger.record_tabular(
                "Number of train steps total",
                self._n_train_steps_total,
            )
            logger.record_tabular(
                "Number of env steps total",
                self._n_env_steps_total,
            )
            logger.record_tabular(
                "Number of rollouts total",
                self._n_rollouts_total,
            )

            times_itrs = gt.get_times().stamps.itrs
            train_time = times_itrs['train'][-1]
            if not self.use_SMM:
                sample_time = times_itrs['sample'][-1]
            else:
                sample_time = times_itrs['policy sample'][-1]
            eval_time = times_itrs['eval'][-1] if epoch > 0 else 0
            epoch_time = train_time + sample_time + eval_time
            total_time = gt.get_times().total

            logger.record_tabular('Train Time (s)', train_time)
            logger.record_tabular('(Previous) Eval Time (s)', eval_time)
            logger.record_tabular('Sample Time (s)', sample_time)
            logger.record_tabular('Epoch Time (s)', epoch_time)
            logger.record_tabular('Total Train Time (s)', total_time)

            logger.record_tabular("Epoch", epoch)
            logger.dump_tabular(with_prefix=False, with_timestamp=False)
        else:
            logger.log("Skipping eval for now.")

    def _can_evaluate(self,epoch):
        """
        One annoying thing about the logger table is that the keys at each
        iteration need to be the exact same. So unless you can compute
        everything, skip evaluation.

        A common example for why you might want to skip evaluation is that at
        the beginning of training, you may not have enough data for a
        validation and training set.

        :return:
        """
        # eval collects its own context, so can eval any time
        return True #if (epoch+1)%5==0 else False

    def _can_train(self):
        return all([self.replay_buffer.num_steps_can_sample(idx) >= self.batch_size for idx in self.train_tasks])

    def _get_action_and_info(self, agent, observation):
        """
        Get an action to take in the environment.
        :param observation:
        :return:
        """
        agent.set_num_steps_total(self._n_env_steps_total)
        return agent.get_action(observation,)

    def _start_epoch(self, epoch):
        self._epoch_start_time = time.time()
        self._exploration_paths = []
        self._do_train_time = 0
        logger.push_prefix('Iteration #%d | ' % epoch)

    def _end_epoch(self):
        logger.log("Epoch Duration: {0}".format(
            time.time() - self._epoch_start_time
        ))
        logger.log("Started Training: {0}".format(self._can_train()))
        logger.pop_prefix()

    ##### Snapshotting utils #####
    def get_epoch_snapshot(self, epoch):
        data_to_save = dict(
            epoch=epoch,
            exploration_policy=self.exploration_policy,
        )
        if self.save_environment:
            data_to_save['env'] = self.training_env
        return data_to_save

    def get_extra_data_to_save(self, epoch):
        """
        Save things that shouldn't be saved every snapshot but rather
        overwritten every time.
        :param epoch:
        :return:
        """
        if self.render:
            self.training_env.render(close=True)
        data_to_save = dict(
            epoch=epoch,
        )
        if self.save_environment:
            data_to_save['env'] = self.training_env
        if self.save_replay_buffer:
            data_to_save['replay_buffer'] = self.replay_buffer
        if self.save_algorithm:
            data_to_save['algorithm'] = self
        return data_to_save

    def collect_paths(self, idx, epoch, run):
        self.task_idx = idx
        self.env.reset_task(idx)

        self.agent.clear_z()
        paths = []
        num_transitions = 0
        num_trajs = 0

        path, num = self.expsampler.obtain_samples(deterministic=self.eval_deterministic,
                                                   max_trajs=self.num_exp_traj_eval, accum_context_for_agent=True, context_agent = self.agent,split=True)
        num_transitions += num
        num_trajs +=self.num_exp_traj_eval
        paths+=path

        while num_transitions < self.num_steps_per_eval:

            path, num = self.sampler.obtain_samples(deterministic=self.eval_deterministic, max_samples=self.num_steps_per_eval - num_transitions, max_trajs=1, accum_context=True)
            paths += path
            num_transitions += num
            num_trajs += 1
            self.agent.infer_posterior(self.agent.context)

        if self.sparse_rewards:
            for p in paths:
                sparse_rewards = np.stack(e['sparse_reward'] for e in p['env_infos']).reshape(-1, 1)
                p['rewards'] = sparse_rewards

        goal = self.env._goal
        for path in paths:
            path['goal'] = goal # goal

        # save the paths for visualization, only useful for point mass
        if self.dump_eval_paths:
            logger.save_extra_data(paths, path='eval_trajectories/task{}-epoch{}-run{}'.format(idx, epoch, run))

        return paths

    def _do_eval(self, indices, epoch):
        final_returns = []
        online_returns = []
        for idx in indices:
            all_rets = []
            for r in range(self.num_evals):
                paths = self.collect_paths(idx, epoch, r)
                all_rets.append([eval_util.get_average_returns([p]) for p in paths])
            final_returns.append(np.mean([np.mean(a) for a in all_rets]))
            # record online returns for the first n trajectories
            n = min([len(a) for a in all_rets])
            all_rets = [a[:n] for a in all_rets]
            all_rets = np.mean(np.stack(all_rets), axis=0) # avg return per nth rollout
            online_returns.append(all_rets)
        n = min([len(t) for t in online_returns])
        online_returns = [t[:n] for t in online_returns]
        return final_returns, online_returns

    def evaluate(self, epoch,num_trajs):
        if self.eval_statistics is None:
            self.eval_statistics = OrderedDict()

        ### sample trajectories from prior for debugging / visualization
        if self.dump_eval_paths:
            # 100 arbitrarily chosen for visualizations of point_robot trajectories
            # just want stochasticity of z, not the policy
            self.agent.clear_z()
            #if not self.use_SMM:
            prior_paths, _ = self.sampler.obtain_samples(deterministic=self.eval_deterministic,
                                                         max_samples=self.max_path_length * 20,
                                                         accum_context=False,
                                                         resample=1)
            #else:
            #    prior_paths, _ = self.smm_sampler.obtain_samples(
            #                                                 max_samples=self.max_path_length * 20,
            #                                                 )
            logger.save_extra_data(prior_paths, path='eval_trajectories/prior-epoch{}'.format(epoch))

        ### train tasks
        # eval on a subset of train tasks for speed
        indices = np.random.choice(self.train_tasks, len(self.eval_tasks))
        eval_util.dprint('evaluating on {} train tasks'.format(len(indices)))
        ### eval train tasks with posterior sampled from the training replay buffer
        train_returns = []
        for idx in indices:
            self.task_idx = idx
            self.env.reset_task(idx)
            paths = []
            for _ in range(self.num_steps_per_eval // self.max_path_length):
                context, context_unbatched = self.sample_context(idx)
                self.agent.infer_posterior(context)
                p, _ = self.sampler.obtain_samples(deterministic=self.eval_deterministic,
                                                   max_samples=self.max_path_length,
                                                   accum_context=False,
                                                   max_trajs=1,
                                                   resample=np.inf)
                paths += p

            if self.sparse_rewards:
                for p in paths:
                    sparse_rewards = np.stack(e['sparse_reward'] for e in p['env_infos']).reshape(-1, 1)
                    p['rewards'] = sparse_rewards

            train_returns.append(eval_util.get_average_returns(paths))
        train_returns = np.mean(train_returns)
        ### eval train tasks with on-policy data to match eval of test tasks
        train_final_returns, train_online_returns = self._do_eval(indices, epoch)
        eval_util.dprint('train online returns')
        eval_util.dprint(train_online_returns)

        ### test tasks
        eval_util.dprint('evaluating on {} test tasks'.format(len(self.eval_tasks)))
        test_final_returns, test_online_returns = self._do_eval(self.eval_tasks, epoch)
        eval_util.dprint('test online returns')
        eval_util.dprint(test_online_returns)

        # save the final posterior
        self.agent.log_diagnostics(self.eval_statistics)

        if hasattr(self.env, "log_diagnostics"):
            self.env.log_diagnostics(paths, prefix=None)

        avg_train_return = np.mean(train_final_returns)
        avg_test_return = np.mean(test_final_returns)
        avg_train_online_return = np.mean(np.stack(train_online_returns), axis=0)
        avg_test_online_return = np.mean(np.stack(test_online_returns), axis=0)
        self.eval_statistics['AverageTrainReturn_all_train_tasks'] = train_returns
        self.eval_statistics['AverageReturn_all_train_tasks'] = avg_train_return
        self.eval_statistics['AverageReturn_all_test_tasks'] = avg_test_return
        logger.save_extra_data(avg_train_online_return, path='online-train-epoch{}'.format(epoch))
        logger.save_extra_data(avg_test_online_return, path='online-test-epoch{}'.format(epoch))

        for key, value in self.eval_statistics.items():
            logger.record_tabular(key, value)
        self.eval_statistics = None

        if self.render_eval_paths:
            self.env.render_paths(paths)

        if self.plotter:
            self.plotter.draw()

    @abc.abstractmethod
    def training_mode(self, mode):
        """
        Set training mode to `mode`.
        :param mode: If True, training will happen (e.g. set the dropout
        probabilities to not all ones).
        """
        pass

    @abc.abstractmethod
    def _do_training(self):
        """
        Perform some update, e.g. perform one gradient step.
        :return:
        """
        pass

class RLAlgorithm(metaclass=abc.ABCMeta):
    def __init__(
            self,
            env,
            exploration_policy: ExplorationPolicy,
            training_env=None,
            num_epochs=100,
            num_episodes=100,
            num_steps_per_epoch=10000,
            num_steps_per_eval=1000,
            num_updates_per_episode=50,
            num_updates_per_env_step=1,
            num_updates_per_epoch=None,
            batch_size=1024,
            max_path_length=1000,
            discount=0.99,
            replay_buffer_size=1000000,
            reward_scale=1,
            min_num_steps_before_training=None,
            render=False,
            save_replay_buffer=False,
            save_algorithm=False,
            save_environment=True,
            eval_sampler=None,
            eval_policy=None,
            replay_buffer=None,
            collection_mode='online',
    ):
        """
        Base class for RL Algorithms

        :param env: Environment used to evaluate.
        :param exploration_policy: Policy used to explore
        :param training_env: Environment used by the algorithm. By default, a
        copy of `env` will be made for training, so that training and
        evaluation are completely independent.
        :param num_epochs:
        :param num_episodes: Used by episodic training mode.
        :param num_steps_per_epoch:
        :param num_steps_per_eval:
        :param num_updates_per_env_step: Used by online training mode.
        :param num_updates_per_epoch: Used by batch training mode.
        :param num_updates_per_episode: Used by episodic training mode.
        :param batch_size:
        :param max_path_length:
        :param discount:
        :param replay_buffer_size:
        :param reward_scale:
        :param min_num_steps_before_training:
        :param render:
        :param save_replay_buffer:
        :param save_algorithm:
        :param save_environment:
        :param eval_sampler:
        :param eval_policy: Policy to evaluate with.
        :param replay_buffer:
        :param collection_mode: String determining how training happens
         - 'online': Train after every step taken in the environment.
         - 'batch': Train after every epoch.
        """
        assert collection_mode in ['online', 'batch', 'episodic']
        if collection_mode == 'batch':
            assert num_updates_per_epoch is not None

        self.training_env = training_env or pickle.loads(pickle.dumps(env))
        self.exploration_policy = exploration_policy
        self.num_epochs = num_epochs
        self.num_env_steps_per_epoch = num_steps_per_epoch
        self.num_steps_per_eval = num_steps_per_eval
        if collection_mode == 'online':
            self.num_updates_per_train_call = num_updates_per_env_step
        elif collection_mode == 'batch':
            self.num_updates_per_train_call = num_updates_per_epoch
        elif collection_mode == 'episodic':
            self.num_updates_per_train_call = num_updates_per_episode
            self.num_episodes = num_episodes
        else:
            raise TypeError("Invalid collection_mode: {}".format(
                collection_mode
            ))
        self.batch_size = batch_size
        self.max_path_length = max_path_length
        self.discount = discount
        self.replay_buffer_size = replay_buffer_size
        self.reward_scale = reward_scale
        self.render = render
        self.collection_mode = collection_mode
        self.save_replay_buffer = save_replay_buffer
        self.save_algorithm = save_algorithm
        self.save_environment = save_environment
        if min_num_steps_before_training is None:
            min_num_steps_before_training = self.num_env_steps_per_epoch
        self.min_num_steps_before_training = min_num_steps_before_training
        if eval_sampler is None:
            if eval_policy is None:
                eval_policy = exploration_policy
            eval_sampler = SMMInPlacePathSampler(
                env=env,
                policy=eval_policy,
                max_samples=self.num_steps_per_eval + self.max_path_length,
                max_path_length=self.max_path_length,
            )
        self.eval_policy = eval_policy
        self.eval_sampler = eval_sampler
        self.eval_statistics = OrderedDict()
        self.need_to_update_eval_statistics = True

        self.action_space = env.action_space
        self.obs_space = env.observation_space
        self.env = env
        if replay_buffer is None:
            replay_buffer = EnvReplayBuffer(
                self.replay_buffer_size,
                self.env,
            )
        self.replay_buffer = replay_buffer

        self._n_env_steps_total = 0
        self._n_train_steps_total = 0
        self._n_rollouts_total = 0
        self._do_train_time = 0
        self._epoch_start_time = None
        self._algo_start_time = None
        self._old_table_keys = None
        self._current_path_builder = PathBuilder()
        self._exploration_paths = []
        self.post_epoch_funcs = []

    def train(self, start_epoch=0):
        self.pretrain()
        if start_epoch == 0:
            params = self.get_epoch_snapshot(-1)
            logger.save_itr_params(-1, params)
        self.training_mode(False)
        self._n_env_steps_total = start_epoch * self.num_env_steps_per_epoch
        gt.reset()
        gt.set_def_unique(False)
        if self.collection_mode == 'online':
            self.train_online(start_epoch=start_epoch)
        elif self.collection_mode == 'batch':
            self.train_batch(start_epoch=start_epoch)
        elif self.collection_mode == 'episodic':
            self.train_episodic(start_episode=start_epoch)
        else:
            raise TypeError("Invalid collection_mode: {}".format(
                self.collection_mode
            ))

    def pretrain(self):
        pass

    def train_online(self, start_epoch=0):
        self._current_path_builder = PathBuilder()
        for epoch in gt.timed_for(
                range(start_epoch, self.num_epochs),
                save_itrs=True,
        ):
            self._start_epoch(epoch)
            set_to_train_mode(self.training_env)
            observation = self._start_new_rollout()
            for _ in range(self.num_env_steps_per_epoch):
                observation = self._take_step_in_env(observation)
                gt.stamp('sample')

                self._try_to_train()
                gt.stamp('train')

            set_to_eval_mode(self.env)
            self._try_to_eval(epoch)
            gt.stamp('eval')
            self._end_epoch(epoch)

    def train_batch(self, start_epoch):
        self._current_path_builder = PathBuilder()
        for epoch in gt.timed_for(
                range(start_epoch, self.num_epochs),
                save_itrs=True,
        ):
            self._start_epoch(epoch)
            set_to_train_mode(self.training_env)
            observation = self._start_new_rollout()
            # This implementation is rather naive. If you want to (e.g.)
            # parallelize data collection, this would be the place to do it.
            for _ in range(self.num_env_steps_per_epoch):
                observation = self._take_step_in_env(observation)
            gt.stamp('sample')

            self._try_to_train()
            gt.stamp('train')

            set_to_eval_mode(self.env)
            self._try_to_eval(epoch)
            gt.stamp('eval')
            self._end_epoch(epoch)

    def train_episodic(self, start_episode=0):
        self._current_path_builder = PathBuilder()
        eval_paths = []
        for episode in gt.timed_for(
                range(start_episode, self.num_episodes),
                save_itrs=True,
        ):
            self._start_epoch(episode)
            set_to_train_mode(self.training_env)
            observation = self._start_new_rollout()

            done = False
            while not done:
                action, agent_info = self._get_action_and_info(
                    observation,
                )
                if self.render:
                    self.training_env.render()
                next_ob, raw_reward, _terminal, env_info = (
                    self.training_env.step(action)
                )
                next_ob = self._proc_observation(next_ob)
                self._n_env_steps_total += 1
                reward = raw_reward * self.reward_scale
                terminal = np.array([_terminal])
                reward = np.array([reward])
                self._handle_step(
                    observation,
                    action,
                    reward,
                    next_ob,
                    terminal,
                    agent_info=agent_info,
                    env_info=env_info,
                )
                gt.stamp('sample')

                observation = next_ob
                done = (_terminal or len(self._current_path_builder) >= self.max_path_length)
            # Get path for the current episode.
            eval_path = self._current_path_builder.get_all_stacked()
            eval_paths.append(eval_path)

            # Empty self._current_path_builder, and perform other algorithm-specific handles.
            self._handle_rollout_ending()

            # Train policy.
            self._try_to_train()
            gt.stamp('train')

            # Evaluate policy.
            self.need_to_update_eval_statistics = False  # Evaluate regardless of whether we trained or not.
            set_to_eval_mode(self.env)
            self._try_to_eval(episode)
            gt.stamp('eval')
            self._end_epoch(episode)

        # Evaluate over all episodes.
        print("Evaluating all {} episodes...".format(len(eval_paths)))
        self.need_to_update_eval_statistics = False  # Evaluate eval_path regardless of whether we trained or not.
        set_to_eval_mode(self.env)
        self._try_to_eval(episode, eval_paths=eval_paths)
        gt.stamp('eval')
        self._end_epoch(episode)

        # Save plots.
        if hasattr(self.env, "draw"):
            self.env.draw(eval_paths, save_dir=logger.get_snapshot_dir())

    def _proc_observation(self, ob):
        return ob

    def _take_step_in_env(self, observation):
        action, agent_info = self._get_action_and_info(
            observation,
        )
        if self.render:
            self.training_env.render()
        next_ob, raw_reward, terminal, env_info = (
            self.training_env.step(action)
        )
        next_ob = self._proc_observation(next_ob)
        self._n_env_steps_total += 1
        reward = raw_reward * self.reward_scale
        terminal = np.array([terminal])
        reward = np.array([reward])
        self._handle_step(
            observation,
            action,
            reward,
            next_ob,
            terminal,
            agent_info=agent_info,
            env_info=env_info,
        )
        if terminal or len(self._current_path_builder) >= self.max_path_length:
            self._handle_rollout_ending()
            new_observation = self._start_new_rollout()
        else:
            new_observation = next_ob
        return new_observation

    def _try_to_train(self):
        if self._can_train():
            self.training_mode(True)
            for i in range(self.num_updates_per_train_call):
                self._do_training()
                self._n_train_steps_total += 1
            self.training_mode(False)

    def _try_to_eval(self, epoch, eval_paths=None):
        logger.save_extra_data(self.get_extra_data_to_save(epoch))
        if self._can_evaluate():
            self.evaluate(epoch, eval_paths=eval_paths)

            params = self.get_epoch_snapshot(epoch)
            logger.save_itr_params(epoch, params)
            table_keys = logger.get_table_key_set()
            if self._old_table_keys is not None:
                assert table_keys == self._old_table_keys, (
                    "Table keys cannot change from iteration to iteration."
                )
            self._old_table_keys = table_keys

            logger.record_tabular(
                "Number of train steps total",
                self._n_train_steps_total,
            )
            logger.record_tabular(
                "Number of env steps total",
                self._n_env_steps_total,
            )
            logger.record_tabular(
                "Number of rollouts total",
                self._n_rollouts_total,
            )

            times_itrs = gt.get_times().stamps.itrs
            train_time = times_itrs['train'][-1]
            sample_time = times_itrs['sample'][-1]
            eval_time = times_itrs['eval'][-1] if epoch > 0 else 0
            epoch_time = train_time + sample_time + eval_time
            total_time = gt.get_times().total

            logger.record_tabular('Train Time (s)', train_time)
            logger.record_tabular('(Previous) Eval Time (s)', eval_time)
            logger.record_tabular('Sample Time (s)', sample_time)
            logger.record_tabular('Epoch Time (s)', epoch_time)
            logger.record_tabular('Total Train Time (s)', total_time)
            logger.record_tabular("Epoch", epoch)
            logger.dump_tabular(with_prefix=False, with_timestamp=False)
        else:
            logger.log("Skipping eval for now.")

    def _can_evaluate(self):
        """
        One annoying thing about the logger table is that the keys at each
        iteration need to be the exact same. So unless you can compute
        everything, skip evaluation.

        A common example for why you might want to skip evaluation is that at
        the beginning of training, you may not have enough data for a
        validation and training set.
        """
        return (
            len(self._exploration_paths) > 0
            and not self.need_to_update_eval_statistics
        )

    def _can_train(self):
        return (
            self.replay_buffer.num_steps_can_sample() >=
            self.min_num_steps_before_training
        )

    def _get_action_and_info(self, observation):
        """
        Get an action to take in the environment.
        :param observation:
        :return:
        """
        self.exploration_policy.set_num_steps_total(self._n_env_steps_total)
        return self.exploration_policy.get_action(
            observation,
        )

    def _start_epoch(self, epoch):
        self._epoch_start_time = time.time()
        self._exploration_paths = []
        self._do_train_time = 0
        logger.push_prefix('Iteration #%d | ' % epoch)

    def _end_epoch(self, epoch):
        logger.log("Epoch Duration: {0}".format(
            time.time() - self._epoch_start_time
        ))
        logger.log("Started Training: {0}".format(self._can_train()))
        logger.pop_prefix()

        for post_epoch_func in self.post_epoch_funcs:
            post_epoch_func(self, epoch)

    def _start_new_rollout(self):
        self.exploration_policy.reset()
        ob = self.training_env.reset()
        return self._proc_observation(ob)

    def _handle_path(self, path):
        """
        Naive implementation: just loop through each transition.
        :param path:
        :return:
        """
        for (
            ob,
            action,
            reward,
            next_ob,
            terminal,
            agent_info,
            env_info
        ) in zip(
            path["observations"],
            path["actions"],
            path["rewards"],
            path["next_observations"],
            path["terminals"],
            path["agent_infos"],
            path["env_infos"],
        ):
            self._handle_step(
                ob,
                action,
                reward,
                next_ob,
                terminal,
                agent_info=agent_info,
                env_info=env_info,
            )
        self._handle_rollout_ending()

    def _handle_step(
            self,
            observation,
            action,
            reward,
            next_observation,
            terminal,
            agent_info,
            env_info,
    ):
        """
        Implement anything that needs to happen after every step
        :return:
        """
        self._current_path_builder.add_all(
            observations=observation,
            actions=action,
            rewards=reward,
            next_observations=next_observation,
            terminals=terminal,
            agent_infos=agent_info,
            env_infos=env_info,
        )
        self.replay_buffer.add_sample(
            observation=observation,
            action=action,
            reward=reward,
            terminal=terminal,
            next_observation=next_observation,
            agent_info=agent_info,
            env_info=env_info,
        )

    def _handle_rollout_ending(self):
        """
        Implement anything that needs to happen after every rollout.
        """
        self.replay_buffer.terminate_episode()
        self._n_rollouts_total += 1
        if len(self._current_path_builder) > 0:
            self._exploration_paths.append(
                self._current_path_builder.get_all_stacked()
            )
            self._current_path_builder = PathBuilder()

    def get_epoch_snapshot(self, epoch):
        data_to_save = dict(
            epoch=epoch,
            exploration_policy=self.exploration_policy,
            eval_policy=self.eval_policy,
        )
        if self.save_environment:
            data_to_save['env'] = self.training_env
        return data_to_save

    def get_extra_data_to_save(self, epoch):
        """
        Save things that shouldn't be saved every snapshot but rather
        overwritten every time.
        :param epoch:
        :return:
        """
        if self.render:
            self.training_env.render(close=True)
        data_to_save = dict(
            epoch=epoch,
        )
        if self.save_environment:
            data_to_save['env'] = self.training_env
        if self.save_replay_buffer:
            data_to_save['replay_buffer'] = self.replay_buffer
        if self.save_algorithm:
            data_to_save['algorithm'] = self
        return data_to_save

    @abc.abstractmethod
    def training_mode(self, mode):
        """
        Set training mode to `mode`.
        :param mode: If True, training will happen (e.g. set the dropout
        probabilities to not all ones).
        """
        pass

    def evaluate(self, epoch, eval_paths=None):
        statistics = OrderedDict()
        statistics.update(self.eval_statistics)

        logger.log("Collecting samples for evaluation")
        if eval_paths:
            test_paths = eval_paths
        else:
            test_paths = self.get_eval_paths()
        statistics.update(eval_util.get_generic_path_information(
            test_paths, stat_prefix="Test",
        ))
        if len(self._exploration_paths) > 0:
            statistics.update(eval_util.get_generic_path_information(
                self._exploration_paths, stat_prefix="Exploration",
            ))
        if hasattr(self.env, "log_diagnostics"):
            self.env.log_diagnostics(test_paths, logger=logger)
        if hasattr(self.env, "get_diagnostics"):
            env_statistics = self.env.get_diagnostics(test_paths)
            if env_statistics is not None:
                statistics.update(env_statistics)

        average_returns = eval_util.get_average_returns(test_paths)
        statistics['AverageReturn'] = average_returns
        for key, value in statistics.items():
            logger.record_tabular(key, value)
            logger.record_tensorboard_scalar(key, value, epoch)
        self.need_to_update_eval_statistics = True

    def get_eval_paths(self):
        return self.eval_sampler.obtain_samples()

    @abc.abstractmethod
    def _do_training(self):
        """
        Perform some update, e.g. perform one gradient step.
        :return:
        """
        pass

def set_to_train_mode(env):
    if hasattr(env, 'train'):
        env.train()

def set_to_eval_mode(env):
    if hasattr(env, 'eval'):
        env.eval()


