import numpy as np
from gym import spaces
from gym import Env

from . import register_env


@register_env('point-robot')
class PointEnv(Env):
    """
    point robot on a 2-D plane with position control
    tasks (aka goals) are positions on the plane

     - tasks sampled from unit square
     - reward is L2 distance
    """

    def __init__(self, randomize_tasks=False, n_tasks=2):

        if randomize_tasks:
            np.random.seed(1337)
            goals = [[np.random.uniform(-1., 1.), np.random.uniform(-1., 1.)] for _ in range(n_tasks)]
        else:
            # some hand-coded goals for debugging
            goals = [np.array([10, -10]),
                     np.array([10, 10]),
                     np.array([-10, 10]),
                     np.array([-10, -10]),
                     np.array([0, 0]),

                     np.array([7, 2]),
                     np.array([0, 4]),
                     np.array([-6, 9])
                     ]
            goals = [g / 10. for g in goals]
        self.goals = goals

        self.reset_task(0)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,))
        self.action_space = spaces.Box(low=-0.1, high=0.1, shape=(2,))

    def reset_task(self, idx):
        ''' reset goal AND reset the agent '''
        self._goal = self.goals[idx]
        self.reset()

    def get_all_task_idx(self):
        return range(len(self.goals))

    def reset_model(self):
        # reset to a random location on the unit square
        self._state = np.random.uniform(-1., 1., size=(2,))
        return self._get_obs()

    def reset(self):
        return self.reset_model()

    def _get_obs(self):
        return np.copy(self._state)

    def step(self, action):
        self._state = self._state + action
        x, y = self._state
        x -= self._goal[0]
        y -= self._goal[1]
        reward = - (x ** 2 + y ** 2) ** 0.5
        done = False
        ob = self._get_obs()
        return ob, reward, done, dict()

    def viewer_setup(self):
        print('no viewer')
        pass

    def render(self):
        print('current state:', self._state)


@register_env('sparse-point-robot')
class SparsePointEnv(PointEnv):
    '''
     - tasks sampled from unit half-circle
     - reward is L2 distance given only within goal radius

     NOTE that `step()` returns the dense reward because this is used during meta-training
     the algorithm should call `sparsify_rewards()` to get the sparse rewards
     '''
    def __init__(self, randomize_tasks=False, n_tasks=2, goal_radius=0.2):
        super().__init__(randomize_tasks, n_tasks)
        self.goal_radius = goal_radius

        if randomize_tasks:
            np.random.seed(1337)
            radius = 1.0
            angles = np.linspace(0, np.pi, num=n_tasks)
            xs = radius * np.cos(angles)
            ys = radius * np.sin(angles)
            goals = np.stack([xs, ys], axis=1)
            np.random.shuffle(goals)
            goals = goals.tolist()

        self.goals = goals
        self.reset_task(0)
        self.goals_np = np.array(goals)[:80,:]

    def sparsify_rewards(self, r):
        ''' zero out rewards when outside the goal radius '''
        mask = (r >= -self.goal_radius).astype(np.float32)
        r = r * mask
        return r

    def reset_model(self):
        self._state = np.array([0, 0])
        return self._get_obs()

    def step(self, action):
        ob, reward, done, d = super().step(action)
        sparse_reward = self.sparsify_rewards(reward)
        # make sparse rewards positive
        if reward >= -self.goal_radius:
            sparse_reward += 1
        explore_reward = self.get_explore_reward(self._state,sparse_reward,action)
        d.update({'sparse_reward': sparse_reward,"info":explore_reward})
        return ob, reward, done, d

    def get_explore_reward(self,_state,sparse_reward,action):
        relative_pos = self.goals_np - _state
        rew = -1 * (relative_pos[:,0] **2 + relative_pos[:,1] **2 ) **0.5
        mask = (rew >= -self.goal_radius).astype(np.float32)
        rew = rew * mask
        explore_reward = np.mean(np.abs(rew-sparse_reward))
        if np.sum(action)<0.1:
            explore_reward = 0
        #print(rew)
        return explore_reward





@register_env('goal-pitfall')
class goal_pitfall_env(PointEnv):
    '''
     - tasks sampled from unit half-circle
     - reward is L2 distance given only within goal radius

     NOTE that `step()` returns the dense reward because this is used during meta-training
     the algorithm should call `sparsify_rewards()` to get the sparse rewards
     '''
    def __init__(self, randomize_tasks=False, n_tasks=2, goal_radius=0.2):
        #super().__init__(randomize_tasks, n_tasks)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,))
        self.action_space = spaces.Box(low=-0.1, high=0.1, shape=(2,))
        self.goal_radius = goal_radius

        if randomize_tasks:
            np.random.seed(1337)
            goals = np.random.rand(n_tasks,2)*-0.4 - 0.8
            pitfalls = np.random.rand(n_tasks, 2) * 0.4 + 0.8
            for i in range(n_tasks):
                if np.random.rand()>0.5:
                    goals[i,:] = -1 * goals[i,:]
                    pitfalls[i,:] = -1 * pitfalls[i,:]
            goals = goals.tolist()
            pitfalls = pitfalls.tolist()


        self.goals = goals
        self.pitfalls = pitfalls
        self.reset_task(0)

    def reset_task(self, idx):
        ''' reset goal AND reset the agent '''
        self._goal = self.goals[idx]
        self._pitfall = self.pitfalls[idx]
        self.reset()



    def reset_model(self):
        self._state = np.array([0, 0])
        return self._get_obs()

    def step(self, action):
        ob, reward, done, d = super().step(action)
        x, y = self._state
        x -= self._pitfall[0]
        y -= self._pitfall[1]
        pitfall_dis =  (x ** 2 + y ** 2) ** 0.5
        pitfall_rew = pitfall_dis - 1
        pitfall_rew = 0 if pitfall_rew>0 else pitfall_rew
        if pitfall_dis<0.3:
            pitfall_rew = pitfall_rew - 20
        reward = reward + pitfall_rew
        return ob, reward, done, d


@register_env('goat-car')
class GoatCarEnv(Env):
    def __init__(self, randomize_tasks=False, n_tasks=2):

        if randomize_tasks:
            np.random.seed(1337)
            goals = np.random.choice(3,n_tasks).tolist()
        goals = np.ones((n_tasks),).tolist()
        self.goals = goals

        self.reset_task(0)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1,))
        self.action_space = spaces.Discrete(4)
        self.info_space = spaces.Discrete(4)

    def reset_task(self, idx):
        ''' reset goal AND reset the agent '''
        self._goal = self.goals[idx]
        self.reset()

    def get_all_task_idx(self):
        return range(len(self.goals))

    def reset_model(self):
        # reset to a random location on the unit square
        self._state = np.ones((1,))
        return self._get_obs()

    def reset(self):
        return self.reset_model()

    def _get_obs(self):
        return np.copy(self._state)

    def step(self, action):
        #print(action)
        action = np.argmax(action)
        if action == 3:
            return self._state, 0,False,dict(info=self._goal)
        else:
            reward = 1 if action == self._goal else -1
            #if reward==1:
            #    print("!")
            return self._state, reward, False,dict(info=-1)

    def viewer_setup(self):
        print('no viewer')
        pass

    def render(self):
        print('current state:', self._state)

