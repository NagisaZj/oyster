import numpy as np
from gym import spaces
from gym import Env

#from . import register_env


class PointEnv_SMM(Env):
    def __init__(self,goal_prior=None,sample_goal = None,shaped_rewards=None,distance_threshold=None,init_object_pos_prior=None,terminate_upon_success=None,terminate_upon_failure=None,goal_radius=1,goal_angle=np.pi,reward_radius=0.2,num_goals_sample=100):
        '''

        :param goal_radius: the radius of a circle, on which goals are distributed.
        :param goal_angle:  the maximal angle that goals possibly exists,(0,2*pi)
        :param reward_radius: maximum distance from goal to state to get reward
        :param num_goals_sample: sample a set of goals to calculate rew
        '''
        self.goal_radius = goal_radius
        self.goal_angle = goal_angle
        self.reward_raidus = reward_radius
        self.num_goals_sample = num_goals_sample
        angles = np.linspace(0, self.goal_angle, num=self.num_goals_sample)
        xs = self.goal_radius * np.cos(angles)
        ys = self.goal_radius * np.sin(angles)
        self.goals = np.stack([xs, ys], axis=1)
        self.reset()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,))
        self.action_space = spaces.Box(low=-0.1, high=0.1, shape=(2,))

    def cal_rew(self,state):
        rew = 0
        for i in range(self.num_goals_sample):
            x = state[0] - self.goals[i,0]
            y = state[1] - self.goals[i,1]
            dis =  (x ** 2 + y ** 2) ** 0.5
            if dis < self.reward_raidus:
                rew = rew + 1
        if rew>0:
            return np.sqrt(rew)/10
        else:
            return -1

    def reset(self):
        self._state = np.array([0, 0])
        return np.copy(self._state)

    def step(self, action):
        action = np.clip(action,-0.1,0.1)
        self._state = self._state + action
        reward = self.cal_rew(self._state)
        done = False
        ob = np.copy(self._state)
        return ob, reward, done, dict()

#@register_env('point-robot')
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


#@register_env('sparse-point-robot')
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
        d.update({'sparse_reward': sparse_reward})
        return ob, reward, done, d


if __name__ =="__main__":
    env = PointEnv_SMM()
    x = np.linspace(-1,1,100)
    y = np.linspace(-1,1,100)
    xm,ym=np.meshgrid(x,y)
    z = np.zeros((100,100),dtype=np.float32)
    for i in range(100):
        for j in range(100):
            z[i,j] = env.cal_rew(np.array([x[i],y[j]],dtype=np.float32))
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(xm,ym,z)
    plt.show()
