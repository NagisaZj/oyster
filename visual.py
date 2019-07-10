import numpy as np
import matplotlib
import matplotlib.pyplot as plt
#%matplotlib inline
import csv
import pickle
import os
import colour


# config
exp_id = '2019_07_07_09_19_18' # PEARL trainrew change
#exp_id = '2019_07_07_09_19_23' # smm trainrew change
exp_id = '2019_07_07_09_19_23'  #smm first trail
#exp_id = '2019_07_07_09_19_18'  # PEARL first trial

#smm only
#exp_id = '2019_07_09_19_47_25'
#exp_id = '2019_07_09_19_49_42'
#exp_id = '2019_07_09_19_50_10'
#exp_id = '2019_07_09_19_50_35'

#changed smm
#exp_id = '2019_07_09_15_50_16'
#exp_id = '2019_07_09_15_50_39'
#exp_id = '2019_07_09_15_51_40'
#exp_id = '2019_07_09_15_52_02'

#smm with another kind of reward
exp_id = '2019_07_09_19_42_14'
#exp_id = '2019_07_09_19_42_54'
#exp_id = '2019_07_09_19_43_34'
#exp_id = '2019_07_09_19_44_01'

#tracking the rewards
#exp_id = '2019_07_10_14_05_27'

tlow, thigh = 80, 100 # task ID range
# see `n_tasks` and `n_eval_tasks` args in the training config json
# by convention, the test tasks are always the last `n_eval_tasks` IDs
# so if there are 100 tasks total, and 20 test tasks, the test tasks will be IDs 81-100
epoch = 200 # training epoch to load data from
gr = 0.2 # goal radius, for visualization purposes

expdir = './output/sparse-point-robot/{}/eval_trajectories/'.format(exp_id) # directory to load data from
expdir = './SMMout/sparse-point-robot/{}/eval_trajectories/'.format(exp_id)
# helpers
def load_pkl(task):
    with open(os.path.join(expdir, 'task{}-epoch{}-run0.pkl'.format(task, epoch)), 'rb') as f:
        data = pickle.load(f)
    return data

def load_pkl_prior():
    with open(os.path.join(expdir, 'prior-epoch{}.pkl'.format(epoch)), 'rb') as f:
        data = pickle.load(f)
    return data

paths = load_pkl_prior()
goals = [load_pkl(task)[0]['goal'] for task in range(tlow, thigh)]

plt.figure(figsize=(8,8))
axes = plt.axes()
axes.set(aspect='equal')
plt.axis([-1.25, 1.25, -0.25, 1.25])
for g in goals:
    circle = plt.Circle((g[0], g[1]), radius=gr)
    axes.add_artist(circle)
rewards = 0
final_rewards = 0
for traj in paths:
    rewards += sum(traj['rewards'])
    final_rewards += traj['rewards'][-1]
    states = traj['observations']
    plt.plot(states[:-1, 0], states[:-1, 1], '-o')
    plt.plot(states[-1, 0], states[-1, 1], '-x', markersize=20)

mpl = 20
num_trajs = 60

all_paths = []
for task in range(tlow, thigh):
    paths = [t['observations'] for t in load_pkl(task)]
    all_paths.append(paths)

# color trajectories in order they were collected
cmap = matplotlib.cm.get_cmap('plasma')
sample_locs = np.linspace(0, 0.9, num_trajs)
colors = [cmap(s) for s in sample_locs]

fig, axes = plt.subplots(3, 3, figsize=(12, 20))
t = 0


all_paths_rew = []
for task in range(tlow, thigh):
    paths = [t['rewards'] for t in load_pkl(task)]
    all_paths_rew.append(paths)
reward = np.zeros((20,1))
final_rew = np.zeros((20,1))
for m in range(20):
    for n in range(len(all_paths_rew[m])):
        #reward[m] = reward[m] + sum(all_paths_rew[m][n])
        reward[m] = reward[m] + all_paths_rew[m][n][-1]
    reward[m] = reward[m] / len(all_paths_rew[m])

print(reward)
print(np.mean(reward))

for j in range(3):
    for i in range(3):
        axes[i, j].set_xlim([-1.25, 1.25])
        axes[i, j].set_ylim([-0.25, 1.25])
        for k, g in enumerate(goals):
            alpha = 1 if k == t else 0.2
            circle = plt.Circle((g[0], g[1]), radius=gr, alpha=alpha)
            axes[i, j].add_artist(circle)
        indices = list(np.linspace(0, len(all_paths[t]), num_trajs, endpoint=False).astype(np.int))
        counter = 0
        for idx in indices:
            states = all_paths[t][idx]
            axes[i, j].plot(states[:-1, 0], states[:-1, 1], '-', color=colors[counter])
            axes[i, j].plot(states[-1, 0], states[-1, 1], '-x', markersize=10, color=colors[counter])
            axes[i, j].set(aspect='equal')
            counter += 1
        axes[i,j].set_title("average reward:%f"%reward[t])
        t += 1
fig.suptitle("iteration:%d, average reward of all tasks:%f"%(epoch,np.mean(reward)))
plt.show()
