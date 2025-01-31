import numpy as np
import matplotlib
import matplotlib.pyplot as plt
#%matplotlib inline
import csv
import pickle
import os
import colour
import torch
from rlkit.torch.networks import SnailEncoder,MlpEncoder
# config
def cal_rew(encoder,path):
    s,a,r = path['observations'],path['actions'],path['rewards']
    s,a,r=torch.FloatTensor(s),torch.FloatTensor(a),torch.FloatTensor(r)
    input = torch.cat([s,a,r],dim=1)
    print(input.shape)
    input = torch.unsqueeze(input,0)
    output = encoder.forward_seq(input)
    print(output.shape)
    var = torch.mean(torch.log(torch.nn.functional.softplus(output[:,:,10:])),dim=2)
    var = torch.mean(torch.nn.functional.softplus(output[:,:,10:]),dim=2)
    #var = torch.log(var)
    print(var.shape)
    #print(r,var)
    return var.view(r.shape[0],r.shape[1])


exp_id = '2019_09_07_14_02_36' #gooooooood
exp_id = '2019_09_07_14_02_39' #not good
exp_id = '2019_09_08_09_02_27' #not good
exp_id = '2019_09_08_09_28_38' #not good
exp_id = '2019_09_08_15_00_41' #not good
exp_id = '2019_09_08_20_08_03' #not good
tlow, thigh = 80, 100 # task ID range
# see `n_tasks` and `n_eval_tasks` args in the training config json
# by convention, the test tasks are always the last `n_eval_tasks` IDs
# so if there are 100 tasks total, and 20 test tasks, the test tasks will be IDs 81-100
epoch = 237 # training epoch to load data from
gr = 0.2 # goal radius, for visualization purposes


expdir = './outputee/sparse-point-robot/{}/eval_trajectories/'.format(exp_id)
expdir = './outputfin/sparse-point-robot/{}/eval_trajectories/'.format(exp_id)
dir = './outputfin/sparse-point-robot/{}/'.format(exp_id)
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

'''all_paths_z_means = []
all_paths_z_vars = []
for task in range(tlow, thigh):
    means = [t['z_means'] for t in load_pkl(task)]
    vars = [t['z_vars'] for t in load_pkl(task)]
    all_paths_z_means.append(means)
    all_paths_z_vars.append(means)'''

reward = np.zeros((20,1))
final_rew = np.zeros((20,1))
for m in range(20):
    for n in range(len(all_paths_rew[m])):
        reward[m] = reward[m] + np.sum(all_paths_rew[m][n])
        #reward[m] = reward[m] + all_paths_rew[m][n][-1]
    reward[m] = reward[m] / len(all_paths_rew[m])

#print(reward)
#print(np.mean(reward))

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


task = 2
fig, axes = plt.subplots(4, 4)
encoder = SnailEncoder(hidden_sizes=[20],
        input_size=5,
        output_size=10,)
#encoder.load_state_dict(torch.load(os.path.join(dir, 'context_encoder.pth')))
ap = [t for t in load_pkl(task+80)]


for m in range(4):
    for n in range(4):
        id = m*4 +n
        axes[m,n].set_xlim([-1.25, 1.25])
        axes[m,n].set_ylim([-0.25, 1.25])
        for k, g in enumerate(goals):
            alpha = 1 if k == task else 0.2
            circle = plt.Circle((g[0], g[1]), radius=gr, alpha=alpha)
            axes[m,n].add_artist(circle)
        states = all_paths[task][id]
        #rew = cal_rew(encoder, ap[id])
        axes[m,n].plot(states[:-1, 0], states[:-1, 1], '-', color=colors[0])
        axes[m,n].plot(states[-1, 0], states[-1, 1], '-x', markersize=10, color=colors[0])
        for i in range(0, 9, 9):
            #axes[m, n].text(states[i, 0], states[i, 1], '%f' % rew[i, 0].data.numpy())
            axes[m, n].text(states[i, 0], states[i, 1], '%f' % np.mean(ap[id]['z_vars'][i]))
        for i in range(19, 20, 10):
            #axes[m, n].text(states[i, 0] + np.random.rand() * 0.0 - 0.0, states[i, 1] + np.random.rand() * 0.0 - 0.0,
            #                '%f' % rew[i, 0].data.numpy())
            axes[m, n].text(states[i, 0], states[i, 1], '%f' % np.min(ap[id]['z_vars'][i]))
for i in range(10):
    print(ap[i]['z_means'])
plt.show()
