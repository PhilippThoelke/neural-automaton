from model import NeuralAutomatonCollector
from train import trajectory, get_board, update as traj_update

from os.path import abspath, dirname, join
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

import torch

res = 20
n_obj = 2

model_name = 'model-0.1815'
model = torch.load(abspath(join(dirname(__file__), 'models', model_name)))

def reset():
    coords = torch.rand((n_obj, 2))
    vel = torch.zeros_like(coords)
    state = torch.cat([trajectory(1, n_obj=n_obj, resolution=res, coords=coords, vel=vel), torch.zeros(model.n_channels - 1, res, res)], dim=0).unsqueeze(0)
    return coords, vel, state

coords, vel, state = reset()

fig, (ax1, ax2) = plt.subplots(ncols=2)
title = fig.suptitle('Frame 0')
target_plot = ax1.imshow(get_board(coords, vel, resolution=res), cmap='gray')
pred_plot = ax2.imshow(state[0,0], cmap='gray')
ax1.set_title('Target')
ax2.set_title('Prediction')
ax1.axis('off')
ax2.axis('off')

def update(frame, state, coords, vel):
    if (frame + 1) % 50 == 0:
        coords[:], vel[:], state[:] = reset()
    title.set_text(f'Frame {(frame + 1) % 50}')

    coords[:], vel[:] = traj_update(coords, vel)
    target_plot.set_data(get_board(coords, vel, resolution=res))

    state[:] = model(state).detach().clip(0, 1)
    pred_plot.set_data(state[0,0].detach().numpy())

anim = FuncAnimation(fig, update, fargs=(state, coords, vel), interval=50)
plt.show()