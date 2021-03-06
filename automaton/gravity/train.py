from automaton.gravity.model import NeuralAutomatonCollector

import os
from os.path import abspath, dirname, join, exists
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

import torch
from torch import nn, optim
from torch.nn import functional as F


def get_board(coords, resolution=32):
    board = torch.zeros((resolution, resolution), device=coords.device)
    for c in coords:
        idx = (c * resolution).int()
        if idx.min() >= 0 and idx.max() < resolution:
            board[idx[0],idx[1]] = 1
    return board


def update(coords, vel, strength=1e-3, dampening=0.999):
    for i in range(coords.size(0)):
        for j in range(i + 1, coords.size(0)):
            direction = coords[i] - coords[j]
            dist_sq = (direction ** 2).sum()
            if dist_sq > 1e-2:
                grav = strength / dist_sq
                vel[i] -= direction * grav
                vel[j] += direction * grav

    coords = coords + vel
    vel = vel * dampening
    return coords, vel


def trajectory(n_steps, n_obj=3, resolution=32, coords=None, vel=None, device='cpu'):
    coords = torch.rand((n_obj, 2), device=device) if coords is None else coords
    vel = torch.zeros_like(coords, device=coords.device) if vel is None else vel

    frames = torch.empty((n_steps, resolution, resolution), device=device)
    for i in range(n_steps):
        frames[i] = get_board(coords, resolution)
        coords, vel = update(coords, vel)
    return frames


def train(save_path, load_model=None, n_traj=16, steps_range=(30, 60), n_obj=2, res=30, lr=1e-4, device='cpu'):
    if load_model is None:
        model = NeuralAutomatonCollector().to(device)
    else:
        model = torch.load(load_model, map_location=device)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    step = 0
    while True:
        optimizer.zero_grad()
        n_steps = torch.randint(*steps_range, (1,))

        # get ground truth simulation
        y = torch.stack([trajectory(n_steps + 1, n_obj=n_obj, resolution=res, device=device) for _ in range(n_traj)])
        x = torch.cat([y[:,0].unsqueeze(1), torch.zeros(n_traj, model.n_channels - 1, res, res, device=device)], dim=1)

        # forward step
        for _ in range(n_steps):
            x = model(x).clip(0, 1)

        # backward step
        mask = y[:,-1] == 1
        loss = loss_fn(x[:,0][mask], y[:,-1][mask]) + 0.1 * loss_fn(x[:,0][~mask], y[:,-1][~mask])
        loss.backward()
        optimizer.step()

        # logging
        loss_np = loss.cpu().detach().numpy()
        del loss
        step += 1

        print(f'step {step}: {loss_np}')

        if step % 10 == 0:
            print('Saving model checkpoint...')
            path = abspath(join(save_path, f'model-{loss_np:.4f}'))
            torch.save(model, path)


if __name__ == '__main__':
    save_path = join(dirname(__file__), 'models')
    if not exists(save_path):
        os.makedirs(save_path)
    train(save_path)