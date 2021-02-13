from model import NeuralAutomatonCollector

import os
from os.path import abspath, dirname, join, exists
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

import torch
from torch import nn, optim
from torch.nn import functional as F

def get_board(coords, vel, resolution=32):
    board = torch.zeros((resolution, resolution))
    for c, v in zip(coords, vel):
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

def trajectory(n_steps, n_obj=3, resolution=32, coords=None, vel=None):
    coords = torch.rand((n_obj, 2)) if coords is None else coords
    vel = torch.zeros_like(coords) if vel is None else vel

    frames = torch.empty((n_steps, resolution, resolution))
    for i in range(n_steps):
        frames[i] = get_board(coords, vel, resolution)
        coords, vel = update(coords, vel)
    return frames

def main():
    n_traj = 8
    n_steps = 20
    n_obj = 2
    res = 20

    model = NeuralAutomatonCollector()
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    step = 0
    losses = []
    while True:
        y = torch.stack([trajectory(n_steps + 1, n_obj=n_obj, resolution=res) for _ in range(n_traj)])

        pred = torch.zeros(n_traj, n_steps, res, res)
        x = torch.cat([y[:,0].unsqueeze(1), torch.zeros(n_traj, model.n_channels - 1, res, res)], dim=1)
        for i in range(n_steps):
            x = model(x).clip(0, 1)
            pred[:,i] = x[:,0]

        optimizer.zero_grad()
        mask = y[:,1:] == 1
        loss = loss_fn(pred[mask], y[:,1:][mask]) + pred.norm(p=1) * 5e-5
        loss.backward()
        optimizer.step()

        losses.append(loss.detach().numpy())
        step += 1

        if step % 50 == 0:
            path = abspath(join(dirname(__file__), 'models', f'model-{np.mean(losses):.4f}'))
            if not exists(dirname(path)):
                os.makedirs(dirname(path))
            torch.save(model, path)

        if step % 25 == 0:
            print(f'step {step}: {np.mean(losses)}')
            losses = []

if __name__ == '__main__':
    main()