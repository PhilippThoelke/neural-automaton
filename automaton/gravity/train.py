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

def train(save_path, load_model=None, n_traj=8, n_steps=20, n_obj=2, sample_delay=1, res=20, lr=1e-3,
          double_step_interval=100, max_steps=100, save_interval=10, log_interval=2, device='cpu'):
    if load_model is None:
        model = NeuralAutomatonCollector().to(device)
    else:
        model = torch.load(load_model, map_location=device)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    step = 0
    losses = []
    while True:
        optimizer.zero_grad()
        
        # get ground truth simulation
        y = torch.stack([trajectory(n_steps + 1, n_obj=n_obj, resolution=res, device=device) for _ in range(n_traj)])
        x = torch.cat([y[:,0].unsqueeze(1), torch.zeros(n_traj, model.n_channels - 1, res, res, device=device)], dim=1)
        y = y[:,sample_delay::sample_delay]
        
        # forward step
        pred = torch.zeros(n_traj, n_steps // sample_delay, res, res, device=device)
        for i in range(n_steps):
            x = model(x).clip(0, 1)
            if (i + 1) % sample_delay == 0:
                pred[:,i // sample_delay] = x[:,0]
                x = x.detach()

        # backward step
        mask = y == 1
        loss = loss_fn(pred[mask], y[mask]) + pred.norm(p=1) * 5e-5
        loss.backward()
        optimizer.step()

        # logging
        losses.append(loss.cpu().detach().numpy())
        del loss
        step += 1

        if step % save_interval == 0:
            print('Saving model checkpoint...')
            path = abspath(join(save_path, f'model-{np.mean(losses):.4f}'))
            torch.save(model, path)
            
        if step % log_interval == 0:
            print(f'step {step}: {np.mean(losses)}')
            losses = []
        
        if step % double_step_interval == 0:
            n_steps = min(n_steps * 2, max_steps)
            sample_delay = min(10, sample_delay * 2)
            print(f'Setting number of simulation steps to {n_steps}')

if __name__ == '__main__':
    save_path = join(dirname(__file__), 'models')
    if not exists(save_path):
        os.makedirs(save_path)
    train(save_path, load_model='/home/philipp/Documents/neural-automaton/automaton/gravity/models/model-0.1815')