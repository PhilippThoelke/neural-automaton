from automaton.game_of_life.model import NeuralAutomaton

import os
from os.path import abspath, dirname, join, exists
from matplotlib import pyplot as plt

import torch
from torch import nn, optim

def tick(state):
    kernel = torch.ones((1, 1, 3, 3), device=state.device)
    kernel[:,:,1,1] = 0

    state_ndim = state.ndim
    if state.ndim == 2:
        state = state.unsqueeze(0)
    neighbors_alive = torch.nn.functional.conv2d(state.unsqueeze(1), kernel, padding=1)[:,0]
    
    state = state.detach().clone()
    state[neighbors_alive < 2] = 0 # underpopulation
    state[neighbors_alive > 3] = 0 # overpopulation
    state[neighbors_alive == 3] = 1 # reproduction

    if state_ndim == 2:
        state = state[0]
    return state

def train(save_path, board_size=64, batch_size=32, threshold=0.8, lr=1e-2, plot=False, save_interval=50, device='cpu'):
    model = NeuralAutomaton().to(device)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    step = 1
    while True:
        optimizer.zero_grad()

        batch = (torch.rand((batch_size, board_size, board_size), device=device) > threshold).float()
        target = tick(batch)
        prediction = model(batch)
        
        loss = loss_fn(prediction, target)
        loss.backward()
        optimizer.step()

        print(f'{step}: {loss.cpu().detach().numpy()}')
        step += 1

        if step % save_interval == 0:
            print('Saving model checkpoint...')
            path = abspath(join(save_path, f'model-{loss.cpu().detach().numpy():.4f}'))
            torch.save(model, path)

            if plot:
                state = (torch.rand((20, 20), device=device) > 0.8).float()
                _, axes = plt.subplots(2, 2)

                axes[0,0].matshow(state)
                axes[0,0].set_title('before')
                axes[0,0].axis('off')

                prediction = model(state)[0].cpu().detach().numpy()
                axes[1,0].matshow(prediction)
                axes[1,0].set_title('prediction')
                axes[1,0].axis('off')

                target = tick(state)
                axes[0,1].matshow(target)
                axes[0,1].set_title('target')
                axes[0,1].axis('off')

                axes[1,1].matshow((target - prediction).abs())
                axes[1,1].set_title('difference')
                axes[1,1].axis('off')

                plt.tight_layout()
                plt.show()

if __name__ == '__main__':
    save_path = join(dirname(__file__), 'models')
    if not exists(save_path):
        os.makedirs(save_path)
    train(save_path)