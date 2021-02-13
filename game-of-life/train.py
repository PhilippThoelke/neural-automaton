from model import NeuralAutomaton

import os
from os.path import abspath, dirname, join, exists
from matplotlib import pyplot as plt

import torch
from torch import nn, optim

def tick(state):
    kernel = torch.ones((1, 1, 3, 3))
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

def main():
    board_size = 64
    batch_size = 32
    threshold = 0.8
    plot = False

    model = NeuralAutomaton()
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-2)

    step = 1
    while True:
        batch = (torch.rand((batch_size, board_size, board_size)) > threshold).float()
        target = tick(batch)
        prediction = model(batch)
        loss = loss_fn(prediction, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f'{step}: {loss.detach().numpy()}')
        step += 1

        if step % 50 == 0:
            path = abspath(join(dirname(__file__), 'models', f'model-{loss.detach().numpy():.4f}'))
            if not exists(dirname(path)):
                os.makedirs(dirname(path))
            torch.save(model, path)

            if plot:
                state = (torch.rand((20, 20)) > 0.8).float()
                _, axes = plt.subplots(2, 2)

                axes[0,0].matshow(state)
                axes[0,0].set_title('before')
                axes[0,0].axis('off')

                prediction = model(state)[0].detach().numpy()
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
    main()
