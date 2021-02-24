from automaton.game_of_life.model import NeuralAutomaton

from os.path import dirname, abspath, join
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

import torch

def run(model_path, size=100):
    model = torch.load(model_path)
    state = (torch.randn((size, size)) > 0.8).float()

    fig, ax = plt.subplots()
    state_plot = ax.matshow(state, cmap='gray')
    ax.axis('off')

    def update(frame, state):
        state[:] = model(state)[0].detach()
        state_plot.set_data(state.detach().numpy())
        return [state_plot]

    anim = FuncAnimation(fig, update, fargs=(state,), interval=100, blit=True)
    plt.show()

if __name__ == '__main__':
    model_path = abspath(join(dirname(__file__), 'models', 'model-0.0001'))
    run(model_path)