import torch
from torch.optim.optimizer import Optimizer
from collections import deque
import copy

class TrajectoryLookahead(Optimizer):
    def __init__(self, optimizer, la_steps=5, trajectory_len=3, la_alpha=0.8):
        if not 0.0 <= la_alpha <= 1.0:
            raise ValueError(f'Invalid slow update rate: {la_alpha}')
        if not 1 <= la_steps:
            raise ValueError(f'Invalid lookahead steps: {la_steps}')
        if not 1 <= trajectory_len:
            raise ValueError(f'Invalid trajectory length: {trajectory_len}')

        self.optimizer = optimizer
        self.la_alpha = la_alpha
        self.la_steps = la_steps
        self.trajectory_len = trajectory_len

        self.state = {}
        for group in self.optimizer.param_groups:
            for p in group['params']:
                self.state[p] = {
                    'slow_param': torch.clone(p.data),
                    'trajectory': deque(maxlen=self.trajectory_len)
                }
        self.la_counter = 0

    def zero_grad(self):
        self.optimizer.zero_grad()

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def step(self, closure=None):
        loss = self.optimizer.step(closure)
        self.la_counter += 1

        for group in self.optimizer.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['trajectory'].append(copy.deepcopy(p.data))

        if self.la_counter >= self.la_steps:
            self.la_counter = 0
            for group in self.optimizer.param_groups:
                for p in group['params']:
                    state = self.state[p]
                    if len(state['trajectory']) > 0:
                        # Average the trajectory
                        avg_param = torch.stack(list(state['trajectory'])).mean(dim=0)

                        # Update slow parameter
                        state['slow_param'].data.add_(self.la_alpha, avg_param - state['slow_param'].data)

                        # Copy updated slow parameter back to the main parameter
                        p.data.copy_(state['slow_param'].data)

                        # Clear the trajectory
                        state['trajectory'].clear()
        return loss
