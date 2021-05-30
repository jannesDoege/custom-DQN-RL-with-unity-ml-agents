import torch
import torch.nn as nn
import torch.functional as F

#simple linear deep q network
class DQN(nn.Module):
    def __init__(self, input_space, action_space, hidden_units_one=128, hidden_units_two=128):
        super(DQN, self).__init__()
        self.input_shape = input_space
        self.action_space = action_space
        self.hidden_units_one = hidden_units_one
        self.hidden_units_two = hidden_units_two

        self.net = nn.Sequential(
            nn.Linear(self.input_shape, self.hidden_units_one),
            nn.ReLU(),
            nn.Linear(self.hidden_units_one, self.hidden_units_two),
            nn.ReLU(),
            nn.Linear(self.hidden_units_two, self.action_space),
            nn.ReLU()
        )



    def forward(self, x):

        return self.net(x)
    