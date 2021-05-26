import torch
import torch.nn as nn
import model
import buffer
import copy

device = "cuda" if torch.cuda.is_available() else "cpu"

class Agent:
    def __init__(self, sync_frame, buffer_length, action_size, observation_size, discount_factor=0.99, learning_rate=1e-3, batch_size=64):
        self.t_frame=0
        self.sync_frame = sync_frame
        self.action_size=action_size
        self.observation_size=observation_size
        self.discount_factor=discount_factor
        self.lr = learning_rate
        self.batch_size = batch_size
        
        self.net = model.DQN(action_size, observation_size).to(device)
        self.target_net = copy.deepcopy(self.net)
        self.rep_buffer = buffer.ReplayBuffer(capacity=buffer_length)

    def step(self, observation: buffer.Experience, env):
        self.rep_buffer.add_exp(buffer.Experience)
        self.t_frame+=1
        self.learn(env, observation)

    def act(self, obs, eps):
        return self.net(obs).max()

    def learn(self, epsilon):
        states, actions, rewards, dones, next_states = self.rep_buffer.sample(batch_size=64)

    def target_net_update(self, env):
        if self.t_frame % self.sync_frame:
            self.target_net.load_state_dict(self.net.state_dict())