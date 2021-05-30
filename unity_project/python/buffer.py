import collections
import numpy as np

Experience = collections.namedtuple("Experience", field_names=["state", "action", "reward", "done", "next_state"])

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def add_exp(self, exp: Experience):
        self.buffer.append(exp)

    def sample(self, batch_size):
        idxs = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in idxs])

        return np.array(states), np.array(actions,  dtype=np.float32), \
               np.array(rewards, dtype=np.float32), \
               np.array(dones, dtype=np.uint8), \
               np.array(next_states)
