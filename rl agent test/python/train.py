import torch
import numpy as np
import mlagents
from mlagents_envs.environment import UnityEnvironment as UE
import tensorboardX
import time

import agent
import hyperparams


def main():
    dqn_agent = agent.Agent(10, 300)


if __name__ == "__main__":
    main()