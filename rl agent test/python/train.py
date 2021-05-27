import mlagents_envs
import torch
import numpy as np
import mlagents
from mlagents_envs.environment import UnityEnvironment as UE, BehaviorName, BehaviorSpec
import tensorboardX
import time

import agent
import hyperparams

#if this is set to None you have to click the play button in the unity scene after creating the env
env_file_name = None

def main():
    if env_file_name == None:
        print("please click the play button in your unity scene")
    env = UE(file_name=env_file_name)
    if env_file_name == None:
        print("environment initialized - if you pause or close the unity scene training will stop")
    env.reset()

    behavior_name = list(env.behavior_specs)[0]
    spec = env.behavior_specs[behavior_name]
    
    decision_steps, terminl_steps = env.get_steps(behavior_name)

    observation_size = spec.observation_specs[0].shape[0]
    action_size = len(decision_steps[0][0][0])

    dqn_agent = agent.Agent(10, 300)


if __name__ == "__main__":
    main()