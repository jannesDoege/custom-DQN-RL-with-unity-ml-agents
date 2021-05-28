import torch
import numpy as np
import mlagents
from mlagents_envs.environment import UnityEnvironment as UE, BehaviorName, BehaviorSpec
from tensorboardX import SummaryWriter
import time

import agent
from hyperparams import hyperparams
from buffer import Experience

#if this is set to None you have to click the play button in the unity scene after creating the env
env_file_name = None

device = "cuda" if torch.cuda.is_available() else "cpu"

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
    observation_size = len(decision_steps[0][0][0])
    action_size = len(decision_steps)
    done_t = False
    rewards = []
    best_m_reward = None
    epsilon = hyperparams["epsilon_start"]

    dqn_agent = agent.Agent(hyperparams["sync_frame"], hyperparams["buffer_length"], action_size, observation_size, hyperparams["discount_factor"],
                            hyperparams["learning_rate"], hyperparams["batch_size"], hyperparams["update_rate"])
    writer = SummaryWriter(comment="-" + behavior_name)
    
    start_time = time.time

    while True:
        env.reset()
        decision_steps, terminl_steps = env.get_steps(behavior_name)
        obs = decision_steps[0][0][0]

        t_reward = 0
        tracked_agent = decision_steps.agent_id[0]

        state_v = torch.tensor(obs).to(device)
        action = dqn_agent.act(obs, epsilon)   

        env.set_actions(behavior_name, action)
        env.step()


        
        #act in env

        dqn_agent.step(Experience(obs, action, reward, done, next_state))

def update_epsilon(eps):
    return eps
    
    train_time = time.time - start_time

    


if __name__ == "__main__":
    main()