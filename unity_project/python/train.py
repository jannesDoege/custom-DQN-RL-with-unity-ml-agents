import torch
import numpy as np
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
    
    decision_steps, terminal_steps = env.get_steps(behavior_name)
    observation_size = len(decision_steps[0][0][0])
    action_size = len(decision_steps)
    done_t = False
    rewards = []
    best_m_reward = None
    epsilon = hyperparams["epsilon_start"]

    DQN_agent = agent.Agent(hyperparams["sync_frame"], hyperparams["buffer_length"], action_size, observation_size, hyperparams["discount_factor"],
                            hyperparams["learning_rate"], hyperparams["batch_size"], hyperparams["update_rate"])
    writer = SummaryWriter(comment="-" + behavior_name)
    
    start_time = time.time

    def update_epsilon():
        return max(hyperparams["epsilon_end"], hyperparams["epsilon_start"] - DQN_agent.t_frame / hyperparams["epsilon_decay_last_frame"])

    while not done_t:
        env.reset()
        decision_steps, terminal_steps = env.get_steps(behavior_name)
        episode_reward = 0
        tracked_agent = decision_steps.agent_id[0]
        done = False
        obs = decision_steps[0][0][0]

        while not done:
            decision_steps, terminal_steps = env.get_steps(behavior_name)   
            obs = decision_steps[0][0][0]

            state_v = torch.tensor(obs).to(device)
            action = DQN_agent.act(obs, epsilon)   

            env.set_actions(behavior_name, action)
            env.step()

            decision_steps, terminal_steps = env.get_steps(behavior_name)   
            if tracked_agent in decision_steps:
                reward = decision_steps[tracked_agent].reward
                episode_reward += reward

            if tracked_agent in terminal_steps:
                reward = terminal_steps[tracked_agent].reward
                episode_reward += reward
                done=True
            
            new_state = decision_steps[0][0][0]
            DQN_agent.step(Experience(state_v, torch.tensor(action).to(device), torch.tensor(reward).to(device), torch.tensor(done).to(device), torch.tensor(new_state).to(device)))
        
        print(DQN_agent.t_frame + ": done ", + len(rewards) + " games, reward: " + m_reward + " eps: " + epsilon)

        epsilon = update_epsilon()
        rewards.append(episode_reward)
        
        writer.add_scalar("epsilon", epsilon, DQN_agent.t_frame)
        writer.add_scalar("reward_100", m_reward, DQN_agent.t_frame)
        writer.add_scalar("reward", reward, DQN_agent.t_frame)

        m_reward = np.mean(rewards[-hyperparams["m_reward_length"]:])
        if m_reward > hyperparams["solved_score"]:
            print("Solved in " + DQN_agent.t_frame + " frames!")
            done_t = True

    
    train_time = time.time - start_time
    print("Training took " + train_time)
    writer.close()



if __name__ == "__main__":
    main()