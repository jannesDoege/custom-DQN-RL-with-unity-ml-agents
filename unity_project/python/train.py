from mlagents_envs.base_env import ActionTuple
import torch
import numpy as np
from mlagents_envs.environment import UnityEnvironment as UE, BehaviorName, BehaviorSpec
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from tensorboardX import SummaryWriter
import time

import agent
from hyperparams import hyperparams
from buffer import Experience

#if this is set to None you have to click the play button in the unity scene after creating the env
env_file_name = None

device = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    channel = EngineConfigurationChannel()   

    if env_file_name == None:
        print("please click the play button in your unity scene")
    env = UE(file_name=env_file_name, side_channels=[channel], no_graphics=True)
    if env_file_name == None:
        print("environment initialized - if you pause or close the unity scene training will stop")
    channel.set_configuration_parameters(time_scale = 10.0)
    
    env.reset()



    behavior_name = list(env.behavior_specs)[0]
    spec = env.behavior_specs[behavior_name]
    
    decision_steps, terminal_steps = env.get_steps(behavior_name)
    observation_size = len(decision_steps[0][0][0])
    action_size = len(decision_steps.action_mask)
    done_t = False
    rewards = []
    best_m_reward = None
    epsilon = hyperparams["epsilon_start"]
    loss = None

    DQN_agent = agent.Agent(hyperparams["sync_frame"], hyperparams["replay_buffer_size"], action_size, observation_size, hyperparams["discount_factor"],
                            hyperparams["learning_rate"], hyperparams["batch_size"], hyperparams["update_rate"])
    print(behavior_name[:-7])
    writer = SummaryWriter(comment="-" + behavior_name[:-7])
    
    start_time = time.time()

    print(DQN_agent.net)

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

            state_v = obs
            action = DQN_agent.act(obs, epsilon) 

            actions = np.zeros((1, 4), dtype=np.int32)
            actions[0][action] = 1

            action_tup = ActionTuple(discrete=actions)

            env.set_actions(behavior_name, action_tup)
            env.step()

            decision_steps, terminal_steps = env.get_steps(behavior_name)   
            if tracked_agent in decision_steps:
                reward = decision_steps[tracked_agent].reward
                episode_reward += reward

            if tracked_agent in terminal_steps:
                reward = terminal_steps[tracked_agent].reward
                episode_reward += reward
                done=True
            
            if not done:
                new_state = decision_steps[0][0][0]
            else:
                new_state = terminal_steps[0][0][0]
            loss = DQN_agent.step(Experience(state_v, action, reward, done, new_state))

        m_reward = np.mean(rewards[-hyperparams["m_reward_length"]:])
        
        epsilon = update_epsilon()
        rewards.append(episode_reward)
        
        print(str(DQN_agent.t_frame) + " frames done " + str(len(rewards)) + " games, reward " + str(m_reward) + " eps " + str(epsilon) + " loss: " + str(loss))



        writer.add_scalar("epsilon", epsilon, DQN_agent.t_frame)
        writer.add_scalar("reward_100", m_reward, DQN_agent.t_frame)
        writer.add_scalar("reward", reward, DQN_agent.t_frame)
        if loss is not None:
            writer.add_scalar("loss", loss, DQN_agent.t_frame)
        
        if len(rewards)  < hyperparams["m_reward_length"]:
            continue

        if m_reward >= hyperparams["solved_score"]:
            print("Solved in " + str(DQN_agent.t_frame) + " frames!")
            done_t = True

    
    train_time = time.time() - start_time
    torch.save(DQN_agent.target_net, "final_net.pt")

    print("Training took " + str(train_time))
    env.close()
    writer.close()



if __name__ == "__main__":
    main()