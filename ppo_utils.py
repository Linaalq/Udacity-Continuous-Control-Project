import matplotlib
import matplotlib.pyplot as plt
import torch
import numpy as np
import random as rand
import torch.nn as nn

def collect_trajectories(env, policy, tmax=2000, nrand=2):
    
    #initialize returning lists and start the game!
    state_list=[]
    reward_list=[]
    prob_list=[]
    action_list=[]
    value_list=[]
    is_done_list=[]
    
    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    
    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]
    
    # number of parallel instances
    n = len(env_info.agents)
    
    #get first states
    state = env_info.vector_observations
    
    while True:
        action, log_prob, _, v = policy(state)
        action_list.append(action.detach())
        log_prob = log_prob.detach()
        v = v.detach()
        env_info = env.step(action.cpu().numpy())[brain_name]
        next_state = env_info.vector_observations
        reward = np.asarray(env_info.rewards)
        reward = torch.from_numpy(reward).unsqueeze(1)
        is_done = np.array([1 if t else 0 for t in env_info.local_done])
        is_done = torch.Tensor(1-is_done).unsqueeze(1)
        
        # store the result
        state_list.append(state)
        reward_list.append(reward)
        prob_list.append(log_prob)
        is_done_list.append(is_done)
        value_list.append(v)
        state = next_state
        
        # stop if any of the trajectories is done
        # we want all the lists to be retangular
        if np.any(env_info.local_done):
            break
    
    return prob_list, state_list, reward_list, action_list, value_list, is_done_list

def clipped_surrogate(policy, old_probs, states, rewards, actions, values, is_done, optimizer, config):
    
    discount= config['hyperparameters']['discount_rate']
    epsilon= config['hyperparameters']['epsilon']
    beta= config['hyperparameters']['beta']
    tau= config['hyperparameters']['tau']
    gradient_clip= config['hyperparameters']['gradient_clip']
    device = config['pytorch']['device']
    
    r = torch.tensor(values[-1], dtype=torch.float, device=device)
    returns = [0 for i in range(len(values)-1)]
    advantages = [0 for i in range(len(values)-1)]
    a = torch.Tensor(np.zeros((20, 1))).float()
    
    for i in reversed(range(len(states) - 1)):
        
        next_value = values[i+1]
        r = rewards[i] + discount * is_done[i] * r
        td_error = rewards[i] + discount * next_value.detach() * is_done[i] - values[i].detach()
        a = a * tau * discount * is_done[i] + td_error
        returns[i] = r
        advantages[i] = a
    
    advantages = torch.cat(advantages, dim=0).float()
    advantages = (advantages - advantages.mean()) / advantages.std()
    
    values = values[:-1]
    states = np.asarray(states[:-1])
    actions = actions[:-1]
    states = torch.cat([torch.tensor(state).float() for state in states], dim=0)
    actions = torch.cat([torch.tensor(action).float() for action in actions], dim=0)
    old_probs = torch.cat(old_probs[:-1],dim=0).float()
    
    # run the policy
    _, new_probs, entropy_loss, new_values = policy(states, actions)
    
    #compute the policy loss
    ratio = (new_probs - old_probs).exp()
    obj = ratio * advantages
    obj_clipped = ratio.clamp(1.0 - epsilon, 1.0 + epsilon) * advantages
    policy_loss = -torch.min(obj, obj_clipped).mean(0) - beta * entropy_loss.mean().float()
    
    #compute the value loss
    value_loss = 0.5 * (torch.cat(returns, dim=0) - new_values).pow(2).mean()

    optimizer.zero_grad()
    (policy_loss + value_loss).backward()
    nn.utils.clip_grad_norm_(policy.parameters(), gradient_clip)
    optimizer.step()
    return