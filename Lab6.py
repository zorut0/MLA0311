import numpy as np
import random
import math

num_ads = 5
num_rounds = 10000

true_probs = [0.05, 0.03, 0.08, 0.04, 0.06]

def simulate_click(ad):
    return 1 if random.random() < true_probs[ad] else 0

epsilon = 0.1
eg_rewards = [0]*num_ads
eg_counts = [0]*num_ads
eg_total_reward = 0

for t in range(num_rounds):
    if random.random() < epsilon:
        ad = random.randint(0, num_ads-1)
    else:
        ad = np.argmax([eg_rewards[i]/eg_counts[i] if eg_counts[i] > 0 else 0 for i in range(num_ads)])
    
    reward = simulate_click(ad)
    eg_total_reward += reward
    eg_counts[ad] += 1
    eg_rewards[ad] += reward

ucb_rewards = [0]*num_ads
ucb_counts = [0]*num_ads
ucb_total_reward = 0

for t in range(num_rounds):
    if t < num_ads:
        ad = t
    else:
        ucb_values = [
            (ucb_rewards[i]/ucb_counts[i]) + math.sqrt(2 * math.log(t) / ucb_counts[i])
            for i in range(num_ads)
        ]
        ad = np.argmax(ucb_values)
    
    reward = simulate_click(ad)
    ucb_total_reward += reward
    ucb_counts[ad] += 1
    ucb_rewards[ad] += reward

alpha = [1]*num_ads
beta = [1]*num_ads
ts_total_reward = 0

for t in range(num_rounds):
    sampled_probs = [np.random.beta(alpha[i], beta[i]) for i in range(num_ads)]
    ad = np.argmax(sampled_probs)
    reward = simulate_click(ad)
    ts_total_reward += reward
    
    if reward == 1:
        alpha[ad] += 1
    else:
        beta[ad] += 1

print("Total Rounds:", num_rounds)
print("Epsilon-Greedy CTR:", eg_total_reward / num_rounds)
print("UCB CTR:", ucb_total_reward / num_rounds)
print("Thompson Sampling CTR:", ts_total_reward / num_rounds)
