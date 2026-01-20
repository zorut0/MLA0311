import numpy as np
import random

N = 5
actions = ["up", "down", "left", "right"]

goal = (4,4)
start = (0,0)

# Belief state
belief = np.ones((N,N)) / (N*N)

def transition(state, action):
    x,y = state
    if action == "up": x = max(0, x-1)
    if action == "down": x = min(N-1, x+1)
    if action == "left": y = max(0, y-1)
    if action == "right": y = min(N-1, y+1)

    # motion noise
    if random.random() < 0.2:
        x,y = state
    return (x,y)

def observation(state):
    # noisy distance to goal
    gx,gy = goal
    sx,sy = state
    dist = abs(gx-sx) + abs(gy-sy)
    return dist + random.choice([-1,0,1])

def update_belief(belief, action, obs):
    new_belief = np.zeros((N,N))

    for i in range(N):
        for j in range(N):
            # prediction
            prev = (i,j)
            next_state = transition(prev, action)

            # observation likelihood
            expected = abs(goal[0]-i)+abs(goal[1]-j)
            prob_obs = 0.8 if obs==expected else 0.2

            new_belief[next_state] += belief[i,j] * prob_obs

    new_belief /= np.sum(new_belief)
    return new_belief

def choose_action(belief):
    # simple heuristic: move toward most probable cell
    x,y = np.unravel_index(np.argmax(belief), belief.shape)

    if x < goal[0]: return "down"
    if x > goal[0]: return "up"
    if y < goal[1]: return "right"
    return "left"

# Simulation
state = start
steps = 0

print("Robot Navigation Using POMDP\n")

while state != goal and steps < 50:
    obs = observation(state)
    action = choose_action(belief)
    belief = update_belief(belief, action, obs)
    state = transition(state, action)

    print(f"Step {steps}: State={state} Action={action}")
    steps += 1

print("\nGoal Reached in", steps, "steps")
