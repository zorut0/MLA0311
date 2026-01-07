import numpy as np
import random
from collections import deque
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers

state_size = 3
action_size = 3
memory = deque(maxlen=2000)

gamma = 0.95
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
learning_rate = 0.001

def build_model():
    model = models.Sequential([
        layers.Dense(24, input_dim=state_size, activation='relu'),
        layers.Dense(24, activation='relu'),
        layers.Dense(action_size, activation='linear')
    ])
    model.compile(loss='mse', optimizer=optimizers.Adam(learning_rate))
    return model

model = build_model()
target_model = build_model()
target_model.set_weights(model.get_weights())

def simulate_env(state, action):
    speed, distance, lane = state
    if action == 2: speed += 1
    if action == 0: speed -= 1
    distance = max(distance - speed, 0)
    reward = 2 if distance > 5 else -5
    done = distance <= 0
    return np.array([speed, distance, lane]), reward, done

# Training
episodes = 200
batch_size = 32

for e in range(episodes):
    state = np.array([1, 20, 1])
    state = np.reshape(state, [1, state_size])
    for time in range(200):
        if np.random.rand() <= epsilon:
            action = random.randrange(action_size)
        else:
            action = np.argmax(model.predict(state, verbose=0))

        next_state, reward, done = simulate_env(state[0], action)
        next_state = np.reshape(next_state, [1, state_size])
        memory.append((state, action, reward, next_state, done))
        state = next_state

        if len(memory) > batch_size:
            minibatch = random.sample(memory, batch_size)
            for s, a, r, s_next, d in minibatch:
                target = r
                if not d:
                    target = r + gamma * np.max(target_model.predict(s_next, verbose=0))
                target_f = model.predict(s, verbose=0)
                target_f[0][a] = target
                model.fit(s, target_f, epochs=1, verbose=0)

        if done:
            break

    if epsilon > epsilon_min:
        epsilon *= epsilon_decay
    target_model.set_weights(model.get_weights())

    print(f"Episode {e+1}/200 completed, epsilon={epsilon:.3f}")
