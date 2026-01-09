import numpy as np
import random
from collections import deque
import tensorflow as tf
from tensorflow.keras import layers

state_dim = 5
action_dim = 3
memory = deque(maxlen=5000)
gamma = 0.95
tau = 0.005

def build_actor():
    inp = layers.Input(shape=(state_dim,))
    x = layers.Dense(32, activation='relu')(inp)
    x = layers.Dense(32, activation='relu')(x)
    out = layers.Dense(action_dim, activation='tanh')(x)
    return tf.keras.Model(inp, out)

def build_critic():
    s = layers.Input(shape=(state_dim,))
    a = layers.Input(shape=(action_dim,))
    x = layers.Concatenate()([s,a])
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dense(32, activation='relu')(x)
    out = layers.Dense(1)(x)
    return tf.keras.Model([s,a], out)

actor = build_actor()
critic = build_critic()
target_actor = build_actor()
target_critic = build_critic()
target_actor.set_weights(actor.get_weights())
target_critic.set_weights(critic.get_weights())

actor_optimizer = tf.keras.optimizers.Adam(0.001)
critic_optimizer = tf.keras.optimizers.Adam(0.002)

def simulate_user(state, action):
    reward = 2 if np.sum(action) > 0 else -1
    next_state = state + np.random.randn(state_dim)*0.1
    return next_state, reward

episodes = 300
batch_size = 64

for ep in range(episodes):
    state = np.random.randn(state_dim)
    for step in range(50):
        action = actor(np.expand_dims(state,0))[0].numpy()
        next_state, reward = simulate_user(state, action)
        memory.append((state, action, reward, next_state))
        state = next_state

        if len(memory) > batch_size:
            batch = random.sample(memory, batch_size)
            states, actions, rewards, next_states = map(np.array, zip(*batch))

            with tf.GradientTape() as tape:
                target_actions = target_actor(next_states)
                y = rewards + gamma * tf.squeeze(target_critic([next_states, target_actions]))
                critic_value = tf.squeeze(critic([states, actions]))
                critic_loss = tf.keras.losses.MSE(y, critic_value)

            grads = tape.gradient(critic_loss, critic.trainable_variables)
            critic_optimizer.apply_gradients(zip(grads, critic.trainable_variables))

            with tf.GradientTape() as tape:
                new_actions = actor(states)
                actor_loss = -tf.reduce_mean(critic([states, new_actions]))

            grads = tape.gradient(actor_loss, actor.trainable_variables)
            actor_optimizer.apply_gradients(zip(grads, actor.trainable_variables))

            # Soft update
            for t, e in zip(target_actor.variables, actor.variables):
                t.assign(t*(1-tau) + e*tau)
            for t, e in zip(target_critic.variables, critic.variables):
                t.assign(t*(1-tau) + e*tau)

    if ep % 50 == 0:
        print("Episode", ep, "completed")
