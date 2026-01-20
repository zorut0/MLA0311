import numpy as np

# ----- Simulated Patient Environment -----

class HealthcareEnv:
    def __init__(self):
        # hidden true effectiveness of 3 treatments
        self.true_weights = np.array([
            [0.6, -0.2, 0.1],   # Treatment A
            [0.3,  0.4, -0.1],  # Treatment B
            [0.1,  0.2, 0.5]    # Treatment C
        ])

    def get_reward(self, context, action):
        # probabilistic outcome
        mean = np.dot(self.true_weights[action], context)
        return 1 if np.random.rand() < self.sigmoid(mean) else 0

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


# ----- Contextual Thompson Sampling Bandit -----

class SafeContextualBandit:
    def __init__(self, n_actions=3, n_features=3):
        self.n_actions = n_actions
        self.A = [np.identity(n_features) for _ in range(n_actions)]
        self.b = [np.zeros(n_features) for _ in range(n_actions)]

        # safety: block risky treatment for severe patients
        self.risk_threshold = 0.7

    def select_action(self, context):
        samples = []

        for a in range(self.n_actions):
            theta = np.linalg.inv(self.A[a]).dot(self.b[a])
            sample = np.random.normal(theta.dot(context), 0.1)

            # ---- Ethical Constraint ----
            if context[1] > self.risk_threshold and a == 2:
                sample = -np.inf   # block unsafe exploration

            samples.append(sample)

        return np.argmax(samples)

    def update(self, action, context, reward):
        self.A[action] += np.outer(context, context)
        self.b[action] += reward * context


# ----- Training Simulation with Metrics -----

env = HealthcareEnv()
bandit = SafeContextualBandit()

rewards = []
actions = []

for patient in range(5000):

    context = np.random.rand(3)

    action = bandit.select_action(context)
    reward = env.get_reward(context, action)

    bandit.update(action, context, reward)

    rewards.append(reward)
    actions.append(action)

print("Learning completed safely.")
print("Average recovery rate:", np.mean(rewards))

for a in range(3):
    print(f"Treatment {a} chosen {actions.count(a)} times")
