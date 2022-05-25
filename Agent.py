from collections import deque

import numpy as np

from network import *


class QAgent:
    def __init__(self, n_actions, lr, gamma, epsilon, epsilon_min, epsilon_max, batch_size, max_mem_length):
        self.n_actions = n_actions
        self.lr = lr  # learning rate for optimizer
        self.gamma = gamma  # Discount factor for past rewards
        self.epsilon = epsilon  # Epsilon greedy parameter
        self.epsilon_min = epsilon_min  # Minimum epsilon greedy parameter
        self.epsilon_max = epsilon_max  # Maximum epsilon greedy parameter
        self.epsilon_interval = (
                epsilon_max - epsilon_min)  # Rate at which to reduce chance of random action being taken
        self.batch_size = batch_size
        self.epsilon_random_frames = 50000  # Number of frames for exploration
        self.epsilon_greedy_frames = 1000000.0  # Maximum replay length
        self.model = set_up_nn(n_actions)
        self.model_target = set_up_nn(n_actions)
        self._memory = deque(maxlen=max_mem_length)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0)
        self.loss_function = tf.keras.losses.Huber()

    @property
    def memory(self):
        return self._memory

    def step(self, state, frame_count):
        if frame_count < self.epsilon_random_frames or self.epsilon > np.random.rand(1)[0]:
            # Take random action
            action = np.random.choice(self.n_actions)
        else:
            # Predict action Q-values
            # From environment state
            state_tensor = tf.convert_to_tensor(state)
            state_tensor = tf.expand_dims(state_tensor, 0)
            action_probs = self.model(state_tensor, training=False)
            # Take best action
            action = tf.argmax(action_probs[0]).numpy()

        # Decay probability of taking random action
        self.epsilon -= self.epsilon_interval / self.epsilon_greedy_frames
        self.epsilon = max(self.epsilon, self.epsilon_min)

        return action

    def remember(self, state, action, state_next, done, reward):
        self.memory.append((state, action, state_next, done, reward))

    def update(self):
        # Get indices of samples for replay buffers
        indices = np.random.choice(range(len(self._memory)), size=self.batch_size)

        # Using list comprehension to sample from replay buffer
        state_sample = np.array([self._memory[i][0] for i in indices])
        state_next_sample = np.array([self._memory[i][2] for i in indices])
        rewards_sample = [self._memory[i][4] for i in indices]
        action_sample = [self._memory[i][1] for i in indices]
        done_sample = tf.convert_to_tensor(
            [float(self._memory[i][3]) for i in indices]
        )
        #sampled_memory = np.array(random.sample(self.memory, self.batch_size))
        # Using list comprehension to sample from replay buffer
        #state_sample = sampled_memory[:, 0]
        #action_sample = list(sampled_memory[:, 1])
        #state_next_sample = sampled_memory[:, 2]
        #done_sample = tf.convert_to_tensor(list(sampled_memory[:, 3].astype(float)))
        #rewards_sample = list(sampled_memory[:, 4])

        # Build the updated Q-values for the sampled future states
        # Use the target model for stability
        future_rewards = self.model_target.predict(state_next_sample)
        # Q value = reward + discount factor * expected future reward
        updated_q_values = rewards_sample + self.gamma * tf.reduce_max(
            future_rewards, axis=1
        )

        # If final frame set the last value to -1
        updated_q_values = updated_q_values * (1 - done_sample) - done_sample

        # Create a mask, so we only calculate loss on the updated Q-values
        masks = tf.one_hot(action_sample, self.n_actions)

        with tf.GradientTape() as tape:
            # Train the model on the states and updated Q-values
            q_values = self.model(state_sample)

            # Apply the masks to the Q-values to get the Q-value for action taken
            q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
            # Calculate loss between new Q-value and old Q-value
            loss = self.loss_function(updated_q_values, q_action)

        # Backpropagation
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

    def update_target_network(self):
        # update the target network with new weights
        self.model_target.set_weights(self.model.get_weights())

    def handle_mem(self):
        self._memory.popleft()

    def save_network(self, filepath):
        self.model.save(filepath)
