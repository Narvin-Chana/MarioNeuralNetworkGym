from collections import deque

import numpy as np

from network import *


class QAgent:
    def __init__(self, n_actions, lr, gamma, epsilon, epsilon_decay, epsilon_min, epsilon_max, batch_size, max_mem_length):
        self.n_actions = n_actions
        self.lr = lr  # learning rate for optimizer
        self.gamma = gamma  # Discount factor for past rewards
        self.epsilon = epsilon  # Epsilon greedy parameter
        self.epsilon_min = epsilon_min  # Minimum epsilon greedy parameter
        self.epsilon_max = epsilon_max  # Maximum epsilon greedy parameter
        self.epsilon_decay = epsilon_decay  # Rate at which to reduce chance of random action being taken
        self.batch_size = batch_size
        self.epsilon_random_frames = 12000  # Number of frames for exploration
        self.epsilon_greedy_frames = 250000.0  # Maximum replay length
        self.model = set_up_nn(n_actions)
        self.model_target = set_up_nn(n_actions)
        self._memory = deque(maxlen=max_mem_length)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0)
        self.loss_function = tf.keras.losses.Huber()
        self.model.compile(loss=self.loss_function, optimizer=self.optimizer)

    @property
    def memory(self):
        """
        Getter for memory
        :return: memory
        """
        return self._memory

    def step(self, state, frame_count):
        """
        Act in the environment and decay epsilon parameter
        :param state: the current state of the game
        :param frame_count: how many frames have been counted
        :return: a random or the best action
        """
        if frame_count < self.epsilon_random_frames or self.epsilon > np.random.rand(1)[0]:
            # Take random action
            action = np.random.choice(self.n_actions)
        else:
            # Predict action Q-values
            # From environment state
            state_tensor = tf.convert_to_tensor(state)
            state_tensor = tf.expand_dims(state_tensor, 0)
            action_probs = self.model.predict(state_tensor)
            # Take best action
            action = tf.argmax(action_probs[0]).numpy()

        return action

    def remember(self, state, action, state_next, done, reward):
        """
        Save information for replay buffer
        :param state: current state of the game
        :param action: action taken in the current state of the game
        :param state_next: the state of the game after taking the specified action
        :param done: whether Mario has died or won
        :param reward: the reward obtained after taking the specified action
        """
        self.memory.append((state, action, state_next, done, reward))

    def update(self):
        """
        Experience replay sampled from memory
        Based on these experiences, build updated Q-values and calculate loss between new and old Q-values
        Finally, backwards pass over network and update it
        """
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

        # Decay probability of taking random action
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon, self.epsilon_min)

    def update_target_network(self):
        # update the target network with new weights
        self.model_target.set_weights(self.model.get_weights())

    def handle_mem(self):
        # clear up memory by removing oldest experience
        self._memory.popleft()

    def save_network(self, filepath):
        # Save the network to the specified path
        self.model.save(filepath)
