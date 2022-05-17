import os
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from network import *
env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, COMPLEX_MOVEMENT)


def main():
    """
    An example pipeline of applying Q learning.
    Retrieved largely from: https://keras.io/examples/rl/deep_q_network_breakout/
    There are more techniques of course.
    Currently, network returns sampled numbers which still need to be translated to the actions
    known to the environment
    """
    file_dir = os.getcwd()
    network_filepath = os.path.join(file_dir, 'model.h5')
    # Could be used if we already have partially trained the network and save an intermediary best
    # best_network = keras.models.load_model(network_filepath)
    n_actions = 12
    gamma = 0.99  # Discount factor for past rewards
    epsilon = 1.0  # Epsilon greedy parameter
    epsilon_min = 0.1  # Minimum epsilon greedy parameter
    epsilon_max = 1.0  # Maximum epsilon greedy parameter
    epsilon_interval = (
            epsilon_max - epsilon_min
    )  # Rate at which to reduce chance of random action being taken
    batch_size = 32  # Size of batch taken from replay buffer
    max_steps_per_episode = 10000
    model = set_up_nn()
    model_target = set_up_nn()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)
    # Using huber loss for stability
    loss_function = tf.keras.losses.Huber()

    action_history = []
    state_history = []
    state_next_history = []
    rewards_history = []
    done_history = []
    episode_reward_history = []
    running_reward = 0
    frame_count = 0
    max_episodes = 10000
    # Number of frames to take random action and observe output
    epsilon_random_frames = 50000
    # Number of frames for exploration
    epsilon_greedy_frames = 1000000.0
    # Maximum replay length
    # Note: The Deepmind paper Mnih et al. (2013) suggests 1000000 however this causes memory issues
    max_memory_length = 100000
    # Train the model after 4 actions
    update_after_actions = 4
    # How often to update the target network
    update_target_network = 10000

    for episode_count in range(max_episodes):
        state = env.reset()
        episode_reward = 0

        for timestep in range(1, max_steps_per_episode):
            frame_count += 1

            # Use epsilon-greedy for exploration
            if frame_count < epsilon_random_frames or epsilon > np.random.rand(1)[0]:
                # Take random action
                action = np.random.choice(n_actions)
            else:
                # Predict action Q-values
                # From environment state
                state_tensor = tf.convert_to_tensor(state)
                state_tensor = tf.expand_dims(state_tensor, 0)
                action_probs = model(state_tensor, training=False)
                # Take best action
                action = tf.argmax(action_probs[0]).numpy()

            # Decay probability of taking random action
            epsilon -= epsilon_interval / epsilon_greedy_frames
            epsilon = max(epsilon, epsilon_min)

            # Apply the sampled action in our environment
            state_next, reward, done, info = env.step(action)
            # env.render()
            state_next = np.array(state_next)

            episode_reward += reward

            # Save actions and states in replay buffer
            action_history.append(action)
            state_history.append(state)
            state_next_history.append(state_next)
            done_history.append(done)
            rewards_history.append(reward)
            state = state_next

            # Update every fourth frame and once batch size is over 32
            if frame_count % update_after_actions == 0 and len(done_history) > batch_size:

                # Get indices of samples for replay buffers
                indices = np.random.choice(range(len(done_history)), size=batch_size)

                # Using list comprehension to sample from replay buffer
                state_sample = np.array([state_history[i] for i in indices])
                state_next_sample = np.array([state_next_history[i] for i in indices])
                rewards_sample = [rewards_history[i] for i in indices]
                action_sample = [action_history[i] for i in indices]
                done_sample = tf.convert_to_tensor(
                    [float(done_history[i]) for i in indices]
                )

                # Build the updated Q-values for the sampled future states
                # Use the target model for stability
                future_rewards = model_target.predict(state_next_sample)
                # Q value = reward + discount factor * expected future reward
                updated_q_values = rewards_sample + gamma * tf.reduce_max(
                    future_rewards, axis=1
                )

                # If final frame set the last value to -1
                updated_q_values = updated_q_values * (1 - done_sample) - done_sample

                # Create a mask, so we only calculate loss on the updated Q-values
                masks = tf.one_hot(action_sample, n_actions)

                with tf.GradientTape() as tape:
                    # Train the model on the states and updated Q-values
                    q_values = model(state_sample)

                    # Apply the masks to the Q-values to get the Q-value for action taken
                    q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
                    # Calculate loss between new Q-value and old Q-value
                    loss = loss_function(updated_q_values, q_action)

                # Backpropagation
                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

            if frame_count % update_target_network == 0:
                # update the target network with new weights
                model_target.set_weights(model.get_weights())
                # Log details
                template = "running reward: {:.2f} at episode {}, frame count {}"
                print(template.format(running_reward, episode_count, frame_count))

            # Limit the state and reward history
            if len(rewards_history) > max_memory_length:
                del rewards_history[:1]
                del state_history[:1]
                del state_next_history[:1]
                del action_history[:1]
                del done_history[:1]

            if done:
                break

        # Update running reward to check condition for solving
        episode_reward_history.append(episode_reward)
        if len(episode_reward_history) > 100:
            del episode_reward_history[:1]
        running_reward = np.mean(episode_reward_history)

        if running_reward > 40:  # Condition to consider the task solved
            print("Solved at episode {}!".format(episode_count))
            break

    # Save best network
    model.save(network_filepath)
    env.close()


main()
