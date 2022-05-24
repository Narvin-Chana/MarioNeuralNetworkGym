import gym

"""
    Allows for the skipping of frames to simplify calculations and gain compute time.
"""


class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = False
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info


"""
    Allows for additional features to be added to the environment.
    Code was sourced from the baselines given by OpenAI at: 
    https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py
    
    We updated it to work with the latest version of Gym and changed it to finish the episode on death.
"""


class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode
        """
        gym.Wrapper.__init__(self, env)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if self.env.unwrapped._is_dying:
            # Also checks for if the player is dead
            done = True
        return obs, reward, done, info

    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        obs = self.env.reset(**kwargs)
        return obs


def wrapper(env, skip_count):
    """Apply a common set of wrappers for games."""
    # Removed since episodes being of a fixed duration was preferred.
    # env = EpisodicLifeEnv(env)
    env = SkipFrame(env, skip_count)
    return env
