import numpy as np
import torch


class Normalizer:
    """
    Normalizes the observations based on running statistics.
    Based on Gym's NormalizeObservation wrapper.
    """
    def __init__(self, obs_shape, eps=1e-8):
        self.eps = eps
        self.obs_shape = obs_shape
        self.obs_rms = RunningMeanStd(shape=obs_shape)

    def normalize(self, obs, update = True):
        # update the running mean and variance
        if update:
            self.obs_rms.update(obs)
        return (obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.eps)


class RunningMeanStd:
    "Tracks the mean, variance, and count of values"

    def __init__(self, epsilon=1e-4, shape=()):
        """Tracks the mean, variance and count of values."""
        self.mean = np.zeros(shape, "float64")
        self.var = np.ones(shape, "float64")
        self.count = epsilon

    def update(self, x):
        """Updates the mean, var and count from a batch of samples."""
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        """Updates from batch mean, variance and count moments."""
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count
        )
def update_mean_var_count_from_moments(
    mean, var, count, batch_mean, batch_var, batch_count
):
    """Updates the mean, var and count using the previous mean, var, count and batch values."""
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count


class CriticTargetScaler:
    """
    Target is the GAE estimate of the true value of the state. The problem is this can be quite large since we can have large rewards.
    We want to scale the target values to be more reasonable so that the critic can learn better.
    """

    def __init__(self, obs_shape, update_rate=0.01, epsilon=1e-8):
        self.obs_shape = obs_shape
        self.target_mean = None
        self.target_std = None
        self.update_rate = update_rate
        self.epsilon = epsilon

    def update(self, target_values):
        # Target values are based on GAE estimates of the values
        if self.target_mean is None:
            self.target_mean = target_values.mean().item()
            self.target_std = target_values.std().item()
        else:
            self.target_mean = (1 - self.update_rate) * self.target_mean + self.update_rate * target_values.mean().item()
            self.target_std = (1 - self.update_rate) * self.target_std + self.update_rate * target_values.std().item()

    def scale(self, target_values):
        return target_values * self.target_std + self.target_mean

    def normalize(self, target_values):
        return (target_values - self.target_mean) / max(self.target_std, self.epsilon)



