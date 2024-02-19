import numpy as np

def sample_motion_plan(motion_plan, sampling_time):
    """
    Sample motion plan at given sampling time.

    :param motion_plan: List of [left_wheel_rpm, right_wheel_rpm, duration_seconds] lists
    :param sampling_time: Sampling time (seconds)
    :return: List of [left_wheel_rpm, right_wheel_rpm, duration_seconds] lists
    """
    sampled_motion_plan = []
    for motion in motion_plan:
        n_samples = int(motion[2] / sampling_time)
        for _ in range(n_samples):
            sampled_motion_plan.append(motion[:2])
    return sampled_motion_plan

def get_gaussian_noise_sample(mu, sigma):
    """
    Get a random sample from a 1D Gaussian distribution with mean mu and standard deviation sigma.

    :param mu: mean of distribution
    :param sigma: standard deviation
    :return: random sample from distribution with given parameters
    """
    size = 1 
    if np.isscalar(mu):
        return np.random.normal(loc=mu, scale=sigma, size=1)

    if mu.ndim == 1:
            size = (mu.shape[0], 1)
    else:
            size = mu.shape
    return np.random.normal(loc=mu.reshape(size), scale=sigma, size=size).reshape(mu.shape)

def make_shifted_matrix(vector):
     indices = np.arange(len(vector))

     shifted_matrix = vector[(indices[:, None] + indices) % len(vector)]

     return shifted_matrix