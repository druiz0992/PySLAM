
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
            sampled_motion_plan.append(motion)
    return sampled_motion_plan