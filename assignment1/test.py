def test_policy_evaluation():

    import numpy as np
    import vi_and_pi as vp

    env = vp.gym.make("Deterministic-4x4-FrozenLake-v0")

    P = env.P
    nS = env.nS
    nA = env.nA

    policy = np.asarray([0, 3, 0, 1, 1, 2, 3, 1, 1, 2, 3, 1, 1, 1, 1, 2], dtype=np.int32)

    value_function = vp.policy_evaluation(P, nS, nA, policy)

    print(value_function)

    return None

if __name__ == "__main__":
    test_policy_evaluation()
