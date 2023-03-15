#!/usr/bin/env python3

################################################################################
#
# Project 1 - 10 armed bandit testbed
# Get plots for three different methods: greedy, epsilon = 0.1 and
#   epsilon = 0.01
#
################################################################################

import random

import matplotlib.pyplot as plt
import numpy as np

RANDOM_SEED = 42
RANDOM_SEED = 421
NUM_ARMS = 10
STEPS_PER_RUN = 2000
NUM_BEDS = 2000

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

def main():
    mean_master = 0
    variance = 1

    avg_reward = np.zeros((3, STEPS_PER_RUN))
    avg_percent_opt_act = np.zeros((3, STEPS_PER_RUN))

    for bed in range(NUM_BEDS):
        if bed % 100 == 0:
            # print(f"testBed: {bed}", end='\r')
            print(f"testBed: {bed}")

        # initial conditions
        mean_arms = np.random.normal(mean_master, variance, size=(NUM_ARMS,))
        optimal_action = np.where(np.max(mean_arms) == mean_arms)[0][0]
        Qt = np.zeros((3, NUM_ARMS))
        step_reward = np.zeros((3, STEPS_PER_RUN))
        step_percent_opt_act = np.zeros((3, STEPS_PER_RUN))

        counter = np.zeros((3, NUM_ARMS), dtype='int')

        for step in range(1, STEPS_PER_RUN + 1):
            # greedy
            idx = 0
            sel_arm = epsilon_greedy_selection(Qt[idx])
            reward = update_values(sel_arm, mean_arms, variance, counter[idx],
                                Qt[idx])
            step_reward[idx, step - 1] = reward
            step_percent_opt_act[idx, step - 1] = \
                counter[idx, optimal_action] / step

            # epsilon = 0.1
            idx = 1
            eps = 0.1
            sel_arm = epsilon_greedy_selection(Qt[idx], epsilon=eps)
            reward = update_values(sel_arm, mean_arms, variance, counter[idx],
                                Qt[idx])
            step_reward[idx, step - 1] = reward
            step_percent_opt_act[idx, step - 1] = \
                counter[idx, optimal_action] / step

            # epsilon = 0.01
            idx = 2
            eps = 0.01
            sel_arm = epsilon_greedy_selection(Qt[idx], epsilon=eps)
            reward = update_values(sel_arm, mean_arms, variance, counter[idx],
                                Qt[idx])
            step_reward[idx, step - 1] = reward
            step_percent_opt_act[idx, step - 1] = \
                counter[idx, optimal_action] / step

        avg_reward += step_reward
        avg_percent_opt_act += step_percent_opt_act

        if bed == 100:
            print(counter)
            print(np.sum(counter, axis=1))
            print(optimal_action)
            print()
            break

    avg_reward /= NUM_BEDS
    avg_percent_opt_act /= NUM_BEDS
    plt.figure()
    plt.title("Average Reward")
    plt.plot(avg_reward[0], '-g')
    plt.plot(avg_reward[1], '-b')
    plt.plot(avg_reward[2], '-r')
    plt.ylabel("reward")
    plt.xlabel("time steps")

    plt.figure()
    plt.title("% Optimal Action")
    plt.plot(avg_percent_opt_act[0], '-g')
    plt.plot(avg_percent_opt_act[1], '-b')
    plt.plot(avg_percent_opt_act[2], '-r')
    plt.ylabel("% optimal action")
    plt.xlabel("time steps")
    plt.show()

def epsilon_greedy_selection(Qt, epsilon=0):
    """
    Select an arm greedily (highest estimated value)

    Parameters
    ----------
    Qt : np.ndarray
        estimated values for each arm
    epsilon : float
        probability of exploring

    Returns
    -------
    sel_arm : int
        selected arm
    """

    if epsilon == 0 or random.random() > epsilon:
        sel_arm = np.where(np.max(Qt) == Qt)[0]
        # check for ties
        if sel_arm.shape[0] > 1:
            arm_idx = random.randint(0, sel_arm.shape[0] - 1)
            sel_arm = sel_arm[arm_idx]
        else:
            sel_arm = sel_arm[0]
    else:
        sel_arm = random.randint(0, Qt.shape[0] - 1)

    return sel_arm


def update_values(sel_arm, mean_arms, variance, counter, Qt):
    """
    Sample from `sel_arm` distribution and update its estimated value

    Parameters
    ----------
    sel_arm : int
        selected arm to use
    mean_arms : np.ndarray
        means of the distribution of each arm
    variance : float
        variance of all arms distributions
    counter : np.ndarray
        counter of how many times each arm has been selected
    Qt : np.ndarray
        estimated values for each arm

    Returns
    -------
    step_reward : float
        the reward obtained in this timestep
    """
    step_reward = np.random.normal(mean_arms[sel_arm], variance)
    counter[sel_arm] += 1
    Qt[sel_arm] += 1 / counter[sel_arm] * (step_reward - Qt[sel_arm])

    return step_reward


if __name__ == "__main__":
    main()
