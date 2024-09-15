""" 
Implements a Bandit Class for the Multi-Armed Bandit Problem 
and a Bandit Algorithm Class which can handle different algorithms
including:
- Epsilon Greedy
- UCB
- Thompson Sampling
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta, norm, rv_continuous, rv_discrete
from collections.abc import Callable
from numba import jit

hist_dtype = [("arm", int), ("reward", float)]


class Bandit:

    def __init__(self, arms: list[rv_discrete | rv_continuous], num_rounds: int):
        """
        Args:
            arms (list): list of distributions for each arm
        """
        self.arms = arms
        self.best_arm: int = np.argmax([arm.mean() for arm in arms])
        # history of rewards and actions for each round
        self.history: list[(int, float)] = np.zeros(num_rounds, dtype=hist_dtype)
        self.sample_means, self.counts = np.zeros(len(arms)), np.zeros(len(arms))
        self.n_history = 0
        self.num_rounds = num_rounds
        self.n_arms = len(arms)

    def reset(self) -> None:
        self.history[:] = 0
        self.sample_means[:] = 0
        self.counts[:] = 0
        self.n_history = 0

    def pull(self, arm) -> float:
        reward = self.arms[arm].rvs()
        self.history[self.n_history] = (arm, reward)
        self.__update_sample_means(arm, reward)
        self.n_history += 1
        return reward

    def __update_sample_means(self, arm, reward) -> None:
        """
        Args:
            arm (int): arm to update
            reward (float): reward to update with
        """
        self.counts[arm] += 1
        self.sample_means[arm] = (
            self.sample_means[arm] * (self.counts[arm] - 1) + reward
        ) / self.counts[arm]

    def regret(self) -> float:
        return (
            self.arms[self.best_arm].mean() * self.n_history
            - self.history["reward"].sum()
        )

    def plot_history(self, algo_name: str) -> None:
        # plot cumulative rewards and actions
        cum_rewards = np.cumsum(self.history["reward"])
        fig, ax = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f"{algo_name} Algorithm", fontsize=14)
        ax[0].plot(cum_rewards)
        ax[0].plot(
            [0, len(self.history)],
            [0, self.arms[self.best_arm].mean() * len(self.history)],
            "r--",
        )
        ax[0].set_xlabel("Round")
        ax[0].set_ylabel("Cumulative Reward")
        ax[0].set_title("Cumulative Reward vs. Round")
        ax[0].legend(["Cumulative Reward", "Optimal Reward"])

        actions = [arm for arm, _ in self.history]
        ax[1].plot(actions, "r.", markersize=0.1)
        ax[1].set_yticks(range(len(self.arms)))
        ax[1].set_xlabel("Round")
        ax[1].set_ylabel("Action")
        ax[1].set_title("Action vs. Round")
        plt.show()


class BanditAlgorithm:
    def __init__(self, name: str, bandit_algo: Callable[[Bandit], int]):
        self.name = name
        self.bandit_algo = bandit_algo

    def run(self, bandit: Bandit, n_rounds: int, *args):
        """
        Args:
            bandit (Bandit): the bandit to run the algorithm on
            n_rounds (int): number of rounds to run the algorithm
        """
        for _ in range(n_rounds):
            arm = self.bandit_algo(bandit, *args)
            bandit.pull(arm)


def run_bandit_algorithm(
    name: str,
    bandit_algo_fn: Callable[[Bandit], int],
    bandit: Bandit,
    n_rounds: int,
    *args,
    **kwargs,
):
    """
    Args:
        name (str): name of the algorithm
        bandit_algo_fn (Callable): bandit algorithm function
        bernoullis (list): list of bernoulli distributions for each arm
        n_rounds (int): number of rounds to run the algorithm
        args (list): additional arguments for the bandit algorithm
        plot (bool): whether to plot the results, defauls to False
    """
    plot = kwargs.get("plot", False)
    bandit_algo = BanditAlgorithm(name, bandit_algo_fn)
    bandit_algo.run(bandit, n_rounds, *args)
    if plot:
        bandit.plot_history(name)
    regret = bandit.regret()
    if bandit.n_history == bandit.num_rounds:
        bandit.reset()
    return regret


def explore_then_commit(
    bandit: Bandit,
    explore_rounds: int,
) -> int:
    """
    Args:
        history (list): list of rewards and actions for past rounds
        explore_rounds (int): number of rounds to explore
    """

    if bandit.n_history < explore_rounds * bandit.n_arms:
        return bandit.n_history % bandit.n_arms
    else:
        return np.argmax(bandit.sample_means)


def epsilon_greedy(
    bandit: Bandit,
    epsilon: float,
) -> int:
    """
    Args:
        history (list): list of rewards and actions for past rounds
        epsilon (float): probability of exploration
    """
    if np.random.rand() < epsilon:
        return np.random.randint(bandit.n_arms)
    else:
        return np.argmax(bandit.sample_means)


def ucb(
    bandit: Bandit,
    c: int = 1,
) -> int:
    """
    Args:
        history (list): list of rewards and actions for past rounds
        c (float): exploration parameter
    """
    if bandit.n_history < bandit.n_arms:
        return bandit.n_history
    else:
        ucbs = bandit.sample_means + c * np.sqrt(
            2 * np.log(len(bandit.history) + 1) / bandit.counts
        )
        return np.argmax(ucbs)


def thompson_sampling(
    bandit: Bandit,
    num_samples: int = 1,
) -> int:
    """
    Args:
        history (list): list of rewards and actions for past rounds
    """
    if bandit.n_history < bandit.n_arms:
        return bandit.n_history
    else:
        # we take samples of beta(x,y) where x = 1 + total rewards and y = 1 + total failures
        samples = [
            sum(
                beta.rvs(
                    1 + bandit.sample_means[i] * bandit.counts[i],
                    1 + (1 - bandit.sample_means[i]) * bandit.counts[i],
                )
                for _ in range(num_samples)
            )
            for i in range(bandit.n_arms)
        ]
        return np.argmax(samples)
