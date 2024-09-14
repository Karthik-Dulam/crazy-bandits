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
from numba import njit


class Bandit:

    def __init__(self, arms: list[rv_discrete | rv_continuous]):
        """
        Args:
            arms (list): list of distributions for each arm
        """
        self.arms = arms
        self.best_arm: int = np.argmax([arm.mean() for arm in arms])
        # history of rewards and actions for each round
        self.history: list[(int, float)] = []
        self.sample_means, self.counts = np.zeros(len(arms)), np.zeros(len(arms))
        self.regret = 0

    def pull(self, arm) -> float:
        reward = self.arms[arm].rvs()
        self.history.append((arm, reward))
        self.__update_sample_means(arm, reward)
        self.__update_regret()
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

    def __update_regret(self) -> None:
        self.regret += self.arms[self.best_arm].mean() - self.history[-1][1]

    def plot_history(self, algo_name: str) -> None:
        # plot cumulative rewards and actions
        cum_rewards = np.cumsum([reward for _, reward in self.history])
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
        ax[1].plot(actions, "b.")
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
            arm = self.bandit_algo(
                bandit.history,
                len(bandit.arms),
                bandit.sample_means,
                bandit.counts,
                *args,
            )
            bandit.pull(arm)


def run_bandit_algorithm(
    name: str,
    bandit_algo_fn: Callable[[Bandit], int],
    bernoullis: list[rv_discrete | rv_continuous],
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
    berBandit = Bandit(bernoullis)
    bandit_algo = BanditAlgorithm(name, bandit_algo_fn)
    bandit_algo.run(berBandit, n_rounds, *args)
    if plot:
        berBandit.plot_history(name)
    return berBandit.regret


@njit
def explore_then_commit(
    history: list[tuple[int, float]],
    n_arms: int,
    sample_means: np.ndarray,
    counts: np.ndarray,
    explore_rounds: int,
) -> int:
    """
    Args:
        history (list): list of rewards and actions for past rounds
        explore_rounds (int): number of rounds to explore
    """

    if len(history) < explore_rounds * n_arms:
        return len(history) % n_arms
    else:
        return np.argmax(sample_means)


@njit
def epsilon_greedy(
    history: list[tuple[int, float]],
    n_arms: int,
    sample_means: np.ndarray,
    counts: np.ndarray,
    epsilon: float,
) -> int:
    """
    Args:
        history (list): list of rewards and actions for past rounds
        epsilon (float): probability of exploration
    """
    if np.random.rand() < epsilon:
        return np.random.randint(n_arms)
    else:
        return np.argmax(sample_means)


@njit
def ucb(
    history: list[tuple[int, float]],
    n_arms: int,
    sample_means: np.ndarray,
    counts: np.ndarray,
    c: float,
) -> int:
    """
    Args:
        history (list): list of rewards and actions for past rounds
        c (float): exploration parameter
    """
    if len(history) < n_arms:
        return len(history)
    else:
        ucbs = sample_means + c * np.sqrt(2 * np.log(len(history) + 1) / counts)
        return np.argmax(ucbs)


@njit
def thompson_sampling(
    history: list[tuple[int, float]],
    n_arms: int,
    sample_means: np.ndarray,
    counts: np.ndarray,
    num_samples: int = 1,
) -> int:
    """
    Args:
        history (list): list of rewards and actions for past rounds
    """
    if len(history) < n_arms:
        return len(history)
    else:
        # we take samples of beta(x,y) where x = 1 + total rewards and y = 1 + total failures
        samples = [
            sum(
                beta.rvs(
                    1 + sample_means[i] * counts[i],
                    1 + (1 - sample_means[i]) * counts[i],
                )
                for _ in range(num_samples)
            )
            for i in range(n_arms)
        ]
        return np.argmax(samples)
