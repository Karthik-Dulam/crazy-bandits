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
from numba.experimental import jitclass
from numba import int32, float64, from_dtype

history = np.dtype([('x', np.int32),('y', np.float64)])                  
spec = [
    ("arms", float64[:]),
    ("best_arm", int32),
    ("history", from_dtype(history)[:]),
    ("sample_means", float64[:]),
    ("counts", float64[:]),
    ("regret", float64),
]

@jitclass
class Bandit:

    def __init__(self, arms: list[rv_discrete | rv_continuous]):
        """
        Args:
            arms (list): list of distributions for each arm
        """
        self.arms = arms
        self.best_arm: int = np.argmax([arm.mean() for arm in arms])
        # history of actions and rewards for each round
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
            arm = self.bandit_algo(bandit, *args)
            bandit.pull(arm)


def run_bandit_algorithm(
    name: str,
    bandit_algo_fn: Callable[[Bandit], int],
    bernoullis: list[rv_discrete | rv_continuous],
    n_rounds: int,
    *args,
    **kwargs,
    # plot: bool = False,
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

@jit
def explore_then_commit(bandit: Bandit, explore_rounds: int) -> int:
    """
    Args:
        history (list): list of rewards and actions for past rounds
        explore_rounds (int): number of rounds to explore
    """
    history = bandit.history
    n = len(bandit.arms)

    if len(history) < explore_rounds * n:
        return len(history) % n
    else:
        return np.argmax(bandit.sample_means)

@jit
def epsilon_greedy(bandit: Bandit, epsilon: float) -> int:
    """
    Args:
        history (list): list of rewards and actions for past rounds
        epsilon (float): probability of exploration
    """
    history = bandit.history
    n = len(bandit.arms)
    if np.random.rand() < epsilon:
        return np.random.randint(n)
    else:
        return np.argmax(bandit.sample_means)

@jit
def ucb(bandit: Bandit, c: float = 1) -> int:
    """
    Args:
        history (list): list of rewards and actions for past rounds
        c (float): exploration parameter
    """
    history = bandit.history
    n = len(bandit.arms)
    if len(history) < n:
        return len(history)
    else:
        ucbs = bandit.sample_means + c * np.sqrt(
            2 * np.log(len(history) + 1) / bandit.counts
        )
        return np.argmax(ucbs)

@jit
def thompson_sampling(bandit: Bandit, num_samples: int = 1) -> int:
    """
    Args:
        history (list): list of rewards and actions for past rounds
    """
    history = bandit.history
    n = len(bandit.arms)
    if len(history) < n:
        return len(history)
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
            for i in range(n)
        ]
        return np.argmax(samples)
