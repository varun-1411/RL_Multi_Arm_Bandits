"""
Modified Gradient Bandit with Adaptive Baseline
================================================

Components:
-----------
1. NonStationaryBandit: Environment with changing reward distributions
2. AdaptiveGradientBandit: Agent with adaptive baseline
3. Experiment runner and visualization functions
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List

# ======================================================
# Global seed for reproducibility
# ======================================================
GLOBAL_SEED = 42
np.random.seed(GLOBAL_SEED)


class NonStationaryBandit:
    """
    Non-stationary 10-armed bandit with random walk.
    
    Attributes:
        n_arms: Number of arms (actions) available
        true_means: Current true mean reward for each arm
        step_count: Number of steps taken
        walk_interval: Steps between random walk updates
    """
    
    def __init__(self, n_arms: int = 10, walk_interval: int = 500):
        """
        Initialize the non-stationary bandit.
        
        Args:
            n_arms: Number of arms
            walk_interval: How often to update true means (random walk)
        """
        self.n_arms = n_arms
        self.walk_interval = walk_interval
        self.step_count = 0
        
        # Initialize true means from N(0, 1)
        self.true_means = np.random.randn(n_arms)
    
    def get_reward(self, action: int) -> float:
        """
        Get reward for pulling an arm.
        
        Args:
            action: Which arm to pull (0 to n_arms-1)
            
        Returns:
            Sampled reward value from N(true_mean[action], 1)
        """
        self.step_count += 1
        
        # Every walk_interval steps, perform random walk on all arm means
        if self.step_count % self.walk_interval == 0:
            self.true_means += np.random.randn(self.n_arms) * 0.1
        
        # Return noisy reward: true_mean + Gaussian noise
        return np.random.randn() + self.true_means[action]
    
    def get_optimal_action(self) -> int:
        """Return the current optimal action (arm with highest true mean)."""
        return np.argmax(self.true_means)
    
    def get_optimal_value(self) -> float:
        """Return the value of the optimal action."""
        return np.max(self.true_means)
    
    def reset(self):
        """Reset the bandit to initial state with new random means."""
        self.step_count = 0
        self.true_means = np.random.randn(self.n_arms)


class AdaptiveGradientBandit:
    """
    Gradient Bandit Agent with Adaptive Baseline.
    
    Attributes:
        n_arms: Number of arms
        alpha: Learning rate for preference updates
        beta: Weight for variance adaptation (0 = standard baseline)
        window_size: Size of sliding window for recent rewards
        H: Preference values for each arm
        reward_history: All rewards received
        recent_rewards: Sliding window of recent rewards
    """
    
    def __init__(self, n_arms: int, alpha: float, beta: float, window_size: int):
        """
        Initialize the adaptive gradient bandit agent.
        
        Args:
            n_arms: Number of arms
            alpha: Step size for preference updates (learning rate)
            beta: Weight for variance adaptation (0 = standard baseline)
            window_size: Number of recent rewards to consider for variance
        """
        self.n_arms = n_arms
        self.alpha = alpha
        self.beta = beta
        self.window_size = window_size
        
        # Initialize preferences to 0 (equal probability for all arms initially)
        self.H = np.zeros(n_arms)
        
        # Track all rewards for computing running average
        self.reward_history: List[float] = []
        
        # Track recent rewards in a sliding window for variance computation
        self.recent_rewards: List[float] = []
        
        # Cache the current baseline for reporting
        self._current_baseline = 0.0
    
    def _softmax(self) -> np.ndarray:
        """
        Compute softmax probabilities from preferences.
        
        Returns:
            Array of action probabilities
        """
        # Subtract max for numerical stability (prevents overflow)
        exp_H = np.exp(self.H - np.max(self.H))
        return exp_H / np.sum(exp_H)
    
    def select_action(self) -> int:
        """
        Select action based on current softmax policy.
        
        Returns:
            Selected action index
        """
        probabilities = self._softmax()
        return np.random.choice(self.n_arms, p=probabilities)
    
    def _compute_variance_adjusted_mean(self) -> float:
        """
        Compute variance-adjusted mean from recent rewards.
        
        Returns:
            Variance-adjusted mean of recent rewards
        """
        if len(self.recent_rewards) < 2:
            return np.mean(self.recent_rewards) if self.recent_rewards else 0.0
        
        rewards = np.array(self.recent_rewards)
        mean = np.mean(rewards)
        
        # Compute squared deviations from mean
        squared_deviations = (rewards - mean) ** 2
        
        # Weights are inversely proportional to squared deviations
        epsilon = 1e-8
        weights = 1.0 / (squared_deviations + epsilon)
        
        # Normalize weights to sum to 1
        weights = weights / np.sum(weights)
        
        # Weighted mean
        variance_adjusted_mean = np.sum(weights * rewards)
        
        return variance_adjusted_mean
    
    def get_baseline(self) -> float:
        """
        Compute and return current adaptive baseline value.
        
        Formula: baseline = (1 - beta) * avg_reward + beta * variance_adjusted_mean
        
        Returns:
            Current baseline value
        """
        if not self.reward_history:
            return 0.0
        
        # Component 1: Running average of all rewards
        avg_reward = np.mean(self.reward_history)
        
        if self.beta == 0:
            # Standard gradient bandit: just use average
            self._current_baseline = avg_reward
            return avg_reward
        
        # Component 2: Variance-adjusted mean of recent rewards
        variance_adjusted_mean = self._compute_variance_adjusted_mean()
        
        # Combine using beta weight
        baseline = (1 - self.beta) * avg_reward + self.beta * variance_adjusted_mean
        self._current_baseline = baseline
        
        return baseline
    
    def update(self, action: int, reward: float) -> None:
        """
        Update preferences using the gradient bandit update rule.
        
        Update rules:
            For selected action a:    H[a] = H[a] + alpha * (R - baseline) * (1 - pi[a])
            For other actions:        H[a] = H[a] - alpha * (R - baseline) * pi[a]
        
        Args:
            action: The action that was taken
            reward: The reward received
        """
        # Store reward in history
        self.reward_history.append(reward)
        
        # Update recent rewards window (sliding window)
        self.recent_rewards.append(reward)
        if len(self.recent_rewards) > self.window_size:
            self.recent_rewards.pop(0)
        
        # Get current baseline and policy
        baseline = self.get_baseline()
        pi = self._softmax()
        
        # Compute the reward advantage
        advantage = reward - baseline
        
        # Update all preferences
        for a in range(self.n_arms):
            if a == action:
                self.H[a] += self.alpha * advantage * (1 - pi[a])
            else:
                self.H[a] -= self.alpha * advantage * pi[a]
    
    def reset(self):
        """Reset the agent to initial state."""
        self.H = np.zeros(self.n_arms)
        self.reward_history = []
        self.recent_rewards = []
        self._current_baseline = 0.0






def run_experiment(n_runs: int, n_steps: int, n_arms: int, alpha: float,
                   beta: float, window_size: int) -> Dict[str, np.ndarray]:
    """
    Run multiple experiments and return averaged results.
    
    Args:
        n_runs: Number of independent runs to average over
        n_steps: Number of steps per run
        n_arms: Number of arms in the bandit
        alpha: Learning rate
        beta: Adaptive baseline weight
        window_size: Window size for variance computation
    
    Returns:
        Dictionary containing:
        - 'rewards': Average reward at each step
        - 'optimal_actions': Fraction of optimal actions at each step
        - 'baselines': Average baseline value at each step
    """
    all_rewards = np.zeros((n_runs, n_steps))
    all_optimal = np.zeros((n_runs, n_steps))
    all_baselines = np.zeros((n_runs, n_steps))
    
    for run in range(n_runs):
        bandit = NonStationaryBandit(n_arms=n_arms)
        agent = AdaptiveGradientBandit(n_arms, alpha, beta, window_size)
        
        for step in range(n_steps):
            optimal_action = bandit.get_optimal_action()
            action = agent.select_action()
            reward = bandit.get_reward(action)
            agent.update(action, reward)
            
            all_rewards[run, step] = reward
            all_optimal[run, step] = 1.0 if action == optimal_action else 0.0
            all_baselines[run, step] = agent._current_baseline
    
    return {
        'rewards': np.mean(all_rewards, axis=0),
        'optimal_actions': np.mean(all_optimal, axis=0),
        'baselines': np.mean(all_baselines, axis=0)
    }


def compute_running_average(data: np.ndarray, window: int = 100) -> np.ndarray:
    """
    Compute running average over a sliding window.
    
    Args:
        data: 1D array of values
        window: Window size for averaging
    
    Returns:
        Array of running averages (same length as input)
    """
    result = np.zeros_like(data)
    for i in range(len(data)):
        start = max(0, i - window + 1)
        result[i] = np.mean(data[start:i+1])
    return result


def plot_average_reward(results: Dict, beta_values: List[float], n_steps: int):
    """
    Plot average reward over time.
    
    Args:
        results: Dictionary of results for each beta value
        beta_values: List of beta values tested
        n_steps: Total number of steps
    """
    plt.figure(figsize=(10, 6))
    
    colors = ['blue', 'orange', 'green']
    
    for beta, color in zip(beta_values, colors):
        running_avg = compute_running_average(results[beta]['rewards'], window=100)
        label = r'$\beta = {}$'.format(beta) if beta > 0 else r'$\beta = 0$ (Standard)'
        plt.plot(running_avg, color=color, label=label, alpha=0.8, linewidth=1.5)
    
    # Add vertical lines for random walk
    for step in range(500, n_steps, 500):
        plt.axvline(x=step, color='red', linestyle='--', alpha=0.3)
    
    plt.xlabel('Steps', fontsize=12)
    plt.ylabel('Average Reward (100-step running average)', fontsize=12)
    plt.title('Average Reward Over Time', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Add annotation for random walk
    plt.text(520, plt.ylim()[1] * 0.05, 'Random Walk', fontsize=9, color='red', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('plot1_average_reward.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: plot1_average_reward.png")


def plot_optimal_action(results: Dict, beta_values: List[float], n_steps: int):
    """
    Plot percentage of optimal action selection.
    
    Args:
        results: Dictionary of results for each beta value
        beta_values: List of beta values tested
        n_steps: Total number of steps
    """
    plt.figure(figsize=(10, 6))
    
    colors = ['blue', 'orange', 'green']
    
    for beta, color in zip(beta_values, colors):
        running_avg = compute_running_average(results[beta]['optimal_actions'], window=100)
        label = r'$\beta = {}$'.format(beta) if beta > 0 else r'$\beta = 0$ (Standard)'
        plt.plot(running_avg * 100, color=color, label=label, alpha=0.8, linewidth=1.5)
    
    # Add vertical lines for random walk
    for step in range(500, n_steps, 500):
        plt.axvline(x=step, color='red', linestyle='--', alpha=0.3)
    
    plt.xlabel('Steps', fontsize=12)
    plt.ylabel('Optimal Action Selection (%)', fontsize=12)
    plt.title('Percentage of Optimal Action Selection Over Time', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 100)
    
    # Add annotation for random walk
    plt.text(520, plt.ylim()[1] * 0.05, 'Random Walk', fontsize=9, color='red', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('plot2_optimal_action.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: plot2_optimal_action.png")


def plot_baseline_values(results: Dict, beta_values: List[float], n_steps: int):
    """
    Plot baseline values over time.
    
    Args:
        results: Dictionary of results for each beta value
        beta_values: List of beta values tested
        n_steps: Total number of steps
    """
    plt.figure(figsize=(10, 6))
    
    colors = ['blue', 'orange', 'green']
    
    for beta, color in zip(beta_values, colors):
        smoothed_baseline = compute_running_average(results[beta]['baselines'], window=50)
        label = r'$\beta = {}$'.format(beta) if beta > 0 else r'$\beta = 0$ (Standard)'
        plt.plot(smoothed_baseline, color=color, label=label, alpha=0.8, linewidth=1.5)
    
    # Add vertical lines for random walk
    for step in range(500, n_steps, 500):
        plt.axvline(x=step, color='red', linestyle='--', alpha=0.3)
    
    plt.xlabel('Steps', fontsize=12)
    plt.ylabel('Baseline Value', fontsize=12)
    plt.title('Baseline Values Over Time', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Add annotation for random walk
    plt.text(520, plt.ylim()[1] * 0.05, 'Random Walk', fontsize=9, color='red', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('plot3_baseline_values.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: plot3_baseline_values.png")


def print_full_statistics(results: Dict, beta_values: List[float], n_steps: int):
    """
    Print comprehensive statistics for the entire experiment.
    
    Args:
        results: Dictionary of results for each beta value
        beta_values: List of beta values tested
        n_steps: Total number of steps
    """
    print("\n" + "=" * 70)
    print("COMPREHENSIVE STATISTICS")
    print("=" * 70)
    
    # Define intervals for analysis
    intervals = [
        ("Full Run (Steps 1-2000)", 0, n_steps),
        ("Early Phase (Steps 1-500)", 0, 500),
        ("Mid Phase (Steps 501-1000)", 500, 1000),
        ("Late Phase (Steps 1001-1500)", 1000, 1500),
        ("Final Phase (Steps 1501-2000)", 1500, 2000),
    ]
    
    for interval_name, start, end in intervals:
        print(f"\n{'-' * 70}")
        print(f"{interval_name}")
        print(f"{'-' * 70}")
        print(f"{'Beta':<20} {'Avg Reward':<20} {'Optimal Action %':<20}")
        print(f"{'-' * 60}")
        
        for beta in beta_values:
            avg_reward = np.mean(results[beta]['rewards'][start:end])
            std_reward = np.std(results[beta]['rewards'][start:end])
            optimal_pct = np.mean(results[beta]['optimal_actions'][start:end]) * 100
            
            beta_str = f"beta = {beta}" if beta > 0 else "beta = 0 (Standard)"
            print(f"{beta_str:<20} {avg_reward:.3f} ± {std_reward:.3f}      {optimal_pct:.1f}%")
    
    # Performance comparison
    print("\n" + "=" * 70)
    print("PERFORMANCE COMPARISON (Relative to Standard beta = 0)")
    print("=" * 70)
    
    baseline_optimal = np.mean(results[0.0]['optimal_actions']) * 100
    baseline_reward = np.mean(results[0.0]['rewards'])
    
    print(f"\n{'Beta':<20} {'Optimal Action change':<25} {'Avg Reward change':<20}")
    print(f"{'-' * 65}")
    
    for beta in beta_values:
        optimal_pct = np.mean(results[beta]['optimal_actions']) * 100
        avg_reward = np.mean(results[beta]['rewards'])
        
        optimal_diff = optimal_pct - baseline_optimal
        reward_diff = avg_reward - baseline_reward
        
        beta_str = f"beta = {beta}" if beta > 0 else "beta = 0 (Standard)"
        optimal_sign = "+" if optimal_diff >= 0 else ""
        reward_sign = "+" if reward_diff >= 0 else ""
        
        print(f"{beta_str:<20} {optimal_sign}{optimal_diff:.2f}%                  {reward_sign}{reward_diff:.4f}")


def main():
    """
    Run experiments and create separate plots for each metric.
    
    This function:
    1. Runs experiments for beta = 0 (standard), beta = 0.3, and beta = 0.6
    2. Creates three separate plots:
       - Plot 1: Average reward over time
       - Plot 2: Percentage of optimal action selection
       - Plot 3: Baseline values over time
    3. Prints comprehensive statistics for all phases
    """
    # Experiment parameters
    n_runs = 200      # Number of independent runs to average
    n_steps = 2000    # Steps per run
    n_arms = 10       # Number of arms
    alpha = 0.1       # Learning rate
    window_size = 50  # Window for variance computation
    
    # Beta values to compare
    beta_values = [0.0, 0.3, 0.6]
    
    print("=" * 70)
    print("ADAPTIVE GRADIENT BANDIT EXPERIMENT")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  - Number of runs: {n_runs}")
    print(f"  - Steps per run: {n_steps}")
    print(f"  - Number of arms: {n_arms}")
    print(f"  - Learning rate (alpha): {alpha}")
    print(f"  - Window size: {window_size}")
    print(f"  - Beta values: {beta_values}")
    print(f"  - Random walk interval: 500 steps")
    print("-" * 70)
    
    # Run experiments
    results = {}
    for beta in beta_values:
        print(f"Running experiments for beta = {beta}...")
        results[beta] = run_experiment(n_runs, n_steps, n_arms, alpha, beta, window_size)
    
    print("\nGenerating separate plots...")
    
    # Create separate plots
    plot_average_reward(results, beta_values, n_steps)
    plot_optimal_action(results, beta_values, n_steps)
    plot_baseline_values(results, beta_values, n_steps)
    
    # Print comprehensive statistics
    # print_full_statistics(results, beta_values, n_steps)

if __name__ == "__main__":
    main()