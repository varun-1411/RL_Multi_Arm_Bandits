class AdaptiveGradientBandit:
    def __init__(self, n_arms: int, alpha: float, beta: float, window_size: int):
        """
        Initialize the adaptive gradient bandit agent.
        
        Args:
            n_arms: Number of arms
            alpha: Step size for preference updates
            beta: Weight for variance adaptation (0 = standard baseline)
            window_size: Number of recent rewards to consider for variance
        """
        pass
    
    def select_action(self) -> int:
        """Select action based on current softmax policy."""
        pass
    
    def update(self, action: int, reward: float) -> None:
        """Update preferences using adaptive baseline."""
        pass
    
    def get_baseline(self) -> float:
        """Return current baseline value."""
        pass

def run_experiment(n_runs: int, n_steps: int, n_arms: int, alpha: float, 
                   beta: float, window_size: int) -> dict:
    """
    Run multiple experiments and return averaged results.
    
    Returns:
        dict with keys: 'rewards', 'optimal_actions', 'baselines'
    """
    pass

class NonStationaryBandit:
    """Non-stationary 10-armed bandit with random walk."""
    pass

def main():
    """Run experiments and plot results."""


if __name__ == "__main__":
    main()

