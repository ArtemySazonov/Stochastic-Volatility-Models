import numpy as np

def european_call(paths: np.ndarray, 
                  strike: float):
    """Compute the payoff of a European call option.
    Args:
        paths:  The simulated paths.
        strike: The strike price.
    
    Returns:
        The payoff of the European call option.
    """
    return np.maximum(paths[:, -1] - strike, 0)

def european_put(paths: np.ndarray,
                 strike: float):
        """Compute the payoff of a European put option.
        Args:
            paths:  The simulated paths.
            strike: The strike price.
        
        Returns:
            The payoff of the European put option.
        """
        return np.maximum(strike - paths[:, -1], 0)

def asian_call(paths: np.ndarray, 
               strike: float):
    """Compute the payoff of an Asian call option.
    Args:
        paths:  The simulated paths.
        strike: The strike price.
    
    Returns:
        The payoff of the Asian call option.
    """
    return np.maximum(np.mean(paths, axis=1) - strike, 0)

def asian_put(paths: np.ndarray,
              strike: float):
    """Compute the payoff of an Asian put option.
    Args:
        paths:  The simulated paths.
        strike: The strike price.
    
    Returns:
        The payoff of the Asian put option.
    """
    return np.maximum(strike - np.mean(paths, axis=1), 0)