import scipy.stats
import numpy as np
from dataclasses import dataclass
from typing import Union, Callable, Optional

@dataclass
class HestonParameters:
    kappa:  Union[float, np.ndarray]
    gamma:  Union[float, np.ndarray]
    rho:    Union[float, np.ndarray]
    vbar:   Union[float, np.ndarray]
    v0:     Union[float, np.ndarray]

@dataclass
class MarketState:
    stock_price:   Union[float, np.ndarray]
    interest_rate: Union[float, np.ndarray]


def get_len_conf_interval(data: np.ndarray, 
                          confidence_level: float = 0.05):
    """Get the confidence interval for a given confidence level.
    Args:
        data:             The data to compute the confidence interval for.
        confidence_level: The confidence level to use.
    
    Returns:
        The confidence interval.
    """

    mean = np.mean(data)
    l = scipy.stats.norm.ppf(confidence_level / 2) * np.sqrt(np.var(data) / len(data))
    return [mean - l, mean + l]

def get_number_of_simulations(data: np.ndarray, 
                              confidence_level: float = 0.05, 
                              absol_error: float = 1e-2):
    """Get the length of the confidence interval for a given confidence level.
    Args:
        data:             The undersampled data to compute the sample variance.
        confidence_level: The confidence level to use.
    
    Returns:
        The number of simulations.
    """
    return int((2*scipy.stats.norm.ppf(confidence_level / 2))**2 * np.var(data) / absol_error)

def get_number_of_batches(data: np.ndarray, 
                          confidence_level: float = 0.05, 
                          absol_error: float = 5e-3, 
                          batch_size: int = 10000):
    """Get the number of batches needed to achieve a given absolute error using the MC simulations.
    Args:
        absol_error: The absolute error to achieve. Defaults to 0.005 (corresponds to 0.5 cents).
        batch_size:  The batch size to use. Defaults to 10000.
    
    Returns:
        The number of batches needed to achieve the given absolute error.
    """
    return int(get_number_of_simulations(data, confidence_level, absol_error) / batch_size)+1

def controlled_simulation(market_state: MarketState,
                          params: HestonParameters,
                          T: float,
                          dt: float,
                          simulate: Callable[[MarketState, HestonParameters, float, float, int], np.ndarray],
                          absolute_error: float = 0.01,
                          confidence_level: float = 0.05,
                          batch_size: int = 10000,
                          undersample_size: int = 1000):
    """Simulate the Heston model using a controlled simulation.
    Args:
        market_state: The market state.
        params:           The Heston parameters.
        T:                The time horizon length, measured in years.
        dt:               The time step, measured in years.
        simulate:         The function to use to simulate the Heston model.
        absolute_error:   The absolute error to achieve. Defaults to 0.01.
        confidence_level: The confidence level to use. Defaults to 0.05.
        batch_size:       The batch size to use. Defaults to 10000.
        undersample_size: The size of the undersampled data for sample variance estimation. Defaults to 1000.

    Returns:
        The simulated data.
    """

    undersample = simulate(market_state, params, dt, T, undersample_size)[:, -1]
    batches = get_number_of_batches(undersample, confidence_level, absolute_error, batch_size)
    data = np.ndarray([[]])
    for i in range(batches):
        data = np.concatenate((data, simulate(params, batch_size)), axis=0)

    return data