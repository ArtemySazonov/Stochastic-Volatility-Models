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


def get_len_conf_interval(data: np.ndarray, 
                          confidence_level: float = 0.05):
    """Get the confidence interval for a given confidence level.
    Args:
        data: The data to compute the confidence interval for.
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
        data: The undersampled data to compute the sample variance.
        confidence_level: The confidence level to use.
    
    Returns:
        The number of simulations.
    """
    return int((2*scipy.stats.norm.ppf(confidence_level / 2))**2 * np.var(data) / absol_error)

def get_number_of_batches(data: np.ndarray, 
                          confidence_level: float = 0.05, 
                          absol_error: float = 5e-3, 
                          batch_size: int = 10000):
    """Get the number of batches needed to achieve a given relative error using the MC simulations.
    Args:
        absol_error: The relative error to achieve. Defaults to 0.005 (corresponds to 0.5 cents).
        batch_size: The batch size to use. Defaults to 10000.
    
    Returns:
        The number of batches needed to achieve the given relative error.
    """
    return int(get_number_of_simulations(data, confidence_level, absol_error) / batch_size)+1

def controlled_simulation(params: HestonParameters,
                          simulate: Callable[[HestonParameters, int], np.ndarray],
                          absolute_error: float = 0.01,
                          batch_size: int = 10000):
    """Controlled-precision simulation using the batch method.
    Args:
        params: The Heston parameters to use.
        simulate: The simulation function to use.
        absolute_error: The absolute error to achieve. Defaults to 0.01.

    Returns:
        The paths.
    """

    data = simulate(params, 100)[:, -1]
    batches = get_number_of_batches(data, 0.05, absolute_error)
    for i in range(batches):
        data = np.concatenate((data, simulate(params, batch_size)[:, -1]), axis=0)

    return data