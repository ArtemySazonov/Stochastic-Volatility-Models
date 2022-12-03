import scipy.stats
import numpy as np

def get_len_conf_interval(data: np.ndarray, confidence_level: float = 0.05):
    """Get the confidence interval for a given confidence level.
    Args:
        data: The data to compute the confidence interval for.
        confidence_level: The confidence level to use.
    
    Returns:
        The confidence interval.
    """

    variance = np.var(data)
    mean = np.mean(data)
    Z = scipy.stats.norm.ppf(confidence_level / 2)
    l = Z * np.sqrt(variance / len(data))
    return [mean - l, mean + l]

def get_number_of_simulations(data: np.ndarray, confidence_level: float = 0.05, absol_error: float = 1e-2):
    """Get the length of the confidence interval for a given confidence level.
    Args:
        data: The undersampled data to compute the sample variance.
        confidence_level: The confidence level to use.
    
    Returns:
        The number of simulations.
    """
    return int((2*scipy.stats.norm.ppf(confidence_level / 2))**2 * np.var(data) / absol_error)

def get_number_of_batches(data: np.ndarray, confidence_level: float = 0.05, absol_error: float = 1e-2, batch_size: int = 10000):
    """Get the number of batches needed to achieve a given relative error using the MC simulations.
    Args:
        absol_error: The relative error to achieve. Defaults to 0.01 (corresponds to 1 cent).
        batch_size: The batch size to use. Defaults to 10000.
    
    Returns:
        The number of batches needed to achieve the given relative error.
    """
    return int(get_number_of_simulations(data, confidence_level, absol_error) / batch_size)+1