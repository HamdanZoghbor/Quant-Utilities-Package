import numpy as np

def arithmetic_mean(data):
    """
    Calculate the arithmetic mean of a list or array of numbers.

    Parameters:
    data (list or np.ndarray): A list or array of numerical values.

    Returns:
    float: The arithmetic mean of the input data.
    """
    if len(data) == 0:
        raise ValueError("Data list cannot be empty.")
    
    data_array = np.array(data)
    mean_value = np.mean(data_array)
    
    return mean_value

def weighted_mean(data, weights):
    """
    Calculate the weighted mean of a list or array of numbers.

    Parameters:
    data (list or np.ndarray): A list or array of numerical values.
    weights (list or np.ndarray): A list or array of weights corresponding to the data.

    Returns:
    float: The weighted mean of the input data.
    """
    if len(data) == 0:
        raise ValueError("Data list cannot be empty.")
    if len(data) != len(weights):
        raise ValueError("Data and weights must be of the same length.")
    
    data_array = np.array(data)
    weights_array = np.array(weights)
    
    if np.sum(weights_array) == 0:
        raise ValueError("Sum of weights cannot be zero.")
    
    weighted_mean_value = np.sum(data_array * weights_array) / np.sum(weights_array)
    
    return weighted_mean_value

def geometric_mean(data):
    """
    Calculate the geometric mean of a list or array of numbers.

    Parameters:
    data (list or np.ndarray): A list or array of numerical values.

    Returns:
    float: The geometric mean of the input data.
    """
    if len(data) == 0:
        raise ValueError("Data list cannot be empty.")
    
    data_array = np.array(data)
    
    if np.any(data_array <= 0):
        raise ValueError("All elements must be positive to compute geometric mean.")
    
    product = np.prod(data_array)
    n = len(data_array)
    geo_mean_value = product ** (1/n)
    
    return geo_mean_value

def population_variance(data):
    """
    Calculate the population variance of a list or array of numbers.

    Parameters:
    data (list or np.ndarray): A list or array of numerical values.

    Returns:
    float: The population variance of the input data.
    """
    if len(data) == 0:
        raise ValueError("Data list cannot be empty.")
    
    data_array = np.array(data)
    mean = np.mean(data_array)
    variance = np.mean((data_array - mean) ** 2)
    
    return variance

def sharpe_ratio(returns, risk_free_rate=0.0, use_population=False):
    """
    Calculate the Sharpe Ratio of a set of returns.

    Parameters:
    returns (list or np.ndarray): A list or array of returns.
    risk_free_rate (float): The risk-free rate to subtract from returns.
    use_population (bool): Whether to use population standard deviation.

    Returns:
    float: The Sharpe Ratio.
    """
    excess_returns = np.array(returns) - risk_free_rate
    mean_excess_return = np.mean(excess_returns)
    if use_population:
        std_dev = np.std(excess_returns, ddof=0)
    else:
        std_dev = np.std(excess_returns, ddof=1)
    
    if std_dev == 0:
        return 0.0
    
    sharpe_ratio_value = mean_excess_return / std_dev
    return sharpe_ratio_value