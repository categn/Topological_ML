import pandas as pd
from scipy.integrate import simpson
import numpy as np

def build_point_clouds(
        df: pd.DataFrame, 
        window_size: int
    ) -> list:

    """
    Constructs point clouds from a multivariate time series using a sliding window approach.

    Parameters
    ----------
    df : A DataFrame containing the log-returns of multiple indices.
    window_size : The number of consecutive time steps to include in each window.

    Returns
    -------
    list of ndarray
        A list of NumPy arrays, each representing a point cloud.
        Each point cloud has shape (window_size, n), where n is the number of indices.
    """

    d = len(df)
    point_clouds = []
    for k in range(d - window_size + 1):       # Slide the window accross the time series
        cloud = df.iloc[k:k+window_size].values   # Take the values corrsponding to that window
        point_clouds.append(cloud)

    return point_clouds

def compute_lp_norms(
        landscapes, 
        p: int = 1, 
        dx: float = None
    ) -> np.ndarray:

    """
    Compute the L^p norm of a collection of persistence landscapes.

    Each landscape consists of multiple layers (1D NumPy arrays). 
    The L^p norm is computed by integrating |layer|^p over the domain using Simpson's rule,
    summing over all layers, and then taking the p-th root.

    Parameters:
    ----------
    landscapes : A list of landscapes, where each landscape is a list of 1D NumPy arrays 
        representing individual layers sampled on the same grid.
    p : The order of the L^p norm (p >= 1).
    dx : The spacing between consecutive points in the domain over which each landscape layer is defined.

    Returns:
    -------
    np.ndarray
        A 1D array containing the L^p norm of each landscape.
    """

    norms = []
    for landscape in landscapes:    # Loop through each landscape
        norm = 0
        for layer in landscape:   # Loop through each layer (1D array) of the landscape
            norm += simpson(np.abs(layer)**p, dx=dx)    # Compute the integral of |layer|^p using Simpson's rule
        norm = norm**(1/p)     # Take the p-th root of the total integral to get the L^p norm
        norms.append(norm)
    return np.array(norms)

def detect_signals(
        norms, 
        alpha: float, 
        beta: int, 
        s: int, 
        t: int
    ) -> set:

    """
    Detect signal windows based on L^p norm anomalies using three conditions.

    For each index j, the function checks:
      - (i) Nj / Nj-1 >= alpha
      - (ii) Nj / Qj >= beta, where Qj is the average of the t previous norms
      - (iii) Cj / Qj >= beta, where Cj is the average of the s future norms

    If all three conditions hold, the time window [j, j+s) is marked as containing a warning signal.

    Parameters:
    ----------
    norms : Time series of L^p norms.
    alpha : Threshold for the jump ratio condition.
    beta : Threshold for the relative increase conditions.
    s : Length of the future window (Cj).
    t : Length of the past window (Qj).

    Returns:
    -------
    set of int
        Sorted set of time indices flagged as warning signals.
    """

    warnings = set()
    for j in range(t, len(norms) - s):
        Nj = norms[j]        # Current norm
        Nj_1 = norms[j - 1]      # Previous norm
        Qj = np.mean(norms[j - t:j])      # Mean of the t previous norms
        Cj = np.mean(norms[j:j + s])      # Mean of the s following norms

        condition_i = Nj / Nj_1 >= alpha      # Sudden increase in current norm vs previous one
        condition_ii = Nj / Qj >= beta        # Current norm much larger than historical average
        condition_iii = Cj / Qj >= beta       # Future average high compared to historical average

        if condition_i and condition_ii and condition_iii:
            warnings.update(range(j, j + s))        # Add all indices in the signal window
    return sorted(warnings)

def get_signal_date(
        signal_days: set, 
        dates
    ):

    """
    Convert signal indices into calendar dates, returning only the first day of each signal burst.
    Consecutive days with signals are grouped, and only the first index of each group is mapped to a date.

    Parameters:
    ----------
    signal_days : Set of time indices flagged as signal days.
    dates : Full list of dates corresponding to the indices in the original time series.

    Returns:
    -------
    list of datetime-like
        List of dates corresponding to the start of each group of consecutive signal days.
    """

    start_days = []

    for i in range(len(signal_days)):
        if i == 0 or signal_days[i] != signal_days[i - 1] + 1:    # If it's the first day, or not consecutive with the previous day, mark as a signal day
            start_days.append(signal_days[i])

    return [dates[i] for i in start_days]    # Get the date corresponding to the index 
