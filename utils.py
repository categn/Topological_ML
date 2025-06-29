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
    ) -> list:

    """
    Detect signal windows based on L^p norm anomalies using three conditions.
    
    For each index j, the function checks:
      - (i) Nj / Nj-1 >= alpha
      - (ii) Nj / Qj >= beta, where Qj is the average of the t previous norms
      - (iii) Cj / Qj >= beta, where Cj is the average of the s future norms

    If all three conditions hold, the time window [j, j+s) is marked as containing a warning signal.

    Parameters
    ----------
    norms : Time series of L^p norms.
    alpha : Threshold for the jump ratio condition.
    beta : Threshold for the relative increase conditions.
    s : Length of the future window (Cj).
    t : Length of the past window (Qj).

    Returns
    -------
    list of int
        Sorted list of time indices flagged as warning signals.
    """
  
    # Calculate once the mean of the previous t norms
    kernel_t = np.ones(t) / t
    Qj_all = np.convolve(norms, kernel_t, mode='valid')

    # Calculate once the mean of the following s norms
    kernel_s = np.ones(s) / s
    Cj_all = np.convolve(norms, kernel_s, mode='valid')

    warnings = set()
    for j in range(t, len(norms) - s):
        Nj = norms[j]               # Current norm
        Nj_1 = norms[j - 1]         # Previous norm
        Qj = Qj_all[j - t]          # Mean of the previous t norms
        Cj = Cj_all[j]              # Mean of the following s norms

        condition_i = Nj / Nj_1 >= alpha        # Sudden increase in current norm vs previous one
        condition_ii = Nj / Qj >= beta          # Current norm much larger than historical average
        condition_iii = Cj / Qj >= beta         # Future average high compared to historical average

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
    signal_days = sorted(signal_days)    # Sort the signal days to ensure they are in order
    start_days = []

    for i in range(len(signal_days)):
        if i == 0 or signal_days[i] != signal_days[i - 1] + 1:    # If it's the first day, or not consecutive with the previous day, mark as a signal day
            start_days.append(signal_days[i])

    return [dates[i] for i in start_days]    # Get the date corresponding to the index 


def evaluate_ews(
        crisis_dates,
        ews_dates,
        lead_window = pd.Timedelta(days=600),
        lag_window = pd.Timedelta(days=100)
    ) -> dict:
    
    """
    Evaluate the effectiveness of EWS in matching real crisis events.

    An EWS is considered a:
    - True Positive (TP) if it occurs within the lead_window before a crisis.
    - Late EWS if it occurs within the lag_window after a crisis.
    - False Positive (FP) if it does not correspond to any crisis within either window.
    - False Negative (FN) if no EWS occurs within either window around a crisis.

    Metrics:
    - Precision: TP / (TP + Late EWS + FP)
    - Recall: TP / (TP + Late EWS + FN)
    - F1 score: Harmonic mean of precision and recall
    - False Alarm Rate (FAR): FP / (TP + Late EWS + FP)
    - Mean lead time: Average number of days between EWS and crisis
    - Mean delay time: Average number of days between crisis and late EWS

    Parameters:
    ----------
    crisis_dates : List of known crisis dates.
    ews_dates : List of early warning signal dates.
    lead_window : Time period before each crisis in which an EWS is considered valid.
    lag_window : Time period after each crisis in which an EWS is considered late.

    Returns:
    -------
    dict
        Dictionary containing evaluation metrics and statistics.
    """

    matched_early = {}
    matched_late = {}
    used_ews = set()

    for c in crisis_dates:
        early_win = [e for e in ews_dates if (c - lead_window) <= e <= c]    # Find EWS in the lead window 
        late_win  = [e for e in ews_dates if c < e <= (c + lag_window)]      # Find EWS in the lag window 

        if early_win:
            first = min(early_win)     # Take the earliest EWS in the lead window
            matched_early[c] = first
            used_ews.add(first)        # Avoid using this EWS again
        elif late_win:
            first = min(late_win)      # Take the earliest EWS in the lag window
            matched_late[c] = first
            used_ews.add(first)        # Avoid using this EWS again

    tp = len(matched_early)      # True Positives
    late_ews  = len(matched_late)      # Late EWS
    fn = len([c for c in crisis_dates if c not in matched_early and c not in matched_late])    # False Negatives
    fp = len([e for e in ews_dates if e not in used_ews])      # False Positives

    # Evaluation metrics
    precision = (tp) / (tp + late_ews + fp) if (tp + late_ews + fp) else np.nan
    recall = (tp) / (tp + late_ews + fn) if (tp + late_ews + fn) else np.nan
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else np.nan
    far = fp / (tp + late_ews + fp) if (tp + late_ews + fp) else np.nan
    lead_times = [(c - matched_early[c]).days for c in matched_early] 
    delay_times = [(matched_late[c] - c).days for c in matched_late]

    return {
        "True Positives": tp,
        "Late EWS": late_ews,
        "False Negatives": fn,
        "False Positives": fp,
        "Precision": round(precision, 3),
        "Recall": round(recall, 3),
        "F1 score": round(f1, 3),
        "False alarm rate": round(far, 3),
        "Mean lead time": round(np.mean(lead_times), 3) if lead_times else np.nan,
        "Mean delay time": round(np.mean(delay_times), 3) if delay_times else np.nan
    }
