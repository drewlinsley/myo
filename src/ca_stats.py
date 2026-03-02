import numpy as np
from src import utils
from scipy import signal
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import warnings


warnings.filterwarnings("error")


def calc_diff_trace(trace):
    """
    Calculates the difference between consecutive values in the trace.

    Args:
        trace (numpy.ndarray): The input fluorescence trace.

    Returns:
        numpy.ndarray: The difference trace.
    """
    diff_trace = np.diff(trace)
    return diff_trace


def find_transient_onsets(diff_trace, prominence=0.5):
    """
    Finds the onset of fluorescence transients based on peaks in the difference trace.

    Args:
        diff_trace (numpy.ndarray): The difference trace of the fluorescence signal.
        prominence (float, optional): The prominence threshold for peak detection. Defaults to 0.5.

    Returns:
        numpy.ndarray: Indices of detected transient onsets.
    """
    peaks, _ = signal.find_peaks(diff_trace, prominence=prominence * np.percentile(diff_trace, 99))
    return peaks


def segment_transients(fluorescence_trace, transient_onsets):
    """
    Segments individual transients from the fluorescence trace based on the transient onsets.

    Args:
        fluorescence_trace (numpy.ndarray): The full fluorescence trace.
        transient_onsets (numpy.ndarray): Indices of detected transient onsets.

    Returns:
        list: A list of numpy arrays, each containing a segmented transient.
    """
    segments = []
    for i in range(len(transient_onsets) - 1):
        segments.append(fluorescence_trace[transient_onsets[i]: transient_onsets[i + 1]])
    return segments


def calculate_baseline(trace, cycle_length, acquisition_frequency, window=0.2):
    """
    Calculates the baseline intensity for a given fluorescence trace.

    Args:
        trace (numpy.ndarray): The fluorescence trace.
        cycle_length (float): The cycle length of the trace (in seconds).
        acquisition_frequency (float): The acquisition frequency of the trace (in Hz).
        window (float, optional): The fraction of the trace to use for baseline calculation. Defaults to 0.2.

    Returns:
        float: The calculated baseline intensity.
    """
    window_size = int(len(trace) * window)
    last_points = trace[-window_size:]
    baseline = np.median(last_points)
    return baseline


def kinetic_fits(decay_time, decay_portion, p0, fit, bounds=(-100, np.inf), maxfev=100000, normalize=False):
    """
    Derive parameters describing kinematics of transients.

    Args:
        decay_time (numpy.ndarray): Time points for the decay portion.
        decay_portion (numpy.ndarray): Intensity values for the decay portion.
        p0 (tuple): Initial guess for the fit parameters.
        fit (str): Type of fit to perform ('exp' or 'gaussian').
        bounds (tuple, optional): Bounds for the fit parameters. Defaults to (-100, np.inf).
        maxfev (int, optional): Maximum number of function evaluations. Defaults to 100000.
        normalize (bool, optional): Whether to normalize the input data. Defaults to False.

    Returns:
        tuple: Fitted coefficients and tau value.

    Raises:
        NotImplementedError: If an unsupported fit type is specified.
    """
    if normalize:
        decay_time = (decay_time - decay_time.min()) / (decay_time.max() - decay_time.min())
        decay_portion = (decay_portion - decay_portion.min()) / (decay_portion.max() - decay_portion.min())

    if fit == "exp":
        # Fit the exponential decay curve
        coeffs, _ = curve_fit(exp_decay, decay_time, decay_portion, p0=p0, maxfev=maxfev)
    elif fit == "gaussian":
        coeffs, _ = curve_fit(gaussian_exp_decay, decay_time, decay_portion, p0=p0, bounds=bounds, method='trf')
    else:
        raise NotImplementedError("{} fit is not yet implemented.".format(fit))
    tau = 1 / coeffs[1]
    return coeffs, tau


def calcium_transient(t, A, tau_rise, tau_fall, t0):
    """
    Calculate the calcium transient intensity at given time points.

    This function models the calcium transient as a combination of exponential rise and decay.

    Args:
        t (float or numpy.ndarray): Time point(s) at which to calculate the transient intensity.
        A (float): Amplitude of the transient.
        tau_rise (float): Time constant for the rising phase of the transient.
        tau_fall (float): Time constant for the falling phase of the transient.
        t0 (float): Time offset, representing the start of the transient.

    Returns:
        float or numpy.ndarray: Calculated intensity of the calcium transient at the given time point(s).
    """
    return A * (1 - np.exp(-(t - t0) / tau_rise)) * np.exp(-(t - t0) / tau_fall)


def calc_transient_parameters(it_transient, baseline, cycle_length, fit="gaussian"):
    """
    Calculates various parameters for a single transient.
    
    Args:
        it_transient (numpy.ndarray): The calcium transient data.
        baseline (float): The baseline intensity.
        cycle_length (float): The cycle length of the calcium transient (in seconds).
        fit (str, optional): The type of fit to use ('gaussian' or 'exp'). Defaults to "gaussian".
        
    Returns:
        dict: A dictionary containing the calculated parameters for the transient.
    """
    peak = np.max(it_transient[:int(len(it_transient) * 0.7)])
    peak_idx = np.argmax(it_transient[:int(len(it_transient) * 0.7)])
    magnitude = peak - baseline
    fmax_f0 = peak / baseline
    redefined_baseline = baseline + 0.03 * magnitude

    # Calculate temporal parameters
    transient_duration = len(it_transient)
    time_points = np.arange(0, transient_duration)
    normalized_transient = (it_transient - redefined_baseline) / magnitude

    # Fit exponential decay to the fall
    decay_start = peak_idx + int(len(it_transient[peak_idx:]) * 0.1)  # Start 10% after
    baseline_end = np.where(it_transient > redefined_baseline)[0][-1]
    decay_portion = it_transient[decay_start: baseline_end]
    decay_time = np.linspace(0, cycle_length, len(decay_portion))

    # Fit exponential decay to the fall
    rise_start = 0  # np.where(it_transient > redefined_baseline)[0][0]
    rise_portion = it_transient[rise_start: peak_idx]
    # rise_time = np.linspace(0, cycle_length, len(rise_portion))

    # Get rise parameters for left side of curve
    rise_t50 = rise_portion[int(len(rise_portion) * 0.5)]

    # Get fall parameters for right side of curve
    fall_t50 = decay_portion[int(len(decay_portion) * 0.5)]

    # Initial guess for the coefficients
    ra0 = it_transient[rise_start]  # Initial amplitude guess
    fa0 = it_transient[decay_start]  # Initial amplitude guess
    b0 = 0.5  # Initial decay rate guess
    c0 = 3.0
    e0 = 0.1  # Decay rate
    rp0 = [ra0, b0, c0, ra0, e0]
    fp0 = [fa0, b0, c0, fa0, e0]

    # Compute coefs
    # try:
    rise_coeffs, rise_tau = kinetic_fits(rise_start, rise_portion, rp0, fit=fit, bounds=(0, 100))
    # except:
        # rp0 = [ra0, -b0, -c0, ra0, -e0]
        # rise_coeffs, rise_tau = kinetic_fits(rise_start, rise_portion, rp0, fit=fit, bounds=(-np.inf, 100))
    fall_coeffs, fall_tau = kinetic_fits(decay_time, decay_portion, fp0, fit=fit, bounds=(-100, np.inf))
    return {
        'peak': peak,
        'magnitude': magnitude,
        'fmax_f0': fmax_f0,
        'redefined_baseline': redefined_baseline,
        'rise_t50': rise_t50,
        'fall_t50': fall_t50,
        'rise_tau': rise_tau,
        'rise_tau_fit_coeffs': rise_coeffs,
        'fall_tau': fall_tau,
        'fall_tau_fit_coeffs': fall_coeffs,
    }


def exp_decay(x, a, b):
    """
    Exponential decay function for curve fitting.

    Args:
        x (numpy.ndarray): Input values.
        a (float): Amplitude.
        b (float): Decay rate.

    Returns:
        numpy.ndarray: Exponential decay values.
    """
    return a * np.exp(-b * x)


def gaussian_exp_decay(x, a, b, c, d, e):
    """
    Combination of Gaussian and exponential decay functions.

    Args:
        x (numpy.ndarray): Input values.
        a, b, c (float): Parameters controlling the Gaussian component.
        d, e (float): Parameters controlling the exponential decay component.

    Returns:
        numpy.ndarray: Combined Gaussian and exponential decay values.
    """
    gaussian = a * np.exp(-(x - b)**2 / (2 * c**2))
    exp_decay = d * np.exp(-e * x)
    return gaussian + exp_decay


def compute_stats(x, frame, mask, output=None, acquisition_frequency=20, show_plot=False, plot_transients=False, compute_per_trace=False, fit="gaussian", chunk_size=16):
    """
    Run routine to compute calcium transient statistics.

    Args:
        x (numpy.ndarray): Input fluorescence trace.
        frame (numpy.ndarray): Image frame for visualization.
        mask (numpy.ndarray): Mask for the region of interest.
        output (str, optional): Path to save output plots. Defaults to None.
        acquisition_frequency (float, optional): Acquisition frequency in Hz. Defaults to 20.
        show_plot (bool, optional): Whether to display plots. Defaults to False.
        plot_transients (bool, optional): Whether to plot individual transients. Defaults to False.
        compute_per_trace (bool, optional): Whether to compute statistics per trace. Defaults to False.
        fit (str, optional): Type of fit to use ('gaussian' or 'exp'). Defaults to "gaussian".
        chunk_size (int, optional): Minimum size for baseline groups. Defaults to 16.

    Returns:
        tuple: A tuple containing the computed statistics, baseline groups, transient parameters, and the mean trace.
    """
    fluorescence_trace = utils.nonzero_mean(x)
    if fluorescence_trace is None:
        # Some frames have no mask. Kill this cell.
        return {}, None, None, None
    diff_trace = calc_diff_trace(fluorescence_trace)
    transient_onsets = find_transient_onsets(diff_trace)
    cycle_length = np.median(np.diff(transient_onsets))
    transients = segment_transients(fluorescence_trace, transient_onsets)

    # Compute mean trace
    tlens = [len(x) for x in transients]
    if not len(tlens):
        return {}, None, None, None
    max_len = max(tlens)
    padded_transients = []
    for transient in transients:
        zs = max_len - len(transient)
        pad = np.empty(zs)
        pad[:] = np.nan
        transient = np.concatenate((transient, pad))
        padded_transients.append(transient)
    mean_trace = np.nanmean(np.stack(padded_transients, 0), 0)
    baseline = calculate_baseline(mean_trace, cycle_length=cycle_length, acquisition_frequency=acquisition_frequency)
    split_baseline = calculate_baseline(mean_trace, cycle_length=cycle_length, acquisition_frequency=acquisition_frequency, window=0.75)
    baseline_idx = np.where(fluorescence_trace < split_baseline)[0]
    baseline_groups = split_contiguous_frames(baseline_idx, chunk_size)

    # Compute parameters
    transient_parameters = []
    if compute_per_trace:
        for transient in transients:
            params = calc_transient_parameters(transient, baseline, cycle_length=cycle_length, fit=fit)
            transient_parameters.append(params)
            if plot_transients:
                plot_transient(transient, fluorescence_trace, params, cycle_length=cycle_length, fit=fit, frame=frame, mask=mask)
    try:
        params = calc_transient_parameters(mean_trace, baseline, cycle_length=cycle_length, fit=fit)
        plot_transient(mean_trace, fluorescence_trace, params, cycle_length=cycle_length, fit=fit, output=output, frame=frame, mask=mask)
        params.pop("rise_tau_fit_coeffs")
        params.pop("fall_tau_fit_coeffs")
        params = {k: [v] for k, v in params.items()}
        return params, baseline_groups, transient_parameters, mean_trace
    except:
        # mean_trace, baseline
        return {}, None, None, None


def plot_transient(transient, fluorescence_trace, params, cycle_length, fit, frame, mask, output=None):
    """
    Plots a single calcium transient with overlaid parameter values.

    Args:
        transient (np.ndarray): The averaged calcium transient data.
        fluorescence_trace (np.ndarray): All timepoints of calcium data
        params (dict): A dictionary containing the calculated parameters for the transient.
        cycle_length (float): The cycle length of the calcium transient (in seconds).

    Returns:
        None
    """
    time = np.arange(0, len(transient))

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(14, 6))
    ax[0].imshow(frame, cmap="viridis")  # More salient than greyscale
    masked_mask = np.ma.masked_where(mask, mask)
    ax[0].imshow(masked_mask, cmap="Reds", alpha=0.5)
    ax[0].axis("off")
    ax[0].set_title("Cell location")
    ax[1].plot(time, transient, color='k', linewidth=2)

    # Plot baseline
    baseline = params['redefined_baseline']
    ax[1].axhline(y=baseline, color='r', linestyle='--', label='Baseline')

    # Plot peak
    peak = params['peak']
    peak_idx = np.argmax(transient)
    ax[1].scatter(time[peak_idx], peak, color='g', marker='x', s=80, label='Peak')

    # Plot temporal parameters
    rise_t50 = params['rise_t50']
    fall_t50 = params['fall_t50']
    rise_t50_idx = np.argmin(np.abs(transient - rise_t50))
    fall_t50_idx = np.argmin(np.abs(transient - fall_t50))
    ax[1].scatter(rise_t50_idx, rise_t50, color='#03fc13', marker='o', s=50, alpha=0.8, label='50% Rise')
    ax[1].scatter(fall_t50_idx, fall_t50, color='#0d4a11', marker='o', s=50, alpha=0.8, label='50% Fall')

    # Plot tau
    decay_start = peak_idx + int(len(transient[peak_idx:]) * 0.1)  # Start 10% after
    baseline_end = np.where(transient > baseline)[0][-1]
    decay_portion = transient[decay_start: baseline_end]
    decay_time = np.linspace(0, cycle_length, len(decay_portion))
    decay_idx = np.arange(decay_start, baseline_end)
    rise_start = 0
    rise_portion = transient[rise_start: peak_idx]
    rise_time = np.linspace(0, cycle_length, len(rise_portion))
    rise_idx = np.arange(rise_start, peak_idx)

    if fit == "exp":
        fit_fun = exp_decay
    elif fit == "gaussian":
        fit_fun = gaussian_exp_decay
    rise_tau_fit = fit_fun(rise_time, *params['rise_tau_fit_coeffs'])[::-1] + params['rise_tau_fit_coeffs'][2]
    fall_tau_fit = fit_fun(decay_time, *params['fall_tau_fit_coeffs'])
    ax[1].plot(rise_idx, rise_tau_fit, color='#ff14f1', linestyle='--', label=f'Rise Tau Fit (tau={params["rise_tau"]:.2f})')
    ax[1].plot(decay_idx, fall_tau_fit, color='#7a0773', linestyle='--', label=f'Fall Tau Fit (tau={params["fall_tau"]:.2f})')

    ax[1].set_xlabel('Frames')
    ax[1].set_ylabel('Fluorescence Intensity')
    ax[1].set_title('Calcium Transient')
    ax[1].legend()

    ax[2].plot(fluorescence_trace)
    ax[2].set_xlabel('Frames')
    ax[2].set_ylabel('Fluorescence Intensity')
    ax[2].set_title('All calcium timepoints')

    try:
        if output is not None:
            plt.savefig(output)
        else:
            plt.show()
    finally:
        plt.close(fig)


def split_contiguous_frames(baseline_idx, min_size):
    """
    Splits an array of frame indices into groups of contiguous frames.

    Args:
        baseline_idx (numpy.ndarray): An array of frame indices.

    Returns:
        list: A list of numpy arrays, where each array represents a group of contiguous frames.
    """
    groups = []
    current_group = []

    for i in range(len(baseline_idx)):
        if i == 0 or baseline_idx[i] != baseline_idx[i - 1] + 1:
            if current_group:
                groups.append(np.array(current_group))
                current_group = []
        current_group.append(baseline_idx[i])

    if current_group:
        groups.append(np.array(current_group))

    groups = [x for x in groups if len(x) >= min_size]
    return groups
