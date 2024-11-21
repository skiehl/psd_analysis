#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Script for estimating the power spectral density (PSD) of a blazar light
curve.
"""

from datetime import datetime
from math import ceil, floor, log10, sqrt
import os.path
import sys

import numpy as np
from scipy.signal import periodogram
from scipy.interpolate import splrep, splev
from scipy.stats import kstest
from statsmodels.distributions import ECDF
from matplotlib import pyplot as plt
from matplotlib import cm
import matplotlib.gridspec as gs
import tables as tb

import warnings
warnings.simplefilter('ignore', np.RankWarning)

__author__ = "Sebastian Kiehlmann"
__copyright__ = "Copyright 2023, Sebastian Kiehlmann"
__credits__ = ["Sebastian Kiehlmann"]
__license__ = "BSD 3"
__version__ = "4.1"
__maintainer__ = "Sebastian Kiehlmann"
__email__ = "skiehlmann@mail.de"
__status__ = "Production"

#==============================================================================
# Matplotlib configuration
#==============================================================================

cmap = cm.Spectral
plt.rcParams.update({'axes.labelsize': 16.,
                     'axes.titlesize': 16.,
                     'figure.figsize': [12, 7.5],
                     'legend.fontsize': 16.,
                     'legend.numpoints': 1,
                     'legend.scatterpoints': 1,
                     'lines.marker': 'None',
                     'lines.linestyle': '-',
                     'xtick.labelsize': 14.,
                     'ytick.labelsize': 14.})

#==============================================================================
# FUNCTIONS
#==============================================================================

def create_file_dir(filename):
    """
    Extracts a path name if part of the file name, checks whether that path
    exists and if not creates this path.

    Parameters
    -----
    filename : sting
        A path and filename.

    Returns
    -----
    None
    """

    #find dir name, if any:
    filename_rev = filename[::-1]
    try:
        ind = filename_rev.index('/')
    except:
        return False

    #extract dir name:
    ind = len(filename) - ind
    dirname = filename[:ind]

    #check dir and create if not there:
    path = os.path.dirname(dirname)
    if not os.path.exists(path):
        os.makedirs(path)

#==============================================================================

def powerlaw(frequencies, index=1., amplitude=10., frequency=0.1):
    """Returns an array of amplitudes following a power-law over the input
    frequencies.

    Parameters
    -----
    frequencies : 1darray
        Frequencies for which to calculate the power-law in arbitrary units.
    index : float, default=1.
        Power-law index.
    amplitude : float, default=10.
        Power-law amplitude at 'frequency' in arbitrary unit.
    frequency : float, default=0.1
        Frequency for the given 'amplitude' in same unit as 'frequencies'.

    Returns
    -----
    out : 1darray
        Array of same length as input 'frequencies'.
    """

    return amplitude * np.power(frequencies / frequency, -index)

#==============================================================================

def kneemodel(frequencies, index=1., amplitude=10., frequency=0.1):
    """Returns an array of amplitudes following a constant profile that changes
    into a power-law around a given frequency.

    Parameters
    -----
    frequencies : 1darray
        Frequencies for which to calculate the power-law in arbitrary units.
    index : float, default=1.
        Power-law index.
    amplitude : float, default=10.
        Constant amplitude at frequencies below 'frequency' in arbitrary unit.
    frequency : float, default=0.1
        Frequency  in same unit as 'frequencies' at which profile changes
        into a power-law.

    Returns
    -----
    out : 1darray
        Array of same length as input 'frequencies'.
    """

    return amplitude * np.power(
            1 + np.power(frequencies / frequency, 2), -index / 2.)

#==============================================================================

def brokenpowerlaw(
        frequencies, index_lo=1., index_hi=2., amplitude=10., frequency=0.1):
    """Returns an array of amplitudes following a broken power-law.

    Parameters
    -----
    frequencies : array
        Frequencies for which to calculate the power-law in arbitrary units.
    index_hi : float, default=2.
        Power-law index at frequencies lower than 'frequency'.
    index_lo : float, default=1.
        Power-law index at frequencies higher than 'frequency'.
    frequency : float, default=0.1
        Frequency of the power-law break in same unit as 'frequencies'.
    amplitude : float, default=10.
        Amplitude at 'frequency' in arbitrary unit.

    Returns
    -----
        Array of same length as input 'frequencies'.
    """

    return np.where(
            frequencies>frequency,
            amplitude * np.power(frequencies / frequency, -index_hi),
            amplitude * np.power(frequencies / frequency, -index_lo))

#==============================================================================

def sampling(time, average='median', factor=0.1):
    """Prints out the total time and the min, max, median and mean sampling
    of a time series. Returns the total time and a suggested upper limit for
    the simulation sampling.

    Parameters
    -----
    time : 1darray
        Time series.
    average : string, default='median'
        Choose average type ('median' or 'mean') for suggested sampling.
    factor : float, default=0.1
        The suggested sampling is the average sampling times this factor.

    Returns
    -----
    out, out : float, float
        Total time and suggested sampling rate (upper limit).
    """

    total_time = time[-1] - time[0]
    deltat =time[1:] - time[:-1]
    sampling_median = np.median(deltat)
    sampling_mean = np.mean(deltat)
    sampling_min = np.min(deltat)
    sampling_max = np.max(deltat)

    if average=='median':
        sampling_sim = sampling_median * factor
    elif average=='mean':
        sampling_sim = sampling_mean * factor

    print(f'Total time:      {total_time:.3f}')
    print(f'Min. sampling:   {sampling_min:.3f}')
    print(f'Max. sampling:   {sampling_max:.3f}')
    print(f'Mean sampling:   {sampling_mean:.3f}')
    print(f'Median sampling: {sampling_median:.3f}')
    print(f'Suggested simulation sampling: <{sampling_sim:.3f}')

    return total_time, sampling_sim

#==============================================================================

def create_timesteps(time, sampling):
    """Create equally sampled time steps.

    Parameters
    ----------
    time : float
        Total time.
    sampling : float
        Time interval between adjacent time steps.

    Returns
    -------
    np.ndarray
        Time steps.
    """

    # get number of data points and adjust total time:
    N = int(ceil(time / sampling)) + 1
    time = sampling * (N - 1)

    return np.linspace(0, time, N)

#==============================================================================

def simulate_lightcurve_tk(time, sampling, spec_shape, spec_args, seed=False):
    """Create an equally sampled, simulated random light curve following a
    noise process given a spectral shape of the power density spectrum.

    Parameters
    -----
    time : float
        Length of the simulation in arbitrary time unit.
    sampling : float
        Length of the sampling interval in same unit as 'time'.
    spec_shape : func
        Function that takes an array of frequencies and 'spec_args' as input
        and calculates a spectrum for those frequencies.
    spec_args : list
        Function arguments to 'spec_shape'.
    seed : bool, default:False
        Sets a seed for the random generator to get a reproducable result.
        For testing only.

    Returns
    -----
    out : np.ndarray
        The simulated red noise light curve.

    Notes
    -----
    This is an implemention of the algorithm described in [1].

    References
    -----
    [1] Timmer and Koenig, 1995, 'On generating power law noise', A&A, 300, 707
    """

    # get number of data points and adjust total time:
    N = int(ceil(time / sampling)) + 1

    # set spectrum:
    freq = np.fft.rfftfreq(N, sampling)
    freq[0] = 1
    spectrum = spec_shape(freq[1:], *spec_args)
    spectrum[0] = 0
    del freq

    # random (complex) Fourier coefficients for inverse Fourier transform:
    if seed:
        np.random.seed(seed)
    coef = np.random.normal(size=(2, spectrum.shape[0]))

    # if N is even the Nyquist frequency is real:
    if N%2==0:
        coef[-1,1] = 0.
    coef = coef[0] +1j * coef[1]
    coef *= np.sqrt(0.5 *spectrum * N / sampling)

    # inverse Fourier transform:
    lightcurve = np.fft.irfft(coef, N)

    return lightcurve

#==============================================================================

def simulate_lightcurves_tk(
        time, sampling, spec_shape, spec_args, nlcs=1, seed=False):
    """Simulate multiple light curves with the T&K algorithm.

    Parameters
    ----------
    time : float
        Length of the simulation in arbitrary time unit.
    sampling : float
        Length of the sampling interval in same unit as 'time'.
    spec_shape : func
        Function that takes an array of frequencies and 'spec_args' as input
        and calculates a spectrum for those frequencies.
    spec_args : list
        Function arguments to 'spec_shape'.
    nlcs : int, default:1
        Number of light curves to be simulated.
    seed : bool, default:False
        Sets a seed for the random generator to get a reproducable result.
        For testing only.

    Returns
    -------
    lightcurves : np.ndarray
        Two dimensional array of simulated red noise light curves. Each row
        along the first dimension contains one light curve.

    Notes
    -----
    The light curves will be initially created as one long light curve and then
    are split into seperate light curves. Therefore, 'nlcs' does not only
    control the number of final light curves. It also affects to what extend
    lower power frequencies are included in the final light curves, which is
    relevant for taking rednoise leakage into account in the PSD estimation.
    A value of nlcs=10 is recommendable.
    """

    # get number of data points per light curve:
    N = int(ceil(time / sampling)) + 1

    # simulate long lightcurve:
    lightcurves = simulate_lightcurve_tk(
            sampling*N*nlcs, sampling, spec_shape, spec_args, seed=seed)[:-1]

    # reshape to short light curves and normalize each to zero mean:

    if nlcs>1:
        shape = (nlcs, N)
        lightcurves = lightcurves[:N*nlcs].reshape(shape)
        lightcurves -= np.repeat(
                np.mean(lightcurves, axis=1), shape[1]).reshape(shape)

    return lightcurves

#==============================================================================

def adjust_lightcurve_pdf(lightcurve, ecdf, iterations=100, verbose=0):
    """Interatively adjust a simulated red noise light curve to match a target
    probability density function (PDF).

    Parameters
    ----------
    lightcurve : np.ndarray
        The input light curve to adjust.
    ecdf : statsmodels.distributions.ECDF
        Target ECDF as an empirical description of the target PDF.
    iterations : int, default: 100
        Number of iterations.
    verbose : int, default=0
        Controls the amount of information printed.

    Returns
    -------
    lc_sim : np.ndarray
        The adjusted simulated light curve.

    Notes
    -----
    This is an implemention of the algorithm described in [1].

    References
    -----
    [1] Emmanoulopoulos et al., 2013, 'Generating artificial light curves:
        revisited and updated', MNRAS, 433, 2, 907
    """

    # calculate discrete Fourier transform:
    dft_norm = np.fft.rfft(lightcurve)

    #---Emmanoulopoulos-et-al-algorithm----------------------------------------
    # calculate amplitudes based on the random Fourier coefficients:
    N = len(lightcurve)
    ampl_adj = np.absolute(dft_norm)

    # create artificial light curve based on ECDF:
    lc_sim = np.interp(
            np.random.uniform(ecdf.y[1], 1., size=N), ecdf.y, ecdf.x)

    # iteration:
    for i in range(iterations):
        # calculate DFT, amplitudes:
        dft_sim = np.fft.rfft(lc_sim)
        ampl_sim = np.absolute(dft_sim)

        # spectral adjustment:
        dft_adj = dft_sim / ampl_sim * ampl_adj
        lc_adj = np.fft.irfft(dft_adj, n=N)

        # amplitude adjustment:
        a = np.argsort(lc_adj)
        s = np.argsort(lc_sim)
        lc_adj[a] = lc_sim[s]

        if np.max(np.absolute(lc_adj - lc_sim) / lc_sim) < 0.01:
            if verbose:
                print(f'Convergence reached after {i+1} iterations.')
            break
        else:
            lc_sim = lc_adj
    else:
        if verbose:
            print(f'No convergence reached within {iterations} iterations.')

    return lc_sim

#==============================================================================

def simulate_lightcurve_emp(
        time, sampling, spec_shape, spec_args, ecdf, iterations=100,
        seed=False):
    """Create an equally sampled, simulated random light curve following a
    noise process given a spectral shape of the power density spectrum and
    a target probability density function expressed by an ECDF.

    Parameters
    -----
    time : float
        Length of the simulation in arbitrary time unit.
    sampling : float
        Length of the sampling interval in same unit as 'time'.
    spec_shape : func
        Function that takes an array of frequencies and 'spec_args' as input
        and calculates a spectrum for those frequencies.
    spec_args : list
        Function arguments to 'spec_shape'
    ecdf : statsmodels.distributions.ECDF
        Target ECDF as an empirical description of the target PDF.
    iterations : int, default: 100
        Number of iterations for the Emmanoulopoulos et al. algorithm.
    seed : bool, default: False
        Sets a seed for the random generator to get a reproducable result.
        For testing only.

    Returns
    -------
    lc_sim : np.ndarray
        The simulated light curve following a target PSD and PDF.

    Notes
    -----
    This is an implemention of the algorithm described in [1].

    References
    -----
    [1] Emmanoulopoulos et al., 2013, 'Generating artificial light curves:
        revisited and updated', MNRAS, 433, 2, 907
    """

    #---check input and set arguments for spectral shape functions-------------
    if spec_shape == powerlaw:
        try:
            float(spec_args)
        except:
            print("Input error: When spec_shape is 'powerlaw', spec_args " \
                  "needs to be a float (spectral index)!")
            return False
        spec_args = [spec_args, 10., 0.1]

    elif spec_shape == kneemodel:
        try:
            spec_args[0] = float(spec_args[0])
            spec_args[1] = float(spec_args[1])
        except:
            print("Input error: When spec_shape is 'kneemodel', spec_args " \
                  "needs to be a list or tuple of two floats (spectral index" \
                  ", knee frequency)!")
            return False
        spec_args = [spec_args[0], 10., spec_args[1]]

    elif spec_shape == brokenpowerlaw:
        try:
            spec_args[0] = float(spec_args[0])
            spec_args[1] = float(spec_args[1])
            spec_args[2] = float(spec_args[2])
        except:
            print("Input error: When spec_shape is 'brokenpowerlaw', " \
                  "spec_args needs to be a list or tuple of three floats " \
                  "(spectral index low, spectral index high, break " \
                  "frequency)!")
            return False
        spec_args = [spec_args[0], spec_args[1], 10., spec_args[2]]

    #---Timmer-Koenig-algorithm------------------------------------------------
    print('Step 1: TK algorithm')
    # get number of data points and adjust total time:
    N = int(ceil(time / sampling)) + 1

    # set spectrum:
    freq = np.fft.rfftfreq(N, sampling)
    freq[0] = 1
    spectrum = spec_shape(freq, *spec_args)
    spectrum[0] = 0
    del freq

    # random (complex) Fourier coefficients for inverse Fourier transform:
    if seed:
        np.random.seed(seed)

    dft_norm = np.random.normal(size=(2, spectrum.shape[0]))

    # if N is even the Nyquist frequency is real:
    if N % 2 == 0:
        dft_norm[-1,1] = 0.
    dft_norm = dft_norm[0] + 1j * dft_norm[1]
    dft_norm *= np.sqrt(0.5 * spectrum * N / sampling)

    #---Emmanoulopoulos-et-al-algorithm----------------------------------------
    print('Step 2: ECDF based sim. light curve')
    # calculate amplitudes based on the random Fourier coefficients:
    ampl_adj = np.absolute(dft_norm)
    del dft_norm

    # create artificial light curve based on ECDF:
    lc_sim = np.interp(
            np.random.uniform(ecdf.y[1], 1., size=N), ecdf.y, ecdf.x)

    # iteration:
    print('Step 3: iterative spectral and amplitude adjustment...')
    for i in range(iterations):
        sys.stdout.write('\r        Progress: {0:.0f} %'.format(
                (i+1) * 100. / iterations))
        sys.stdout.flush()

        # calculate DFT, amplitudes:
        dft_sim = np.fft.rfft(lc_sim)
        ampl_sim = np.absolute(dft_sim)

        # spectral adjustment:
        dft_adj = dft_sim /ampl_sim *ampl_adj
        lc_adj = np.fft.irfft(dft_adj, n=N)

        # amplitude adjustment:
        a = np.argsort(lc_adj)
        s = np.argsort(lc_sim)
        lc_adj[a] = lc_sim[s]

        if np.max(np.absolute(lc_adj -lc_sim) /lc_sim) < 0.01:
            print(f'\r       Convergence reached after {i+1} iterations.')
            break
        else:
            lc_sim = lc_adj
    else:
        print('\n        No convergence reached.')

    return lc_sim

#==============================================================================

def simulate_lightcurves_emp(
        time, sampling, spec_shape, spec_args, ecdf, nlcs=1, adjust_iter=100,
        verbose=0, seed=False):
    """Simulate multiple light curves with the Emmanoulpopoulous et al.
    algorithm.

    Parameters
    ----------
    time : float
        Length of the simulation in arbitrary time unit.
    sampling : float
        Length of the sampling interval in same unit as 'time'.
    spec_shape : func
        Function that takes an array of frequencies and 'spec_args' as input
        and calculates a spectrum for those frequencies.
    spec_args : list
        Function arguments to 'spec_shape'.
    ecdf : statsmodels.distributions.ECDF
        Target ECDF as an empirical description of the target PDF.
    nlcs : int, default: 1
        Number of light curves to be simulated.
    adjust_iter : int, default: 100
        Number of iterations for the Emmanoulopoulos et al. algorithm.
    verbose : int, default=0
        Controls the amount of information printed.
    seed : bool, default:False
        Sets a seed for the random generator to get a reproducable result.
        For testing only.

    Returns
    -------
    lightcurves : np.ndarray
        Two dimensional array of simulated red noise light curves. Each row
        along the first dimension contains one light curve.

    Notes
    -----
    * This is an implemention of the algorithm described in [1].
    * The light curves will be initially created as one long light curve and
      then are split into seperate light curves. Therefore, 'nlcs' does not
      only control the number of final light curves. It also affects to what
      extend lower power frequencies are included in the final light curves,
      which is relevant for taking rednoise leakage into account in the PSD
      estimation. A value of nlcs=10 is recommendable.

    References
    -----
    [1] Emmanoulopoulos et al., 2013, 'Generating artificial light curves:
        revisited and updated', MNRAS, 433, 2, 907
    """

    #---check input and set arguments for spectral shape functions-------------
    if spec_shape == powerlaw:
        try:
            float(spec_args)
        except:
            print("Input error: When spec_shape is 'powerlaw', spec_args " \
                  "needs to be a float (spectral index)!")
            return False
        spec_args = [spec_args, 10., 0.1]

    elif spec_shape == kneemodel:
        try:
            spec_args[0] = float(spec_args[0])
            spec_args[1] = float(spec_args[1])
        except:
            print("Input error: When spec_shape is 'kneemodel', spec_args " \
                  "needs to be a list or tuple of two floats (spectral index" \
                  ", knee frequency)!")
            return False
        spec_args = [spec_args[0], 10., spec_args[1]]

    elif spec_shape == brokenpowerlaw:
        try:
            spec_args[0] = float(spec_args[0])
            spec_args[1] = float(spec_args[1])
            spec_args[2] = float(spec_args[2])
        except:
            print("Input error: When spec_shape is 'brokenpowerlaw', " \
                  "spec_args needs to be a list or tuple of three floats " \
                  "(spectral index low, spectral index high, break " \
                  "frequency)!")
            return False
        spec_args = [spec_args[0], spec_args[1], 10., spec_args[2]]

    #---create light curve: TK algorithm---------------------------------------
    if verbose:
        print(f'Create long light curve of total time: {time*nlcs:.1f}.')

    # simulate light curves:
    lightcurves = simulate_lightcurves_tk(
            time, sampling, spec_shape, spec_args, nlcs=nlcs, seed=seed)

    #---iterate through light curves-------------------------------------------
    for i in range(nlcs):
        # shell feedback:
        if verbose:
            sys.stdout.write(
                    '\rAdjust amplitudes of short light curves: ' \
                    '{0:.0f} %'.format(i*100./nlcs))
            sys.stdout.flush()

        # adjust amplitude PDF: EMP algorithm:
        lightcurves[i] = adjust_lightcurve_pdf(lightcurves[i], ecdf,
                                               iterations=adjust_iter)
    else:
        if verbose:
            print('\rAdjust amplitudes of short light curves: done.')

    return lightcurves

#==============================================================================

def resample(time, lightcurve, sampling, resample_n=1):
    """Resample a evenly binned light curve.

    Parameters
    ----------
    time : np.ndarray
        Time steps of the original light curve.
    lightcurve : np.ndarray
        Flux density of the original light curve.
    sampling : float or np.ndarray
        Provide a float to resample to even time steps, where the time interval
        is given by this float. Provide the target times in a np.ndarray to
        resample the original data to specific times.
    resample_n : int, default: 1
        If larger than 1 the light curve will be resampled multiple times with
        different zero points.

    Returns
    -------
    time_res : np.ndarray
        The resampled time steps.
    lc_res : np.ndarray
    The resampled flux densities.
    """

    #---check input------------------------------------------------------------
    shape = lightcurve.shape
    resample_n = 1 if resample_n<1 else int(resample_n)

    if resample_n>1 and not isinstance(sampling, float):
        print("WARNING: Multiple resampling of a light curve only possible " \
              "for even sampling. 'resample_n' in resample() set to 1.")
        resample_n = 1

    elif resample_n>1 and len(shape)>1:
        print("WARNING: Multiple light curves will each be resampled only " \
              " once. 'resample_n' in resample() set to 1.")
        resample_n = 1

    #---even sampling----------------------------------------------------------
    if isinstance(sampling, float):
        # create new time steps:
        N = int(floor((time[-1] - time[0]) / sampling))
        time_res = np.linspace(time[0], time[0]+sampling*N, N+1)

    #---uneven sampling--------------------------------------------------------
    elif isinstance(sampling, np.ndarray):
        # sampling interval is not within time interval:
        if sampling[-1]<time[0] or sampling[0]>time[-1]:
            print("WARNING: New time steps are not within given time interval"\
                  ". Cannot resample light curve. Aborted!")
            return False

        # limit resampling time steps to within time (no extrapolation):
        if sampling[0]<time[0] or sampling[-1]>time[-1]:
            i = np.min(np.where(sampling>=time[0])[0])
            j = np.max(np.where(sampling<=time[-1])[0]) + 1
            time_res = sampling[i:j]
            del i, j

        # sampling interval is within time interval:
        else:
            time_res = sampling

        N = len(time_res)

    #---invalid input----------------------------------------------------------
    else:
        print(f"WARNING: type '{type(sampling)}' for input 'sampling' in " \
              "resample() is invalid. Give float or np.1darray. Aborted!")
        return False

    #---resample single light curve once---------------------------------------
    if len(shape)==1 and resample_n==1:
        lc_res = np.interp(time_res, time, lightcurve)

    #---resample single light curve multiple times-----------------------------
    elif len(shape)==1:
        # create time zero point offsets:
        time_offset = np.linspace(0, sampling, resample_n, endpoint=False)
        # delete last last time step, if out of time limit with largest offset:
        if time_res[-1]+time_offset[-1]>time[-1]:
            time_res = time_res[:-1]
        # iterate through resamples:
        lc_res = np.zeros((resample_n, len(time_res)))
        for i, off in enumerate(time_offset):
            lc_res[i] = np.interp(time_res+off, time, lightcurve)

    #---resample multiple light curves-----------------------------------------
    else:
        lc_res = np.zeros((shape[0], N))
        #iterate through light curves:
        for i in range(shape[0]):
            lc_res[i] = np.interp(time_res, time, lightcurve[i])

    return time_res, lc_res

#==============================================================================

def rebin(time, lightcurve, bins, bincenters=None, binlimits='lower'):
    """Bins and averages a light curve according to its time steps.

    Parameters
    -----
    time : np.1darray
        Time steps of the light curve, the light curve is bined according to
        these time stamps.
    lightcurve : np.ndarray
        The signal that is bined and averaged. If 'lightcurve' is a 2darray
        each row is treated as a single light curve, binned and averaged.
    bins : float or array-like
        Defines the lower or upper or two sided bin limits (when 'bincenters'
        is not set) or the bin spread (when 'bincenters' is set).
        When 'bincenters' is not set, there are 3 options:
        1) If 'bins' is a float equally sized bins are created, starting at
        the first time stamp.
        2) If bins is a 1darray it defines the lower of upper bin limits
        depending on 'binlimts'.
        3) If 'bins' is a list of two arrays, the first array defines the
        lower bin limits, the second array the upper limits.
    bincenters : np.1darray, default=None
        When bin centers are given, 'bins' defines the spread of each bin,
        with 3 options:
        1) If 'bins' is a float the bin limits are given by bincenters-/+bins,
        yielding constant bin sizes of 2 times 'bins'.
        2) If 'bins' is a 1darray of the same size as 'bincenters' the bin
        limits are given as in (1) for each bin.
        3) If 'bins' is a list of two 1darrays, the first array defines the
        lower bin spreads, the second array the upper bin spreads from the
        center value.
    binlimits : str, default='lower'
        If 'bincenters' is not set and 'bins' is a 1darray, 'binlimits' sets
        wheather 'bins' are lower or upper bin limits. 'bincenters' overwrites
        'binlimits'.

    Returns
    -----
    out : 1daraay

    """

    time = np.array(time)
    lightcurve = np.array(lightcurve)
    bins = np.array(bins)

    #---create bin limits------------------------------------------------------
    # bin centers and spread given:
    if bincenters is not None:
        bincenters = np.array(bincenters)

        if len(bins.shape)<=1:
            bins = np.array([bincenters-bins, bincenters+bins])
        else:
            bins = np.array([bincenters-bins[0], bincenters+bins[1]])

    # fixed bin size:
    elif len(bins.shape)==0:
        binsize = bins
        bins = np.arange(time[0], time[-1]+binsize, binsize)
        bins = np.array([bins, bins+binsize])
        del binsize

    # lower or upper limits:
    elif len(bins.shape)==1:
        if binlimits=='lower':
            binlimits = bins
            bins = np.zeros((2, len(binlimits)))
            bins[0,:] = binlimits
            bins[1,:-1] = binlimits[1:]
            bins[1,-1] = np.inf
        else:
            binlimits = bins
            bins = np.zeros((2, len(binlimits)))
            bins[1,:] = binlimits
            bins[0,1:] = binlimits[:-1]
            bins[0,0] = -np.inf

    # upper and lower limits given:
    else:
        pass

    bins = bins.T

    #---bin data---------------------------------------------------------------
    # iterate through bins to get data binning:
    selections = []
    for limlo, limhi in bins:
        sel = np.where(np.logical_and(limlo<=time, time<limhi))[0]
        selections.append(sel)

    # create bined data array:
    shape = lightcurve.shape
    if len(shape)==1:
        lightcurve_binned = np.ones(len(bins)) *np.nan
        # iterate through bins:
        for i, sel in enumerate(selections):
            if len(sel)>0:
                lightcurve_binned[i] = np.mean(lightcurve[sel])
    else:
        lightcurve_binned = np.ones((shape[0], len(bins))) *np.nan
        # iterate through light curves:
        for i, lc in enumerate(lightcurve):
            # iterate through bins:
            for j, sel in enumerate(selections):
                if len(sel)>0:
                    lightcurve_binned[i,j] = np.mean(lc[sel])

    return lightcurve_binned

#==============================================================================

def add_errors(lightcurve, errors):
    """Add Gaussian errors to a light curve.

    Parameters
    ----------
    lightcurve : np.ndarray
        The light curve(s).
    errors : float or np.ndarray
        If a float is given, random errors are drawn from a Gaussian
        distribution with zero mean and a standard deviation given by this
        float.
        If a np.array is given that matches the input lightcurve in length,
        the values are randomly shuffled. Then errors are drawn from a Gaussian
        distribution with zero mean and the standard deviation corresponding to
        each individual data point given by the shuffled values
        If a np.array is given that does not match the input lightcurve in
        length, random uncertainties are drawn from the ECDF of 'errors'. Then
        errors are drawn from a Gaussian distribution with zero mean and the
        standard deviation corresponding to each individual data point given by
        random draws from the ECDF.

    Returns
    -------
    np.ndarray
        The input light curve plus randomly drawn errors.
    """

    shape = lightcurve.shape

    #---constant error scale---------------------------------------------------
    if isinstance(errors, float):
        errors = np.random.normal(scale=errors, size=shape)

    #---observed errors: shuffle-----------------------------------------------
    # shuffle observed error, if as many errors are given as light curve points
    elif isinstance(errors, np.ndarray) and len(shape)==1 \
            and errors.shape[0]==shape[0]:
        errors = np.random.shuffle(errors)
        errors = np.random.normal(size=shape) *errors
    elif isinstance(errors, np.ndarray) and len(shape)==2 \
            and errors.shape[0]==shape[1]:
        errors = np.tile(errors, shape[0]).reshape(shape)
        map(np.random.shuffle, errors)
        errors = np.random.normal(size=shape) *errors

    #---observed errors: draw from ECDF----------------------------------------
    # draw random error scales from the error ECDF, if the number of light
    # curve data points differs from the number of given errors
    elif isinstance(errors, np.ndarray):
        ecdf = ECDF(errors)
        errors = np.interp(np.random.uniform(low=ecdf.y[1], size=shape),
                           ecdf.y, ecdf.x)
        errors = np.random.normal(scale=errors, size=shape)

    #---invalid input for errors-----------------------------------------------
    else:
        print(f"WARNING: Data type '{type(errors)}' for input variable " \
              "'errors' in add_errors() is not supported. Give float or " \
              "np.ndarray. Aborted!")
        return False

    return lightcurve + errors

#==============================================================================

def smooth_pg(freq, pg, bins_per_order=10, interpolate=False, verbose=0):
    """Smooth periodogram.

    Parameters
    ----------
    freq : np.ndarray
        Frequencies of the periodogram.
    pg : np.ndarray
        Powers of the periodogram.
    bins_per_order : int, default: 10
        Number of bins per order of magnitude in the covered frequency space.
    interpolate : bool, default: False
        If True, linearly interpolate empty frequency bins. Otherwise, empty
        frequency bins will contain np.nan.
    verbose : int, default=0
        Controls the amount of information printed.

    Returns
    -------
    freq_bin : np.ndarray
        Center frequencies of the binned periodogram.
    pg_bin : np.ndarray
        Power of the binned periodogram.
    pg_uncert : np.ndarray
        Uncertainties of the power of the binned periodogram.
    """

    # set frequency bins:
    order_low = floor(log10(np.min(freq)))
    order_high = ceil(log10(np.max(freq)))
    bins = np.logspace(order_low, order_high,
                       bins_per_order*(order_high-order_low)+1)
    i = np.where(bins<np.min(freq))[0]
    i = i[-1] if len(i)>0 else 0
    j = np.where(bins>np.max(freq))[0]
    j = j[0] +1 if len(j)>0 else len(bins)
    bins = bins[i:j]

    # prepare arrays for bined periodogram:
    freq_bin = np.zeros((3, len(bins)-1))
    freq_bin[0] *= np.nan
    freq_bin[1] = bins[:-1]
    freq_bin[2] = bins[1:]
    pg_bin = np.zeros(len(bins)-1) * np.nan
    pg_uncert = np.zeros(len(bins)-1) * np.nan

    # average bins:
    bind = np.digitize(freq, bins)
    for i in range(len(bins)-1):
        sel = np.where(bind==i+1)[0]
        if len(sel)>0:
            freq_bin[0,i] = np.exp(np.mean(np.log(freq[sel])))
            pg_bin[i] = np.power(10, np.mean(np.log10(pg[sel])))
            pg_uncert[i] = np.power(10, np.std(np.log10(pg[sel])))

    if interpolate:
        # find indices where to interpolate:
        interp_here = np.where(np.isnan(pg_bin))[0]
        interp_to = np.zeros(len(interp_here))
        isf = np.where(np.isfinite(pg_bin))[0]

        for i,h in enumerate(interp_here):
            interp_to[i] = isf[int(np.min(np.where(h<isf)[0]))]
        del isf

        # interpolate linearly in log-log space:
        for i,j in zip(interp_here, interp_to):
            freq_bin[0,i] = 10**(log10(freq_bin[0,i-1]) \
                                 +(log10(freq_bin[0,j]) \
                                 -log10(freq_bin[0,i-1])) /(j-i+1))
            pg_bin[i] = 10**(log10(pg_bin[i-1]) \
                             +(log10(pg_bin[j]) \
                             -log10(pg_bin[i-1])) /(j-i+1))
            if verbose:
                print(f'Data point {i+1} interpolated.')

    return freq_bin, pg_bin, pg_uncert

#==============================================================================

def estimate_periodogram(
        time, signal, split='interactive', split_limit=4,
        resample_rate='median', resample_n=1, resample_split_n=1,
        bins_per_order=10, interpolate=False,
        return_periodograms=False, verbose=0):
    """Estimates the raw, frequency binned periodogram of an unevenly sampled
    light curve. The unevenly sampled input data is resampled at a sampling
    rate of choice. The data may be split at long observation gaps. Then a
    periodogram is calculated for every split data set and for the full data
    set resampled at the longest time step. Resampling may be done multiple
    times at different zero points to estimate the uncertainty of the
    resampling. Individual periodograms are averaged over log-spaced frequency
    bins.

    Parameters
    -----
    time : 1darray
        Time data points of a light curve.
    signal : 1darray
        Intensity data points of a light curve.
    split : string/float/bool, default='interactive'
        Set whether to split the data set at observation gaps longer than a
        certain time step. If set to 'False' or 0. the data will not be split.
        If given a float, this defines the time step threshold. If set to
        'interactive' a histogram of all time steps will be shown and the user
        may decide afterwards whether or not to split the data.
    split_limit : int, default=10
        Sets the minimum number of data points a split data set hase to
        contain. Data sets with fewer numbers are not considered in the
        periodograms. Required minimum number of data points: 4
    resample_rate : string/float, default='median'
        Sets the sampling rate at which the data will be evenly resampled.
        If set to 'median' the median time step is used as sampling rate. If
        set to 'mean' the mean time step is used as sampling rate. If given a
        float, this value is used as sampling rate.
        If data is split, the split data sets will be resamped at this rate,
        the full data set will be resampled at the largest time step. If data
        is not split, the full data set is resampled at this sampling rate.
    resample_n : int, default=1
        Sets how often the full data set will be resampled at different points
        of the original data set.
    resample_split_n : int, default=1
        If data is split, sets how often the split data set will be resampled
        at different points of the original data set.
    bins_per_order : int, default=10
        All individual periodograms (due to splitting and/or multiple
        resampling) will be averaged over log-scaled frequency bins. This sets
        the number of frequency bins per order of 10.
    interpolate : bool, default=False
        If True empty frequency bins are linearly interpolated, otherwise they
        contain NANs
    return_periodograms : bool, default=False
        If 'True' all individual periodograms are returned in a list
        additionnally to the average periodogram.
    verbose : int, default=0
        Controls the amount of information printed.

    Returns
    -----
    out : 5darray
        If 'return_periodograms' is 'False' (default) a structured array is
        returned, indexed 'freq' for the mean bined frequency, 'freqlo' for
        lower bin limits, 'freqhi' for upper bin limits, 'power' for the
        power density and 'uncert' for the uncertainty of the power density in
        each bin.
    out, out : 5darray, list
        If 'return_periodograms' is 'True' the structured array as decribed
        above is returned and a list of structured 2darrays of all individual
        periodograms, indexed with 'freq' for the frequencies and 'power' for
        the spectral density.

    References
    -----
        [1] Uttley et al., 2002, "Measuring the broad-band power spectra of
        active galactic nuclei with RXTE", MNRAS, 332, 231

        [2] Max-Moerbeck et al., 2014, "A method for the estimation of the
        significance of cross-correlations in unevenly sampled red-noise time
        series", MNRAS, 445, 437
    """

    #---check argument inputs--------------------------------------------------
    if not resample_rate in ['median', 'mean']:
        try:
            resample_rate = float(resample_rate)
        except ValueError:
            print("WARNING: Invalid input for argument 'resample_rate'. " \
                  "Either set to 'median', 'mean' or a float.\n" \
                  "Function aborted!")
            return False

    if split_limit < 4:
        split_limit = 4

    #---interactive data splitting---------------------------------------------
    time -= time[0]
    deltat = np.diff(time)
    deltat_max = np.max(deltat)

    if split=='interactive':
        # show histogram of time steps:
        plt.hist(deltat, bins=max(10, len(deltat)/20), log=True)
        plt.xlabel('Time steps $\\Delta t_i$')
        plt.ylabel('counts')
        plt.title('Split the data set at large observation gaps $\\Delta ' \
                 't_\\mathrm{split}$?', fontsize=16.)
        plt.ylim(ymin=0.5)
        plt.show()
        plt.clf()
        plt.close()

        # user input: split data or not:
        split = input(
                "Split the data set at long observation gaps? " \
                "Type 'no' or a number: ")
        while True:
            try:
                split = float(split)
                break
            except ValueError:
                if split in ['no', 'No', 'n', 'N']:
                    split = False
                    break
                else:
                    split = input(
                            "This was not a valid input. " \
                            "Please type 'no' or a number: ")

    #---data splitting---------------------------------------------------------
    if split and split<=deltat_max:
        # get splitting indices:
        split_ind = np.concatenate(([0],
                                    np.where(deltat>=split)[0]+1,
                                    [len(time)+1]))
        nsplits = len(split_ind) -1
        split_ind = [(split_ind[i], split_ind[i+1]) for i in range(nsplits)]

        split_time = []
        split_signal = []
        for i, ind in enumerate(split_ind, start=1):
            if ind[1]-ind[0]>=split_limit:
                split_time.append(time[ind[0]:ind[1]])
                split_signal.append(signal[ind[0]:ind[1]])
        del split_ind

        if verbose > 1:
            print(f'Data split into {len(split_time)} sets at observation ' \
                  f'gaps > {split:.2f}.')
            if nsplits!=len(split_time):
                  print(f'{nsplits-len(split_time)} observation periods had ' \
                        f'fewer than {split_limit} data points and are not ' \
                        'included in the split data sets.')
            for i in range(len(split_time)):
                print('  Set {0:d} time: {1:.1f} - {2:.1f}, total: {3:.1f}' \
                      ''.format(
                        i+1, split_time[i][0], split_time[i][-1],
                        split_time[i][-1]-split_time[i][0]))

        nsplits = len(split_time)

        #---resample full data set---------------------------------------------
        if verbose > 1:
            print(f'Resample full data set {resample_n} times at ' \
                  f'{deltat_max:.1f} sampling:')

        resample_n = int(resample_n)
        res_full_timestep = deltat_max
        del deltat_max

        __, res_full_signal = resample(time, signal, res_full_timestep,
                                       resample_n=resample_n)

        if verbose > 1:
            print('  Done.')

        # if resampled data is too short, do not keep:
        if (resample_n==1 and len(res_full_signal)<split_limit) or \
                (resample_n>1 and res_full_signal.shape[1]<split_limit):
            res_full_signal = None
            if verbose:
                print('NOTE: The total data set resampled at the largest gap '\
                      f'duration has fewer than {split_limit} data points. ' \
                      'Low frequency periodograms will not be included.')

        #---resample split data sets-------------------------------------------
        if verbose > 1 and isinstance(resample_rate, str):
            print(f'Resample {nsplits} split data set {resample_split_n} ' \
                  f'times at {resample_rate} sampling:')
        elif verbose > 1:
            print(f'Resample {nsplits} split data set {resample_split_n} ' \
                  f'times at {resample_rate:.1f} sampling:')

        res_split_timestep = []
        res_split_signal = []

        for i in range(nsplits):
            # set time step for resampling:
            if resample_rate=='median':
                timestep = np.median(np.diff(split_time[i]))
            elif resample_rate=='mean':
                timestep = np.mean(np.diff(split_time[i]))
            else:
                timestep = resample_rate

            __, res_signal = resample(split_time[i], split_signal[i], timestep,
                                      resample_n=resample_split_n)

            # store resampled data only if long enough:
            if (resample_split_n==1 and len(res_signal)>=split_limit) or \
                        (resample_split_n>1 and \
                        res_signal.shape[1]>=split_limit):
                res_split_timestep.append(timestep)
                res_split_signal.append(res_signal)
            elif verbose > 1:
                print(f'The {i+1}. split data set resampled has fewer than ' \
                      f'{split_limit} data points. Periodograms will not be ' \
                     'included.')

            del timestep, res_signal

        nsplits = len(res_split_signal)

        if verbose > 1:
            print('  Done.')

    #---no data splitting------------------------------------------------------
    else:
        if split:
            split = False
            if verbose > 1:
                print('Critical gap time is not exceeded. No data splitting.')
        if verbose > 1 and isinstance(resample_rate, str):
            print(f'Resample full data set {resample_n} times at ' \
                  f'{resample_rate} sampling:')
        elif verbose > 1:
            print(f'Resample full data set {resample_n} times at ' \
                  '{resample_rate:.1f} sampling:')

        # set time step for resampling:
        if resample_rate=='median':
            res_full_timestep = np.median(deltat)
        elif resample_rate=='mean':
            res_full_timestep = np.mean(deltat)
        else:
            res_full_timestep = resample_rate

        __, res_full_signal = resample(
                time, signal, res_full_timestep, resample_n=resample_n)

        if verbose > 1:
            print('  Done.')

    #---periodograms-----------------------------------------------------------
    # NOTE: to calculate the periodogram we use the scipy implementation
    # the signal is linearly de-trended to avoid aliasing contamination
    # and a Hann window function is applied.

    if verbose and split:
        pg_n = nsplits*resample_split_n
        pg_n = pg_n +resample_n if res_full_signal is not None else pg_n
        print(f'Calculate {pg_n} periodograms:')
        del pg_n
    elif verbose > 1:
        print('Calculate {resample_n} periodograms:')

    frequencies = []
    periodograms = []

    # full data set:
    if res_full_signal is not None and resample_n==1:
        freq, pg = periodogram(
                res_full_signal, fs=1./res_full_timestep, window='hann',
                detrend='linear')
        frequencies.append(freq[1:])
        periodograms.append(pg[1:])
    elif res_full_signal is not None:
        for i in range(resample_n):
            freq, pg = periodogram(
                    res_full_signal[i], fs=1./res_full_timestep, window='hann',
                    detrend='linear')
            frequencies.append(freq[1:])
            periodograms.append(pg[1:])
    del res_full_timestep, res_full_signal

    # split data sets:
    if split and resample_split_n==1:
        for __ in range(nsplits):
            freq, pg = periodogram(
                    res_split_signal[0], fs=1./res_split_timestep[0],
                    window='hann', detrend='linear')
            frequencies.append(freq[1:])
            periodograms.append(pg[1:])
            del res_split_timestep[0], res_split_signal[0]

    elif split :
        for __ in range(nsplits):
            for i in range(resample_split_n):
                freq, pg = periodogram(
                        res_split_signal[0][i], fs=1./res_split_timestep[0],
                        window='hann', detrend='linear')
                frequencies.append(freq[1:])
                periodograms.append(pg[1:])
            del res_split_timestep[0], res_split_signal[0]

    if verbose > 1:
        print('  Done.')

    #---average periodogram: bin periodograms----------------------------------
    if verbose > 1:
        print('Calculate average periodogram:')

    frequencies_con = np.concatenate(frequencies)
    periodograms_con = np.concatenate(periodograms)

    # delete or prepare individual periodograms for returning:
    if return_periodograms:
        periodograms_all = []
        for i in range(len(frequencies)):
            periodograms_all.append(np.zeros(len(frequencies[i]),
                    dtype=[('freq', np.float64), ('power', np.float64)]))
            periodograms_all[i]['freq'] = frequencies[i]
            periodograms_all[i]['power'] = periodograms[i]
    else:
        del frequencies, periodograms

    # calculate averaged periodogram:
    freq_bin, pg_bin, pg_uncert = smooth_pg(
            frequencies_con, periodograms_con, bins_per_order=bins_per_order,
            interpolate=interpolate, verbose=verbose)

    # write result into structured array:
    periodogram_avg = np.zeros(
            len(pg_bin), dtype=[
                    ('freq',np.float64), ('freqlo',np.float64),
                    ('freqhi',np.float64), ('power',np.float64),
                    ('uncert', np.float64)])
    periodogram_avg['freq'] = freq_bin[0]
    periodogram_avg['freqlo'] = freq_bin[1]
    periodogram_avg['freqhi'] = freq_bin[2]
    periodogram_avg['power'] = pg_bin
    periodogram_avg['uncert'] = pg_uncert

    if verbose > 1:
        print('  Done.')

    # return average periodogram and optionally individual periodograms:
    if return_periodograms:
        return periodogram_avg, periodograms_all
    else:
        return periodogram_avg

#==============================================================================

def create_psd_db(
        dbfile, data_time, data_signal, data_err, ntime, sim_sampling,
        spec_shape='powerlaw', split='interactive', split_limit=4,
        resample_rate='median', resample_n=1, resample_split_n=1,
        bins_per_order=10, interpolate=False, scaling='pdf',
        time_bins=None, verbose=1):
    """Creates a data base file to store simulated power-law PSDs (power
    spectra densities) in. This data base includes general parameters,
    simulation parameters, the corresponding data light curve and PSD and
    (later-on) the simulated PSDs.

    Parameters
    -----
    dbfile : string
        Filename of the database.
    data_time : 1darray
        Time data points of a light curve.
    data_signal : 1darray
        Intensity data points of a light curve.
    data_err : 1darray
        One sigma errors of the light curve.
    ntime : int
        Factor by which the simulated light curves will be longer than the data
        lightcurve to include lower frequency contribution in the sim. PSDs.
        E.g. ntime=1000. Optimal number of simulations to run is multiples of
        ntime.
    sim_sampling : float
        Time sampling interval for the simulated light curves. Should be
        shorter than 'data_sampling'.
    spec_shape : str, default='powerlaw'
        Set the spectral model type for the simulated light curves.
    split : string/float/bool, default='interactive'
        Set whether to split the data set at observation gaps longer than a
        certain time step. If set to 'False' or 0. the data will not be split.
        If given a float, this defines the time step threshold. If set to
        'interactive' a histogram of all time steps will be shown and the user
        may decide afterwards whether or not to split the data.
    split_limit : int, default=10
        Sets the minimum number of data points a split data set hase to
        contain. Data sets with fewer numbers are not considered in the
        periodograms. Required minimum number of data points: 4
    resample_rate : string/float, default='median'
        Sets the sampling rate at which the data will be evenly resampled.
        If set to 'median' the median time step is used as sampling rate. If
        set to 'mean' the mean time step is used as sampling rate. If given a
        float, this value is used as sampling rate.
        If data is split, the split data sets will be resamped at this rate,
        the full data set will be resampled at the largest time step. If data
        is not split, the full data set is resampled at this sampling rate.
    resample_n : int, default=1
        Sets how often the full data set will be resampled at different points
        of the original data set.
    resample_split_n : int, default=1
        If data is split, sets how often the split data set will be resampled
        at different points of the original data set.
    bins_per_order : int, default=10
        All individual periodograms (due to splitting and/or multiple
        resampling) will be averaged over log-scaled frequency bins. This sets
        the number of frequency bins per order of 10.
    interpolate : bool, default=False
        Set whether or not empty bins of the power spectrum should be linearly
        interpolated or not.
    scaling : str, default='pdf'
        Set the method to scale the simulated light curve amplitudes. Set to
        'pdf' the Emmanoulopoulos et al. algorithm is used to simulate light
        curves. Set to 'std' the Timmer and Koenig algorithm is used and light
        curve amplitudes are scaled through the flux standard deviation.
    time_bins : 2darray, default=None
        Averaging time intervals of the observed data. Stored in the data base.
        If set, then simulated light curves will not be interpolated to the
        observed time grid but averaged over the time bins.
    verbose : int, default=1
        Controls the amount of information printed.
    """

    #---check input------------------------------------------------------------
    if spec_shape not in ['powerlaw', 'kneemodel', 'brokenpowerlaw']:
        print(f"WARNING: spectral shape '{spec_shape}' in create_psd_db() is "\
              "not defined. Set to 'powerlaw', 'kneemodel' or " \
              "'brokenpowerlaw'. No data base file created!")
        return False

    #---check if file exists---------------------------------------------------
    if os.path.isfile(dbfile):
        userio = input(
                f'Data file {dbfile} already exists. Overwrite file (y/n)? ')
        if not userio.lower() in ('y', 'yes', 'make it so!'):
            print('Existing file not overwritten.')
            return False

    #---interactive data splitting---------------------------------------------
    data_time -= data_time[0]
    deltat = np.diff(data_time)

    if split=='interactive':
        # show histogram of time steps:
        plt.clf()
        plt.hist(deltat, bins=max(10, len(deltat)/20), log=True)
        plt.xlabel('Time steps $\\Delta t_i$')
        plt.ylabel('counts')
        plt.title('Split the data set at large observation gaps $\\Delta ' \
                 't_\\mathrm{split}$?', fontsize=16.)
        plt.ylim(ymin=0.5)
        plt.show(block=False)

        # user input: split data or not:
        split = input(
                "Split the data set at long observation gaps? " \
                "Type 'no' or a number: ")
        while True:
            try:
                split = float(split)
                break
            except ValueError:
                if split in ['no', 'No', 'n', 'N']:
                    split = 0.
                    break
                else:
                    split = input(
                            "This was not a valid input. " \
                            "Please type 'no' or a number: ")
    plt.clf()
    plt.close()

    #---calculate data PSD-----------------------------------------------------
    if verbose > 1:
        print('Calculate data raw periodogram..')

    data_periodogram = estimate_periodogram(
            data_time, data_signal, split=split, resample_rate=resample_rate,
            resample_n=resample_n, resample_split_n=resample_split_n,
            bins_per_order=bins_per_order, interpolate=interpolate,
            split_limit=split_limit, verbose=verbose)
    nfreq = len(data_periodogram)

    #---create data base file--------------------------------------------------
    # define table columns:
    class GenParamCols(tb.IsDescription):
        ntime = tb.Float32Col()
        time = tb.Float32Col()
        sampling = tb.Float32Col()
        binning = tb.BoolCol()
        split = tb.Float32Col()
        resample_rate = tb.Float32Col()
        resample_n = tb.Int32Col()
        resample_split_n = tb.Int32Col()
        bins_per_order = tb.Int32Col()
        interpolate = tb.Int32Col()
        split_limit = tb.Int32Col()
        scaling = tb.StringCol(3)
        spectrum = tb.StringCol(14)

    class SimParamColsPL(tb.IsDescription):
        index = tb.Float32Col()

    class SimParamColsKM(tb.IsDescription):
        index = tb.Float32Col()
        freq = tb.Float32Col()

    class SimParamColsBPL(tb.IsDescription):
        indexlo = tb.Float32Col()
        indexhi = tb.Float32Col()
        freq = tb.Float32Col()

    class ScaleParamCol(tb.IsDescription):
        amplitude = tb.Float32Col()

    # create directory if necessary:
    create_file_dir(dbfile)

    # create a file and groups:
    h5file = tb.open_file(dbfile, mode = 'w',
                          title = 'Power-law PSD simulation data base')
    grp_lc = h5file.create_group('/', 'lightcurve', 'Data light curve')
    grp_psd = h5file.create_group('/', 'psds', 'Power spectral densities')

    # create parameter tables:
    tb_genpar = h5file.create_table(grp_psd, 'genpar', GenParamCols,
                                    'General parameters')
    if spec_shape=='powerlaw':
        h5file.create_table(grp_psd, 'simpar', SimParamColsPL,
                            'Simulation parameters')
    elif spec_shape=='kneemodel':
        h5file.create_table(grp_psd, 'simpar', SimParamColsKM,
                            'Simulation parameters')
    elif spec_shape=='brokenpowerlaw':
        h5file.create_table(grp_psd, 'simpar', SimParamColsBPL,
                            'Simulation parameters')
    h5file.create_table(grp_psd, 'scalepar', ScaleParamCol,
                        'Scaling parameter')

    # write general parameters:
    row_pars = tb_genpar.row
    row_pars['ntime'] = ntime
    row_pars['time'] = data_time[-1]
    row_pars['sampling'] = sim_sampling
    row_pars['binning'] = False if time_bins is None else True
    row_pars['split'] = split
    if resample_rate == 'mean':
        store_resrate = -1.
    elif resample_rate == 'median':
        store_resrate = -2.
    else:
        store_resrate = resample_rate
    row_pars['resample_rate'] = store_resrate
    del store_resrate
    row_pars['resample_n'] = resample_n
    row_pars['resample_split_n'] = resample_split_n
    row_pars['bins_per_order'] = bins_per_order
    row_pars['interpolate'] = interpolate
    row_pars['split_limit'] = split_limit
    row_pars['scaling'] = scaling
    row_pars['spectrum'] = spec_shape

    row_pars.append()
    tb_genpar.flush()

    # store light curve:
    h5file.create_array(grp_lc, 'lightcurve',
                       np.array([data_time, data_signal, data_err]),
                      'Data light curve')

    # store time bins:
    if time_bins is not None:
        time_bins = np.array(time_bins)
        h5file.create_array(grp_lc, 'timebins', time_bins, 'Time bins')

    # store frequencies and data PSD:
    h5file.create_array(grp_psd, 'frequencies', data_periodogram['freq'],
                       'PSD frequencies')
    h5file.create_array(grp_psd, 'datapsd', data_periodogram['power'],
                       'Data PSD')
    h5file.create_array(grp_psd, 'datapsderr', data_periodogram['uncert'],
                       'Data PSD uncertainty')

    # create extendable array for simulation PSDs:
    h5file.create_earray(grp_psd, 'simpsd', tb.Float64Atom(), (0,nfreq),
                        'Simulation PSDs', expectedrows=ntime)

    # feedback:
    if verbose > 0:
        print(f'New data base file for simulated power-law PSDs: {dbfile}.')

    if verbose > 1:
        print(h5file)

    h5file.close()

#==============================================================================

def run_sim(
        dbfile, iterations, index=None, indexlo=None, indexhi=None,
        freq=None, verbose=1):
    """Runs light curve simulations for given spectral model input parameters,
    estimates the power spectral densities (PSDs) and stores the PSDs in a
    given data base.

    Parameters
    -----
    dbfile : string
        Filename of the database the simulation results are stored in.
    iterations : int
        Number of PSDs to calculate.
    index : float, default=None
        Power-law spectral index. Needs to be set if the data base is set to
        'powerlaw' or 'kneemodel' spectral model.
    indexlo : float, default=None
        Power-law spectral index at low frequencies. Needs to be set if the
        data base is set to 'brokenpowerlaw' spectral model.
    indexhi : float, default=None
        Power-law spectral index at high frequencies. Needs to be set if the
        data base is set to 'brokenpowerlaw' spectral model.
    freq : float, default=None
        Turn over frequency at which the spectral model bends. Needs to be set
        if the data base is set to 'kneemodel' or 'brokenpowerlaw' spectral
        model.
    verbose : int, default=1
        Controls the amount of information printed.
    """

    # set simulation parameter:
    adjust_iter = 200   # set the number of iterations if EMP algorithm is used

    #---open data base file----------------------------------------------------
    h5file = tb.open_file(
        dbfile, mode = 'a', title = 'Power-law PSD simulation data base')
    tb_par = h5file.root.psds.simpar
    tb_par_row = tb_par.row
    tb_scalepar = h5file.root.psds.scalepar
    tb_scalepar_row = tb_scalepar.row
    arr_psd = h5file.root.psds.simpsd

    # read general parameters:
    ntime = int(h5file.root.psds.genpar[0]['ntime'])
    sampling = h5file.root.psds.genpar[0]['sampling']
    try:
        binning = h5file.root.psds.genpar[0]['binning']
    except:
        print('Note: this data base is an old one and does not allow binning' \
              ' yet. It is used as non-binned data sets.')
        binning = False
    split = h5file.root.psds.genpar[0]['split']
    resample_rate = h5file.root.psds.genpar[0]['resample_rate']
    resample_n = h5file.root.psds.genpar[0]['resample_n']
    resample_split_n = h5file.root.psds.genpar[0]['resample_split_n']
    bins_per_order = h5file.root.psds.genpar[0]['bins_per_order']
    interpolate = h5file.root.psds.genpar[0]['interpolate']
    split_limit = h5file.root.psds.genpar[0]['split_limit']
    if resample_rate < -1.5:
        resample_rate = 'median'
    elif resample_rate < 0.:
        resample_rate = 'mean'
    scaling = h5file.root.psds.genpar[0]['scaling'].decode('utf-8')
    spec_shape = h5file.root.psds.genpar[0]['spectrum'].decode('utf-8')

    #---check input------------------------------------------------------------
    if spec_shape=='powerlaw' and not isinstance(index, float):
        print("WARNING: This data base is set to power-law spectrum. " \
              "Function parameter 'index' has to be a float. " \
              "Simulation aborted!")
        h5file.close()
        return False

    elif spec_shape=='kneemodel' and not \
            (isinstance(index, float) and isinstance(freq, float)):
        print("WARNING: This data base is set to knee model spectum. " \
              "Function parameters 'index' and 'freq' have to be floats. " \
              "Simulation aborted!")
        h5file.close()
        return False

    elif spec_shape=='brokenpowerlaw' and not (isinstance(indexlo, float) \
            and isinstance(indexhi, float) and isinstance(freq, float)):
        print("WARNING: This data base is set to broken power-law spectum. " \
              "Function parameters 'indexlo', 'indexhi' and 'freq' have to " \
              "be floats. Simulation aborted!")
        h5file.close()
        return False

    #---get scaling variables--------------------------------------------------
    time_data = h5file.root.lightcurve.lightcurve[0]
    if binning:
        bins = h5file.root.lightcurve.timebins[:]
        if len(bins.shape)==2:
            bin0 = bins[0,0]
            binN = bins[1,-1]
        else:
            bin0 = bins[0]
            binN = bins[-1]
    flux = h5file.root.lightcurve.lightcurve[1]
    errors = h5file.root.lightcurve.lightcurve[2]

    # determine the data flux empirical cumulative distribution function (ECDF)
    if scaling=='pdf':
        ecdf = ECDF(flux)

    # calculate data signal variance:
    elif scaling=='std':
        variance = np.std(flux)**2
        mean_err_var = np.mean(np.power(errors, 2))

        if mean_err_var < variance:
            variance -= mean_err_var
        else:
            # if variance is smaller than errors, let's assume half of the
            # variance is intrinsic; because variance must not ne zero:
            variance /= 2

    del flux

    #---run simulations--------------------------------------------------------
    # create sim. time steps for binned data:
    if binning:
        total_time = time_data[-1] +bin0 +binN
        time_data += bin0
    # create sim. time steps for unbinned data:
    else:
        total_time = time_data[-1]

    time_sim = create_timesteps(total_time, sampling)

    # set number of simulations:
    if iterations%ntime==0:
        simulations = int(iterations /ntime)
    else:
        simulations = int(iterations /ntime) +1

    # feedback:
    if verbose > 0:
        print('Number of simulations: ' \
            f'{simulations} long lightcurves split into {ntime} short light ' \
            f'curves, yielding {simulations*ntime} PSDs.')

    # feedback:
    if verbose > 0:
        sim_start = datetime.now()
        print('Start simulations on {0:s} ...'.format(
                sim_start.strftime('%a %H:%M:%S')))
        sys.stdout.flush()

    # iterate through light curve simulations:
    for i in range(simulations):
        # feedback:
        if verbose > 1:
            sys.stdout.write(
                f'Simulate and analyse light curves {i*ntime+1}-{(i+1)*ntime} ' \
                f'of {simulations*ntime}.\n')
            sys.stdout.flush()
            sim_inter = datetime.now()

        #---create light curves: TK algorithm----------------------------------
        # feedback:
        if verbose > 1:
            sys.stdout.write('  Simulate long light curve..')
            sys.stdout.flush()

        # simulate light curve: power-law model:
        if spec_shape=='powerlaw':
            lightcurves = simulate_lightcurves_tk(
                    time_sim[-1], sampling, powerlaw, [index, 10., 0.1],
                    nlcs=ntime)

        # simulate light curve: knee model:
        elif spec_shape=='kneemodel':
            lightcurves = simulate_lightcurves_tk(
                    time_sim[-1], sampling, kneemodel, [index, 10., freq],
                    nlcs=ntime)

        # simulate light curve: knee model:
        elif spec_shape=='brokenpowerlaw':
            lightcurves = simulate_lightcurves_tk(
                    time_sim[-1], sampling, brokenpowerlaw,
                    [indexlo, indexhi, 10., freq], nlcs=ntime)

        else:
            print("Something's wrong... this should not happen!")
            break

        shape = lightcurves.shape

        # feedback:
        if verbose > 1:
            sys.stdout.write(' done in {0}.\n'.format(
                    datetime.now()-sim_inter))
            sys.stdout.flush()
            sim_inter = datetime.now()

        #---match PDF: EMP algorithm-------------------------------------------
        if scaling=='pdf':
            for j in range(ntime):
                # feedback:
                if verbose > 1:
                    sys.stdout.write(
                            '\r  Adjust PDFs of short light curves: ' \
                            f'{j*100./ntime} %')
                    sys.stdout.flush()

                # adjust amplitude PDF: EMP algorithm:
                lightcurves[j] = adjust_lightcurve_pdf(
                        lightcurves[j], ecdf, iterations=adjust_iter)
            # feedback:
            if verbose > 1:
                sys.stdout.write(
                        '\r  Adjust PDFs of short light curves: done in ' \
                        '{0}.\n'.format(
                                datetime.now()-sim_inter))
                sys.stdout.flush()

            # factors needed for storage:
            factors = np.zeros(shape[0])

        #---match variance-----------------------------------------------------
        if scaling=='std':
            # feedback:
            if verbose > 1:
                sys.stdout.write('  Adjust variances of short light curves..')
                sys.stdout.flush()

            # match variances:
            factors = np.std(lightcurves, axis=1) /sqrt(variance)
            lightcurves /= np.repeat(factors, shape[1]).reshape(shape)

            # feedback:
            if verbose > 1:
                sys.stdout.write(' done in {0}.\n'.format(
                    datetime.now()-sim_inter))
                sys.stdout.flush()

            # adjust factors for storage:
            factors = 10. / factors**2

        sim_inter = datetime.now()

        #---resample-----------------------------------------------------------
        if binning:
            if verbose > 1:
                sys.stdout.write('  Bin light curves..')
                sys.stdout.flush()

            lightcurves = rebin(
                    time_sim, lightcurves, bins, bincenters=time_data+bin0)

        else:
            if verbose > 1:
                sys.stdout.write('  Resample light curves..')
                sys.stdout.flush()

            lightcurves = resample(time_sim, lightcurves, time_data)[1]

        if verbose > 1:
            sys.stdout.write(' done in {0}.\n'.format(datetime.now()-sim_inter))
            sys.stdout.flush()
            sim_inter = datetime.now()

        #---simulate light curve errors----------------------------------------
        if verbose > 1:
            sys.stdout.write('  Simulate observational errors..')
            sys.stdout.flush()

        lightcurves = add_errors(lightcurves, errors)

        if verbose > 1:
            sys.stdout.write(' done in {0}.\n'.format(datetime.now()-sim_inter))
            sys.stdout.flush()
            sim_inter = datetime.now()

        #---estimate periodograms----------------------------------------------
        # iterate through light curves:
        for j, lc in enumerate(lightcurves):
            # feedback:
            if verbose > 1:
                sys.stdout.write('\r  Estimate periodograms: {0:.1f} %'.format(
                        (i*ntime+j+1)*100./simulations/ntime))
                sys.stdout.flush()

            # periodogram:
            psd = estimate_periodogram(
                    time_data, lc, split=split, split_limit=split_limit,
                    resample_rate=resample_rate, resample_n=resample_n,
                    resample_split_n=resample_split_n,
                    bins_per_order=bins_per_order, interpolate=interpolate)

            # store PSD:
            arr_psd.append(np.expand_dims(psd['power'], axis=0))
            # store parameters:
            tb_par_row['index'] = index
            tb_par_row.append()
            tb_par.flush()
            tb_scalepar_row['amplitude'] = factors[j]
            tb_scalepar_row.append()
            tb_scalepar.flush()

        if verbose > 1:
            sys.stdout.write('\r  Estimate periodograms: done in {0}.\n'.format(
                    datetime.now()-sim_inter))
            sys.stdout.flush()
            sim_inter = datetime.now()

        del lightcurves

        # feedback:
        if verbose > 1:
            print('Total progress: {0:.0f} % done in {1}.'.format(
                    (i+1)*100./simulations, datetime.now()-sim_start))

    # feedback:
    if verbose > 0:
        print('{0} simulations done in {1}.'.format(
                    ntime*simulations, datetime.now()-sim_start))

    # close data base file:
    h5file.close()

#==============================================================================

def smooth(x, window_len=11, window='hanning'):
    """Smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    Parameters
    -----
   x : np.1darray
       The input signal ;
   window_len : int
       The dimension of the smoothing window; should be an odd integer, will
       be increased to the next odd integer otherwise.
   window : string, default='hanning'
       The type of window from 'flat', 'hanning', 'hamming', 'bartlett',
       'blackman'. Flat window will produce a moving average smoothing.

    Returns
    -----
    output : np.1darray
       The smoothed signal.
    """

    if x.ndim != 1:
       raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
       raise ValueError("Input vector needs to be bigger than window size.")

    if window_len<3:
       return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
       raise ValueError("Window is on of 'flat', 'hanning', 'hamming', " \
               "'bartlett', 'blackman'")

    if not window_len%2:
        window_len = int(window_len) + 1

    s = np.r_[x[window_len-1:0:-1], x, x[-1:-window_len:-1]]

    if window=='flat':
       w = np.ones(window_len,'d')
    else:
       w = eval(f'np.{window}(window_len)')

    y = np.convolve(w / w.sum(), s, mode='valid')
    drop = window_len // 2

    return y[drop:-drop]

#==============================================================================

def evaluate_sim(
        dbfile, output_dir='', output_suffix='dbfile', sign_level=0.05,
        smooth_confint=False):
    """Evaluates a data base of simulated power spectral densities. Results are
    saved in plots and a text file.

    Parameters
    -----
    dbfile : string
        Directory and file name of the simulation data base.
    output_dir : string, default=''
        Directory where the results are saved.
    output_suffix : string, default='dbfile'
        A suffix that is used for all resulting files. If 'dbfile' (default)
        the file name of the data base is used as suffix.
    sign_level : float, default=0.05
        Significance level at which the confidence interval for the spectral
        index is determined.
    smooth_confint : int, default=False
        If set, the p-values used to calculate the confidence interval are
        smoothed before fitted with a cubic spline. The number given sets the
        number of data points averaged. The number needs to be uneven. For even
        numbers the number will be increased by one.
    """

    #---access data base-------------------------------------------------------
    print(f'Access data base: {dbfile}')

    h5file = tb.open_file(
            dbfile, mode = 'r', title = 'Power-law PSD simulation data base')

    lightcurve = h5file.root.lightcurve.lightcurve[:]
    frequencies = h5file.root.psds.frequencies[:]
    datapsd = h5file.root.psds.datapsd[:]
    simpsd = h5file.root.psds.simpsd
    simpar = h5file.root.psds.simpar
    genpar = h5file.root.psds.genpar
    scaling = genpar[0]['scaling']
    scalepar = h5file.root.psds.scalepar

    nfreq = len(frequencies)
    indices = np.unique(simpar.cols.index[:])
    nind = len(indices)

    #---create output directory and file---------------------------------------
    # create directory if necessary:
    if len(output_dir)>0 and output_dir[-1]!='/':
        output_dir = '%s/' % output_dir
    create_file_dir(output_dir)

    # create text file:
    if output_suffix=='dbfile':
        try:
            i = dbfile[::-1].index('/')
        except:
            i = len(dbfile)
        i = len(dbfile) -i
        output_suffix = '%s_' % dbfile[i:-3]
        resfile = '%s%s%s' % (output_dir, output_suffix, 'results.txt')
    write_to = open(resfile, 'w')

    #---write general information to results file------------------------------
    resample_rate = genpar.cols.resample_rate[0]
    if resample_rate<-1.5:
        resample_rate = 'median'
    elif resample_rate<-0.5:
        resample_rate = 'mean'
    else:
        resample_rate = f'{resample_rate:.2f}'

    text = 'Data base file: {0:s}\n' \
           '--------------------------------------------------\n' \
           'Light curve:\n' \
           'Total time: {1:.2f}\n' \
           'Split at gaps: {2:.2f}\n' \
           'Full data resampled {3:d} times.\n' \
           'Split data resampled {4:d} times.\n' \
           'Resample rate: {5:s}\n' \
           '--------------------------------------------------\n' \
           'Light curve simulation:\n' \
           'Total time: {6:.2f} * {7:.0f}\n' \
           'Initial sampling rate: {8:.2f}\n' \
           '--------------------------------------------------\n' \
           'Power spectral density: {9:s}\n' \
           'Amplitude scaling: {10:s}\n' \
           'Bins per order: {11:d}\n\n'.format(
               dbfile, genpar.cols.time[0], genpar.cols.split[0],
               genpar.cols.resample_n[0], genpar.cols.resample_split_n[0],
               resample_rate, genpar.cols.time[0], genpar.cols.ntime[0],
               genpar.cols.sampling[0], genpar.cols.spectrum[0],
               genpar.cols.scaling[0], genpar.cols.bins_per_order[0])
    write_to.write(text)
    del resample_rate, text

    #---plot light curve-------------------------------------------------------
    print('Create figure: light curve.. ', end='')
    sys.stdout.flush()

    fig, axes = plt.subplots(1)
    fig.suptitle('Observed lightcurve', fontsize=16.)
    axes.set_xlabel('time')
    axes.set_ylabel('signal')
    axes.plot(
            lightcurve[0], lightcurve[1], marker='o', linestyle=':', color='k')
    plt.savefig(
            f'{output_dir}{output_suffix}1_lightcurve.png', dpi=100)
    plt.clf()
    plt.close()
    del fig, axes
    print('done.')

    #---average models, chi square---------------------------------------------
    print('Calculate average models and calculate chi squares.. ', end='')
    sys.stdout.flush()

    # set storage arrays:
    simpsd_avg = np.zeros((nind, nfreq))
    simpsd_rms = np.zeros((nind, nfreq))
    chisqs_obs = np.zeros(nind)

    # iterate through spectral indices:
    for i, index in enumerate(indices):
        # select simulations:
        sel = np.where(simpar.cols.index[:] == index)[0]
        simpsd_sel = simpsd[:][sel]
        shape = simpsd_sel.shape

        with warnings.catch_warnings():
            # do not show warning for NAN slices:
            warnings.simplefilter("ignore")
            # calculate average simulation:
            simpsd_avg[i] = np.nanmean(simpsd_sel, axis=0)
            # calculate RMS spread for all frequencies (omit the root):
            simpsd_rms[i] = np.nanmean(np.power(simpsd_sel \
                    -np.tile(simpsd_avg[i], shape[0]).reshape(shape), 2),
                    axis=0)
            # calculate chi sqare:
            chisqs_obs[i] = np.nansum(np.power(datapsd -simpsd_avg[i], 2) \
                    /simpsd_rms[i])

    print('done.')

    #---plot average model PSDs------------------------------------------------
    print('Create figure: average model PSDs.. ', end='')
    sys.stdout.flush()

    fig, axes = plt.subplots(1)
    fig.suptitle(
            'PSD results:\nObserved raw PSD and average model PSDs ' \
            f'$\\beta={indices[0]:.2f}-{indices[-1]:.2f}$',
            fontsize=16.)
    axes.set_xlabel('frequency')
    axes.set_ylabel('power')
    for i, index in enumerate(indices):
        color = cmap(i*1./(len(indices)-1))
        axes.plot(
                frequencies, simpsd_avg[i], marker='o', ms=1., mfc=color,
                mec=color, color=color)
    axes.plot(
            frequencies, datapsd, marker='o', ms=2., linewidth=2, color='k',
            label='obs.')
    axes.legend(loc='best', ncol=4)
    axes.set_xscale('log')
    axes.set_yscale('log')
    # NOTE: runtime error occurs here when log scale is chosen and NANs are
    # included
    plt.savefig(
            f'{output_dir}{output_suffix}2_avgPSDs.png', dpi=100)
    plt.clf()
    plt.close()
    del fig, axes, frequencies
    print('done.')

    #---plot fitted model PSD amplitudes---------------------------------------
    # create legend entries only for a few spectral indices:
    max_legendentries = 20
    label = np.floor(np.linspace(0, nind-1, max_legendentries))

    if scaling=='std':
        print('Create figure: model amplitudes.. ', end='')
        sys.stdout.flush()

        fig, axes = plt.subplots(1)
        fig.suptitle('PSD results:\nFitted PSD model amplitudes', fontsize=16.)
        axes.set_xlabel('fit. model amplitude $A_\\mathrm{fit}$')
        #axes.set_ylabel('count')
        for i, index in enumerate(indices):
            # select simulations:
            sel = np.where(simpar.cols.index[:] == index)[0]
            # plot histogram:
            axes.hist(
                    scalepar.cols.amplitude[:][sel],
                    range=(0, ceil(np.max(scalepar.cols.amplitude[:]))),
                    bins=100, color=cmap(i*1./nind), alpha=0.5,
                    label='$\\beta={0}$'.format(index if i in label else ''))
        axes.legend(loc='upper right', ncol=5)
        plt.savefig(
                f'{output_dir}{output_suffix}3_fitAmpl.png', dpi=100)
        plt.clf()
        plt.close()
        del fig, axes
        print('done.')

    #---minimize chi square----------------------------------------------------
    print('Minimize chi square.. ', end='')
    sys.stdout.flush()

    # select spectral indices with small chi square:
    for sel_order in range(1, 10):
        sel = np.where(chisqs_obs<=10.**sel_order*np.min(chisqs_obs))[0]
        if len(sel)>4:
            break

    # set two arbitrary starting numbers:
    best_index = {'former': 0.}
    min_chisq = {'former': 0.}
    poly = {}

    # create x-values for polynomial evaluation:
    x = np.linspace(
            indices[sel][0], indices[sel][-1],
            int((indices[sel][-1] -indices[sel][0]) *100))

    # iterate through polynomial orders:
    for poly_order in range(3, len(sel)-1):
        # fit and evaluate polynomial:
        poly_coef = np.polyfit(indices[sel], chisqs_obs[sel], poly_order)
        poly['new'] = np.polyval(poly_coef, x)
        del poly_coef
        best_index['new'] = x[np.argmin(poly['new'])]
        min_chisq['new'] = np.min(poly['new'])

        # break once convergence reached (at order of 0.01):
        if round(best_index['former'], 2)==round(best_index['new'], 2):
            poly_order = poly_order -1
            best_index = best_index['former']
            min_chisq = min_chisq['former']
            poly = poly['former']
            break
        # otherwise store current results:
        else:
            best_index['former'] = best_index['new']
            min_chisq['former'] = min_chisq['new']
            poly['former'] = poly['new']
    else:
        best_index = best_index['former']
        min_chisq = min_chisq['former']
        poly = poly['former']

    print('done.')

    #---plot chi squares and polynomial fit------------------------------------
    print('Create figure: chi square fit.. ', end='')
    sys.stdout.flush()

    fig, axes = plt.subplots(1)
    fig.suptitle('PSD results:\nChi square minimization', fontsize=16.)
    axes.set_xlabel('spectral index $\\beta$')
    axes.set_ylabel('$\\chi^2_\\mathrm{obs}$')
    i = np.where(chisqs_obs<10.**sel_order*np.min(chisqs_obs))[0]
    axes.plot(
            indices[i], chisqs_obs[i], marker='o', linestyle='None',
            label='measured')
    axes.plot(
            x, poly, color='g', label=f'{poly_order}. order poly. fit')
    axes.axvline(best_index, color='g', linestyle=':')
    axes.axhline(min_chisq, color='g', linestyle=':')
    axlim = axes.axis()
    axes.text(
            axlim[0]+0.01*(axlim[1] -axlim[0]), min_chisq,
            f'$\\chi^2_\\mathrm{{min}}={min_chisq:.2f}$',
            verticalalignment='bottom', fontsize=16.)
    axes.text(
            best_index+0.01*(axlim[1] -axlim[0]),
            axlim[2]+0.01*(axlim[3] -axlim[2]),
            f'$\\beta_\\mathrm{{opt}}={best_index:.2f}$',
            verticalalignment='bottom', fontsize=16.)
    axes.legend(loc='best')
    plt.savefig(
            f'{output_dir}{output_suffix}4_chisq.png', dpi=100)
    plt.clf()
    plt.close()
    del fig, axes, i, sel_order, poly_order, x, poly, axlim
    print('done.')

    #---goodness-of-fit: p-value-----------------------------------------------
    print('Goodness-of-fit: determine p-value.. ', end='')
    sys.stdout.flush()

    sel = np.argmin(np.absolute(indices-best_index))
    bestmodel_psd = simpsd_avg[sel]
    bestmodel_rms = simpsd_rms[sel]
    del simpsd_avg, simpsd_rms

    bestmodel_index = indices[sel]
    sel = np.where(simpar.cols.index[:] == bestmodel_index)[0]
    simpsd_sel = simpsd[:][sel]
    shape = simpsd_sel.shape

    # test model simulations against best model:
    chisqs_sim = np.nansum(np.power(simpsd_sel \
            -np.tile(bestmodel_psd, shape[0]).reshape(shape), 2) \
            /np.tile(bestmodel_rms, shape[0]).reshape(shape), axis=1)

    # calculate p-value:
    pvalue = len(np.where(chisqs_sim > min_chisq)[0]) *1. /shape[0]

    print('done.')

    #---plot sim chi square distribution---------------------------------------
    print('Create figure: chi square distribution.. ', end='')
    sys.stdout.flush()

    fig, axes = plt.subplots(1)
    fig.suptitle(
            'PSD results:\n$\\chi^2$ distribution at $\\beta={0:.2f}$ ' \
            '($\\beta_\\mathrm{{opt}}={1:.2f}$)'.format(
                    bestmodel_index, best_index),
            fontsize=16.)
    axes.set_xlabel('$\\chi^2$')
    axes.set_ylabel('rel. count')
    axes.hist(chisqs_sim, bins=20, density=True,
              alpha=0.5,
              label='sim')
    axes.axvline(min_chisq,
                 color='r', lw=2.,
                 label='obs')
    axes.text(
            0.84, 0.96, '$N={0}$\n$P(\\chi^2_\\mathrm{{sim}}>\\chi^2' \
            '_\\mathrm{{obs}})={1:.1f} \\%$'.format(shape[0], pvalue*100.),
            horizontalalignment='right', verticalalignment='top',
            fontsize=16., transform=axes.transAxes)
    axes.legend(loc='best')
    plt.savefig(
            f'{output_dir}{output_suffix}5_p-value.png', dpi=100)
    plt.clf()
    plt.close()
    del fig, axes
    print('done.')

    #---confidence interval----------------------------------------------------
    print('Determine confidence interval for best spectral index.. ', end='')
    sys.stdout.flush()

    # iterate though spectral indices:
    pvalues = np.zeros(nind)
    for i, index in enumerate(indices):
        # select simulations:
        sel = np.where(simpar.cols.index[:] == index)[0]
        simpsd_sel = simpsd[:][sel]
        shape = simpsd_sel.shape

        # test model simulations against best model:
        chisqs_sim = np.nansum(np.power(simpsd_sel \
                -np.tile(bestmodel_psd, shape[0]).reshape(shape), 2) \
                /np.tile(bestmodel_rms, shape[0]).reshape(shape), axis=1)

        # calculate p-value:
        pvalues[i] = len(np.where(chisqs_sim > min_chisq)[0]) *1. / shape[0]
        del chisqs_sim

    # cubic spline fit to p-values over spectral indices after smoothing:
    if smooth_confint:
        pvalues_smoothed = smooth(pvalues, smooth_confint)
        splines = splrep(indices, pvalues_smoothed, s=0)
    # cubic spline fit to p-values over spectral indices without smoothing:
    else:
        splines = splrep(indices, pvalues, s=0)
    # evaluate fit with spectral index accuracy at order 0.001:
    x = np.linspace(indices[0], indices[-1],
                    int((indices[-1] -indices[0]) *1000))
    y = splev(x, splines, der=0)
    del splines

    # lower confidence limit:
    i = np.where(y<1.-sign_level)[0]
    if len(i)==0:
        confint_lo = np.nan
    elif i[0]==0 or x[i[0]]>best_index:
        confint_lo = np.nan
    else:
        confint_lo = (x[i[0]] +x[i[0]-1]) /2.

    # upper confidence limit:
    i = np.where(y<1.-sign_level)[0]
    if len(i)==0:
        confint_hi = np.nan
    elif i[-1]==len(x)-1 or x[i[-1]]<best_index:
        confint_hi = np.nan
    else:
        confint_hi = (x[i[-1]] +x[i[-1]+1]) /2.
    del i

    print('done.')

    #---plot chi squares and cubic spline fit----------------------------------
    print('Create figure: confidence interval.. ', end='')
    sys.stdout.flush()

    fig, axes = plt.subplots(1)
    fig.suptitle(
            'PSD results:\nConfidence interval for $\\beta_\\mathrm{opt}$ at '\
            f'significance level $\\alpha={sign_level*100.:.2f}\\,\\%$',
            fontsize=16.)
    axes.set_xlabel('spectral index $\\beta$')
    axes.set_ylabel('p-value')
    axes.plot(indices, pvalues,
              marker='o', linestyle='None',
              label='measured')
    axes.plot(x, y, color='g', label='cubic spline fit')
    axes.axvline(best_index, color='r', label='$\\beta_\\mathrm{opt}$')
    axes.axvline(confint_lo, linestyle=':', color='r')
    axes.axvline(confint_hi, linestyle=':', color='r',
                 label='Conf. int. limits')
    axes.legend(loc='best')
    plt.savefig(f'{output_dir}{output_suffix}6_confint.png', dpi=100)
    plt.clf()
    plt.close()
    del fig, axes, x, y
    print('done.')

    #---write out results------------------------------------------------------
    text = '--------------------------------------------------\n' \
           'Best index:           {0:2f}\n' \
           'Min. chisq:           {1:.2f}\n' \
           'p-value:              {2:.2f}\n' \
           'Confidence interval\nat {3:.2f} sign. level:  {4:.2f} - {5:.2f}\n'\
           '\n--------------------------------------------------\n'.format(
           best_index, min_chisq, pvalue, sign_level, confint_lo, confint_hi)
    write_to.write(text)
    del text

    write_to.write('Spectral   obs.         p-value\n')
    write_to.write('index      chi sq.\n')
    for i, index in enumerate(indices):
        write_to.write('{0:.2f}       {1:.2e}     {2:.2f}\n'.format(
                index, chisqs_obs[i], pvalues[i]))

    print(f'Results written to: {resfile}\n')

    #---close open files-------------------------------------------------------
    h5file.close()
    write_to.close()

#==============================================================================

def test_reliability(
        dbfile, output_dir='', output_suffix='dbfile', sign_level=0.05,
        smooth_confint=False):
    """Tests the reliability of the simulation result. Each simulation is used
    as data input for the simulation evaluation. Derived optimal spectral
    indices, false negatives and confidence intervals are tested.

    Parameters
    -----
    dbfile : string
        Directory and file name of the simulation data base.
    output_dir : string, default=''
        Directory where the results are saved.
    output_suffix : string, default='dbfile'
        A suffix that is used for all resulting files. If 'dbfile' (default)
        the file name of the data base is used as suffix.
    sign_level : float, default=0.05
        Significance level at which the confidence interval for the spectral
        index is determined.
    smooth_confint : int, default=False
        If set, the p-values used to calculate the confidence interval are
        smoothed before fitted with a cubic spline. The number given sets the
        number of data points averaged. The number needs to be uneven. For even
        numbers the number will be increased by one.
    """

    #---access data base-------------------------------------------------------
    print(f'Access data base: {dbfile}')

    h5file = tb.open_file(
            dbfile, mode='r', title='Power-law PSD simulation data base')

    frequencies = h5file.root.psds.frequencies[:]
    simpsd = h5file.root.psds.simpsd
    simpar = h5file.root.psds.simpar
    genpar = h5file.root.psds.genpar

    #isf = np.where(np.isfinite(datapsd))[0]
    nfreq = len(frequencies)
    indices = np.unique(simpar.cols.index[:])
    nind = len(indices)
    nsim = len(simpar.cols.index)

    #---create output directory and file---------------------------------------
    # create directory if necessary:
    if len(output_dir)>0 and output_dir[-1]!='/':
        output_dir = f'{output_dir}/'
    create_file_dir(output_dir)

    # create text file:
    if output_suffix=='dbfile':
        try:
            i = dbfile[::-1].index('/')
        except:
            i = len(dbfile)
        i = len(dbfile) -i
        output_suffix = f'{dbfile[i:-3]}_'
        resfile = f'{output_dir}{output_suffix}reliability.txt'
    write_to = open(resfile, 'w')

    #---write general information to results file------------------------------
    resample_rate = genpar.cols.resample_rate[0]
    if resample_rate<-1.5:
        resample_rate = 'median'
    elif resample_rate<-0.5:
        resample_rate = 'mean'
    else:
        resample_rate = f'{resample_rate:.2f}'

    text = 'Data base file: {0}\n' \
           '--------------------------------------------------\n' \
           'Light curve:\n' \
           'Total time: {1:.2f}\n' \
           'Split at gaps: {2:.2f}\n' \
           'Full data resampled {3} times.\n' \
           'Split data resampled {4} times.\n' \
           'Resample rate: {5}\n' \
           '--------------------------------------------------\n' \
           'Light curve simulation:\n' \
           'Total time: {6:.2f} * {7}\n' \
           'Initial sampling rate: {8:.2f}\n' \
           '--------------------------------------------------\n' \
           'Power spectral density: {9}\n' \
           'Amplitude scaling: {10}\n' \
           'Bins per order: {11}\n\n'.format(
            dbfile, genpar.cols.time[0], genpar.cols.split[0],
            genpar.cols.resample_n[0], genpar.cols.resample_split_n[0],
            resample_rate, genpar.cols.time[0], genpar.cols.ntime[0],
            genpar.cols.sampling[0], genpar.cols.spectrum[0],
            genpar.cols.scaling[0], genpar.cols.bins_per_order[0])
    write_to.write(text)
    del resample_rate, text

    #---calculate average models-----------------------------------------------
    print('Calculate average models.. ', end='')
    sys.stdout.flush()

    # set storage arrays/lists:
    simpsd_avg = np.zeros((nind, nfreq))
    simpsd_rms = np.zeros((nind, nfreq))
    selections = []

    # iterate through spectral indices:
    for i, index in enumerate(indices):
        # select simulations:
        sel = np.where(simpar.cols.index[:] == index)[0]
        selections.append(sel)
        simpsd_sel = simpsd[:][sel]
        shape = simpsd_sel.shape

        # calculate average simulation:
        simpsd_avg[i] = np.mean(simpsd_sel, axis=0)
        # calculate RMS spread for all frequencies (omit the root):
        simpsd_rms[i] = np.mean(np.power(simpsd_sel \
                -np.tile(simpsd_avg[i], shape[0]).reshape(shape), 2), axis=0)

    del sel, shape

    print('done.')

    #---calculate and minimize chi squares-------------------------------------
    # set storage arrays:
    chisqs_obs = np.zeros((nsim, nind))
    min_chisqs = np.zeros(nsim)
    best_indices = np.zeros(nsim)
    best_models = np.zeros(nsim, dtype=int)

    # iterate through simulations:
    for i, psd in enumerate(simpsd[:]):
        sys.stdout.write(
                f'\rCalculate and minimize chi squares.. {i*100./nsim:.0f} %')
        sys.stdout.flush()

        #---calculate chi sqare------------------------------------------------
        chisqs_obs[i,:] = np.nansum(np.power(
                np.tile(psd, nind).reshape((nind, nfreq)) -simpsd_avg, 2) \
                /simpsd_rms, axis=1)

        #---minimize chi square------------------------------------------------
        # select spectral indices with small chi square:
        for sel_order in range(1, 10):
            sel = np.where(
                    chisqs_obs[i,:]<=10.**sel_order*np.min(chisqs_obs[i,:]))[0]
            if len(sel)>4:
                break

        # set two arbitrary starting numbers:
        best_index = {'former': 0.}
        min_chisq = {'former': 0.}

        # create x-values for polynomial evaluation:
        x = np.linspace(indices[sel][0], indices[sel][-1],
                        int((indices[sel][-1] -indices[sel][0]) *100))

        # iterate through polynomial orders:
        for poly_order in range(3, len(sel)-1):
            # fit and evaluate polynomial:
            poly_coef = np.polyfit(indices[sel], chisqs_obs[i,:][sel],
                                   poly_order)
            poly = np.polyval(poly_coef, x)
            del poly_coef
            best_index['new'] = x[np.argmin(poly)]
            min_chisq['new'] = np.min(poly)

            # break once convergence reached (at order of 0.01):
            if round(best_index['former'], 2)==round(best_index['new'], 2):
                best_indices[i] = best_index['former']
                min_chisqs[i] = min_chisq['former']
                best_models[i] = np.argmin(np.absolute(indices
                                                       -best_indices[i]))
                break
            # otherwise store current results:
            else:
                best_index['former'] = best_index['new']
                min_chisq['former'] = min_chisq['new']
        else:
            best_indices[i] = best_index['former']
            min_chisqs[i] = min_chisq['former']

    del x, poly, poly_order, sel_order, best_index, min_chisq

    print('\rCalculate and minimize chi squares.. done.')

    #---calculate p-values and confidence intervals----------------------------
    # set storage arrays:
    rejection = np.zeros(nsim, dtype=bool)
    confidence = np.zeros(nsim)

    # set x values to evaluate cubic splines fit of p-values:
    x = np.linspace(indices[0], indices[-1],
                    int((indices[-1] -indices[0]) *1000))

    # iterate through simulations:
    for i, chisq in enumerate(min_chisqs):
        sys.stdout.write(
                f'\rp-values and confidence intervals.. {i*100./nsim:.0f} %')
        sys.stdout.flush()

        #---calculate p-values-------------------------------------------------
        # set storage arrays:
        pvalues = np.zeros(nind)

        # iterate through models:
        for j, sel in enumerate(selections):
            N = len(np.where(chisqs_obs[:,best_models[i]][sel]>=chisq)[0])
            pvalues[j] = float(N) /len(sel)

        #---check model rejection, i.e. false negatives------------------------
        rejection[i] = pvalues[best_models[i]]<=sign_level

        #---calculate confidence interval--------------------------------------
        # cubic spline fit to p-values over spectral indices after smoothing:
        if smooth_confint:
            pvalues_smoothed = smooth(pvalues, smooth_confint)
            splines = splrep(indices, pvalues_smoothed, s=0)
        # cubic spline fit to p-values over spectral indices without smoothing:
        else:
            splines = splrep(indices, pvalues, s=0)
        # evaluate fit with spectral index accuracy at order 0.001:
        y = splev(x, splines, der=0)
        del splines

        sel = np.where(y<1.-sign_level)[0]
        # confidence interval not defined at this significance level:
        if len(sel)==0:
            confidence[i] = np.nan

        # confidence interval defined:
        else:
            # lower confidence limit:
            if sel[0]==0:
                confint_lo = np.min(indices)
            else:
                confint_lo = (x[sel[0]] +x[sel[0]-1]) /2.

            # upper confidence limit:
            if sel[-1]==len(x)-1:
                confint_hi = np.max(indices)
            else:
                confint_hi = (x[sel[-1]] +x[sel[-1]+1]) /2.

            # check if true value is in confidence interval:
            if simpar.cols.index[i]<confint_lo \
            or simpar.cols.index[i]>confint_hi:
                confidence[i] = 0.
            else:
                confidence[i] = 1.
        del sel

    # calculate rates of false negatives per spectral index:
    rate_falseneg = np.zeros(nind)
    for i, sel in enumerate(selections):
        rate_falseneg[i] = np.sum(rejection[sel]) / float(len(sel))

    # calculate rates of in/out of confidence interval per spectral index:
    confint_in = np.zeros(nind)
    confint_out = np.zeros(nind)
    confint_unk = np.zeros(nind)
    for i, sel in enumerate(selections):
        norm = float(len(sel))
        confint_in[i] = len(np.where(confidence[sel]==1)[0]) / norm
        confint_out[i] = len(np.where(confidence[sel]==0)[0]) / norm
        confint_unk[i] = len(np.where(np.isnan(confidence[sel]))[0]) / norm
    del norm

    print('\rp-values and confidence intervals.. done.')

    #---plot: deviation from intrinsic value-----------------------------------
    # select only a few spectral indices to plot:
    max_plots = 40
    nplots = nind if nind<=max_plots else max_plots
    plot = np.floor(np.linspace(0, nind, nplots)).astype(int)
    subselections = [selections[i] for i in range(nind) if i in plot]

    print('Create figure: accuracy.. ', end='')
    sys.stdout.flush()

    accuracy = simpar.cols.index[:] -best_indices

    fig = plt.figure(figsize=(6., 3.*nplots))
    grid = gs.GridSpec(nplots, 1, hspace=0.1)
    axes = []

    for i, sel in enumerate(subselections):
        sys.stdout.write(
                f'\rCreate figure: accuracy.. {i*100./nplots:.0f} %')
        sys.stdout.flush()

        axes.append(plt.subplot(grid[i]))
        # plot histogram:
        lolim = int(floor(np.min(accuracy)))
        uplim = int(ceil(np.max(accuracy)))
        ks = kstest(accuracy[sel]-np.mean(accuracy[sel]), 'norm')[1]
        axes[i].hist(
                accuracy[sel],
                range=(lolim-0.05, uplim+0.05), bins=10*(uplim-lolim)+1,
                color=cmap(i*1./nplots), alpha=0.5,
                label=f'$\\beta={indices[plot[i]]:.2f}$')
        axes[i].text(
                0.97, 0.7, 'median:  {0:.2f}\n  mean:  {1:.2f}\n   std.:  ' \
                '{2:.2f}\n     KS: {3:.2f}'.format(
                    np.median(accuracy[sel]), np.mean(accuracy[sel]),
                    np.std(accuracy[sel]), ks),
                fontsize=16.,
                horizontalalignment='right', verticalalignment='top',
                transform=axes[i].transAxes)
        axes[i].legend(loc='upper right')
    print('\rCreate figure: accuracy.. almost done.. ', end='')
    sys.stdout.flush()

    axes[0].set_title(
            'PSD reliability:\nspectral index accuracy', fontsize=16.)
    axes[-1].set_xlabel('$\\beta_\\mathrm{intr}-\\beta_\\mathrm{obs}$')

    plt.savefig(
            f'{output_dir}{output_suffix}7_accuracy.png', dpi=100,
            bbox_inches='tight')
    plt.clf()
    plt.close()
    del fig, axes, lolim, uplim, ks
    print('\rCreate figure: accuracy.. done.        ')

    #---plot: rate of false negatives------------------------------------------
    print('Create figure: false negatives.. ', end='')
    sys.stdout.flush()

    fig, axes = plt.subplots(1)
    fig.suptitle('PSD reliability:\nfalse negative rates')
    axes.set_xlabel('$\\beta_\\mathrm{intr}$')
    axes.set_ylabel('rate of false negatives $[\%]$')
    ymax = ceil(np.max(rate_falseneg*100))
    axes.set_ylim(0, ymax)
    axes.set_xlim(np.min(indices)-0.1, np.max(indices)+0.1)
    for ind, fnr in zip(indices, rate_falseneg):
        axes.axvline(ind, ymax=fnr*100./ymax, color='k', linewidth=1.)
        axes.text(
                ind, fnr*100, '%.1f %%' % (fnr*100),
                horizontalalignment='center', verticalalignment='bottom',
                fontsize=6.)
    #plt.plot(indices, rate_falseneg*100, linestyle='None', marker='o')
    plt.savefig(
            f'{output_dir}{output_suffix}8_falseneg.png', dpi=100,
            bbox_inches='tight')
    plt.clf()
    plt.close()
    del fig, axes
    print('done.')

    #---plot: confidence intervals---------------------------------------------
    print('Create figure: confidence intervals.. ', end='')
    sys.stdout.flush()

    fig, axes = plt.subplots(1)
    fig.suptitle(
            'PSD reliability:\n$\\beta_\\mathrm{intr}$ in confidence ' \
            f'interval at significance level $\\alpha={sign_level:.2f}$')
    axes.set_xlabel('$\\beta_\\mathrm{intr}$')
    axes.set_ylabel('rates')
    axes.set_xlim(np.min(indices)-0.1, np.max(indices)+0.1)
    axes.bar(indices, confint_in,
             width=0.02, linewidth=0, color=cmap(0.7), label='in')
    axes.bar(indices, confint_unk, bottom=confint_in,
             width=0.02, linewidth=0, color=cmap(0.5), label='undef.')
    axes.bar(indices, confint_out, bottom=confint_in+confint_unk,
             width=0.02, linewidth=0, color=cmap(0.1), label='out')
    axes.axhline(1.-sign_level, linestyle=':', color='k')
    axes.legend(loc='lower right')
    plt.savefig('%s%s9_confidence.png' % (output_dir, output_suffix),
                dpi=100, bbox_inches='tight')
    plt.clf()
    plt.close()
    del fig, axes
    print('done.')

    #---write out results------------------------------------------------------
    text = '--------------------------------------------------\n' \
           'Spectral    _____________accuracy______________    false neg.    '\
           'confidence interval\n' \
           'index       dev. med.    dev. mean    dev. std.    rate          '\
           'in      out     unkn.\n'
    write_to.write(text)

    for i, index in enumerate(indices):
        acc_med = np.median(accuracy[selections[i]])
        acc_mean = np.mean(accuracy[selections[i]])
        acc_std = np.std(accuracy[selections[i]])
        text = '{0:.2f}        {1:+.2f}        {2:+.2f}        {3:.2f} ' \
               '         {4:.2f}          {5:.2f}    {6:.2f}    {7:.2f}\n' \
               ''.format(
                   index, acc_med, acc_mean, acc_std, rate_falseneg[i],
                   confint_in[i], confint_out[i], confint_unk[i])
        write_to.write(text)
    del text

    print(f'Results written to: {resfile}\n')

    #---close open files-------------------------------------------------------
    h5file.close()
    write_to.close()

#==============================================================================

def dbinfo(dbfile, details=None):
    """Prints information about the simulations stored in a PSD data base file.
    Lists the tested spectral indices and the number of simulations run for
    each spectral index.

    Parameters
    -----
    dbfile : string
        Directory and file name of the simulation data base.
    details : string, default=None
        If 'all' spectral indices and according number of simulations are
        printed, otherwise not.
    """

    #---access data base-------------------------------------------------------
    print(f'Data base file: {dbfile}')

    h5file = tb.open_file(
            dbfile, mode = 'r', title = 'Power-law PSD simulation data base')

    simpar = h5file.root.psds.simpar
    indices = np.sort(np.unique(simpar.cols.index[:]))

    print(f'{len(indices)} spectral indices tested.')

    if details=='all':
        print('Spectral    Number of')
        print('index:      simulations:')

        for index in indices:
            n_sim = len(np.where(simpar.cols.index[:]==index)[0])
            print(f'{index:.2f}        {n_sim}')

    print(f'Total number of simulations: {len(simpar.cols.index)}.\n')

    h5file.close()

#==============================================================================
