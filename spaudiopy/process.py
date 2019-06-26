# -*- coding: utf-8 -*-

from itertools import repeat

import numpy as np
import resampy
import pickle
from scipy import signal as scysig
from joblib import Memory
import multiprocessing

from . import utils
from . import sph
from . import sig

# Prepare Caching
cachedir = './__cache_dir'
memory = Memory(cachedir)


@memory.cache
def resample_hrirs(hrir_l, hrir_r, fs_hrir, fs_target, jobs_count=None):
    """
    Resample HRIRs to new SamplingRate(t), using multiprocessing.

    Parameters
    ----------
    hrir_l : (g, h) numpy.ndarray
        h(t) for grid position g.
    hrir_r : (g, h) numpy.ndarray
        h(t) for grid position g.
    fs_hrir : int
        Current fs(t) of hrirs.
    fs_target : int
        Target fs(t) of hrirs.
    jobs_count : int, optional
        [CPU Cores], Number of Processes, switches implementation for n > 1.

    Returns
    -------
    hrir_l_resampled : (g, h_n) numpy.ndarray
        h_n(t) resampled for grid position g.
    hrir_r_resampled : (g, h_n) numpy.ndarray
        h_n(t) resampled for grid position g.
    fs_hrir : int
        New fs(t) of hrirs.
    """
    if jobs_count is None:
        jobs_count = multiprocessing.cpu_count()
    print('Resample HRTFs')
    hrir_l_resampled = np.zeros([hrir_l.shape[0],
                                 int(hrir_l.shape[1] * fs_target / fs_hrir)])
    hrir_r_resampled = np.zeros_like(hrir_l_resampled)

    if jobs_count == 1:
        for i, (l, r) in enumerate(zip(hrir_l, hrir_r)):
            hrir_l_resampled[i, :] = resampy.resample(l, fs_hrir, fs_target)
            hrir_r_resampled[i, :] = resampy.resample(r, fs_hrir, fs_target)
    elif jobs_count > 1:
        print("Using %i processes..." % jobs_count)
        with multiprocessing.Pool(processes=jobs_count) as pool:
            results = pool.starmap(resampy.resample,
                                   map(lambda x: (x, fs_hrir, fs_target),
                                       hrir_l))
            hrir_l_resampled = np.array(results)
            results = pool.starmap(resampy.resample,
                                   map(lambda x: (x, fs_hrir, fs_target),
                                       hrir_r))
            hrir_r_resampled = np.array(results)

    fs_hrir = fs_target
    return hrir_l_resampled, hrir_r_resampled, fs_hrir


def haversine_dist(azi1, colat1, azi2, colat2):
    """
    Calculate the great circle distance between two points on the sphere.

    Parameters
    ----------
    azi1 : (n,) array_like
    colat1 : (n,) array_like.
    azi2 : (n,) array_like
    colat2: (n,) array_like

    Returns
    -------
    c : (n,) array_like
        Haversine distance between pairs of points.
    """
    lat1 = np.pi / 2 - colat1
    lat2 = np.pi / 2 - colat2

    dlon = azi2 - azi1
    dlat = lat2 - lat1

    haversin_A = np.sin(dlat / 2) ** 2
    haversin_B = np.sin(dlon / 2) ** 2

    haversin_alpha = haversin_A + np.cos(lat1) * np.cos(lat2) * haversin_B

    c = 2 * np.arcsin(np.sqrt(haversin_alpha))
    return c


def match_loudness(sig_in, sig_target):
    """
    Match loundess of input to target, based on RMS and avoid clipping.

    Parameters
    ----------
    sig_in : (n, c) array_like
        Input(t) samples n, channel c.
    sig_target : (n, c) array_like
        Target(t) samples n, channel c.

    Returns
    -------
    sig_out : (n, c) array_like
        Output(t) samples n, channel c.
    """
    L_in = np.max(np.sqrt(np.mean(np.square(sig_in), axis=0)))
    L_target = np.max(np.sqrt(np.mean(np.square(sig_target), axis=0)))
    sig_out = sig_in * L_target / L_in
    peak = np.max(np.abs(sig_out))
    if peak > 1:
        sig_out = sig_out / peak
        print('Audio normalized')
    return sig_out


def _intensity_sample(i, W, X, Y, Z, win):
    buf = len(win)
    # global shared_array
    shared_array[int(i + buf // 2), :] = np.asarray(
        [np.trapz(win * W[i:i + buf] * X[i:i + buf]),
         np.trapz(win * W[i:i + buf] * Y[i:i + buf]),
         np.trapz(win * W[i:i + buf] * Z[i:i + buf])])


@memory.cache
def pseudo_intensity(Ambi_B, win_len=33, f_bp=None, smoothing_order=5,
                     jobs_count=None):
    """DOA prototyping.

    Parameters
    ----------
    Ambi_B : sig.AmbiBSignal
        Input signal.
    win_len : int optional
        Sliding window length.
    f_bp : tuple(f_lo, f_hi), optional
        Cutoff frequencies for bandpass, 'None' to disable.
    smoothing_order : int, optional
        Apply hanning(smoothing_order) smoothing to output.
    jobs_count : int, optional
        [CPU Cores], Number of Processes, switches implementation for n > 1.

    Returns
    -------
    I_azi, I_colat, I_r : array_like
        Pseudo intensity vector.
    """
    # WIP
    if jobs_count is None:
        jobs_count = multiprocessing.cpu_count()

    assert(win_len % 2)
    win = np.hanning(win_len)
    fs = Ambi_B.fs
    # Z_0 = 413.3
    # T_int = 1/fs * win_len
    # a = 1 / (np.sqrt(2) * T_int * Z_0)

    # get first order signals
    W = utils.asarray_1d(Ambi_B.W)
    X = utils.asarray_1d(Ambi_B.X)
    Y = utils.asarray_1d(Ambi_B.Y)
    Z = utils.asarray_1d(Ambi_B.Z)

    # Bandpass signals
    if f_bp is not None:
        f_lo = f_bp[0]
        f_hi = f_bp[1]
        b, a = scysig.butter(N=2, Wn=(f_lo / (fs / 2), f_hi / (fs / 2)),
                             btype='bandpass')
        W = scysig.filtfilt(b, a, W)
        X = scysig.filtfilt(b, a, X)
        Y = scysig.filtfilt(b, a, Y)
        Z = scysig.filtfilt(b, a, Z)

    # Initialize intensity vector
    I_vec = np.c_[np.zeros(len(Ambi_B)),
                  np.zeros(len(Ambi_B)), np.zeros(len(Ambi_B))]

    if jobs_count == 1:
        print('Using single job')
        # I = p*v for each sample
        for i in range(len(Ambi_B) - win_len):
            I_vec[int(i + win_len // 2), :] = np.asarray(
                [np.trapz(win * W[i:i + win_len] * X[i:i + win_len]),
                 np.trapz(win * W[i:i + win_len] * Y[i:i + win_len]),
                 np.trapz(win * W[i:i + win_len] * Z[i:i + win_len])])
    else:
        # preparation
        shared_array_shape = np.shape(I_vec)
        _arr_base = _create_shared_array(shared_array_shape)
        _arg_itr = zip(range(len(Ambi_B) - win_len),
                       repeat(W), repeat(X), repeat(Y), repeat(Z),
                       repeat(win))
        # execute
        with multiprocessing.Pool(processes=jobs_count,
                                  initializer=_init_shared_array,
                                  initargs=(_arr_base,
                                            shared_array_shape,)) as pool:
            pool.starmap(_intensity_sample, _arg_itr)
        # reshape
        I_vec = np.frombuffer(_arr_base.get_obj()).reshape(
            shared_array_shape)

    if smoothing_order > 0:
        assert(smoothing_order % 2)
        I_vec = np.apply_along_axis(scysig.convolve, 0, I_vec,
                                    np.hanning(smoothing_order), 'same')
    # Normalize
    # I_norm = np.linalg.norm(I_vec, axis=1)
    # I_norm[I_norm < 10e8] = 10e8  # handle Zeros
    # I_vec = I_vec * (1 / I_norm[:, np.newaxis])

    return utils.cart2sph(I_vec[:, 0], I_vec[:, 1], I_vec[:, 2])


def ambeo_a2b(Ambi_A, filter_coeffs=None):
    """Convert A 'MultiSignal' (type I: FLU, FRD, BLD, BRU) to B AmbiBSignal.

    Parameters
    ----------
    Ambi_A : sig.MultiSignal
        Input signal.
    filter_coeffs : string
        Picklable file that contains b0_d, a0_d, b1_d, a1_d.

    Returns
    -------
    Ambi_B : sig.AmbiBSignal
        B-format output signal.
    """
    _B = sph.soundfield_to_b(Ambi_A.get_signals())
    Ambi_B = sig.AmbiBSignal([_B[0, :], _B[1, :], _B[2, :], _B[3, :]],
                             fs=Ambi_A.fs)
    if filter_coeffs is not None:
        b0_d, a0_d, b1_d, a1_d = pickle.load(open(filter_coeffs, "rb"))
        Ambi_B.W = scysig.lfilter(b0_d, a0_d, Ambi_B.W)
        Ambi_B.X = scysig.lfilter(b1_d, a1_d, Ambi_B.X)
        Ambi_B.Y = scysig.lfilter(b1_d, a1_d, Ambi_B.Y)
        Ambi_B.Z = scysig.lfilter(b1_d, a1_d, Ambi_B.Z)
    return Ambi_B


def b_to_stereo(Ambi_B):
    """Downmix B format first order Ambisonics to Stereo.

    Parameters
    ----------
    Ambi_B : sig.AmbiBSignal
        B-format output signal.

    Returns
    -------
    L, R : array_like
    """
    L = Ambi_B.W + (Ambi_B.X + Ambi_B.Y) / (np.sqrt(2))
    R = Ambi_B.W + (Ambi_B.X - Ambi_B.Y) / (np.sqrt(2))
    return L, R


def lagrange_delay(N, delay):
    """
    Return fractional delay filter using lagrange interpolation.
    For best results, delay should be near N/2 +/- 1.

    Parameters
    ----------
    N : int
        Filter order.
    delay : float
        Delay in samples.

    Returns
    -------
    h : (N+1,) array_like
        FIR Filter.
    """
    n = np.arange(N + 1)
    h = np.ones(N + 1)
    for k in range(N + 1):
        index = np.where(n != k)
        h[index] = h[index] * (delay - k) / (n[index] - k)
    return h


def frac_octave_filterbank(n, N_out, fs, f_low, f_high=None, mode='energy',
                           overlap=0.5, l=3):
    """ Fractional octave band filterbank.
    Design of digital fractional-octave-band filters with energy conservation
    and perfect reconstruction.

    Parameters
    ----------
    n : int
        Octave fraction, e.g. n=3 third-octave bands.
    N_out : int
        Number of non-negative frequency bins [0, fs/2].
    fs : int
        Sampling frequency in Hz.
    f_low : int
        Center frequency of first full band in Hz.
    f_high : int
        Cutoff frequency in Hz, above which no further bands are generated.
    mode : 'energy' or 'pressure'
        'energy' produces -3dB at crossover, 'pressure' -6dB.
    overlap : float
        Band overlap, should be between [0, 0.5].
    l : int
        Band transition slope, implemented as recursion order `l`.

    Returns
    -------
    g : (b, N) np.ndarray
        Band gains for non-negative frequency bins.
    ff : (b, 3) np.ndarray
        Filter frequencies as [f_lo, f_c, f_hi].

    Notes
    -----
    Antoni, J. (2010). Orthogonal-like fractional-octave-band filters.
    The Journal of the Acoustical Society of America, 127(2), 884–895.

    Examples
    --------
    >>> fs = 44100
    >>> N = 2**16
    >>> gs, ff = frac_octave_filterbank(n=1, N_out=N, fs=fs, f_low=100, f_high=8000)
    >>> f = np.linspace(0, fs//2, N)
    >>> fig, ax = plt.subplots(2, 1)
    >>> ax[0].semilogx(f, gs.T)
    >>> ax[0].set_title('Band gains')
    >>> ax[1].semilogx(f, np.sum(np.abs(gs)**2, axis=0))
    >>> ax[1].set_title('$\sum |g| ^ 2$')
    >>> for a_i in ax:
    >>>     a_i.grid(True)
    >>>     a_i.set_xlim([20, fs//2])
    >>>     a_i.set_xlabel('f in Hz')
    >>>     a_i.set_ylabel('Amplitude')
    """
    # fft bins
    N = (N_out - 1) * 2
    # frequency axis
    freq = np.fft.rfftfreq(N, d=1. / fs)
    f_alias = fs // 2
    if f_high is None:
        f_high = f_alias
    else:
        f_high = np.min([f_high, f_alias])
    assert (overlap <= 0.5)
    # center frequencies
    f_c = []
    # first is f_low
    f_c.append(f_low)
    # check next cutoff frequency
    while (f_c[-1] * (2 ** (1 / (2 * n)))) < f_high:
        f_c.append(2 ** (1 / n) * f_c[-1])
    f_c = np.array(f_c)

    # cut-off freqs
    f_lo = f_c / (2 ** (1 / (2 * n)))
    f_hi = f_c * (2 ** (1 / (2 * n)))

    # convert
    w_s = 2 * np.pi * fs
    # w_m
    w_c = 2 * np.pi * f_c
    # w_1
    w_lo = 2 * np.pi * f_lo
    # w_1+1
    w_hi = 2 * np.pi * f_hi

    # DFT line that corresponds to the lower bandedge frequency
    k_i = np.floor(N * w_lo / w_s).astype(int)
    # DFT bins in the frequency band
    N_i = np.diff(k_i)
    # band overlap (twice)
    P = np.round(overlap * (N * (w_c - w_lo) / w_s)).astype(int)

    g = np.ones([len(f_c) + 1, len(freq)])
    for b_idx in range(len(f_c)):
        p = np.arange(-P[b_idx], P[b_idx] + 1)

        # phi within [-1, 1]
        phi = (p / P[b_idx])
        phi[np.isnan(phi)] = 1.
        # recursion eq. 20
        for l_i in range(l):
            phi = np.sin(np.pi / 2 * phi)

        if mode == 'energy':
            # shift phi to [0, 1]
            phi = 0.5 * (phi + 1)
            a = np.sin(np.pi / 2 * phi)
            b = np.cos(np.pi / 2 * phi)

        if mode == 'pressure':
            # This is not part of Antony (2010)
            a = np.sin(np.pi / 2 * phi)
            a = 0.5 * (a + 1)
            b = 1 - a

        # Hi
        g[b_idx, k_i[b_idx] - P[b_idx]: k_i[b_idx] + P[b_idx] + 1] = b
        g[b_idx, k_i[b_idx] + P[b_idx]:] = 0.
        # Lo
        g[b_idx + 1, k_i[b_idx] - P[b_idx]: k_i[b_idx] + P[b_idx] + 1] = a
        g[b_idx + 1, : k_i[b_idx] - P[b_idx]] = 0.

    # Corresponding frequency limits
    ff = np.c_[f_lo, f_c, f_hi]
    # last band
    ff[-1, -1] = fs / 2
    ff[-1, 1] = np.sqrt(ff[-1, 0] * ff[-1, -1])
    # first band
    ff = np.vstack([np.array([0, np.sqrt(1 * ff[0, 0]), ff[0, 0]]), ff])
    return g, ff


def subband_levels(x, width, fs, power=False, axis=-1):
    """Computes the level/power in each subband of subband signals."""
    N = x.shape[1]

    if power is False:
        # normalization wrt bandwidth/sampling interval
        L = np.sqrt(1 / width * fs / 2 * np.sum(np.abs(x) ** 2, axis=axis))
    else:
        L = 1 / N * 1 / width * fs / 2 * np.sum(np.abs(x) ** 2, axis=axis)

    return L


def energy_decay(p):
    """Energy decay curve (EDC) in dB by Schroeder backwards integration.

    Parameters
    ----------
    p : array_like

    Returns
    -------
    rd : array_like
    """
    a = np.trapz(p**2)
    b = np.cumsum(p[::-1]**2)[::-1]
    return 10 * np.log10(b / a)


def half_sided_Hann(N):
    """Design half-sided Hann tapering window of order N."""
    assert (N >= 3)
    w_full = scysig.hann(2 * ((N + 1) // 2) + 1)
    # get half sided window
    w_taper = np.ones(N + 1)
    w_taper[-((N - 1) // 2):] = w_full[-((N + 1) // 2):-1]
    return w_taper


# Parallel worker stuff -->
def _create_shared_array(shared_array_shape):
    """Allocate ctypes array from shared memory with lock."""
    d_type = 'd'
    shared_array_base = multiprocessing.Array(d_type, shared_array_shape[0] *
                                              shared_array_shape[1])
    return shared_array_base


def _init_shared_array(shared_array_base, shared_array_shape):
    """Makes 'shared_array' available to child processes."""
    global shared_array
    shared_array = np.frombuffer(shared_array_base.get_obj())
    shared_array = shared_array.reshape(shared_array_shape)
# < --Parallel worker stuff
