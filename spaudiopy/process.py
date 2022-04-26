# -*- coding: utf-8 -*-
"""Collection of audio processing tools.

.. plot::
    :context: reset

    import numpy as np
    import matplotlib.pyplot as plt
    plt.rcParams['axes.grid'] = True

    import spaudiopy as spa

**Memory cached functions**

.. autofunction:: spaudiopy.process.resample_hrirs(hrir_l, hrir_r, fs_hrir, fs_target, jobs_count=None)

"""

from itertools import repeat

import numpy as np
import resampy
import pickle
from scipy import signal
from joblib import Memory
import multiprocessing

from . import utils
from . import sph
from . import sig

# Prepare Caching
cachedir = './.spa_cache_dir'
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
    jobs_count : int or None, optional
        Number of parallel jobs, 'None' employs 'cpu_count'.

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
    hrir_l_resampled = np.zeros([hrir_l.shape[0],
                                 int(hrir_l.shape[1] * fs_target / fs_hrir)])
    hrir_r_resampled = np.zeros_like(hrir_l_resampled)

    if jobs_count == 1:
        hrir_l_resampled = resampy.resample(hrir_l, fs_hrir, fs_target, axis=1)
        hrir_r_resampled = resampy.resample(hrir_r, fs_hrir, fs_target, axis=1)
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


def resample_signal(s_time, fs_current, fs_target, axis=-1):
    """
    Resample time signal.

    Parameters
    ----------
    s_time : numpy.ndarray
        Time signal, or signals stacked.
    fs_current : int
    fs_target : int
    axis : int, optional
        Axis along which to resample. The default is -1.

    Returns
    -------
    single_spec_resamp : numpy.ndarray.

    """
    s_time = np.atleast_2d(s_time)
    s_time_resamp = resampy.resample(s_time, fs_current, fs_target, axis=axis)
    return np.squeeze(s_time_resamp)


def resample_spectrum(single_spec, fs_current, fs_target, axis=-1):
    """
    Resample single sided spectrum, as e.g. from np.fft.rfft().

    Parameters
    ----------
    single_spec : numpy.ndarray
        Single sided spectrum, or spectra stacked.
    fs_current : int
    fs_target : int
    axis : int, optional
        Axis along which to resample. The default is -1.

    Returns
    -------
    single_spec_resamp : numpy.ndarray.

    """
    single_spec = np.atleast_2d(single_spec)
    s_time = np.fft.irfft(single_spec, axis=axis)
    s_time_resamp = resampy.resample(s_time, fs_current, fs_target, axis=axis)
    single_spec_resamp = np.fft.rfft(s_time_resamp, axis=axis)
    return np.squeeze(single_spec_resamp)


def ilds_from_hrirs(hrirs, f_cut=1000, INDB=True):
    """Calculate ILDs from HRIRs by high-passed broad-band RMS.

    Parameters
    ----------
    hrirs : sig.HRIRs
    f_cut : float, optional
        Low-pass cutoff frequency. The default is 1000.

    Returns
    -------
    ild : array_like
        ILD per grid point, positive value indicates left ear louder.
    INDB : bool, optional
        ILD in dB RMS ratio, otherwise as RMS difference. The default is TRUE.
    """
    assert(isinstance(hrirs, sig.HRIRs))
    fs = hrirs.fs
    sos = signal.butter(4, f_cut, 'high', fs=fs, output='sos')

    hrirs_l_f = signal.sosfiltfilt(sos, hrirs.left, axis=-1)
    hrirs_r_f = signal.sosfiltfilt(sos, hrirs.right, axis=-1)

    if INDB:
        rms_diff = utils.db(utils.rms(hrirs_l_f, axis=-1) /
                            utils.rms(hrirs_r_f, axis=-1))
    else:
        rms_diff = utils.rms(hrirs_l_f, axis=-1) - \
                   utils.rms(hrirs_r_f, axis=-1)

    return rms_diff


def itds_from_hrirs(hrirs, f_cut=1000, upsample=4):
    """Calculate ITDs from HRIRs by upsampled and filtered cross-correlation.

    Parameters
    ----------
    hrirs : sig.HRIRs
    f_cut : float, optional
        Low-pass cutoff frequency. The default is 1000.
    upsample : int, optional
        Upsampling factor. The default is 8.

    Returns
    -------
    itd : array_like
        ITD in seconds per grid point, positive value indicates left ear first.

    """
    assert(isinstance(hrirs, sig.HRIRs))
    fs = hrirs.fs
    sos = signal.butter(4, f_cut, 'low', fs=fs, output='sos')

    hrirs_l_us, hrirs_r_us, _ = resample_hrirs(hrirs.left, hrirs.right,
                                               hrirs.fs, upsample*hrirs.fs)

    hrirs_l_us = signal.sosfiltfilt(sos, hrirs_l_us, axis=-1)
    hrirs_r_us = signal.sosfiltfilt(sos, hrirs_r_us, axis=-1)

    maxidx = np.zeros(hrirs.grid_points)
    for idx, hrirs_dir in enumerate(zip(hrirs_l_us, hrirs_r_us)):
        maxidx[idx] = np.argmax(np.correlate(hrirs_dir[0], hrirs_dir[1],
                                             mode='same'))
    maxidx -= hrirs_l_us.shape[1]//2
    # alternative
    # maxidx = np.argmax(hrirs_l_us, axis=-1) - np.argmax(hrirs_r_us, axis=-1)
    itd = -maxidx / (upsample*fs)
    return itd


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
        Ambi_B.W = signal.lfilter(b0_d, a0_d, Ambi_B.W)
        Ambi_B.X = signal.lfilter(b1_d, a1_d, Ambi_B.X)
        Ambi_B.Y = signal.lfilter(b1_d, a1_d, Ambi_B.Y)
        Ambi_B.Z = signal.lfilter(b1_d, a1_d, Ambi_B.Z)
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
    mode : 'energy' or 'amplitude'
        'energy' produces -3dB at crossover, 'amplitude' -6dB.
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
    This filterbank is originally designed such that the sum of gains squared
    sums to unity. The alternative 'amplitude' mode ensures that the gains sum
    directly to unity.

    References
    ----------
    Antoni, J. (2010). Orthogonal-like fractional-octave-band filters.
    The Journal of the Acoustical Society of America, 127(2), 884–895.

    Examples
    --------
    .. plot::
        :context: close-figs

        fs = 44100
        N = 2**16
        gs, ff = spa.process.frac_octave_filterbank(n=1, N_out=N, fs=fs,
                                                    f_low=100, f_high=8000)
        f = np.linspace(0, fs//2, N)
        fig, ax = plt.subplots(2, 1, constrained_layout=True)
        ax[0].semilogx(f, gs.T)
        ax[0].set_title('Band gains')
        ax[1].semilogx(f, np.sum(np.abs(gs)**2, axis=0))
        ax[1].set_title(r'$\sum |g| ^ 2$')
        for a_idx in ax:
            a_idx.grid(True)
            a_idx.set_xlim([20, fs//2])
            a_idx.set_xlabel('f in Hz')
            a_idx.set_ylabel('Amplitude')

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

        # shift phi to [0, 1]
        phi = 0.5 * (phi + 1)
        a = np.sin(np.pi / 2 * phi)
        b = np.cos(np.pi / 2 * phi)

        # Hi
        g[b_idx, k_i[b_idx] - P[b_idx]: k_i[b_idx] + P[b_idx] + 1] = b
        g[b_idx, k_i[b_idx] + P[b_idx]:] = 0.
        # Lo
        g[b_idx + 1, k_i[b_idx] - P[b_idx]: k_i[b_idx] + P[b_idx] + 1] = a
        g[b_idx + 1, : k_i[b_idx] - P[b_idx]] = 0.

    if mode in ['energy']:
        g = g
    elif mode in ['amplitude', 'pressure']:
        # This is not part of Antony (2010), see 'notes'
        g = g**2
    else:
        raise ValueError("Mode not implemented: " + mode)

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
    """Design half-sided Hann tapering window of order N (>=3)."""
    assert (N >= 3)
    w_full = signal.hann(2 * ((N + 1) // 2) + 1)
    # get half sided window
    w_taper = np.ones(N + 1)
    w_taper[-((N - 1) // 2):] = w_full[-((N + 1) // 2):-1]
    return w_taper


def gain_clipping(gain, threshold):
    """Limit gain factor by soft clipping function. Limits gain factor to +6dB
    beyond threshold point. (Pass values as factors/ratios, not dB!)

    Parameters
    ----------
    gain : array_like
    threshold : float

    Returns
    -------
    gain_clipped : array_like

    Examples
    --------
    .. plot::
        :context: close-figs

        x = np.linspace(-10, 10, 1000)
        lim_threshold = 2.5
        y = spa.process.gain_clipping(x, lim_threshold)
        plt.figure()
        plt.plot(x, x, '--', label='In')
        plt.plot(x, y, label='Out')
        plt.legend()
        plt.xlabel('In')
        plt.ylabel('Out')
        plt.grid(True)

    """
    gain = gain / threshold  # offset by threshold
    gain[gain > 1] = 1 + np.tanh(gain[gain > 1] - 1)  # soft clipping to 2
    return gain * threshold


def pulsed_noise(t_noise, t_pause, fs, reps=10, t_fade=0.02, pink_noise=True,
                 normalize=True):
    """Pulsed noise train, pink or white.

    Parameters
    ----------
    t_noise : float
        t in s for pulse.
    t_pause : float
        t in s between pulses.
    fs : int
        Sampling frequency.
    reps : int, optional
        Repetitions (independent). The default is 10.
    t_fade : float, optional
        t in s for fade in and out. The default is 0.02.
    pink_noise : bool, optional
        Use 'pink' (1/f) noise. The default is True
    normalize : bool, optional
        Normalize output. The default is True.

    Returns
    -------
    s_out : array_like
        output signal.

    """
    s_out = []

    for _ in range(reps):
        s_noise = np.random.randn(int(fs*t_noise))

        if pink_noise:
            X = np.fft.rfft(s_noise)
            nbins = len(X)
            # divide by sqrt(n), power spectrum
            X_pink = X / np.sqrt(np.arange(nbins)+1)
            s_noise = np.fft.irfft(X_pink)

        s_pause = np.zeros(int(fs*t_noise))

        # fades
        mask_n = int(fs*t_fade)
        mask_in = np.sin(np.linspace(0, np.pi/2, mask_n))**2
        mask_out = np.cos(np.linspace(0, np.pi/2, mask_n))**2

        # apply
        s_noise[:mask_n] *= mask_in
        s_noise[-mask_n:] *= mask_out

        s_out = np.r_[s_out, s_noise, s_pause]

    if normalize:
        s_out /= np.max(abs(s_out))

    return s_out
