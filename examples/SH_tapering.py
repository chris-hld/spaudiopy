#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Examples for:
# Hold, C., Gamper, H., Pulkki, V., Raghuvanshi, N., & Tashev, I. J. (2019).
# IMPROVING BINAURAL AMBISONICS DECODING BY SPHERICAL HARMONICS DOMAIN TAPERING
# AND COLORATION COMPENSATION.
# In ICASSP, IEEE International Conference on Acoustics, Speech and
# Signal Processing - Proceedings.

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin2
from spaudiopy import sph, utils, plots
from spaudiopy import process as pcs

# sampling rate (Hz)
fs = 48000
# speed of sound (m/s)
c = 343
# sphere radius (m)
r_0 = 8.75 / 100
# target spherical order N (>= 3)
N = 5


# %% Coloration compensation filter
def coloration_compensation(N, N_full, kr, w_taper=None):
    """Get coloration compensation G(kr)|N for diffuse field of order N."""
    h_comp = sph.pressure_on_sphere(N_full, kr) / \
             sph.pressure_on_sphere(N, kr, w_taper)
    # catch NaNs
    h_comp[np.isnan(h_comp)] = 1
    return h_comp


# evaluated frequencies, 1000 points
f = np.linspace(0, fs / 2, 1000)
k = (2 * np.pi * f) / c
kr = k * r_0

# get aliasing free N > kr
N_full = int(np.ceil(kr[-1]))

# %% Spatial dirac in SH domain
dirac_azi = np.deg2rad(90)
dirac_colat = np.deg2rad(90)

# cross section
azi = np.linspace(0, 2 * np.pi, 720, endpoint=True)
colat = np.pi / 2 * np.ones_like(azi)

# Bandlimited Dirac pulse
dirac_untapered = 4 * np.pi / (N + 1) ** 2 * \
                  sph.bandlimited_dirac(N, azi - dirac_azi)
dirac_tapered = 4 * np.pi / (N + 1) ** 2 * \
                sph.bandlimited_dirac(N, azi - dirac_azi,
                                      w_n=pcs.half_sided_Hann(N))

# Coloration compensation of windowing
compensation_untapered = coloration_compensation(N, N_full, kr)
compensation_tapered = coloration_compensation(N, N_full, kr,
                                               w_taper=pcs.half_sided_Hann(N))

# Get an FIR filter
ntaps = 1024 + 1
assert (ntaps % 2), "Does not produce uneven number of filter taps."
filter_taps = firwin2(ntaps, f / (fs // 2), compensation_tapered)

# %% --- PLOTS ---
plots.polar(azi, dirac_untapered, title='Dirac untapered')
plots.polar(azi, dirac_tapered, title='Dirac tapered')
plots.spectrum(filter_taps, fs, scale_mag=True,
               title='Coloration Equalization')
plt.show()

