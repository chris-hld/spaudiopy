#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Examples for:
# Hold, C., Gamper, H., Pulkki, V., Raghuvanshi, N., & Tashev, I. J. (2019).
# Improving Binaural Ambisonics Decoding by Spherical Harmonics Domain
# Tapering and Coloration Compensation.
# In IEEE International Conference on Acoustics, Speech and Signal Processing.

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin2
from spaudiopy import sph, utils, plots
from spaudiopy import process as pcs

# sampling rate (Hz)
fs = 48000
# evaluated frequencies, 1000 points
f = np.linspace(0, fs / 2, 1000)
# target spherical harmonics order N (>= 3)
N = 5

# tapering windows
w_Hann = pcs.half_sided_Hann(N)
w_rE = sph.max_rE_weights(N)
# Choose here:
w_taper = w_Hann


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
                sph.bandlimited_dirac(N, azi - dirac_azi, w_n=w_taper)

# Coloration compensation of windowing
compensation_untapered = sph.binaural_coloration_compensation(N, f)
compensation_tapered = sph.binaural_coloration_compensation(N, f,
                                                            w_taper=w_taper)

# Get an FIR filter
ntaps = 128 + 1
assert (ntaps % 2), "Does not produce uneven number of filter taps."
filter_taps_untapered = firwin2(ntaps, f / (fs // 2), compensation_untapered)
filter_taps_tapered = firwin2(ntaps, f / (fs // 2), compensation_tapered)

# %% --- PLOTS ---
plots.polar(azi, dirac_untapered, title='Dirac untapered')
plots.polar(azi, dirac_tapered, title='Dirac tapered')
plots.spectrum([filter_taps_untapered, filter_taps_tapered], fs, scale_mag=True,
               title='Coloration Equalization', labels=['untapered', 'tapered'])

plt.show()
