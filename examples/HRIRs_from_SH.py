# ---
# jupyter:
#   jupytext:
#     comment_magics: 'false'
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.1'
#       jupytext_version: 0.8.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
#   language_info:
#     codemirror_mode:
#       name: ipython
#       version: 3
#     file_extension: .py
#     mimetype: text/x-python
#     name: python
#     nbconvert_exporter: python
#     pygments_lexer: ipython3
#     version: 3.6.8
# ---

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
from scipy import signal as scysignal

import soundfile as sf

from spaudiopy import plots, sph, process, utils, grids, IO


# %% Setup
N = 35  # modal order of grid
N_hrtf = 35  # modal order of HRTFs

azi, colat, weights = grids.gauss(N)  # grid positions
plt.figure()
plt.scatter(azi, colat)

# %% Load HRTF
try:
    file = loadmat(
    '../data/0 HRIRs neutral head orientation/SphericalHarmonics/FABIAN_HRIR_measured_HATO_0.mat')
except FileNotFoundError:
    import requests, zipfile, io
    print("Downloading from https://depositonce.tu-berlin.de/handle/11303/6153.3 ...")
    r = requests.get('https://depositonce.tu-berlin.de/bitstream/11303/6153.3/8/FABIAN_HRTFs_NeutralHeadOrientation.zip')
    with zipfile.ZipFile(io.BytesIO(r.content)) as zip_ref:
        zip_ref.extractall('../data/')
    file = loadmat(
    '../data/0 HRIRs neutral head orientation/SphericalHarmonics/FABIAN_HRIR_measured_HATO_0.mat')

# Extracting the data is a bit ugly here...
SamplingRate = int(file['SamplingRate'])
SH_l = file['SH'][0][0][0]
SH_r = file['SH'][0][0][1]
f = np.squeeze(file['SH'][0][0][5])
print("Positive frequncy bins:", len(f))
print("SH shape:", SH_l.shape)
plt.plot(np.abs(SH_l[:, 8]))
plt.xlabel('SH coefficient')
plt.ylabel('Amplitude')
plt.title('Left HRTF SH at f={:.3} Hz'.format(f[8]))

# %% Inverse SHT
HRTF_l = sph.inverse_sht(SH_l, azi, colat, 'complex')
HRTF_r = sph.inverse_sht(SH_r, azi, colat, 'complex')

assert HRTF_l.shape == HRTF_r.shape
print("HRTF shape:", HRTF_l.shape)
plt_idx = int(HRTF_l.shape[0] / 2)
plots.freq_resp(f, [HRTF_l[plt_idx, :],
                    HRTF_r[plt_idx, :]],
                title=r"HRTF for $\phi={:.2}, \theta={:.2}$".format(
                    azi[plt_idx], colat[plt_idx]),
                labels=['left', 'right'])

# %% [markdown]
# The inverse spherical harmonics transform renders from the spherical harmonics representation to the defined grid.
# Since the original input to the SHT are HRTFs we obtain again the time-frequency transfer functions.
#
# Now, from time-frequency domain to time domain HRIRs by the inverse Fourier transform.

# %%
hrir_l = np.fft.irfft(HRTF_l)  # creates 256 samples(t)
hrir_r = np.fft.irfft(HRTF_r)  # creates 256 samples(t)

assert hrir_l.shape == hrir_r.shape
print("hrir shape:", hrir_l.shape)
plt.figure()
plt.plot(utils.db(hrir_l[plt_idx, :]), label='left')
plt.plot(utils.db(hrir_r[plt_idx, :]), label='right')
plt.legend()
plt.grid()
plt.xlabel('t in samples')
plt.ylabel('A in dB')
plt.title('HRIR ETC')

# %% Headphone compensation / applying inverse common transfer function
sofa_data = IO.load_sofa_data('../data/0 HRIRs neutral head orientation/SOFA/FABIAN_CTF_measured_inverted_smoothed.sofa')
h_headphone = sofa_data['Data.IR']
h_samplerate = sofa_data['Data.SamplingRate']

assert SamplingRate == h_samplerate
plots.spectrum(h_headphone, h_samplerate, ylim=[-60, -30],
               labels='HP Filter', title='Headphone compensation spectrum')

hrir_l_hp = np.apply_along_axis(lambda m:
                                scysignal.convolve(m, h_headphone),
                                axis=1, arr=hrir_l)
hrir_r_hp = np.apply_along_axis(lambda m:
                                scysignal.convolve(m, h_headphone),
                                axis=1, arr=hrir_r)

print("Compensated HRIR:", hrir_l_hp.shape)

freq = np.fft.rfftfreq(hrir_l_hp.shape[1], d=1. / SamplingRate)
plots.freq_resp(freq, [np.fft.rfft(hrir_l_hp[plt_idx, :]),
                       np.fft.rfft(hrir_r_hp[plt_idx, :])],
                labels=['HRTF left', 'HRTF right'],
                title='Compensated HRTF')

# %% Resample to 48k
fs_target = 48000
hrir_l_hp48k, hrir_r_hp48k, _ = process.resample_hrirs(hrir_l_hp, hrir_r_hp,
                                                       SamplingRate,
                                                       fs_target)
print("Resampled HRIR:", hrir_l_hp48k.shape)
freq = np.fft.rfftfreq(hrir_l_hp48k.shape[1], d=1. / SamplingRate)
plots.freq_resp(freq, [np.fft.rfft(hrir_l_hp48k[plt_idx, :]),
                   np.fft.rfft(hrir_r_hp48k[plt_idx, :])],
                labels=['HRTF left', 'HRTF right'],
                title='Resampled HRTF')

# %% Save to .mat
# IO.get_default_hrirs() does the job now, have a look!
SAVENEW = False
if SAVENEW:
    savemat('../data/HRTF_default', {'hrir_l': hrir_l_hp,
                                     'hrir_r': hrir_r_hp,
                                     'azi': azi, 'elev': colat,
                                     'SamplingRate': SamplingRate})
    savemat('../data/HRTF_default48k', {'hrir_l': hrir_l_hp48k,
                                       'hrir_r': hrir_r_hp48k,
                                       'azi': azi, 'elev': colat,
                                       'SamplingRate': fs_target})

# %%
plt.show()
