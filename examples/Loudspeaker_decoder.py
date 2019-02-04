# ---
# jupyter:
#   jupytext:
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
import sounddevice as sd

from spaudiopy import utils, IO, sig, decoder, sph, plots, grids


# %% User setup
setupname = "graz"
NONUNIFORM = False
LISTEN = True

if setupname is "aalto_full":
    ls_dirs = np.array([[-18, -54, -90, -126, -162, -198, -234, -270, -306,
                         -342, 0, -72, -144, -216, -288, -45, -135, -225,
                         -315, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -10, -10, -10, -10, -10,
                         45, 45, 45, 45, 90]])
    ls_dirs[1, :] = 90 - ls_dirs[1, :]
    normal_limit = 85
    aperture_limit = 90
    opening_limit = 150
    blacklist = None
elif setupname is "aalto_partial":
    ls_dirs = np.array([[-80, -45, 0, 45, 80, -60, -30, 30, 60],
                        [0, 0, 0, 0, 0, 60, 60, 60, 60]])
    ls_dirs[1, :] = 90 - ls_dirs[1, :]
    normal_limit = 85
    aperture_limit = 90
    opening_limit = 150
    blacklist = None
elif setupname is "graz":
    ls_dirs = np.array([[0, 23.7, 48.2, 72.6, 103.1, -100.9, -69.8, -44.8, -21.4,
                         22.7, 67.9, 114.2, -113.3, -65.4, -22.7,
                         46.8, 133.4, -133.4, -43.4],
                        [90.0, 89.6, 89.4, 89.3, 89.4, 89.4, 89.6, 89.5, 89.5,
                         61.5, 61.5, 62.1, 61.6, 61.5, 62.0,
                         33.0, 33.0, 33.4, 32.3]])
    # [90, 90, 90, 90, 90, 90, 90, 90, 90, 60, 60, 60, 60, 60, 60, 30, 30, 30, 30]])
    normal_limit = 85
    aperture_limit = 90
    opening_limit = 135
    blacklist = None
else:
    raise ValueError

x, y, z = utils.sph2cart(utils.deg2rad(ls_dirs[0, :]),
                         utils.deg2rad(ls_dirs[1, :]))
if NONUNIFORM:
    x = np.r_[x, 1.5]
    y = np.r_[y, 1.5]
    z = np.r_[z, 1.5]

listener_position = [0, 0, 0]


# %% Show setup
ls_setup = decoder.LoudspeakerSetup(x, y, z, listener_position)
ls_setup.pop_triangles(normal_limit, aperture_limit, opening_limit, blacklist)

ls_setup.show()
plots.hull_normals(ls_setup)

# Test source location
src = np.array([1, .6, .2])
src_azi, src_colat, _ = utils.cart2sph(*src.tolist())

# %% VBAP
gains_VBAP = decoder.vbap(src, ls_setup)


# %% ALLRAD
# Ambisonic setup
N_e = ls_setup.get_characteristic_order()
ls_setup.setup_for_ambisonic(N_kernel=9)
gains_ALLRAP = decoder.ALLRAP(src, ls_setup, N=N_e)

# Show ALLRAP hulls
ambisonics_hull, kernel_hull = decoder._ALLRAP_hulls(ls_setup, N_kernel=9)
plots.hull(ambisonics_hull, title='Ambisonic hull')
plots.hull(kernel_hull, title='Kernel hull')


# %% test multiple sources
_grid, _weights = grids.load_Fliege_Maier_nodes(10)
G_vbap = decoder.vbap(_grid, ls_setup)
G_allrap = decoder.ALLRAP(_grid, ls_setup)

# %% Look at some measures
plots.decoder_performance(ls_setup, 'VBAP')
plots.decoder_performance(ls_setup, 'ALLRAP')

# %% Binauralize
fs = 44100
hrirs = IO.load_hrir(fs)

l_vbap_IR, r_vbap_IR = ls_setup.binauralize(gains_VBAP, fs)

l_allrap_IR, r_allrap_IR = ls_setup.binauralize(gains_ALLRAP, fs)


# %%
fig = plt.figure()
fig.add_subplot(3, 1, 1)
plt.plot(hrirs.select_direction(src_azi, src_colat)[0])
plt.plot(hrirs.select_direction(src_azi, src_colat)[1])
plt.grid(True)
plt.title("hrir")
fig.add_subplot(3, 1, 2)
plt.plot(l_vbap_IR)
plt.plot(r_vbap_IR)
plt.grid(True)
plt.title("binaural VBAP")
fig.add_subplot(3, 1, 3)
plt.plot(l_allrap_IR)
plt.plot(r_allrap_IR)
plt.grid(True)
plt.title("binaural ALLRAP")
plt.tight_layout()

# Listen to some
if LISTEN:
    s_in = sig.MonoSignal.from_file('../data/piano_mono.flac', fs)
    s_in.trim(2.6, 6)

    s_out_vbap = sig.MultiSignal([s_in.filter(l_vbap_IR),
                                  s_in.filter(r_vbap_IR)],
                                 fs=fs)
    s_out_allrap = sig.MultiSignal([s_in.filter(l_allrap_IR),
                                    s_in.filter(r_allrap_IR)],
                                   fs=fs)
    s_out_hrir = sig.MultiSignal([s_in.filter(
                                      hrirs.select_direction(src_azi,
                                                             src_colat)[0]),
                                  s_in.filter(
                                      hrirs.select_direction(src_azi,
                                                             src_colat)[1])],
                                 fs=fs)
    print("input")
    sd.play(s_in.signal,
            int(s_in.fs))
    sd.wait()
    print("hrir")
    sd.play(s_out_hrir.get_signals().T,
            int(s_in.fs))
    sd.wait()
    print("vbap")
    sd.play(s_out_vbap.get_signals().T,
            int(s_in.fs))
    sd.wait()
    print("allrap")
    sd.play(s_out_allrap.get_signals().T,
            int(s_in.fs))
    sd.wait()

    fig = plt.figure()
    fig.add_subplot(4, 1, 1)
    plt.plot(s_in.signal)
    plt.grid(True)
    plt.title("dry")
    fig.add_subplot(4, 1, 2)
    plt.plot(s_out_hrir.get_signals().T)
    plt.grid(True)
    plt.title("hrir")
    fig.add_subplot(4, 1, 3)
    plt.plot(s_out_vbap.get_signals().T)
    plt.grid(True)
    plt.title("binaural VBAP")
    fig.add_subplot(4, 1, 4)
    plt.plot(s_out_allrap.get_signals().T)
    plt.grid(True)
    plt.title("binaural ALLRAP")
    plt.tight_layout()

plt.show()
