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

from spaudiopy import utils, IO, sph, plots, grids

# %% ..
pi = np.pi

# %%
NFFT = 128
fs = 48000
N = 1

# %%
# Grids
t = grids.load_t_design(2)
t_az, t_colat, t_r = utils.cart2sph(t[:, 0], t[:, 1], t[:, 2])
azi = t_az
colat = t_colat

# %%
# First, check condition number to which SH order the SHT is stable
# Tetraeder is not suited for SHT N>1:
sph.check_cond_SHT(3, t_az, t_colat, 'real')

# %%
# different versions of SH matrix
#from sound_field_analysis import gen as sf_gen
#from sound_field_analysis import sph as sf_sph
#import micarray
#Y1 = micarray.modal.angular.sht_matrix(1, t_az, t_colat)
#Y2 = sf_sph.sph_harm_all(1, t_az, t_colat)
Y3 = sph.SH_matrix(1, t_az, t_colat)
# Real and Complex SHs
Y_nm_c = sph.SH_matrix(N, azi, colat, 'complex')
Y_nm_r = sph.SH_matrix(N, azi, colat, 'real')

# %%
# Look at some SHTs
sig = np.array([1, 1, 1, 1])
sig_t = np.c_[np.eye(4), np.eye(4)]  # second axis s(t)
sig_B = sph.tetra_to_B(sig)
F_nm = sph.SHT(sig, N, azi, colat, SH_type='real')
F_nm_t = sph.SHT(sig_t, N, azi, colat, SH_type='real')

# %%
F_nm_c = sph.SHT(sig, N, azi, colat, SH_type='complex')
F_nm_c_t = sph.SHT(sig_t, N, azi, colat, SH_type='complex')

# %%
F_nm_lst = sph.SHT_lstsq(sig, N, azi, colat, SH_type='complex')
F_nm_lst_t = sph.SHT_lstsq(sig_t, N, azi, colat, SH_type='real')

# %%
# Check inverse SHT
f = sph.inverseSHT(F_nm, azi, colat, SH_type='real')
f_c_t = sph.inverseSHT(F_nm_c_t, azi, colat, SH_type='complex')
f_lst = sph.inverseSHT(F_nm_lst, azi, colat, SH_type='complex')
utils.test_diff(sig, f)
utils.test_diff(sig_t, f_c_t)
utils.test_diff(sig, f_lst)

# %%
# Check B format conversion
B_sig = np.array([1, 1, 0, 0])  # W, X, Y, Z
F_B = sph.B_to_SH(B_sig)
B_sig_re = sph.SH_to_B(F_B)
utils.test_diff(B_sig, B_sig_re)

# %%
# Some plots
plots.sph_coeffs(F_nm, title="Ambeo: all channels max")
plots.sph_coeffs(F_B, title="B_to_SH: W+X")

# %%
plots.subplot_sph_coeffs([np.array([1, 0, 0, 0]),
                          np.array([0, 1, 0, 0]),
                          np.array([0, 0, 1, 0]),
                          np.array([0, 0, 0, 1])],
                         title=["0", "1, -1", "1, 0", "1, 1"])

# %%
plots.sph_coeffs(np.sqrt(2) * np.array([1, 0, 0, 1]), 'complex',
                 title="Both * sqrt2 complex")

# %%
# Look at simple B format generator
sig2 = np.ones(8)
B = sph.src_to_B(sig2, np.pi / 4, np.pi / 4)
B_nm = sph.B_to_SH(B)
plots.sph_coeffs(B_nm[:, 0], title="Sig 2 B")

# %%
plt.show()
