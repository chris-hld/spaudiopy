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


# %% Spherical Harmonics Order
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
sph.check_cond_sht(3, t_az, t_colat, 'real')

# %% Real and Complex SHs
Y_nm_c = sph.sh_matrix(N, azi, colat, 'complex')
Y_nm_r = sph.sh_matrix(N, azi, colat, 'real')

# %%
# Look at some SHTs
sig = np.array([1, 1, 1, 1])
sig_t = np.c_[np.eye(4), np.eye(4)]  # second axis s(t)
sig_B = sph.soundfield_to_b(sig)
F_nm = sph.sht(sig, N, azi, colat, SH_type='real')
F_nm_t = sph.sht(sig_t, N, azi, colat, SH_type='real')

# %%
F_nm_c = sph.sht(sig, N, azi, colat, SH_type='complex')
F_nm_c_t = sph.sht(sig_t, N, azi, colat, SH_type='complex')

# %%
F_nm_lst = sph.sht_lstsq(sig, N, azi, colat, SH_type='complex')
F_nm_lst_t = sph.sht_lstsq(sig_t, N, azi, colat, SH_type='real')

# %% inverse SHT
f = sph.inverse_sht(F_nm, azi, colat, SH_type='real')
f_c_t = sph.inverse_sht(F_nm_c_t, azi, colat, SH_type='complex')
f_lst_t = sph.inverse_sht(F_nm_lst_t, azi, colat, SH_type='real')

# %% Checks
print("Single dimension signal:")
utils.test_diff(sig, f)
print("Complex valued SHT of time signal:")
utils.test_diff(sig_t, f_c_t)
print("Real valued least-squares SHT of time signal:")
utils.test_diff(sig_t, f_lst_t)

# %%
# Check B format conversion
B_sig = np.array([1, 1, 0, 0])  # W, X, Y, Z
F_B = sph.b_to_sh(B_sig)
B_sig_re = sph.sh_to_b(F_B)
print("B format to SH conversion:")
utils.test_diff(B_sig, B_sig_re)

# %%
# Some plots
plots.sh_coeffs(F_nm, title="Ambeo: all channels max")
plots.sh_coeffs(F_B, title="b_to_sh: W+X")

# %%
plots.sh_coeffs_subplot([np.array([1, 0, 0, 0]),
                         np.array([0, 1, 0, 0]),
                         np.array([0, 0, 1, 0]),
                         np.array([0, 0, 0, 1])],
                        titles=["0", "1, -1", "1, 0", "1, 1"])

# %%
plots.sh_coeffs(np.sqrt(2) * np.array([1, 0, 0, -1]), 'complex',
                title="sqrt(2) * [1, 0, 0, -1] complex coeffs")

# %%
# Look at simple B format generator
sig2 = np.ones(8)
B = sph.src_to_B(sig2, np.pi / 4, np.pi / 4)
B_nm = sph.b_to_sh(B)
plots.sh_coeffs(B_nm[:, 0], title="Sig 2 B")

# %%
plt.show()
