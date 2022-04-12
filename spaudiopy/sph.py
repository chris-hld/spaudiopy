# -*- coding: utf-8 -*-
"""Spherical Harmonics.

.. plot::
    :context: reset

    import numpy as np
    import matplotlib.pyplot as plt
    plt.rcParams['axes.grid'] = True

    import spaudiopy as spa

    spa.plots.sh_coeffs_subplot([np.sqrt(4*np.pi) * np.array([1, 0, 0, 0]),
                                 np.sqrt(4/3*np.pi) * np.array([0, 1, 0, 0]),
                                 np.sqrt(4/3*np.pi) * np.array([0, 0, 1, 0]),
                                 np.sqrt(4/3*np.pi) * np.array([0, 0, 0, 1])],
                                titles=["$Y_{0, 0}$", "$Y_{1, -1}$",
                                       "$Y_{1, 0}$", "$Y_{1, 1}$"])
"""

import numpy as np
from scipy import special as scyspecial
from scipy.linalg import lstsq

from . import utils


def sh_matrix(N, azi, colat, SH_type='complex', weights=None):
    r"""Matrix of spherical harmonics up to order N for given angles.

    Computes a matrix of spherical harmonics up to order :math:`N`
    for the given angles/grid.

    .. math::

        \mathbf{Y} = \left[ \begin{array}{ccccc}
        Y_0^0(\theta[0], \phi[0]) & Y_1^{-1}(\theta[0], \phi[0]) &
        Y_1^0(\theta[0], \phi[0]) &
        \dots & Y_N^N(\theta[0], \phi[0])  \\
        Y_0^0(\theta[1], \phi[1]) & Y_1^{-1}(\theta[1], \phi[1]) &
        Y_1^0(\theta[1], \phi[1]) &
        \dots & Y_N^N(\theta[1], \phi[1])  \\
        \vdots & \vdots & \vdots & \vdots & \vdots \\
        Y_0^0(\theta[Q-1], \phi[Q-1]) & Y_1^{-1}(\theta[Q-1], \phi[Q-1]) &
        Y_1^0(\theta[Q-1], \phi[Q-1]) &
        \dots & Y_N^N(\theta[Q-1], \phi[Q-1])
        \end{array} \right]

    where

    .. math::

        Y_n^m(\theta, \phi) = \sqrt{\frac{2n + 1}{4 \pi}
                                    \frac{(n-m)!}{(n+m)!}} P_n^m(\cos \theta)
                              e^{i m \phi}

    When using `SH_type='real'`, the real spherical harmonics
    :math:`Y_{n,m}(\theta, \phi)` are implemented as a relation to
    :math:`Y_n^m(\theta, \phi)`.

    Parameters
    ----------
    N : int
        Maximum SH order.
    azi : (Q,) array_like
        Azimuth.
    colat : (Q,) array_like
        Colatitude.
    SH_type :  'complex' or 'real' spherical harmonics.
    weights : (Q,) array_like, optional
        Quadrature weights.

    Returns
    -------
    Ymn : (Q, (N+1)**2) numpy.ndarray
        Matrix of spherical harmonics.

    Notes
    -----
    The convention used here is also known as N3D-ACN (for SH_type='real').

    """
    azi = utils.asarray_1d(azi)
    colat = utils.asarray_1d(colat)
    if azi.ndim == 0:
        Q = 1
    else:
        Q = len(azi)
    if weights is None:
        weights = np.ones(Q)
    if SH_type == 'complex':
        Ymn = np.zeros([Q, (N+1)**2], dtype=np.complex_)
    elif SH_type == 'real':
        Ymn = np.zeros([Q, (N+1)**2], dtype=np.float_)
    else:
        raise ValueError('SH_type unknown.')

    idx = 0
    for n in range(N+1):
        for m in range(-n, n+1):
            if SH_type == 'complex':
                Ymn[:, idx] = weights * scyspecial.sph_harm(m, n, azi, colat)
            elif SH_type == 'real':
                if m == 0:
                    Ymn[:, idx] = weights * np.real(
                                  scyspecial.sph_harm(0, n, azi, colat))
                if m < 0:
                    Ymn[:, idx] = weights * np.sqrt(2) * (-1) ** abs(m) * \
                                  np.imag(
                                  scyspecial.sph_harm(abs(m), n, azi, colat))
                if m > 0:
                    Ymn[:, idx] = weights * np.sqrt(2) * (-1) ** abs(m) * \
                                  np.real(
                                  scyspecial.sph_harm(abs(m), n, azi, colat))

            idx += 1
    return Ymn


def sht(f, N, azi, colat, SH_type, weights=None, Y_nm=None):
    """Spherical harmonics transform of f for appropriate point sets.

    If f is a QxS matrix then the transform is applied to each column
    of f, and returns the coefficients at each column of F_nm
    respectively.

    Parameters
    ----------
    f : (Q, S)
        The spherical function(S) evaluated at Q directions 'azi/colat'.
    N : int
        Maximum SH order.
    azi : (Q,) array_like
        Azimuth.
    colat : (Q,) array_like
        Colatitude.
    SH_type :  'complex' or 'real' spherical harmonics.
    weights : (Q,) array_like, optional
        Quadrature weights.
    Y_nm : (Q, (N+1)**2) numpy.ndarray, optional
        Matrix of spherical harmonics.

    Returns
    -------
    F_nm : ((N+1)**2, S) numpy.ndarray
        Matrix of spherical harmonics coefficients of spherical function(S).
    """
    if f.ndim == 1:
        f = f[:, np.newaxis]  # upgrade to handle 1D arrays
    if Y_nm is None:
        Y_nm = sh_matrix(N, azi, colat, SH_type)
    if weights is None:
        Npoints = Y_nm.shape[0]
        Y_nm_transform = (4*np.pi / Npoints) * Y_nm.conj()
    else:
        Y_nm_transform = Y_nm.conj()
        # weights should sum to 4pi
        f = np.dot(np.diag(weights), f)
    return np.matmul(Y_nm_transform.T, f)


def sht_lstsq(f, N, azi, colat, SH_type, Y_nm=None):
    """Spherical harmonics transform  of f as least-squares solution.

    If f is a QxS matrix then the transform is applied to each column
    of f, and returns the coefficients at each column of F_nm
    respectively.

    Parameters
    ----------
    f : (Q, S)
        The spherical function(S) evaluated at Q directions 'azi/colat'.
    N : int
        Maximum SH order.
    azi : (Q,) array_like
        Azimuth.
    colat : (Q,) array_like
        Colatitude.
    SH_type :  'complex' or 'real' spherical harmonics.
    Y_nm : (Q, (N+1)**2) numpy.ndarray, optional
        Matrix of spherical harmonics.

    Returns
    -------
    F_nm : ((N+1)**2, S) numpy.ndarray
        Matrix of spherical harmonics coefficients of spherical function(S).
    """
    if f.ndim == 1:
        f = f[:, np.newaxis]  # upgrade to handle 1D arrays
    if Y_nm is None:
        Y_nm = sh_matrix(N, azi, colat, SH_type)
    return lstsq(Y_nm, f)[0]


def inverse_sht(F_nm, azi, colat, SH_type, N=None, Y_nm=None):
    """Perform the inverse spherical harmonics transform.

    Parameters
    ----------
    F_nm : ((N+1)**2, S) numpy.ndarray
        Matrix of spherical harmonics coefficients of spherical function(S).
    azi : (Q,) array_like
        Azimuth.
    colat : (Q,) array_like
        Colatitude.
    SH_type :  'complex' or 'real' spherical harmonics.
    N : int, optional
        Maximum SH order.
    Y_nm : (Q, (N+1)**2) numpy.ndarray, optional
        Matrix of spherical harmonics.

    Returns
    -------
    f : (Q, S)
        The spherical function(S) evaluated at Q directions 'azi/colat'.
    """
    assert(F_nm.ndim == 2)
    if N is None:
        N = int(np.sqrt(F_nm.shape[0]) - 1)
    if Y_nm is None:
        Y_nm = sh_matrix(N, azi, colat, SH_type)
    # perform the inverse transform up to degree N
    return np.matmul(Y_nm, F_nm[:(N + 1) ** 2, :])


def check_cond_sht(N, azi, colat, SH_type, lim=None):
    """Check if condition number for a least-squares SHT(N) is greater 'lim'."""
    if lim is None:
        lim = N + N / 2
    Y = sh_matrix(N, azi, colat, SH_type)
    c = np.zeros(N + 1)
    YYn = np.matmul(Y.conj().T, Y)
    c = np.linalg.cond(YYn)
    if np.any(c > lim):
        print("High condition number! " + str(c))
    return c


def calculate_grid_weights(azi, zen, order=None):
    """Approximate quadrature weights by pseudo-inverse.
    
    Parameters
    ----------
    azi : (Q,) array_like
        Azimuth.
    zen : (Q,) array_like
        Zenith / Colatitude.
    order : int, optional
        Supported order N, searched if not provided.
    
    Returns
    -------
    weights : (Q,) array_like
        Grid / Quadrature weights.
    
    References
    ---------
    Fornberg, B., & Martel, J. M. (2014). On spherical harmonics based 
    numerical quadrature over the surface of a sphere. 
    Advances in Computational Mathematics.

    """
    if order is None:  # search for max supported SHT order
        for itOrder in range(1,100):
            cond =  check_cond_sht(itOrder, azi, zen, 'real', np.inf)
            if cond > 2*(itOrder+1):  # experimental condition
                order = itOrder-1
                break
    assert(order>0)
    Y = sh_matrix(order, azi, zen, 'real')
    P_leftinv = np.linalg.pinv(Y)
    weights = np.sqrt(4*np.pi) * P_leftinv[0, :]
    if (np.abs(np.sum(weights) - 4*np.pi) > 0.01) or np.any(weights < 0):
        print('Could not calculate weights')
    return weights


def n3d_to_sn3d(F_nm, sh_axis=0):
    """Convert N3D (orthonormal) to SN3D (Schmidt semi-normalized) signals.

    Parameters
    ----------
    F_nm : ((N_sph+1)**2, S) numpy.ndarray
        Matrix of spherical harmonics coefficients of spherical function(S).
    sh_axis : int, optional
        SH axis. The default is 0.

    Returns
    -------
    F_nm : ((N_sph+1)**2, S) numpy.ndarray
        Matrix of spherical harmonics coefficients of spherical function(S).

    """
    assert(F_nm.ndim == 2)
    # Input SH order
    N = int(np.sqrt(F_nm.shape[sh_axis]) - 1)
    # 1/sqrt(2n+1) conversion factor
    n_norm = np.array([1/np.sqrt(2*n + 1) for n in range(N + 1)])
    # Broadcast
    n_norm = np.expand_dims(repeat_per_order(n_norm), axis=sh_axis-1)
    return n_norm * F_nm


def sn3d_to_n3d(F_nm, sh_axis=0):
    """Convert SN3D (Schmidt semi-normalized) to N3D (orthonormal) signals.

    Parameters
    ----------
    F_nm : ((N_sph+1)**2, S) numpy.ndarray
        Matrix of spherical harmonics coefficients of spherical function(S).
    sh_axis : int, optional
        SH axis. The default is 0.

    Returns
    -------
    F_nm : ((N_sph+1)**2, S) numpy.ndarray
        Matrix of spherical harmonics coefficients of spherical function(S).

    """
    assert(F_nm.ndim == 2)
    # Input SH order
    N = int(np.sqrt(F_nm.shape[sh_axis]) - 1)
    # sqrt(2n+1) conversion factor
    n_norm = np.array([np.sqrt(2*n + 1) for n in range(N + 1)])
    # Broadcast
    n_norm = np.expand_dims(repeat_per_order(n_norm), axis=sh_axis-1)
    return n_norm * F_nm


def platonic_solid(shape):
    """Return coordinates of shape='tetrahedron' only, yet."""
    if shape in ['tetra', 'tetrahedron']:
        u = np.array([[1, 1, 1],
                      [1, -1, -1],
                      [-1, 1, -1],
                      [-1, -1, 1]]/np.sqrt(3))
    return u


def sh_to_b(F_nm, W_weight=None):
    """Convert first order SH input signals to B-format.

    Parameters
    ----------
    F_nm : (4, S) numpy.ndarray
        First order spherical harmonics function coefficients.
    W-weight : float
        Weight on W channel.

    Returns
    -------
    B_nm : (4, S) numpy.ndarray
        B-format signal(S).
    """
    # B-format
    if W_weight is None:
        # Traditionally  W_weight = 1 / np.sqrt(2)
        W_weight = 1
    # Conversion matrix SH to B
    M = np.sqrt(4*np.pi) * np.array([[W_weight, 0, 0, 0],
                                    [0, 0, 0, 1/np.sqrt(3)],
                                    [0, 1/np.sqrt(3), 0, 0],
                                    [0, 0, 1/np.sqrt(3), 0]])
    return np.apply_along_axis(np.dot, 0, F_nm, M.T)


def b_to_sh(B, W_weight=None):
    """Convert B-format input to first order SH signals.

    Parameters
    ----------
    B_nm : (4, S) numpy.ndarray
        B-format signal(S).

    W-weight : float
        Weight on W channel.

    Returns
    -------
    F_nm : (4, S) numpy.ndarray
        First order spherical harmonics function coefficients.
    """
    # B-format
    if W_weight is None:
        # Traditionally  W_weight = 1 / np.sqrt(2)
        W_weight = 1
    # Conversion matrix SH to B
    M = np.sqrt(4*np.pi) * np.array([[W_weight, 0, 0, 0],
                                    [0, 0, 0, 1/np.sqrt(3)],
                                    [0, 1/np.sqrt(3), 0, 0],
                                    [0, 0, 1/np.sqrt(3), 0]])
    M_inv = np.linalg.inv(M)
    return np.apply_along_axis(np.dot, 0, B, M_inv.T)


def soundfield_to_b(sig, W_weight=None):
    """Convert soundfield tetraeder mic input signals to B-format by SHT.

    Parameters
    ----------
    sig : (4, S)
        Tetraeder mic signals(S).
    W-weight : float
        Weight on W channel.

    Returns
    -------
    B_nm : (4, S) numpy.ndarray
        B-format signal(S).
    """
    # get tetraeder position
    N = 1
    t = platonic_solid('tetrahedron')
    t_az, t_colat, t_r = utils.cart2sph(t[:, 0], t[:, 1], t[:, 2])
    # SHT of input signal
    F_nm = sht(sig, N, azi=t_az, colat=t_colat, SH_type='real')
    return sh_to_b(F_nm, W_weight)


def src_to_b(sig, src_azi, src_colat):
    """Get B format signal channels for source in direction azi/colat."""
    sig = utils.asarray_1d(sig)
    src_azi = utils.asarray_1d(src_azi)
    src_colat = utils.asarray_1d(src_colat)
    gw = np.ones(len(src_azi))
    gx, gy, gz = utils.sph2cart(src_azi, src_colat)
    g = np.c_[gw, gx, gy, gz]
    return np.outer(g, sig)


def src_to_sh(sig, src_azi, src_zen, N_sph, SH_type='real'):
    """Source signal(s) plane wave encoded in spherical harmonics.
    

    Parameters
    ----------
    sig : (num_src, S) numpy.ndarray
        Source signal(s).
    src_azi : array_like
    src_zen : array_like
    N_sph : int
    SH_type : 'real' (default) or 'complex', optional

    Returns
    -------
    ((N_sph+1)**2, S) numpy.ndarray
        Source signal(s) in SHD.
    
    Examples
    --------
    .. plot::
        :context: close-figs

        src_sig = np.array([1, 0.5], ndmin=2).T * np.random.randn(2, 1000)
        N_sph = 3
        src_azi = [np.pi/2, -np.pi/3]
        src_zen = [np.pi/4, np.pi/2]

        sig_nm = spa.sph.src_to_sh(src_sig, src_azi, src_zen, N_sph)

        spa.plots.sh_rms_map(sig_nm)

    """
    sig = np.atleast_2d(sig)
    src_azi = utils.asarray_1d(src_azi)
    src_zen = utils.asarray_1d(src_zen)

    Y_nm = sh_matrix(N_sph, src_azi, src_zen, SH_type=SH_type)
    return Y_nm.conj().T @ sig


def bandlimited_dirac(N, d, w_n=None):
    r"""Order N spatially bandlimited Dirac pulse at central angle d.

    Parameters
    ----------
    N : int
        SH order.
    d : (Q,) array_like
        Central angle in rad.
    w_n : (N,) array_like, optional. Default is None.
        Tapering window w_n.

    Returns
    -------
    dirac : (Q,) array_like
        Amplitude at central angle d.

    Notes
    -----
    Normalize with

    .. math::  \sum^N \frac{2N + 1}{4 \pi} = \frac{(N+1)^2}{4 \pi} .

    References
    ----------
    Rafaely, B. (2015). Fundamentals of Spherical Array Processing. Springer.,
    eq. (1.60).

    Examples
    --------
    .. plot::
        :context: close-figs

        dirac_azi = np.deg2rad(0)
        dirac_colat = np.deg2rad(90)
        N = 5

        # cross section
        azi = np.linspace(0, 2 * np.pi, 720, endpoint=True)

        # Bandlimited Dirac pulse
        dirac_bandlim = 4 * np.pi / (N + 1) ** 2 * \
                            spa.sph.bandlimited_dirac(N, azi - dirac_azi)

        spa.plots.polar(azi, dirac_bandlim)

    """
    d = utils.asarray_1d(d)
    if w_n is None:
        w_n = np.ones(N + 1)
    assert(len(w_n) == N + 1), "Provide weight per order."
    g_n = np.zeros([(N + 1)**2, len(d)])
    for n, i in enumerate(range(N + 1)):
        g_n[i, :] = w_n[i] * (2 * n + 1) / (4 * np.pi) * \
                    scyspecial.eval_legendre(n, np.cos(d))
    dirac = np.sum(g_n, axis=0)
    return dirac


def max_rE_weights(N):
    """Return max-rE modal weight coefficients for spherical harmonics order N.

    See Also
    --------
    :py:func:`spaudiopy.sph.unity_gain` : Unit amplitude compensation.

    References
    ----------
    Zotter, F., & Frank, M. (2012). All-Round Ambisonic Panning and Decoding.
    Journal of Audio Engineering Society, eq. (10).

    Examples
    --------
    .. plot::
        :context: close-figs

        dirac_azi = np.deg2rad(45)
        dirac_colat = np.deg2rad(45)
        N = 5

        # cross section
        azi = np.linspace(0, 2 * np.pi, 720, endpoint=True)

        # Bandlimited Dirac pulse, with max r_E tapering window
        w_n = spa.sph.max_rE_weights(N)
        w_n = spa.sph.unity_gain(w_n)
        dirac_tapered = spa.sph.bandlimited_dirac(N, azi - dirac_azi, w_n=w_n)

        spa.plots.polar(azi, dirac_tapered)

    """
    theta = np.deg2rad(137.9) / (N + 1.51)
    a_n = scyspecial.eval_legendre(np.arange(N + 1), np.cos(theta))
    return a_n


def r_E(p, g):
    """Calculate r_E vector and magnitude from loudspeaker gains.

    Parameters
    ----------
    p : (Q, 3) numpy.ndarray
        Q loudspeaker position vectors.
    g : (S, Q) numpy.ndarray
        Q gain vectors per source S.

    Returns
    -------
    rE : (S, 3) numpy.ndarray
        rE vector.
    rE_mag : (S,) array_like
        rE magnitude (radius).

    References
    ----------
    Zotter, F., & Frank, M. (2012). All-Round Ambisonic Panning and Decoding.
    Journal of Audio Engineering Society, eq. (16).
    """
    p = np.atleast_2d(p)
    g = np.atleast_2d(g)
    assert(p.shape[0] == g.shape[1]), 'Provide gain per speaker!'
    E = np.sum(g**2, axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        rE = np.diag(1 / E) @ (g**2 @ p)
    rE_mag = np.sqrt(np.sum(rE**2, axis=1))
    # catch division by zero NaN
    rE_mag = np.nan_to_num(rE_mag)
    return rE, rE_mag


def project_on_sphere(x, y, z):
    """Little helper that projects x, y, z onto unit sphere."""
    phi, theta, r = utils.cart2sph(x, y, z)
    r = np.ones_like(r)
    return utils.sph2cart(phi, theta, r)


def repeat_per_order(c):
    """Repeat each coefficient in 'c' m times per spherical order n.

    Parameters
    ----------
    c : (N,) array_like
        Coefficients up to SH order N.

    Returns
    -------
    c_reshaped : ((N+1)**2,) array like
        Reshaped input coefficients.
    """
    c = utils.asarray_1d(c)
    N = len(c) - 1
    return np.repeat(c, 2*np.arange(N+1)+1)


def spherical_hn2(n, z, derivative=False):
    """Spherical Hankel function of the second kind.

    Parameters
    ----------
    n : int, array_like
        Order of the spherical Hankel function (n >= 0).
    z : complex or float, array_like
        Argument of the spherical Hankel function.
    derivative : bool, optional
        If True, the value of the derivative (rather than the function
        itself) is returned.

    Returns
    -------
    hn2 : array_like


    References
    ----------
    http://mathworld.wolfram.com/SphericalHankelFunctionoftheSecondKind.html
    """
    with np.errstate(invalid='ignore'):
        yi = 1j * scyspecial.spherical_yn(n, z, derivative)
    return scyspecial.spherical_jn(n, z, derivative) - yi


def mode_strength(n, kr, sphere_type='rigid'):
    """Mode strength b_n(kr) for an incident plane wave on sphere.

    Parameters
    ----------
    n : int
        Degree.
    kr : array_like
        kr vector, product of wavenumber k and radius r_0.
    sphere_type : 'rigid' or 'open'

    Returns
    -------
    b_n : array_like
        Mode strength b_n(kr).

    References
    ----------
    Rafaely, B. (2015). Fundamentals of Spherical Array Processing. Springer.
    eq. (4.4) and (4.5).
    """
    if sphere_type == 'open':
        b_n = 4*np.pi*1j**n * scyspecial.spherical_jn(n, kr)
    elif sphere_type == 'rigid':
        b_n = 4*np.pi*1j**n * (scyspecial.spherical_jn(n, kr) -
                               (scyspecial.spherical_jn(n, kr, True) /
                                spherical_hn2(n, kr, True)) *
                               spherical_hn2(n, kr))
    else:
        raise ValueError('sphere_type Not implemented.')
    return b_n


def pressure_on_sphere(N, kr, weights=None):
    """Calculate the diffuse field pressure frequency response of a spherical
    scatterer, up to SH order N.

    Parameters
    ----------
    N : int
        SH order.
    kr : array_like
        kr vector, product of wavenumber k and radius r_0.
    weights : (N+1,) array_like
        SH order weights.

    Returns
    -------
    p : array_like
        Pressure p(kr)|N.

    References
    ----------
    Ben-Hur, Z., Brinkmann, F., Sheaffer, J., et.al. (2017).
    Spectral equalization in binaural signals represented by order-truncated
    spherical harmonics. The Journal of the Acoustical Society of America,
    eq. (11).
    """
    p_N = np.zeros_like(kr)
    if weights is None:
        weights = np.ones(N + 1)
    for n in range(N + 1):
        p_N += weights[n] * (2 * n + 1) * np.abs(mode_strength(n, kr))**2
    return 1 / (4 * np.pi) * np.sqrt(p_N)


def binaural_coloration_compensation(N, f, r_0=0.0875, w_taper=None):
    """Spectral equalization gain G(kr)|N for diffuse field of order N.
    This filter compensates the high frequency roll of that occurs for order
    truncated SH signals. It models the human head as a rigid sphere of radius
    r_0 (e.g. 0.0875m) and compensates the binaural signals.

    Parameters
    ----------
    N : int
        SH order.
    f : array_like
        Time-frequency in Hz.
    r_0 : radius
        Rigid sphere radius (approx. human head).
    w_taper : (N+1,) array_like
        SH order weights for tapering. See e.g. 'process.half_sided_Hann'.

    Returns
    -------
    gain : array_like
        Filter gain(kr).

    See Also
    --------
    :py:func:`spaudiopy.process.gain_clipping` : Limit maximum gain.

    References
    ----------
    Hold, C., Gamper, H., Pulkki, V., Raghuvanshi, N., & Tashev, I. J. (2019).
    Improving Binaural Ambisonics Decoding by Spherical Harmonics Domain
    Tapering and Coloration Compensation.
    In IEEE International Conference on Acoustics, Speech and Signal
    Processing.

    Examples
    --------
    .. plot::
        :context: close-figs

        fs = 48000
        f = np.linspace(0, fs / 2, 1000)
        # target spherical harmonics order N (>= 3)
        N = 5
        # tapering window
        w_rE = spa.sph.max_rE_weights(N)

        compensation_tapered = spa.sph.binaural_coloration_compensation(
                                N, f, w_taper=w_rE)
        compensation_tapered_lim = spa.process.gain_clipping(
                                    compensation_tapered,
                                    spa.utils.from_db(12))
        spa.plots.freq_resp(f, [compensation_tapered,
                                compensation_tapered_lim],
                            ylim=(-5, 25),
                            labels=[r'$N=5, max_{rE}$', 'with soft lim'])

    """
    c = 343  # speed of sound (m/s)
    k = (2 * np.pi * f) / c
    kr = k * r_0
    # get aliasing free N > kr
    N_full = int(np.ceil(kr[-1]))

    gain = pressure_on_sphere(N_full, kr) / \
           pressure_on_sphere(N, kr, weights=w_taper)  # noqa: E127
    # catch NaNs
    gain[np.isnan(gain)] = 1.
    return gain


def unity_gain(w_n):
    """Make modal weighting / tapering unit amplitude in steering direction.

    Parameters
    ----------
    w_n : (N+1,) array_like
        Modal weighting factors.

    Returns
    -------
    w_n : (N+1,) array_like
        Modal weighting factors, adjusted for unit amplitude.

    Examples
    --------
    See :py:func:`spaudiopy.sph.max_rE_weights`.

    """
    w_n = utils.asarray_1d(w_n)
    a_n = 0
    for n, w in enumerate(w_n):
        a_n += (2*n + 1) / (4 * np.pi) * w
    return w_n / a_n


def hypercardioid_modal_weights(N_sph):
    """Modal weights for beamformer resulting in a hyper-cardioid.

    Parameters
    ----------
    N_sph : int
        SH order.

    Returns
    -------
    w_n : (N+1,) array_like
        Modal weighting factors.

    Notes
    -----
    Also called max-DI or normalized PWD.

    Examples
    --------
    .. plot::
        :context: close-figs

        N = 5
        w_n = spa.sph.hypercardioid_modal_weights(N)
        w_nm = spa.sph.repeat_per_order(w_n) * \
            spa.sph.sh_matrix(N, np.pi/4, np.pi/4, 'real')
        spa.plots.sh_coeffs(w_nm)

    """
    c_n = np.repeat((4*np.pi)/(N_sph+1)**2, N_sph+1)
    return c_n


def cardioid_modal_weights(N_sph):
    """Modal weights for beamformer resulting in a cardioid.

    Parameters
    ----------
    N_sph : int
        SH order.

    Returns
    -------
    w_n : (N+1,) array_like
        Modal weighting factors.

    Examples
    --------
    .. plot::
        :context: close-figs

        N = 5
        w_n = spa.sph.cardioid_modal_weights(N)
        w_nm = spa.sph.repeat_per_order(w_n) * \
            spa.sph.sh_matrix(N, np.pi/4, np.pi/4, 'real')
        spa.plots.sh_coeffs(w_nm)

    """
    c_n = np.array([(np.math.factorial(N_sph)*np.math.factorial(N_sph)) /
                    (np.math.factorial(N_sph+n+1) * np.math.factorial(N_sph-n))
                    for n in range(N_sph+1)])
    # Note: 4pi to compensate for correct normalization, unit amplitude
    return 4*np.pi*c_n


def maxre_modal_weights(N_sph, UNITAMP=True):
    """Modal weights for beamformer resulting with max-rE weighting.

    Parameters
    ----------
    N_sph : int
        SH order.
    UNITAMP : bool, optional (default:True)

    Returns
    -------
    w_n : (N+1,) array_like
        Modal weighting factors.

    Notes
    -----
    Can be compensated for unit amplitude.

    Examples
    --------
    .. plot::
        :context: close-figs

        N = 5
        w_n = spa.sph.maxre_modal_weights(N)
        w_nm = spa.sph.repeat_per_order(w_n) * \
            spa.sph.sh_matrix(N, np.pi/4, np.pi/4, 'real')
        spa.plots.sh_coeffs(w_nm)

    """
    c_n = max_rE_weights(N_sph)
    # This is an iSHT in the same direction as unit PW
    a = bandlimited_dirac(N_sph, 0, c_n) if UNITAMP else 1
    return c_n/a


def butterworth_modal_weights(N_sph, k, n_c, UNITAMP=True):
    """Modal weights for spatial butterworth filter / beamformer.

    Parameters
    ----------
    N_sph : int
        SH order.
    k : int (float)
        Filter order
    n_c : int (float)
        Cut-on SH order.
    UNITAMP : bool, optional (default:True)

    Returns
    -------
    w_n : (N+1,) array_like
        Modal weighting factors.

    Notes
    -----
    Can be compensated for unit amplitude.

    References
    ----------
    Devaraju, B. (2015). Understanding filtering on the sphere.

    Examples
    --------
    .. plot::
        :context: close-figs

        N = 5
        w_n = spa.sph.butterworth_modal_weights(N, 5, 3)
        w_nm = spa.sph.repeat_per_order(w_n) * \
            spa.sph.sh_matrix(N, np.pi/4, np.pi/4, 'real')
        spa.plots.sh_coeffs(w_nm)

    """
    c_n = 1/np.sqrt(1+(np.arange(N_sph+1) / n_c)**(2*k))
    # This is an iSHT in the same direction as unit PW
    a = bandlimited_dirac(N_sph, 0, c_n) if UNITAMP else 1
    return c_n/a


def spat_filterbank_reconstruction_factor(w_nm, num_secs, mode=None):
    """Reconstruction factor for restoring amplitude/energy preservation.


    Parameters
    ----------
    w_nm : ((N+1)**2,), array_like
        SH beam coefficients.
    num_secs : int
        Number of spatial filters.
    mode : 'amplitude' or 'energy'

    Raises
    ------
    ValueError
        If mode is not specified.

    Returns
    -------
    beta : float
        Reconstruction factor.

    References
    ----------
    Hold, C., Politis, A., Mc Cormack, L., & Pulkki, V. (2021).
    Spatial Filter Bank Design in the Spherical Harmonic Domain. EUSIPCO 2021.

    """
    w_nm = np.atleast_2d(w_nm)
    assert(mode)
    if mode.lower() in ['amplitude', 'amp']:
        beta = np.sqrt(4*np.pi) / (w_nm[0, 0] * num_secs)
    elif mode.lower() in ['energy', 'en']:
        beta = (4*np.pi) / (w_nm[0, :].conj()@w_nm[0, :] * num_secs)
    else:
        raise ValueError("Mode not implemented: " + mode)
    return beta


def design_spat_filterbank(N_sph, sec_azi, sec_zen, c_n, SH_type, mode):
    """Design spatial filter bank analysis and reconstruction matrix.

    Parameters
    ----------
    N_sph : int
        SH order.
    sec_azi : (J,) array_like
        Sector azimuth steering directions.
    sec_zen : (J,) array_like
        Sector zenith/colatitude steering directions.
    c_n : (N,) array_like
        SH Modal weights, describing (axisymmetric) pattern.
    SH_type : 'real' or 'complex'
    mode : 'perfect' or 'energy'
        Design achieves perfect reconstruction or energy reconstruction.

    Raises
    ------
    ValueError
        If mode not specified.

    Returns
    -------
    A : (J, (N+1)**2) numpy.ndarray
        Analysis matrix.
    B : (J, (N+1)**2) numpy.ndarray
        Resynthesis matrix.

    References
    ----------
    Hold, C., Schlecht, S. J., Politis, A., & Pulkki, V. (2021). 
    Spatial Filter Bank in the Spherical Harmonic Domain : 
    Reconstruction and Application. WASPAA 2021.

    Examples
    --------
    .. plot::
        :context: close-figs

        N_sph = 3
        sec_dirs = spa.utils.cart2sph(*spa.grids.load_t_design(2*N_sph).T)
        c_n = spa.sph.maxre_modal_weights(N_sph)
        [A, B] = spa.sph.design_spat_filterbank(N_sph, sec_dirs[0], sec_dirs[1],
                                                c_n, 'real', 'perfect')
        # diffuse input SH signal
        in_nm = np.random.randn((N_sph+1)**2, 1000)
        # Sector signals (Analysis)
        s_sec = A @ in_nm
        # Reconstruction to SH domain
        out_nm = B.conj().T @ s_sec

        # Test perfect reconstruction
        print(spa.utils.test_diff(in_nm, out_nm))

    """
    sec_azi = utils.asarray_1d(sec_azi)
    sec_zen = utils.asarray_1d(sec_zen)
    c_n = utils.asarray_1d(c_n)
    num_secs = len(sec_azi)

    # Analysis matrix
    A = repeat_per_order(c_n) * sh_matrix(N_sph, sec_azi, sec_zen, SH_type)

    # Preservation property
    if mode.lower() == 'perfect':
        pres = 'amplitude'
    elif mode.lower() == 'energy':
        pres = 'energy'
    else:
        raise ValueError("Mode not implemented: " + mode)

    beta = spat_filterbank_reconstruction_factor(A[0, :], num_secs, mode=pres)

    # Reconstruction matrix
    if mode.lower() == 'perfect':
        B = beta * repeat_per_order(1/(c_n/c_n[0])) * \
                       sh_matrix(N_sph, sec_azi, sec_zen, SH_type)
    elif mode.lower() == 'energy':
        B = np.sqrt(beta) * repeat_per_order(1/(c_n/c_n[0])) * \
                                sh_matrix(N_sph, sec_azi, sec_zen, SH_type)
    else:
        raise ValueError("Mode not implemented: " + mode)

    return A, B
