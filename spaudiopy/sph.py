# -*- coding: utf-8 -*-

import numpy as np
from scipy import special as scyspecial
from scipy.linalg import lstsq

from . import utils


def SH_matrix(N, azi, colat, SH_type='complex', weights=None):
    r"""Matrix of spherical harmonics up to order N for given angles.

    Computes a matrix of spherical harmonics up to order :math:`N`
    for the given angles/grid.

    .. math::

        \mathbf{Y} = \left[ \begin{array}{cccccc}
        Y_0^0(\theta[0], \phi[0]) & Y_1^{-1}(\theta[0], \phi[0]) & Y_1^0(\theta[0], \phi[0]) & Y_1^1(\theta[0], \phi[0]) & \dots & Y_N^N(\theta[0], \phi[0])  \\
        Y_0^0(\theta[1], \phi[1]) & Y_1^{-1}(\theta[1], \phi[1]) & Y_1^0(\theta[1], \phi[1]) & Y_1^1(\theta[1], \phi[1]) & \dots & Y_N^N(\theta[1], \phi[1])  \\
        \vdots & \vdots & \vdots & \vdots & \vdots & \vdots \\
        Y_0^0(\theta[Q-1], \phi[Q-1]) & Y_1^{-1}(\theta[Q-1], \phi[Q-1]) & Y_1^0(\theta[Q-1], \phi[Q-1]) & Y_1^1(\theta[Q-1], \phi[Q-1]) & \dots & Y_N^N(\theta[Q-1], \phi[Q-1])
        \end{array} \right]

    where

    .. math::

        Y_n^m(\theta, \phi) = \sqrt{\frac{2n + 1}{4 \pi} \frac{(n-m)!}{(n+m)!}} P_n^m(\cos \theta) e^{i m \phi}


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

    """
    azi = utils.asarray_1d(azi)
    colat = utils.asarray_1d(colat)
    if azi.ndim == 0:
        Q = 1
    else:
        Q = len(azi)
    if weights is None:
        weights = np.ones(Q)
    if SH_type is 'complex':
        Ymn = np.zeros([Q, (N+1)**2], dtype=complex)
    elif SH_type is 'real':
        Ymn = np.zeros([Q, (N+1)**2], dtype=float)
    else:
        raise ValueError('SH_type unknown.')

    i = 0
    for n in range(N+1):
        for m in range(-n, n+1):
            if SH_type is 'complex':
                Ymn[:, i] = weights * scyspecial.sph_harm(m, n, azi, colat)
            elif SH_type is 'real':
                if m == 0:
                    Ymn[:, i] = weights * np.real(
                            scyspecial.sph_harm(m, n, azi, colat))
                if m < 0:
                    Ymn[:, i] = weights * np.sqrt(2) * (-1) ** m * np.imag(
                            scyspecial.sph_harm(np.abs(m), n, azi, colat))
                if m > 0:
                    Ymn[:, i] = weights * np.sqrt(2) * (-1) ** m * np.real(
                            scyspecial.sph_harm(np.abs(m), n, azi, colat))

            i += 1
    return Ymn


def SHT(f, N, azi, colat, SH_type, weights=None, Y_nm=None):
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
    if Y_nm is None:
        Y_nm = SH_matrix(N, azi, colat, SH_type)
    if weights is None:
        Npoints = len(azi)
        Y_nm_transform = (4*np.pi / Npoints) * Y_nm.conj()
    else:
        Y_nm_transform = Y_nm.conj()
        # weights should sum to 4pi
        f = np.dot(np.diag(weights), f)
    return np.matmul(Y_nm_transform.T, f)


def SHT_lstsq(f, N, azi, colat, SH_type, Y_nm=None):
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
    if Y_nm is None:
        Y_nm = SH_matrix(N, azi, colat, SH_type)
    return lstsq(Y_nm, f)[0]


def inverseSHT(F_nm, azi, colat, SH_type, Y_nm=None):
    """Perform the inverse spherical harmonics transform.

    Parameters
    ----------
    F_nm : ((N+1)**2, S) numpy.ndarray
        Matrix of spherical harmonics coefficients of spherical function(S).
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
    ----------
    f : (Q, S)
        The spherical function(S) evaluated at Q directions 'azi/colat'.
    """
    if Y_nm is None:
        N = int(np.sqrt(F_nm.shape[0]) - 1)
        Y_nm = SH_matrix(N, azi, colat, SH_type)
    # perform the inverse transform up to degree N
    return np.matmul(Y_nm, F_nm)


def platonic_solid(shape):
    """Returns coordinates of shape='tetrahedron' only, yet."""
    if shape in ['tetra', 'tetrahedron']:
        u = np.array([[1, 1, 1],
                      [1, -1, -1],
                      [-1, 1, -1],
                      [-1, -1, 1]]/np.sqrt(3))
    return u


def SH_to_B(F_nm, W_weight=None):
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
    # Without np.sqrt(4*np.pi)
    M = np.array([[W_weight, 0, 0, 0],
                  [0, 0, 0, 1/np.sqrt(3)],
                  [0, 1/np.sqrt(3), 0, 0],
                  [0, 0, 1/np.sqrt(3), 0]])
    return np.apply_along_axis(np.dot, 0, F_nm, M.T)


def B_to_SH(B, W_weight=None):
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
    # Without np.sqrt(4*np.pi)
    M = np.array([[W_weight, 0, 0, 0],
                  [0, 0, 0, 1/np.sqrt(3)],
                  [0, 1/np.sqrt(3), 0, 0],
                  [0, 0, 1/np.sqrt(3), 0]])
    M_inv = np.linalg.inv(M)
    return np.apply_along_axis(np.dot, 0, B, M_inv.T)


def soundfield_to_B(sig, W_weight=None):
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
    F_nm = SHT(sig, N, azi=t_az, colat=t_colat, SH_type='real')
    return SH_to_B(F_nm, W_weight)


def src_to_B(signal, src_azi, src_colat):
    """Get B format signal channels for source in direction azi/colat."""
    gw = 1
    gx, gy, gz = utils.sph2cart(src_azi, src_colat)
    g = np.array([gw, gx, gy, gz])
    return np.outer(g, signal)


def check_cond_SHT(N, azi, colat, SH_type, lim=None):
    """Check if condition number for a least-squares SHT is greater 'lim'."""
    A = SH_matrix(N, azi, colat, SH_type)
    c = np.zeros(N + 1)
    for iN in range(N + 1):
        # get coeffs up to iter order iN
        Y = A[:, :(iN + 1)**2]
        YYn = np.matmul(Y.conj().T, Y)
        c[iN] = np.linalg.cond(YYn)
    if lim is None:
        lim = N + N / 2
    if np.any(c > lim):
        print("High condition number! " + str(c))
    return c


def bandlimited_dirac(N, d, a_n=None):
    """Order N spatially bandlimited Dirac pulse at angular distance d.
    Zotter, F., & Frank, M. (2012). All-Round Ambisonic Panning and Decoding.
    Journal of Audio Engineering Society, eq. (7).
    """
    if a_n is None:
        a_n = np.ones(N + 1)
    g_n = np.zeros([(N + 1)**2, len(d)])
    for n, i in enumerate(range(N + 1)):
        g_n[i, :] = a_n[i] * (2 * n + 1) / (4 * np.pi) * \
                    scyspecial.eval_legendre(n, np.cos(d))
    s = np.sum(g_n, 0)
    return s


def max_rE_weights(N):
    """Return max rE window coefficients for order N.
    Zotter, F., & Frank, M. (2012). All-Round Ambisonic Panning and Decoding.
    Journal of Audio Engineering Society, eq. (10).
    """
    a_n = np.zeros(N + 1)
    for n, i in enumerate(range(N + 1)):
        a_n[i] = scyspecial.eval_legendre(n, np.cos(np.deg2rad(137.9 /
                                                               (N + 1.51))))
    return a_n


def r_E(p, g):
    """r_E vector and magnitude calculated from loudspeaker position vector p
    and their gains g.
    Zotter, F., & Frank, M. (2012). All-Round Ambisonic Panning and Decoding.
    Journal of Audio Engineering Society, eq. (16).
    """
    p = np.atleast_2d(p)
    g = np.atleast_2d(g)
    assert(p.shape[0] == g.shape[1]), 'Provide gain per speaker!'
    E = np.sum(g**2, axis=1)
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


def repeat_order_coeffs(c):
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
    c_reshaped = np.zeros((N + 1) ** 2)
    idx = 0
    for n in range(N+1):
        for m in range(-n, n+1):
            c_reshaped[idx] = c[n]
            idx += 1
    return c_reshaped
