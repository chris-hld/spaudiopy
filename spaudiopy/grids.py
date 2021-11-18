# -*- coding: utf-8 -*-
"""Sampling grids.

.. plot::
    :context: reset

    import numpy as np
    import matplotlib.pyplot as plt
    plt.rcParams['axes.grid'] = True

    import spaudiopy as spa

"""

import os
import numpy as np
from warnings import warn
from scipy.io import loadmat
from . import utils


def load_t_design(degree):
    """Return the unit coordinates of minimal T-designs.

    The designs have been copied from:
    http://neilsloane.com/sphdesigns/
    and should be referenced as:

        "McLaren's Improved Snub Cube and Other New Spherical Designs in
        Three Dimensions", R. H. Hardin and N. J. A. Sloane, Discrete and
        Computational Geometry, 15 (1996), pp. 429-441.

    Parameters
    ----------
    degree : int
        T-design degree between 1 and 21.

    Returns
    -------
    vecs : (M, 3) numpy.ndarray
        Coordinates of points.

    Notes
    -----
    Degree must be >= 2 * SH_order for spherical harmonic transform (SHT).

    Examples
    --------
    .. plot::
        :context: close-figs

        vecs = spa.grids.load_t_design(degree=2*5)
        spa.plots.hull(spa.decoder.get_hull(*vecs.T))

    """
    if degree > 21:
        raise ValueError('Designs of order > 21 are not implemented.')
    elif degree < 1:
        raise ValueError('Order should be at least 1.')
    # extract
    current_file_dir = os.path.dirname(__file__)
    file_path = os.path.join(current_file_dir,
                             '../data/Grids/t_designs_1_21.mat')
    mat = loadmat(file_path)
    t_designs_obj = mat['t_designs']
    t_designs = t_designs_obj[0].tolist()
    # degree t>=2N should be used for SHT
    vecs = t_designs[degree - 1]
    return vecs


def load_n_design(degree):
    """Return the unit coordinates of spherical N-design
    (Chebyshev-type quadrature rules). Seem to be equivalent but more
    modern t-designs.

    The designs have been copied from:
    https://homepage.univie.ac.at/manuel.graef/quadrature.php

    Parameters
    ----------
    degree : int
       Degree of exactness N between 1 and 124.

    Returns
    -------
    vecs : (M, 3) numpy.ndarray
        Coordinates of points.

    Examples
    --------
    .. plot::
        :context: close-figs

        vecs = spa.grids.load_n_design(degree=2*5)
        spa.plots.hull(spa.decoder.get_hull(*vecs.T))

    """
    if degree > 124:
        raise ValueError('Designs of order > 124 are not implemented.')
    elif degree < 1:
        raise ValueError('Order should be at least 1.')
    # extract
    current_file_dir = os.path.dirname(__file__)
    file_path = os.path.join(current_file_dir,
                             '../data/Grids/n_designs_1_124.mat')
    mat = loadmat(file_path)
    try:
        n_design = mat['N' + f'{degree:03}']
    except KeyError:
        warn(f"Degree {degree} not defined, trying {degree+1} ...")
        n_design = load_n_design(degree + 1)
    return n_design


def load_lebedev(degree):
    """Return the unit coordinates of Lebedev grid.

    The designs have been copied from:
    https://people.sc.fsu.edu/~jburkardt/datasets/sphere_lebedev_rule/sphere_lebedev_rule.html

    Parameters
    ----------
    degree : int
       Degree of precision p between 3 and 131.

    Returns
    -------
    vecs : (M, 3) numpy.ndarray
        Coordinates of points.
    weights : array_like
        Quadrature weights.

    Examples
    --------
    .. plot::
        :context: close-figs

        vecs, weights = spa.grids.load_lebedev(degree=2*5)
        spa.plots.hull(spa.decoder.get_hull(*vecs.T))

    """
    if degree > 131:
        raise ValueError('Designs of order > 131 are not implemented.')
    elif degree < 3:
        raise ValueError('Order should be at least 3.')
    # extract
    current_file_dir = os.path.dirname(__file__)
    file_path = os.path.join(current_file_dir,
                             '../data/Grids/lebedevQuadratures_3_131.mat')
    mat = loadmat(file_path)
    try:
        design = mat['lebedev_' + f'{degree:03}']
        vecs = design[:, :3]
        weights = 4*np.pi * design[:, 3]
        if np.any(weights < 0):
            warn(f"Lebedev grid {degree} has negative weights.")
    except KeyError:
        warn(f"Degree {degree} not defined, trying {degree+1} ...")
        vecs, weights = load_lebedev(degree + 1)
    return vecs, weights


def load_Fliege_Maier_nodes(grid_order):
    """Return Fliege-Maier grid nodes with associated weights.

    The designs have been copied from:
    http://www.personal.soton.ac.uk/jf1w07/nodes/nodes.html
    and should be referenced as:

        "A two-stage approach for computing cubature formulae for the sphere.",
        Jorg Fliege and Ulrike Maier, Mathematik 139T, Universitat Dortmund,
        Fachbereich Mathematik, Universitat Dortmund, 44221. 1996.

    Parameters
    ----------
    grid_order : int
        Grid order between 2 and 30

    Returns
    -------
    vecs : (M, 3) numpy.ndarray
        Coordinates of points.
    weights : array_like
        Quadrature weights.

    Examples
    --------
    .. plot::
        :context: close-figs

        vecs, weights = spa.grids.load_Fliege_Maier_nodes(grid_order=5)
        spa.plots.hull(spa.decoder.get_hull(*vecs.T))

    """
    if grid_order > 30:
        raise ValueError('Designs of order > 30 are not implemented.')
    elif grid_order < 2:
        raise ValueError('Order should be at least 2.')
    # extract
    current_file_dir = os.path.dirname(__file__)
    file_path = os.path.join(current_file_dir,
                             '../data/Grids/fliegeMaierNodes_1_30.mat')
    mat = loadmat(file_path)
    fliege_maier_nodes = np.squeeze(mat['fliegeNodes'])
    # grid_order >= N+1 should be used for SHT
    vecs = fliege_maier_nodes[grid_order - 1][:, :-1]
    weights = fliege_maier_nodes[grid_order - 1][:, -1]  # sum(weights) == 4pi
    return vecs, weights


def load_maxDet(degree):
    """Return Maximum Determinant (Fekete, Extremal) points on the sphere.

    The designs have been copied from:
    https://web.maths.unsw.edu.au/~rsw/Sphere/MaxDet/

    Parameters
    ----------
    degree : int
       Degree between 1 and 200.

    Returns
    -------
    vecs : (M, 3) numpy.ndarray
        Coordinates of points.
    weights : array_like
        Quadrature weights.

    Examples
    --------
    .. plot::
        :context: close-figs

        vecs, weights = spa.grids.load_maxDet(degree=5)
        spa.plots.hull(spa.decoder.get_hull(*vecs.T))

    """
    if degree > 200:
        raise ValueError('Designs of order > 200 are not implemented.')
    elif degree < 1:
        raise ValueError('Order should be at least 1.')
    # extract
    current_file_dir = os.path.dirname(__file__)
    file_path = os.path.join(current_file_dir,
                             '../data/Grids/maxDetPoints_1_200.mat')
    mat = loadmat(file_path)
    try:
        design = mat['maxDet_' + f'{degree:03}']
        vecs = design[:, :3]
        weights = design[:, 3]
        if np.any(weights < 0):
            warn(f"Grid {degree} has negative weights.")
    except KeyError:
        warn(f"Degree {degree} not defined, trying {degree+1} ...")
        vecs, weights = load_maxDet(degree + 1)
    return vecs, weights


def equal_angle(n):
    """Equi-angular sampling points on a sphere.

    According to (cf. Rafaely book, sec.3.2)

    Parameters
    ----------
    n : int
        Maximum order.

    Returns
    -------
    azi : array_like
        Azimuth.
    colat : array_like
        Colatitude.
    weights : array_like
        Quadrature weights.

    Examples
    --------
    .. plot::
        :context: close-figs

        azi, colat, weights = spa.grids.equal_angle(n=5)
        spa.plots.hull(spa.decoder.get_hull(*spa.utils.sph2cart(azi, colat)))

    """
    azi = np.linspace(0, 2*np.pi, 2*n+2, endpoint=False)
    colat, d_colat = np.linspace(0, np.pi, 2*n+2, endpoint=False, retstep=True)
    colat += d_colat/2

    weights = np.zeros_like(colat)
    p = np.arange(1, 2*n+2, 2)
    for i, theta in enumerate(colat):
        weights[i] = 2*np.pi/(n+1) * np.sin(theta) * np.sum(np.sin(p*theta)/p)

    azi = np.tile(azi, 2*n+2)
    colat = np.repeat(colat, 2*n+2)
    weights = np.repeat(weights, 2*n+2)
    weights /= n+1     # sum(weights) == 4pi
    return azi, colat, weights


def gauss(n):
    """Gauss-Legendre sampling points on sphere.

    According to (cf. Rafaely book, sec.3.3)

    Parameters
    ----------
    n : int
        Maximum order.

    Returns
    -------
    azi : array_like
        Azimuth.
    colat : array_like
        Colatitude.
    weights : array_like
        Quadrature weights.

    Examples
    --------
    .. plot::
        :context: close-figs

        azi, colat, weights = spa.grids.gauss(n=5)
        spa.plots.hull(spa.decoder.get_hull(*spa.utils.sph2cart(azi, colat)))

    """
    azi = np.linspace(0, 2*np.pi, 2*n+2, endpoint=False)
    x, weights = np.polynomial.legendre.leggauss(n+1)
    colat = np.arccos(x)
    azi = np.tile(azi, n+1)
    colat = np.repeat(colat, 2*n+2)
    weights = np.repeat(weights, 2*n+2)
    weights *= np.pi / (n+1)      # sum(weights) == 4pi
    return azi, colat, weights


def equal_polar_angle(n):
    """Equi-angular sampling points on a circle.

    Parameters
    ----------
    n : int
        Maximum order

    Returns
    -------
    pol : array_like
        Polar angle.
    weights : array_like
        Weights.
    """
    num_mic = 2*n+1
    pol = np.linspace(0, 2*np.pi, num=num_mic, endpoint=False)
    weights = 1/num_mic * np.ones(num_mic)
    return pol, weights
