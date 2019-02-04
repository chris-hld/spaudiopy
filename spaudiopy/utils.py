# -*- coding: utf-8 -*-
"""
@author: chris

A few helpers.
"""
import numpy as np


def asarray_1d(a, **kwargs):
    """Squeeze the input and check if the result is one-dimensional.

    Returns *a* converted to a `numpy.ndarray` and stripped of
    all singleton dimensions.  Scalars are "upgraded" to 1D arrays.
    The result must have exactly one dimension.
    If not, an error is raised.
    """
    result = np.squeeze(np.asarray(a, **kwargs))
    if result.ndim == 0:
        result = result.reshape((1,))
    elif result.ndim > 1:
        raise ValueError("array must be one-dimensional")
    return result


def deg2rad(deg):
    """Convert from degree [0, 360) to radiant [0, 2*pi)."""
    return deg % 360 / 180 * np.pi


def rad2deg(rad):
    """Convert from radiant [0, 2*pi) to degree [0, 360)."""
    return rad / np.pi * 180 % 360


def cart2sph(x, y, z):
    """Vectorized conversion of cartesian to spherical coordinates."""
    x = asarray_1d(x)
    y = asarray_1d(y)
    z = asarray_1d(z)
    r = np.sqrt(x**2 + y**2 + z**2)
    azi = np.arctan2(y, x)
    azi = azi % (2 * np.pi)  # [0, 2pi)
    colat = np.arccos(z / r)
    return azi, colat, r


def sph2cart(azi, colat, r=1):
    """Vectorized conversion of spherical to cartesian coordinates."""
    azi = asarray_1d(azi)
    colat = asarray_1d(colat)
    r = asarray_1d(r)
    x = r * np.cos(azi) * np.sin(colat)
    y = r * np.sin(azi) * np.sin(colat)
    z = r * np.cos(colat)
    return x, y, z


def matlab_sph2cart(az, elev, r):
    """Matlab port with ELEVATION."""
    z = r * np.sin(elev)
    rcoselev = r * np.cos(elev)
    x = rcoselev * np.cos(az)
    y = rcoselev * np.sin(az)
    return x, y, z


def vecs2dirs(vecs):
    """Helper to convert [x, y, z] to [phi ,theta]."""
    phi, theta, _ = cart2sph(vecs[:, 0], vecs[:, 1], vecs[:, 2])
    return np.c_[phi, theta]


def angle_between(v1, v2, vi=None):
    """Angle between point v1 and v2(s) with initial point vi."""
    v1 = asarray_1d(v1)
    if vi is not None:
        v1 = v1 - vi
        v2 = v2 - vi

    a = np.dot(v1, v2.T) / (np.linalg.norm(v1.T, axis=0) *
                            np.linalg.norm(v2.T, axis=0))
    return np.arccos(np.clip(a, -1.0, 1.0))


def area_triangle(p1, p2, p3):
    """calculate area of any triangle given coordinates of its corners p."""
    return 0.5 * np.linalg.norm(np.cross((p2 - p1), (p3 - p1)))


def dB(data, power=False):
    """Calculate the 20*log10(abs(x)).

    Parameters
    ----------
    data : array_like
       signals to be converted to db
    power : boolean
       data is a power signal and only needs factor 10

    Returns
    -------
    db : array_like
       (20 or 10) * log10(abs(data))

    """
    if power:
        factor = 10
    else:
        factor = 20
    return factor * np.log10(np.abs(data))


def stack(vector_1, vector_2):
    """Stack two 2D vectors along the same-sized or the smaller dimension."""
    vector_1, vector_2 = np.atleast_2d(vector_1, vector_2)
    M1, N1 = vector_1.shape
    M2, N2 = vector_2.shape

    if (M1 == M2 and (M1 < N1 or M2 < N2)):
        out = np.vstack([vector_1, vector_2])
    elif (N1 == N2 and (N1 < M1 or N2 < M2)):
        out = np.hstack([vector_1, vector_2])
    else:
        raise ValueError('vector_1 and vector_2 dont have a common dimension.')
    return np.squeeze(out)


def test_diff(v1, v2, VERBOSE=True):
    """Test if the absolute difference between v1 and v2 is greater 10-e8."""
    d = np.sum(np.abs(v1 - v2))
    if VERBOSE:
        if np.any(d > 10e-8):
            print('Diff: ', d)
        else:
            print('Close enough')
    return d
