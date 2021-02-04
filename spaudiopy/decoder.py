# -*- coding: utf-8 -*-
"""Loudspeaker decoders.

.. plot::
    :context: reset

    import numpy as np
    import matplotlib.pyplot as plt
    plt.rcParams['axes.grid'] = True

    import spaudiopy as spa

    # Loudspeaker Setup
    ls_dirs = np.array([[-80, -45, 0, 45, 80, -60, -30, 30, 60],
                        [0, 0, 0, 0, 0, 60, 60, 60, 60]])
    ls_x, ls_y, ls_z = spa.utils.sph2cart(spa.utils.deg2rad(ls_dirs[0, :]),
                                          spa.utils.deg2rad(90 - ls_dirs[1, :]))

"""

import copy
import multiprocessing
from itertools import repeat
from warnings import warn

import numpy as np
import scipy.spatial as scyspat
from scipy import signal

from spaudiopy import utils, sph, IO, plots, grids

shared_array = None


class LoudspeakerSetup:
    """Creates a 'hull' object containing all information for further decoding.

    .. plot::
        :context: close-figs

        ls_setup = spa.decoder.LoudspeakerSetup(ls_x, ls_y, ls_z)
        ls_setup.pop_triangles(normal_limit=85, aperture_limit=90,
                               opening_limit=150)
        ls_setup.show()

    """

    def __init__(self, x, y, z, listener_position=None):
        """
        Parameters
        ----------
        x : array_like
        y : array_like
        z : array_like
        listener_position : (3,), cartesian, optional
            Offset, will be substracted from the loudspeaker positions.

        """
        self.x = utils.asarray_1d(x)
        self.y = utils.asarray_1d(y)
        self.z = utils.asarray_1d(z)
        if listener_position is None:
            listener_position = [0, 0, 0]
        self.listener_position = utils.asarray_1d(listener_position)

        # Listener position as origin
        self.x -= listener_position[0]
        self.y -= listener_position[1]
        self.z -= listener_position[2]
        # TODO: Better handling of this, e.g. not effective when updating hull:
        self.listener_position -= self.listener_position
        _, _, self.d = utils.cart2sph(self.x, self.y, self.z)

        # amplitude decay exponent
        self.a = 1

        # Triangulation of points
        _hull = get_hull(self.x, self.y, self.z)
        self.points = _hull.points
        self.npoints = _hull.npoints
        self.nsimplex = _hull.nsimplex
        self.vertices = _hull.vertices
        self.simplices = _hull.simplices
        self.simplices = sort_vertices(self.simplices)
        self.centroids = calculate_centroids(self)
        self.face_areas = calculate_face_areas(self)
        self.face_normals = calculate_face_normals(self)
        self.vertex_normals = calculate_vertex_normals(self)
        self.barycenter = calculate_barycenter(self)

        # All simplices enclosing listener valid per default for rendering
        self._encloses_listener = check_listener_inside(self,
                                                        self.listener_position)
        self.valid_simplices = self._encloses_listener

        # see 'ambisonics_setup()'
        self.ambisonics_hull = []
        self.kernel_hull = []
        self.characteristic_order = None

        # some checks
        assert(len(self.d) == self.npoints)

    @classmethod
    def from_sph(cls, azi, colat, r=1, listener_position=None):
        """ Alternative constructor, using spherical coordinates in rad.

        Parameters
        ----------
        azi : array_like, spherical
        colat : array_like, spherical
        r : array_like, spherical
        listener_position : (azi, colat, r), spherical, optional
            Offset, will be substracted from the loudspeaker positions.

        """
        x, y, z = utils.sph2cart(azi, colat, r)
        if listener_position is None:
            listener_position = [0, 0, 0]
        listener_position = utils.asarray_1d(listener_position)
        listener_position = utils.sph2cart(*listener_position)
        return cls(x, y, z, listener_position=listener_position)

    def is_simplex_valid(self, simplex):
        """Tests if simplex is in valid simplices (independent of orientation).
        """
        # find face in all faces
        in_s = np.isin(self.valid_simplices, simplex).sum(axis=-1) == 3
        return np.any(in_s)

    def pop_triangles(self, normal_limit=85, aperture_limit=None,
                      opening_limit=None, blacklist=None):
        """Refine triangulation by removing them from valid simplices.
        Bypass by passing 'None'.

        Parameters
        ----------
        normal_limit : float, optional
        aperture_limit : float, optional
        opening_limit : float, optional
        blacklist : list, optional

        """
        if normal_limit is not None:
            self.valid_simplices = check_normals(self, normal_limit)
        if aperture_limit is not None:
            self.valid_simplices = check_aperture(self, aperture_limit)
        if opening_limit is not None:
            self.valid_simplices = check_opening(self, opening_limit)
        if blacklist is not None:
            self.valid_simplices = apply_blacklist(self, blacklist)

    def get_characteristic_order(self):
        """Characteristic Ambisonics order."""
        if self.characteristic_order is None:
            N_e = characteristic_ambisonic_order(self)
            if N_e < 1:
                raise ValueError
        else:
            N_e = self.characteristic_order
        return N_e

    def ambisonics_setup(self, N_kernel=50, update_hull=False,
                         imaginary_ls=None):
        """Prepare loudspeaker hull for ambisonic rendering.
        Sets the `kernel_hull` as an n-design of twice `N_kernel`,
        and updates the ambisonic hull with an additional imaginary loudspeaker,
        if desired.

        Parameters
        ----------
        N_kernel : int, optional
        update_hull : bool, optional
        imaginary_ls : (L, 3), cartesian, optional
            Imaginary loudspeaker positions, if set to 'None' calls
            'find_imaginary_loudspeaker()' for 'update_hull'.

        Examples
        --------
        .. plot::
            :context: close-figs

            ls_setup.ambisonics_setup(update_hull=True)
            N_e = ls_setup.characteristic_order
            ls_setup.ambisonics_hull.show(title=f"Ambisonic Hull, $N_e={N_e}$")

        """
        self.characteristic_order = self.get_characteristic_order()
        if N_kernel is None:
            print('Setting Ambisonics order =', self.characteristic_order)
            N_kernel = self.characteristic_order
        if(not update_hull and imaginary_ls is not None):
            warn('Not updating hull but imaginary_ls position given.')

        ambi_ls = self.points
        if update_hull:
            if imaginary_ls is None:
                new_imaginary_ls = find_imaginary_loudspeaker(self)
                # add imaginary speaker to hull
                ambi_ls = np.vstack([ambi_ls, new_imaginary_ls])
                # mark imaginary speaker (last one)
                imaginary_ls_idx = ambi_ls.shape[0] - 1
            else:
                imaginary_ls = np.atleast_2d(imaginary_ls)
                assert(imaginary_ls.shape[1] == 3)
                # add imaginary loudspeaker(s) to hull
                ambi_ls = np.vstack([ambi_ls, imaginary_ls])
                # mark imaginary speaker (last one(s))
                imaginary_ls_idx = np.arange(ambi_ls.shape[0] -
                                             imaginary_ls.shape[0],
                                             ambi_ls.shape[0])
        else:
            imaginary_ls_idx = None

        # This new triangulation is now the rendering setup
        ambisonics_hull = LoudspeakerSetup(ambi_ls[:, 0],
                                           ambi_ls[:, 1],
                                           ambi_ls[:, 2])
        # mark imaginary speaker index
        ambisonics_hull.imaginary_ls_idx = imaginary_ls_idx
        # discretization hull
        virtual_speakers = grids.load_n_design(2 * N_kernel)
        # Avoid any extra calculation on this dense grid
        kernel_hull = get_hull(virtual_speakers[:, 0],
                               virtual_speakers[:, 1],
                               virtual_speakers[:, 2])

        del ambisonics_hull.ambisonics_hull
        del ambisonics_hull.kernel_hull
        self.ambisonics_hull = ambisonics_hull
        self.kernel_hull = kernel_hull
        self.kernel_hull.N_kernel = N_kernel

    def binauralize(self, ls_signals, fs, orientation=(0, 0), hrirs=None):
        """Create binaural signals that the loudspeaker signals produce on this
        setup (no delays).

        Parameters
        ----------
        ls_signals : (L, S) np.ndarray
            Loudspeaker signals.
        fs : int
        orientation : (azi, colat) tuple, optional
            Listener orientation offset (azimuth, colatitude) in rad.
        hrirs : sig.HRIRs, optional

        Returns
        -------
        l_sig : array_like
        r_sig : array_like

        """
        if hrirs is None:
            hrirs = IO.load_hrirs(fs)
        assert(hrirs.fs == fs)
        ls_signals = np.atleast_2d(ls_signals)
        assert ls_signals.shape[0] == self.npoints, \
            'Provide signal per loudspeaker!'
        # distance attenuation
        relative_position = self.points - \
                            self.listener_position
        ls_azi, ls_colat, ls_r = utils.cart2sph(*relative_position.T)
        ls_signals = np.diag(1 / ls_r ** self.a) @ ls_signals
        # convolve with hrir
        l_sig = np.zeros(ls_signals.shape[1] + len(hrirs) - 1)
        r_sig = np.zeros_like(l_sig)
        for ch, ls_sig in enumerate(ls_signals):
            if any(abs(ls_sig) > 10e-6):  # Gate at -100dB
                hrir_l, hrir_r = hrirs.nearest_hrirs(ls_azi[ch] -
                                                     orientation[0],
                                                     ls_colat[ch] -
                                                     orientation[1])
                # sum all loudspeakers
                l_sig += signal.convolve(ls_sig, hrir_l)
                r_sig += signal.convolve(ls_sig, hrir_r)
        return l_sig, r_sig

    def loudspeaker_signals(self, ls_gains, sig_in=None):
        """Render loudspeaker signals.

        Parameters
        ----------
        ls_gains : (S, L) np.ndarray
        sig_in : (S,) array like, optional

        Returns
        -------
        sig_out : (L, S) np.ndarray

        """
        ls_gains = np.atleast_2d(ls_gains)
        if sig_in is None:
            sig_in = np.ones(ls_gains.shape[0])
        sig_in = utils.asarray_1d(sig_in)
        assert(ls_gains.shape[1] == len(self.points)), \
            'Provide gain per speaker!'
        return (sig_in[:, np.newaxis] * ls_gains).T

    def show(self, title='Loudspeaker Setup', **kwargs):
        """Plot hull object, calls plots.hull()."""
        plots.hull(self, title=title, **kwargs)


def get_hull(x, y, z):
    """Wrapper for scipy.spatial.ConvexHull."""
    return scyspat.ConvexHull(np.c_[x, y, z], incremental=False)


def calculate_centroids(hull):
    """Calculate centroid for each simplex."""
    centroids = np.zeros((len(hull.simplices), 3))
    for face_i, face in enumerate(hull.simplices):
        # extract vertices face
        v = hull.points[face, :]
        centroids[face_i, :] = np.mean(v, axis=0)
    return centroids


def calculate_face_areas(hull):
    """Calculate area for each simplex."""
    face_areas = np.zeros(len(hull.simplices))
    for face_i, face in enumerate(hull.simplices):
        v = hull.points[face, :]
        face_areas[face_i] = utils.area_triangle(v[0, :], v[1, :], v[2, :])
    return face_areas


def calculate_face_normals(hull, eps=10e-6, normalize=False):
    """Calculate outwards pointing normal for each simplex."""
    face_normals = np.zeros((len(hull.simplices), 3))
    barycenter = np.mean(hull.points, axis=0)
    for face_i, face in enumerate(hull.simplices):
        # extract vertices face
        v = hull.points[face, :]
        centroid = np.mean(v, axis=0)
        # normal vector is cross product of two sides, initial point v0
        v_n = np.cross(v[1, :] - v[0, :], v[2, :] - v[0, :])
        # compare of face normal points in direction of barycenter
        criterion = np.dot(centroid + v_n - centroid, barycenter - centroid)
        # Make normal vector pointing outwards
        if criterion > eps:
            v_n = -v_n
        if normalize:
            v_n = v_n / np.linalg.norm(v_n)
        face_normals[face_i, :] = v_n
    return face_normals


def calculate_vertex_normals(hull, normalize=False):
    """Calculate normal for each vertex from simplices normals."""
    faces = hull.simplices
    vertex_normals = np.zeros([hull.npoints, 3])
    for p in hull.vertices:
        is_in_face = [p in row for row in faces]
        a = hull.face_areas[is_in_face]
        N = hull.face_normals[is_in_face]
        # weighted sum
        N_w = a[:, np.newaxis] * N
        vertex_n = np.sum(N_w, axis=0)
        if normalize:
            vertex_n = vertex_n / np.linalg.norm(vertex_n)
        vertex_normals[p, :] = vertex_n
    return vertex_normals


def calculate_barycenter(hull):
    """Barycenter of hull object."""
    return np.mean(hull.points, axis=0)


def check_listener_inside(hull, listener_position=None):
    """Return valid simplices for which the listener is inside the hull."""
    if listener_position is None:
        listener_position = hull.listener_position
    listener_position = np.asarray(listener_position)
    valid_simplices = []
    for face, centroid in zip(hull.simplices, hull.centroids):
        # centroid to listener
        v1 = listener_position - centroid
        # centroid to barycenter
        v2 = hull.barycenter - centroid
        # listener inside if both point in the same direction
        if np.dot(v1, v2) < 0:
            print("Listener not inside:", face)
        else:
            valid_simplices.append(face)
    return np.array(valid_simplices)


def check_normals(hull, normal_limit=85, listener_position=None):
    """Return valid simplices that point towards listener."""
    if listener_position is None:
        listener_position = hull.listener_position
    valid_simplices = []
    for face, v_n, c in zip(hull.simplices, hull.face_normals,
                            hull.centroids):
        if hull.is_simplex_valid(face):
            if utils.angle_between(c, v_n, vi=listener_position) > \
                    utils.deg2rad(normal_limit):
                print("Face not pointing towards listener: " + str(face))
            else:
                valid_simplices.append(face)
    return np.array(valid_simplices)


def check_aperture(hull, aperture_limit=90, listener_position=None):
    """Return valid simplices, where the aperture form the listener is small.
    """
    if listener_position is None:
        listener_position = hull.listener_position
    valid_simplices = []
    for face in hull.valid_simplices:
        # extract vertices face
        v = hull.points[face, :]
        a = utils.angle_between(v[0, :], v[1, :], vi=listener_position)
        b = utils.angle_between(v[1, :], v[2, :], vi=listener_position)
        c = utils.angle_between(v[0, :], v[2, :], vi=listener_position)
        if np.any(np.r_[a, b, c] > utils.deg2rad(aperture_limit)):
            print("Face produces critically large aperture: " + str(face))
        else:
            valid_simplices.append(face)
    return np.array(valid_simplices)


def check_opening(hull, opening_limit=135):
    """Return valid simplices with all opening angles within simplex > limit.
    """
    valid_simplices = []
    for face in hull.valid_simplices:
        # extract vertices face
        v = hull.points[face, :]
        a = utils.angle_between(v[1, :], v[2, :], vi=v[0, :])
        b = utils.angle_between(v[0, :], v[2, :], vi=v[1, :])
        c = utils.angle_between(v[0, :], v[1, :], vi=v[2, :])
        if np.any(np.r_[a, b, c] > utils.deg2rad(opening_limit)):
            print("Found large opening angle in face: " + str(face))
        else:
            valid_simplices.append(face)
    return np.array(valid_simplices)


def apply_blacklist(hull, blacklist=None):
    """Specify a blacklist to exclude simplices from valid simplices."""
    if blacklist is not None:
        valid_simplices = []
        for face in hull.valid_simplices:
            if not all(elem in face for elem in blacklist):
                valid_simplices.append(face)
            else:
                print("Blacklist face: " + str(face))
    return np.array(valid_simplices)


def sort_vertices(simplices):
    """Start the simplices with smallest vertex entry."""
    out = np.zeros_like(simplices)
    for i, face in enumerate(simplices):
        face = face[::-1]
        out[i, :] = np.roll(face, -np.argmin(face))
    return out


def find_imaginary_loudspeaker(hull):
    """Find imaginary loudspeaker coordinates for smoother hull.

    References
    ----------
    Zotter, F., & Frank, M. (2012). All-Round Ambisonic Panning and Decoding.
    Journal of Audio Engineering Society, Sec. 1.1.

    """
    # detect edges
    rim_edges = []
    for face in hull.valid_simplices:
        # extract the three edges
        edges = np.zeros([3, 2])
        edges[0, :] = face[[0, 1]]
        edges[1, :] = face[[1, 2]]
        edges[2, :] = face[[0, 2]]

        # detect if edge occurs once
        for edge in edges:
            occured_once = np.count_nonzero(np.isin(
                            hull.valid_simplices, edge).sum(axis=1) == 2) == 1
            if occured_once:
                rim_edges.append(edge)
    # Check that all rim vertices are connected
    unique, counts = np.unique(rim_edges, return_counts=True)
    if not (counts >= 2).all():
        raise NotImplementedError("More than one rim found.")
    if (counts == 3).all():
        raise RuntimeError("No rim detected. Consider not updating the hull.")

    # Zotter, F., & Frank, M. (2012). All-Round Ambisonic Panning and Decoding.
    # Journal of Audio Engineering Society, sec. 1.1
    avg_valid_n = np.zeros([1, 3])
    avg_d = 0
    for face in hull.valid_simplices:
        # find valid face in all faces
        mask = np.isin(hull.simplices, face).sum(axis=-1) == 3
        avg_valid_n += hull.face_areas[mask] * hull.face_normals[mask]
        avg_d += np.mean(hull.d[hull.simplices[mask]])
    # r**3 seems necessary
    imaginary_loudspeaker_coordinates = -avg_valid_n / \
                                        (avg_d / len(hull.valid_simplices))**3
    return imaginary_loudspeaker_coordinates


def _invert_triplets(simplices, points):
    """Invert loudspeaker triplets."""
    inverted_ls_triplets = []
    for face in simplices:
        # extract vertices face (valid LS positions)
        v = points[face, :]
        v_inv = np.linalg.lstsq(v, np.eye(3), rcond=None)[0]
        inverted_ls_triplets.append(v_inv.T)
    return inverted_ls_triplets


# part of parallel vbap:
def _vbap_gains_single_source(src_idx, src, inverted_ls_triplets,
                              valid_simplices):
    for face_idx, ls_base in enumerate(inverted_ls_triplets):
        # projecting src onto loudspeakers
        projection = np.dot(ls_base, src[src_idx, :])
        # normalization
        projection /= np.sqrt(np.sum(projection**2))
        if np.all(projection > -10e-6):
            assert(np.count_nonzero(projection) <= 3)
            # print(f"Source {src_idx}: Gains {projection}")
            shared_array[src_idx, valid_simplices[face_idx]] = projection
            break  # found valid gains


def vbap(src, hull, valid_simplices=None, retain_outside=False, jobs_count=1):
    """Loudspeaker gains for Vector Base Amplitude Panning decoding.

    Parameters
    ----------
    src : (n, 3) numpy.ndarray
        Cartesian coordinates of n sources to be rendered.
    hull : LoudspeakerSetup
    valid_simplices : (nsimplex, 3) numpy.ndarray
        Valid simplices employed for rendering, defaults hull.valid_simplices.
    retain_outside : bool, optional
        Render on the 'ambisonic hull' to fade out amplitude.
    jobs_count : int or None, optional
        Number of parallel jobs, 'None' employs 'cpu_count'.

    Returns
    -------
    gains : (n, L) numpy.ndarray
        Panning gains for L loudspeakers to render n sources.

    References
    ----------
    Pulkki, V. (1997). Virtual Sound Source Positioning Using Vector Base
    Amplitude Panning. AES, 144(5), 357–360.

    Examples
    --------
    .. plot::
        :context: close-figs

        ls_setup = spa.decoder.LoudspeakerSetup(ls_x, ls_y, ls_z)
        ls_setup.pop_triangles(normal_limit=85, aperture_limit=90,
                               opening_limit=150)

        spa.plots.decoder_performance(ls_setup, 'VBAP')

        ls_setup.ambisonics_setup(update_hull=True)
        spa.plots.decoder_performance(ls_setup, 'VBAP', retain_outside=True)
        plt.suptitle('VBAP with imaginary loudspeaker')

    """
    if jobs_count is None:
        jobs_count = multiprocessing.cpu_count()
    if retain_outside:
        assert(valid_simplices is None)
        if hull.ambisonics_hull:
            hull = hull.ambisonics_hull
            if hull.imaginary_ls_idx is None:
                raise ValueError('No imaginary loudspeaker. Update hull!')
        else:
            raise ValueError('Run LoudspeakerSetup.ambisonics_setup() first!')

    if valid_simplices is None:
        valid_simplices = hull.valid_simplices

    src = np.atleast_2d(src)
    assert(src.shape[1] == 3)
    src_count = src.shape[0]

    ls_count = hull.npoints
    # Base
    inverted_ls_triplets = _invert_triplets(valid_simplices, hull.points)

    gains = np.zeros([src_count, ls_count])

    if (jobs_count == 1) or (src_count < 10):
        for src_idx in range(src_count):
            for face_idx, ls_base in enumerate(inverted_ls_triplets):
                # projecting src onto loudspeakers
                projection = np.dot(ls_base, src[src_idx, :])
                # normalization
                projection /= np.sqrt(np.sum(projection**2))
                if np.all(projection > -10e-6):
                    assert(np.count_nonzero(projection) <= 3)
                    # print(f"Source {src_idx}: Gains {projection}")
                    gains[src_idx, valid_simplices[face_idx]] = projection
                    break  # found valid gains
    else:
        warn("Using %i processes..." % jobs_count)
        # preparation
        shared_array_shape = np.shape(gains)
        _arr_base = _create_shared_array(shared_array_shape)
        _arg_itr = zip(range(src_count),
                       repeat(src), repeat(inverted_ls_triplets),
                       repeat(valid_simplices))
        # execute
        with multiprocessing.Pool(processes=jobs_count,
                                  initializer=_init_shared_array,
                                  initargs=(_arr_base,
                                            shared_array_shape,)) as pool:
            pool.starmap(_vbap_gains_single_source, _arg_itr)
        # reshape
        gains = np.frombuffer(_arr_base.get_obj()).reshape(
                                shared_array_shape)

    # Distance compensation
    gains = (hull.d[np.newaxis, :] ** hull.a) * gains
    if retain_outside:
        # remove imaginary loudspeaker
        gains = np.delete(gains, hull.imaginary_ls_idx, axis=1)
    return gains


def vbip(src, hull, valid_simplices=None, retain_outside=False, jobs_count=1):
    """Loudspeaker gains for Vector Base Intensity Panning decoding.

    Parameters
    ----------
    src : (n, 3) numpy.ndarray
        Cartesian coordinates of n sources to be rendered.
    hull : LoudspeakerSetup
    valid_simplices : (nsimplex, 3) numpy.ndarray
        Valid simplices employed for rendering, defaults hull.valid_simplices.
    retain_outside : bool, optional
        Render on the 'ambisonic hull', amplitude will not fade out with VBIP.
    jobs_count : int or None, optional
        Number of parallel jobs, 'None' employs 'cpu_count'.

    Returns
    -------
    gains : (n, L) numpy.ndarray
        Panning gains for L loudspeakers to render n sources.

    Examples
    --------
    .. plot::
        :context: close-figs

        ls_setup = spa.decoder.LoudspeakerSetup(ls_x, ls_y, ls_z)
        ls_setup.pop_triangles(normal_limit=85, aperture_limit=90,
                               opening_limit=150)

        spa.plots.decoder_performance(ls_setup, 'VBIP')

        ls_setup.ambisonics_setup(update_hull=True)
        spa.plots.decoder_performance(ls_setup, 'VBIP', retain_outside=True)
        plt.suptitle('VBIP with imaginary loudspeaker')

    """
    src = np.atleast_2d(src)
    assert(src.shape[1] == 3)
    # Treat VBAP output as squared gains
    g_sq = vbap(src, hull, valid_simplices=valid_simplices,
                retain_outside=retain_outside, jobs_count=jobs_count)
    # Get rid of potential numerical error
    g_sq[g_sq < 0.] = 0.

    # Renormalize
    g = np.sqrt(g_sq)
    g_norm = np.linalg.norm(g, axis=1)
    g[g_norm > 0., :] /= (g_norm[g_norm > 0.][:, np.newaxis])
    return g


def characteristic_ambisonic_order(hull):
    """Find the characteristic order for specified loudspeaker layout.

    References
    ----------
    Zotter, F., & Frank, M. (2012). All-Round Ambisonic Panning and
    Decoding. Journal of Audio Engineering Society, Sec. 7.

    """
    _hull = copy.copy(hull)
    # projection for loudspeakers not on unit sphere
    _xp, _yp, _zp = sph.project_on_sphere(_hull.points[:, 0],
                                          _hull.points[:, 1],
                                          _hull.points[:, 2],)
    _hull.points = np.c_[_xp, _yp, _zp]
    # project centroids (for non-uniform setups)
    src = np.asarray(sph.project_on_sphere(hull.centroids[:, 0],
                                           hull.centroids[:, 1],
                                           hull.centroids[:, 2])).T
    # all loudspeaker triplets enclosing listener
    gains = vbap(src, _hull, valid_simplices=_hull._encloses_listener)
    # Energy vector of each center
    rE, rE_mag = sph.r_E(_hull.points, gains)
    # eq. (16)
    spread = 2 * np.arccos(rE_mag) * (180 / np.pi)
    N_e = 2 * 137.9 / np.average(spread) - 1.51
    # ceil might be optimistic
    return int(np.ceil(N_e))


def allrap(src, hull, N_sph=None, jobs_count=1):
    """Loudspeaker gains for All-Round Ambisonic Panning.

    Parameters
    ----------
    src : (N, 3)
        Cartesian coordinates of N sources to be rendered.
    hull : LoudspeakerSetup
    N_sph : int
        Decoding order, defaults to hull.characteristic_order.
    jobs_count : int or None, optional
        Number of parallel jobs, 'None' employs 'cpu_count'.

    Returns
    -------
    gains : (N, L) numpy.ndarray
        Panning gains for L loudspeakers to render N sources.

    References
    ----------
    Zotter, F., & Frank, M. (2012). All-Round Ambisonic Panning and Decoding.
    Journal of Audio Engineering Society, Sec. 4.

    Examples
    --------
    .. plot::
        :context: close-figs

        ls_setup = spa.decoder.LoudspeakerSetup(ls_x, ls_y, ls_z)
        ls_setup.pop_triangles(normal_limit=85, aperture_limit=90,
                               opening_limit=150)
        ls_setup.ambisonics_setup(update_hull=True)

        spa.plots.decoder_performance(ls_setup, 'ALLRAP')

    """
    if hull.ambisonics_hull:
        ambisonics_hull = hull.ambisonics_hull
    else:
        raise ValueError('Run LoudspeakerSetup.ambisonics_setup() first!')
    if hull.kernel_hull:
        kernel_hull = hull.kernel_hull
    else:
        raise ValueError('Run LoudspeakerSetup.ambisonics_setup() first!')
    if N_sph is None:
        N_sph = hull.characteristic_order

    src = np.atleast_2d(src)
    assert(src.shape[1] == 3)

    # normalize direction
    src = src / np.linalg.norm(src, axis=1)[:, np.newaxis]
    # virtual t-design loudspeakers
    J = len(kernel_hull.points)
    # virtual speakers expressed as VBAP phantom sources (Kernel)
    G_k = vbap(src=kernel_hull.points, hull=ambisonics_hull,
               jobs_count=jobs_count)

    # SH tapering coefficients
    a_n = sph.max_rE_weights(N_sph)
    a_n = sph.unity_gain(a_n)
    a_nm = sph.repeat_per_order(a_n)

    # sources
    _s_azi, _s_colat, _s_r = utils.cart2sph(src[:, 0],
                                            src[:, 1],
                                            src[:, 2])
    Y_s = sph.sh_matrix(N_sph, _s_azi, _s_colat, SH_type='real')
    # kernel
    _k_azi, _k_colat, _k_r = utils.cart2sph(kernel_hull.points[:, 0],
                                            kernel_hull.points[:, 1],
                                            kernel_hull.points[:, 2])
    Y_k = sph.sh_matrix(N_sph, _k_azi, _k_colat, SH_type='real')

    # discretized (band-limited) ambisonic panning function
    G_bld = Y_s @ np.diag(a_nm) @ Y_k.T

    # remove imaginary loudspeaker
    if ambisonics_hull.imaginary_ls_idx is not None:
        G_k = np.delete(G_k, ambisonics_hull.imaginary_ls_idx, axis=1)

    gains = 4 * np.pi / J * G_bld @ G_k
    return gains


def allrap2(src, hull, N_sph=None, jobs_count=1):
    """Loudspeaker gains for All-Round Ambisonic Panning 2.

    Parameters
    ----------
    src : (N, 3)
        Cartesian coordinates of N sources to be rendered.
    hull : LoudspeakerSetup
    N_sph : int
        Decoding order, defaults to hull.characteristic_order.
    jobs_count : int or None, optional
        Number of parallel jobs, 'None' employs 'cpu_count'.

    Returns
    -------
    gains : (N, L) numpy.ndarray
        Panning gains for L loudspeakers to render N sources.

    References
    ----------
    Zotter, F., & Frank, M. (2018). Ambisonic decoding with panning-invariant
    loudness on small layouts (AllRAD2). In 144th AES Convention.

    Examples
    --------
    .. plot::
        :context: close-figs

        ls_setup = spa.decoder.LoudspeakerSetup(ls_x, ls_y, ls_z)
        ls_setup.pop_triangles(normal_limit=85, aperture_limit=90,
                               opening_limit=150)
        ls_setup.ambisonics_setup(update_hull=True)

        spa.plots.decoder_performance(ls_setup, 'ALLRAP2')

    """
    if hull.ambisonics_hull:
        ambisonics_hull = hull.ambisonics_hull
    else:
        raise ValueError('Run LoudspeakerSetup.ambisonics_setup() first!')
    if hull.kernel_hull:
        kernel_hull = hull.kernel_hull
    else:
        raise ValueError('Run LoudspeakerSetup.ambisonics_setup() first!')
    if N_sph is None:
        N_sph = hull.characteristic_order

    src = np.atleast_2d(src)
    assert(src.shape[1] == 3)

    # normalize direction
    src = src / np.linalg.norm(src, axis=1)[:, np.newaxis]
    # virtual t-design loudspeakers
    J = len(kernel_hull.points)
    # virtual speakers expressed as VBAP phantom sources (Kernel)
    G_k = vbap(src=kernel_hull.points, hull=ambisonics_hull,
               jobs_count=jobs_count)

    # SH tapering coefficients
    a_n = sph.max_rE_weights(N_sph)
    a_n = sph.unity_gain(a_n)
    # sqrt(E) normalization (eq.6)
    a_w = np.sqrt(np.sum((2 * (np.arange(N_sph + 1) + 1)) / (4 * np.pi) *
                         a_n**2))
    a_n /= a_w
    a_nm = sph.repeat_per_order(a_n)

    # sources
    _s_azi, _s_colat, _s_r = utils.cart2sph(src[:, 0],
                                            src[:, 1],
                                            src[:, 2])
    Y_s = sph.sh_matrix(N_sph, _s_azi, _s_colat, SH_type='real')
    # kernel
    _k_azi, _k_colat, _k_r = utils.cart2sph(kernel_hull.points[:, 0],
                                            kernel_hull.points[:, 1],
                                            kernel_hull.points[:, 2])
    Y_k = sph.sh_matrix(N_sph, _k_azi, _k_colat, SH_type='real')

    # discretized (band-limited) ambisonic panning function
    G_bld = Y_s @ np.diag(a_nm) @ Y_k.T

    # remove imaginary loudspeaker
    if ambisonics_hull.imaginary_ls_idx is not None:
        G_k = np.delete(G_k, ambisonics_hull.imaginary_ls_idx, axis=1)

    gains = np.sqrt(4 * np.pi / J * G_bld**2 @ G_k**2)
    return gains


def allrad(F_nm, hull, N_sph=None, jobs_count=1):
    """Loudspeaker signals of All-Round Ambisonic Decoder.

    Parameters
    ----------
    F_nm : ((N_sph+1)**2, S) numpy.ndarray
        Matrix of spherical harmonics coefficients of spherical function(S).
    hull : LoudspeakerSetup
    N_sph : int
        Decoding order.
    jobs_count : int or None, optional
        Number of parallel jobs, 'None' employs 'cpu_count'.

    Returns
    -------
    ls_sig : (L, S) numpy.ndarray
        Loudspeaker L output signal S.

    References
    ----------
    Zotter, F., & Frank, M. (2012). All-Round Ambisonic Panning and Decoding.
    Journal of Audio Engineering Society, Sec. 6.

    Examples
    --------
    .. plot::
        :context: close-figs

        ls_setup = spa.decoder.LoudspeakerSetup(ls_x, ls_y, ls_z)
        ls_setup.pop_triangles(normal_limit=85, aperture_limit=90,
                               opening_limit=150)
        ls_setup.ambisonics_setup(update_hull=True)

        spa.plots.decoder_performance(ls_setup, 'ALLRAD')

    """
    if hull.ambisonics_hull:
        ambisonics_hull = hull.ambisonics_hull
    else:
        raise ValueError('Run LoudspeakerSetup.ambisonics_setup() first!')
    if hull.kernel_hull:
        kernel_hull = hull.kernel_hull
    else:
        raise ValueError('Run LoudspeakerSetup.ambisonics_setup() first!')
    if N_sph is None:
        N_sph = hull.characteristic_order

    N_sph_in = int(np.sqrt(F_nm.shape[0]) - 1)
    assert(N_sph == N_sph_in)  # for now
    if N_sph_in > kernel_hull.N_kernel:
        warn("Undersampling the sphere. Needs higher N_Kernel.")

    # virtual t-design loudspeakers
    J = len(kernel_hull.points)
    # virtual speakers expressed as VBAP phantom sources (Kernel)
    G_k = vbap(src=kernel_hull.points, hull=ambisonics_hull,
               jobs_count=jobs_count)

    # SH tapering coefficients
    a_n = sph.max_rE_weights(N_sph)
    a_n = sph.unity_gain(a_n)
    a_nm = sph.repeat_per_order(a_n)

    # virtual Ambisonic decoder
    _k_azi, _k_colat, _k_r = utils.cart2sph(kernel_hull.points[:, 0],
                                            kernel_hull.points[:, 1],
                                            kernel_hull.points[:, 2])
    # band-limited Dirac
    Y_bld = sph.sh_matrix(N_sph, _k_azi, _k_colat, SH_type='real')

    # ALLRAD Decoder
    D = 4 * np.pi / J * G_k.T @ Y_bld
    # apply tapering to decoder matrix
    D = D @ np.diag(a_nm)

    # remove imaginary loudspeakers
    if ambisonics_hull.imaginary_ls_idx is not None:
        D = np.delete(D, ambisonics_hull.imaginary_ls_idx, axis=0)

    # loudspeaker output signals
    ls_sig = D @ F_nm

    return ls_sig


def allrad2(F_nm, hull, N_sph=None, jobs_count=1):
    """Loudspeaker signals of All-Round Ambisonic Decoder 2.

    Parameters
    ----------
    F_nm : ((N_sph+1)**2, S) numpy.ndarray
        Matrix of spherical harmonics coefficients of spherical function(S).
    hull : LoudspeakerSetup
    N_sph : int
        Decoding order, defaults to hull.characteristic_order.
    jobs_count : int or None, optional
        Number of parallel jobs, 'None' employs 'cpu_count'.

    Returns
    -------
    ls_sig : (L, S) numpy.ndarray
        Loudspeaker L output signal S.

    References
    ----------
    Zotter, F., & Frank, M. (2018). Ambisonic decoding with panning-invariant
    loudness on small layouts (AllRAD2). In 144th AES Convention.

    Examples
    --------
    .. plot::
        :context: close-figs

        ls_setup = spa.decoder.LoudspeakerSetup(ls_x, ls_y, ls_z)
        ls_setup.pop_triangles(normal_limit=85, aperture_limit=90,
                               opening_limit=150)
        ls_setup.ambisonics_setup(update_hull=True)

        spa.plots.decoder_performance(ls_setup, 'ALLRAD2')

    """
    if not hull.ambisonics_hull:
        raise ValueError('Run LoudspeakerSetup.ambisonics_setup() first!')
    if hull.kernel_hull:
        kernel_hull = hull.kernel_hull
    else:
        raise ValueError('Run LoudspeakerSetup.ambisonics_setup() first!')
    if N_sph is None:
        N_sph = hull.characteristic_order

    N_sph_in = int(np.sqrt(F_nm.shape[0]) - 1)
    assert(N_sph == N_sph_in)  # for now
    if N_sph_in > kernel_hull.N_kernel:
        warn("Undersampling the sphere. Needs higher N_Kernel.")

    # virtual t-design loudspeakers
    J = len(kernel_hull.points)
    # virtual speakers expressed as phantom sources (Kernel)
    G_k = allrap2(src=kernel_hull.points, hull=hull, N_sph=N_sph,
                  jobs_count=jobs_count)
    # tapering already applied in kernel, sufficient?

    # virtual Ambisonic decoder
    _k_azi, _k_colat, _k_r = utils.cart2sph(kernel_hull.points[:, 0],
                                            kernel_hull.points[:, 1],
                                            kernel_hull.points[:, 2])
    # band-limited Dirac
    Y_bld = sph.sh_matrix(N_sph, _k_azi, _k_colat, SH_type='real')

    # ALLRAD2 Decoder
    D = 4 * np.pi / J * G_k.T @ Y_bld

    # loudspeaker output signals
    ls_sig = D @ F_nm
    return ls_sig


def sad(F_nm, hull, N_sph=None):
    """Loudspeaker signals of Sampling Ambisonic Decoder.

    Parameters
    ----------
    F_nm : ((N_sph+1)**2, S) numpy.ndarray
        Matrix of spherical harmonics coefficients of spherical function(S).
    hull : LoudspeakerSetup
    N_sph : int
        Decoding order, defaults to hull.characteristic_order.

    Returns
    -------
    ls_sig : (L, S) numpy.ndarray
        Loudspeaker L output signal S.

    References
    ----------
    ch. 4.9.1, Zotter, F., & Frank, M. (2019). Ambisonics.
    Springer Topics in Signal Processing.

    Examples
    --------
    .. plot::
        :context: close-figs

        ls_setup = spa.decoder.LoudspeakerSetup(ls_x, ls_y, ls_z)
        ls_setup.pop_triangles(normal_limit=85, aperture_limit=90,
                               opening_limit=150)

        spa.plots.decoder_performance(ls_setup, 'SAD')

    """
    if N_sph is None:
        if hull.characteristic_order:
            N_sph = hull.characteristic_order
        else:
            N_sph = hull.get_characteristic_order()

    L = hull.npoints
    N_sph_in = int(np.sqrt(F_nm.shape[0]) - 1)
    assert(N_sph_in >= N_sph)  # for now

    ls_azi, ls_colat, ls_r = utils.cart2sph(*hull.points.T)
    Y_ls = sph.sh_matrix(N_sph, ls_azi, ls_colat, SH_type='real')

    D = Y_ls
    D *= np.sqrt(4*np.pi / (N_sph+1)**2)
    D *= np.sqrt(4*np.pi / L)  # Energy to unity (on t-design)
    # loudspeaker output signals
    ls_sig = D @ F_nm[:(N_sph+1)**2, :]
    return ls_sig


def mad(F_nm, hull, N_sph=None):
    """Loudspeaker signals of Mode-Matching Ambisonic Decoder.

    Parameters
    ----------
    F_nm : ((N_sph+1)**2, S) numpy.ndarray
        Matrix of spherical harmonics coefficients of spherical function(S).
    hull : LoudspeakerSetup
    N_sph : int
        Decoding order, defaults to hull.characteristic_order.

    Returns
    -------
    ls_sig : (L, S) numpy.ndarray
        Loudspeaker L output signal S.

    References
    ----------
    ch. 4.9.2, Zotter, F., & Frank, M. (2019). Ambisonics.
    Springer Topics in Signal Processing.

    Examples
    --------
    .. plot::
        :context: close-figs

        ls_setup = spa.decoder.LoudspeakerSetup(ls_x, ls_y, ls_z)
        ls_setup.pop_triangles(normal_limit=85, aperture_limit=90,
                               opening_limit=150)

        spa.plots.decoder_performance(ls_setup, 'MAD')

    """
    if N_sph is None:
        if hull.characteristic_order:
            N_sph = hull.characteristic_order
        else:
            N_sph = hull.get_characteristic_order()

    L = hull.npoints
    N_sph_in = int(np.sqrt(F_nm.shape[0]) - 1)
    assert(N_sph_in >= N_sph)  # for now

    ls_azi, ls_colat, ls_r = utils.cart2sph(*hull.points.T)
    Y_ls = sph.sh_matrix(N_sph, ls_azi, ls_colat, SH_type='real')

    D = (np.linalg.pinv(Y_ls)).T
    D *= np.sqrt(L / (N_sph+1)**2)  # Energy to unity (on t-design)

    # loudspeaker output signals
    ls_sig = D @ F_nm[:(N_sph+1)**2, :]
    return ls_sig


def epad(F_nm, hull, N_sph=None):
    r"""Loudspeaker signals of Energy-Preserving Ambisonic Decoder.

    Parameters
    ----------
    F_nm : ((N_sph+1)**2, S) numpy.ndarray
        Matrix of spherical harmonics coefficients of spherical function(S).
    hull : LoudspeakerSetup
    N_sph : int
        Decoding order, defaults to hull.characteristic_order.

    Returns
    -------
    ls_sig : (L, S) numpy.ndarray
        Loudspeaker L output signal S.

    Notes
    -----
    Number of loudspeakers should be greater or equal than SH channels, i.e.

    .. math::  L \geq (N_{sph}+1)^2 .

    References
    ----------
    Zotter, F., Pomberger, H., & Noisternig, M. (2012). Energy-preserving
    ambisonic decoding. Acta Acustica United with Acustica, 98(1), 37–47.

    Examples
    --------
    .. plot::
        :context: close-figs

        ls_setup = spa.decoder.LoudspeakerSetup(ls_x, ls_y, ls_z)
        ls_setup.pop_triangles(normal_limit=85, aperture_limit=90,
                               opening_limit=150)

        spa.plots.decoder_performance(ls_setup, 'EPAD')

        spa.plots.decoder_performance(ls_setup, 'EPAD', N_sph=2,
                                      title='$N_{sph}=2$')

    """
    if N_sph is None:
        if hull.characteristic_order:
            N_sph = hull.characteristic_order
        else:
            N_sph = hull.get_characteristic_order()

    L = hull.npoints
    if (L < (N_sph+1)**2):
        warn('EPAD needs more loudspeakers for this N_sph!'
             f' ({L} < {(N_sph+1)**2})')

    N_sph_in = int(np.sqrt(F_nm.shape[0]) - 1)
    assert(N_sph_in >= N_sph)  # for now

    # SVD of LS base
    ls_azi, ls_colat, ls_r = utils.cart2sph(*hull.points.T)
    Y_ls = sph.sh_matrix(N_sph, ls_azi, ls_colat, SH_type='real')
    U, S, VH = np.linalg.svd(Y_ls)
    # Set singular values to identity and truncate
    S_new = np.eye(L, (N_sph+1)**2)
    D = U @ S_new @ VH
    # Scale to unity
    D *= np.sqrt(4 * np.pi / L)  # Amplitude to unity
    D *= np.sqrt(L / (N_sph+1)**2)  # Energy to unity

    # loudspeaker output signals
    ls_sig = D @ F_nm[:(N_sph+1)**2, :]
    return ls_sig


def nearest_loudspeaker(src, hull):
    """Loudspeaker gains for nearest loudspeaker selection (NLS) decoding,
    based on euclidean distance.

    Parameters
    ----------
    src : (N, 3)
        Cartesian coordinates of N sources to be rendered.
    hull : LoudspeakerSetup

    Returns
    -------
    gains : (N, L) numpy.ndarray
        Panning gains for L loudspeakers to render N sources.

    Examples
    --------
    .. plot::
        :context: close-figs

        ls_setup = spa.decoder.LoudspeakerSetup(ls_x, ls_y, ls_z)
        ls_setup.pop_triangles(normal_limit=85, aperture_limit=90,
                               opening_limit=150)

        spa.plots.decoder_performance(ls_setup, 'NLS')

    """
    src = np.atleast_2d(src)
    src_count = src.shape[0]
    gains = np.zeros([src_count, hull.npoints])

    # Loudspeakers projected on unit sphere
    ls_points = hull.points / hull.d[:, np.newaxis]
    p = np.inner(src, ls_points)
    idx = np.argmax(p, axis=1)
    for g, i in zip(gains, idx):
        g[i] = 1.0 * (hull.d[i] ** hull.a)
    return gains


# Parallel worker stuff -->
def _create_shared_array(shared_array_shape, d_type='d'):
    """Allocate ctypes array from shared memory with lock."""
    shared_array_base = multiprocessing.Array(d_type, shared_array_shape[0] *
                                              shared_array_shape[1])
    return shared_array_base


def _init_shared_array(shared_array_base, shared_array_shape):
    """Make 'shared_array' available to child processes."""
    global shared_array
    shared_array = np.frombuffer(shared_array_base.get_obj())
    shared_array = shared_array.reshape(shared_array_shape)
# < --Parallel worker stuff
