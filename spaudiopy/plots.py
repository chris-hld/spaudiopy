# -*- coding: utf-8 -*-
"""Plotting helpers.

.. plot::
    :context: reset

    import numpy as np
    import matplotlib.pyplot as plt
    plt.rcParams['axes.grid'] = True

    import spaudiopy as spa

"""

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm, colors, tri
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

from . import utils
from . import sph
from . import decoder


def spectrum(x, fs, ylim=None, scale_mag=False, **kwargs):
    """Positive (single sided) amplitude spectrum of time signal x.
    kwargs are forwarded to plots.freq_resp().

    Parameters
    ----------
    x : np.array, list of np.array
        Time domain signal.
    fs : int
        Sampling frequency.

    """
    if not isinstance(x, (list, tuple)):
        x = [x]
    bins = len(x[0])
    freq = np.fft.rfftfreq(bins, d=1. / fs)

    specs = []
    for s in x:
        # rfft returns half sided spectrum
        mag = np.abs(np.fft.rfft(s))
        # Scale the amplitude (factor two for mirrored frequencies)
        mag = mag / len(s)
        assert(mag.ndim == 1)
        if bins % 2:
            # odd
            mag[1:] *= 2.
        else:
            # even
            # should be spec[1:-1] *= 2., but this looks "correct" for plotting
            mag[1:] *= 2.
        # scale by factor bins/2
        if scale_mag:
            mag = mag * bins/2
        specs.append(mag)

    freq_resp(freq, specs, ylim=ylim, **kwargs)


def freq_resp(freq, amp, to_db=True, smoothing_n=None, title=None,
              labels=None, xlim=(10, 25000), ylim=(-30, 20)):
    """ Plot amplitude of frequency response over time frequency f.

    Parameters
    ----------
    f : frequency array
    amp : array_like, list of array_like

    """
    if not isinstance(amp, (list, tuple)):
        amp = [amp]
    if labels is not None:
        if not isinstance(labels, (list, tuple)):
            labels = [labels]

    assert(all(len(a) == len(freq) for a in amp))

    if to_db:
        # Avoid zeros in spec for dB
        amp = [utils.db(np.clip(a, 10e-15, None)) for a in amp]

    if smoothing_n is not None:
        smoothed = []
        for a in amp:
            smooth = np.zeros_like(a)
            for idx in range(len(a)):
                k_lo = idx / (2**(1/(2*smoothing_n)))
                k_hi = idx * (2**(1/(2*smoothing_n)))
                smooth[idx] = np.mean(a[np.floor(k_lo).astype(int):
                                        np.ceil(k_hi).astype(int) + 1])
            smoothed.append(smooth)
        amp = smoothed

    fig, ax = plt.subplots()
    [ax.semilogx(freq, a.flat) for a in amp]

    if title is not None:
        plt.title(title)
    if smoothing_n is not None:
        if labels is None:
            labels = [None] * len(amp)
        # fake line for extra legend entry
        ax.plot([], [], '*', color='black')
        labels.append(r"$\frac{%d}{8}$ octave smoothing" % smoothing_n)
    if labels is not None:
        ax.legend(labels)

    plt.xlabel('Frequency in Hz')
    plt.ylabel('Amplitude in dB')
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.grid(True)


def transfer_function(freq, H, title=None, xlim=(10, 25000)):
    """Plot transfer function H (magnitude and phase) over time frequency f."""
    fig, ax1 = plt.subplots()
    H += 10e-15
    ax1.semilogx(freq, utils.db(H),
                 color=plt.rcParams['axes.prop_cycle'].by_key()['color'][0],
                 label='Amplitude')
    ax1.set_xlabel('Frequency in Hz')
    ax1.set_ylabel('Amplitude in dB')
    ax1.set_xlim(xlim)
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.semilogx(freq, np.unwrap(np.angle(H)),
                 color=plt.rcParams['axes.prop_cycle'].by_key()['color'][1],
                 label='Phase', zorder=0)
    ax2.set_ylabel('Phase in rad')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc=0)
    if title is not None:
        plt.title(title)


def zeropole(b, a, zPlane=False, title=None):
    """Plot Zero Pole diagram from a and b coefficients."""
    z = np.roots(b)
    p = np.roots(a)
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    ax.scatter(np.real(z), np.imag(z), marker='o', label='Zeros')
    ax.scatter(np.real(p), np.imag(p), marker='x', label='Poles')

    if zPlane:
        circle1 = plt.Circle([0, 0], 1, facecolor='none', edgecolor='r')
        ax.add_artist(circle1)
        plt.xlim(np.min([-1.1, plt.xlim()[0]]), np.max([1.1, plt.xlim()[1]]))
        plt.ylim(np.min([-1.1, plt.ylim()[0]]), np.max([1.1, plt.ylim()[1]]))
        plt.xlabel(r'$\Re(z)$')
        plt.ylabel(r'$\Im(z)$')
    else:
        plt.axhline(0, color='black', zorder=0)
        plt.axvline(0, color='red', zorder=0)
        plt.xlabel(r'$\Re(s)$')
        plt.ylabel(r'$\Im(s)$')

    plt.legend(loc=2)
    plt.grid(True)
    if title is not None:
        plt.title(title)


def compare_ambi(Ambi_A, Ambi_B):
    """Compare A and B format signals."""
    t = np.linspace(0, (len(Ambi_B)-1)/Ambi_B.fs, len(Ambi_B))

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(t, Ambi_A.channel[0].signal,
             t, Ambi_A.channel[1].signal,
             t, Ambi_A.channel[2].signal,
             t, Ambi_A.channel[3].signal, alpha=0.3)
    plt.legend(['Channel 1', 'Channel 2', 'Channel 3', 'Channel 4'])
    plt.title('A-format')
    plt.subplot(2, 1, 2)
    plt.plot(t, Ambi_B.channel[0].signal,
             t, Ambi_B.channel[1].signal,
             t, Ambi_B.channel[2].signal,
             t, Ambi_B.channel[3].signal, alpha=0.3)
    plt.legend(['Channel W', 'Channel X', 'Channel Y', 'Channel Z'])
    plt.xlabel('t in s')
    plt.title('B-format')


def spherical_function(f, azi, colat, title=None):
    """Plot function 1D vector f over azi and colat."""
    f = utils.asarray_1d(np.real_if_close(f))
    azi = utils.asarray_1d(azi)
    colat = utils.asarray_1d(colat)
    x, y, z = utils.sph2cart(azi, colat, r=abs(f))

    # triangulate in the underlying parametrization
    triang = tri.Triangulation(colat, azi)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    p_tri = ax.plot_trisurf(x, y, z,
                            cmap=plt.cm.coolwarm,
                            # antialiased=False,
                            triangles=triang.triangles, shade=True,
                            edgecolor='none', linewidth=0.06, alpha=0.25)

    # Draw axis lines
    x0 = np.array([1, 0, 0])
    y0 = np.array([0, 1, 0])
    z0 = np.array([0, 0, 1])
    for i in range(3):
        ax.plot([-x0[i], x0[i]], [-y0[i], y0[i]], [-z0[i], z0[i]], 'k',
                alpha=0.3)

    # overlay data points, radius as color
    p_sc = ax.scatter(x, y, z, c=f, cmap=plt.cm.viridis,
                      vmin=np.min([0, f.min()]),
                      vmax=np.max([1, f.max()]))

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.locator_params(nbins=5)

    cbar = plt.colorbar(p_sc, ax=ax, shrink=0.5, aspect=10)
    cbar.set_label(r"$f\,(\Omega)$")

    plt.grid(True)
    set_aspect_equal3d(ax)
    ax.view_init(25, 230)
    if title is not None:
        plt.title(title)


def sh_coeffs(F_nm, SH_type=None, azi_steps=5, el_steps=3, title=None):
    """Plot spherical harmonics coefficients as function on the sphere."""
    F_nm = utils.asarray_1d(F_nm)
    F_nm = F_nm[:, np.newaxis]
    if SH_type is None:
        SH_type = 'complex' if np.iscomplexobj(F_nm) else 'real'

    azi_steps = np.deg2rad(azi_steps)
    el_steps = np.deg2rad(el_steps)
    phi_plot, theta_plot = np.meshgrid(np.arange(0., 2 * np.pi + azi_steps,
                                                 azi_steps),
                                       np.arange(10e-3, np.pi + el_steps,
                                                 el_steps))

    f_plot = sph.inverse_sht(F_nm, phi_plot.ravel(), theta_plot.ravel(),
                             SH_type)
    f_r = np.abs(f_plot)
    f_ang = np.angle(f_plot)

    x_plot, y_plot, z_plot = utils.sph2cart(phi_plot.ravel(),
                                            theta_plot.ravel(),
                                            f_r.ravel())

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    m = cm.ScalarMappable(cmap=cm.hsv,
                          norm=colors.Normalize(vmin=-np.pi, vmax=np.pi))
    m.set_array(f_ang)
    c = m.to_rgba(f_ang.reshape(phi_plot.shape))

    ax.plot_surface(x_plot.reshape(phi_plot.shape),
                    y_plot.reshape(phi_plot.shape),
                    z_plot.reshape(phi_plot.shape),
                    facecolors=c,
                    edgecolor='none', linewidth=0.06, alpha=0.68, shade=True)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)

    # Draw axis lines
    x0 = np.array([1, 0, 0])
    y0 = np.array([0, 1, 0])
    z0 = np.array([0, 0, 1])
    for i in range(3):
        ax.plot([-x0[i], x0[i]], [-y0[i], y0[i]], [-z0[i], z0[i]], 'k',
                alpha=0.3)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.locator_params(nbins=5)

    cb = plt.colorbar(m, ticks=[-np.pi, 0, np.pi], shrink=0.5, aspect=10)
    cb.set_label("Phase in rad")
    cb.set_ticklabels([r'$-\pi$', r'$0$', r'$\pi$'])

    plt.grid(True)
    set_aspect_equal3d(ax)
    ax.view_init(25, 230)
    if title is not None:
        plt.title(title)


def sh_coeffs_subplot(F_l, SH_type=None, azi_steps=5, el_steps=3, titles=None):
    """Plot spherical harmonics coefficients list as function on the sphere."""
    N_plots = len(F_l)
    azi_steps = np.deg2rad(azi_steps)
    el_steps = np.deg2rad(el_steps)
    phi_plot, theta_plot = np.meshgrid(np.arange(0., 2 * np.pi + azi_steps,
                                                 azi_steps),
                                       np.arange(10e-3, np.pi + el_steps,
                                                 el_steps))

    fig = plt.figure(figsize=plt.figaspect(1 / N_plots))  # constrained_layout=True)
    ax_l = []
    for i_p, ff in enumerate(F_l):
        F_nm = utils.asarray_1d(ff)
        F_nm = F_nm[:, np.newaxis]
        if SH_type is None:
            SH_type = 'complex' if np.iscomplexobj(F_nm) else 'real'

        f_plot = sph.inverse_sht(F_nm, phi_plot.ravel(), theta_plot.ravel(),
                                 SH_type)
        f_r = np.abs(f_plot)
        f_ang = np.angle(f_plot)

        x_plot, y_plot, z_plot = utils.sph2cart(phi_plot.ravel(),
                                                theta_plot.ravel(),
                                                f_r.ravel())

        ax = fig.add_subplot(1, N_plots, i_p + 1, projection='3d')

        m = cm.ScalarMappable(cmap=cm.hsv,
                              norm=colors.Normalize(vmin=-np.pi, vmax=np.pi))
        m.set_array(f_ang)
        c = m.to_rgba(f_ang.reshape(phi_plot.shape))

        ax.plot_surface(x_plot.reshape(phi_plot.shape),
                        y_plot.reshape(phi_plot.shape),
                        z_plot.reshape(phi_plot.shape),
                        facecolors=c,
                        edgecolor='none', linewidth=0.06, alpha=0.68,
                        shade=True)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)

        # Draw axis lines
        x0 = np.array([1, 0, 0])
        y0 = np.array([0, 1, 0])
        z0 = np.array([0, 0, 1])
        for i in range(3):
            ax.plot([-x0[i], x0[i]], [-y0[i], y0[i]], [-z0[i], z0[i]], 'k',
                    alpha=0.3)

        if i_p == 0:
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')

        ax.locator_params(nbins=3)
        plt.grid(True)
        ax.view_init(25, 230)
        if titles is not None:
            ax.set_title(titles[i_p])
        set_aspect_equal3d(ax)
        ax_l.append(ax)

    cbar = plt.colorbar(m, ax=ax_l,
                        shrink=0.5, orientation='horizontal')
    cbar.set_ticks([-np.pi, 0, np.pi])
    cbar.set_ticklabels([r'$-\pi$', r'$0$', r'$\pi$'])


def hull(hull, simplices=None, mark_invalid=True, title=None, ax_lim=None,
         color=None, clim=None):
    """Plot loudspeaker setup and valid simplices from its hull object.

    Parameters
    ----------
    hull : decoder.LoudspeakerSetup
    simplices : optional
    mark_invalid : bool, optional
        mark invalid simplices from hull object.
    title : string, optional
    ax_lim : float, optional
        Axis limits in m.
    color : array_like, optional
        Custom colors for simplices.
    clim : (2,), optional
        `vmin` and `vmax` for colors.

    """
    if simplices is None:
        simplices = hull.simplices

    if mark_invalid:
        if not hasattr(hull, 'valid_simplices'):
            # Can happen when not a LoudspeakerSetup but generic hull object
            mark_invalid = False
            valid_s = simplices
        else:
            is_valid_s = np.array([hull.is_simplex_valid(s)
                                   for s in simplices])
            valid_s = simplices[is_valid_s]
            invalid_s = simplices[~is_valid_s]
            if np.all(is_valid_s):
                # nothing invalid to show
                mark_invalid = False
    else:
        valid_s = simplices

    if color is not None:
        if clim is None:
            clim = (None, None)
        color = utils.asarray_1d(color)
        assert(len(color) == simplices.shape[0])
        m = cm.ScalarMappable(cmap=cm.Spectral,
                              norm=colors.Normalize(vmin=clim[0],
                                                    vmax=clim[1]))
        m.set_array(color)
        colset = m.to_rgba(color)
    else:
        colset = None

    # extract data
    x = hull.points[:, 0]
    y = hull.points[:, 1]
    z = hull.points[:, 2]

    fig = plt.figure(constrained_layout=True)
    ax = fig.gca(projection='3d')

    # valid
    polyc = ax.plot_trisurf(x, y, z,
                            triangles=valid_s,
                            cmap=cm.Spectral if color is None else None,
                            edgecolor='black', linewidth=0.3, alpha=0.6,
                            zorder=2)
    # apply colors if given
    polyc.set_facecolors(colset)
    # invalid
    if mark_invalid:
        ax.plot_trisurf(x, y, z,
                        triangles=invalid_s, linestyle='--',
                        edgecolor='black', linewidth=0.35,
                        color=(0., 0., 0., 0.), alpha=0.1, zorder=2)
    # loudspeaker no
    for s, co in enumerate(np.c_[x, y, z]):
        ax.text(co[0], co[1], co[2], s, zorder=1)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.locator_params(tight=True, nbins=5)

    if ax_lim is None:
        ax_lim = np.max(np.array([abs(x), abs(y), abs(z)])).T
        ax_lim = 1.1*np.max([ax_lim, 1.0])  # 1.1 looks good
    set_aspect_equal3d(ax, XYZlim=(-ax_lim, ax_lim))

    ax.view_init(25, 230)
    if color is not None:
        fig.colorbar(m, ax=ax, fraction=0.024, pad=0.04)
    if title is not None:
        plt.title(title)


def hull_normals(hull, plot_face_normals=True, plot_vertex_normals=True):
    """Plot loudspeaker setup with vertex and face normals."""
    x = hull.points[:, 0]
    y = hull.points[:, 1]
    z = hull.points[:, 2]

    fig = plt.figure(constrained_layout=True)
    ax = fig.gca(projection='3d')
    ax.plot_trisurf(x, y, z,
                    triangles=hull.simplices,
                    edgecolor='black', linewidth=0.5,
                    color=(0., 0., 0., 0.), alpha=0.15, zorder=3)

    for s, co in enumerate(np.c_[x, y, z]):
        ax.text(co[0], co[1], co[2], s, zorder=2)

    if plot_vertex_normals:
        ax.quiver(x, y, z,
                  hull.vertex_normals[:, 0],
                  hull.vertex_normals[:, 1],
                  hull.vertex_normals[:, 2],
                  arrow_length_ratio=0.1,
                  colors=(0.52941176, 0.56862745, 0.7372549, 0.8),
                  alpha=0.6, zorder=0, label='vertex normal')

    if plot_face_normals:
        c_x = hull.centroids[:, 0]
        c_y = hull.centroids[:, 1]
        c_z = hull.centroids[:, 2]
        ax.quiver(c_x, c_y, c_z,
                  hull.face_normals[:, 0],
                  hull.face_normals[:, 1],
                  hull.face_normals[:, 2],
                  arrow_length_ratio=0.1,
                  colors=(0.16078431, 0.64705882, 0.17647059, 1),
                  alpha=0.6, zorder=1, label='simplex normal')

    ax.scatter(hull.listener_position[0], hull.listener_position[1],
               hull.listener_position[2],
               s=48, marker='+', label='listener')

    ax.scatter(hull.barycenter[0], hull.barycenter[1], hull.barycenter[2],
               s=48, marker='*', label='barycenter')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.locator_params(tight=True, nbins=5)

    set_aspect_equal3d(ax, XYZlim=[-1, 1])
    plt.legend(loc='best')
    ax.view_init(25, 230)


def polar(theta, r, title=None, rlim=(-40, 0), ax=None):
    """Polar plot in dB that allows negative values for 'r'."""
    if ax is None:
        fig = plt.figure()
        ax = fig.gca(projection='polar')
    ax.plot(theta, utils.db(np.clip(r, 0, None)), label='$+$')
    ax.plot(theta, utils.db(abs(np.clip(r, None, 0))), label='$-$')
    ax.set_rmin(rlim[0])
    ax.set_rmax(rlim[1])
    ax.set_rticks(np.linspace(rlim[0], rlim[1], 5))
    ax.text(np.pi/8, 5, 'dB', horizontalalignment='left')
    plt.legend(loc='lower right')
    if title is not None:
        plt.title(title)


def decoder_performance(hull, renderer_type, azi_steps=5, ele_steps=3,
                        show_ls=True, title=None, **kwargs):
    """Shows energy, spread and angular error measures on grid.
    For renderer_type={'VBAP', 'VBIP', 'ALLRAP', 'NLS'},
    as well as {'ALLRAD', 'ALLRAD2', 'EPAD', 'MAD'}.
    All kwargs are forwarded to the decoder function.

    Zotter, F., & Frank, M. (2019). Ambisonics.
    Springer Topics in Signal Processing.
    """
    azi_steps = np.deg2rad(azi_steps)
    ele_steps = np.deg2rad(ele_steps)
    phi_vec = np.arange(-np.pi, np.pi + azi_steps, azi_steps)
    theta_vec = np.arange(0., np.pi + ele_steps, ele_steps)
    phi_plot, theta_plot = np.meshgrid(phi_vec, theta_vec)
    _grid_x, _grid_y, grid_z = utils.sph2cart(phi_plot.ravel(),
                                              theta_plot.ravel())

    # Prepare for SH based rendering
    if renderer_type.lower() in ['allrad', 'allrad2', 'epad', 'mad']:
        if 'N_sph' in kwargs:
            N_sph = kwargs.pop('N_sph')
        else:
            N_sph = hull.get_characteristic_order()
        Y_in = sph.sh_matrix(N_sph, phi_plot.flatten(), theta_plot.flatten(),
                             SH_type='real').T

    # Switch renderer
    if renderer_type.lower() == 'vbap':
        G = decoder.vbap(np.c_[_grid_x, _grid_y, grid_z], hull, **kwargs)
    elif renderer_type.lower() == 'vbip':
        G = decoder.vbip(np.c_[_grid_x, _grid_y, grid_z], hull, **kwargs)
    elif renderer_type.lower() == 'allrap':
        G = decoder.allrap(np.c_[_grid_x, _grid_y, grid_z], hull,
                           **kwargs)
    elif renderer_type.lower() == 'allrap2':
        G = decoder.allrap2(np.c_[_grid_x, _grid_y, grid_z], hull,
                            **kwargs)
    elif renderer_type.lower() == 'nls':
        G = decoder.nearest_loudspeaker(np.c_[_grid_x, _grid_y, grid_z], hull,
                                        **kwargs)
    elif renderer_type.lower() == 'allrad':
        G = decoder.allrad(Y_in, hull, N_sph=N_sph, **kwargs).T
    elif renderer_type.lower() == 'allrad2':
        G = decoder.allrad2(Y_in, hull, N_sph=N_sph, **kwargs).T
    elif renderer_type.lower() == 'epad':
        G = decoder.epad(Y_in, hull, N_sph=N_sph, **kwargs).T
    elif renderer_type.lower() == 'mad':
        G = decoder.mad(Y_in, hull, N_sph=N_sph, **kwargs).T
    else:
        raise ValueError('Unknown renderer_type')

    # Measures
    # Amplitude
    A = np.sum(G, axis=1)
    # Energy
    E = np.sum(G**2, axis=1)  # * (4 * np.pi / G.shape[1])  # (eq. 15)
    # project points onto unit sphere
    ls_points = hull.points / hull.d[:, np.newaxis]
    rE, rE_mag = sph.r_E(ls_points, G / hull.d[np.newaxis, :] ** hull.a)
    # Zotter book (eq. 2.11) adds factor 5/8
    spread = 2 * np.arccos(np.clip(rE_mag, 0, 1)) * 180 / np.pi
    # angular error
    col_dot = np.einsum('ij,ij->i', np.array([_grid_x, _grid_y, grid_z]).T,
                        (rE / (rE_mag[:, np.newaxis] + 10e-15)))
    ang_error = np.rad2deg(np.arccos(np.clip(col_dot, -1.0, 1.0)))

    # Show them
    fig, axes = plt.subplots(2, 2, sharex='all', sharey='all')
    axes = axes.ravel()
    for ip, _data in enumerate([A, E, spread, ang_error]):
        _data = _data.reshape(phi_plot.shape)
        ax = axes[ip]
        ax.set_aspect('equal')
        # draw mesh, value corresponds to center of mesh
        p = ax.pcolormesh(phi_plot, theta_plot, _data,
                          shading='gouraud', vmin=0,
                          vmax=np.max([1.0, np.max(_data)])
                          if ip in [0, 1] else 90)

        if show_ls:
            for s, co in enumerate(ls_points):
                # map to pixels grid
                _azi_ls, _colat_ls, _ = utils.cart2sph(*co)
                ax.plot(_azi_ls, _colat_ls, marker='2', color='grey')
                # ax.text(_x_plot, _y_plot, s)  # LS Number
        # Labeling etc
        ax.grid(True)
        ax.set_xlim([-np.pi - azi_steps/2, np.pi + azi_steps/2])
        ax.set_ylim([0 - ele_steps/2, np.pi + ele_steps/2])
        ax.invert_yaxis()
        ax.set_xticks(np.linspace(-np.pi, np.pi, 5))
        ax.set_xticklabels([r'$-\pi$', r'$-\pi/2$', r'$0$',
                            r'$\pi/2$', r'$\pi$'])
        ax.set_yticks(np.linspace(0, np.pi, 3))
        ax.set_yticklabels([r'$0$', r'$\pi/2$', r'$\pi$'])
        cbar = fig.colorbar(p, ax=ax, fraction=0.024, pad=0.04)
        cbar.outline.set_linewidth(0.5)
        if ip == 0:
            ax.set_title(r'$A$')
            cbar.set_ticks([0, 0.5, np.max([1.0, np.max(_data)])])
        if ip == 1:
            ax.set_title(r'$E$')
            cbar.set_ticks([0, 0.5, np.max([1.0, np.max(_data)])])
        elif ip == 2:
            ax.set_xlabel('Azimuth')
            ax.set_ylabel('Colatitude')

            ax.set_title(r'$\sigma_E$')
            cbar.set_ticks([0, 45, 90])
            cbar.set_ticklabels([r'$0^{\circ}$', r'$45^{\circ}$',
                                 r'$90^{\circ}$'])
        elif ip == 3:
            ax.set_title(r'$\Delta \angle$')
            cbar.set_ticks([0, 45, 90])
            cbar.set_ticklabels([r'$0^{\circ}$', r'$45^{\circ}$',
                                 r'$90^{\circ}$'])

    if title is None:
        title = renderer_type
    else:
        title = renderer_type + ', ' + str(title)
    plt.suptitle(title)
    plt.subplots_adjust(wspace=0.25)


def doa(azi, colat, fs, p=None, size=250):
    """Direction of Arrival, with optional p(t) scaling the size."""
    # t in ms
    t_ms = np.linspace(0, len(azi) / fs, len(azi), endpoint=False) * 1000

    # shift azi to [np.pi, np.pi]
    azi[azi > np.pi] = azi[azi > np.pi] % -np.pi
    # colar to elevation
    ele = np.pi/2 - colat

    if p is not None:
        s_plot = np.clip(p / np.max(p), 10e-15, None)
    else:
        s_plot = np.ones_like(azi)
    # scale
    s_plot *= size

    fig, ax = plt.subplots(constrained_layout=True)
    ax.set_aspect('equal')

    # plot in reverse order so that first reflections are on top
    p = ax.scatter(azi[::-1], ele[::-1], s=s_plot[::-1], c=t_ms[::-1],
                   alpha=0.35)
    ax.set_xlabel("Azimuth in rad")
    ax.set_ylabel("Elevation in rad")
    ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    ax.set_xticklabels([r'$-\pi$', r'$-\pi / 2$', r'$0$',
                        r'$\pi / 2$', r'$\pi$'])
    ax.set_yticks([-np.pi/2, 0, np.pi/2])
    ax.set_yticklabels([r'$-\pi / 2$', r'$0$', r'$\pi / 2$'])

    # show t as colorbar
    cbar = plt.colorbar(p, ax=ax, orientation='horizontal')
    cbar.set_label("t in ms")

    try:
        # produce a legend with a cross section of sizes from the scatter
        handles, labels = p.legend_elements(prop="sizes", alpha=0.3, num=5,
                                            func=lambda x: x/size)
        ax.legend(handles, labels, loc="upper right", title="p(t)")
    except AttributeError:  # mpl < 3.3.0
        pass


def set_aspect_equal3d(ax=None, XYZlim=None):
    """
    Set 3D axis to equal aspect.

    Parameters
    ----------
    ax : axis, optional
        ax object. The default is None.
    XYZlim : [min, max], optional
        min and max in m. The default is None.

    Returns
    -------
    None.

    """
    if ax is None:
        ax = plt.gca()

    if XYZlim is None:
        xyzlim = np.array([ax.get_xlim3d(),
                           ax.get_ylim3d(),
                           ax.get_zlim3d()]).T
        XYZlim = [min(xyzlim[0]), max(xyzlim[1])]

    ax.set_xlim3d(XYZlim)
    ax.set_ylim3d(XYZlim)
    ax.set_zlim3d(XYZlim)
    try:  # mpl < 3.1.0
        ax.set_aspect('equal')
    except NotImplementedError:  # mpl >= 3.3.0
        ax.set_box_aspect((1, 1, 1))
