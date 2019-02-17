# -*- coding: utf-8 -*-

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm, colors
from mpl_toolkits.mplot3d import Axes3D  # for (projection='3d')

from plotly import offline as pltlyof
import plotly.graph_objs as pltlygo

from . import utils
from . import sig
from . import sph
from . import decoder


def fresp(f, amp, title=None, labels=None,
          xlim=[10, 25000], ylim=[-30, 20]):
    """
    Plot amplitude frequency response over time frequency f.

    Parameters
    ----------
    f : frequency array
    *amp : array_like, list of array_like

    """
    fig = plt.figure()
    if not isinstance(amp, (list, tuple)):
        amp = [amp]
    [plt.semilogx(f, a) for a in amp]

    if title is not None:
        plt.title(title)
    if labels is not None:
        plt.legend(labels)
    plt.xlabel('Frequency in Hz')
    plt.ylabel('Amplitude in dB')
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.grid(True)
    fig.tight_layout()


def H(f, H, title=None, xlim=[10, 25000]):
    """Plot transfer function H (amplitude and phase) over time frequency f."""
    fig, ax1 = plt.subplots()
    ax1.semilogx(f, 20 * np.log10(np.abs(H)),
                 color=plt.rcParams['axes.prop_cycle'].by_key()['color'][0],
                 label='Amplitude')
    ax1.set_xlabel('Frequency in Hz')
    ax1.set_ylabel('Amplitude in dB')
    ax1.set_xlim(xlim)
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.semilogx(f, np.unwrap(np.angle(H)),
                 color=plt.rcParams['axes.prop_cycle'].by_key()['color'][1],
                 label='Phase')
    ax2.set_ylabel('Phase in rad')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc=0)

    fig.tight_layout()
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


def iplot_Sphere(x, y, z, last=-1):
    """Plot Sphere."""
    last = min(len(x), last)  # clip last to length of input
    trace1 = pltlygo.Scatter3d(
        x=x[:last],
        y=y[:last],
        z=z[:last],
        mode='markers',
        marker=dict(
            size=3,
            color=np.arange(len(x[:last])),
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(
                title='Sample (t)'
            ),
            opacity=0.6
        ),
    )

    data = [trace1]
    layout = pltlygo.Layout(
        scene=dict(
            xaxis=dict(range=[-1, 1]),
            yaxis=dict(range=[-1, 1]),
            zaxis=dict(range=[-1, 1])),
        margin=dict(l=0, r=0, b=0, t=0)
    )
    fig = pltlygo.Figure(data=data, layout=layout)
    pltlyof.iplot(fig, image='png')


def pseudoI(I_phi, I_theta, I_r, last=-1):
    """Plot Pseudo-Intensity."""
    plt.figure()
    plt.plot(I_r[:last], label=r'$r$')
    plt.plot(I_phi[:last], label=r'$\phi$')
    plt.plot(I_theta[:last], label=r'$\theta$')
    plt.legend()
    plt.xlabel('t in samples')
    plt.title('Pseudo-Intensity')
    plt.figure()
    iplot_Sphere(*utils.sph2cart(I_phi, I_theta, I_r), last=last)


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


def sph_coeffs(F_nm, SH_type=None, azi_steps=5, el_steps=3, title=None):
    """Plot spherical harmonics coefficients as function on the sphere."""
    F_nm = utils.asarray_1d(F_nm)
    if SH_type is None:
        SH_type = 'complex' if np.iscomplexobj(F_nm) else 'real'

    azi_steps = np.deg2rad(azi_steps)
    el_steps = np.deg2rad(el_steps)
    phi_plot, theta_plot = np.meshgrid(np.arange(0., 2 * np.pi + azi_steps,
                                                 azi_steps),
                                       np.arange(10e-3, np.pi + el_steps,
                                                 el_steps))

    f_plot = sph.inverse_sht(F_nm, phi_plot.ravel(), theta_plot.ravel(), SH_type)
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
        ax.plot([-x0[i], x0[i]], [-y0[i], y0[i]], [-z0[i], z0[i]], 'k', alpha=0.3)

    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_zlabel('z', fontsize=12)

    cb = plt.colorbar(m, ticks=[-np.pi, 0, np.pi], shrink=0.3, aspect=8)
    cb.set_ticklabels([r'$-\pi$', r'$0$', r'$\pi$'])

    plt.grid(True)
    ax.set_aspect('equal')
    ax.view_init(25, 230)
    if title is not None:
        plt.title(title)
    fig.tight_layout()


def subplot_sph_coeffs(F_l, SH_type=None, azi_steps=5, el_steps=3, title=None):
    """Plot spherical harmonics coefficients as function on the sphere."""
    N_plots = len(F_l)
    azi_steps = np.deg2rad(azi_steps)
    el_steps = np.deg2rad(el_steps)
    phi_plot, theta_plot = np.meshgrid(np.arange(0., 2 * np.pi + azi_steps,
                                                 azi_steps),
                                       np.arange(10e-3, np.pi + el_steps,
                                                 el_steps))

    fig = plt.figure(figsize=plt.figaspect(1 / N_plots))
    ax_l = []
    for i_p, ff in enumerate(F_l):
        F_nm = utils.asarray_1d(ff)
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

        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('y', fontsize=12)
        ax.set_zlabel('z', fontsize=12)

        plt.grid(True)
        ax.view_init(25, 230)
        if title is not None:
            ax.set_title(title[i_p])
        ax.set_aspect('equal')
        ax_l.append(ax)

    cb = plt.colorbar(m, ticks=[-np.pi, 0, np.pi], shrink=0.3, aspect=8,
                      ax=ax_l)
    cb.set_ticklabels([r'$-\pi$', r'$0$', r'$\pi$'])


def hull(hull, simplices=None, mark_invalid=True, title=None):
    """Plot loudspeaker setup and valid simplices from its hull object."""
    if simplices is None:
        simplices = hull.simplices

    if mark_invalid:
        is_valid_s = np.array([hull.is_simplex_valid(s) for s in simplices])
        valid_s = simplices[is_valid_s]
        invalid_s = simplices[~is_valid_s]
        if np.all(is_valid_s):
            # nothing invalid to show
            mark_invalid = False
    else:
        valid_s = simplices

    x = hull.points[:, 0]
    y = hull.points[:, 1]
    z = hull.points[:, 2]
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d', aspect='equal')
    # valid
    ax.plot_trisurf(x, y, z,
                    triangles=valid_s,
                    edgecolor='black', linewidth=0.3,
                    cmap=plt.cm.Spectral, alpha=0.6, zorder=2)
    # invalid
    if mark_invalid:
        ax.plot_trisurf(x, y, z,
                        triangles=invalid_s, linestyle='--',
                        edgecolor='black', linewidth=0.35,
                        color=(0., 0., 0., 0.), alpha=0.1, zorder=2)
    # loudspeaker no
    for s, co in enumerate(np.c_[x, y, z]):
        ax.text(co[0], co[1], co[2], s, zorder=1)

    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_zlabel('z', fontsize=12)

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    if title is not None:
        plt.title(title)


def hull_normals(hull, plot_face_normals=True, plot_vertex_normals=True):
    """Plot loudspeaker setup with vertex and face normals."""
    x = hull.points[:, 0]
    y = hull.points[:, 1]
    z = hull.points[:, 2]
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d', aspect='equal')
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
                  colors=(0.52941176, 0.56862745, 0.7372549 , 0.8),
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

    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_zlabel('z', fontsize=12)

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    plt.legend(loc='best')


def polar(theta, a, title=None, rlim=[-40, 0]):
    """Polar plot that allows negative values for 'a'."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='polar')
    ax.plot(theta, utils.dB(np.clip(a, 0, None)), label='pos')
    ax.plot(theta, utils.dB(abs(np.clip(a, None, 0))), label='neg')
    ax.set_rmin(rlim[0])
    ax.set_rmax(rlim[1])
    plt.legend(loc='upper left')
    if title is not None:
        plt.title(title)


def decoder_performance(hull, renderer_type, azi_steps=5, el_steps=3, N=None):
    """Currently rE_mag, E and spread for renderer_type='VBAP' or 'ALLRAP'."""
    azi_steps = np.deg2rad(azi_steps)
    el_steps = np.deg2rad(el_steps)
    phi_plot, theta_plot = np.meshgrid(np.arange(0., 2 * np.pi + azi_steps,
                                                 azi_steps),
                                       np.arange(0., np.pi + el_steps,
                                                 el_steps))
    _grid_x, _grid_y, grid_z = utils.sph2cart(phi_plot.ravel(),
                                              theta_plot.ravel())

    # Switch renderer
    if renderer_type.lower() == 'vbap':
        G = decoder.vbap(np.c_[_grid_x, _grid_y, grid_z], hull)
    if renderer_type.lower() == 'allrap':
        G = decoder.ALLRAP(np.c_[_grid_x, _grid_y, grid_z], hull, N)

    # Measures
    E = np.sum(G**2, axis=1)  # * (4 * np.pi / G.shape[1])  # (eq. 15)
    rE, rE_mag = sph.r_E(hull.points, G)
    # TODO remove np.clip and handle non-uniform
    spread = 2 * np.arccos(np.clip(rE_mag, 0, 1)) * 180 / np.pi  # (eq. 16)

    # Show them
    fig, axes = plt.subplots(3, 1, sharex='all', figsize=plt.figaspect(2))
    for ip, var_str in enumerate(['rE_mag', 'E', 'spread']):
        _data = eval(var_str)
        _data = _data.reshape(phi_plot.shape)
        # shift 0 azi to middle
        _data = np.roll(_data, - int(_data.shape[1]/2), axis=1)
        ax = axes[ip]
        p = ax.imshow(_data, vmin=0, vmax=180 if var_str is "spread" else
                      np.max([1.0, np.max(_data)]))
        ax.set_xticks(np.linspace(0, _data.shape[1] - 1, 5))
        ax.set_xticklabels(['$-\pi$', '$-\pi/2$', '$0$', '$\pi/2$', '$\pi$'])
        ax.set_yticks(np.linspace(0, _data.shape[0] - 1, 3))
        ax.set_yticklabels(['$0$', '$\pi/2$', '$\pi$'])
        ax.set_title(var_str)
        fig.colorbar(p, ax=ax, fraction=0.024, pad=0.04)

    ax.set_xlabel('Azimuth')
    ax.set_ylabel('Colatitude')
    plt.suptitle(renderer_type)
