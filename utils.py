import numpy as np
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import astropy.units as u

def vpos_pars():
    pole = SkyCoord(169.3, -2.8, unit='deg', frame='galactic')
    tol = np.arccos(-(0.1*(4*np.pi)/(2*np.pi) - 1)) * u.rad
    return pole, tol

def aitoff(figsize=(8,6), dpi=100, plot_vpos=True):
    plt.figure(figsize=figsize, dpi=dpi)
    ax = plt.subplot(111, projection="aitoff")

    ticks = np.array([-1, -2/3, -1/3, 0, 1/3, 2/3, 1])
    ax.set_xticks(ticks*np.pi)
    ax.set_yticks(ticks*np.pi/2)
    ax.set_xticklabels([r"{0}$^\circ$".format(int((-i+1)*180)) for i in ticks])
    ax.grid(True)

    if plot_vpos:
        co, theta = vpos_pars()
        phi = np.linspace(0, 2*np.pi, 100)
        rcos = theta*np.cos(phi)
        rsin = theta*np.sin(phi)

        con = SkyCoord(co.l+180*u.deg, -co.b, frame='galactic')
        co_ring = SkyCoord(co.l + rcos, co.b + rsin, frame='galactic')
        con_ring = SkyCoord(co.l+rcos-np.pi*u.rad,-co.b-rsin, frame='galactic')

        k = {'c': 'g', 'zorder': 100}
        plot_aitoff(ax, co.l, co.b, marker='x', s=100, **k)
        plot_aitoff(ax, con.l, con.b, marker='+', s=100, **k)
        plot_aitoff(ax, co_ring.l, co_ring.b, plot=True, **k)

        # left counter-loop
        sel = con_ring.l < 180*u.deg
        plot_aitoff(ax, con_ring.l[sel], con_ring.b[sel], plot=True, **k)

        # right counter-loop
        sel = con_ring.l > 180*u.deg
        plot_aitoff(ax, con_ring.l[sel], con_ring.b[sel], plot=True, **k)

    return ax

def plot_aitoff(ax, lon, lat, plot=None, **kwargs):
    plot_lon, plot_lat = lonlat2mpl(lon, lat)
    if plot:
        ax.plot(plot_lon, plot_lat, **kwargs)
    else:
        ax.scatter(plot_lon, plot_lat, **kwargs)

def lonlat2mpl(lon, lat):
    plot_lon = (-lon + 180*u.deg).wrap_at(180*u.deg).rad
    plot_lat = lat.rad
    return plot_lon, plot_lat
