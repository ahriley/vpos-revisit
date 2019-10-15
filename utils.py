import numpy as np
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import astropy.units as u
import pandas as pd
import copy

def vpos_pars():
    pole = SkyCoord(169.3, -2.8, unit='deg', frame='galactic')
    tol = np.arccos(-(0.1*(4*np.pi)/(2*np.pi) - 1)) * u.rad
    return pole, tol

def aitoff(ax, xticks=None, yticks=None):
    ax.set_longitude_grid_ends(90)
    ax.grid(True, ls='--', which='major')

    # NOTE: needs to get adjusted to do what I want
    if xticks and yticks:
        xlabels = [r"{0:.0f}$^\circ$".format(tick.value) for tick in xticks]
        ylabels = [r"{0:.0f}$^\circ$".format(tick.value) for tick in yticks]
        xticks, yticks = lonlat2mpl(xticks, yticks)
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        ax.set_xticklabels(xlabels)
        ax.set_yticklabels(ylabels)
    else:
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    return ax

def load_globulars():
    catalog = 'data/Vasiliev19.txt'
    cols = ['name', 'ra', 'dec', 'dist', 'vLOS', 'e_vLOS', 'pmRA', 'pmDE',
            'e_pmRA', 'e_pmDE', 'pm_corr', 'Rscale', 'Nstar']
    df = pd.read_csv(catalog, sep='\t', names=cols, skiprows=2)
    df['name'] = df.apply(lambda x: x['name'].split('(')[0].strip(), axis=1)
    df.set_index('name', inplace=True)

    # add classification from 2018MNRAS.481..918A (using 2005MNRAS.360..631M)
    classifier = 'data/Arakelyan18.csv'
    df2 = pd.read_csv(classifier)
    df2.set_index('Name', inplace=True)
    df['type'] = df2['Type']

    # these two GCs are in Vasiliev19 and not Arakelyan18
    df.at['Crater', 'type'] = 'UN'
    df.at['FSR 1716', 'type'] = 'UN'
    return df

def plot_aitoff(ax, lon, lat, plot=None, **kwargs):
    plot_lon, plot_lat = lonlat2mpl(lon, lat)
    if plot:
        ax.plot(plot_lon, plot_lat, **kwargs)
    else:
        ax.scatter(plot_lon, plot_lat, **kwargs)

def plot_vpos(ax, pole=None, tol=None, counter=True):
    co = vpos_pars()[0] if pole is None else pole
    theta = vpos_pars()[1] if tol is None else tol
    phi = np.linspace(0, 2*np.pi, 100)
    rcos = theta*np.cos(phi)
    rsin = theta*np.sin(phi)

    co_ring = SkyCoord(co.l + rcos, co.b + rsin, frame='galactic')

    k = {'c': 'g', 'zorder': 100}
    plot_aitoff(ax, co.l, co.b, marker='x', s=100, **k)
    plot_aitoff(ax, co_ring.l, co_ring.b, plot=True, **k)

    if counter:
        con = SkyCoord(co.l+180*u.deg, -co.b, frame='galactic')
        con_ring = SkyCoord(co.l+rcos-np.pi*u.rad,-co.b-rsin, frame='galactic')
        plot_aitoff(ax, con.l, con.b, marker='+', s=100, **k)

        # left counter-loop
        sel = con_ring.l < 180*u.deg
        plot_aitoff(ax, con_ring.l[sel], con_ring.b[sel], plot=True, **k)

        # right counter-loop
        sel = con_ring.l > 180*u.deg
        plot_aitoff(ax, con_ring.l[sel], con_ring.b[sel], plot=True, **k)


def lonlat2mpl(lon, lat):
    plot_lon = (-lon + 180*u.deg).wrap_at(180*u.deg).rad
    plot_lat = lat.rad
    return plot_lon, plot_lat
