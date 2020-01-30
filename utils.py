import numpy as np
import matplotlib.pyplot as plt
import astropy.coordinates as coord
from astropy.coordinates.representation import CartesianDifferential as CD
import astropy.units as u
import pandas as pd
import copy
from scipy.special import comb

vpos = {'pole': coord.SkyCoord(169.3, -2.8, unit='deg', frame='galactic'),
        'tol': np.arccos(-(0.1*(4*np.pi)/(2*np.pi) - 1)) * u.rad}

galcen = {'distance': 8.122,                # 2018A&A...615L..15G
          'distance_err': 0.031,
          'z_sun': 20.8,                    # 2019MNRAS.482.1417B
          'z_sun_err': 0.3,
          'v_sun': [12.9, 245.6, 7.78],     # 2018RNAAS...2..210D
          'v_sun_err': [3, 1.4, 0.09]}
frame = coord.Galactocentric(galcen_distance=galcen['distance'] * u.kpc,
                             galcen_v_sun=CD(galcen['v_sun'] * u.km/u.s),
                             z_sun=galcen['z_sun'] * u.pc)
galcen['frame'] = frame

def aitoff(ax, ticklabels=False):
    ax.set_longitude_grid_ends(90)
    ax.grid(True, ls='--', which='major')

    xticks = coord.Angle(np.arange(13)*30 * u.deg)
    xticks[0] = 0.001*u.deg
    yticks = coord.Angle([-60, -30, 0, 30, 60]*u.deg)
    xticks, yticks = lonlat2mpl(xticks, yticks)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)

    if ticklabels:
        x = ['', '', r'60$^\circ$', '', r'120$^\circ$', '', '', '',
             r'240$^\circ$', '', r'300$^\circ$', '', '']
        ax.set_xticklabels(x)
        ax.tick_params(axis='both', pad=2000)

        ax.set_yticklabels([])
        ax.text(0, -np.pi/3, r'$-60^\circ$', ha='center', va='center')
        ax.text(0, -np.pi/6, r'$-30^\circ$', ha='center', va='center')
        ax.text(0, np.pi/3, r'$60^\circ$', ha='center', va='center')
        ax.text(0, np.pi/6, r'$30^\circ$', ha='center', va='center')
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

def set_paper_random_seed():
    from astropy.cosmology import Planck15
    seed = int(Planck15.age(z=0)/u.kyr)
    np.random.seed(seed=seed)

def prob_nCk_sphere(n, k, theta, beta=0*u.deg):
    p = (1 - np.cos(theta)) / np.cos(beta)
    sum = 0
    for i in range(n-k+1):
        nCk = comb(N=n, k=k+i, exact=True, repetition=False)
        sum += nCk * p**(k+i) * (1-p)**(n-k-i)
    return sum

def plot_aitoff(ax, lon, lat, plot=None, lower=None, **kwargs):
    if lower is not None:
        # restrict normals to [lower, lower+180)
        lon = copy.copy(lon)
        lat = copy.copy(lat)
        upper = lower + 180*u.deg
        ok = lower <= lon
        ok &= lon < upper
        lon[~ok] = lon[~ok] + 180*u.deg
        lat[~ok] = -lat[~ok]

    plot_lon, plot_lat = lonlat2mpl(lon, lat)
    if plot:
        ax.plot(plot_lon, plot_lat, **kwargs)
    else:
        ax.scatter(plot_lon, plot_lat, **kwargs)

def plot_vpos(ax, pole=None, tol=None, counter=True):
    co = vpos['pole'] if pole is None else pole
    theta = vpos['tol'] if tol is None else tol
    phi = np.linspace(0, 2*np.pi, 100)
    rcos = theta*np.cos(phi)
    rsin = theta*np.sin(phi)

    co_ring = coord.SkyCoord(co.l + rcos, co.b + rsin, frame='galactic')

    k = {'c': 'g', 'zorder': 100}
    plot_aitoff(ax, co.l, co.b, marker='x', s=100, **k)
    plot_aitoff(ax, co_ring.l, co_ring.b, plot=True, **k)

    if counter:
        con = coord.SkyCoord(co.l+180*u.deg, -co.b, frame='galactic')
        con_ring = coord.SkyCoord(co.l+rcos-np.pi*u.rad, -co.b-rsin,
                                  frame='galactic')
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
