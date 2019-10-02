import numpy as np
import astropy.coordinates as coord
import astropy.units as u
import gala.potential as gp
import gala.dynamics as gd
from gala.coordinates import GreatCircleICRSFrame
import utils
import pickle
import os
import matplotlib.pyplot as plt
import pandas as pd
import multiprocessing as mp

countfile = 'data/processed/counting-orbits.npy'
potential = gp.MilkyWayPotential()
v_sun = coord.CartesianDifferential([10, 248, 7]*u.km/u.s)
gc_frame = coord.Galactocentric(galcen_distance=8.178*u.kpc,
                                z_sun=20.8*u.pc,
                                galcen_v_sun=v_sun)

# load in nominal stream normals
result_file = result_file = 'data/processed/my-compilation-output.pkl'
with open(result_file, 'rb') as f:
    result = pickle.load(f)
    table = result['table']
    nom_normals = result['nom_normals']
    MC_normals = result['MC_normals']

# load satellite data
globs = utils.load_globulars()
globs['e_dist'] = globs['dist'] * 0.046
gals = pd.read_csv('data/mw-satellites.csv', index_col=0)
sats = pd.concat([globs, gals], sort=False)

# convert present day 6-D info to SkyCoord
cols = ['ra', 'dec', 'dist', 'vLOS', 'pmRA', 'pmDE']
ra, dec, dist, vlos, pmra, pmdec = sats[cols].T.to_numpy()
icrs = coord.SkyCoord(ra=ra*u.deg, dec=dec*u.deg, distance=dist*u.kpc,
                      pm_ra_cosdec=pmra*u.mas/u.yr, pm_dec=pmdec*u.mas/u.yr,
                      radial_velocity=vlos*u.km/u.s)

# get stream endpoints as SkyCoords
endpts = coord.SkyCoord(ra=table[['lon1', 'lon2']].values*u.deg,
                        dec=table[['lat1', 'lat2']].values*u.deg,
                        distance=table[['dist1', 'dist2']].values*u.kpc,
                        frame='icrs')

# Monte Carlo sample errors
Nsamples = 1000
cols = ['e_dist', 'e_vLOS', 'e_pmRA', 'e_pmDE']
e_dist, e_vlos, e_pmra, e_pmdec = sats[cols].T.to_numpy()
icrs_err = coord.SkyCoord(ra=0*u.deg, dec=0*u.deg, distance=e_dist*u.kpc,
                          pm_ra_cosdec=e_pmra*u.mas/u.yr,
                          pm_dec=e_pmra*u.mas/u.yr,
                          radial_velocity=e_vlos*u.km/u.s)
sample_pars = (Nsamples,len(icrs))
dist = np.random.normal(icrs.distance.value, icrs_err.distance.value,
                        sample_pars) * icrs.distance.unit
pm_ra_cosdec = np.random.normal(icrs.pm_ra_cosdec.value,
                                icrs_err.pm_ra_cosdec.value,
                                sample_pars) * icrs.pm_ra_cosdec.unit
pm_dec = np.random.normal(icrs.pm_dec.value,
                          icrs_err.pm_dec.value,
                          sample_pars) * icrs.pm_dec.unit
rv = np.random.normal(icrs.radial_velocity.value,
                      icrs_err.radial_velocity.value,
                      sample_pars) * icrs.radial_velocity.unit
icrs_samples = coord.SkyCoord(ra=ra*u.deg, dec=dec*u.deg, distance=dist,
                              pm_ra_cosdec=pm_ra_cosdec,
                              pm_dec=pm_dec, radial_velocity=rv)

# integrate orbits
timestep = 0.1 * u.Myr
totaltime = 1*u.Gyr
nsteps = (totaltime / timestep).to(u.dimensionless_unscaled)

# tolerance for being considered "associated"
septolmin = 0.5
dtolmin = 1
disttols = np.mean(table[['e_dist1', 'e_dist2']], axis=1).values
disttols = [d if d > dtolmin else dtolmin for d in disttols] * u.kpc
septols = np.mean(table[['e_lb1', 'e_lb2']], axis=1).values
septols = [a if a > septolmin else septolmin for a in septols] * u.deg

# function that computes orbits and associates with stream
def orbitxstreams(ICs):
    galcen = ICs.transform_to(gc_frame)
    w0 = gd.PhaseSpacePosition(galcen.data)
    orbit = potential.integrate_orbit(w0, dt=-timestep, n_steps=nsteps)

    sc = coord.SkyCoord(x=orbit.x, y=orbit.y, z=orbit.z,
                        v_x=orbit.v_x, v_y=orbit.v_y, v_z=orbit.v_z,
                        frame=gc_frame)
    sc = sc.transform_to(coord.ICRS).T

    # compute 3-D separation from each endpoint, all along orbit
    distmins = []
    for stream, disttol, septol in zip(endpts, disttols, septols):
        distmin_stream = []
        for pt in stream:
            sel = pt.separation(sc) < septol
            sel &= np.abs(sc.distance - pt.distance) < disttol
            distmin_stream.append(sel.any(axis=1))
        distmins.append(distmin_stream)
    distmins = np.array(distmins)
    return distmins.all(axis=1)

# compute orbits and associate (parallelized!)
if os.path.isfile(countfile):
    match = np.load(countfile)
else:
    ncpu = mp.cpu_count()
    pool = mp.Pool(ncpu, maxtasksperchild=1)
    out = pool.map(orbitxstreams, icrs_samples)
    match = np.sum(out, axis=0) / 1000
    np.save(countfile, match)

for row, mins in zip(table.iterrows(), match):
    name, data = row
    sel = mins > 0.05
    match_names = list(sats.index[sel])
    if len(match_names) == 0:
        continue
    print(name, data['dist1'], match_names)
