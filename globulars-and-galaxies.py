import numpy as np
import pandas as pd
import utils
import astropy.coordinates as coord
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.coordinates.representation import CartesianDifferential as CD
from gala.coordinates import GreatCircleICRSFrame
import pickle

# comment for different random seed
utils.set_paper_random_seed()

# number of samples, save location, catalog
N = 2000
outfiles = ['data/processed/globulars-output.pkl',
            'data/processed/satellites-output.pkl']
cats = [utils.load_globulars(),
        pd.read_csv('data/mw-satellites.csv', index_col=0)]

# assumed MW/VPOS properties
vpos_pole, vpos_tol = utils.vpos['pole'], utils.vpos['tol']
galcen_frame = utils.galcen['frame']

# asummed distance error (in Vasiliev19, distance modulus +- 0.1mag)
cats[0]['e_dist'] = 0.046 * cats[0]['dist']

for cat, outfile in zip(cats, outfiles):
    # nominal orbital pole (using most likely values)
    sc = SkyCoord(ra=cat['ra'].values*u.deg, dec=cat['dec'].values*u.deg,
                    distance=cat['dist'].values*u.kpc,
                    pm_ra_cosdec=cat['pmRA'].values*u.mas/u.yr,
                    pm_dec=cat['pmDE'].values*u.mas/u.yr,
                    radial_velocity=cat['vLOS'].values*u.km/u.s)
    sc = sc.transform_to(galcen_frame)
    pos = np.array([sc.x, sc.y, sc.z])
    vel = np.array([sc.v_x, sc.v_y, sc.v_z])
    L = np.cross(pos.T, vel.T).T
    Lval, lat, lon = coord.cartesian_to_spherical(L[0], L[1], L[2])
    nom_poles = SkyCoord(lon, lat, frame='galactic')
    r_MW, b_MW, l_MW = coord.cartesian_to_spherical(sc.x, sc.y, sc.z)
    sc_MW = SkyCoord(l=l_MW, b=b_MW, frame='galactic')
    cat['dist_gal'] = r_MW
    cat['h_nom'] = Lval

    # Monte Carlo sample in heliocentric
    positions = []
    coords_converted = []
    for GCname, GC in cat.iterrows():
        # construct means and covariances for proper motion sampling
        means = np.array([GC['pmRA'], GC['pmDE']])
        cov = [[GC['e_pmRA']**2,GC['pm_corr']*\
                    GC['e_pmRA']*GC['e_pmDE']],
                [GC['pm_corr']*GC['e_pmRA']*\
                    GC['e_pmDE'], GC['e_pmDE']**2]]
        cov = np.array(cov)

        # perform sampling, store in pos
        pos = np.zeros((N,6))
        pos[:,0:2] = np.random.multivariate_normal(mean=means, cov=cov, size=N)
        pos[:,2] = np.random.normal(GC['vLOS'],GC['e_vLOS'],N)
        pos[:,3] = np.random.normal(GC['dist'],GC['e_dist'],N)
        pos[:,4] = np.ones(N)*GC['ra']
        pos[:,5] = np.ones(N)*GC['dec']

        positions.append(np.swapaxes(pos, axis1=0, axis2=1))
    positions = np.array(positions)

    # convert that sample to galactocentric to get Monte Carlo orbital poles
    poles = []
    for i in range(N):
        sc = SkyCoord(ra=positions[:,4,i]*u.deg,
                        dec=positions[:,5,i]*u.deg,
                        distance=positions[:,3,i]*u.kpc,
                        pm_ra_cosdec=positions[:,0,i]*u.mas/u.yr,
                        pm_dec=positions[:,1,i]*u.mas/u.yr,
                        radial_velocity=positions[:,2,i]*u.km/u.s,
                        frame='icrs')

        sc = sc.transform_to(galcen_frame)
        assert galcen_frame.galcen_distance == sc.galcen_distance

        # compute orbital pole
        pos = np.array([sc.x, sc.y, sc.z])
        vel = np.array([sc.v_x, sc.v_y, sc.v_z])
        L = np.cross(pos.T, vel.T).T
        Lval, lat_i, lon_i = coord.cartesian_to_spherical(L[0], L[1], L[2])
        poles.append([lon_i.deg, lat_i.deg, Lval])
    poles = np.array(poles)
    MC_poles = SkyCoord(poles[:,0,:], poles[:,1,:],
                        unit='deg', frame='galactic')
    MC_poles = MC_poles.T
    hvals = poles[:,2,:].T

    # spherical standard distance
    sep = nom_poles.separation(MC_poles.T).T
    cat['sph_std'] = np.sqrt(np.sum(sep.deg**2, axis=1) / N)
    sep = np.arccos(np.abs(np.cos(sep))).to(u.deg).value
    cat['sph_std_abs'] = np.sqrt(np.sum(sep**2, axis=1) / N)

    # theta_obs : angle between VPOS normal and observed pole
    sep = vpos_pole.separation(nom_poles)
    cat['theta_obs'] = np.arccos(np.abs(np.cos(sep))).to(u.deg)

    # theta_pred : min angle from VPOS to stream track w/ endpoint as pole
    phi1 = np.arange(0,360,0.001)*u.deg
    theta_pred, pole_pred = [], []
    for loc in sc_MW:
            end = loc.transform_to(coord.ICRS)
            streamfr = GreatCircleICRSFrame(pole=end, ra0=0*u.deg)
            gc = SkyCoord(phi1=phi1, phi2=0*u.deg, frame=streamfr)
            assert np.allclose(end.separation(gc).deg, 90)
            gc = gc.transform_to(coord.Galactic)
            ii = np.argmin(vpos_pole.separation(gc).deg)
            theta_pred.append(vpos_pole.separation(gc[ii]).deg)
            pole_pred.append([gc[ii].l.deg, gc[ii].b.deg])
    cat['theta_pred'] = theta_pred
    pole_pred = np.array(pole_pred).T
    pole_pred = SkyCoord(pole_pred[0], pole_pred[1],
                         unit='deg', frame='galactic')
    assert (vpos_pole.separation(pole_pred).deg == cat['theta_pred']).all()
    assert (cat['theta_obs'] > cat['theta_pred']).all()

    # p_inVPOS : how many Monte Carlo poles lie within vpos_tol
    sep = vpos_pole.separation(MC_poles)
    sep = np.arccos(np.abs(np.cos(sep)))
    cat['p_inVPOS'] = np.sum(sep < vpos_tol, axis=1) / N

    # p_gVPOS : falsely finding a pole to be misaligned with VPOS
    # p_gobs : finding intrinsically aligned pole as far from VPOS as observed
    ra0 = 0*u.deg
    gVPOS, gobs = [], []
    for obspole, alignpole, MCpoles in zip(nom_poles, pole_pred, MC_poles):
        fr_obs = GreatCircleICRSFrame(pole=obspole, ra0=ra0)
        fr_align = GreatCircleICRSFrame(pole=vpos_pole, ra0=ra0)
        MCtrans = MCpoles.transform_to(fr_obs)
        sc = SkyCoord(phi1=MCtrans.phi1, phi2=MCtrans.phi2, frame=fr_align)
        sep = vpos_pole.separation(sc)
        sep = np.arccos(np.abs(np.cos(sep)))
        gVPOS.append(np.sum(sep > vpos_tol) / N)
        gobs.append(np.sum(sep > vpos_pole.separation(obspole)) / N)
    cat['p_gVPOS'] = gVPOS
    cat['p_gobs'] = gobs

    results = {'table': cat, 'nom_normals': nom_poles, 'MC_normals': MC_poles,
                'opt_normals': pole_pred, 'MC_hvals': hvals}

    with open(outfile, 'wb') as f:
        pickle.dump(results, f)
