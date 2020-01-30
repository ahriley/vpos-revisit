import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord, cartesian_to_spherical
import astropy.coordinates as coord
import astropy.units as u
import utils
from gala.coordinates import GreatCircleICRSFrame, pole_from_endpoints
import pickle

# comment for different random seed
utils.set_paper_random_seed()

# number of samples, save location, catalog
N = 2000
outfiles = ['data/processed/streams-output.pkl',
            'data/processed/Pawlowski2012-output.pkl']
infiles = ['data/streams.csv', 'data/Pawlowski2012.csv']

# assumed MW/VPOS properties
vpos_pole, vpos_tol = utils.vpos['pole'], utils.vpos['tol']
galcen_frame = utils.galcen['frame']

for infile, outfile in zip(infiles, outfiles):
    endpts = pd.read_csv(infile, index_col=0)

    # nominal stream pole (using most likely values)
    lon, lat, dist = [], [], []
    for name, pts in endpts.iterrows():
        lon.append([pts['lon'+str(i+1)] for i in range(2)])
        lat.append([pts['lat'+str(i+1)] for i in range(2)])
        dist.append([pts['dist'+str(i+1)] for i in range(2)])
    pts = SkyCoord(lon*u.deg, lat*u.deg,\
                   distance=dist*u.kpc, frame=pts['frame'])
    endpts['length'] = pts[:,0].separation(pts[:,1]).deg
    pts = pts.transform_to(galcen_frame)
    r_MW, b_MW, l_MW = coord.cartesian_to_spherical(pts.x, pts.y, pts.z)
    pts_MW = SkyCoord(l=l_MW, b=b_MW, frame='galactic')
    l, b = [], []
    for stream in pts_MW:
        stream = stream.transform_to(coord.ICRS)
        pole = pole_from_endpoints(stream[0], stream[1])
        pole = pole.transform_to(coord.Galactic)
        l.append(pole.l)
        b.append(pole.b)
    poles_MW = SkyCoord(l=l, b=b, frame='galactic')
    endpts['dist_gal'] = np.mean(r_MW, axis=1)

    # Monte Carlo sampling for stream normals
    N_up = int(N*1.1)
    lon_arr = []; lat_arr = []
    for name, pts in endpts.iterrows():
        temp = []
        for i in ['1', '2']:
            offset = np.random.normal(loc=0, scale=pts['e_lb'+i], size=N_up)
            phi = np.random.uniform(size=N_up)*2*np.pi

            ra = pts['lon'+i] + offset * np.cos(phi)
            dec = pts['lat'+i] + offset * np.sin(phi)
            sel = (dec > -90) & (dec < 90)
            ra = ra[sel][:N]
            dec = dec[sel][:N]
            dist = np.random.normal(loc=pts['dist'+i],\
                                    scale=pts['e_dist'+i], size=N_up)
            dist = dist[dist > 0][:N]
            assert len(dist) == N
            assert len(dec) == N

            sc = SkyCoord(ra*u.deg, dec*u.deg,\
                          distance=dist*u.kpc, frame=pts['frame'])
            sc = sc.transform_to(galcen_frame)
            cart = np.array([sc.x, sc.y, sc.z])
            temp.append(cart)
        temp = np.array(temp)
        cross = np.cross(temp[0].T, temp[1].T)
        x, y, z = cross.T / np.linalg.norm(cross, axis=1)
        r, lat, lon = cartesian_to_spherical(x, y, z)
        lon_arr.append(lon.deg)
        lat_arr.append(lat.deg)
    lon = np.array(lon_arr)
    lat = np.array(lat_arr)
    norms = SkyCoord(lon, lat, unit='deg', frame='galactic')

    # spherical standard distance
    sep = poles_MW.separation(norms.T).T
    endpts['sph_std'] = np.sqrt(np.sum(sep.deg**2, axis=1) / N)
    sep = np.arccos(np.abs(np.cos(sep))).to(u.deg).value
    endpts['sph_std_abs'] = np.sqrt(np.sum(sep**2, axis=1) / N)

    # theta_obs : angle between VPOS normal and nominal stream normal
    sep = vpos_pole.separation(poles_MW)
    endpts['theta_obs'] = np.arccos(np.abs(np.cos(sep))).to(u.deg)

    # theta_pred : mininum angle from VPOS pole to track w/ endpoint as pole
    phi1 = np.arange(0,360,0.001)*u.deg
    theta_pred, pole_pred = [], []
    for stream_ends in pts_MW:
        theta_pred_stream, pole_pred_stream = [], []
        for i in range(2):
            end = stream_ends[i].transform_to(coord.ICRS)
            streamfr = GreatCircleICRSFrame(pole=end, ra0=0*u.deg)
            gc = SkyCoord(phi1=phi1, phi2=0*u.deg, frame=streamfr)
            assert np.allclose(end.separation(gc).deg, 90)
            gc = gc.transform_to(coord.Galactic)
            ii = np.argmin(vpos_pole.separation(gc).deg)
            theta_pred_stream.append(vpos_pole.separation(gc[ii]).deg)
            pole_pred_stream.append([gc[ii].l.deg, gc[ii].b.deg])
        theta_pred.append(theta_pred_stream)
        pole_pred.append(pole_pred_stream)
    theta_pred = np.array(theta_pred)
    endpts['theta_pred'] = np.max(theta_pred, axis=1)
    pole_pred = np.array(pole_pred)
    iimax = np.argmax(theta_pred, axis=1)
    temp = np.array([polepair[ii] for polepair,ii in zip(pole_pred, iimax)])
    pole_pred = SkyCoord(temp[:,0], temp[:,1], unit='deg', frame='galactic')
    assert (vpos_pole.separation(pole_pred).deg == endpts['theta_pred']).all()
    assert (endpts['theta_obs'] > endpts['theta_pred']).all()

    # p_inVPOS : how many Monte Carlo poles lie within vpos_tol
    sep = vpos_pole.separation(norms)
    sep = np.arccos(np.abs(np.cos(sep)))
    endpts['p_inVPOS'] = np.sum(sep < vpos_tol, axis=1) / N

    # p_gVPOS : falsely finding a normal to be misaligned with VPOS
    # p_gobs : finding intrinsically aligned pole as far from VPOS as observed
    ra0 = 0*u.deg
    gVPOS, gobs = [], []
    for obspole, alignpole, MCpoles in zip(poles_MW, pole_pred, norms):
        fr_obs = GreatCircleICRSFrame(pole=obspole, ra0=ra0)
        fr_align = GreatCircleICRSFrame(pole=vpos_pole, ra0=ra0)
        MCtrans = MCpoles.transform_to(fr_obs)
        sc = SkyCoord(phi1=MCtrans.phi1, phi2=MCtrans.phi2, frame=fr_align)
        sep = vpos_pole.separation(sc)
        sep = np.arccos(np.abs(np.cos(sep)))
        gVPOS.append(np.sum(sep > vpos_tol) / N)
        gobs.append(np.sum(sep > vpos_pole.separation(obspole)) / N)
    endpts['p_gVPOS'] = gVPOS
    endpts['p_gobs'] = gobs

    results = {'table': endpts, 'nom_normals': poles_MW, 'MC_normals': norms,
                'opt_normals': pole_pred}

    with open(outfile, 'wb') as f:
        pickle.dump(results, f)
