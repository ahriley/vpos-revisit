import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import utils
import astropy.coordinates as coord

sns.set_context("paper", font_scale=1.0)

studies = ['streams', 'Pawlowski2012']
colors = ['C0', 'C1']
labels = ['This work', 'Pawlowski+12']

sagcoords = [[273.8, -14.5], [273.8, -13.5]]

plt.figure(figsize=(4,4), dpi=300)
for study, c, label, sag in zip(studies, colors, labels, sagcoords):
    result_file = 'data/processed/'+study+'-output.pkl'
    with open(result_file, 'rb') as f:
        result = pickle.load(f)
        table = result['table']
        MC_normals_orig = result['MC_normals']

    # deal with Sagittarius separately
    N = MC_normals_orig.shape[1]
    l_MC = np.vstack((MC_normals_orig.l.deg, np.full(N, sag[0])))
    b_MC = np.vstack((MC_normals_orig.b.deg, np.full(N, sag[1])))
    MC_normals = coord.SkyCoord(l=l_MC, b=b_MC, unit='deg', frame='galactic')

    sep = MC_normals.separation(utils.vpos['pole'])
    sep = 1 - np.abs(np.cos(sep))

    cdfs = []
    for vals in sep.T:
        cdfs.append(np.sort(vals))
    cdfs = np.array(cdfs)

    conf = np.percentile(cdfs, q=[2.5, 16, 84, 97.5], axis=0)

    yvals = np.linspace(0,1,len(table)+1)
    plt.fill_betweenx(yvals, conf[0], conf[3], color=c, alpha=0.3)
    plt.fill_betweenx(yvals, conf[1], conf[2], color=c, alpha=0.5, label=label)
plt.plot(yvals, yvals, 'k--', label='Isotropic')
plt.axvline(0.2, c='0.5', label=r'$\theta_{\mathrm{inVPOS}}$')
plt.xlim(0,1)
plt.ylim(0,1)
plt.legend(loc='lower right')
plt.xlabel(r'$1 - \cos\theta$');
plt.savefig('plots/pdfs/stream-cdf.pdf', bbox_inches='tight')
plt.savefig('plots/pngs/stream-cdf.png', bbox_inches='tight')
