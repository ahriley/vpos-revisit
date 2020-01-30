import matplotlib.pyplot as plt
import seaborn as sns
import astropy.units as u
import pickle
import utils
import astropy.coordinates as coord

sns.set_context("paper", font_scale=1.0)
bigdots = {'s': 75, 'edgecolors': '0.3', 'linewidths': 1, 'zorder': 100}
smalldots = {'s': 3, 'alpha': 0.02, 'rasterized': True}

comp = 'streams'
result_file = 'data/processed/'+comp+'-output.pkl'
with open(result_file, 'rb') as f:
    result = pickle.load(f)
    nom_normals = result['nom_normals']
    MC_normals = result['MC_normals']

plt.figure(figsize=(8,6))
ax = plt.subplot(111, projection='aitoff')
ax = utils.aitoff(ax)
utils.plot_vpos(ax, counter=False)
colors = sns.color_palette("husl", len(nom_normals))
low = 120*u.deg
for nom, sc, c in zip(nom_normals, MC_normals, colors):
    utils.plot_aitoff(ax, sc.l, sc.b, lower=low, color=c, **smalldots)
    utils.plot_aitoff(ax, nom.l, nom.b, lower=low, color=c, **bigdots)

# add Sagittarius (choosing a color that is different from its neighbors)
sag = coord.SkyCoord(l=273.8, b=-13.5, unit='deg', frame='galactic')
utils.plot_aitoff(ax, sag.l, sag.b, lower=low, color=colors[30], **bigdots)

plt.savefig('plots/pdfs/stream-normals.pdf', bbox_inches='tight')
plt.savefig('plots/pngs/stream-normals.png', bbox_inches='tight')
