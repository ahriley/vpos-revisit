import matplotlib.pyplot as plt
import seaborn as sns
import astropy.units as u
import pickle
import utils
import astropy.coordinates as coord

sns.set_context("paper", font_scale=1.0)
big = {'s': 50, 'edgecolors': '0.3', 'linewidths': 1, 'zorder': 100}
small = {'s': 3, 'alpha': 0.05, 'rasterized': True}

dwarf_color = 'limegreen'
stream_color = 'lightcoral'

plt.figure(figsize=(8,6), dpi=100)
ax = plt.subplot(111, projection='aitoff')
ax = utils.aitoff(ax, ticklabels=True)
utils.plot_vpos(ax)

# load and plot dwarf poles (only ones that were in Pawlowski2012)
comp = 'satellites'
result_file = 'data/processed/'+comp+'-output.pkl'
with open(result_file, 'rb') as f:
    result = pickle.load(f)
    table = result['table']
    nom_normals = result['nom_normals']
    MC_normals = result['MC_normals']

inp12 = ['Carina I', 'Draco I', 'Fornax', 'Leo I', 'Leo II', 'LMC', 'SMC',
         'Sagittarius I', 'Sculptor', 'Sextans', 'Ursa Minor']
sel = [a in inp12 for a in table.index]

for nom, sc in zip(nom_normals[sel], MC_normals[sel]):
    utils.plot_aitoff(ax, sc.l, sc.b, color=dwarf_color, **small)
    utils.plot_aitoff(ax, nom.l, nom.b, color=dwarf_color, **big)

# load and plot stream normals (from Pawlowski2012 data)
comp = 'Pawlowski2012'
result_file = 'data/processed/'+comp+'-output.pkl'
with open(result_file, 'rb') as f:
    result = pickle.load(f)
    table = result['table']
    nom_normals = result['nom_normals']
    MC_normals = result['MC_normals']

sel = [True] * len(table)

low = 120*u.deg
for nom, sc in zip(nom_normals[sel], MC_normals[sel]):
    utils.plot_aitoff(ax, sc.l, sc.b, lower=low, color=stream_color, **small)
    utils.plot_aitoff(ax, nom.l, nom.b, lower=low, color=stream_color, **big)

# add Sagittarius stream
sag = coord.SkyCoord(l=273.8, b=-13.5, unit='deg', frame='galactic')
utils.plot_aitoff(ax, sag.l, sag.b, lower=low, color=stream_color, **big)

# YH GC disk from Pawlowski+12
yh = coord.SkyCoord(l=144, b=-4.3, unit='deg', frame='galactic')
utils.plot_aitoff(ax, yh.l, yh.b, color='navy', marker='D', s=200,
                  facecolors='none', linewidths=2, zorder=100)
utils.plot_aitoff(ax, yh.l, yh.b, color='navy', marker='+', s=200,
                  linewidths=2, zorder=100)

# minor axis of MW satellites from Pawlowski+15
sats = coord.SkyCoord(l=164, b=-6.9, unit='deg', frame='galactic')
utils.plot_aitoff(ax, sats.l, sats.b, color='purple', marker='h', s=300,
                  facecolors='none', linewidths=2, zorder=100)
utils.plot_aitoff(ax, sats.l, sats.b, color='purple', marker='+', s=300,
                  linewidths=2, zorder=100)

plt.savefig('plots/pdfs/vpos-classical.pdf', bbox_inches='tight')
plt.savefig('plots/pngs/vpos-classical.png', bbox_inches='tight')
