import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import utils
import astropy.coordinates as coord

sns.set_context("paper", font_scale=1.0)
bigdots = {'s': 50, 'edgecolors': '0.3', 'linewidths': 1, 'zorder': 100}
smalldots = {'s': 3, 'alpha': 0.02, 'rasterized': True}

comp = 'globulars'
result_file = 'data/processed/'+comp+'-output.pkl'

with open(result_file, 'rb') as f:
    result = pickle.load(f)
    table = result['table']
    nom_normals = result['nom_normals']
    MC_normals = result['MC_normals']

plt.figure(figsize=(8,6))
for classification, N in zip(['OH', 'YH', 'BD', 'SG'], range(4)):
    ax = plt.subplot(220+N+1, projection='aitoff')
    sel = table['type'] == classification

    ax = utils.aitoff(ax)
    utils.plot_vpos(ax)
    colors = sns.color_palette("husl", len(table[sel]))
    for nom, sc, c in zip(nom_normals[sel], MC_normals[sel], colors):
        utils.plot_aitoff(ax, sc.l, sc.b, color=c, **smalldots)
        utils.plot_aitoff(ax, nom.l, nom.b, color=c, **bigdots)
plt.tight_layout(h_pad=-7.0)
plt.savefig('plots/pdfs/GC-poles.pdf', bbox_inches='tight')
plt.savefig('plots/pngs/GC-poles.png', bbox_inches='tight')
