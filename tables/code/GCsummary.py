import numpy as np
import pandas as pd
import astropy.units as u
import pickle
import utils

comp = 'globulars'
result_file = 'data/processed/'+comp+'-output.pkl'

# load results
with open(result_file, 'rb') as f:
    result = pickle.load(f)
    table = result['table']
    nom_normals = result['nom_normals']
    MC_normals = result['MC_normals']
    opt_normals = result['opt_normals']

# GCsummary: summary results broken down by Type
toprow = 'Type & $N$ & $f_\\text{inVPOS}$ & $f_\\text{notVPOS}$ & $f_\\text{inconclusive}$ & $\\sum p_\\text{inVPOS} / N$'
pops = ['OH', 'YH', 'BD', 'SG', 'UN', 'All', 'F18']
with open('tables/tex/GCsummary.tex', 'w') as f:
    f.write('\\begin{tabular}{cccccc}\n\t\\toprule\n')
    f.write('\t'+toprow+' \\\\\n\t\\midrule\n')
    for pop in pops:
        if pop == 'F18':
            fritz = pd.read_csv('data/Fritz2018-table4.csv', index_col=0)
            # add 2 for the Magellanic Clouds
            Ntot, Nyes, Nnot, Nidk = [39+2, 17+2, 12, 10]
            Nsum = np.sum(fritz['p_inVPOS']) + 2
            f.write('\t\\midrule\n')
        else:
            if pop == 'All':
                f.write('\t\\midrule\n')
                mini = table
            else:
                mini = table[table['type'] == pop]
            Ntot = len(mini)
            Nyes = np.sum(mini['p_inVPOS'] > 0.5)
            Nnot = np.sum(mini['p_inVPOS'] < 0.05)
            Nidk = Ntot - Nyes - Nnot
            Nsum = np.sum(mini['p_inVPOS'])

        string = '\t' + pop + ' & ' + str(Ntot) + ' & '
        string += '{:.3f} & '.format(Nyes/Ntot)
        string += '{:.3f} & '.format(Nnot/Ntot)
        string += '{:.3f} & '.format(Nidk/Ntot)
        string += '{:.3f} \\\\\n'.format(Nsum/Ntot)
        f.write(string)
    f.write('\t\\bottomrule\n\\end{tabular}\n')
