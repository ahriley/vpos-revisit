import numpy as np
import pandas as pd
import astropy.units as u
import pickle
import utils

# convert GC information to latex-formatted row of table
def GC_to_tablerow(index):
    name = table.index[index]
    row = table.iloc[index]
    normal = nom_normals[index]

    low, med, high = np.percentile(seps[index], q=[15.9, 50, 84.1])
    val = med if normal.separation(utils.vpos['pole']) < 90*u.deg else -med

    string = '\t' + name + ' & ' + row['type']
    string += ' & {:.1f}'.format(normal.l.deg)
    string += ' & {:.1f}'.format(normal.b.deg)
    string += ' & {:.1f}'.format(row['sph_std'])
    string += ' & {:.1f}'.format(row['theta_pred'])
    # string += ' & {:.1f}'.format(row['theta_obs'])
    string += ' & ${:.1f}^{{+{:.1f}}}_{{-{:.1f}}}$'.format(val, high-med, med-low)
    string += ' & {:.3f}'.format(row['p_inVPOS'])
    string += ' & {:.3f}'.format(row['p_gVPOS'])
    string += ' & {:.3f}'.format(row['p_gobs'])
    string += ' \\\\\n'

    return string

# generating stream table, given start/end desired
def make_table(filename, end, start=0):
    with open(filename, 'w') as f:
        f.write('\\begingroup\n\\renewcommand{\\arraystretch}{1.25}\n')
        f.write('\\begin{tabular}{lcccccrccc}\n\t\\toprule\n')
        f.write('\t'+col_string+' \\\\\n')
        f.write('\t'+unit_string+' \\\\\n\t\\midrule\n')
        for ii in range(start, end):
            row = GC_to_tablerow(index=ii)
            f.write(row)
        f.write('\t\\bottomrule\n\\end{tabular}\n')
        f.write('\\endgroup')

comp = 'globulars'
result_file = 'data/processed/'+comp+'-output.pkl'

# load results
with open(result_file, 'rb') as f:
    result = pickle.load(f)
    table = result['table']
    nom_normals = result['nom_normals']
    MC_normals = result['MC_normals']

seps = np.arccos(np.abs(np.cos(MC_normals.separation(utils.vpos['pole']))))
seps = seps.to(u.deg).value

# GCprops: results for individual GCs
cols = ['Name', 'Type', '$l_\\text{pole}$', '$b_\\text{pole}$',
        '$\\Delta_\\text{pole}$', '$\\theta_\\text{pred}$',
        '$\\theta_\\text{obs}$', '$p_\\text{inVPOS}$', '$p_\\text{>VPOS}$',
        '$p_\\text{>obs}$']
units = ['', '', '[deg]', '[deg]', '[deg]', '[deg]', '[deg]', '', '', '']
col_string = ' & '.join(cols)
unit_string = ' & '.join(units)

# short version of the table for the text
make_table('tables/tex/GCprops.tex', end=5)

# full version for the Appendix (split into 3 tables)
make_table('tables/tex/GCprops-full1.tex', end=50)
make_table('tables/tex/GCprops-full2.tex', end=100, start=50)
make_table('tables/tex/GCprops-full3.tex', end=len(table), start=100)
