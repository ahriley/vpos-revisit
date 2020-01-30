import numpy as np
import pandas as pd
import astropy.units as u
import pickle
import utils
import astropy.coordinates as coord

# convert stream information to latex-formatted row of table
def stream_to_tablerow(index):
    name = table.index[index]
    row = table.iloc[index]
    sc = endpts[index]

    low, med, high = np.percentile(seps[index], q=[15.9, 50, 84.1])
    widthstring = '{:.1f}'.format(row['width']) if row['width'] != 0 else '--'
    pre = '\\multirow{2}{*}{'

    mapper = {'gold': 1, 'silver': 2, 'red': 3}

    # checking reporting range 120 < l < 300
    l = nom_normals[index].l
    b = nom_normals[index].b
    if (l < 120*u.deg) or (l > 300*u.deg):
        l = l + 180*u.deg
        b = -b

    # formatting data from pandas row to latex string
    string = '\t' + pre + row.name + '} & '
    string += pre + str(mapper[row['flag']]) + '} & '
    string += '{:.2f} & '.format(sc[0].ra.value)
    string += '{:.2f} & '.format(sc[0].dec.value)
    string += '${:.1f} \\pm {:.1f}$ & '.format(row['dist1'], row['e_dist1'])
    string += '${:.1f}$ & '.format(row['e_lb1'])
    string += pre + '{:.1f}'.format(row['length']) + '} & '
    string += pre + widthstring + '} & '
    string += pre + '{:.1f}'.format(l.value) + '} & '
    string += pre + '{:.1f}'.format(b.value) + '} & '
    string += pre + '${:.1f}^{{+{:.1f}}}_{{-{:.1f}}}$'.format(med, high-med, med-low) + '} & '
    string += pre + '{:.3f}'.format(row['p_inVPOS']) + '} & '
    string += pre + '\\citet{' + row['reference'] + '}} \\\\\n'
    string += '\t &  & '
    string += '{:.2f} & '.format(sc[1].ra.value)
    string += '{:.2f} & '.format(sc[1].dec.value)
    string += '${:.1f} \\pm {:.1f}$ & '.format(row['dist2'], row['e_dist2'])
    string += '${:.1f}$ & '.format(row['e_lb2'])
    string += ' &  &  &  &  &  &  \\\\\n'
    return string

# generating stream table, given start/end desired
def make_table(filename, end, start=0):
    with open(filename, 'w') as f:
        f.write('\\begin{tabular}{lcrrccccccccr}\n\t\\toprule\n')
        f.write('\t'+col_string+' \\\\\n')
        f.write('\t'+unit_string+' \\\\\n\t\\midrule\n')
        for ii in range(start, end):
            row = stream_to_tablerow(index=ii)
            f.write(row)
        f.write('\t\\bottomrule\n\\end{tabular}\n')

comp = 'streams'
result_file = 'data/processed/'+comp+'-output.pkl'

# load results
with open(result_file, 'rb') as f:
    result = pickle.load(f)
    table = result['table']
    nom_normals = result['nom_normals']
    MC_normals = result['MC_normals']

vpos_pole = utils.vpos['pole']
seps = np.arccos(np.abs(np.cos(MC_normals.separation(vpos_pole))))
seps = seps.to(u.deg).value
endpts = coord.SkyCoord(ra=table[['lon1', 'lon2']].values,\
                        dec=table[['lat1', 'lat2']].values, unit='deg')

# column names and units
cols = ['Name', 'Class', 'RA', 'Dec', 'Distance', '$\Delta \\theta$', 'Length',
        'Width', '$l_\\text{normal}$', '$b_\\text{normal}$',
        '$\\theta_\\text{obs}$', '$p_\\text{inVPOS}$', 'Ref.']
units = ['', '', '[deg]', '[deg]', '[kpc]', '[deg]', '[deg]', '[deg]', '[deg]',
         '[deg]', '[deg]', '', '']
col_string = ' & '.join(cols)
unit_string = ' & '.join(units)

# first few rows as an example in the paper text
make_table('tables/tex/streamprops.tex', end=4)

# all rows, for the Appendix (split into two tables)
split = 35
make_table('tables/tex/streamprops-full1.tex', end=split)
make_table('tables/tex/streamprops-full2.tex', end=len(table), start=split)
