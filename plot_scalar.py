"""
Plot scalar outputs from scalar_output.h5 file.

Usage:
    plot_scalar.py <file> [options]

Options:
    --times=<times>      Range of times to plot over; pass as a comma separated list with t_min,t_max.  Default is whole timespan.
    --output=<output>    Output directory; if blank, a guess based on <file> location will be made.
"""
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pathlib
import h5py

import logging
logger = logging.getLogger(__name__.split('.')[-1])

from docopt import docopt
args = docopt(__doc__)
file = args['<file>']

if args['--output'] is not None:
    output_path = pathlib.Path(args['--output']).absolute()
else:
    data_dir = args['<file>'].split('/')[0]
    data_dir += '/'
    output_path = pathlib.Path(data_dir).absolute()

f = h5py.File(file, 'r')
data = {}
t = f['scales/sim_time'][:]
for key in f['tasks']:
    data[key] = f['tasks/'+key][:]
f.close()

if args['--times']:
    subrange = True
    t_min, t_max = args['--times'].split(',')
    t_min = float(t_min)
    t_max = float(t_max)
    print("plotting over range {:g}--{:g}, data range {:g}--{:g}".format(t_min, t_max, min(t), max(t)))
else:
    subrange = False

energy_keys = ['T0', 'E0']

fig_E, ax_E = plt.subplots()
for key in energy_keys:
    ax_E.plot(t, data[key], label=key)

i_ten = int(0.9*data['E0'].shape[0])
print("KE benchmark {:14.12g} +- {:4.2g} (averaged from {:g}-{:g})".format(np.mean(data['E0'][i_ten:]), np.std(data['E0'][i_ten:]), t[i_ten], t[-1]))
print("KE benchmark {:14.12g} (at t={:g})".format(np.mean(data['E0'][-1]), t[-1]))
print("total simulation time {:6.2g}".format(t[-1]-t[0]))

if subrange:
    ax_E.set_xlim(t_min,t_max)
ax_E.set_xlabel('time')
ax_E.set_ylabel('energy density')
ax_E.legend(loc='lower left')
fig_E.savefig('{:s}/energies.pdf'.format(str(output_path)))
ax_E.set_yscale('log')
fig_E.savefig('{:s}/log_energies.pdf'.format(str(output_path)))
