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

energy_keys = ['KE','PE']

fig_E, ax_E = plt.subplots(nrows=2)
for key in energy_keys:
    ax_E[0].plot(t, data[key], label=key)
ax_E[1].plot(t, data['KE'], label='KE')

for ax in ax_E:
    if subrange:
        ax.set_xlim(t_min,t_max)
    ax.set_xlabel('time')
    ax.set_ylabel('energy density')
    ax.legend(loc='lower left')
fig_E.savefig('{:s}/energies.pdf'.format(str(output_path)))
for ax in ax_E:
    ax.set_yscale('log')
fig_E.savefig('{:s}/log_energies.pdf'.format(str(output_path)))

fig_L, ax_L = plt.subplots(nrows=2)
ax_L[0].plot(t, data['Lz'], label='Lz')
ax_L[1].plot(t, np.abs(data['Lz']), label='Lz')

for ax in ax_L:
    if subrange:
        ax.set_xlim(t_min,t_max)
    ax.set_xlabel('time')
    ax.set_ylabel('Angular Momentum')
    ax.legend(loc='lower left')
ax_L[1].set_yscale('log')
fig_L.savefig('{:s}/angular_momentum.pdf'.format(str(output_path)))

benchmark_set = ['KE', 'Re', 'Ro']
i_ten = int(0.9*data[benchmark_set[0]].shape[0])
for benchmark in benchmark_set:
    print("{:s} benchmark {:14.12g} +- {:4.2g} (averaged from {:g}-{:g})".format(benchmark, np.mean(data[benchmark][i_ten:]), np.std(data[benchmark][i_ten:]), t[i_ten], t[-1]))
for benchmark in benchmark_set:
    print("{:s} benchmark {:14.12g} (at t={:g})".format(benchmark, data[benchmark][-1], t[-1]))
print("total simulation time {:6.2g}".format(t[-1]-t[0]))
