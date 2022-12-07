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
data_slice = (slice(None),0,0,0)
for key in f['tasks']:
    data[key] = f['tasks/'+key][data_slice]
f.close()

MHD = 'ME' in data

if args['--times']:
    subrange = True
    t_min, t_max = args['--times'].split(',')
    t_min = float(t_min)
    t_max = float(t_max)
    print("plotting over range {:g}--{:g}, data range {:g}--{:g}".format(t_min, t_max, min(t), max(t)))
else:
    subrange = False

energy_keys = ['KE']
energy_keys.append('PE')
if MHD:
    energy_keys.insert(1, 'ME')

fig_E, ax_E = plt.subplots(nrows=2, sharex=True)
for key in energy_keys:
    ax_E[0].plot(t, data[key], label=key)
for key in energy_keys[:-1]:
    ax_E[1].plot(t, data[key], label=key)

for ax in ax_E:
    if subrange:
        ax.set_xlim(t_min,t_max)
    ax.set_xlabel('time')
    ax.set_ylabel('energy density')
    ax.legend(loc='lower left')
fig_E.tight_layout()
fig_E.savefig('{:s}/energies.pdf'.format(str(output_path)))
fig_E.savefig('{:s}/energies.png'.format(str(output_path)), dpi=300)
for ax in ax_E:
    ax.set_yscale('log')
fig_E.savefig('{:s}/log_energies.pdf'.format(str(output_path)))
fig_E.savefig('{:s}/log_energies.png'.format(str(output_path)), dpi=300)

if 'DRKE' in data:
    fluc_energy_keys = ['DRKE','MCKE','FKE']
    for key in fluc_energy_keys:
        ax_E[0].plot(t, data[key], label=key)
    for key in fluc_energy_keys[:-1]:
        ax_E[1].plot(t, data[key], label=key)
    ax_E[0].legend()
    ax_E[1].legend()
    ymin, ymax = ax_E[1].get_ylim()
    ax_E[1].set_ylim(max(ymin, 1e-14), min(ymax,1))
    fig_E.tight_layout()
    fig_E.savefig('{:s}/log_energies_fluc.pdf'.format(str(output_path)))

fig_tau, ax_tau = plt.subplots(nrows=2, sharex=True)
for i in range(2):
    ax_tau[i].plot(t, data['τ_u'], label=r'$\tau_{u}$')
    ax_tau[i].plot(t, data['τ_s'], label=r'$\tau_{s}$')
    ax_tau[i].plot(t, data['τ_p'], label=r'$\tau_{p}$')
    if MHD:
        ax_tau[i].plot(t, data['τ_A'], label=r'$\tau_{A}$')
        ax_tau[i].plot(t, data['τ_φ'], label=r'$\tau_{\phi}$')

for ax in ax_tau:
    if subrange:
        ax.set_xlim(t_min,t_max)
    ax.set_xlabel('time')
    ax.set_ylabel(r'$<\tau>$')
    ax.legend(loc='lower left')
ax_tau[1].set_yscale('log')
ylims = ax_tau[1].get_ylim()
ax_tau[1].set_ylim(max(1e-14, ylims[0]), ylims[1])
fig_tau.tight_layout()
fig_tau.savefig('{:s}/tau_error.pdf'.format(str(output_path)))
fig_tau.savefig('{:s}/tau_error.png'.format(str(output_path)), dpi=300)

fig_L, ax_L = plt.subplots(nrows=2, sharex=True)
ax_L[0].plot(t, data['Lx'], label='Lx')
ax_L[0].plot(t, data['Ly'], label='Ly')
ax_L[0].plot(t, data['Lz'], label='Lz')
ax_L[1].plot(t, np.abs(data['Lx']), label='Lx')
ax_L[1].plot(t, np.abs(data['Ly']), label='Ly')
ax_L[1].plot(t, np.abs(data['Lz']), label='Lz')

for ax in ax_L:
    if subrange:
        ax.set_xlim(t_min,t_max)
    ax.set_ylabel('Angular Momentum')
    ax.legend()
ax_L[1].set_xlabel('time')
ax_L[1].set_yscale('log')
fig_L.tight_layout()
fig_L.savefig('{:s}/angular_momentum.pdf'.format(str(output_path)))
fig_L.savefig('{:s}/angular_momentum.png'.format(str(output_path)), dpi=300)


if 'Λz' in data:
    fig_L, ax_L = plt.subplots(nrows=2, sharex=True)
    ax_L[0].plot(t, data['Λx'], label='Λx')
    ax_L[0].plot(t, data['Λy'], label='Λy')
    ax_L[0].plot(t, data['Λz'], label='Λz')
    ax_L[1].plot(t, np.abs(data['Λx']), label='Λx')
    ax_L[1].plot(t, np.abs(data['Λy']), label='Λy')
    ax_L[1].plot(t, np.abs(data['Λz']), label='Λz')

    for ax in ax_L:
        if subrange:
            ax.set_xlim(t_min,t_max)
        ax.set_ylabel('Angular momentum flux moment')
        ax.legend(loc='lower left')
    ax_L[1].set_xlabel('time')
    ax_L[1].set_yscale('log')
    fig_L.tight_layout()
    fig_L.savefig('{:s}/angular_momentum_flux_moment.pdf'.format(str(output_path)))
    fig_L.savefig('{:s}/angular_momentum_flux_moment.png'.format(str(output_path)), dpi=300)


fig_f, ax_f = plt.subplots(nrows=2, sharex=True)
for ax in ax_f:
    ax.plot(t, data['Re'], label='Re')
    ax_r = ax.twinx()
    ax_r.plot(t, data['Ro'], label='Ro', color='tab:orange')
    if subrange:
        ax.set_xlim(t_min,t_max)
    ax.set_xlabel('time')
    ax.set_ylabel('fluid parameters')
    ax.legend(loc='lower left')

ax_f[1].set_yscale('log')
ax_r.set_yscale('log') # relies on it being the last instance; poor practice

fig_f.tight_layout()
fig_f.savefig('{:s}/Re_and_Ro.pdf'.format(str(output_path)))
fig_f.savefig('{:s}/Re_and_Ro.png'.format(str(output_path)), dpi=300)

benchmark_set = ['KE', 'PE', 'Re', 'Ro', 'Lx', 'Ly', 'Lz', 'τ_u', 'τ_s', 'τ_p']
if MHD:
    benchmark_set.insert(1, 'ME/KE')
    benchmark_set.insert(1, 'ME')
    benchmark_set += ['τ_A', 'τ_φ']
    data['ME/KE'] = data['ME']/data['KE']

if 'Λz' in data:
    benchmark_set.append('Λx')
    benchmark_set.append('Λy')
    benchmark_set.append('Λz')

i_ten = int(0.9*data[benchmark_set[0]].shape[0])
print("total simulation time {:6.2g}".format(t[-1]-t[0]))
print("benchmark values (averaged from {:g}-{:g})".format(t[i_ten], t[-1]))
for benchmark in benchmark_set:
    try:
        print("{:3s} = {:20.12e} +- {:4.2e}".format(benchmark, np.mean(data[benchmark][i_ten:]), np.std(data[benchmark][i_ten:])))
    except:
        print("{:3s} missing".format(benchmark))
