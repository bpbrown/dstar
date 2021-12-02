"""
Plot radial profile outputs from joint analysis file.

Usage:
    plot_slices.py <files>... [options]

Options:
    --MESA               Overlay MESA L_MLT
    --output=<output>    Output directory; if blank a guess based on likely case name will be made
"""
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pathlib
import h5py
import scipy.integrate as sci

import logging
logger = logging.getLogger(__name__.split('.')[-1])

from docopt import docopt
args = docopt(__doc__)

import dedalus.public as de
from dedalus.tools import post
from dedalus.tools.general import natural_sort
files = natural_sort(args['<files>'])
case = args['<files>'][0].split('/')[0]

if args['--output'] is not None:
    output_path = pathlib.Path(args['--output']).absolute()
else:
    data_dir = case +'/'
    output_path = pathlib.Path(data_dir).absolute()

fields = ['s(r)', 'F_h(r)', 'F_κ(r)', 'F_KE(r)', 'F_source(r)']

def accumulate_files(filename,start,count,file_list):
    if start==0:
        file_list.append(filename)

file_list = []
post.visit_writes(files,  accumulate_files, file_list=file_list)
logger.debug(file_list)

data = {}
r = None
times = None
for file in file_list:
    logger.debug("opening file: {}".format(file))
    f = h5py.File(file, 'r')
    data_slices = (slice(None), 0, 0, slice(None))
    for task in f['tasks']:
        if '(r)' in task: # fluxes denoted with 'f(r)'
            logger.info("task: {}".format(task))
            if task in data:
                data[task] = np.append(data[task], f['tasks'][task][data_slices], axis=0)
            else:
                data[task] = np.array(f['tasks'][task][data_slices])
            if r is None:
                r = f['tasks'][task].dims[3][0][:]
    if times is None:
        times = f['scales/sim_time'][:]
    else:
        times = np.append(times, f['scales/sim_time'][:])
    f.close()

print(file_list)
for task in data:
    print(task, data[task].shape)

def time_avg(f, axis=0):
    n_avg = f.shape[axis]
    return np.squeeze(np.sum(f, axis=axis))/n_avg

s_avg = time_avg(data['s(r)'])
fig_s, ax_s = plt.subplots(figsize=(4.5,4/1.5))
fig_s.subplots_adjust(top=0.9, right=0.95, bottom=0.2, left=0.15)

for si in data['s(r)']:
    ax_s.plot(r, si, alpha=0.3)
ax_s.plot(r, s_avg, linewidth=2, color='black')
fig_s.savefig('{:s}/thermal_profile.pdf'.format(str(output_path)))

F_h = time_avg(data['F_h(r)'])
F_κ = time_avg(data['F_κ(r)'])
F_KE = time_avg(data['F_KE(r)'])
Q_source = time_avg(data['F_source(r)'])
#F_μ_avg = time_avg(data['<Fμr>'])

fig_Q, ax_Q = plt.subplots(figsize=(4.5,4/1.5))
fig_Q.subplots_adjust(top=0.9, right=0.8, bottom=0.2, left=0.2)
ax_Q.plot(r, Q_source)
ax_Q.set_ylabel(r'$\mathcal{S}(r)$')
ax_Q.set_xlabel(r'$r/R$')
fig_Q.savefig('{:s}/source_function.pdf'.format(str(output_path)))

L_S = 4*np.pi*sci.cumtrapz(r**2*Q_source, x=r, initial=0)
norm = 1/L_S[-1]

L_h = 4*np.pi*r**2*F_h*norm
L_κ = 4*np.pi*r**2*F_κ*norm
L_KE = 4*np.pi*r**2*F_KE*norm
#L_μ = 4*np.pi*r**2*theta_avg(F_μ_avg)*norm
L_S = L_S*norm

fig_hr, ax_hr = plt.subplots(figsize=(4.5,4/1.5))
fig_hr.subplots_adjust(top=0.9, right=0.95, bottom=0.2, left=0.15)
ax_hr.plot(r, L_h + L_KE + L_κ, color='black', label=r'$L_\mathrm{tot}$', linewidth=3)
ax_hr.plot(r, L_h, label=r'$L_\mathrm{h}$')
ax_hr.plot(r, L_KE, label=r'$L_\mathrm{KE}$')
ax_hr.plot(r, L_κ, label=r'$L_\kappa$')
ax_hr.plot(r, L_S, label=r'$L_\mathcal{S}$')
ax_hr.axhline(y=0, linestyle='dashed', color='darkgrey', zorder=0)
#ax_hr.plot(r, L_μ, label=r'$L_\mu$')
if args['--MESA']:
    from mesa_structure import mesa
    max_r = 0.85
    mesa_structure = mesa('./mdwarf', max_fractional_r = max_r, deg=8)
    r_mesa = mesa_structure['r']
    L_MLT = mesa_structure['L_conv']/mesa_structure['L_star']
    L_S_mesa = mesa_structure['L_S_tot']/mesa_structure['L_S_tot'][-1]
    ax_hr.plot(r_mesa, L_MLT, color='darkslateblue', linestyle='dashed', label='MESA MLT')
    ax_hr.plot(r_mesa, L_S_mesa, color='firebrick', linestyle='dashed', label='MESA S')
ax_hr.legend()
ax_hr.set_ylabel(r'$L/L_{\scrS(r=1)}$')
ax_hr.set_xlabel(r'$r/R$')
fig_hr.savefig('{:s}/flux_balance.pdf'.format(str(output_path)))

# total_flux = data['<hr>'] + data['<Fsr>'] + data['<FKEr>'] + data['<Fμr>']
# print(total_flux.shape)
# total_flux = theta_avg(total_flux, axis=1)
# print(total_flux.shape)
# total_L_top = norm*4*np.pi*r[-1]**2*total_flux[:,-1]
# fig_t, ax_t = plt.subplots(figsize=(4.5,4/1.5))
# ax_t.plot(times, total_L_top)
# fig_t.savefig('{:s}/luminosity_trace.pdf'.format(str(output_path)))


if args['--MESA']:
    from structure import lane_emden
    structure = lane_emden(n_rho=3, m=1.5, norm_at_core=True)
    # set rho0
    ln_rho = []
    rho = []
    T = []
    for i, r_i in enumerate(r):
        rho.append(structure['rho'].interpolate(r=r_i)['g'][0])
        ln_rho.append(structure['ln_rho'].interpolate(r=r_i)['g'][0])
        T.append(structure['T'].interpolate(r=r_i)['g'][0])

    rho = np.array(rho)
    ln_rho = np.array(ln_rho)
    T = np.array(T)
    Q_S = theta_avg(Q_S)
    S = Q_S/rho/T
    S /= S[0]

    σ = 0.11510794072958948
    Q0_over_Q1 = 10.969517734412433
    Q1 = 1/(Q0_over_Q1 + 1) # normalize to 1 at r=0
    Source = (Q0_over_Q1*np.exp(-(r*0.85)**2/(2*σ**2)) + 1)*Q1

    fig, ax = plt.subplots(figsize=(4.5,4/1.5))
    fig.subplots_adjust(top=0.9, right=0.95, bottom=0.2, left=0.15)
    ax.plot(r, S, linewidth=3)
    ax.plot(r, Source)

    L_S  = 4*np.pi*sci.cumtrapz(r**2*rho*T*S, x=r, initial=0)
    L_So = 4*np.pi*sci.cumtrapz(r**2*rho*T*Source, x=r, initial=0)
    ax2 = ax.twinx()
    ax2.plot(r, L_S, linewidth=3)
    ax2.plot(r, L_So)
    ax.plot(r_mesa, L_MLT, color='darkslateblue', linestyle='dashed', label='MESA MLT')
    ax.plot(r_mesa, L_S_mesa, color='firebrick', linestyle='dashed', label='MESA S')

    fig.savefig('{:s}/source_test.pdf'.format(str(output_path)))

    fig, ax = plt.subplots(figsize=(4.5,4/1.5))
    fig.subplots_adjust(top=0.9, right=0.95, bottom=0.2, left=0.15)
    ax.plot(r, rho)
    ax.plot(r, T)

    print(max(np.log(rho)), min(np.log(rho)))
    print(max(T), min(T), np.exp(-3/1.5))
    fig.savefig('{:s}/profile_test.pdf'.format(str(output_path)))
