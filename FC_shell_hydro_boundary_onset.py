"""
Dedalus script for full sphere fully compressible convection,
using a Lane-Emden structure and internal heat source.
Designed for modelling fully-convective stars.

Usage:
    FC_hydro.py [options]

Options:
    --Ekman=<Ekman>                      Ekman number    [default: 1e-4]
    --ConvectiveRossbySq=<Co2>           Squared Convective Rossby = Ra*Ek**2/Pr [default: 1e-1]
    --Mach=<Ma>                          Mach number [default: 1e-2]
    --Prandtl=<Prandtl>                  Prandtl number  [default: 1]
    --gamma=<gamma>                      Ideal gas gamma [default: 5/3]
    --n_rho=<n_rho>                      Density scale heights [default: 3]

    --Legendre                           Use Legendre polynomials in radius

    --Ntheta=<Ntheta>                    Latitudinal modes [default: 32]
    --Nr=<Nr>                            Radial modes [default: 32]
    --dealias=<dealias>                  Degree of deailising   [default: 1.5]

    --benchmark                          Use benchmark initial conditions
    --spectrum                           Use a spectrum of benchmark perturbations
    --ell_benchmark=<ell_benchmark>      Integer value of benchmark perturbation m=+-ell [default: 3]

    --thermal_equilibrium                Start in thermal equilibrum

    --max_dt=<max_dt>                    Largest possible timestep [default: 0.25]
    --safety=<safety>                    CFL safety factor [default: 0.4]

    --run_time_sim=<run_time>            How long to run, in rotating time units
    --run_time_iter=<niter>              How long to run, in iterations

    --slice_dt=<slice_dt>                Cadence at which to output slices, in rotation times (P_rot = 4pi) [default: 10]
    --scalar_dt=<scalar_dt>              Time between scalar outputs, in rotation times (P_rot = 4pi) [default: 2]

    --restart=<restart>                  Merged chechpoint file to restart from.
                                         Make sure "--label" is set to avoid overwriting the previous run.

    --label=<label>                      Additional label for run output directory

    --tol=<tol>             Tolerance for opitimization loop [default: 1e-5]
    --eigs=<eigs>           Target number of eigenvalues to search for [default: 20]

    --ncc_cutoff=<ncc_cutoff>            Amplitude to truncate NCC terms [default: 1e-10]
    --plot_sparse                        Plot sparsity structures for L+M and it's LU decomposition
"""
import numpy as np

import pathlib
import os
import sys
import h5py
from fractions import Fraction

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.rank
ncpu = comm.size

from docopt import docopt
args = docopt(__doc__)

import logging
logger = logging.getLogger(__name__)
for system in ['matplotlib', 'h5py', 'evaluator']:
    dlog = logging.getLogger(system)
    dlog.setLevel(logging.WARNING)

data_dir = sys.argv[0].split('.py')[0]
data_dir += '_Co{}_Ma{}_Ek{}_Pr{}'.format(args['--ConvectiveRossbySq'],args['--Mach'],args['--Ekman'],args['--Prandtl'])
data_dir += '_Th{}_R{}'.format(args['--Ntheta'], args['--Nr'])
if args['--label']:
    data_dir += '_{:s}'.format(args['--label'])

import dedalus.tools.logging as dedalus_logging
dedalus_logging.add_file_handler(data_dir+'/logs/dedalus_log', 'DEBUG')

Nθ = int(args['--Ntheta'])
Nr = int(args['--Nr'])
Nφ = Nθ*2

Legendre = args['--Legendre']

N_eigs = int(float(args['--eigs']))
tol = float(args['--tol'])

ncc_cutoff = float(args['--ncc_cutoff'])

nρ = float(args['--n_rho'])
radius = 1

Ek = Ekman = float(args['--Ekman'])
Co2 = ConvectiveRossbySq = float(args['--ConvectiveRossbySq'])
Ma = float(args['--Mach'])
Ma2 = Ma*Ma
γ = gamma = float(Fraction(args['--gamma']))
Pr = Prandtl = float(args['--Prandtl'])

import dedalus.public as de
from dedalus.extras import flow_tools


logger.debug(sys.argv)
logger.debug('-'*40)
logger.info("saving data in {}".format(data_dir))
logger.info("Run parameters")
logger.info("Ek = {}, Co2 = {}, Ma = {}, Pr = {}".format(Ek,Co2,Ma,Pr))

dealias = float(args['--dealias'])

ε = Ma2
m_ad = 1/(γ-1)
m_poly = m_ad - ε
Ro = r_outer = 1
Ri = r_inner = 0.7
nh = nρ/m_poly

dtype = np.complex128
coords = de.SphericalCoordinates('phi', 'theta', 'r')
dist = de.Distributor(coords, dtype=dtype)
radii = (Ri, Ro)
if Legendre:
    basis = de.ShellBasis(coords, alpha=(0,0), shape=(Nφ, Nθ, Nr), radii=radii, dtype=dtype)
else:
    basis = de.ShellBasis(coords, shape=(Nφ, Nθ, Nr), radii=radii, dtype=dtype)
basis_ncc = basis.meridional_basis
b_S2 = basis.S2_basis()
phi, theta, r = basis.local_grids()

eφ = dist.VectorField(coords, bases=basis_ncc)
eφ['g'][0] = 1
eθ = dist.VectorField(coords, bases=basis_ncc)
eθ['g'][1] = 1
er = dist.VectorField(coords, bases=basis_ncc)
er['g'][2] = 1

p = dist.Field(name='p', bases=basis)
Υ = dist.Field(name='Υ', bases=basis)
θ = dist.Field(name='θ', bases=basis)
s = dist.Field(name='s', bases=basis)
u = dist.VectorField(coords, name='u', bases=basis)
τ_s1 = dist.Field(name='τ_s1', bases=b_S2)
τ_s2 = dist.Field(name='τ_s2', bases=b_S2)
τ_u1 = dist.VectorField(coords, name='τ_u1', bases=b_S2)
τ_u2 = dist.VectorField(coords, name='τ_u2', bases=b_S2)

# Parameters and operators
div = lambda A: de.Divergence(A, index=0)
lap = lambda A: de.Laplacian(A, coords)
grad = lambda A: de.Gradient(A, coords)
curl = lambda A: de.Curl(A)
cross = lambda A, B: de.CrossProduct(A, B)
trans = lambda A: de.TransposeComponents(A)
radial = lambda A: de.RadialComponent(A)
angular = lambda A: de.AngularComponent(A, index=1)
trace = lambda A: de.Trace(A)
power = lambda A, B: de.Power(A, B)
lift_basis = basis.clone_with(k=0)
lift = lambda A, n: de.Lift(A,lift_basis,n)
integ = lambda A: de.Integrate(A, coords)
azavg = lambda A: de.Average(A, coords.coords[0])
shellavg = lambda A: de.Average(A, coords.S2coordsys)
avg = lambda A: de.Integrate(A, coords)/(4/3*np.pi*radius**3)

# NCCs and variables of the problem
ez = dist.VectorField(coords, name='ez', bases=basis_ncc)
ez['g'][1] = -np.sin(theta)
ez['g'][2] =  np.cos(theta)
ez_g = de.Grid(ez).evaluate()
ez_g.name='ez_g'

logger.info("establishing polytrope with m = {:}, nρ = {:}, nh = {:}".format(m_ad, nρ, nh))

from structure import polytrope_shell
structure = polytrope_shell(Nr, radii, nh, m=m_poly, Legendre=Legendre, dtype=dtype, comm=MPI.COMM_SELF)

T = dist.Field(name='T', bases=basis_ncc)
lnρ = dist.Field(name='lnρ', bases=basis_ncc)

if T['g'].size > 0 :
    for i, r_i in enumerate(r[0,0,:]):
         T['g'][:,:,i] = structure['T'](r=r_i).evaluate()['g']
         lnρ['g'][:,:,i] = structure['lnρ'](r=r_i).evaluate()['g']

h0 = T.copy()
h0.name = 'h0'
θ0 = np.log(h0).evaluate()
Υ0 = lnρ.evaluate()
Υ0.name = 'Υ0'
ρ0 = np.exp(Υ0).evaluate()
ρ0.name = 'ρ0'
ρ0_inv = np.exp(-Υ0).evaluate()
ρ0_inv.name = '1/ρ0'
grad_h0 = grad(h0).evaluate()
grad_θ0 = grad(θ0).evaluate()
grad_Υ0 = grad(Υ0).evaluate()

s0 = (1/γ*θ0 - (γ-1)/γ*Υ0).evaluate()

r_g = dist.Field(bases=basis_ncc)
r_g['g'] = r
r_g.name='r'

grad_s0 = grad(s0).evaluate()

e = grad(u) + trans(grad(u))
viscous_terms = div(e) - 2/3*grad(div(u))

logger.info("NCC expansions:")
for ncc in [ρ0, ρ0*grad_h0, ρ0*h0, ρ0*h0*grad_s0, grad_θ0, ρ0*grad_s0, r_g*h0*grad_Υ0, r_g*h0]:
    logger.info("{}: {}".format(ncc.evaluate(), np.where(np.abs(ncc.evaluate()['c']) >= ncc_cutoff)[0].shape))
#Problem
omega = dist.Field(name='omega')
ddt = lambda A: omega*A
problem = de.EVP([u, Υ, θ, s, τ_u1, τ_u2, τ_s1, τ_s2], eigenvalue=omega)
problem.add_equation((ρ0*ddt(u) # assumes grad_s0 = 0
                      + Co2*ρ0*grad_h0*θ
                      + Co2*ρ0*h0*grad(θ)
                      - Co2*ρ0*h0*grad(s)
                      - Co2*ρ0*h0*grad_s0*θ
                      - Ek*viscous_terms
                      + ρ0*cross(ez, u)
                      + lift(τ_u1,-1) + lift(τ_u2,-2),
                      0 ))
problem.add_equation((r_g*h0*(ddt(Υ) + div(u) + u@grad_Υ0) + 1/Ek*lift(τ_u2,-1)@er, 0))
problem.add_equation((θ - (γ-1)*Υ - γ*s, 0)) #EOS, s_c/cP = 1
problem.add_equation((ρ0*(ddt(s) + u@grad_s0)
                      - Ek/Pr*(lap(θ)+2*grad_θ0@grad(θ))
                      + lift(τ_s1,-1) + lift(τ_s2,-2), 0 ))
# Boundary conditions
problem.add_equation((radial(u(r=Ri)), 0))
problem.add_equation((radial(angular(e(r=Ri))), 0))
problem.add_equation((s(r=Ri), 0))
problem.add_equation((radial(u(r=Ro)), 0))
problem.add_equation((radial(angular(e(r=Ro))), 0))
problem.add_equation((s(r=Ro), 0))
logger.info("Problem built")

# Solver
solver = problem.build_solver(ncc_cutoff=ncc_cutoff)
target_m = 10
target = 0 + 1j*0

subproblem = solver.subproblems_by_group[(target_m, None, None)]
solver.solve_sparse(subproblem, N=N_eigs, target=target, rebuild_matrices=True)
i_evals = np.argsort(solver.eigenvalues.real)
evals = solver.eigenvalues[i_evals]
logger.info('m = {:d}, ω_max = {:.3g}, {:.3g}i'.format(target_m, evals[-1].real, evals[-1].imag))
print(evals)


target_m=10
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.scatter(evals.real, evals.imag, alpha=0.5, zorder=5)
ax.axhline(y=0, linestyle='dashed', color='xkcd:dark grey', alpha=0.5, zorder=0)
ax.axvline(x=0, linestyle='dashed', color='xkcd:dark grey', alpha=0.5, zorder=0)
ax.set_title('m = {:d}, '.format(target_m)+r'$N_\theta = '+'{:d}'.format(Nθ)+r'$, $N_r = '+'{:d}'.format(Nr)+r'$')
ax.set_xlabel(r'$\omega_R$')
ax.set_ylabel(r'$\omega_I$')
ax.scatter(target.real, target.imag, marker='x', label='target',  color='xkcd:dark green', alpha=0.2, zorder=1)
ax.legend()
fig_filename = 'eigenspectrum'
if args['--Legendre']:
    fig_filename += '_Legendre'
fig.savefig(data_dir+'/'+fig_filename+'.png', dpi=300)
