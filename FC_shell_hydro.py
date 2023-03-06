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
    --mesh=<mesh>                        Processor mesh for 3-D runs; if not set a sensible guess will be made

    --benchmark                          Use benchmark initial conditions
    --spectrum                           Use a spectrum of benchmark perturbations
    --ell_benchmark=<ell_benchmark>      Integer value of benchmark perturbation m=+-ell [default: 3]

    --thermal_equilibrium                Start in thermal equilibrum

    --max_dt=<max_dt>                    Largest possible timestep [default: 0.25]
    --safety=<safety>                    CFL safety factor [default: 0.2]

    --run_time_sim=<run_time>            How long to run, in rotating time units
    --run_time_iter=<niter>              How long to run, in iterations

    --slice_dt=<slice_dt>                Cadence at which to output slices, in rotation times (P_rot = 4pi) [default: 10]
    --scalar_dt=<scalar_dt>              Time between scalar outputs, in rotation times (P_rot = 4pi) [default: 2]

    --restart=<restart>                  Merged chechpoint file to restart from.
                                         Make sure "--label" is set to avoid overwriting the previous run.

    --label=<label>                      Additional label for run output directory

    --ncc_cutoff=<ncc_cutoff>            Amplitude to truncate NCC terms [default: 1e-10]
    --plot_sparse                        Plot sparsity structures for L+M and it's LU decomposition
"""
import numpy as np
from dedalus.tools.parallel import Sync

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
dlog = logging.getLogger('matplotlib')
dlog.setLevel(logging.WARNING)
dlog = logging.getLogger('evaluator')
dlog.setLevel(logging.WARNING)

data_dir = sys.argv[0].split('.py')[0]
data_dir += '_Co{}_Ma{}_Ek{}_Pr{}'.format(args['--ConvectiveRossbySq'],args['--Mach'],args['--Ekman'],args['--Prandtl'])
data_dir += '_Th{}_R{}'.format(args['--Ntheta'], args['--Nr'])
if args['--thermal_equilibrium']:
    data_dir += '_therm'
if args['--benchmark']:
    data_dir += '_benchmark'
if args['--label']:
    data_dir += '_{:s}'.format(args['--label'])

import dedalus.tools.logging as dedalus_logging
dedalus_logging.add_file_handler(data_dir+'/logs/dedalus_log', 'DEBUG')

mesh = args['--mesh']
if mesh is not None:
    mesh = mesh.split(',')
    mesh = [int(mesh[0]), int(mesh[1])]
else:
    log2 = np.log2(ncpu)
    if log2 == int(log2):
        mesh = [int(2**np.ceil(log2/2)),int(2**np.floor(log2/2))]
logger.info("running on processor mesh={}".format(mesh))

Nθ = int(args['--Ntheta'])
Nr = int(args['--Nr'])
Nφ = Nθ*2

if args['--run_time_iter']:
    niter = int(float(args['--run_time_iter']))
else:
    niter = np.inf
if args['--run_time_sim']:
    run_time = float(args['--run_time_sim'])
else:
    run_time = np.inf

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
scrC = 1/(gamma-1)*Co2/Ma2
logger.info("scrC = {:}, Co2 = {:}, Ma2 = {:}".format(scrC, Co2, Ma2))

dealias = float(args['--dealias'])

m_poly = m_ad = 1/(γ-1)
Ro = r_outer = 1
Ri = r_inner = 0.7
nh = nρ/m_poly
c0 = -(Ri-Ro*np.exp(-nh))/(Ro-Ri)
c1 = Ri*Ro/(Ro-Ri)*(1-np.exp(-nh))

dtype = np.float64
coords = de.SphericalCoordinates('phi', 'theta', 'r')
dist = de.Distributor(coords, mesh=mesh, dtype=dtype)
if args['--Legendre']:
    basis = de.ShellBasis(coords, alpha=(0,0), shape=(Nφ, Nθ, Nr), radii=(Ri, Ro), dtype=dtype)
    basis_ncc = de.ShellBasis(coords, alpha=(0,0), shape=(1, 1, Nr), radii=(Ri, Ro), dtype=dtype)
else:
    basis = de.ShellBasis(coords, shape=(Nφ, Nθ, Nr), radii=(Ri, Ro), dtype=dtype)
    basis_ncc = de.ShellBasis(coords, shape=(1, 1, Nr), radii=(Ri, Ro), dtype=dtype)
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
A = dist.VectorField(coords, name="A", bases=basis)
φ = dist.Field(name="φ", bases=basis)
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
ddt = lambda A: de.TimeDerivative(A)
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
ex = dist.VectorField(coords, bases=basis, name='ex')
ex['g'][2] = np.sin(theta)*np.cos(phi)
ex['g'][1] = np.cos(theta)*np.cos(phi)
ex['g'][0] = -np.sin(phi)
ey = dist.VectorField(coords, bases=basis, name='ey')
ey['g'][2] = np.sin(theta)*np.sin(phi)
ey['g'][1] = np.cos(theta)*np.sin(phi)
ey['g'][0] = np.cos(phi)
ez = dist.VectorField(coords, name='ez', bases=basis)
ez['g'][1] = -np.sin(theta)
ez['g'][2] =  np.cos(theta)
ez_g = de.Grid(ez).evaluate()
ez_g.name='ez_g'

r_cyl = dist.VectorField(coords, name='r_cyl', bases=basis)
r_cyl['g'][2] =  r*np.sin(theta)
r_cyl['g'][1] = -r*np.cos(theta)

r_vec = dist.VectorField(coords, name='r_vec', bases=basis)
r_vec['g'][2] = r
r_vec_g = de.Grid(r_vec).evaluate()

r_S2 = dist.VectorField(coords, name='r_S2')
r_S2['g'][2] = 1

logger.info("establishing polytrope with m = {:}, nρ = {:}, nh = {:}".format(m_ad, nρ, nh))

T = dist.Field(name='T', bases=basis_ncc)

T = dist.Field(bases=basis_ncc, name='T')
T['g'] = c0 + c1/r
lnρ = m_poly*(np.log(T)).evaluate()
lnρ.name = 'lnρ'

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

h0_g = de.Grid(h0).evaluate()
h0_inv_g = de.Grid(1/h0).evaluate()
grad_h0_g = de.Grid(grad(h0)).evaluate()
ρ0_g = de.Grid(ρ0).evaluate()

ρ0_grad_h0_g = de.Grid(ρ0*grad(h0)).evaluate()
ρ0_h0_g = de.Grid(ρ0*h0).evaluate()

# Entropy source function
def source_function(r):
    return 1

source_func = dist.Field(name='S', bases=basis)
source_func['g'] = source_function(r)

# for RHS source function, need θ0 on the full ball grid (rather than just the radial grid)
θ0_RHS = dist.Field(name='θ0_RHS', bases=basis)
θ0.change_scales(1)
θ0_RHS.require_grid_space()
if θ0['g'].size > 0:
    θ0_RHS['g'] = θ0['g']
ε = Ma2
source = Ek/Pr*(ε*ρ0/h0*source_func)
source.name='source'

# terms related to the background being adiabatic rather than thermally equilibrated
thermal_terms = Ek/Pr*(lap(θ0_RHS) + grad(θ0_RHS)@grad(θ0_RHS))

# combine both together into total source term
#total_source = (source + thermal_terms).evaluate()
total_source = source.evaluate()
source_g = de.Grid(total_source).evaluate()

e = grad(u) + trans(grad(u))

ω = curl(u)
viscous_terms = div(e) - 2/3*grad(div(u))
trace_e = trace(e)
Phi = trace(e@e) - 1/3*(trace_e*trace_e)

# angular momentum conservation in shells
m, ell, n = dist.coeff_layout.local_group_arrays(basis.domain, scales=1)
mask = (ell==1)*(n==0)

τ_L = dist.VectorField(coords, bases=basis, name='τ_L')
τ_L.valid_modes[2] *= mask
τ_L.valid_modes[0] = False
τ_L.valid_modes[1] = False
L_cons_ncc = dist.Field(bases=basis_ncc, name='L_cons_ncc')
# suppress aliasing errors in the L_cons_ncc
padded = (1,1,4)
L_cons_ncc.change_scales(padded)
phi_pad, theta_pad, r_pad = dist.local_grids(basis, scales=padded)

R_avg = (Ro+Ri)/2
if args['--Legendre']:
    L_cons_ncc['g'] = (r_pad/R_avg)**3
else:
    L_cons_ncc['g'] = (r_pad/R_avg)**3*np.sqrt((r_pad/Ro-1)*(1-r_pad/Ri))
L_cons_ncc.change_scales(1)


logger.info("NCC expansions:")
for ncc in [ρ0, ρ0*grad(h0), ρ0*h0, ρ0*grad(θ0), h0*grad(Υ0), L_cons_ncc*ρ0]:
    logger.info("{}: {}".format(ncc.evaluate(), np.where(np.abs(ncc.evaluate()['c']) >= ncc_cutoff)[0].shape))

#Problem
problem = de.IVP([u, Υ, θ, s, τ_u1, τ_u2, τ_s1, τ_s2, τ_L])
problem.add_equation((ρ0*ddt(u) # assumes grad_s0 = 0
                      + Co2*ρ0*h0*grad(θ)
                      + Co2*ρ0*grad_h0*θ
                      - Co2*ρ0*h0*grad(s)
                      - Ek*viscous_terms
                      + lift(τ_u1,-1) + lift(τ_u2,-2) + τ_L,
                      -ρ0_g*((u@grad(u)) + cross(ez_g, u))
                      -Co2*ρ0_grad_h0_g*(np.expm1(θ)-θ)
                      -Co2*ρ0_h0_g*np.expm1(θ)*grad(θ)
                      +Co2*ρ0_h0_g*np.expm1(θ)*grad(s) ))
problem.add_equation((L_cons_ncc*ρ0*u, 0))
eq = problem.equations[-1]
eq['LHS'].valid_modes[2] *= mask
eq['LHS'].valid_modes[0] = False
eq['LHS'].valid_modes[1] = False
problem.add_equation((h0*ddt(Υ) + h0*div(u) + h0*u@grad_Υ0 + 1/Ek*lift(τ_u2,-1)@er,
                      -h0_g*(u@grad(Υ)) ))
problem.add_equation((θ - (γ-1)*Υ - γ*s, 0)) #EOS, s_c/cP = 1
#TO-DO:
# add ohmic heat
problem.add_equation((ρ0*(ddt(s))
                      - Ek/Pr*lap(θ) - Ek/Pr*2*grad_θ0@grad(θ)
                      + lift(τ_s1,-1) + lift(τ_s2,-2),
                      - ρ0_g*(u@grad(s))
                      + Ek/Pr*grad(θ)@grad(θ)
                      + Ek/Co2*0.5*h0_inv_g*Phi
                      + source_g ))
# Boundary conditions
problem.add_equation((radial(u(r=Ri)), 0))
problem.add_equation((radial(angular(e(r=Ri))), 0))
problem.add_equation((radial(grad(s)(r=Ri)), 0))
problem.add_equation((radial(u(r=Ro)), 0))
problem.add_equation((radial(angular(e(r=Ro))), 0))
problem.add_equation((s(r=Ro), 0))
logger.info("Problem built")

if args['--thermal_equilibrium']:
    logger.info("solving for thermal equilbrium")
    dist_eq = de.Distributor(coords, comm=None, dtype=dtype)
    s_r = dist_eq.Field(name='s(r)', bases=basis_ncc)
    θ_r = dist_eq.Field(name='θ(r)', bases=basis_ncc)
    grad_θ0_r = dist_eq.VectorField(coords, name='grad_θ0(r)', bases=basis_ncc)
    if grad_θ0_r['g'].size > 0 :
        logger.info(grad_θ0_r['g'].shape)
        logger.info(grad_θ0['g'].shape)
        for i, r_i in enumerate(r[0,0,:]):
            grad_θ0_r['g'][:,:,:,i] = grad_θ0(r=r_i).evaluate()['g'].real
    equilibrium = de.NLBVP([θ_r, s_r, τ_s1, τ_s2])
    equilibrium.add_equation((- Ek/Pr*(lap(θ_r)+2*grad_θ0_r@grad(θ_r))# - grad(θ_r)@grad(θ_r))
                              + lift(τ_s1,-1) + lift(τ_s2,-2), source))
    equilibrium.add_equation((θ_r - γ*s_r, 0)) #EOS, s_c/cP = 1
    equilibrium.add_equation((radial(grad(s_r)(r=Ri)), 0))
    equilibrium.add_equation((s_r(r=Ro), 0))
    eq_solver = equilibrium.build_solver(ncc_cutoff=ncc_cutoff)
    pert_norm = np.inf
    tolerance = 1e-8
    while pert_norm > tolerance:
        eq_solver.newton_iteration()
        pert_norm = sum(pert.allreduce_data_norm('c', 2) for pert in eq_solver.perturbations)
        logger.debug(f'Perturbation norm: {pert_norm:.3e}')
    logger.info('equilbrium acquired')
    if s_r['g'].size > 0:
        s['g'] += s_r['g']
        θ['g'] += θ_r['g']

# Solver
solver = problem.build_solver(de.RK222, ncc_cutoff=ncc_cutoff)

if args['--benchmark']:
    amp = 1e-1
    𝓁 = int(args['--ell_benchmark'])
    norm = 1/(2**𝓁*np.math.factorial(𝓁))*np.sqrt(np.math.factorial(2*𝓁+1)/(4*np.pi))
    s['g'] += amp*norm*r**𝓁*(1-r**2)*(np.cos(𝓁*phi)+np.sin(𝓁*phi))*np.sin(theta)**𝓁
    logger.info("benchmark run with perturbations at ell={} with norm={}".format(𝓁, norm))
elif args['--spectrum']:
    𝓁_min = 1
    for 𝓁 in np.arange(𝓁_min, int(args['--ell_benchmark'])+1):
        norm = 1/(2**𝓁*np.math.factorial(𝓁))*np.sqrt(np.math.factorial(2*𝓁+1)/(4*np.pi))
        s['g'] += amp*norm*r**𝓁*(1-r**2)*(np.cos(𝓁*phi)+np.sin(𝓁*phi))*np.sin(theta)**𝓁
    logger.info("bandwide run with perturbations at ell={}--{}".format(𝓁_min, 𝓁))
else:
    amp = 1e-5
    noise = dist.Field(name='noise', bases=basis)
    noise.fill_random('g', seed=42, distribution='standard_normal')
    noise.low_pass_filter(scales=0.25)
    s['g'] += amp*noise['g']

max_dt = float(args['--max_dt'])
dt = max_dt/10
if not args['--restart']:
    mode = 'overwrite'
else:
    write, dt = solver.load_state(args['--restart'])
    mode = 'append'

solver.stop_iteration = niter
solver.stop_sim_time = run_time

# Analysis
ur = u@er
uθ = u@eθ
uφ = u@eφ

ρ_cyl = dist.Field(bases=basis)
ρ_cyl['g'] = r*np.sin(theta)
Ωz = uφ/ρ_cyl # this is not ω_z; misses gradient terms; this is angular differential rotation.

u_fluc = u - azavg(ur)*er - azavg(uθ)*eθ - azavg(uφ)*eφ

ρ = ρ0*np.exp(Υ)
h = h0*np.exp(θ)
c_P = γ/(γ-1)
T = h/c_P
KE = 0.5*ρ*u@u
DRKE = 0.5*ρ*(azavg(uφ)**2)
MCKE = 0.5*ρ*(azavg(ur)**2 + azavg(uθ)**2)
FKE = KE - DRKE - MCKE #0.5*dot(u_fluc, u_fluc)

Ma2_ad = 1/(γ-1)*u@u/h

PE = Co2*ρ*T*s
PE.name = 'PE'

L = cross(r_vec,ρ*u)
L.name='L'

enstrophy = curl(u)@curl(u)
enstrophy_fluc = curl(u_fluc)@curl(u_fluc)

Re2 = u@u*(ρ/Ek)**2
Re2_fluc = u_fluc@u_fluc*(ρ/Ek)**2

scalar_dt = float(args['--scalar_dt'])
traces = solver.evaluator.add_file_handler(data_dir+'/traces', sim_dt=scalar_dt, max_writes=None)
traces.add_task(avg(KE), name='KE')
traces.add_task(avg(DRKE), name='DRKE')
traces.add_task(avg(MCKE), name='MCKE')
traces.add_task(avg(FKE), name='FKE')
traces.add_task(avg(Ma2_ad), name='Ma2')
traces.add_task(integ(KE)/Ek**2, name='E0')
traces.add_task(np.sqrt(avg(enstrophy)), name='Ro')
traces.add_task(np.sqrt(avg(Re2)), name='Re')
traces.add_task(np.sqrt(avg(enstrophy_fluc)), name='Ro_fluc')
traces.add_task(np.sqrt(avg(Re2_fluc)), name='Re_fluc')
traces.add_task(avg(PE), name='PE')
traces.add_task(integ(L@ex), name='Lx')
traces.add_task(integ(L@ey), name='Ly')
traces.add_task(integ(L@ez), name='Lz')
traces.add_task(shellavg(np.abs(τ_s1)), name='τ_s1')
traces.add_task(shellavg(np.abs(τ_s2)), name='τ_s2')
traces.add_task(shellavg(np.sqrt(τ_u1@τ_u1)), name='τ_u1')
traces.add_task(shellavg(np.sqrt(τ_u2@τ_u2)), name='τ_u2')
traces.add_task(shellavg(np.sqrt(τ_L@τ_L)), name='τ_L')

slice_dt = float(args['--slice_dt'])
slices = solver.evaluator.add_file_handler(data_dir+'/slices', sim_dt = slice_dt, max_writes = 10, mode=mode)
# for equatorial slices
slices.add_task(s(theta=np.pi/2), name='s')
slices.add_task(enstrophy(theta=np.pi/2), name='enstrophy')
# for mollweide
slices.add_task(s(r=0.95), name='s r0.95')
slices.add_task((er@u)(r=0.95), name='ur r0.95')
# averaged quantities
slices.add_task(azavg(Ωz), name='<Ωz>')
slices.add_task(azavg(s), name='<s>')
slices.add_task(shellavg(s), name='s(r)')
slices.add_task(shellavg(ρ*er@u*(p+0.5*u@u)), name='F_h(r)')
slices.add_task(shellavg(ρ*er@u*u@u), name='F_KE(r)')
slices.add_task(shellavg(-Co2*Ek/Pr*T*er@grad(s)), name='F_κ(r)')
slices.add_task(shellavg(Co2*source), name='F_source(r)')

report_cadence = 100
flow = flow_tools.GlobalFlowProperty(solver, cadence=report_cadence)
flow.add_property(Re2, name='Re2')
flow.add_property(enstrophy, name='Ro2')
flow.add_property(Re2_fluc, name='Re2_fluc')
flow.add_property(enstrophy_fluc, name='Ro2_fluc')
flow.add_property(KE, name='KE')
flow.add_property(Ma2_ad, name='Ma2')
flow.add_property(PE, name='PE')
flow.add_property(L@ez, name='Lz')
flow.add_property(np.abs(τ_s1), name='|τ_s1|')
flow.add_property(np.abs(τ_s2), name='|τ_s2|')
flow.add_property(np.sqrt(τ_u1@τ_u1), name='|τ_u1|')
flow.add_property(np.sqrt(τ_u2@τ_u2), name='|τ_u2|')
flow.add_property(np.sqrt(τ_L@τ_L), name='|τ_L|')

# CFL
cfl_safety_factor = float(args['--safety'])
CFL = flow_tools.CFL(solver, initial_dt=dt, cadence=1, safety=cfl_safety_factor, max_dt=max_dt, threshold=0.1)
CFL.add_velocity(u)

good_solution = True
vol = 4*np.pi/3
while solver.proceed and good_solution:
    dt = CFL.compute_timestep()
    if solver.iteration % report_cadence == 0 and solver.iteration > 0:
        KE_avg = flow.volume_integral('KE')/vol # volume average needs a defined volume
        E0 = flow.volume_integral('KE')/Ek**2 # integral rather than avg
        Re_avg = np.sqrt(flow.volume_integral('Re2')/vol)
        Ro_avg = np.sqrt(flow.volume_integral('Ro2')/vol)
        Re_fluc_avg = np.sqrt(flow.volume_integral('Re2_fluc')/vol)
        Ro_fluc_avg = np.sqrt(flow.volume_integral('Ro2_fluc')/vol)
        PE_avg = flow.volume_integral('PE')/vol
        Lz_avg = flow.volume_integral('Lz')/vol
        Ma_avg = np.sqrt(flow.volume_integral('Ma2')/vol)
        max_τ = np.max([flow.max('|τ_u1|'), flow.max('|τ_u2|'), flow.max('|τ_s1|'), flow.max('|τ_s2|')])
        τ_L = flow.max('|τ_L|')

        log_string = "iter: {:d}, dt={:.1e}, t={:.2e} ({:.2e})".format(solver.iteration, dt, solver.sim_time, solver.sim_time*Ek)
        log_string += ", Ma={:.2e}, KE={:.2e}, PE={:.2e}".format(Ma_avg, KE_avg, PE_avg)
        log_string += ", Re={:.1e}, Ro={:.1e}".format(Re_fluc_avg, Ro_fluc_avg)
        log_string += ", Lz={:.1e}, τ={:.1e}, τ_L={:.1e}".format(Lz_avg, max_τ, τ_L)
        logger.info(log_string)
        good_solution = np.isfinite(E0)
    solver.step(dt)

solver.log_stats()
logger.debug("mode-stages/DOF = {}".format(solver.total_modes/(Nφ*Nθ*Nr)))
