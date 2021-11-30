"""
Dedalus script for full sphere anelastic convection,
using a Lane-Emden structure and internal heat source.
Designed for modelling fully-convective stars.

Usage:
    mdwarf_hydro.py [options]

Options:
    --Ekman=<Ekman>                      Ekman number    [default: 5e-5]
    --ConvectiveRossbySq=<Co2>           Squared Convective Rossby = Ra*Ek**2/Pr [default: 7e-3]
    --Prandtl=<Prandtl>                  Prandtl number  [default: 1]
    --MagneticPrandtl=<Pm>               Magnetic Prandtl number [default: 1]
    --n_rho=<n_rho>                      Density scale heights [default: 3]

    --Ntheta=<Ntheta>                    Latitudinal modes [default: 32]
    --Nr=<Nr>                            Radial modes [default: 32]
    --dealias=<dealias>                  Degree of deailising   [default: 1.5]
    --mesh=<mesh>                        Processor mesh for 3-D runs; if not set a sensible guess will be made

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

    --ncc_cutoff=<ncc_cutoff>            Amplitude to truncate NCC terms [default: 1e-10]
    --plot_sparse                        Plot sparsity structures for L+M and it's LU decomposition
"""
import numpy as np
from dedalus.tools.parallel import Sync

import pathlib
import os
import sys
import h5py

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
data_dir += '_Ek{}_Co{}_Pr{}_Pm{}'.format(args['--Ekman'],args['--ConvectiveRossbySq'],args['--Prandtl'],args['--MagneticPrandtl'])
data_dir += '_Th{}_R{}'.format(args['--Ntheta'], args['--Nr'])
if args['--benchmark']:
    data_dir += '_benchmark'
if args['--label']:
    data_dir += '_{:s}'.format(args['--label'])
logger.info("saving data in {}".format(data_dir))

from dedalus.tools.config import config
config['logging']['filename'] = os.path.join(data_dir,'logs/dedalus_log')
config['logging']['file_level'] = 'DEBUG'
with Sync() as sync:
    if sync.comm.rank == 0:
        if not os.path.exists('{:s}/'.format(data_dir)):
            os.mkdir('{:s}/'.format(data_dir))
        logdir = os.path.join(data_dir,'logs')
        if not os.path.exists(logdir):
            os.mkdir(logdir)

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

n_rho = float(args['--n_rho'])
radius = 1

Ek = Ekman = float(args['--Ekman'])
Co2 = ConvectiveRossbySq = float(args['--ConvectiveRossbySq'])
Pr = Prandtl = float(args['--Prandtl'])
Pm = MagneticPrandtl = float(args['--MagneticPrandtl'])

logger.debug(sys.argv)
logger.debug('-'*40)
logger.info("Run parameters")
logger.info("Ek = {}, Co2 = {}, Pr = {}, Pm = {}".format(Ek,Co2,Pr, Pm))

import dedalus.public as de
from dedalus.extras import flow_tools

from structure import lane_emden

dealias = float(args['--dealias'])

c = de.SphericalCoordinates('phi', 'theta', 'r')
d = de.Distributor(c, mesh=mesh, dtype=np.float64)
b = de.BallBasis(c, shape=(Nφ,Nθ,Nr), radius=radius, dealias=dealias, dtype=np.float64)
b_S2 = b.S2_basis()
phi, theta, r = b.local_grids()

p = d.Field(name='p', bases=b)
s = d.Field(name='s', bases=b)
u = d.VectorField(c, name='u', bases=b)
A = d.VectorField(c, name="A", bases=b)
φ = d.Field(name="φ", bases=b)
τ_p = d.Field(name='τ_p')
τ_φ = d.Field(name="τ_φ")
τ_s = d.Field(name='τ_s', bases=b_S2)
τ_u = d.VectorField(c, name='τ_u', bases=b_S2)
τ_A = d.VectorField(c, name="τ_A", bases=b_S2)

# Parameters and operators
# Parameters and operators
div = lambda A: de.Divergence(A, index=0)
lap = lambda A: de.Laplacian(A, c)
grad = lambda A: de.Gradient(A, c)
curl = lambda A: de.Curl(A)
dot = lambda A, B: de.DotProduct(A, B)
cross = lambda A, B: de.CrossProduct(A, B)
ddt = lambda A: de.TimeDerivative(A)
trans = lambda A: de.TransposeComponents(A)
radial = lambda A: de.RadialComponent(A)
angular = lambda A: de.AngularComponent(A, index=1)
trace = lambda A: de.Trace(A)
power = lambda A, B: de.Power(A, B)
lift_basis = b.clone_with(k=2)
lift = lambda A, n: de.LiftTau(A,lift_basis,n)
integ = lambda A: de.Integrate(A, c)
azavg = lambda A: de.Average(A, c.coords[0])
shellavg = lambda A: de.Average(A, c.S2coordsys)
avg = lambda A: de.Integrate(A, c)/(4/3*np.pi*radius**3)

ell_func = lambda ell: ell+1
ellp1 = lambda A: de.SphericalEllProduct(A, c, ell_func)

# NCCs and variables of the problem
ez = d.VectorField(c, name='ez', bases=b)
ez['g'][1] = -np.sin(theta)
ez['g'][2] =  np.cos(theta)
ez_g = de.Grid(ez).evaluate()
ez_g.name='ez_g'

r_cyl = d.VectorField(c, name='r_cyl', bases=b)
r_cyl['g'][2] =  r*np.sin(theta)
r_cyl['g'][1] = -r*np.cos(theta)

r_vec = d.VectorField(c, name='r_vec', bases=b)
r_vec['g'][2] = r
r_vec_g = de.Grid(r_vec).evaluate()

r_S2 = d.VectorField(c, name='r_S2')
r_S2['g'][2] = 1

structure = lane_emden(Nr, n_rho=n_rho, m=1.5, comm=MPI.COMM_SELF)

bk2 = b.clone_with(k=2)
bk1 = b.clone_with(k=1)
T = d.Field(name='T', bases=b.radial_basis)
lnρ = d.Field(name='lnρ', bases=b.radial_basis)

if T['g'].size > 0 :
    # TO-DO: clean this up and make work for lane-emden solve in np.float64 rather than np.complex128
    for i, r_i in enumerate(r[0,0,:]):
         T['g'][:,:,i] = structure['T'](r=r_i).evaluate()['g'].real
         lnρ['g'][:,:,i] = structure['lnρ'](r=r_i).evaluate()['g'].real

lnT = np.log(T).evaluate()
lnT.name='lnT'
grad_lnT = grad(lnT).evaluate()
grad_lnT.name='grad_lnT'
grad_lnT1 = d.VectorField(c,name='grad_lnT1', bases=bk2.radial_basis)
grad_lnT.require_scales(1)
grad_lnT1['g'] = grad_lnT['g']
ρ = np.exp(lnρ).evaluate()
ρ.name='ρ'
ρ2 = d.Field(name='ρ2', bases=bk2.radial_basis)
ρ.require_scales(1)
ρ2['g'] = ρ['g']
grad_lnρ = grad(lnρ).evaluate()
grad_lnρ.name='grad_lnρ'
ρT = (ρ*T).evaluate()
ρT.name='ρT'
ρT2 = d.Field(name='ρT2', bases=bk2.radial_basis)
ρT.require_scales(1)
ρT2['g'] = ρT['g']

# Entropy source function, inspired from MESA model
def source_function(r):
    # from fits to MESA profile on r = [0,0.85]
    σ = 0.11510794072958948
    Q0_over_Q1 = 10.969517734412433
    # normalization from Brown et al 2020
    Q1 = σ**-2/(Q0_over_Q1 + 1) # normalize to σ**-2 at r=0
    logger.info("Source function: Q0/Q1 = {:}, σ = {:}, Q1 = {:}".format(Q0_over_Q1, σ, Q1))
    return (Q0_over_Q1*np.exp(-r**2/(2*σ**2)) + 1)*Q1

source_func = d.Field(name='S', bases=b)
source_func['g'] = source_function(r)
source = de.Grid(Ek/Pr*ρT*source_func).evaluate()
source.name='source'

B = curl(A)
J = -lap(A) #curl(B)

#e = 0.5*(grad(u) + trans(grad(u)))
e = grad(u) + trans(grad(u))
e.store_last = True

ω = curl(u)
#viscous_terms = div(e) + dot(grad_lnρ, e) - 2/3*grad(div(u)) - 2/3*grad_lnρ*div(u)
viscous_terms = div(e) - 2/3*grad(div(u))
trace_e = trace(e)
trace_e.store_last = True
Phi = trace(dot(e, e)) - 1/3*(trace_e*trace_e)

#Problem
problem = de.IVP([p, u, s, φ, A, τ_p, τ_u, τ_s, τ_φ, τ_A])
problem.add_equation((ρ2*ddt(u) + ρ2*grad(p) - Co2*ρT2*grad(s) - Ek*viscous_terms + lift(τ_u,-1),
                      - ρ*dot(u, e) - ρ*cross(ez_g, u) + cross(J,B)))
problem.add_equation((T*dot(grad_lnρ, u) + T*div(u) + τ_p, 0))
#TO-DO: consider: add ohmic heating?
problem.add_equation((ρT2*ddt(s) - Ek/Pr*T*(lap(s)+ dot(grad_lnT1, grad(s))) + lift(τ_s,-1),
                      - ρT*dot(u, grad(s)) + source + 1/2*Ek/Co2*Phi))
problem.add_equation((div(A) + τ_φ, 0)) # coulomb gauge
problem.add_equation((ρ2*ddt(A) + ρ2*grad(φ) - Ek/Pm*lap(A) + lift(τ_A,-1),
                        ρ2*cross(u, B) ))
# Boundary conditions
problem.add_equation((radial(u(r=radius)), 0))
problem.add_equation((radial(angular(e(r=radius))), 0))
problem.add_equation((integ(p), 0))
problem.add_equation((s(r=radius), 0))
problem.add_equation((integ(φ), 0))
problem.add_equation((dot(r_S2, grad(A)(r=radius))+ellp1(A)(r=radius)/radius, 0))
logger.info("Problem built")

logger.info("NCC expansions:")
for ncc in [ρ2, T, ρT2, (T*grad_lnρ).evaluate(), (T*grad_lnT1).evaluate()]:
    logger.info("{}: {}".format(ncc, np.where(np.abs(ncc['c']) >= ncc_cutoff)[0].shape))

if args['--thermal_equilibrium']:
    logger.info("solving for thermal equilbrium")
    equilibrium = de.LBVP([s, τ_s])
    equilibrium.add_equation((-(lap(s)+ dot(grad_lnT, grad(s))) + lift(τ_s,-1),
                              ρ*source))
    equilibrium.add_equation((s(r=radius), 0))
    eq_solver = equilibrium.build_solver(ncc_cutoff=ncc_cutoff)
    eq_solver.solve()

# Solver
solver = problem.build_solver(de.SBDF2, ncc_cutoff=ncc_cutoff)

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
    noise = d.Field(name='noise', bases=b)
    noise.fill_random('g', seed=42, distribution='standard_normal')
    noise.low_pass_filter(scales=0.25)
    s['g'] += amp*noise['g']

mag_amp = 1e-4
invert_B_to_A = False
if invert_B_to_A:
    B_IC = d.VectorField(c, name="B_IC", bases=b)
    B_IC['g'][2] = 0 # radial
    B_IC['g'][1] = -mag_amp*3./2.*r*(-1+4*r**2-6*r**4+3*r**6)*(np.cos(phi)+np.sin(phi))
    B_IC['g'][0] = -mag_amp*3./4.*r*(-1+r**2)*np.cos(theta)* \
                                 ( 3*r*(2-5*r**2+4*r**4)*np.sin(theta)
                                 +2*(1-3*r**2+3*r**4)*(np.cos(phi)-np.sin(phi)))
    logger.info("set initial conditions for B")
    IC_problem = de.LBVP([φ, A, τ_φ, τ_A])
    IC_problem.add_equation((div(A) + τ_φ, 0))
    IC_problem.add_equation((curl(A) + grad(φ) + lift(τ_A, -1), B_IC))
    IC_problem.add_equation((integ(φ), 0))
    IC_problem.add_equation((dot(r_S2, grad(A)(r=radius))+ellp1(A)(r=radius)/radius, 0))
    IC_solver = IC_problem.build_solver()
    IC_solver.solve()
    logger.info("solved for initial conditions for A")
else:
    # Marti convective dynamo benchmark values
    A_analytic_2 = (3/2*r**2*(1-4*r**2+6*r**4-3*r**6)
                       *np.sin(theta)*(np.sin(phi)-np.cos(phi))
                   +3/8*r**3*(2-7*r**2+9*r**4-4*r**6)
                       *(3*np.cos(theta)**2-1)
                   +9/160*r**2*(-200/21*r+980/27*r**3-540/11*r**5+880/39*r**7)
                         *(3*np.cos(theta)**2-1)
                   +9/80*r*(1-100/21*r**2+245/27*r**4-90/11*r**6+110/39*r**8)
                        *(3*np.cos(theta)**2-1)
                   +1/8*r*(-48/5*r+288/7*r**3-64*r**5+360/11*r**7)
                       *np.sin(theta)*(np.sin(phi)-np.cos(phi))
                   +1/8*(1-24/5*r**2+72/7*r**4-32/3*r**6+45/11*r**8)
                       *np.sin(theta)*(np.sin(phi)-np.cos(phi)))
    A_analytic_1 = (-27/80*r*(1-100/21*r**2+245/27*r**4-90/11*r**6+110/39*r**8)
                            *np.cos(theta)*np.sin(theta)
                    +1/8*(1-24/5*r**2+72/7*r**4-32/3*r**6+45/11*r**8)
                        *np.cos(theta)*(np.sin(phi)-np.cos(phi)))
    A_analytic_0 = (1/8*(1-24/5*r**2+72/7*r**4-32/3*r**6+45/11*r**8)
                       *(np.cos(phi)+np.sin(phi)))

    A['g'][0] = mag_amp*A_analytic_0
    A['g'][1] = mag_amp*A_analytic_1
    A['g'][2] = mag_amp*A_analytic_2


max_dt = float(args['--max_dt'])
dt = max_dt/10
if not args['--restart']:
    mode = 'overwrite'
else:
    write, dt = solver.load_state(args['--restart'])
    mode = 'append'

solver.stop_iteration = niter
solver.stop_sim_time = run_time

KE = 0.5*ρ*dot(u,u)
KE.store_last = True
ME = 0.5*dot(B,B)
ME.store_last = True
PE = Co2*ρ*T*s
Lz = dot(cross(r_vec,ρ*u), ez)
enstrophy = dot(curl(u),curl(u))
enstrophy.store_last = True

scalar_dt = float(args['--scalar_dt'])
traces = solver.evaluator.add_file_handler(data_dir+'/traces', sim_dt=scalar_dt, max_writes=np.inf)
traces.add_task(avg(KE), name='KE')
traces.add_task(avg(ME), name='ME')
traces.add_task(integ(KE)/Ek**2, name='E0')
traces.add_task(np.sqrt(avg(enstrophy)), name='Ro')
traces.add_task(np.sqrt(2/Ek*avg(KE)), name='Re')
traces.add_task(avg(PE), name='PE')
traces.add_task(avg(Lz), name='Lz')
traces.add_task(np.abs(τ_p), name='τ_p')
traces.add_task(np.abs(τ_φ), name='τ_φ')
traces.add_task(shellavg(np.abs(τ_s)), name='τ_s')
traces.add_task(shellavg(np.sqrt(dot(τ_u,τ_u))), name='τ_u')
traces.add_task(shellavg(np.sqrt(dot(τ_A,τ_A))), name='τ_A')

# Analysis
eφ = d.VectorField(c, bases=b)
eφ['g'][0] = 1
er = d.VectorField(c, bases=b)
er['g'][2] = 1
ρ_cyl = d.Field(bases=b)
ρ_cyl['g'] = r*np.sin(theta)
Ωz = dot(u, eφ)/ρ_cyl # this is not ω_z; misses gradient terms; this is angular differential rotation.
Bφ = dot(B, eφ)
Aφ = dot(A, eφ)

slice_dt = float(args['--slice_dt'])
slices = solver.evaluator.add_file_handler(data_dir+'/slices', sim_dt = slice_dt, max_writes = 10, virtual_file=True, mode=mode)
slices.add_task(s(theta=np.pi/2), name='s')
slices.add_task(enstrophy(theta=np.pi/2), name='enstrophy')
slices.add_task(azavg(Ωz), name='<Ωz>')
slices.add_task(azavg(Bφ), name='<Bφ>')
slices.add_task(azavg(Aφ), name='<Aφ>')
slices.add_task(azavg(s), name='<s>')
slices.add_task(shellavg(s), name='s(r)')
slices.add_task(dot(B,er)(r=radius), name='Br') # is this sufficient?  Should we be using radial(B) instead?

report_cadence = 100
flow = flow_tools.GlobalFlowProperty(solver, cadence=report_cadence)
flow.add_property(np.sqrt(KE*2)/Ek, name='Re')
flow.add_property(np.sqrt(enstrophy), name='Ro')
flow.add_property(KE, name='KE')
flow.add_property(ME, name='ME')
flow.add_property(PE, name='PE')
flow.add_property(Lz, name='Lz')
flow.add_property(np.abs(τ_s), name='|τ_s|')
flow.add_property(np.abs(τ_p), name='|τ_p|')
flow.add_property(np.sqrt(dot(τ_u,τ_u)), name='|τ_u|')
flow.add_property(np.sqrt(dot(τ_A,τ_A)), name='|τ_A|')

# CFL
cfl_safety_factor = float(args['--safety'])
CFL = flow_tools.CFL(solver, initial_dt=dt, cadence=1, safety=cfl_safety_factor, max_dt=max_dt, threshold=0.1)
CFL.add_velocity(u)
CFL.add_velocity(B)

good_solution = True
vol = 4*np.pi/3
while solver.proceed and good_solution:
    dt = CFL.compute_timestep()
    if solver.iteration % report_cadence == 0 and solver.iteration > 0:
        KE_avg = flow.volume_integral('KE')/vol # volume average needs a defined volume
        E0 = flow.volume_integral('KE')/Ek**2 # integral rather than avg
        Re_avg = flow.volume_integral('Re')/vol
        Ro_avg = flow.volume_integral('Ro')/vol
        PE_avg = flow.volume_integral('PE')/vol
        ME_avg = flow.volume_integral('ME')/vol
        Lz_avg = flow.volume_integral('Lz')/vol
        max_τ = np.max([flow.max('|τ_u|'), flow.max('|τ_s|'), flow.max('|τ_p|')])

        log_string = "iter: {:d}, dt={:.1e}, t={:.3e} ({:.2e})".format(solver.iteration, dt, solver.sim_time, solver.sim_time*Ek)
        log_string += ", KE={:.2e}, ME={:.2e}, PE={:.2e}".format(KE_avg, ME_avg, PE_avg)
        log_string += ", Re={:.1e}, Ro={:.1e}".format(Re_avg, Ro_avg)
        log_string += ", Lz={:.1e}, τ={:.1e}".format(Lz_avg, max_τ)
        logger.info(log_string)
        good_solution = np.isfinite(E0)
    solver.step(dt)

solver.log_stats()
logger.debug("mode-stages/DOF = {}".format(solver.total_modes/(Nφ*Nθ*Nr)))

if args['--plot_sparse']:
    # Plot matrices
    import matplotlib
    import matplotlib.pyplot as plt

    # Plot options
    fig = plt.figure(figsize=(9,3))
    cmap = matplotlib.cm.get_cmap("winter_r")
    clim = (-10, 0)
    lim_margin = 0.05

    def plot_sparse(A):
        I, J = A.shape
        A_mag = np.log10(np.abs(A.A))
        ax.pcolor(A_mag[::-1], cmap=cmap, vmin=clim[0], vmax=clim[1])
        ax.set_xlim(-lim_margin, I+lim_margin)
        ax.set_ylim(-lim_margin, J+lim_margin)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal', 'box')
        ax.text(0.95, 0.95, 'nnz: %i' %A.nnz, horizontalalignment='right', verticalalignment='top', transform=ax.transAxes)
        ax.text(0.95, 0.95, '\ncon: %.1e' %np.linalg.cond(A.A), horizontalalignment='right', verticalalignment='top', transform=ax.transAxes)

    for sp in solver.subproblems:
        m = sp.group[0]
        l = sp.group[1]
        print("sparsity structure for m={} ({})".format(m, sp.group))
        # Plot LHS
        ax = fig.add_subplot(1, 3, 1)
        LHS = (sp.M_min + sp.L_min) @ sp.pre_right
        plot_sparse(LHS)
        ax.set_title('LHS (m = %i)' %m)
        # Plot L
        ax = fig.add_subplot(1, 3, 2)
        L = sp.LHS_solver.LU.L
        plot_sparse(L)
        ax.set_title('L (m = %i)' %m)
        # Plot U
        ax = fig.add_subplot(1, 3, 3)
        U = sp.LHS_solver.LU.U
        plot_sparse(U)
        ax.set_title('U (m = %i)' %m)
        plt.tight_layout()
        plt.savefig(data_dir+"/m_{:d}_l_{:d}.pdf".format(m,l))
        fig.clear()
