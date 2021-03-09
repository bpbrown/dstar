"""
Dedalus script for full sphere anelastic convection,
using a Lane-Emden structure and internal heat source.
Designed for modelling fully-convective stars.

Usage:
    mdwarf_hydro.py [options]

Options:
    --Ekman=<Ekman>                      Ekman number    [default: 3e-4]
    --ConvectiveRossbySq=<Co2>           Squared Convective Rossby = Ra*Ek**2/Pr [default: 0.3]
    --Prandtl=<Prandtl>                  Prandtl number  [default: 1]
    --n_rho=<n_rho>                      Density scale heights [default: 3]

    --L_max=<L_max>                      Max spherical harmonic [default: 30]
    --N_max=<N_max>                      Max radial polynomial  [default: 31]
    --mesh=<mesh>                        Processor mesh for 3-D runs; if not set a sensible guess will be made

    --benchmark                          Use benchmark initial conditions
    --ell_benchmark=<ell_benchmark>      Integer value of benchmark perturbation m=+-ell [default: 3]

    --max_dt=<max_dt>                    Largest possible timestep [default: 0.1]
    --safety=<safety>                    CFL safety factor [default: 0.4]

    --run_time_diffusion=<run_time_d>    How long to run, in diffusion times [default: 20]
    --run_time_rotation=<run_time_rot>   How long to run, in rotation timescale; overrides run_time_diffusion if set
    --run_time_iter=<run_time_i>         How long to run, in iterations

    --dt_output=<dt_output>              Time between outputs, in rotation times (P_rot = 4pi) [default: 2]
    --scalar_dt_output=<dt_scalar_out>   Time between scalar outputs, in rotation times (P_rot = 4pi) [default: 2]

    --restart=<restart>                  Merged chechpoint file to restart from.
                                         Make sure "--label" is set to avoid overwriting the previous run.

    --label=<label>                      Additional label for run output directory

    --ncc_cutoff=<ncc_cutoff>            Amplitude to truncate NCC terms [default: 1e-10]
    --debug                              Produce debugging output for NCCs
"""
import numpy as np
import dedalus.public as de
from dedalus.core import arithmetic, timesteppers, problems, solvers
from dedalus.extras.flow_tools import GlobalArrayReducer

from mpi4py import MPI
import time

from docopt import docopt
args = docopt(__doc__)

import logging
logger = logging.getLogger(__name__)
dlog = logging.getLogger('matplotlib')
dlog.setLevel(logging.WARNING)
from structure import lane_emden

comm = MPI.COMM_WORLD
rank = comm.rank
ncpu = comm.size

mesh = args['--mesh']
if mesh is not None:
    mesh = mesh.split(',')
    mesh = [int(mesh[0]), int(mesh[1])]
else:
    log2 = np.log2(ncpu)
    if log2 == int(log2):
        mesh = [int(2**np.ceil(log2/2)),int(2**np.floor(log2/2))]
logger.info("running on processor mesh={}".format(mesh))

Lmax = int(args['--L_max'])
Nmax = int(args['--N_max'])
#niter = int(float(args['--niter']))
ncc_cutoff = float(args['--ncc_cutoff'])

n_rho = float(args['--n_rho'])
radius = 1

Ek = Ekman = float(args['--Ekman'])
Co2 = ConvectiveRossbySq = float(args['--ConvectiveRossbySq'])
Pr = Prandtl = float(args['--Prandtl'])
logger.info("Ek = {}, Co2 = {}, Pr = {}".format(Ek,Co2,Pr))
# load balancing for real variables and parallel runs
if Lmax % 2 == 1:
    nm = 2*(Lmax+1)
else:
    nm = 2*(Lmax+2)

L_dealias = 3/2
N_dealias = 3/2

start_time = time.time()
c = de.coords.SphericalCoordinates('phi', 'theta', 'r')
d = de.distributor.Distributor((c,), mesh=mesh)
b = de.basis.BallBasis(c, (nm,Lmax+1,Nmax+1), radius=radius, dealias=(L_dealias,L_dealias,N_dealias), dtype=np.float64)
b_S2 = b.S2_basis()
phi1, theta1, r1 = b.local_grids((1,1,1))
phi, theta, r = b.local_grids((L_dealias,L_dealias,N_dealias))
phig,thetag,rg= b.global_grids((L_dealias,L_dealias,N_dealias))
theta_target = thetag[0,(Lmax+1)//2,0]

u = de.field.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=np.float64)
p = de.field.Field(dist=d, bases=(b,), dtype=np.float64)
s = de.field.Field(dist=d, bases=(b,), dtype=np.float64)
τ_u = de.field.Field(dist=d, bases=(b_S2,), tensorsig=(c,), dtype=np.float64)
τ_s = de.field.Field(dist=d, bases=(b_S2,), dtype=np.float64)

# Parameters and operators
div = lambda A: de.operators.Divergence(A, index=0)
lap = lambda A: de.operators.Laplacian(A, c)
grad = lambda A: de.operators.Gradient(A, c)
dot = lambda A, B: arithmetic.DotProduct(A, B)
cross = lambda A, B: arithmetic.CrossProduct(A, B)
ddt = lambda A: de.operators.TimeDerivative(A)
trans = lambda A: de.operators.TransposeComponents(A)
radial = lambda A: de.operators.RadialComponent(A)
angular = lambda A: de.operators.AngularComponent(A, index=1)
trace = lambda A: de.operators.Trace(A)
power = lambda A, B: de.operators.Power(A, B)
LiftTau = lambda A, n: de.operators.LiftTau(A,b,n)
d_exp = lambda A: de.operators.UnaryGridFunction(np.exp, A)
d_log = lambda A: de.operators.UnaryGridFunction(np.log, A)

# NCCs and variables of the problem
ez = de.field.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=np.float64)
ez.set_scales(b.dealias)
ez['g'][1] = -np.sin(theta)
ez['g'][2] =  np.cos(theta)
ez_g = de.operators.Grid(ez).evaluate()

structure = lane_emden(Nmax, n_rho=n_rho, m=1.5, comm=MPI.COMM_SELF)

T = de.field.Field(dist=d, bases=(b.radial_basis,), dtype=np.float64)
lnρ = de.field.Field(dist=d, bases=(b.radial_basis,), dtype=np.float64)
ρT_inv = de.field.Field(dist=d, bases=(b,), dtype=np.float64)

#T['g'] = structure['T']['g']
#lnρ['g'] = structure['lnρ']['g']
logger.info("shape and size of T['g'] {} & {}".format(T['g'].shape, T['g'].size))
logger.info("shape and size of ρT_inv['g'] {} & {}".format(ρT_inv['g'].shape, ρT_inv['g'].size))
if T['g'].size > 0 :
    for i, r_i in enumerate(r1[0,0,:]):
         T['g'][:,:,i] = structure['T'](r=r_i).evaluate()['g']
         lnρ['g'][:,:,i] = structure['lnρ'](r=r_i).evaluate()['g']

lnT = d_log(T).evaluate()
T_inv = power(T,-1).evaluate()
grad_lnT = grad(lnT).evaluate()
ρ = d_exp(lnρ).evaluate()
grad_lnρ = grad(lnρ).evaluate()
ρ_inv = d_exp(-lnρ).evaluate()
ρT_inv_rb = (T_inv*ρ_inv).evaluate()
ρT_inv_rb.require_scales(1)
if ρT_inv_rb['g'].size > 0:
    ρT_inv['g'] = ρT_inv_rb['g']

# Entropy source function, inspired from MESA model
source = de.field.Field(dist=d, bases=(b,), dtype=np.float64)
source.require_scales(L_dealias)
# from fits to MESA profile on r = [0,0.85]
σ = 0.11510794072958948
Q0_over_Q1 = 10.969517734412433
# normalization from Brown et al 2020
Q1 = σ**-2/(Q0_over_Q1 + 1) # normalize to σ**-2 at r=0
source['g'] = (Q0_over_Q1*np.exp(-r**2/(2*σ**2)) + 1)*Q1
logger.info("Source function: Q0/Q1 = {:}, σ = {:}, Q1 = {:}".format(Q0_over_Q1, σ, Q1))

#e = 0.5*(grad(u) + trans(grad(u)))
e = grad(u) + trans(grad(u))
e.store_last = True

viscous_terms = div(e) + dot(grad_lnρ, e) - 2/3*grad(div(u)) - 2/3*grad_lnρ*div(u)
trace_e = trace(e)
trace_e.store_last = True
Phi = trace(dot(e, e)) - 1/3*(trace_e*trace_e)
# Problem
if args['--debug']:
    import matplotlib.pyplot as plt
    if T['g'].size > 0:
        fig, ax = plt.subplots(nrows=3, ncols=2)
        ax[0,0].plot(r[0,0,:], T['g'][0,0,:])
        ax[0,1].plot(r[0,0,:], lnρ['g'][0,0,:])
        ax[1,0].plot(r[0,0,:], grad_lnT['g'][2][0,0,:])
        ax[1,1].plot(r[0,0,:], grad_lnρ['g'][2][0,0,:])
        ax[2,0].plot(r[0,0,:], T_inv['g'][0,0,:])
        ax[2,1].plot(r[0,0,:], ρ_inv['g'][0,0,:])
        ax[0,0].set_ylabel('T')
        ax[0,1].set_ylabel(r'$\ln \rho$')
        ax[1,0].set_ylabel('gradT')
        ax[1,1].set_ylabel('gradlnrho')
        ax[2,0].set_ylabel('1/T')
        ax[2,1].set_ylabel('1/ρ')
        plt.tight_layout()
        fig.savefig('nccs_p{}.pdf'.format(rank))
        print(grad_lnρ['g'][2])
        print("rho inv: {}".format(ρ_inv['g']))

    if T['c'].size > 0:
        fig, ax = plt.subplots(nrows=3, ncols=2)
        ax[0,0].plot(np.abs(T['c'][0,0,:]))
        ax[0,1].plot(np.abs(lnρ['c'][0,0,:]))
        ax[1,0].plot(np.abs(grad_lnT['c'][1][0,0,:])) # index 1 is spin 0
        ax[1,1].plot(np.abs(grad_lnρ['c'][1][0,0,:])) # index 1 is spin 0
        ax[2,0].plot(np.abs(T_inv['c'][0,0,:]))
        ax[2,1].plot(np.abs(ρ_inv['c'][0,0,:]))
        ax[0,0].set_ylabel('T')
        ax[0,1].set_ylabel(r'$\ln \rho$')
        ax[1,0].set_ylabel('gradT')
        ax[1,1].set_ylabel('gradlnrho')
        ax[2,0].set_ylabel('1/T')
        ax[2,1].set_ylabel('1/ρ')
        for axi in ax:
            for axii in axi:
                axii.axhline(y=ncc_cutoff, linestyle='dashed', color='xkcd:grey')
                axii.set_yscale('log')
        plt.tight_layout()
        fig.savefig('nccs_coeff_p{}.pdf'.format(rank))

problem = problems.IVP([u, p, s, τ_u, τ_s], ncc_cutoff=ncc_cutoff)
problem.add_equation((ddt(u) + grad(p) - Co2*T*grad(s) - Ek*ρ_inv*viscous_terms + LiftTau(τ_u,-1),
                      - dot(u, e) - cross(ez_g, u)), condition = "ntheta != 0")
problem.add_equation((dot(grad_lnρ, u) + div(u), 0), condition = "ntheta != 0")
problem.add_equation((u, 0), condition = "ntheta == 0")
problem.add_equation((p, 0), condition = "ntheta == 0")
problem.add_equation((ddt(s) - Ek/Pr*ρ_inv*(lap(s)+ dot(grad_lnT, grad(s))) + LiftTau(τ_s,-1),
                     - dot(u, grad(s)) + Ek/Pr*source + 1/2*Ek/Co2*ρT_inv*Phi))
# Boundary conditions
problem.add_equation((radial(u(r=radius)), 0), condition = "ntheta != 0")
problem.add_equation((radial(angular(e(r=radius))), 0), condition = "ntheta != 0")
problem.add_equation((τ_u, 0), condition = "ntheta == 0")
problem.add_equation((s(r=radius), 0))
logger.info("Problem built")

amp = 1e-2
if args['--benchmark']:
    𝓁 = int(args['--ell_benchmark'])
    norm = 1/(2**𝓁*np.math.factorial(𝓁))*np.sqrt(np.math.factorial(2*𝓁+1)/(4*np.pi))
    s.require_scales(L_dealias)
    s['g'] += amp*norm*r**𝓁*(1-r**2)*(np.cos(𝓁*phi)+np.sin(𝓁*phi))*np.sin(theta)**𝓁
    logger.info("benchmark run with perturbations at ell={} with norm={}".format(𝓁, norm))
else:
    # need a noise generator
    raise NotImplementedError("noise ICs not implemented")
    s['g'] += amp*noise

# Solver
solver = solvers.InitialValueSolver(problem, timesteppers.SBDF2)

reducer = GlobalArrayReducer(d.comm_cart)
weight_theta = b.local_colatitude_weights(3/2)
weight_r = b.radial_basis.local_weights(3/2)*r**2
vol_test = np.sum(weight_r*weight_theta+0*s['g'])*np.pi/(Lmax+1)/L_dealias
vol_test = reducer.reduce_scalar(vol_test, MPI.SUM)
vol = 4*np.pi/3*(radius)
vol_correction = vol/vol_test

logger.info(vol)

report_cadence = 10
energy_report_cadence = 10
dt = float(args['--max_dt'])
timestepper_history = [0,1]
hermitian_cadence = 100

main_start = time.time()
good_solution = True
while solver.ok and good_solution:
    if solver.iteration % energy_report_cadence == 0:
        q = (ρ*power(u,2)).evaluate()
        E0 = np.sum(vol_correction*weight_r*weight_theta*0.5*q['g'])
        E0 *= (np.pi)/(Lmax+1)/L_dealias
        E0 = reducer.reduce_scalar(E0, MPI.SUM)
        T0 = np.sum(vol_correction*weight_r*weight_theta*0.5*s['g']**2)
        T0 *= (np.pi)/(Lmax+1)/L_dealias
        T0 = reducer.reduce_scalar(T0, MPI.SUM)
        logger.info("iter: {:d}, dt={:e}, t={:e}, E0={:e}, T0={:e}".format(solver.iteration, dt, solver.sim_time, E0, T0))
        good_solution = np.isfinite(E0)
    elif solver.iteration % report_cadence == 0:
        logger.info("iter: {:d}, dt={:e}, t={:e}".format(solver.iteration, dt, solver.sim_time))
    if solver.iteration % hermitian_cadence in timestepper_history:
        for field in solver.state:
            field['g']
    solver.step(dt)

end_time = time.time()

startup_time = main_start - start_time
main_loop_time = end_time - main_start
DOF = nm*(Lmax+1)*(Nmax+1)
niter = solver.iteration
if rank==0:
    print('performance metrics:')
    print('    startup time   : {:}'.format(startup_time))
    print('    main loop time : {:}'.format(main_loop_time))
    print('    main loop iter : {:d}'.format(niter))
    print('    wall time/iter : {:f}'.format(main_loop_time/niter))
    print('          iter/sec : {:f}'.format(niter/main_loop_time))
    print('DOF-cycles/cpu-sec : {:}'.format(DOF*niter/(ncpu*main_loop_time)))
