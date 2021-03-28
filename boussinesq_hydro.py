"""
Dedalus script for full sphere anelastic convection,
applied to the Marti Boussinesq benchmark.

Usage:
    boussinesq_hydro.py [options]

Options:
    --Ekman=<Ekman>                      Ekman number    [default: 3e-4]
    --ConvectiveRossbySq=<Co2>           Squared Convective Rossby = Ra*Ek**2/Pr [default: 2.85e-2]
    --Prandtl=<Prandtl>                  Prandtl number  [default: 1]

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
from dedalus.tools.parallel import Sync

from mpi4py import MPI
import time

import pathlib
import os
import sys
import h5py

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
ncc_cutoff = float(args['--ncc_cutoff'])

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

data_dir = sys.argv[0].split('.py')[0]
data_dir += '_Ek{}_Co{}_Pr{}'.format(args['--Ekman'],args['--ConvectiveRossbySq'],args['--Prandtl'])
data_dir += '_L{}_N{}'.format(args['--L_max'], args['--N_max'])
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
curl = lambda A: de.operators.Curl(A)
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
ez.require_scales(1)
ez_g = de.operators.Grid(ez).evaluate()

T = de.field.Field(dist=d, bases=(b.radial_basis,), dtype=np.float64)
lnρ = de.field.Field(dist=d, bases=(b.radial_basis,), dtype=np.float64)
ρT_inv = de.field.Field(dist=d, bases=(b,), dtype=np.float64)

T['g'] = 0.5*(1-r1**2)
lnρ['g'] = 0

r_vec = de.field.Field(dist=d, bases=(b.radial_basis,), tensorsig=(c,), dtype=np.float64)
r_vec.set_scales(b.dealias)
r_vec['g'][2] = r
r_vec.require_scales(1)
r_vec_g = de.operators.Grid(r_vec).evaluate()

logger.info("shape and size of T['g'] {} & {}".format(T['g'].shape, T['g'].size))
logger.info("shape and size of ρT_inv['g'] {} & {}".format(ρT_inv['g'].shape, ρT_inv['g'].size))

lnT = d_log(T).evaluate()
T_inv = power(T,-1).evaluate()
grad_T = grad(T).evaluate()
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
source['g'] = 3

#e = 0.5*(grad(u) + trans(grad(u)))
e = grad(u) + trans(grad(u))
e.store_last = True

viscous_terms = div(e) + dot(grad_lnρ, e) - 2/3*grad(div(u)) - 2/3*grad_lnρ*div(u)
trace_e = trace(e)
trace_e.store_last = True
Phi = trace(dot(e, e)) - 1/3*(trace_e*trace_e)

ρ_inv['g'] = 1
problem = problems.IVP([p, u,  τ_u, s,  τ_s], ncc_cutoff=ncc_cutoff)
#problem.add_equation((ddt(u) + grad(p) - Co2*T*grad(s) - Ek*ρ_inv*viscous_terms + LiftTau(τ_u,-1),
# problem.add_equation((ddt(u) + grad(p) + Co2*grad_T*s - Ek*ρ_inv*viscous_terms + LiftTau(τ_u,-1),
problem.add_equation((div(u), 0), condition = "ntheta != 0")
problem.add_equation((p, 0), condition = "ntheta == 0")
problem.add_equation((ddt(u) + grad(p)  - Ek*lap(u) - Co2*r_vec*s + LiftTau(τ_u,-1),
                      - dot(u, e) - cross(ez_g, u)), condition = "ntheta != 0")
problem.add_equation((u, 0), condition = "ntheta == 0")
problem.add_equation((ddt(s) - Ek/Pr*(lap(s)) + LiftTau(τ_s,-1),
                     - dot(u, grad(s)) + Ek/Pr*source ))
# Boundary conditions
problem.add_equation((radial(u(r=radius)), 0), condition = "ntheta != 0")
problem.add_equation((radial(angular(e(r=radius))), 0), condition = "ntheta != 0")
problem.add_equation((τ_u, 0), condition = "ntheta == 0")
problem.add_equation((s(r=radius), 0))
logger.info("Problem built")


amp = 1e-5
if args['--benchmark']:
    amp = 1e-1
    𝓁 = int(args['--ell_benchmark'])
    norm = 1/(2**𝓁*np.math.factorial(𝓁))*np.sqrt(np.math.factorial(2*𝓁+1)/(4*np.pi))
    s.require_scales(L_dealias)
    s['g'] += amp*norm*r**𝓁*(1-r**2)*(np.cos(𝓁*phi)+np.sin(𝓁*phi))*np.sin(theta)**𝓁
    s['g'] += 0.5*(1-r**2)
    logger.info("benchmark run with perturbations at ell={} with norm={}".format(𝓁, norm))
else:
    # need a noise generator
    raise NotImplementedError("noise ICs not implemented")
    s['g'] += amp*noise

# Solver
solver = solvers.InitialValueSolver(problem, timesteppers.SBDF2)

reducer = GlobalArrayReducer(d.comm_cart)
weight_theta = b.local_colatitude_weights(3/2)
weight_r = b.local_radial_weights(3/2) #b.radial_basis.local_weights(3/2)*r**2
vol_test = np.sum(weight_r*weight_theta+0*s['g'])*np.pi/(Lmax+1)/L_dealias
vol_test = reducer.reduce_scalar(vol_test, MPI.SUM)
vol = 4*np.pi/3*(radius)
vol_correction = vol/vol_test

logger.info(vol)

def vol_avg(q):
    Q = np.sum(vol_correction*weight_r*weight_theta*q['g'])
    Q *= (np.pi)/(Lmax+1)/L_dealias
    Q /= (4/3*np.pi)
    return reducer.reduce_scalar(Q, MPI.SUM)

int_test = de.field.Field(dist=d, bases=(b,), dtype=np.float64)
int_test['g']=1
int_test.require_scales(L_dealias)
logger.info("vol_avg(1)={}".format(vol_avg(int_test)))
logger.info("vol_test={}".format(vol_test))
logger.info("vol_correction={}".format(vol_correction))

if rank == 0:
    scalar_file = pathlib.Path('{:s}/scalar_output.h5'.format(data_dir)).absolute()
    if os.path.exists(str(scalar_file)):
        scalar_file.unlink()
    scalar_f = h5py.File('{:s}'.format(str(scalar_file)), 'a')
    parameter_group = scalar_f.create_group('parameters')
    parameter_group['ConvectiveRossbySq'] = ConvectiveRossbySq
    parameter_group['Ekman'] = Ekman
    parameter_group['Prandtl'] = Prandtl
    parameter_group['L'] = Lmax
    parameter_group['N'] = Nmax
    parameter_group['dt_max'] = float(args['--max_dt'])

    scale_group = scalar_f.create_group('scales')
    scale_group.create_dataset(name='sim_time', shape=(0,), maxshape=(None,), dtype=np.float64)
    task_group = scalar_f.create_group('tasks')
    scalar_keys = ['KE', 'PE', 'Re', 'Ro', 'Lz', 'E0']
    for key in scalar_keys:
        task_group.create_dataset(name=key, shape=(0,), maxshape=(None,), dtype=np.float64)
    scalar_index = 0
    scalar_f.close()
    from collections import OrderedDict
    scalar_data = OrderedDict()


report_cadence = 100
energy_report_cadence = 100
dt = float(args['--max_dt'])
timestepper_history = [0,1]
hermitian_cadence = 100

main_start = time.time()
good_solution = True
while solver.ok and good_solution:
    if solver.iteration % energy_report_cadence == 0:
        #q = (ρ*power(u,2)).evaluate() # can't eval in parallel with ρ
        #q = (power(u,2)).evaluate()
        #E0 = vol_avg(q)

        #q = (0.5*dot(u,u)).evaluate()
        q = (0.5*dot(u,u)).evaluate()
        KE = vol_avg(q)
        E0 = KE/Ek**2

        q = (dot(curl(u),curl(u))).evaluate()
        Ro = np.sqrt(vol_avg(q))

        q = (dot(u,u)).evaluate()
        Re = np.sqrt(vol_avg(q))/Ek

        q = (T*s).evaluate()
        PE = Co2*vol_avg(q)

        #q = (dot(cross(r_vec_g,u), ez_g)).evaluate()
        q = (dot(cross(r_vec,u), ez)).evaluate()
        Lz = vol_avg(q)

        logger.info("iter: {:d}, dt={:.2e}, t={:.3e} ({:.3e}), KE={:.2e} ({:.4e}), PE={:.2e}, Lz={:.2e}".format(solver.iteration, dt, solver.sim_time, solver.sim_time*Ek, KE, E0, PE, Lz))
        good_solution = np.isfinite(E0)

        if rank == 0:
            scalar_data['PE'] = PE
            scalar_data['KE'] = KE
            scalar_data['Re'] = Re
            scalar_data['Ro'] = Ro
            scalar_data['Lz'] = Lz
            scalar_data['E0'] = E0

            scalar_f = h5py.File('{:s}'.format(str(scalar_file)), 'a')
            scalar_f['scales/sim_time'].resize(scalar_index+1, axis=0)
            scalar_f['scales/sim_time'][scalar_index] = solver.sim_time
            for key in scalar_data:
                scalar_f['tasks/'+key].resize(scalar_index+1, axis=0)
                scalar_f['tasks/'+key][scalar_index] = scalar_data[key]
            scalar_index += 1
            scalar_f.close()

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
