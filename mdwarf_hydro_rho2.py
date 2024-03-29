"""
Dedalus script for full sphere anelastic convection,
using a Lane-Emden structure and internal heat source.
Designed for modelling fully-convective stars.

Usage:
    mdwarf_hydro.py [options]

Options:
    --Ekman=<Ekman>                      Ekman number    [default: 5e-5]
    --ConvectiveRossbySq=<Co2>           Squared Convective Rossby = Ra*Ek**2/Pr [default: 7e-3]
    --Prandtl=<Prandtl>                  Prandtl number  [default: 0.5]
    --n_rho=<n_rho>                      Density scale heights [default: 3]

    --L_max=<L_max>                      Max spherical harmonic [default: 30]
    --N_max=<N_max>                      Max radial polynomial  [default: 31]
    --dealias=<dealias>                  Degree of deailising   [default: 1.5]
    --mesh=<mesh>                        Processor mesh for 3-D runs; if not set a sensible guess will be made

    --benchmark                          Use benchmark initial conditions
    --spectrum                           Use a spectrum of benchmark perturbations
    --ell_benchmark=<ell_benchmark>      Integer value of benchmark perturbation m=+-ell [default: 3]

    --thermal_equilibrium                Start in thermal equilibrum

    --max_dt=<max_dt>                    Largest possible timestep [default: 0.25]
    --safety=<safety>                    CFL safety factor [default: 0.4]

    --run_time=<run_time>                How long to run, in rotating time units
    --niter=<niter>                      How long to run, in iterations

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

import pathlib
import os
import sys
import h5py

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
if args['--niter']:
    niter = int(float(args['--niter']))
else:
    niter = np.inf
if args['--run_time']:
    run_time = float(args['--run_time'])
else:
    run_time = np.inf

ncc_cutoff = float(args['--ncc_cutoff'])

n_rho = float(args['--n_rho'])
radius = 1

Ek = Ekman = float(args['--Ekman'])
Co2 = ConvectiveRossbySq = float(args['--ConvectiveRossbySq'])
Pr = Prandtl = float(args['--Prandtl'])
logger.info("Ek = {}, Co2 = {}, Pr = {}".format(Ek,Co2,Pr))

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

# load balancing for real variables and parallel runs
if Lmax % 2 == 1:
    nm = 2*(Lmax+1)
else:
    nm = 2*(Lmax+2)

L_dealias = float(args['--dealias'])
N_dealias = float(args['--dealias'])

start_time = time.time()
c = de.coords.SphericalCoordinates('phi', 'theta', 'r')
d = de.distributor.Distributor((c,), mesh=mesh)
b = de.basis.BallBasis(c, (nm,Lmax+1,Nmax+1), radius=radius, dealias=(L_dealias,L_dealias,N_dealias), dtype=np.float64)
b_S2 = b.S2_basis()
phi1, theta1, r1 = b.local_grids((1,1,1))
phi, theta, r = b.local_grids((L_dealias,L_dealias,N_dealias))
phig,thetag,rg= b.global_grids((L_dealias,L_dealias,N_dealias))
theta_target = thetag[0,(Lmax+1)//2,0]

u = de.field.Field(name='u', dist=d, bases=(b,), tensorsig=(c,), dtype=np.float64)
p = de.field.Field(name='p', dist=d, bases=(b,), dtype=np.float64)
s = de.field.Field(name='s', dist=d, bases=(b,), dtype=np.float64)
τ_u = de.field.Field(name='τ_u', dist=d, bases=(b_S2,), tensorsig=(c,), dtype=np.float64)
τ_s = de.field.Field(name='τ_s', dist=d, bases=(b_S2,), dtype=np.float64)

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
ez_g = de.operators.Grid(ez).evaluate()

r_cyl = de.field.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=np.float64)
r_cyl.set_scales(b.dealias)
r_cyl['g'][2] =  r*np.sin(theta)
r_cyl['g'][1] = -r*np.cos(theta)

r_vec = de.field.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=np.float64)
r_vec.set_scales(b.dealias)
r_vec['g'][2] = r
r_vec_g = de.operators.Grid(r_vec).evaluate()

structure = lane_emden(Nmax, n_rho=n_rho, m=1.5, comm=MPI.COMM_SELF)

T = de.field.Field(dist=d, bases=(b.radial_basis,), dtype=np.float64)
lnρ = de.field.Field(dist=d, bases=(b.radial_basis,), dtype=np.float64)
ρT_inv = de.field.Field(dist=d, bases=(b,), dtype=np.float64)

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
ρT_inv = (T_inv*ρ_inv).evaluate()

# Entropy source function, inspired from MESA model
def source_function(r):
    # from fits to MESA profile on r = [0,0.85]
    σ = 0.11510794072958948
    Q0_over_Q1 = 10.969517734412433
    # normalization from Brown et al 2020
    Q1 = σ**-2/(Q0_over_Q1 + 1) # normalize to σ**-2 at r=0
    logger.info("Source function: Q0/Q1 = {:}, σ = {:}, Q1 = {:}".format(Q0_over_Q1, σ, Q1))
    return (Q0_over_Q1*np.exp(-r**2/(2*σ**2)) + 1)*Q1

source = de.field.Field(dist=d, bases=(b,), dtype=np.float64)
source['g'] = source_function(r1)


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
problem = problems.IVP([u, p, s, τ_u, τ_s])
problem.add_equation((ρ*ddt(u) + ρ*grad(p) - Co2*ρ*T*grad(s) - Ek*viscous_terms + LiftTau(τ_u,-1),
                      - ρ*dot(u, e) - ρ*cross(ez_g, u)))
#                       - ρ*cross(ω+ez,u)), condition = "ntheta != 0")
#                      - div(ρ*u*u) - ρ*cross(ez_g, u)), condition = "ntheta != 0")
problem.add_equation((dot(grad_lnρ, u) + div(u), 0))
problem.add_equation((ρ*ddt(s) - Ek/Pr*(lap(s)+ dot(grad_lnT, grad(s))) + LiftTau(τ_s,-1),
                      - ρ*dot(u, grad(s)) + Ek/Pr*ρ*source + 1/2*Ek/Co2*T_inv*Phi))
# Boundary conditions
problem.add_equation((radial(u(r=radius)), 0), condition = "ntheta != 0")
problem.add_equation((radial(angular(e(r=radius))), 0), condition = "ntheta != 0")
problem.add_equation((p(r=radius), 0), condition = "ntheta == 0")
problem.add_equation((s(r=radius), 0))
logger.info("Problem built")

logger.info("NCC expansions:")
for ncc in [ρ, grad_lnρ, T, (T*ρ).evaluate()]:
    logger.info("{}: {}".format(ncc, np.where(np.abs(ncc['c']) >= ncc_cutoff)[0].shape))

if args['--thermal_equilibrium']:
    logger.info("solving for thermal equilbrium")
    equilibrium = problems.LBVP([s, τ_s])
    equilibrium.add_equation((-(lap(s)+ dot(grad_lnT, grad(s))) + LiftTau(τ_s,-1),
                              ρ*source))
    equilibrium.add_equation((s(r=radius), 0))
    eq_solver = solvers.LinearBoundaryValueSolver(equilibrium, ncc_cutoff=ncc_cutoff)
    eq_solver.solve()

# Solver
solver = solvers.InitialValueSolver(problem, timesteppers.SBDF2, ncc_cutoff=ncc_cutoff)
timestepper_history = [0,1]

def load_state(solver, path, index=-1):
    """
    Load state from HDF5 file.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to Dedalus HDF5 savefile
    index : int, optional
        Local write index (within file) to load (default: -1)

    Returns
    -------
    write : int
        Global write number of loaded write
    dt : float
        Timestep at loaded write
    """
    path = pathlib.Path(path)
    logger.info("Loading solver state from: {}".format(path))
    with h5py.File(str(path), mode='r') as file:
        # Load solver attributes
        write = file['scales']['write_number'][index]
        try:
            dt = file['scales']['timestep'][index]
        except KeyError:
            dt = None
        solver.iteration = solver.initial_iteration = file['scales']['iteration'][index]
        solver.sim_time = solver.initial_sim_time = file['scales']['sim_time'][index]
        # Log restart info
        logger.info("Loading iteration: {}".format(solver.iteration))
        logger.info("Loading write: {}".format(write))
        logger.info("Loading sim time: {}".format(solver.sim_time))
        logger.info("Loading timestep: {}".format(dt))
        # Load fields
        for field in solver.state:
            # assumes named fields in hdf5 file
            dset = file['tasks'][field.name]
            # Find matching layout
            for layout in solver.dist.layouts:
                if np.allclose(layout.grid_space, dset.attrs['grid_space']):
                    break
            else:
                raise ValueError("No matching layout")
            # Set scales to match saved data
            scales = np.array(dset.shape[2:]) / np.array(layout.global_shape(field.domain,1))
            scales[~layout.grid_space] = 1
            # Extract local data from global dset
            # this line is missing the tensorsig info
            #dset_slices = (index,) + layout.slices(field.domain, tuple(scales))[0]

            # old slicing
            start = self.start(scales)
            local_shape = self.local_shape(scales)
            return tuple(slice(s, s+l) for (s, l) in zip(start, local_shape))

            # re-produces field internal variable, that's not currently stored.  Store instead.
            local_slices = tuple(slice(None) for cs in field.tensorsig) + tuple(field.layout.slices(field.domain, scales))
            dset_slices = (index,) + local_slices
            logger.info(dset)
            #logger.info(dset_slices)
            #logger.info(dset[-1,:,layout.slices(field.domain, tuple(scales))[0]])
            logger.info(dset_slices)
            logger.info(dset[dset_slices])
            local_dset = dset[dset_slices]
            # Copy to field
            field_slices = tuple(slice(n) for n in local_dset.shape)
            field.require_scales(scales)
            field[layout][field_slices] = local_dset
            field.require_scales(solver.domain.dealias)
    return write, dt


amp = 1e-2
if args['--restart']:
    write, dt = load_state(solver, args['--restart'])
elif args['--benchmark']:
    s.require_scales(L_dealias)
    𝓁 = int(args['--ell_benchmark'])
    norm = 1/(2**𝓁*np.math.factorial(𝓁))*np.sqrt(np.math.factorial(2*𝓁+1)/(4*np.pi))
    s['g'] += amp*norm*r**𝓁*(1-r**2)*(np.cos(𝓁*phi)+np.sin(𝓁*phi))*np.sin(theta)**𝓁
    logger.info("benchmark run with perturbations at ell={} with norm={}".format(𝓁, norm))
elif args['--spectrum']:
    s.require_scales(L_dealias)
    𝓁_min = 1
    for 𝓁 in np.arange(𝓁_min, int(args['--ell_benchmark'])+1):
        norm = 1/(2**𝓁*np.math.factorial(𝓁))*np.sqrt(np.math.factorial(2*𝓁+1)/(4*np.pi))
        s['g'] += amp*norm*r**𝓁*(1-r**2)*(np.cos(𝓁*phi)+np.sin(𝓁*phi))*np.sin(theta)**𝓁
    logger.info("bandwide run with perturbations at ell={}--{}".format(𝓁_min, 𝓁))
else:
    # need a noise generator
    raise NotImplementedError("noise ICs not implemented")
    s['g'] += amp*noise


for field in solver.state:
    logger.info(field)

reducer = GlobalArrayReducer(d.comm_cart)
weight_theta = b.local_colatitude_weights(L_dealias)
weight_r = b.local_radial_weights(N_dealias)
vol_test = np.sum(weight_r*weight_theta+0*s['g'])*np.pi/(Lmax+1)/L_dealias
vol_test = reducer.reduce_scalar(vol_test, MPI.SUM)
vol = 4*np.pi/3*(radius)
vol_correction = vol/vol_test

logger.info(vol)

energy_report_cadence  = report_cadence = 100
dt = float(args['--max_dt'])

hermitian_cadence = 100

solver.stop_iteration = niter
solver.stop_sim_time = run_time

if rank == 0:
    scalar_file = pathlib.Path('{:s}/scalar_output.h5'.format(data_dir)).absolute()
    if os.path.exists(str(scalar_file)):
        scalar_file.unlink()
    scalar_f = h5py.File('{:s}'.format(str(scalar_file)), 'a')
    parameter_group = scalar_f.create_group('parameters')
    parameter_group['ConvectiveRossbySq'] = ConvectiveRossbySq
    parameter_group['Ekman'] = Ekman
    parameter_group['Prandtl'] = Prandtl
    parameter_group['n_rho'] = n_rho
    parameter_group['L'] = Lmax
    parameter_group['N'] = Nmax
    parameter_group['dt_max'] = float(args['--max_dt'])

    scale_group = scalar_f.create_group('scales')
    scale_group.create_dataset(name='sim_time', shape=(0,), maxshape=(None,), dtype=np.float64)
    task_group = scalar_f.create_group('tasks')
    scalar_keys = ['KE', 'PE', 'Re', 'Ro', 'Lz', 'τ_u2', 'τ_s2']
    for key in scalar_keys:
        task_group.create_dataset(name=key, shape=(0,), maxshape=(None,), dtype=np.float64)
    scalar_index = 0
    scalar_f.close()
    from collections import OrderedDict
    scalar_data = OrderedDict()

bulk_output = solver.evaluator.add_file_handler(data_dir+'/snapshots',sim_dt=50,max_writes=10)
bulk_output.add_task(s, name='s')
bulk_output.add_task(dot(curl(u),curl(u)), name='enstrophy')

checkpoint = solver.evaluator.add_file_handler(data_dir+'/checkpoints',sim_dt=50,max_writes=1)
checkpoint.add_tasks(solver.state)

def vol_avg(q):
    Q = np.sum(vol_correction*weight_r*weight_theta*q['g'])
    Q *= (np.pi)/(Lmax+1)/L_dealias
    Q /= (4/3*np.pi)
    return reducer.reduce_scalar(Q, MPI.SUM)

def L_inf(q):
    if q['g'].size == 0:
        Q = 0
    else:
        Q = np.max(np.abs(q['g']))
    return reducer.reduce_scalar(Q, MPI.MAX)


int_test = de.field.Field(dist=d, bases=(b,), dtype=np.float64)
int_test['g']=1
int_test.require_scales(L_dealias)
logger.info("vol_avg(1)={}".format(vol_avg(int_test)))
logger.info("vol_test={}".format(vol_test))
logger.info("vol_correction={}".format(vol_correction))


# CFL
dr = np.gradient(r1[0,0])
safety = float(args['--safety'])
dt_max = float(args['--max_dt'])
threshold = 0.1
logger.info("max dt={:.2g}".format(dt_max))
logger.info("dr : {}".format(dr))
def calculate_dt(dt_old):
  u.require_scales(1)
  local_freq = np.abs(u['g'][2]/dr) + np.abs(u['g'][1]*(Lmax+2)) + np.abs(u['g'][0]*(Lmax+2))
  global_freq = reducer.global_max(local_freq)
  if global_freq == 0.:
      dt = np.inf
  else:
      dt = 1 / global_freq
  dt *= safety
  if dt > dt_max: dt = dt_max
  if dt < dt_old*(1+threshold) and dt > dt_old*(1-threshold): dt = dt_old
  return dt

q_KE = 0.5*ρ*dot(u,u)
q_Ro = dot(curl(u),curl(u))
q_Re = dot(ρ*u,ρ*u)
q_PE = ρ*T*s
q_Lz = ρ*dot(cross(r_vec_g,u), ez_g)

main_start = time.time()
good_solution = True
while solver.ok and good_solution:
    if solver.iteration % energy_report_cadence == 0:

        KE = vol_avg(q_KE.evaluate())
        Ro = np.sqrt(vol_avg(q_Ro.evaluate()))
        Re = np.sqrt(vol_avg(q_Re.evaluate()))/Ek
        PE = Co2*vol_avg(q_PE.evaluate())
        Lz = vol_avg(q_Lz.evaluate())

        τ_u2 = L_inf(τ_u)
        τ_s2 = L_inf(τ_s)

        logger.info("iter: {:d}, dt={:.2e}, t={:.3e}, KE={:.2e}, PE={:.2e}, Re={:.2e}, Ro={:.2e}, Lz={:.2e}, taus=({:.2e},{:.2e})".format(solver.iteration, dt, solver.sim_time, KE, PE, Re, Ro, Lz,τ_u2,τ_s2))
        good_solution = np.isfinite(KE) and max(τ_u2, τ_s2) < 1e2

        if rank == 0:
            scalar_data['PE'] = PE
            scalar_data['KE'] = KE
            scalar_data['Re'] = Re
            scalar_data['Ro'] = Ro
            scalar_data['Lz'] = Lz
            scalar_data['τ_u2'] = τ_u2
            scalar_data['τ_s2'] = τ_s2

            scalar_f = h5py.File('{:s}'.format(str(scalar_file)), 'a')
            scalar_f['scales/sim_time'].resize(scalar_index+1, axis=0)
            scalar_f['scales/sim_time'][scalar_index] = solver.sim_time
            for key in scalar_data:
                scalar_f['tasks/'+key].resize(scalar_index+1, axis=0)
                scalar_f['tasks/'+key][scalar_index] = scalar_data[key]
            scalar_index += 1
            scalar_f.close()

    elif solver.iteration % report_cadence == 0:
        logger.info("iter: {:d}, dt={:.2e}, t={:.3e}".format(solver.iteration, dt, solver.sim_time))
    if solver.iteration % hermitian_cadence in timestepper_history:
        for field in solver.state:
            field['g']
    solver.step(dt)
    dt = calculate_dt(dt)

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
