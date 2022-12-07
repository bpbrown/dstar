"""
Dedalus script for full sphere anelastic convection,
applied to the Marti Boussinesq benchmark.

Usage:
    boussinesq_hydro.py [options]

Options:
    --Ekman=<Ekman>                      Ekman number    [default: 3e-4]
    --ConvectiveRossbySq=<Co2>           Squared Convective Rossby = Ra*Ek**2/Pr [default: 2.85e-2]
    --Prandtl=<Prandtl>                  Prandtl number  [default: 1]

    --Ntheta=<Ntheta>                    Latitudinal modes [default: 32]
    --Nr=<Nr>                            Radial modes [default: 32]
    --mesh=<mesh>                        Processor mesh for 3-D runs; if not set a sensible guess will be made

    --benchmark                          Use benchmark initial conditions
    --ell_benchmark=<ell_benchmark>      Integer value of benchmark perturbation m=+-ell [default: 3]

    --max_dt=<max_dt>                    Largest possible timestep [default: 0.1]
    --safety=<safety>                    CFL safety factor [default: 0.4]
    --fixed_dt                           Fix dt

    --run_time_diffusion=<run_time_d>    How long to run, in diffusion times [default: 20]
    --run_time_rotation=<run_time_rot>   How long to run, in rotation timescale; overrides run_time_diffusion if set
    --run_time_iter=<run_time_i>         How long to run, in iterations

    --dt_output=<dt_output>              Time between outputs, in rotation times (P_rot = 4pi) [default: 2]
    --scalar_dt_output=<dt_scalar_out>   Time between scalar outputs, in rotation times (P_rot = 4pi) [default: 2]

    --restart=<restart>                  Merged chechpoint file to restart from.
                                         Make sure "--label" is set to avoid overwriting the previous run.

    --label=<label>                      Additional label for run output directory

    --ncc_cutoff=<ncc_cutoff>            Amplitude to truncate NCC terms [default: 1e-10]
    --plot_sparse                        Plot sparsity structures for L+M and it's LU decomposition

"""
import logging
logger = logging.getLogger(__name__)
for system in ['matplotlib', 'evaluator']:
    dlog = logging.getLogger(system)
    dlog.setLevel(logging.WARNING)

import sys
import numpy as np
from mpi4py import MPI

from docopt import docopt
args = docopt(__doc__)

data_dir = './'+sys.argv[0].split('.py')[0]
data_dir += '_Ek{}_Co{}_Pr{}'.format(args['--Ekman'],args['--ConvectiveRossbySq'],args['--Prandtl'])
data_dir += '_Th{}_R{}'.format(args['--Ntheta'], args['--Nr'])
if args['--benchmark']:
    data_dir += '_benchmark'
if args['--label']:
    data_dir += '_{:s}'.format(args['--label'])
logger.info("saving data in {}".format(data_dir))

import dedalus.tools.logging as dedalus_logging
dedalus_logging.add_file_handler(data_dir+'/logs/dedalus_log', 'DEBUG')

import dedalus.public as de
from dedalus.extras import flow_tools

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

Nθ = int(args['--Ntheta'])
Nr = int(args['--Nr'])
Nφ = Nθ*2

ncc_cutoff = float(args['--ncc_cutoff'])

radius = 1

Ek = Ekman = float(args['--Ekman'])
Co2 = ConvectiveRossbySq = float(args['--ConvectiveRossbySq'])
Pr = Prandtl = float(args['--Prandtl'])
logger.info("Ek = {}, Co2 = {}, Pr = {}".format(Ek,Co2,Pr))

dealias = 3/2

dtype = np.float64

c = de.SphericalCoordinates('phi', 'theta', 'r')
d = de.Distributor(c, mesh=mesh, dtype=dtype)
b = de.BallBasis(c, shape=(Nφ,Nθ,Nr), radius=radius, dealias=dealias, dtype=dtype)
phi, theta, r = b.local_grids()

b_ncc = de.BallBasis(c, shape=(1,1,Nr), radius=radius, dealias=dealias, dtype=dtype)

u = d.VectorField(c, name='u', bases=b)
p = d.Field(name='p', bases=b)
s = d.Field(name='s', bases=b)
τ_p = d.Field(name="τ_p")
τ_u = d.VectorField(c, name="τ_u", bases=b.S2_basis())
τ_s = d.Field(name="τ_s", bases=b.S2_basis())

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
lift_basis = b #b.clone_with(k=2)
lift = lambda A, n: de.Lift(A, b, n)
integ = lambda A: de.Integrate(A, c)
avg = lambda A: de.Integrate(A, c)/(4/3*np.pi*radius**3)
shellavg = lambda A: de.Average(A, c.S2coordsys)

# NCCs and variables of the problem
bk1 = b.clone_with(k=1) # ez on k+1 level to match curl(u)
ez = d.VectorField(c, name='ez', bases=bk1)
ez['g'][1] = -np.sin(theta)
ez['g'][2] =  np.cos(theta)
ex = d.VectorField(c, bases=b, name='ex')
ex['g'][2] = np.sin(theta)*np.cos(phi)
ex['g'][1] = np.cos(theta)*np.cos(phi)
ex['g'][0] = -np.sin(phi)
ey = d.VectorField(c, bases=b, name='ey')
ey['g'][2] = np.sin(theta)*np.sin(phi)
ey['g'][1] = np.cos(theta)*np.sin(phi)
ey['g'][0] = np.cos(phi)

z = d.Field(name='z', bases=b)
x = d.Field(name='x', bases=b)
y = d.Field(name='y', bases=b)
x['g'] = r*np.sin(theta)*np.cos(phi)
y['g'] = r*np.sin(theta)*np.sin(phi)
z['g'] = r*np.cos(theta)

r_vec = d.VectorField(c, name='r_vec', bases=b_ncc)
r_vec['g'][2] = r

# Entropy source function; here constant volume heating rate
source_func = d.Field(name='S', bases=b)
source_func['g'] =  Ek/Pr*3
source = de.Grid(source_func).evaluate()

# for boundary condition
e = grad(u) + trans(grad(u))
e.store_last = True

m, ell, n = d.coeff_layout.local_group_arrays(b.domain, scales=1)
mask = (ell==1)*(n==0)

τ_L = d.VectorField(c, bases=b, name='τ_L')
τ_L.valid_modes[2] *= mask
τ_L.valid_modes[0] = False
τ_L.valid_modes[1] = False

problem = de.IVP([p, u, s, τ_p, τ_u, τ_s, τ_L])
problem.add_equation((div(u) + τ_p, 0))
problem.add_equation((ddt(u) + grad(p)  - Ek*lap(u) - Co2*r_vec*s + τ_L + lift(τ_u,-1),
                      cross(u, curl(u) + ez) ))
#problem.add_equation((τ_L, 0))
problem.add_equation((2*u, 0))
eq = problem.equations[-1]
print(eq['LHS'].valid_modes.shape)
eq['LHS'].valid_modes[2] *= mask
eq['LHS'].valid_modes[0] = False
eq['LHS'].valid_modes[1] = False

problem.add_equation((ddt(s) - Ek/Pr*(lap(s)) + lift(τ_s,-1),
                     - (dot(u, grad(s))) + source ))
# Boundary conditions
problem.add_equation((radial(u(r=radius)), 0))
problem.add_equation((radial(angular(e(r=radius))), 0))
problem.add_equation((s(r=radius), 0))
problem.add_equation((integ(p), 0))  # Pressure gauge
logger.info("Problem built")

s['g'] = 0.5*(1-r**2) # static solution

if args['--benchmark']:
    amp = 1e-1
    𝓁 = int(args['--ell_benchmark'])
    norm = 1/(2**𝓁*np.math.factorial(𝓁))*np.sqrt(np.math.factorial(2*𝓁+1)/(4*np.pi))
    s['g'] += amp*norm*r**𝓁*(1-r**2)*(np.cos(𝓁*phi)+np.sin(𝓁*phi))*np.sin(theta)**𝓁
    logger.info("benchmark run with perturbations at ell={} with norm={}".format(𝓁, norm))
else:
    amp = 1e-5
    noise = d.Field(name='noise', bases=b)
    noise.fill_random('g', seed=42, distribution='standard_normal')
    noise.low_pass_filter(scales=0.25)
    s['g'] += amp*noise['g']

# Solver
solver = problem.build_solver(de.SBDF2, ncc_cutoff=ncc_cutoff)

KE = 0.5*dot(u,u)
PE = Co2*s
L = cross(r_vec,u)
enstrophy = dot(curl(u),curl(u))

coeffs = solver.evaluator.add_file_handler(data_dir+'/coeffs', sim_dt = 100, max_writes = 10)
coeffs.add_task(u, name='ρu', layout='c')

traces = solver.evaluator.add_file_handler(data_dir+'/traces', sim_dt=10, max_writes=None)
traces.add_task(avg(KE), name='KE')
traces.add_task(integ(KE)/Ek**2, name='E0')
traces.add_task(np.sqrt(avg(enstrophy)), name='Ro')
traces.add_task(np.sqrt(2*avg(KE))/Ek, name='Re')
traces.add_task(avg(PE), name='PE')
traces.add_task(integ(dot(L,ex)), name='Lx')
traces.add_task(integ(dot(L,ey)), name='Ly')
traces.add_task(integ(dot(L,ez)), name='Lz')
traces.add_task(integ(-x*div(L)), name='Λx')
traces.add_task(integ(-y*div(L)), name='Λy')
traces.add_task(integ(-z*div(L)), name='Λz')
traces.add_task(shellavg(np.sqrt(dot(τ_u,τ_u))), name='τ_u')
traces.add_task(shellavg(np.sqrt(dot(τ_L,τ_L))), name='τ_L')
traces.add_task(shellavg(np.abs(τ_s)), name='τ_s')
traces.add_task(np.abs(τ_p), name='τ_p')

report_cadence = 100
flow = flow_tools.GlobalFlowProperty(solver, cadence=report_cadence)
flow.add_property(np.sqrt(KE*2)/Ek, name='Re')
flow.add_property(np.sqrt(enstrophy), name='Ro')
flow.add_property(KE, name='KE')
flow.add_property(PE, name='PE')
flow.add_property(dot(L,ex), name='Lx')
flow.add_property(dot(L,ey), name='Ly')
flow.add_property(dot(L,ez), name='Lz')
flow.add_property(np.sqrt(dot(τ_u,τ_u)), name='|τ_u|')
flow.add_property(np.sqrt(dot(τ_L,τ_L)), name='|τ_L|')
flow.add_property(np.abs(τ_s), name='|τ_s|')
flow.add_property(np.abs(τ_p), name='|τ_p|')

max_dt = float(args['--max_dt'])
if args['--fixed_dt']:
    dt = max_dt
else:
    dt = max_dt/10
if not args['--restart']:
    mode = 'overwrite'
else:
    write, dt = solver.load_state(args['--restart'])
    mode = 'append'

cfl_safety_factor = float(args['--safety'])
CFL = flow_tools.CFL(solver, initial_dt=dt, cadence=1, safety=cfl_safety_factor, max_dt=max_dt, threshold=0.1)
CFL.add_velocity(u)

if args['--run_time_rotation']:
    solver.stop_sim_time = float(args['--run_time_rotation'])
else:
    solver.stop_sim_time = float(args['--run_time_diffusion'])/Ek

if args['--run_time_iter']:
    solver.stop_iteration = int(float(args['--run_time_iter']))

good_solution = True
vol = 4*np.pi/3
while solver.proceed and good_solution:
    if not args['--fixed_dt']:
        dt = CFL.compute_timestep()
    if solver.iteration % report_cadence == 0 and solver.iteration > 0:
        KE_avg = flow.volume_integral('KE')/vol # volume average needs a defined volume
        E0 = flow.volume_integral('KE')/Ek**2 # integral rather than avg
        Re_avg = flow.volume_integral('Re')/vol
        Ro_avg = flow.volume_integral('Ro')/vol
        PE_avg = flow.volume_integral('PE')/vol
        Lx_int = flow.volume_integral('Lx')
        Ly_int = flow.volume_integral('Ly')
        Lz_int = flow.volume_integral('Lz')
        τ_u_m = flow.max('|τ_u|')
        τ_s_m = flow.max('|τ_s|')
        τ_L_m = flow.max('|τ_L|')
        log_string = "iter: {:d}, dt={:.1e}, t={:.3e} ({:.2e})".format(solver.iteration, dt, solver.sim_time, solver.sim_time*Ek)
        log_string += ", KE={:.2e} ({:.6e}), PE={:.2e}".format(KE_avg, E0, PE_avg)
        log_string += ", Re={:.1e}, Ro={:.1e}".format(Re_avg, Ro_avg)
        log_string += ", L=({:.1e},{:.1e},{:.1e})".format(Lx_int, Ly_int, Lz_int)
        log_string += ", τ=({:.1e},{:.1e},{:.1e})".format(τ_u_m, τ_s_m, τ_L_m)
        logger.info(log_string)
        good_solution = np.isfinite(E0)
    solver.step(dt)

solver.log_stats()
