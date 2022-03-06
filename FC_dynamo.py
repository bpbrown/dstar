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
    --MagneticPrandtl=<Pm>               Magnetic Prandtl number [default: 1]
    --gamma=<gamma>                      Ideal gas gamma [default: 5/3]
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
data_dir += '_Co{}_Ma{}_Ek{}_Pr{}_Pm{}'.format(args['--ConvectiveRossbySq'],args['--Mach'],args['--Ekman'],args['--Prandtl'],args['--MagneticPrandtl'])
data_dir += '_Th{}_R{}'.format(args['--Ntheta'], args['--Nr'])
if args['--thermal_equilibrium']:
    data_dir += '_therm'
if args['--benchmark']:
    data_dir += '_benchmark'
if args['--label']:
    data_dir += '_{:s}'.format(args['--label'])

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
Ma = float(args['--Mach'])
Ma2 = Ma*Ma
γ = gamma = float(Fraction(args['--gamma']))
Pr = Prandtl = float(args['--Prandtl'])
Pm = MagneticPrandtl = float(args['--MagneticPrandtl'])

import dedalus.public as de
from dedalus.extras import flow_tools


logger.debug(sys.argv)
logger.debug('-'*40)
logger.info("saving data in {}".format(data_dir))
logger.info("Run parameters")
logger.info("Ek = {}, Co2 = {}, Ma = {}, Pr = {}, Pm = {}".format(Ek,Co2,Ma,Pr, Pm))
scrC = 1/(gamma-1)*Co2/Ma2
logger.info("scrC = {:}, Co2 = {:}, Ma2 = {:}".format(scrC, Co2, Ma2))

from structure import lane_emden

dealias = float(args['--dealias'])

c = de.SphericalCoordinates('phi', 'theta', 'r')
d = de.Distributor(c, mesh=mesh, dtype=np.float64)
b = de.BallBasis(c, shape=(Nφ,Nθ,Nr), radius=radius, dealias=dealias, dtype=np.float64)
b_S2 = b.S2_basis()
phi, theta, r = b.local_grids()

p = d.Field(name='p', bases=b)
Υ = d.Field(name='Υ', bases=b)
θ = d.Field(name='θ', bases=b)
s = d.Field(name='s', bases=b)
u = d.VectorField(c, name='u', bases=b)
A = d.VectorField(c, name="A", bases=b)
φ = d.Field(name="φ", bases=b)
τ_φ = d.Field(name="τ_φ")
τ_s = d.Field(name='τ_s', bases=b_S2)
τ_u = d.VectorField(c, name='τ_u', bases=b_S2)
τ_A = d.VectorField(c, name="τ_A", bases=b_S2)

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
lift = lambda A, n: de.Lift(A,lift_basis,n)
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

m_ad = 1/(γ-1)
logger.info("solving lane emden with m = {:}".format(m_ad))
structure = lane_emden(Nr, n_rho=n_rho, m=m_ad, comm=MPI.COMM_SELF)

bk2 = b.clone_with(k=2)
bk1 = b.clone_with(k=1)
T = d.Field(name='T', bases=b.radial_basis)
lnρ = d.Field(name='lnρ', bases=b.radial_basis)

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

h0_g = de.Grid(h0).evaluate()
h0_inv_g = de.Grid(1/h0).evaluate()
grad_h0_g = de.Grid(grad(h0)).evaluate()
ρ0_g = de.Grid(ρ0).evaluate()

ρ0_grad_h0_g = de.Grid(ρ0*grad(h0)).evaluate()
ρ0_h0_g = de.Grid(ρ0*h0).evaluate()

# Entropy source function, inspired from MESA model
def source_function(r):
    # from fits to MESA profile on r = [0,0.85]
    σ = 0.11510794072958948
    Q0_over_Q1 = 10.969517734412433
    # normalization from Brown et al 2020
    Q1 = σ**-2/(Q0_over_Q1 + 1) # normalize to σ**-2 at r=0
    logger.info("Source function: Q0/Q1 = {:.3g}, σ = {:.3g}, Q1 = {:.3g}".format(Q0_over_Q1, σ, Q1))
    return (Q0_over_Q1*np.exp(-r**2/(2*σ**2)) + 1)*Q1

source_func = d.Field(name='S', bases=b)
source_func['g'] = source_function(r)

# for RHS source function, need θ0 on the full ball grid (rather than just the radial grid)
θ0_RHS = d.Field(name='θ0_RHS', bases=b)
θ0.change_scales(1)
θ0_RHS.require_grid_space()
if θ0['g'].size > 0:
    θ0_RHS['g'] = θ0['g']
ε = Ma2
source = de.Grid(Ek/Pr*(ε*ρ0/h0*source_func)).evaluate() # + lap(θ0_RHS) + dot(grad(θ0_RHS),grad(θ0_RHS)) ) ).evaluate()
source.name='source'

B = curl(A)
J = -lap(A) #curl(B)

#e = 0.5*(grad(u) + trans(grad(u)))
e = grad(u) + trans(grad(u))
e.store_last = True

ω = curl(u)
viscous_terms = div(e) - 2/3*grad(div(u))
trace_e = trace(e)
trace_e.store_last = True
Phi = trace(dot(e, e)) - 1/3*(trace_e*trace_e)

logger.info("NCC expansions:")
for ncc in [ρ0, ρ0*grad(h0), ρ0*h0, ρ0*grad(θ0), h0*grad(Υ0)]:
    logger.info("{}: {}".format(ncc.evaluate(), np.where(np.abs(ncc.evaluate()['c']) >= ncc_cutoff)[0].shape))


#Problem
problem = de.IVP([u, Υ, θ, s, φ, A, τ_u, τ_s, τ_φ, τ_A])
problem.add_equation((ρ0*(ddt(u) + scrC*(h0*grad(θ) + grad_h0*θ)
                      - scrC*h0*grad(s))
                      - Ek*viscous_terms
                      + lift(τ_u,-1),
                      ρ0_g*(-dot(u,grad(u)) - cross(ez_g, u))
                      -scrC*ρ0_grad_h0_g*(np.expm1(θ)-θ)
                      -scrC*ρ0_h0_g*np.expm1(θ)*grad(θ)
                      -scrC*ρ0_h0_g*np.expm1(θ)*grad(s)
                      + np.exp(-Υ)*cross(J,B) ))
problem.add_equation((h0*(ddt(Υ) + div(u) + dot(u, grad_Υ0)),
                      -h0_g*dot(u, grad(Υ)) ))
problem.add_equation((θ - (γ-1)*Υ - γ*s, 0)) #EOS, s_c/cP = 1
#TO-DO:
# add ohmic heat
problem.add_equation((ρ0*(ddt(s))
                      - Ek/Pr*(lap(θ)+2*dot(grad_θ0,grad(θ)))
                      + lift(τ_s,-1),
                      - ρ0_g*dot(u,grad(s))
                      + Ek/Pr*dot(grad(θ),grad(θ))
                      + Ek/scrC*0.5*h0_inv_g*Phi
                      + source ))
problem.add_equation((div(A) + τ_φ, 0)) # coulomb gauge
# currently sets ρ = ρ0*exp(Υ) -> ρ0 (neglects exp(Υ), should appear in laplacian)
problem.add_equation((ρ0*(ddt(A) + grad(φ)) - Ek/Pm*lap(A) + lift(τ_A,-1),
                        ρ0_g*cross(u, B)))
# Boundary conditions
problem.add_equation((radial(u(r=radius)), 0))
problem.add_equation((radial(angular(e(r=radius))), 0))
problem.add_equation((s(r=radius), 0))
problem.add_equation((integ(φ), 0))
problem.add_equation((dot(r_S2, grad(A)(r=radius))+ellp1(A)(r=radius)/radius, 0))
logger.info("Problem built")

if args['--thermal_equilibrium']:
    logger.info("solving for thermal equilbrium")
    equilibrium = de.LBVP([s, τ_s])
    equilibrium.add_equation((-Ek/Pr*T*(lap(s)+ dot(grad_lnT1, grad(s))) + lift(τ_s,-1), source))
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

# Analysis
eφ = d.VectorField(c, bases=b)
eφ['g'][0] = 1
eθ = d.VectorField(c, bases=b)
eθ['g'][1] = 1
er = d.VectorField(c, bases=b)
er['g'][2] = 1

ur = dot(u, er)
uθ = dot(u, eθ)
uφ = dot(u, eφ)
Br = dot(B, er)
Bθ = dot(B, eθ)
Bφ = dot(B, eφ)
Aφ = dot(A, eφ)

ρ_cyl = d.Field(bases=b)
ρ_cyl['g'] = r*np.sin(theta)
Ωz = uφ/ρ_cyl # this is not ω_z; misses gradient terms; this is angular differential rotation.

u_fluc = u - azavg(ur)*er - azavg(uθ)*eθ - azavg(uφ)*eφ
u_fluc.store_last = True

B_fluc = B - azavg(Br)*er - azavg(Bθ)*eθ - azavg(Bφ)*eφ
B_fluc.store_last = True

ρ = ρ0*np.exp(Υ)
h = h0*np.exp(θ)
KE = 0.5*ρ*dot(u,u)
DRKE = 0.5*ρ*(azavg(uφ)**2)
MCKE = 0.5*ρ*(azavg(ur)**2 + azavg(uθ)**2)
FKE = KE - DRKE - MCKE #0.5*dot(u_fluc, u_fluc)
KE.store_last = True
DRKE.store_last = True
MCKE.store_last = True
FKE.store_last = True

Ma2_ad = 1/(γ-1)*dot(u,u)/h
Ma2_ad.store_last = True

ME = 0.5*dot(B,B)
TME = 0.5*(azavg(Bφ)**2)
PME = 0.5*(azavg(Br)**2 + azavg(Bθ)**2)
FME = ME - TME - PME #0.5*dot(B_fluc, B_fluc)
ME.store_last = True
TME.store_last = True
PME.store_last = True
FME.store_last = True

PE = scrC*ρ*T*s
PE.name = 'PE'
PE.store_last = True

Lz = dot(cross(r_vec,ρ*u), ez)
Lz.name='Lz'
Lz.store_last = True

enstrophy = dot(curl(u),curl(u))
enstrophy.store_last = True
enstrophy_fluc = dot(curl(u_fluc),curl(u_fluc))
enstrophy_fluc.store_last = True

Re2 = dot(u,u)*(ρ/Ek)**2
Re2.store_last=True
Re2_fluc = dot(u_fluc,u_fluc)*(ρ/Ek)**2
Re2_fluc.store_last=True

scalar_dt = float(args['--scalar_dt'])
traces = solver.evaluator.add_file_handler(data_dir+'/traces', sim_dt=scalar_dt, max_writes=np.inf)
traces.add_task(avg(KE), name='KE')
traces.add_task(avg(DRKE), name='DRKE')
traces.add_task(avg(MCKE), name='MCKE')
traces.add_task(avg(FKE), name='FKE')
traces.add_task(avg(Ma2_ad), name='Ma2')
traces.add_task(avg(ME), name='ME')
traces.add_task(avg(TME), name='TME')
traces.add_task(avg(PME), name='PME')
traces.add_task(avg(FME), name='FME')
traces.add_task(integ(KE)/Ek**2, name='E0')
traces.add_task(np.sqrt(avg(enstrophy)), name='Ro')
traces.add_task(np.sqrt(avg(Re2)), name='Re')
traces.add_task(np.sqrt(avg(enstrophy_fluc)), name='Ro_fluc')
traces.add_task(np.sqrt(avg(Re2_fluc)), name='Re_fluc')
traces.add_task(avg(PE), name='PE')
traces.add_task(avg(Lz), name='Lz')
traces.add_task(np.abs(τ_φ), name='τ_φ')
traces.add_task(shellavg(np.abs(τ_s)), name='τ_s')
traces.add_task(shellavg(np.sqrt(dot(τ_u,τ_u))), name='τ_u')
traces.add_task(shellavg(np.sqrt(dot(τ_A,τ_A))), name='τ_A')

slice_dt = float(args['--slice_dt'])
slices = solver.evaluator.add_file_handler(data_dir+'/slices', sim_dt = slice_dt, max_writes = 10, virtual_file=True, mode=mode)
slices.add_task(s(theta=np.pi/2), name='s')
slices.add_task(enstrophy(theta=np.pi/2), name='enstrophy')
slices.add_task(azavg(Ωz), name='<Ωz>')
slices.add_task(azavg(Bφ), name='<Bφ>')
slices.add_task(azavg(Aφ), name='<Aφ>')
slices.add_task(azavg(s), name='<s>')
slices.add_task(shellavg(s), name='s(r)')
slices.add_task(shellavg(ρ*dot(er, u)*(p+0.5*dot(u,u))), name='F_h(r)')
slices.add_task(shellavg(ρ*dot(er, u)*dot(u,u)), name='F_KE(r)')
slices.add_task(shellavg(-Co2*Ek/Pr*T*dot(er, grad(s))), name='F_κ(r)')
slices.add_task(shellavg(Co2*source), name='F_source(r)')
slices.add_task(Br(r=radius), name='Br') # is this sufficient?  Should we be using radial(B) instead?

report_cadence = 100
flow = flow_tools.GlobalFlowProperty(solver, cadence=report_cadence)
flow.add_property(Re2, name='Re2')
flow.add_property(enstrophy, name='Ro2')
flow.add_property(Re2_fluc, name='Re2_fluc')
flow.add_property(enstrophy_fluc, name='Ro2_fluc')
flow.add_property(KE, name='KE')
flow.add_property(ME, name='ME')
flow.add_property(Ma2_ad, name='Ma2')
flow.add_property(PE, name='PE')
flow.add_property(Lz, name='Lz')
flow.add_property(np.abs(τ_s), name='|τ_s|')
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
        Re_avg = np.sqrt(flow.volume_integral('Re2')/vol)
        Ro_avg = np.sqrt(flow.volume_integral('Ro2')/vol)
        Re_fluc_avg = np.sqrt(flow.volume_integral('Re2_fluc')/vol)
        Ro_fluc_avg = np.sqrt(flow.volume_integral('Ro2_fluc')/vol)
        PE_avg = flow.volume_integral('PE')/vol
        ME_avg = flow.volume_integral('ME')/vol
        Lz_avg = flow.volume_integral('Lz')/vol
        Ma_avg = np.sqrt(flow.volume_integral('Ma2')/vol)
        max_τ = np.max([flow.max('|τ_u|'), flow.max('|τ_s|')])

        log_string = "iter: {:d}, dt={:.1e}, t={:.3e} ({:.2e})".format(solver.iteration, dt, solver.sim_time, solver.sim_time*Ek)
        log_string += ", Ma={:.2e}, KE={:.2e}, ME={:.2e}, PE={:.2e}".format(Ma_avg, KE_avg, ME_avg, PE_avg)
        log_string += ", Re={:.1e}/{:.1e}, Ro={:.1e}/{:.1e}".format(Re_avg, Re_fluc_avg, Ro_avg, Ro_fluc_avg)
        log_string += ", Lz={:.1e}, τ={:.1e}".format(Lz_avg, max_τ)
        logger.info(log_string)
        good_solution = np.isfinite(E0)
    solver.step(dt)

solver.log_stats()
logger.debug("mode-stages/DOF = {}".format(solver.total_modes/(Nφ*Nθ*Nr)))
