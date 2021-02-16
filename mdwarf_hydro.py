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

    --L_max=<L_max>                      Max spherical harmonic [default: 31]
    --N_max=<N_max>                      Max radial polynomial  [default: 31]
    --mesh=<mesh>                        Processor mesh for 3-D runs; if not set a sensible guess will be made

    --max_dt=<max_dt>                    Largest possible timestep [default: 0.5]
    --safety=<safety>                    CFL safety factor [default: 0.4]

    --run_time_diffusion=<run_time_d>    How long to run, in diffusion times [default: 20]
    --run_time_rotation=<run_time_rot>   How long to run, in rotation timescale; overrides run_time_diffusion if set
    --run_time_iter=<run_time_i>         How long to run, in iterations

    --dt_output=<dt_output>              Time between outputs, in rotation times (P_rot = 4pi) [default: 2]
    --scalar_dt_output=<dt_scalar_out>   Time between scalar outputs, in rotation times (P_rot = 4pi) [default: 2]

    --restart=<restart>                  Merged chechpoint file to restart from.
                                         Make sure "--label" is set to avoid overwriting the previous run.

    --label=<label>                      Additional label for run output directory
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

n_rho = float(args['--n_rho'])
radius = 1

Ek = Ekman = float(args['--Ekman'])
Co2 = ConvectiveRossbySq = float(args['--ConvectiveRossbySq'])
Pr = Prandtl = float(args['--Prandtl'])

# load balancing for real variables and parallel runs
if Lmax % 2 == 1:
    nm = 2*(Lmax+1)
else:
    nm = 2*(Lmax+2)

L_dealias = 3/2
N_dealias = 3/2

c = de.coords.SphericalCoordinates('phi', 'theta', 'r')
d = de.distributor.Distributor((c,), mesh=mesh)
b = de.basis.BallBasis(c, (nm,Lmax+1,Nmax+1), radius=radius, dealias=(L_dealias,L_dealias,N_dealias), dtype=np.float64)
b_S2 = b.S2_basis()
phi, theta, r = b.local_grids((L_dealias,L_dealias,N_dealias))
phig,thetag,rg= b.global_grids((L_dealias,L_dealias,N_dealias))
theta_target = thetag[0,(Lmax+1)//2,0]

weight_theta = b.local_colatitude_weights(3/2)
weight_r = b.radial_basis.local_weights(3/2)*r**2

u = de.field.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=np.float64)
p = de.field.Field(dist=d, bases=(b,), dtype=np.float64)
S = de.field.Field(dist=d, bases=(b,), dtype=np.float64)
τ_u = de.field.Field(dist=d, bases=(b_S2,), tensorsig=(c,), dtype=np.float64)
τ_S = de.field.Field(dist=d, bases=(b_S2,), dtype=np.float64)

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
d_exp = lambda A: de.operators.UnaryGridFunctionField(np.exp, A)
d_log = lambda A: de.operators.UnaryGridFunctionField(np.log, A)

# NCCs and variables of the problem
ez = de.field.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=np.float64)
ez.set_scales(b.dealias)
ez['g'][1] = -np.sin(theta)
ez['g'][2] =  np.cos(theta)
ez_g = de.operators.Grid(ez).evaluate()

structure = lane_emden(Nmax, n_rho=n_rho, m=1.5)
T = de.field.Field(dist=d, bases=(b.radial_basis,), dtype=np.float64)
T['g'] = structure['T']['g']
lnρ = de.field.Field(dist=d, bases=(b.radial_basis,), dtype=np.float64)
lnρ['g'] = structure['lnρ']['g']

lnT = d_log(T).evaluate()
T_inv = power(T,-1).evaluate()
grad_lnT = grad(lnT).evaluate()
ρ = d_exp(lnρ).evaluate()
grad_lnρ = grad(lnρ).evaluate()
ρ_inv = d_exp(-lnρ).evaluate()

# Entropy source function, inspired from MESA model
#source = de.field.Field(dist=d, bases=(b.radial_basis,), dtype=np.float64)
source = de.field.Field(dist=d, bases=(b,), dtype=np.float64)
source.require_scales(L_dealias)
# from fits to MESA profile on r = [0,0.85]
σ = 0.11510794072958948
Q0_over_Q1 = 10.969517734412433
Q1 = σ**2/(Q0_over_Q1 + 1) # normalize to σ**2 at r=0
source['g'] = (Q0_over_Q1*np.exp(-r**2/(2*σ**2)) + 1)*Q1
# normalization from Brown et al 2020
#source['g'] /= source(r=0).evaluate()/σ**2
logger.info("Source function: Q0/Q1 = {:}, σ = {:}, Q1 = {:}".format(Q0_over_Q1, σ, Q1))

#e = 0.5*(grad(u) + trans(grad(u)))
e = grad(u) + trans(grad(u))
e.store_last = True

viscous_terms = div(e) + dot(grad_lnρ, e) - 2/3*grad(div(u)) - 2/3*grad_lnρ*div(u)
trace_e = trace(e)
trace_e.store_last = True
Phi = trace(dot(e, e)) - 1/3*(trace_e*trace_e)
# Problem
problem = problems.IVP([u, p, S, τ_u, τ_S])
problem.add_equation((ddt(u) + grad(p) - Co2*T*grad(S) - Ek*ρ_inv*viscous_terms + LiftTau(τ_u,-1),
                      - dot(u, e) - cross(ez_g, u)), condition = "ntheta != 0")
problem.add_equation((dot(grad_lnρ, u) + div(u), 0), condition = "ntheta != 0")
problem.add_equation((u, 0), condition = "ntheta == 0")
problem.add_equation((p, 0), condition = "ntheta == 0")
problem.add_equation((ddt(S) - Ek/Pr*ρ_inv*(lap(S) + dot(grad_lnT, grad(S))) + LiftTau(τ_S,-1),
                     - dot(u, grad(S)) + Ek/Pr*source + Ek/Co2*ρ_inv*T_inv*Phi))
# Boundary conditions
problem.add_equation((radial(u(r=radius)), 0), condition = "ntheta != 0")
problem.add_equation((radial(angular(e(r=radius))), 0), condition = "ntheta != 0")
problem.add_equation((τ_u, 0), condition = "ntheta == 0")
problem.add_equation((S(r=radius), 0))
logger.info("Problem built")
