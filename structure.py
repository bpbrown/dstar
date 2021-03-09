import numpy as np
from dedalus.core import coords, distributor, basis, field, operators, problems, solvers, timesteppers, arithmetic
from dedalus.tools.cache import CachedFunction
from dedalus.tools import logging

def lane_emden(Nmax, Lmax=0, m=1.5, n_rho=3, radius=1,
               ncc_cutoff = 1e-10, tolerance = 1e-10, dtype=np.complex128, comm=None):
    c = coords.SphericalCoordinates('phi', 'theta', 'r')
    d = distributor.Distributor((c,), comm=comm)
    b = basis.BallBasis(c, (1, 1, Nmax+1), radius=radius, dtype=dtype)
    br = b.radial_basis
    phi, theta, r = b.local_grids((1, 1, 1))
    # Fields
    f = field.Field(dist=d, bases=(br,), dtype=dtype, name='f')
    R = field.Field(dist=d, dtype=dtype, name='R')
    τ = field.Field(dist=d, dtype=dtype, name='τ')
    # Parameters and operators
    lap = lambda A: operators.Laplacian(A, c)
    Pow = lambda A,n: operators.Power(A,n)
    LiftTau = lambda A: operators.LiftTau(A, br, -1)
    problem = problems.NLBVP([f, R, τ], ncc_cutoff=ncc_cutoff)
    problem.add_equation((lap(f) + LiftTau(τ), - R**2 * Pow(f,m)))
    problem.add_equation((f(r=0), 1))
    problem.add_equation((f(r=radius), np.exp(-n_rho/m, dtype=dtype))) # explicit typing to match domain

    # Solver
    solver = solvers.NonlinearBoundaryValueSolver(problem)
    # Initial guess
    f['g'] = np.cos(np.pi/2 * r)**2
    R['g'] = 5

    # Iterations
    def error(perts):
        return np.sum([np.sum(np.abs(pert['c'])) for pert in perts])
    err = np.inf
    while err > tolerance:
        solver.newton_iteration()
        err = error(solver.perturbations)

    T = field.Field(dist=d, bases=(br,), dtype=dtype, name='f')
    ρ = field.Field(dist=d, bases=(br,), dtype=dtype, name='f')
    lnρ = field.Field(dist=d, bases=(br,), dtype=dtype, name='f')
    T['g'] = f['g']
    ρ['g'] = f['g']**m
    lnρ['g'] = np.log(ρ['g'])

    structure = {'T':T,'lnρ':lnρ}
    for key in structure:
        structure[key].require_scales(1)
    structure['r'] = r
    structure['problem'] = {'c':c, 'b':b, 'problem':problem}
    return structure

if __name__=="__main__":
    import logging
    logger = logging.getLogger(__name__)

    LE = lane_emden(63)
    print(LE['T']['g'])
    print(LE['lnρ']['g'])
