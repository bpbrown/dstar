import numpy as np
from dedalus.core import coords, distributor, basis, field, operators, problems, solvers, timesteppers, arithmetic
from dedalus.tools.cache import CachedFunction
from dedalus.tools import logging

def lane_emden(Nmax, Lmax=0, m=1.5, n_rho=3, radius=1,
               ncc_cutoff = 1e-10, tolerance = 1e-10, dtype=np.float64):
    c = coords.SphericalCoordinates('phi', 'theta', 'r')
    d = distributor.Distributor((c,))
    if Lmax > 0:
        nm = 2*(Lmax+1)
    else:
        nm = 1
    b = basis.BallBasis(c, (nm,Lmax+1, Nmax+1), radius=radius, dtype=dtype)
    phi, theta, r = b.local_grids((1, 1, 1))
    # Fields
    T = field.Field(dist=d, bases=(b,), dtype=dtype, name='T')
    τ = field.Field(dist=d, bases=(b.S2_basis(),), dtype=dtype, name='τ')
    C = field.Field(dist=d, dtype=dtype, name='C')

    T.require_scales(1)
    T['g'] = np.cos(np.pi/2 * r)*0.9
    C['g'] = 2
    T_top = field.Field(dist=d, bases=(b.S2_basis(),), dtype=dtype, name='T_top')

    # Parameters and operators
    lap = lambda A: operators.Laplacian(A, c)
    Pow = lambda A,n: operators.Power(A,n)
    LiftTau = lambda A: operators.LiftTau(A, b, -1)

    T_center = 1
    T_top['g'] = np.exp(-n_rho/m)
    # from poisson:
    # lap(phi) = -C1*rho
    # from HS balance:
    # grad(phi) ~ grad(T)
    # therefore:
    # lap(T) = -C2*rho = -C3*T**n
    problem = problems.NLBVP([T,τ,C], ncc_cutoff=ncc_cutoff)
    #problem.add_equation((lap(T) + LiftTau(τ), -C*Pow(T,m)))
    problem.add_equation((lap(T) + LiftTau(τ), -T))
    #problem.add_equation((lap(T) + LiftTau(τ), T*T))
    #problem.add_equation((T(r=0), T_center))
    problem.add_equation((T(r=radius), T_top))

    # Solver
    solver = solvers.NonlinearBoundaryValueSolver(problem)


    # Iterations
    def error(perts):
        return np.sum([np.sum(np.abs(pert['c'])) for pert in perts])
    err = np.inf
    while err > tolerance:
        solver.newton_iteration()
        err = error(solver.perturbations)
    return T

def test_heat_ball(Nmax, Lmax, dtype):
    # Bases
    c = coords.SphericalCoordinates('phi', 'theta', 'r')
    d = distributor.Distributor((c,))
    if Lmax > 0:
        nm = 2*(Lmax+1)
    else:
        nm = 1
    b = basis.BallBasis(c, (nm,Lmax+1, Nmax+1), radius=1, dtype=dtype)
    b_S2 = b.S2_basis()
    phi, theta, r = b.local_grids((1, 1, 1))

    # Fields
    u = field.Field(name='u', dist=d, bases=(b,), dtype=dtype)
    τu = field.Field(name='τu', dist=d, bases=(b.S2_basis(),), dtype=dtype)
    F = field.Field(name='a', dist=d, bases=(b,), dtype=dtype)
    F['g'] = 6
    # Problem
    Lap = lambda A: operators.Laplacian(A, c)
    LiftTau = lambda A: operators.LiftTau(A, b, -1)
    problem = problems.LBVP([u, τu])
    problem.add_equation((Lap(u) + LiftTau(τu), F))
    problem.add_equation((u(r=1), 0))
    # Solver
    solver = solvers.LinearBoundaryValueSolver(problem)
    solver.solve()
    return u

def test_heat_ball_nlbvp(Nmax, Lmax, dtype):
    # Bases
    c = coords.SphericalCoordinates('phi', 'theta', 'r')
    d = distributor.Distributor((c,))
    if Lmax > 0:
        nm = 2*(Lmax+1)
    else:
        nm = 1
    b = basis.BallBasis(c, (nm,Lmax+1, Nmax+1), radius=1, dtype=dtype)
    b_S2 = b.S2_basis()
    phi, theta, r = b.local_grids((1, 1, 1))

    # Fields
    u = field.Field(name='u', dist=d, bases=(b,), dtype=dtype)
    τu = field.Field(name='τu', dist=d, bases=(b.S2_basis(),), dtype=dtype)
    F = field.Field(name='F', dist=d, bases=(b,), dtype=dtype)
    F['g'] = 6
    # Problem
    Lap = lambda A: operators.Laplacian(A, c)
    LiftTau = lambda A: operators.LiftTau(A, b, -1)
    problem = problems.NLBVP([u, τu])
    problem.add_equation((Lap(u) + LiftTau(τu), F))
    problem.add_equation((u(r=1), 0))
    # Solver
    solver = solvers.NonlinearBoundaryValueSolver(problem)
    u['g'] = 1
    tolerance = 1e-6
    # Iterations
    def error(perts):
        return np.sum([np.sum(np.abs(pert['c'])) for pert in perts])
    err = np.inf
    while err > tolerance:
        solver.newton_iteration()
        err = error(solver.perturbations)
    return u

if __name__=="__main__":
    import logging
    logger = logging.getLogger(__name__)

    T = test_heat_ball(63, 0, np.float64)
    TL = T['g']
    logger.info("test")
    T = test_heat_ball_nlbvp(63, 0, np.float64)
    print("T error : |NLBVP - LBVP| = {:.2g}".format(np.max(np.abs(T['g']-TL))))

    # works with Lmax=3, but not with Lmax=0
    # well, not really
    T = lane_emden(63, Lmax=3)
    print(T['g'])
