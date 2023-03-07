import numpy as np
import dedalus.public as de

import logging
logger = logging.getLogger(__name__)
dlog = logging.getLogger('subsystems')
dlog.setLevel(logging.WARNING)

def lane_emden(Nr, m=1.5, n_rho=3, radius=1,
               ncc_cutoff = 1e-10, tolerance = 1e-10, dtype=np.complex128, comm=None):
    # TO-DO: clean this up and make work for ncc ingestion in main script in np.float64 rather than np.complex128
    c = de.SphericalCoordinates('phi', 'theta', 'r')
    d = de.Distributor((c,), comm=comm, dtype=dtype)
    b = de.BallBasis(c, (1, 1, Nr), radius=radius, dtype=dtype)
    br = b.radial_basis
    phi, theta, r = b.local_grids()
    # Fields
    f = d.Field(name='f', bases=b)
    R = d.Field(name='R')
    τ = d.Field(name='τ', bases=b.S2_basis(radius=radius))
    # Parameters and operators
    lap = lambda A: de.Laplacian(A, c)
    lift_basis = b.clone_with(k=2) # match laplacian
    lift = lambda A: de.Lift(A, lift_basis, -1)
    problem = de.NLBVP([f, R, τ])
    problem.add_equation((lap(f) + lift(τ), - R**2 * f**m))
    problem.add_equation((f(r=0), 1))
    problem.add_equation((f(r=radius), np.exp(-n_rho/m, dtype=dtype))) # explicit typing to match domain

    # Solver
    solver = problem.build_solver(ncc_cutoff=ncc_cutoff)
    # Initial guess
    f['g'] = np.cos(np.pi/2 * r)**2
    R['g'] = 5

    # Iterations
    logger.debug('beginning Lane-Emden NLBVP iterations')
    pert_norm = np.inf
    while pert_norm > tolerance:
        solver.newton_iteration()
        pert_norm = sum(pert.allreduce_data_norm('c', 2) for pert in solver.perturbations)
        logger.debug(f'Perturbation norm: {pert_norm:.3e}')
    T = d.Field(name='T', bases=br)
    ρ = d.Field(name='ρ', bases=br)
    lnρ = d.Field(name='lnρ', bases=br)
    T['g'] = f['g']
    ρ['g'] = f['g']**m
    lnρ['g'] = np.log(ρ['g'])

    structure = {'T':T,'lnρ':lnρ}
    for key in structure:
        structure[key].change_scales(1)
    structure['r'] = r
    structure['problem'] = {'c':c, 'b':b, 'problem':problem}
    return structure

def polytrope_shell(Nr, radii, nh, m=1.5, Legendre=False,
                    comm=None, dtype=np.float64):

    Ri, Ro = radii
    c0 = -(Ri-Ro*np.exp(-nh))/(Ro-Ri)
    c1 = Ri*Ro/(Ro-Ri)*(1-np.exp(-nh))

    coords = de.SphericalCoordinates('phi', 'theta', 'r')
    dist = de.Distributor(coords, comm=comm, dtype=dtype)
    if Legendre:
        basis = de.ShellBasis(coords, alpha=(0,0), shape=(1, 1, Nr), radii=radii, dtype=dtype)
    else:
        basis = de.ShellBasis(coords, shape=(1, 1, Nr), radii=radii, dtype=dtype)

    phi, theta, r = basis.local_grids()

    T = dist.Field(name='T', bases=basis)
    T['g'] = c0 + c1/r
    lnρ = (m*np.log(T)).evaluate()
    lnρ.name = 'lnρ'

    structure = {'T':T,'lnρ':lnρ}
    for key in structure:
        structure[key].change_scales(1)
    structure['r'] = r
    structure['problem'] = {'c':coords, 'b':basis, 'problem':None}
    return structure

def thermal_equilibrium_heated_polytrope_shell(Nr, radii, Legendre=False, radius=1,
               ncc_cutoff = 1e-10, tolerance = 1e-10, comm=None):
    dist = de.Distributor(coords, comm=MPI.COMM_SELF, dtype=dtype)
    if Legendre:
        basis = de.ShellBasis(coords, alpha=(0,0), shape=(1, 1, Nr), radii=radii, dtype=dtype)
    else:
        basis = de.ShellBasis(coords, shape=(1, 1, Nr), radii=radii, dtype=dtype)
    s = dist.Field(name='s(r)', bases=basis)
    θ = dist.Field(name='θ(r)', bases=basis)
    b_S2 = basis.S2_basis()
    τ_s1 = dist.Field(name='τ_s1', bases=b_S2)
    τ_s2 = dist.Field(name='τ_s2', bases=b_S2)
    lift_basis = basis.clone_with(k=0)
    lift = lambda A, n: de.Lift(A,lift_basis,n)
    grad_θ0 = dist.VectorField(coords, name='grad_θ0(r)', bases=basis)
    source = dist.Field(bases=basis)
    # now we need to populate grad_θ0 and source, with global values; how are we going to do that?
    equilibrium = de.NLBVP([θ_r, s_r, τ_s1, τ_s2])
    equilibrium.add_equation((- Ek/Pr*(lap(θ) + 2*grad_θ0_r@grad(θ) + grad(θ)@grad(θ))
                              + lift_r(τ_s1,-1) + lift_r(τ_s2,-2), source))
    equilibrium.add_equation((θ_r - γ*s, 0)) #EOS, s_c/cP = 1
    equilibrium.add_equation((radial(grad(s)(r=Ri)), 0))
    equilibrium.add_equation((s(r=Ro), 0))
    eq_solver = equilibrium.build_solver(ncc_cutoff=ncc_cutoff)

    s['g'] = 1e-2*Ma2*np.cos(np.pi/2*(r-Ri)/(Ro-Ri))
    θ['g'] = γ*s['g']

    pert_norm = np.inf
    tolerance = 1e-8
    while pert_norm > tolerance:
        eq_solver.newton_iteration()
        pert_norm = sum(pert.allreduce_data_norm('c', 2) for pert in eq_solver.perturbations)
        logger.debug(f'Perturbation norm: {pert_norm:.3e}')
    logger.info('equilbrium acquired')


if __name__=="__main__":
    LE = lane_emden(64, dtype=np.float64)
    logger.info('T: \n {}'.format(LE['T']['g']))
    logger.info('lnρ: \n {}'.format(LE['lnρ']['g']))
