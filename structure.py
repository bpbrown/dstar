import numpy as np
import dedalus.public as de

import logging
logger = logging.getLogger(__name__)
dlog = logging.getLogger('subsystems')
dlog.setLevel(logging.WARNING)

def lane_emden(Nr, m=1.5, n_rho=3, radius=1, verbose=False,
               ncc_cutoff = 1e-10, tolerance = 1e-10, dtype=np.float64, comm=None):
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
        if verbose: logger.debug(f'Perturbation norm: {pert_norm:.3e}')
    logger.debug(f'final Perturbation norm: {pert_norm:.3e}')
    logger.debug('R = {:}'.format(R['g'][0,0,0]))
    T = f.copy()
    T.name='T'
    lnρ = (m*np.log(T)).evaluate()
    lnρ.name='lnρ'

    structure = {'T':T,'lnρ':lnρ}
    for key in structure:
        structure[key].change_scales(1)
    structure['r'] = r
    structure['problem'] = {'c':c, 'b':b, 'problem':problem}
    return structure

if __name__=="__main__":
    LE = lane_emden(64, dtype=np.float64, verbose=True)
    logger.info('T: \n {}'.format(LE['T']['g']))
    logger.info('lnρ: \n {}'.format(LE['lnρ']['g']))
