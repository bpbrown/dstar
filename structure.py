import numpy as np
import dedalus.public as de

import logging
logger = logging.getLogger(__name__)
dlog = logging.getLogger('subsystems')
dlog.setLevel(logging.WARNING)

def lane_emden(Nr, m=1.5, n_rho=3, radius=1, verbose=False,
               ncc_cutoff = 1e-10, tolerance = 1e-10, dtype=np.float64, comm=None):
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

def plot_model_star(Nr, structure, dtype=np.float64, ncc_cutoff=1e-10):
    c = de.SphericalCoordinates('phi', 'theta', 'r')
    d = de.Distributor((c,), dtype=dtype)
    b = de.BallBasis(c, (1, 1, Nr), radius=1, dtype=dtype)
    phi, theta, r = b.local_grids()
    grad = lambda A: de.Gradient(A, c)
    lap = lambda A: de.Laplacian(A, c)
    dot = lambda A, B: de.DotProduct(A, B)

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
    ε = 1 #1e-2
    source = de.Grid((ε*ρ0/h0*source_func)).evaluate() # + lap(θ0_RHS) + dot(grad(θ0_RHS),grad(θ0_RHS)) ) ).evaluate()
    source2 = de.Grid((ε*ρ0/h0*source_func) + lap(θ0_RHS) + dot(grad(θ0_RHS),grad(θ0_RHS)) ).evaluate()
    source.name='source'

    fig, ax = plt.subplots()
    i = (0, 0, slice(None))
    #ax.plot(r[i], source['g'][i])
    #ax.plot(r[i], source2['g'][i])
    ax.plot(r[i], (lap(θ0_RHS) + dot(grad(θ0_RHS),grad(θ0_RHS))).evaluate()['g'][i])
    ax.plot(r[i], (1/T*lap(T)).evaluate()['g'][i], linestyle='dashed')
    fig.savefig('star_source.pdf')

    er = d.VectorField(c, name='er')
    er['g'][2] = 1
    fig, ax = plt.subplots(nrows=2)
    logger.info("NCC expansions:")
    for ncc in [ρ0, ρ0*grad(h0), ρ0*h0, ρ0*grad(θ0), h0*grad(Υ0)]:
        ncc = ncc.evaluate()
        if ncc['g'].ndim == 4:
            i_ncc = (2, 0, 0, slice(None))
            i_c_ncc = (1, 0, 0, slice(None))
            #ncc = dot(er, ncc).evaluate()
            #i_c_ncc = i_ncc = i
        else:
            i_c_ncc = i_ncc = i
        i_good = np.where(np.abs(ncc['c']) >= ncc_cutoff)
        i_bad = np.where(np.abs(ncc['c']) < ncc_cutoff)
        logger.info("{}: {}".format(ncc, i_good[0].shape))
        ncc_cut = ncc.copy()
        ncc_cut['c'][i_bad] = 0
        p = ax[0].plot(r[i], ncc['g'][i_ncc], label=ncc)
        ax[0].plot(r[i], ncc_cut['g'][i_ncc], color=p[0].get_color(), linestyle='dashed', alpha=0.5, linewidth=3)
        p = ax[1].plot(np.abs(ncc['c'][i_c_ncc]), label=ncc)
        ax[1].plot(np.abs(ncc_cut['c'][i_c_ncc]), color=p[0].get_color(), linestyle='dashed', alpha=0.5)
    ax[0].set_ylabel('f(r)')
    ax[0].set_xlabel('r')
    ax[1].legend()
    ax[1].set_yscale('log')
    ax[1].set_ylabel('|f(k)|')
    ax[1].set_xlabel('k')
    ax[1].axhline(y=ncc_cutoff, linestyle='dashed', color='xkcd:dark grey', alpha=0.5)
    fig.savefig('star_nccs.pdf')

if __name__=="__main__":
    dlog = logging.getLogger('matplotlib')
    dlog.setLevel(logging.WARNING)

    import matplotlib.pyplot as plt
    nr = 64

    LE = lane_emden(nr, radius=1, dtype=np.float64, verbose=True)
    logger.info('T:   {}--{}'.format(LE['T'](r=0).evaluate()['g'],LE['T'](r=1).evaluate()['g']))
    logger.info('lnρ: {}--{}'.format(LE['lnρ'](r=0).evaluate()['g'], LE['lnρ'](r=1).evaluate()['g']))
    plot_model_star(nr, LE, ncc_cutoff=1e-6)
