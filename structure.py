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


def solve_for_K(Nr, f_in, m=1.5, n_rho=3, radius=1, verbose=False,
               ncc_cutoff = 1e-10, tolerance = 1e-10, dtype=np.float64, comm=None):
    c = de.SphericalCoordinates('phi', 'theta', 'r')
    d = de.Distributor((c,), comm=comm, dtype=dtype)
    b = de.BallBasis(c, (1, 1, Nr), radius=radius, dtype=dtype, dealias=2)
    br = b.radial_basis
    phi, theta, r = b.local_grids()
    # Fields
    K = d.Field(name='K', bases=b)
    f = d.Field(name='f', bases=b)
    τ = d.Field(name='τ', bases=b.S2_basis(radius=radius))
    f['g'] = f_in['g']

    er = d.VectorField(c, name='er')
    er['g'][2] = 1

    # Parameters and operators
    grad = lambda A: de.Gradient(A, c)
    div = lambda A: de.Divergence(A, index=0)
    dot = lambda A, B: de.DotProduct(A, B)

    lift_basis = b.clone_with(k=2) # match laplacian
    lift = lambda A: de.Lift(A, lift_basis, -1)
    problem = de.LBVP([K, τ])
    problem.add_equation((div(K*grad(f))+lift(τ),0))
    problem.add_equation((K(r=radius), 1))
    # problem = de.LBVP([K])
    # problem.add_equation((dot(er, K*grad(f)),-1))

    # Solver
    solver = problem.build_solver(ncc_cutoff=ncc_cutoff)
    solver.solve()
    K.change_scales(1)
    print(K['g'])
    return K


def thermal_equilibrium(Nr, source_in, Υ0_in, θ0_in,
                        γ=5/3, radius=1,
                        tolerance=1e-10, ncc_cutoff=1e-10,
                        dtype=np.float64, comm=None):
    c = de.SphericalCoordinates('phi', 'theta', 'r')
    d = de.Distributor((c,), comm=comm, dtype=dtype)
    b = de.BallBasis(c, (1, 1, Nr), radius=radius, dtype=dtype)
    br = b.radial_basis
    phi, theta, r = b.local_grids()
    # Fields
    θ = d.Field(name='θ', bases=b)
    s = d.Field(name='s', bases=b)
    τ_s = d.Field(name='τ_s', bases=b.S2_basis())

    dot = lambda A, B: de.DotProduct(A, B)
    lap = lambda A: de.Laplacian(A, c)
    grad = lambda A: de.Gradient(A, c)
    lift_basis = b.clone_with(k=2) # match laplacian
    lift = lambda A: de.Lift(A, lift_basis, -1)

    Υ = d.Field(name='Υ', bases=b)
    Υ['g'] = Υ0_in['g']
    θ0 = d.Field(name='θ0', bases=b)
    θ0['g'] = θ0_in['g']
    source = d.Field(name='source', bases=b)
    source['g'] = source_in['g']

    θ['g'] = θ0['g']
    print(θ0['g'])
    print(θ0['g'].shape)
    print(grad(θ).evaluate()['g'].shape)
    print(grad(θ0).evaluate()['g'].shape)
    dot(grad(θ0),grad(θ)).evaluate()

    Pr = 1
    Ek = 1e-4

    equilibrium = de.NLBVP([θ, s, τ_s])
    equilibrium.add_equation((
                          - (lap(θ)+2*dot(grad(θ0),grad(θ)))
                          + lift(τ_s),
                           dot(grad(θ),grad(θ))
                          + Pr/Ek*source ))
    equilibrium.add_equation((γ*s - θ, -(γ-1)*Υ))
    equilibrium.add_equation((s(r=radius), 0))
    solver = equilibrium.build_solver(ncc_cutoff=ncc_cutoff)
    logger.debug('beginning thermal equilbrium NLBVP iterations')
    pert_norm = np.inf
    while pert_norm > tolerance:
        solver.newton_iteration()
        pert_norm = sum(pert.allreduce_data_norm('c', 2) for pert in solver.perturbations)
        logger.debug('Perturbation norm: {:.3g}'.format(pert_norm))

    return s, θ

def lane_emden_h(Nr, m=1.5, n_rho=3, radius=1, verbose=False,
               ncc_cutoff = 1e-10, tolerance = 1e-10, dtype=np.float64, comm=None):
    c = de.SphericalCoordinates('phi', 'theta', 'r')
    d = de.Distributor((c,), comm=comm, dtype=dtype)
    b = de.BallBasis(c, (1, 1, Nr), radius=radius, dtype=dtype)
    br = b.radial_basis
    phi, theta, r = b.local_grids()
    # Fields
    θ = d.Field(name='θ', bases=b)
    φ = d.Field(name='φ', bases=b)
    s = d.Field(name='s', bases=b)
    Υ = d.Field(name='Υ', bases=b)
    γ = 5/3
    R = d.Field(name='R')
    τ = d.Field(name='τ', bases=b.S2_basis(radius=radius))
    τp = d.Field(name='τp')
    # Parameters and operators
    lap = lambda A: de.Laplacian(A, c)
    grad = lambda A: de.Gradient(A, c)
    lift_basis = b.clone_with(k=2) # match laplacian
    lift = lambda A: de.Lift(A, lift_basis, -1)

    problem = de.NLBVP([θ, φ, Υ, s, R, τ, τp])
    problem.add_equation((grad(θ) - grad(s), - np.exp(-θ)*grad(φ)))
    problem.add_equation((lap(φ) + lift(τ) + τp, R**2*np.exp(Υ)))
    problem.add_equation((Υ - m*θ, 0))
    problem.add_equation((γ*s - θ + (γ-1)*Υ, 0))
    problem.add_equation((Υ(r=0), 0))
    problem.add_equation((Υ(r=radius), -n_rho))
    problem.add_equation((φ(r=0), 0))

    # Solver
    solver = problem.build_solver(ncc_cutoff=ncc_cutoff)
    # Initial guess
    θ['g'] = -n_rho/m*r**3 #np.log(np.cos(np.pi/2 * r)**2)
    φ['g'] = -θ['g']
    Υ['g'] = m*θ['g']
    s['g'] = (1/γ)*θ['g'] - (γ-1)/γ*Υ['g']
    R['g'] = 5

    # Iterations
    logger.debug('beginning Lane-Emden NLBVP iterations')
    pert_norm = np.inf
    while pert_norm > tolerance:
        solver.newton_iteration()
        pert_norm = sum(pert.allreduce_data_norm('c', 2) for pert in solver.perturbations)
        if verbose: logger.debug('Perturbation norm: {:.3g}, R={:.3g}'.format(pert_norm,R['g'][0,0,0]))
    logger.debug(f'final Perturbation norm: {pert_norm:.3e}')
    logger.debug('R = {:}'.format(R['g'][0,0,0]))
    T = np.exp(θ).evaluate()
    T.name='T'
    lnρ = Υ.copy()
    lnρ.name='lnρ'

    structure = {'T':T,'lnρ':lnρ}
    for key in structure:
        structure[key].change_scales(1)
    structure['r'] = r
    structure['problem'] = {'c':c, 'b':b, 'problem':problem}
    return structure

def plot_model_star(Nr, structure, γ=5/3, dtype=np.float64, ncc_cutoff=1e-10):
    c = de.SphericalCoordinates('phi', 'theta', 'r')
    d = de.Distributor((c,), dtype=dtype)
    b = de.BallBasis(c, (1, 1, Nr), radius=1, dtype=dtype)
    phi, theta, r = b.local_grids()
    grad = lambda A: de.Gradient(A, c)
    lap = lambda A: de.Laplacian(A, c)
    dot = lambda A, B: de.DotProduct(A, B)
    div = lambda A: de.Divergence(A, index=0)

    integ = lambda A: de.Integrate(A, c)

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
        #Q1 = σ**-2/(Q0_over_Q1 + 1) # normalize to σ**-2 at r=0
        Q1 = 1/(Q0_over_Q1 + 1) # normalize to σ**-2 at r=0
        logger.info("Source function: Q0/Q1 = {:.3g}, σ = {:.3g}, Q1 = {:.3g}".format(Q0_over_Q1, σ, Q1))
        return (Q0_over_Q1*np.exp(-r**2/(2*σ**2)) + 1)*Q1

    source_func = d.Field(name='S', bases=b)
    source_func['g'] = source_function(r)
    print(integ(source_func).evaluate()['g'])
    # for RHS source function, need θ0 on the full ball grid (rather than just the radial grid)
    θ0_RHS = d.Field(name='θ0_RHS', bases=b)
    θ0.change_scales(1)
    θ0_RHS.require_grid_space()
    if θ0['g'].size > 0:
        θ0_RHS['g'] = θ0['g']
    ε = 1e-1 #1e-2
    source = (ε*ρ0/h0*source_func).evaluate()
    source_g = de.Grid(source).evaluate()
    source2 = ((ε*ρ0/h0*source_func) + lap(θ0_RHS) + dot(grad(θ0_RHS),grad(θ0_RHS))).evaluate()
    source2_g = de.Grid(source2).evaluate()
    source.name='source'
    print(integ(source).evaluate()['g'])
    s0 = θ0 - (γ-1)*Υ0
    logger.info("grad(s0): {:.3g}/{:.3g}".format(np.min(grad(s0).evaluate()['g']),np.max(grad(s0).evaluate()['g'])))
    logger.info("     s0 : {:.3g}/{:.3g}".format(np.min(s0.evaluate()['g']),np.max(s0.evaluate()['g'])))

    s_eq, θ_eq = thermal_equilibrium(Nr, source2, Υ0, θ0)
    fig, ax = plt.subplots()
    i = (0, 0, slice(None))
    ax.plot(r[i], s_eq['g'][i])
    ax2 = ax.twinx()
    ax.plot(r[i], θ_eq['g'][i])
    fig.savefig('star_entropy.pdf')

    fig, ax = plt.subplots()
    i = (0, 0, slice(None))
    ax.plot(r[i], source_g['g'][i], linestyle='dashed')
    ax.plot(r[i], source2['g'][i])
    ax.plot(r[i], (lap(θ0_RHS) + dot(grad(θ0_RHS),grad(θ0_RHS))).evaluate()['g'][i])
    ax.plot(r[i], (1/T*lap(T)).evaluate()['g'][i], linestyle='dashed')
    fig.savefig('star_source.pdf')


    er = d.VectorField(c, name='er')
    er['g'][2] = 1

    fig, ax = plt.subplots()
    ax.plot(r[i], div(grad(T)).evaluate()['g'][i])
    ax.plot(r[i], (lap(θ0_RHS) + dot(grad(θ0_RHS),grad(θ0_RHS))).evaluate()['g'][i])
    ax2 = ax.twinx()
    ax2.plot(r[i], grad(T).evaluate()['g'][2][i], linestyle='dashed')
    K_vec = (1/grad(T)).evaluate()
    K_vec['g'][1][:] = 0
    K_vec['g'][0][:] = 0
    Kg = dot(er, K_vec).evaluate()
    ax2.plot(r[i], (Kg*grad(T)).evaluate()['g'][2][i], linestyle='dashed')
    ax2.plot(r[i], (div(Kg*grad(T))).evaluate()['g'][i], linestyle='dotted')
    fig.savefig('star_F.pdf')


    # K = solve_for_K(Nr, T)
    fig, ax = plt.subplots()
    # ax.plot(r[i], K['g'][i])
    ax.plot(r[i], Kg['g'][i])
    fig.savefig('star_K.pdf')

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

    LE = lane_emden(nr, radius=1, dtype=np.float64, verbose=True, tolerance=1e-6)
    logger.info('T:   {}--{}'.format(LE['T'](r=0).evaluate()['g'],LE['T'](r=1).evaluate()['g']))
    logger.info('lnρ: {}--{}'.format(LE['lnρ'](r=0).evaluate()['g'], LE['lnρ'](r=1).evaluate()['g']))
    plot_model_star(nr, LE, ncc_cutoff=1e-6)
