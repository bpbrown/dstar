"""
Test obtaining ICs in A from B via LBVP.

Usage:
    c2001_case1.py [options]

Options:
    --Ntheta=<Ntheta>       Latitude coeffs [default: 32]
    --Nr=<Nr>               Radial coeffs  [default: 32]
"""
import logging
logger = logging.getLogger(__name__)
for system in ['h5py', 'matplotlib', 'distributor', 'transforms', 'subsystems']:
     logging.getLogger(system).setLevel(logging.WARNING)

import numpy as np
import dedalus.public as de

def test_IC(Nr, Nθ, radius=1):
    L2_error = lambda A, B: de.integ(de.dot(A-B,A-B)).evaluate()['g'][0,0,0]
    L2_set = {}

    # parameters
    Ri = r_inner = 7/13
    Ro = r_outer = 20/13
    Nφ = 2*Nθ
    dealias = 3/2
    dtype = np.float64

    # Bases
    c = de.SphericalCoordinates('phi', 'theta', 'r')
    d = de.Distributor(c, dtype=np.float64)
    b = de.BallBasis(c, shape=(Nφ,Nθ,Nr), radius=radius, dealias=dealias, dtype=np.float64)
    b_S2 = b.S2_basis()
    bk2 = b.clone_with(k=2)
    bk1 = b.clone_with(k=1)

    # Fields
    A = d.VectorField(c, name="A", bases=b)
    φ = d.Field(name="φ", bases=bk1)
    τ_φ = d.Field(name="τ_φ")
    τ_A = d.VectorField(c, name="τ_A", bases=b_S2)

    # Substitutions
    phi, theta, r = d.local_grids(b)

    div = lambda A: de.Divergence(A, index=0)
    lap = lambda A: de.Laplacian(A, c)
    grad = lambda A: de.Gradient(A, c)
    curl = lambda A: de.Curl(A)

    dot = lambda A, B: de.DotProduct(A, B)
    radial = lambda A: de.RadialComponent(A)
    integ = lambda A: de.Integrate(A, c)

    ell_func = lambda ell: ell+1
    ellp1 = lambda A: de.SphericalEllProduct(A, c, ell_func)

    lift = lambda A, n: de.Lift(A, bk2, n)

    A_analytic = d.VectorField(c, bases=b, name='A_analytic')
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

    print(A_analytic['g'].shape)
    print(A_analytic_0.shape)
    A_analytic['g'][0] = A_analytic_0
    A_analytic['g'][1] = A_analytic_1
    A_analytic['g'][2] = A_analytic_2

    # We want to solve for an initial vector potential A
    # with curl(A) = B0, but you're best actually solving -lap(A)=curl(B)
    # (thanks Jeff Oishi & Calum Skene).  We will do this as a BVP.
    mag_amp = 1

    B_IC = d.VectorField(c, name="B_IC", bases=b)
    B_IC['g'][2] = 0 # radial
    B_IC['g'][1] = -mag_amp*3./2.*r*(-1+4*r**2-6*r**4+3*r**6)*(np.cos(phi)+np.sin(phi))
    B_IC['g'][0] = -mag_amp*3./4.*r*(-1+r**2)*np.cos(theta)*( 3*r*(2-5*r**2+4*r**4)*np.sin(theta) + 2*(1-3*r**2+3*r**4)*(np.cos(phi)-np.sin(phi)))
    J_IC = de.curl(B_IC)

    logger.debug('L2 error between curl(A0) and B_IC: {:.3g}'.format(L2_error(B_IC,curl(A_analytic))))


    IC_problem = de.LBVP([φ, A, τ_φ, τ_A])
    IC_problem.add_equation((div(A) + τ_φ, 0))
    IC_problem.add_equation((-lap(A) + grad(φ) + lift(τ_A, -1), J_IC))
    IC_problem.add_equation((integ(φ), 0))
    IC_problem.add_equation((radial( grad(A)(r=radius))+ellp1(A)(r=radius)/radius, 0))
    IC_solver = IC_problem.build_solver()
    IC_solver.solve()
    logger.info("solved for initial conditions for A")

    logger.debug('L2 error between curl(A) and B_IC: {:.3g}'.format(L2_error(B_IC,curl(A))))

    label = r'-lap(A) = J0'
    B = de.curl(A).evaluate()
    L2_set[label] = L2_error(B_IC, curl(A))
    print(label, L2_error(B_IC, curl(A)), L2_error(A_analytic, A),  integ(div(A)).evaluate()['g'][0,0,0])

    return L2_set

if __name__=='__main__':
    import matplotlib.pyplot as plt

    from docopt import docopt
    args = docopt(__doc__)

    Nr_max = int(args['--Nr'])
    Ntheta_max = int(args['--Ntheta'])

    def log_set(N_min, N_max):
        log2_N_min = int(np.log2(N_min))
        log2_N_max = int(np.log2(N_max))
        return np.logspace(log2_N_min, log2_N_max, base=2, num=(log2_N_max-log2_N_min+1), dtype=int)

    Nr_min = 8
    Nr_set = log_set(Nr_min, Nr_max)
    Ntheta = Nr_min
    L2 = {}
    for i, Nr in enumerate(Nr_set):
        L2_set = test_IC(Nr, Ntheta)
        for label in L2_set:
            if i == 0:
                L2[label] = []
            L2[label].append(L2_set[label])

    fig, ax = plt.subplots(nrows=2)
    linestyles = ['solid', 'solid', 'dashed']

    for label, linestyle in zip(L2, linestyles):
        ax[0].plot(Nr_set, L2[label], label=label, linestyle=linestyle)
    ax[0].set_yscale('log')
    ax[0].set_xscale('log', base=2)
    ax[0].set_xlabel(r'$N_r$')
    ax[0].set_ylabel(r'$L_2(B-B_0)$')
    ax[0].legend(title=r'$N_\theta = '+'{:d}'.format(Ntheta)+r'$')

    Ntheta_set = log_set(8, Ntheta_max)
    L2 = {}
    for i, Ntheta in enumerate(Ntheta_set):
        Nr = Ntheta
        L2_set = test_IC(Nr, Ntheta)
        for label in L2_set:
            if i == 0:
                L2[label] = []
            L2[label].append(L2_set[label])

    for label, linestyle in zip(L2, linestyles):
        ax[1].plot(Ntheta_set, L2[label], label=label, linestyle=linestyle)
    ax[1].set_yscale('log')
    ax[1].set_xscale('log', base=2)
    ax[1].set_xlabel(r'$N_\theta$')
    ax[1].set_ylabel(r'$L_2(B-B_0)$')
    ax[1].legend(title=r'$N_r = N_\theta$')

    fig.tight_layout()

    filename = 'test_IC_Nr{:d}-{:d}_Nt{:d}-{:d}'.format(Nr_set[0], Nr_set[-1], Ntheta_set[0], Ntheta_set[-1])
    fig.savefig(filename+'.pdf')
    fig.savefig(filename+'.png', dpi=300)
