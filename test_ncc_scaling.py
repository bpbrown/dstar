import numpy as np
import matplotlib.pyplot as plt
from structure import lane_emden
import dedalus.public as de

import logging
logger = logging.getLogger(__name__)
dlog = logging.getLogger('matplotlib')
dlog.setLevel(logging.WARNING)

ncc_cutoff = 1e-10

LE = lane_emden(63)

T = LE['T']
lnρ = LE['lnρ']
c = LE['problem']['c']
r = LE['r']

d_exp = lambda A: de.operators.UnaryGridFunction(np.exp, A)
d_log = lambda A: de.operators.UnaryGridFunction(np.log, A)
power = lambda A, B: de.operators.Power(A, B)
grad = lambda A: de.operators.Gradient(A, c)

lnT = d_log(T).evaluate()
T_inv = power(T,-1).evaluate()
grad_lnT = grad(lnT).evaluate()
ρ = d_exp(lnρ).evaluate()
grad_lnρ = grad(lnρ).evaluate()
ρ_inv = d_exp(-lnρ).evaluate()
ρT_inv = (T_inv*ρ_inv).evaluate()

# the LHS anelastic NCCs are T, grad_lnT, grad_lnρ, ρ_inv

fig, ax = plt.subplots(nrows=2, ncols=2)
ax[0,0].plot(r[0,0,:], T['g'][0,0,:])
ax[0,1].plot(r[0,0,:], ρ_inv['g'][0,0,:])
ax[1,0].plot(r[0,0,:], grad_lnT['g'][2][0,0,:])
ax[1,1].plot(r[0,0,:], grad_lnρ['g'][2][0,0,:])
ax[0,0].set_ylabel('T')
ax[0,1].set_ylabel('1/ρ')
ax[1,0].set_ylabel('gradT')
ax[1,1].set_ylabel('gradlnrho')
plt.tight_layout()
fig.savefig('nccs.pdf')

fig, ax = plt.subplots(nrows=2, ncols=2)
ax[0,0].plot(np.abs(T['c'][0,0,:]))
ax[0,1].plot(np.abs(ρ_inv['c'][0,0,:]))
ax[1,0].plot(np.abs(grad_lnT['c'][1][0,0,:])) # index 1 is spin 0
ax[1,1].plot(np.abs(grad_lnρ['c'][1][0,0,:])) # index 1 is spin 0
ax[0,0].set_ylabel('T')
ax[0,1].set_ylabel('1/ρ')
ax[1,0].set_ylabel('gradT')
ax[1,1].set_ylabel('gradlnrho')
for axi in ax:
    for axii in axi:
        axii.axhline(y=ncc_cutoff, linestyle='dashed', color='xkcd:grey')
        axii.set_yscale('log')
plt.tight_layout()
fig.savefig('nccs_coeff.pdf')



T_1 = (1/T).evaluate()
grad_lnT_1 = (1/grad_lnT).evaluate()
grad_lnρ_1 = (1/grad_lnρ).evaluate()
ρ_inv_1 = (1/ρ_inv).evaluate()

fig, ax = plt.subplots(nrows=2, ncols=2)
ax[0,0].plot(r[0,0,:], T_1['g'][0,0,:])
ax[0,1].plot(r[0,0,:], ρ_inv_1['g'][0,0,:])
ax[1,0].plot(r[0,0,:], grad_lnT_1['g'][2][0,0,:])
ax[1,1].plot(r[0,0,:], grad_lnρ_1['g'][2][0,0,:])
ax[0,0].set_ylabel('T')
ax[0,1].set_ylabel('1/ρ')
ax[1,0].set_ylabel('gradT')
ax[1,1].set_ylabel('gradlnrho')
plt.tight_layout()
fig.savefig('nccs_1.pdf')

fig, ax = plt.subplots(nrows=2, ncols=2)
ax[0,0].plot(np.abs(T_1['c'][0,0,:]))
ax[0,1].plot(np.abs(ρ_inv_1['c'][0,0,:]))
ax[1,0].plot(np.abs(grad_lnT_1['c'][1][0,0,:])) # index 1 is spin 0
ax[1,1].plot(np.abs(grad_lnρ_1['c'][1][0,0,:])) # index 1 is spin 0
ax[0,0].set_ylabel('T')
ax[0,1].set_ylabel('1/ρ')
ax[1,0].set_ylabel('gradT')
ax[1,1].set_ylabel('gradlnrho')
for axi in ax:
    for axii in axi:
        axii.axhline(y=ncc_cutoff, linestyle='dashed', color='xkcd:grey')
        axii.set_yscale('log')
plt.tight_layout()
fig.savefig('nccs_1_coeff.pdf')
