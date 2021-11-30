import numpy as np
import matplotlib.pyplot as plt
from structure import lane_emden
import dedalus.public as de

import logging
logger = logging.getLogger(__name__)
dlog = logging.getLogger('matplotlib')
dlog.setLevel(logging.WARNING)

ncc_cutoff = 1e-10

LE = lane_emden(64)

T = LE['T']
lnρ = LE['lnρ']
c = LE['problem']['c']
r = LE['r']

grad = lambda A: de.Gradient(A, c)

lnT = np.log(T).evaluate()
lnT.name='lnT'
grad_lnT = grad(lnT).evaluate()
grad_lnT.name='grad_lnT'
ρ = np.exp(lnρ).evaluate()
ρ.name='ρ'
grad_lnρ = grad(lnρ).evaluate()
grad_lnρ.name='grad_lnρ'
ρT = (ρ*T).evaluate()
ρT.name='ρT'

# the LHS anelastic NCCs are T, grad_lnT, grad_lnρ, ρ_inv

fig, ax = plt.subplots(nrows=2, ncols=2)
ax[0,0].plot(r[0,0,:], T['g'][0,0,:])
ax[0,1].plot(r[0,0,:], ρT['g'][0,0,:])
ax[1,0].plot(r[0,0,:], grad_lnT['g'][2][0,0,:])
ax[1,1].plot(r[0,0,:], grad_lnρ['g'][2][0,0,:])
ax[0,0].set_ylabel('T')
ax[0,1].set_ylabel('ρT')
ax[1,0].set_ylabel('gradT')
ax[1,1].set_ylabel('gradlnrho')
plt.tight_layout()
fig.savefig('nccs.pdf')

fig, ax = plt.subplots(nrows=2, ncols=2)
ax[0,0].plot(np.abs(T['c'][0,0,:]))
ax[0,1].plot(np.abs(ρT['c'][0,0,:]))
ax[1,0].plot(np.abs(grad_lnT['c'][1][0,0,:])) # index 1 is spin 0
ax[1,1].plot(np.abs(grad_lnρ['c'][1][0,0,:])) # index 1 is spin 0
ax[0,0].set_ylabel('T')
ax[0,1].set_ylabel('ρT')
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
ρT_1 = (1/ρT).evaluate()

fig, ax = plt.subplots(nrows=2, ncols=2)
ax[0,0].plot(r[0,0,:], T_1['g'][0,0,:])
ax[0,1].plot(r[0,0,:], ρT_1['g'][0,0,:])
ax[1,0].plot(r[0,0,:], grad_lnT_1['g'][2][0,0,:])
ax[1,1].plot(r[0,0,:], grad_lnρ_1['g'][2][0,0,:])
ax[0,0].set_ylabel('T')
ax[0,1].set_ylabel('ρT')
ax[1,0].set_ylabel('gradT')
ax[1,1].set_ylabel('gradlnrho')
plt.tight_layout()
fig.savefig('nccs_1.pdf')

fig, ax = plt.subplots(nrows=2, ncols=2)
ax[0,0].plot(np.abs(T_1['c'][0,0,:]))
ax[0,1].plot(np.abs(ρT_1['c'][0,0,:]))
ax[1,0].plot(np.abs(grad_lnT_1['c'][1][0,0,:])) # index 1 is spin 0
ax[1,1].plot(np.abs(grad_lnρ_1['c'][1][0,0,:])) # index 1 is spin 0
ax[0,0].set_ylabel('T')
ax[0,1].set_ylabel('ρT')
ax[1,0].set_ylabel('gradT')
ax[1,1].set_ylabel('gradlnrho')
for axi in ax:
    for axii in axi:
        axii.axhline(y=ncc_cutoff, linestyle='dashed', color='xkcd:grey')
        axii.set_yscale('log')
plt.tight_layout()
fig.savefig('nccs_1_coeff.pdf')
