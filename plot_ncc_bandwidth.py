import numpy as np
import matplotlib.pyplot as plt
from structure import lane_emden
import dedalus.public as de

import logging
logger = logging.getLogger(__name__)
dlog = logging.getLogger('matplotlib')
dlog.setLevel(logging.WARNING)

ncc_cutoff = 1e-10

LE = lane_emden(64, dtype=np.float64)

T = LE['T']
lnρ = LE['lnρ']
c = LE['problem']['c']
r = LE['r']
b = LE['problem']['b']

d = de.Distributor((c,), dtype=np.float64)

grad = lambda A: de.Gradient(A, c)

lnT = np.log(T).evaluate()
lnT.name='lnT'
grad_lnT = (grad(lnT)).evaluate()
grad_lnT.name='grad_lnT'
ρ = np.exp(lnρ).evaluate()
ρ.name='ρ'
grad_lnρ = (grad(lnρ)).evaluate()
grad_lnρ.name='grad_lnρ'
ρT = (ρ*T).evaluate()
ρT.name='ρT'

Tgrad_lnT = (T*grad(lnT)).evaluate()
Tgrad_lnT.name='Tgrad_lnT'

Tgrad_lnρ = (T*grad(lnρ)).evaluate()
Tgrad_lnρ.name='Tgrad_lnρ'

logger.info("NCC expansions:")
for ncc in [ρ, T, ρT, grad_lnT, grad_lnρ, Tgrad_lnT, Tgrad_lnρ]:
    logger.info("{}: {}".format(ncc, np.where(np.abs(ncc['c']) >= ncc_cutoff)[0].shape))

T_inv = (1/T).evaluate()
T_cond = d.Field(name='T_cond', bases=b.radial_basis)


# the LHS anelastic NCCs are T, grad_lnT, grad_lnρ, ρ_inv


fig, ax = plt.subplots(nrows=2, ncols=2)
ax[0,0].plot(r[0,0,:], T['g'][0,0,:])
for i in range(20):
    T_inv = (1/T).evaluate()
    T_cond = d.Field(name='T_cond', bases=b.radial_basis)
    T_cond['c'][0,0,0:i] = T_inv['c'][0,0,0:i]
    T_pre = d.Field(name='T_pre', bases=b.radial_basis)
    T_pre['g'] = T_cond['g']*T['g'] #(T_cond*T).evaluate()
    ncc = T_pre
    logger.info("i = {}, {}: {}".format(i, ncc, np.where(np.abs(ncc['c']) >= ncc_cutoff)[0].shape))
    ax[0,0].plot(r[0,0,:], T_pre['g'][0,0,:])
ax[0,1].plot(r[0,0,:], ρT['g'][0,0,:])
ax[1,0].plot(r[0,0,:], grad_lnT['g'][2][0,0,:])
ax[1,1].plot(r[0,0,:], grad_lnρ['g'][2][0,0,:])
ax[1,0].plot(r[0,0,:], Tgrad_lnT['g'][2][0,0,:])
ax[1,1].plot(r[0,0,:], Tgrad_lnρ['g'][2][0,0,:])
ax[0,0].set_ylabel('T')
ax[0,1].set_ylabel('ρT')
ax[1,0].set_ylabel('gradlnT')
ax[1,1].set_ylabel('gradlnrho')
plt.tight_layout()
fig.savefig('nccs.pdf')

fig, ax = plt.subplots(nrows=2, ncols=2)
ax[0,0].plot(np.abs(T['c'][0,0,:]))
for i in range(20):
    T_inv = (1/T).evaluate()
    T_cond = d.Field(name='T_cond', bases=b.radial_basis)
    T_cond['c'][0,0,0:i] = T_inv['c'][0,0,0:i]
    T_pre = d.Field(name='T_pre', bases=b.radial_basis)
    T_pre['g'] = T_cond['g']*T['g'] #(T_cond*T).evaluate()
    ncc = T_pre
    logger.info("i = {}, {}: {}".format(i, ncc, np.where(np.abs(ncc['c']) >= ncc_cutoff)[0].shape))
    ax[0,0].plot(np.abs(T_pre['c'][0,0,:]))
ax[0,1].plot(np.abs(ρT['c'][0,0,:]))
ax[1,0].plot(np.abs(grad_lnT['c'][1][0,0,:])) # index 1 is spin 0
ax[1,1].plot(np.abs(grad_lnρ['c'][1][0,0,:])) # index 1 is spin 0
ax[1,0].plot(np.abs(Tgrad_lnT['c'][1][0,0,:])) # index 1 is spin 0
ax[1,1].plot(np.abs(Tgrad_lnρ['c'][1][0,0,:])) # index 1 is spin 0
ax[0,0].set_ylabel('T')
ax[0,1].set_ylabel('ρT')
ax[1,0].set_ylabel('gradlnT')
ax[1,1].set_ylabel('gradlnrho')
for axi in ax:
    for axii in axi:
        axii.axhline(y=ncc_cutoff, linestyle='dashed', color='xkcd:grey')
        axii.set_yscale('log')
        axii.set_xlim(0,20)
plt.tight_layout()
fig.savefig('nccs_coeff.pdf')
