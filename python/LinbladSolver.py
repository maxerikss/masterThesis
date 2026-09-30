import numpy as np
import qutip as qt
from matplotlib import pyplot as plt


# Quantities
omega_r = 0
omega_q = 1
omega_s = 3

beta = 0.05

g = 1
gd = 1

theta_q = 0.1
theta_s = 0.5
C = np.cos(2*theta_q) * np.cos(2*theta_s)
S = np.sin(2*theta_q) * np.sin(2*theta_s)

kappac = 0.5
kappa = 1
Gamma_1s = 0.1
Gamma_phis = 0.2
Gamma_2s = Gamma_1s/2 + Gamma_phis
Gamma_1t = 0.0
Gamma_phit = 0.2
Gamma_2t = Gamma_1t/2 + Gamma_phit

## Qutip stuff
# cavity
N = 5

a = qt.tensor(qt.destroy(N), qt.qeye(2), qt.qeye(2))
ad = a.dag()

# Inner qubit
sm = qt.tensor(qt.qeye(N), qt.sigmam(), qt.qeye(2))
sp = sm.dag()
sz = qt.tensor(qt.qeye(N), qt.sigmaz(), qt.qeye(2))

# Outer qubit
tm = qt.tensor(qt.qeye(N), qt.qeye(2), qt.sigmam())
tp = tm.dag()
tz = qt.tensor(qt.qeye(N), qt.qeye(2), qt.sigmaz())

# Dissipators
c_ops = [
    np.sqrt(kappa) * a,
    np.sqrt(Gamma_1s) * sm,
    np.sqrt(Gamma_1t) * tm,
    np.sqrt(Gamma_phis/2) * sz,
    np.sqrt(Gamma_phit/2) * tz 
]

# Initial state
initKetState = qt.tensor(qt.basis(N,0), qt.basis(2,0), qt.basis(2,0)) # system qubit in |0> (excited) eigval 1
#initKetState = qt.tensor(qt.basis(N,0), qt.basis(2,0), qt.basis(2,1)) # system qubit in |1> (ground) eigval -1
initRhoState = qt.ket2dm(initKetState)

# Hamiltonian
def Hamiltonian(Delta_r, Delta_q, Delta_s,
                a, ad,
                sm, sp, sz,
                tm, tp, tz,
                gd, C, S, kappac):
    H_det = Delta_r * ad * a + Delta_q/2 * sz + g * (ad * sm + a * sp)
    H_sys = Delta_s/2 * tz
    H_int = gd * (C * tz * sz + S * (tp * sm + tm * sp))
    H_drive = 1j * np.sqrt(kappac) * beta * (a - ad)
    return H_det + H_sys + H_int + H_drive

"""
omega_d_values = np.linspace(-5, 5, 250)
r_values = []
n_values = []
tz_values = []

Delta_r = omega_r - omega_d_values
Delta_q = omega_q - omega_d_values

rAnalyticalGround = 1 - kappac / (kappa/2 + 1j * Delta_r + g**2 / (Gamma_phis + 1j * (Delta_q - 2 * gd * C)) )
rAnalyticalExcited = 1 - kappac / (kappa/2 + 1j * Delta_r + g**2 / (Gamma_phis + 1j * (Delta_q + 2 * gd * C)) )

for omega_d in omega_d_values:
    Delta_r = omega_r - omega_d
    Delta_q = omega_q - omega_d
    Delta_s = omega_s - omega_d

    H = Hamiltonian(Delta_r, Delta_q, Delta_s,
                    a, ad,
                    sm, sp, sz,
                    tm, tp, -1,
                    gd, C, S, kappac)

    density_ss = qt.steadystate(H, c_ops)

    alpha = qt.expect(a, density_ss)
    n = qt.expect(ad*a, density_ss)
    tInv = qt.expect(tz, density_ss)

    betaOut = beta + np.sqrt(kappac) * alpha

    r_values.append(betaOut / beta)
    n_values.append(n)
    tz_values.append(tInv)
    
print(np.max(n_values))

Delta_r_values = omega_r - omega_d_values
fig, ax1 = plt.subplots(1,1)

ax1.plot(Delta_r_values, np.abs(r_values), color='blue')
ax1.plot(Delta_r_values, np.abs(rAnalyticalGround), color='red')
ax1.set_xlabel(r'$\Delta_r = \omega_r - \omega_d$')
ax1.set_ylabel(r'$|r|$')
ax1.set_ylim(-0.05, 1.05)

#ax2 = ax1.twinx()
#ax2.plot(Delta_r_values, tz_values, color='red')
#ax2.set_ylim(-1.05, 1.05)

plt.savefig("./tauGround.pdf")
"""

def calculateReflectionCompare(omega_r=omega_r, omega_q=omega_q, omega_s=omega_s,
                                a=a, ad=ad,
                                sm=sm, sp=sp, sz=sz,
                                tm=tm, tp=tp, tz=tz,
                                gd=gd, C=C, S=S, kappac=kappac,
                                c_ops=c_ops):
    omega_d_values = np.linspace(-5, 5, 250)
    r_groundValues = []
    r_excitedValues = []

    r_groundAnalyticalValues = []
    r_excitedAnalyticalValues = []

    for omega_d in omega_d_values:
        Delta_r = omega_r - omega_d
        Delta_q = omega_q - omega_d
        Delta_s = omega_s - omega_d

        H_ground = Hamiltonian(Delta_r, Delta_q, Delta_s,
                    a, ad,
                    sm, sp, sz,
                    tm, tp, -1,
                    gd, C, S, kappac)
        H_excited = Hamiltonian(Delta_r, Delta_q, Delta_s,
                    a, ad,
                    sm, sp, sz,
                    tm, tp, 1,
                    gd, C, S, kappac)

        ground_ss = qt.steadystate(H_ground, c_ops)
        excited_ss = qt.steadystate(H_excited, c_ops)

        alpha_ground = qt.expect(a, ground_ss)
        alpha_excited = qt.expect(a, excited_ss)

        betaOut_ground = beta + np.sqrt(kappac) * alpha_ground
        betaOut_excited = beta + np.sqrt(kappac) * alpha_excited

        r_groundValues.append(betaOut_ground / beta)
        r_excitedValues.append(betaOut_excited / beta)

        r_groundAnalyticalValues.append( 1 - kappac / (kappa/2 + 1j * Delta_r + g**2 / (Gamma_phis + 1j * (Delta_q - 2 * gd * C)) ))
        r_excitedAnalyticalValues.append(1 - kappac / (kappa/2 + 1j * Delta_r + g**2 / (Gamma_phis + 1j * (Delta_q + 2 * gd * C)) ))

    Delta_r_values = omega_r - omega_d_values
    fig, ax1 = plt.subplots(1,1)

    ax1.plot(Delta_r_values, np.abs(r_groundValues), color='blue', label='QuTiP')
    ax1.plot(Delta_r_values, np.abs(r_groundAnalyticalValues), color='red', label='Analytical')
    ax1.set_xlabel(r'$\Delta_r = \omega_r - \omega_d$')
    ax1.set_ylabel(r'$|r|$')
    ax1.set_ylim(-0.05, 1.05)

    fig.legend()

    fig.savefig("./tauGround.pdf")

    fig, ax1 = plt.subplots(1,1)
    
    ax1.plot(Delta_r_values, np.abs(r_excitedValues), color='blue', label='QuTiP')
    ax1.plot(Delta_r_values, np.abs(r_excitedAnalyticalValues), color='red', label='Analytical')
    ax1.set_xlabel(r'$\Delta_r = \omega_r - \omega_d$')
    ax1.set_ylabel(r'$|r|$')
    ax1.set_ylim(-0.05, 1.05)

    fig.legend()

    fig.savefig("./tauExcited.pdf")
        

calculateReflectionCompare()