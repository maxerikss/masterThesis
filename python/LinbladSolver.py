import numpy as np
import qutip as qt
from matplotlib import pyplot as plt
from pathlib import Path

#  Matplotlib setup
plt.rc('font',**{'family':'serif','serif':['Computer Modern'], 'size':'18'})
plt.rc('text.latex', preamble=r'\usepackage{amssymb,amsmath,amsfonts,amsthm}')
plt.rcParams['text.usetex'] = True

# Quantities
omega_r = 0
omega_q = 1
omega_s = 8
omega_d = 0

beta = 0.05

g = 0.5
gd = 2

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

def writeParameters(filename, sweep):
    with open(filename, 'w') as f:
        if sweep[0] == 'omegad':
            f.write(f"omega_d = [{sweep[1]}, {sweep[2]}]\n")
        else:
            f.write(f"omega_d = {omega_d}\n")
        if sweep[0] == 'omegaq':
            f.write(f"omega_q = [{sweep[1]}, {sweep[2]}]\n")
        else:
            f.write(f"omega_q = {omega_q}\n")
        f.write(f"omega_r = {omega_r}\n")
        f.write(f"omega_s = {omega_s}\n")
        f.write(f"beta = {beta}\n")
        f.write(f"g = {g}\n")
        f.write(f"gd = {gd}\n")
        f.write(f"theta_q = {theta_q}\n")
        f.write(f"theta_s = {theta_s}\n")
        f.write(f"C = {C}\n")
        f.write(f"S = {S}\n")
        f.write(f"kappac = {kappac}\n")
        f.write(f"kappa = {kappa}\n")
        f.write(f"Gamma_1s = {Gamma_1s}\n")
        f.write(f"Gamma_phis = {Gamma_phis}\n")
        f.write(f"Gamma_2s = {Gamma_2s}\n")
        f.write(f"Gamma_1t = {Gamma_1t}\n")
        f.write(f"Gamma_phit = {Gamma_phit}\n")
        f.write(f"Gamma_2t = {Gamma_2t}\n")

def calculateSteadyState(sweep=['omegad', -5, 5], omega_r=omega_r, omega_q=omega_q, omega_s=omega_s, omega_d=omega_d,
                            a=a, ad=ad,
                            sm=sm, sp=sp, sz=sz,
                            tm=tm, tp=tp, tz=tz,
                            gd=gd, C=C, S=S, kappac=kappac,
                            c_ops=c_ops):

    density_Ground = np.array([])
    density_Excited = np.array([])
    
    if sweep[0] == 'omegad':
        omega_d_values = np.linspace(sweep[1], sweep[2], 250)
        sweepInfo = [sweep, omega_d_values]

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

            density_Ground = np.append(density_Ground, ground_ss)
            density_Excited = np.append(density_Excited, excited_ss)


    elif sweep[0] == 'omegaq':
        omega_q_values = np.linspace(sweep[1], sweep[2], 250)
        sweepInfo = [sweep[0], omega_q_values]

        for omega_q in omega_q_values:
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

            density_Ground = np.append(density_Ground, ground_ss)
            density_Excited = np.append(density_Excited, excited_ss)

    return density_Ground, density_Excited, sweepInfo


def calculateReflectionCompare(density, omega_r=omega_r, omega_q=omega_q, omega_s=omega_s, omega_d=omega_d,
                                a=a, ad=ad,
                                sm=sm, sp=sp, sz=sz,
                                tm=tm, tp=tp, tz=tz,
                                gd=gd, C=C, S=S, kappac=kappac,
                                c_ops=c_ops):
    r_groundValues = []
    r_excitedValues = []

    r_groundAnalyticalValues = []
    r_excitedAnalyticalValues = []

    density_Ground = density[0]
    density_Excited = density[1]
    sweepInfo = density[2]

    alpha_ground = np.array([])
    alpha_excited = np.array([])

    for i in range(density_Ground.size):
        alpha_ground = np.append(alpha_ground, qt.expect(a, density_Ground[i]))
        alpha_excited = np.append(alpha_excited, qt.expect(a, density_Excited[i]))

    betaOut_ground = beta + np.sqrt(kappac) * alpha_ground
    betaOut_excited = beta + np.sqrt(kappac) * alpha_excited

    r_groundValues = (betaOut_ground / beta)
    r_excitedValues = (betaOut_excited / beta)

    if sweepInfo[0] == 'omegad':
        for omega_d in sweepInfo[1]:
            Delta_r = omega_r - omega_d
            Delta_q = omega_q - omega_d
            r_groundAnalyticalValues.append( 1 - kappac / (kappa/2 + 1j * Delta_r + g**2 / (Gamma_phis + 1j * (Delta_q - 2 * gd * C)) ))
            r_excitedAnalyticalValues.append(1 - kappac / (kappa/2 + 1j * Delta_r + g**2 / (Gamma_phis + 1j * (Delta_q + 2 * gd * C)) ))

        X = omega_r - sweepInfo[1] # Delta_r 
        Xlabel = r'$\Delta_r = \omega_r - \omega_d$'

    if sweepInfo[0] == 'omegaq':
        for omega_q in sweepInfo[1]:
            Delta_r = omega_r - omega_d
            Delta_q = omega_q - omega_d
            r_groundAnalyticalValues.append( 1 - kappac / (kappa/2 + 1j * Delta_r + g**2 / (Gamma_phis + 1j * (Delta_q - 2 * gd * C)) ))
            r_excitedAnalyticalValues.append(1 - kappac / (kappa/2 + 1j * Delta_r + g**2 / (Gamma_phis + 1j * (Delta_q + 2 * gd * C)) ))

        X = omega_r - sweepInfo[1] # 
        Xlabel = r'$\omega_r - \omega_q$'


    output_dir = Path("calculateReflectionCompare")
    output_dir.mkdir(parents=True, exist_ok=True)

    writeParameters(output_dir / "parameters.txt", [sweepInfo[0], np.min(sweepInfo[1]), np.max(sweepInfo[1])])

    fig, ax1 = plt.subplots(1,1)

    ax1.plot(X, np.abs(r_groundValues), color='blue', label='QuTiP')
    ax1.plot(X, np.abs(r_groundAnalyticalValues), color='red', label='Analytical')
    ax1.set_xlabel(Xlabel)
    ax1.set_ylabel(r'$|r|$')
    ax1.set_ylim(-0.05, 1.05)

    ax1.set_title(rf'$\Gamma_1^\tau = 0, S \approx {round(S,2)}, C = \approx {round(C,2)}, \tau_z = -1$')

    ax1.legend()

    fig.tight_layout()
    fig.savefig(output_dir / "tauGround.pdf")

    fig, ax1 = plt.subplots(1,1)
    
    ax1.plot(X, np.abs(r_excitedValues), color='blue', label='QuTiP')
    ax1.plot(X, np.abs(r_excitedAnalyticalValues), color='red', label='Analytical')
    ax1.set_xlabel(Xlabel)
    ax1.set_ylabel(r'$|r|$')
    ax1.set_ylim(-0.05, 1.05)

    ax1.set_title(rf'$\Gamma_1^\tau = 0, S \approx {round(S,2)}, C = \approx {round(C,2)}, \tau_z = 1$')

    ax1.legend()

    fig.tight_layout()
    fig.savefig(output_dir / "tauExcited.pdf")
        
def calculateMeasurementRate(density, omega_r=omega_r, omega_q=omega_q, omega_s=omega_s, omega_d=omega_d,
                                a=a, ad=ad,
                                sm=sm, sp=sp, sz=sz,
                                tm=tm, tp=tp, tz=tz,
                                gd=gd, C=C, S=S, kappac=kappac,
                                c_ops=c_ops):
    omega_d_values = np.linspace(-5, 5, 250)
    alpha_groundValues = np.array([])
    alpha_excitedValues = np.array([])

    density_Ground = density[0]
    density_Excited = density[1]

    for i in range(density_Ground.size):
        alpha_ground = qt.expect(a, density_Ground[i])
        alpha_excited = qt.expect(a, density_Excited[i])
        alpha_groundValues = np.append(alpha_groundValues, alpha_ground)
        alpha_excitedValues = np.append(alpha_excitedValues, alpha_excited)

    Gamma_m = kappa/2 * np.abs(alpha_groundValues - alpha_excitedValues)**2
    Delta_r_values = omega_r - omega_d_values

    fig, ax1 = plt.subplots(1,1)

    ax1.plot(Delta_r_values, Gamma_m, color='blue')
    ax1.set_xlabel(r'$\Delta_r = \omega_r - \omega_d$')
    ax1.set_ylabel(r'$\Gamma_m$')

    fig.tight_layout()
    fig.savefig("./Gamma_m.pdf")



density = calculateSteadyState(sweep=['omegaq', -15, 15])
calculateMeasurementRate(density)
calculateReflectionCompare(density)

