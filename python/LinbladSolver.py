import numpy as np
import qutip as qt
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from pathlib import Path
import time

#  Matplotlib setup
plt.rc('font',**{'family':'serif','serif':['Computer Modern'], 'size':'18'})
plt.rc('text.latex', preamble=r'\usepackage{amssymb,amsmath,amsfonts,amsthm}')
plt.rcParams['text.usetex'] = True

"""
# Quantities
# all frequencies are in the RWA
omega_r = 0 # frequency of the cavity
omega_q = 3 # frequency of the inner qubit
omega_s = 3 # frequency of the outer qubit
omega_d = 0 # frequency of the drive

beta = 0.05 # coherent drive amplitude

g = 2 # light-matter coupling strength between the cavity and inner qubit
gd = 1 # coupling strength between inner and outer qubit

theta_q = 0.1 # mixing angle of the inner qubit
theta_s = 0.5 # mixing angle of the outer qubit
C = np.cos(2*theta_q) * np.cos(2*theta_s)
S = np.sin(2*theta_q) * np.sin(2*theta_s)

kappac = 0.5 # cavity coupling rate to the transmission line
kappa = 1 # total cavity decay rate

Gamma_1s = 0.1 # relaxation rate of inner qubit
Gamma_phis = 0.2 # dephasing rate of inner qubit
Gamma_2s = Gamma_1s/2 + Gamma_phis

Gamma_1t = 0.00 # relaxation rate of outer qubit
Gamma_phit = 0.2 # dephasing rate of outer qubit
Gamma_2t = Gamma_1t/2 + Gamma_phit
"""
params = {
    "omega_r": 0,    # frequency of the cavity
    "omega_q": 3,    # frequency of the inner qubit
    "omega_s": 3,    # frequency of the outer qubit
    "omega_d": 0,    # frequency of the drive

    "beta": 0.05,    # coherent drive amplitude

    "g": 2,          # light-matter coupling strength between the cavity and inner qubit
    "gd": 1,         # coupling strength between inner and outer qubit

    "theta_q": 0.1,  # mixing angle of the inner qubit
    "theta_s": 0.5,  # mixing angle of the outer qubit

    "kappac": 0.5,   # cavity coupling rate to the transmission line
    "kappa": 1,      # total cavity decay rate

    "Gamma_1s": 0.1,    # relaxation rate of inner qubit
    "Gamma_phis": 0.2,  # dephasing rate of inner qubit

    "Gamma_1t": 0.00,   # relaxation rate of outer qubit
    "Gamma_phit": 0.2,  # dephasing rate of outer qubit
}

## Qutip stuff
# cavity
N = 10 # size of cavity Hilbert space

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
    np.sqrt(params['kappa']) * a,
    np.sqrt(params['Gamma_1s']) * sm,
    np.sqrt(params['Gamma_1t']) * tm,
    np.sqrt(params['Gamma_phis']/2) * sz,
    np.sqrt(params['Gamma_phit']/2) * tz 
]

##### Functions

# Hamiltonian
def Hamiltonian(Delta_r, Delta_q, Delta_s, params,
                a=a, ad=ad,
                sm=sm, sp=sp, sz=sz,
                tm=tm, tp=tp, tz=tz):
    """
    Calculates the Hamiltonian in the rotating frame of the drive frequency omega_d.

    Parameters:
        Delta_r (float):
            Detuning of the cavity from the drive frequency, Delta_r = omega_r - omega_d
        Delta_q (float):
            Detuning of the inner qubit from the drive frequency, Delta_q = omega_q - omega_d
        Delta_s (float):
            Detuning of the outer qubit from the drive frequency, Delta_s = omega_s - omega_d
        a (qutip.Qobj):
            Annihilation operator for the cavity
        ad (qutip.Qobj):
            Creation operator for the cavity
        sm (qutip.Qobj):
            Lowering operator for the inner qubit, sigma_-
        sp (qutip.Qobj):
            Raising operator for the inner qubit, sigma_+
        sz (qutip.Qobj):
            Pauli Z operator for the inner qubit, sigma_z
        tm (qutip.Qobj):
            Lowering operator for the outer qubit, tau_-
        tp (qutip.Qobj):
            Raising operator for the outer qubit, tau_+
        tz (qutip.Qobj):
            Pauli Z operator for the outer qubit, tau_z
        g (float):
            Light-matter coupling strength between the cavity and inner qubit
        gd (float):
            Coupling strength between inner and outer qubit
        C (float):
            Cosine term including the mixing angles of the inner and outer qubits, C = cos(2*theta_q) * cos(2*theta_s)
        S (float):
            Sine term including the mixing angles of the inner and outer qubits, S = sin(2*theta_q) * sin(2*theta_s)
        kappac (float):
            Cavity coupling rate to the transmission line
        beta (float):
            Coherent drive amplitude
    
    Returns:
        Hamiltonian (qutip.Qobj):
            The Hamiltonian in the rotating frame of the drive frequency omega_d.
    """
    g = params['g']
    gd = params['gd']
    kappac = params['kappac']
    beta = params['beta']
    theta_q = params['theta_q']
    theta_s = params['theta_s']
    C = np.cos(2*theta_q) * np.cos(2*theta_s)
    S = np.sin(2*theta_q) * np.sin(2*theta_s)

    H_det = Delta_r * ad * a + Delta_q/2 * sz + g * (ad * sm + a * sp)
    H_sys = Delta_s/2 * tz
    H_int = gd * (C * tz * sz + S * (tp * sm + tm * sp))
    H_drive = 1j * np.sqrt(kappac) * beta * (a - ad)
    return H_det + H_sys + H_int + H_drive

def writeParameters(filename, sweep, params):
    """
    Saves the parameters used in a text file for later reference. The main use is in other functions that generate plots.

    Parameters:
        filename (Path):
            The path to the file where the parameters will be saved.
        sweep (list):
            A list containing the sweep parameter and its range. The first element is a string indicating which parameter is being swept. Second and third elements are the minimum and maximum values of the sweep.
        params (dict):
            Dictionary of the fixed parameters (omega_r, beta, g, ...). Entries for the swept parameter are overridden by the sweep range.
    """
    # Map sweep names to the parameter they replace
    sweptKeys = {"omegad": "omega_d", "omegaq": "omega_q"}
    sweptKey = sweptKeys.get(sweep[0])
    # Order in which parameters are written to the file
    PARAM_ORDER = [
                "omega_d", "omega_q", "omega_r", "omega_s", "beta", "g", "gd",
                "theta_q", "theta_s", "C", "S", "kappac", "kappa",
                "Gamma_1s", "Gamma_phis", "Gamma_2s",
                "Gamma_1t", "Gamma_phit", "Gamma_2t",
                ]

    with open(filename, 'w') as f:
        for key in PARAM_ORDER:
            if key == sweptKey:
                f.write(f"{key} = [{sweep[1]}, {sweep[2]}]\n")
            else:
                f.write(f"{key} = {params[key]}\n")

def calculateSteadyState(params, sweep=['omegad', -5, 5, 250],
                            a=a, ad=ad,
                            sm=sm, sp=sp, sz=sz,
                            tm=tm, tp=tp, tz=tz,
                            c_ops=c_ops):
    """
    Calculates the steady state density matrices when considering the outer qubit frozen in either the ground state, tz=-1, or the excited state, tz=1.

    Parameters:
        sweep (list):
            List of the sweep parameters. The first element is a string of what variable is being swept. 
            The second and third are the maximum and minimum values of the sweep, and the fourth element is the resolution.
        omega_r (float):
            The frequency of the cavity in the rotating frame.
        omega_q (float):
            The frequency of the inner qubit in the rotating frame.
        omega_s (float):
            The frequency of the outer qubit in the rotating frame.
        a (qutip.Qobj):
            Annihilation operator for the cavity
        ad (qutip.Qobj):
            Creation operator for the cavity
        sm (qutip.Qobj):
            Lowering operator for the inner qubit, sigma_-
        sp (qutip.Qobj):
            Raising operator for the inner qubit, sigma_+
        sz (qutip.Qobj):
            Pauli Z operator for the inner qubit, sigma_z
        tm (qutip.Qobj):
            Lowering operator for the outer qubit, tau_-
        tp (qutip.Qobj):
            Raising operator for the outer qubit, tau_+
        tz (qutip.Qobj):
            Pauli Z operator for the outer qubit, tau_z
        g (float):
            Light-matter coupling strength between the cavity and inner qubit
        gd (float):
            Coupling strength between inner and outer qubit
        C (float):
            Cosine term including the mixing angles of the inner and outer qubits, C = cos(2*theta_q) * cos(2*theta_s)
        S (float):
            Sine term including the mixing angles of the inner and outer qubits, S = sin(2*theta_q) * sin(2*theta_s)
        kappac (float):
            Cavity coupling rate to the transmission line
        beta (float):
            Coherent drive amplitude
        c_ops (list):
            List of collapse operators.
    
    returns:
        density_Ground (qutip.Qobj):
            The steady state density matrix when the outer qubit is frozen in the ground state.
        density_Excited (qutip.Qobj):
            The steady state density matrix when the outer qubit is frozen in the excited state
        sweepInfo (list):
            A list containg information on the sweep. The first element is a string of the variable being swept. The second element is a numpy array with all the values.
    """

    density_Ground = np.array([])
    density_Excited = np.array([])
    
    if sweep[0] == 'omegad':
        omega_d_values = np.linspace(sweep[1], sweep[2], sweep[3])
        sweepInfo = [sweep[0], omega_d_values]

        for omega_d in omega_d_values:
            Delta_r = params['omega_r'] - omega_d
            Delta_q = params['omega_q'] - omega_d
            Delta_s = params['omega_s'] - omega_d

            H_ground = Hamiltonian(Delta_r, Delta_q, Delta_s, params, tz=-1)
            H_excited = Hamiltonian(Delta_r, Delta_q, Delta_s, params, tz=1)

            ground_ss = qt.steadystate(H_ground, c_ops)
            excited_ss = qt.steadystate(H_excited, c_ops)

            density_Ground = np.append(density_Ground, ground_ss)
            density_Excited = np.append(density_Excited, excited_ss)


    elif sweep[0] == 'omegaq':
        omega_q_values = np.linspace(sweep[1], sweep[2], sweep[3])
        sweepInfo = [sweep[0], omega_q_values]

        for omega_q in omega_q_values:
            Delta_r = params['omega_r'] - params['omega_d']
            Delta_q = omega_q - params['omega_d']
            Delta_s = params['omega_s'] - params['omega_d']

            H_ground = Hamiltonian(Delta_r, Delta_q, Delta_s, params, tz=-1)
            H_excited = Hamiltonian(Delta_r, Delta_q, Delta_s, params, tz=1)

            ground_ss = qt.steadystate(H_ground, c_ops)
            excited_ss = qt.steadystate(H_excited, c_ops)

            density_Ground = np.append(density_Ground, ground_ss)
            density_Excited = np.append(density_Excited, excited_ss)

    return density_Ground, density_Excited, sweepInfo

def MEsolve(params, rho0='ground', sweep=['none'], tMax=25, tRes=100,
                        a=a, ad=ad,
                        sm=sm, sp=sp, sz=sz,
                        tm=tm, tp=tp, tz=tz,
                        c_ops=c_ops, 
                        e_ops=[a, tz, sz, ad*a]):
    """
    Calculating the full time solution of the density matrix with varying initial conditions.

    Parameters:
        rho0 (string):
            Either 'ground' or 'excited', determines the initial state of the system.
        sweep (list):
            Information on the sweep conditions. The first element determines the type of sweep, if 'none' only the time evolution will be calculated once.
            If the first element is not 'none', the second and third and fourth element will be the minimal value, maximum value and the number of values.
        tMax (float):
            The maximum time the density matrix is calculated with.
        tRes (float):
            The number of timesteps.
        omega_r (float):
            The frequency of the cavity in the rotating frame.
        omega_q (float):
            The frequency of the inner qubit in the rotating frame.
        omega_s (float):
            The frequency of the outer qubit in the rotating frame.
        a (qutip.Qobj):
            Annihilation operator for the cavity
        ad (qutip.Qobj):
            Creation operator for the cavity
        sm (qutip.Qobj):
            Lowering operator for the inner qubit, sigma_-
        sp (qutip.Qobj):
            Raising operator for the inner qubit, sigma_+
        sz (qutip.Qobj):
            Pauli Z operator for the inner qubit, sigma_z
        tm (qutip.Qobj):
            Lowering operator for the outer qubit, tau_-
        tp (qutip.Qobj):
            Raising operator for the outer qubit, tau_+
        tz (qutip.Qobj):
            Pauli Z operator for the outer qubit, tau_z
        g (float):
            Light-matter coupling strength between the cavity and inner qubit
        gd (float):
            Coupling strength between inner and outer qubit
        C (float):
            Cosine term including the mixing angles of the inner and outer qubits, C = cos(2*theta_q) * cos(2*theta_s)
        S (float):
            Sine term including the mixing angles of the inner and outer qubits, S = sin(2*theta_q) * sin(2*theta_s)
        kappac (float):
            Cavity coupling rate to the transmission line
        beta (float):
            Coherent drive amplitude
        c_ops (list):
            List of collapse operators.
        e_ops (list):
            list of the operators that will have their expectation values calculated.
    
    Returns:
        densityList (list[Qobj]):
            A list of density matrices. If sweep mode is 'none' returns 1D list. If sweep is something else, it will return 
            a 2D list with rows being time, and columns being the sweeping parameter
        expectList (list):
            A 2D or 3D list depending on if sweep mode is 'none' with the expectation value of the chosen operators
        sweepInfo (list):
            Contains information about the sweep. The first element is a string of the sweep mode. The second elements is an array
            of the sweep, and the thrird element is an array of the time sweep.
    """
    if rho0 == 'ground':
        initKetState = qt.tensor(qt.basis(N,0), qt.basis(2,1), qt.basis(2,1)) # system qubit in |1> (ground) eigval -1
    elif rho0 == 'excited':
        initKetState = qt.tensor(qt.basis(N,0), qt.basis(2,1), qt.basis(2,0)) # system qubit in |0> (excited) eigval 1

    initRhoState = qt.ket2dm(initKetState)

    tList = np.linspace(0, tMax, tRes)

    if sweep[0] == 'none':
        sweepInfo = [sweep[0], np.nan, tList]

        Delta_r = omega_r - omega_d
        Delta_q = omega_q - omega_d
        Delta_s = omega_s - omega_d

        H = Hamiltonian(Delta_r, Delta_q, Delta_s)

        result = qt.mesolve(H, initRhoState, tlist=tList, c_ops=c_ops, e_ops=e_ops)
        densityList = result.states
        expectList = result.expect

    if sweep[0] == 'omegad':
        omega_d_values = np.linspace(sweep[1], sweep[2], sweep[3])
        sweepInfo = [sweep[0], omega_d_values, tList]

        densityList = []
        expectList = [[] for _ in range(len(e_ops))]

        for omega_d in omega_d_values:
            Delta_r = omega_r - omega_d
            Delta_q = omega_q - omega_d
            Delta_s = omega_s - omega_d

            H = Hamiltonian(Delta_r, Delta_q, Delta_s)

            result = qt.mesolve(H, initRhoState, tlist=tList, c_ops=c_ops, e_ops=e_ops)
            densityList.append(result.states)
            expect = result.expect
            for i in range(len(e_ops)):
                expectList[i].append(expect[i])

        expectList = [np.array(x) for x in expectList]

    if sweep[0] == 'omegaq':
        omega_q_values = np.linspace(sweep[1], sweep[2], sweep[3])
        sweepInfo = [sweep[0], omega_q_values, tList]

        densityList =[]
        expectList = [[] for _ in range(len(e_ops))]

        for omega_q in omega_q_values:
            Delta_r = omega_r - omega_d
            Delta_q = omega_q - omega_d
            Delta_s = omega_s - omega_d

            H = Hamiltonian(Delta_r, Delta_q, Delta_s)

            result = qt.mesolve(H, initRhoState, tlist=tList, c_ops=c_ops, e_ops=e_ops)
            densityList.append(result.states)
            expect = result.expect
            for i in range(len(e_ops)):
                expectList[i].append(expect[i])
           
        expectList = [np.array(x) for x in expectList]

    return densityList, expectList, sweepInfo

def calculateReflectionCompare(density, omega_r=omega_r, omega_q=omega_q, omega_s=omega_s, omega_d=omega_d,
                                a=a, ad=ad,
                                sm=sm, sp=sp, sz=sz,
                                tm=tm, tp=tp, tz=tz,
                                g=g, gd=gd, C=C, S=S, kappac=kappac, beta=beta,
                                c_ops=c_ops):
    """
    Calculates the reflection coefficient using an analytical, but simplified model, and using a density matrix calculated using
    QuTiP and comparing the results by plotting.

    Parameters:
        density (tuple):
            A tuple consisting of the steady state density matrix with the outer qubit frozen in the ground state and a steady state density matrix 
            the outer qubit frozen in the excited state.
        omega_r (float):
            The frequency of the cavity in the rotating frame.
        omega_q (float):
            The frequency of the inner qubit in the rotating frame.
        omega_s (float):
            The frequency of the outer qubit in the rotating frame.
        a (qutip.Qobj):
            Annihilation operator for the cavity
        ad (qutip.Qobj):
            Creation operator for the cavity
        sm (qutip.Qobj):
            Lowering operator for the inner qubit, sigma_-
        sp (qutip.Qobj):
            Raising operator for the inner qubit, sigma_+
        sz (qutip.Qobj):
            Pauli Z operator for the inner qubit, sigma_z
        tm (qutip.Qobj):
            Lowering operator for the outer qubit, tau_-
        tp (qutip.Qobj):
            Raising operator for the outer qubit, tau_+
        tz (qutip.Qobj):
            Pauli Z operator for the outer qubit, tau_z
        g (float):
            Light-matter coupling strength between the cavity and inner qubit
        gd (float):
            Coupling strength between inner and outer qubit
        C (float):
            Cosine term including the mixing angles of the inner and outer qubits, C = cos(2*theta_q) * cos(2*theta_s)
        S (float):
            Sine term including the mixing angles of the inner and outer qubits, S = sin(2*theta_q) * sin(2*theta_s)
        kappac (float):
            Cavity coupling rate to the transmission line
        beta (float):
            Coherent drive amplitude
    """
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
                                g=g, gd=gd, C=C, S=S, kappac=kappac, beta=beta, 
                                c_ops=c_ops):
    omega_d_values = np.linspace(-5, 5, 250)
    alpha_groundValues = np.array([])
    alpha_excitedValues = np.array([])

    density_Ground = density[0]
    density_Excited = density[1]
    sweepInfo = density[2]

    for i in range(density_Ground.size):
        alpha_ground = qt.expect(a, density_Ground[i])
        alpha_excited = qt.expect(a, density_Excited[i])
        alpha_groundValues = np.append(alpha_groundValues, alpha_ground)
        alpha_excitedValues = np.append(alpha_excitedValues, alpha_excited)

    Gamma_m = kappa/2 * np.abs(alpha_groundValues - alpha_excitedValues)**2

    output_dir = Path("calculateMeasurementRate")
    output_dir.mkdir(parents=True, exist_ok=True)

    writeParameters(output_dir / "parameters.txt", [sweepInfo[0], np.min(sweepInfo[1]), np.max(sweepInfo[1])])

    if sweepInfo[0] == 'omegad':
        X = omega_r - sweepInfo[1]
        Xlabel = r'$\Delta_r = \omega_r - \omega_d$'
    elif sweepInfo[0] == 'omegaq':
        X = omega_r - sweepInfo[1]
        Xlabel = r'$\omega_r - \omega_q$'

    fig, ax1 = plt.subplots(1,1)

    ax1.plot(X, Gamma_m, color='blue')
    ax1.set_xlabel(Xlabel)
    ax1.set_ylabel(r'$\Gamma_m$')

    fig.tight_layout()
    fig.savefig(output_dir / "Gamma_m.pdf")

def calculateFullPlot(rho0='ground', sweep=['none', np.nan, np.nan, np.nan], tMax = 25, tRes=100,
                    omega_r=omega_r, omega_q=omega_q, omega_s=omega_s, omega_d=omega_d,
                    a=a, ad=ad,
                    sm=sm, sp=sp, sz=sz,
                    tm=tm, tp=tp, tz=tz,
                    g=g, gd=gd, C=C, S=S, kappac=kappac, beta=beta,
                    c_ops=c_ops, 
                    e_ops=[a, tz, sz, ad*a]):
    """
    Future amazing docstring that will be soooo goood.
    """
    densityList, expectList, sweepInfo = MEsolve(rho0=rho0, sweep=sweep, tMax=tMax, tRes=tRes, e_ops=e_ops)

    if sweepInfo[0] == 'none':
        aList = expectList[0]
        rList = 1 + np.sqrt(kappac) * aList / beta
        tList = sweepInfo[2]

        fig, axes = plt.subplots(2,2)
        fig.set_size_inches(10,10)
        ax00: Axes = axes[0, 0]
        ax01: Axes = axes[0, 1]
        ax10: Axes = axes[1, 0]
        ax11: Axes = axes[1, 1]

        panels = [
                (ax00, np.abs(rList), r"$|r|$"),
                (ax01, np.real(expectList[1]), r"$\langle \tau_z \rangle$"),
                (ax10, np.real(expectList[2]), r"$\langle \sigma_z \rangle$"),
                (ax11, np.real(expectList[3]), r"$\langle a^\dagger a \rangle$"),
                ]

        for ax, data, label in panels:
            ax.plot(tList, data, color='blue')
            ax.set_xlabel(r'Time')
            ax.set_ylabel(r'$|r|$')

        output_dir = Path("calculateFullPlot")
        output_dir.mkdir(parents=True, exist_ok=True)

        fig.tight_layout()
        figFileName = rho0 + "TimeEvolve.pdf"
        fig.savefig(output_dir / figFileName)

        parameterFileName = rho0 + "NoneParameters.txt"

        writeParameters(output_dir / parameterFileName, [sweepInfo[0], np.min(sweepInfo[2]), np.max(sweepInfo[2])])    

    if sweepInfo[0] == 'omegad':
        aList = expectList[0]

        rList = 1 + np.sqrt(kappac) * aList / beta
        tList = sweepInfo[2]
        Delta_r = omega_r - sweepInfo[1] 

        fig, axes = plt.subplots(2,2)
        fig.set_size_inches(15,10)
        ax00: Axes = axes[0, 0]
        ax01: Axes = axes[0, 1]
        ax10: Axes = axes[1, 0]
        ax11: Axes = axes[1, 1]

        panels = [
                (ax00, np.abs(rList), r"$|r|$"),
                (ax01, np.real(expectList[1]), r"$\langle \tau_z \rangle$"),
                (ax10, np.real(expectList[2]), r"$\langle \sigma_z \rangle$"),
                (ax11, np.real(expectList[3]), r"$\langle a^\dagger a \rangle$"),
                ]
        for ax, data, label in panels:
            mesh = ax.pcolormesh(tList, Delta_r, data, cmap="viridis", shading="auto", rasterized=True)
            cbar = fig.colorbar(mesh, ax=ax, label=label)
            cbar.solids.set_rasterized(True)
            ax.set_xlabel("Time")
            ax.set_ylabel(r"$\Delta_r$")

        output_dir = Path("calculateFullPlot")
        output_dir.mkdir(parents=True, exist_ok=True)

        figFileName = rho0 + "OmegadSweepTimeEvolve.pdf"

        fig.tight_layout()
        fig.savefig(output_dir / figFileName)

        parameterFileName = rho0 + "OmegadParameters.txt"

        writeParameters(output_dir / parameterFileName, [sweepInfo[0], np.min(sweepInfo[2]), np.max(sweepInfo[2])])    





## Running calculations
startTime = time.time()



calculateFullPlot(rho0='excited', sweep=['omegad', -10, 10, 100], tMax=15, tRes=60)
calculateFullPlot(rho0='ground', sweep=['omegad', -10, 10, 100], tMax=15, tRes=60)

endTime = time.time()
print(f"Time taken: {round(endTime - startTime, 2)} seconds")

#density = calculateSteadyState(sweep=['omegad', -10, 10, 250])
#calculateMeasurementRate(density)
#calculateReflectionCompare(density)
"""
densityList, expectList, sweepInfo = MEsolve(rho0='excited', sweep=['omegad', -10, 10, 100], tMax=25, tRes=100)
aList = expectList[0]

rList = 1 + np.sqrt(kappac) * aList / beta
tList = sweepInfo[2] # tList
Delta_r = omega_r - sweepInfo[1] # omega_d_values

fig, ax = plt.subplots()

im = ax.pcolormesh(
    tList,
    Delta_r,
    np.abs(rList),
    shading='auto'
    )

fig.tight_layout()

ax.set_xlabel('Time')
ax.set_ylabel(r'$\Delta_r$')
fig.colorbar(im, ax=ax, label=r'$|r|$')

plt.savefig("./test.pdf")
"""


"""
density, expect, sweepInfo = calculateMEsolve(rho0='excited')
tList = sweepInfo[2]

rList = 1 + np.sqrt(kappac) * expect[0] / beta

print(f"Last value of |r|: {np.abs(rList[-1])}")
print(f"Last value of <tz>: {expect[1][-1]}")

fig, ax1 = plt.subplots(1,1)

ax1.plot(tList, np.abs(rList), color='blue')
ax1.plot(tList, expect[1], color='red')
ax1.set_xlabel(r"t")
ax1.set_ylabel(r'$|r|$')

fig.tight_layout()
fig.savefig("./test.pdf")
"""