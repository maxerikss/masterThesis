import numpy as np
import qutip as qt
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from pathlib import Path
import time

#  Matplotlib setup
plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': '18'})
plt.rc('text.latex', preamble=r'\usepackage{amssymb,amsmath,amsfonts,amsthm}')
plt.rcParams['text.usetex'] = True

## QuTiP operators (these do not depend on the physical parameters)
N = 20  # size of cavity Hilbert space

# Cavity
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


##### Helper functions

def withDerived(params):
    """
    Returns a copy of the parameter dictionary with the derived quantities
    (C, S, Gamma_2s, Gamma_2t) recomputed from the base parameters.
    Called at the start of every public function, so the derived values are
    never out of date, even if you edit params after creating it.
    """
    p = dict(params)
    p['C'] = np.cos(2 * p['theta_q']) * np.cos(2 * p['theta_s'])
    p['S'] = np.sin(2 * p['theta_q']) * np.sin(2 * p['theta_s'])
    p['Gamma_2s'] = p['Gamma_1s'] / 2 + p['Gamma_phis']
    p['Gamma_2t'] = p['Gamma_1t'] / 2 + p['Gamma_phit']
    return p


def makeCollapseOps(params):
    """
    Builds the list of dissipators (collapse operators) from the parameter dictionary.
    """
    return [
        np.sqrt(params['kappa']) * a,
        np.sqrt(params['Gamma_1s']) * sm,
        np.sqrt(params['Gamma_1t']) * tm,
        np.sqrt(params['Gamma_phis'] / 2) * sz,
        np.sqrt(params['Gamma_phit'] / 2) * tz,
    ]


def getDetunings(params, sweepName, value):
    """
    Returns (Delta_r, Delta_q, Delta_s) for the given parameters. If sweepName is
    'omegad' or 'omegaq', `value` replaces omega_d or omega_q respectively.
    For any other sweepName (e.g. 'none') `value` is ignored.
    """
    omega_d = params['omega_d']
    omega_q = params['omega_q']
    if sweepName == 'omegad':
        omega_d = value
    elif sweepName == 'omegaq':
        omega_q = value
    return (params['omega_r'] - omega_d,
            omega_q - omega_d,
            params['omega_s'] - omega_d)


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
        params (dict):
            Parameter dictionary. Uses g, gd, kappac, beta, theta_q, theta_s.
        a, ad, sm, sp, sz, tm, tp, tz:
            Operators. Pass tz=-1 or tz=1 to freeze the outer qubit in the ground/excited state.

    Returns:
        Hamiltonian (qutip.Qobj):
            The Hamiltonian in the rotating frame of the drive frequency omega_d.
    """
    p = withDerived(params)
    g = p['g']
    gd = p['gd']
    kappac = p['kappac']
    beta = p['beta']
    C = p['C']
    S = p['S']

    H_det = Delta_r * ad * a + Delta_q / 2 * sz + g * (ad * sm + a * sp)
    H_sys = Delta_s / 2 * tz
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
            Dictionary of the fixed parameters. Entries for the swept parameter are replaced by the sweep range.
    """
    p = withDerived(params)

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
                f.write(f"{key} = {p[key]}\n")


def calculateSteadyState(params, sweep=['omegad', -5, 5, 250]):
    """
    Calculates the steady state density matrices when considering the outer qubit frozen in either the ground state, tz=-1, or the excited state, tz=1.

    Parameters:
        params (dict):
            Parameter dictionary.
        sweep (list):
            List of the sweep parameters. The first element is a string of what variable is being swept ('omegad' or 'omegaq').
            The second and third are the minimum and maximum values of the sweep, and the fourth element is the resolution.

    Returns:
        density_Ground (list[qutip.Qobj]):
            The steady state density matrices when the outer qubit is frozen in the ground state.
        density_Excited (list[qutip.Qobj]):
            The steady state density matrices when the outer qubit is frozen in the excited state.
        sweepInfo (list):
            A list containing information on the sweep. The first element is a string of the variable being swept. The second element is a numpy array with all the values.
    """
    p = withDerived(params)
    c_ops = makeCollapseOps(p)

    if sweep[0] not in ('omegad', 'omegaq'):
        raise ValueError(f"Unknown sweep type: {sweep[0]}")

    values = np.linspace(sweep[1], sweep[2], int(sweep[3]))
    sweepInfo = [sweep[0], values]

    density_Ground = []
    density_Excited = []

    for value in values:
        Delta_r, Delta_q, Delta_s = getDetunings(p, sweep[0], value)

        H_ground = Hamiltonian(Delta_r, Delta_q, Delta_s, p, tz=-1)
        H_excited = Hamiltonian(Delta_r, Delta_q, Delta_s, p, tz=1)

        density_Ground.append(qt.steadystate(H_ground, c_ops))
        density_Excited.append(qt.steadystate(H_excited, c_ops))

    return density_Ground, density_Excited, sweepInfo


def MEsolve(params, rho0='ground', sweep=['none'], tMax=25, tRes=100,
            e_ops=[a, tz, sz, ad * a]):
    """
    Calculating the full time solution of the density matrix with varying initial conditions.

    Parameters:
        params (dict):
            Parameter dictionary.
        rho0 (string):
            Either 'ground' or 'excited', determines the initial state of the system.
        sweep (list):
            Information on the sweep conditions. The first element determines the type of sweep, if 'none' only the time evolution will be calculated once.
            If the first element is not 'none', the second, third and fourth element will be the minimal value, maximum value and the number of values.
        tMax (float):
            The maximum time the density matrix is calculated with.
        tRes (int):
            The number of timesteps.
        e_ops (list):
            List of the operators that will have their expectation values calculated.

    Returns:
        densityList (list[Qobj]):
            A list of density matrices. If sweep mode is 'none' returns 1D list. If sweep is something else, it will return
            a 2D list with rows being the sweeping parameter, and columns being time.
        expectList (list):
            Expectation values of the chosen operators. For a sweep, each entry has shape (number of sweep values, number of time steps).
        sweepInfo (list):
            Contains information about the sweep. The first element is a string of the sweep mode. The second element is an array
            of the sweep, and the third element is an array of the time sweep.
    """
    p = withDerived(params)
    c_ops = makeCollapseOps(p)

    if rho0 == 'ground':
        initKetState = qt.tensor(qt.basis(N, 0), qt.basis(2, 1), qt.basis(2, 1))  # outer qubit in |1> (ground) eigval -1
    elif rho0 == 'excited':
        initKetState = qt.tensor(qt.basis(N, 0), qt.basis(2, 1), qt.basis(2, 0))  # outer qubit in |0> (excited) eigval 1
    else:
        raise ValueError(f"rho0 must be 'ground' or 'excited', got {rho0}")

    initRhoState = qt.ket2dm(initKetState)

    tList = np.linspace(0, tMax, tRes)

    if sweep[0] == 'none':
        sweepInfo = [sweep[0], np.nan, tList]

        Delta_r, Delta_q, Delta_s = getDetunings(p, 'none', None)
        H = Hamiltonian(Delta_r, Delta_q, Delta_s, p)

        result = qt.mesolve(H, initRhoState, tlist=tList, c_ops=c_ops, e_ops=e_ops)
        densityList = result.states
        expectList = result.expect

    elif sweep[0] in ('omegad', 'omegaq'):
        values = np.linspace(sweep[1], sweep[2], int(sweep[3]))
        sweepInfo = [sweep[0], values, tList]

        densityList = []
        expectList = [[] for _ in range(len(e_ops))]

        for value in values:
            Delta_r, Delta_q, Delta_s = getDetunings(p, sweep[0], value)
            H = Hamiltonian(Delta_r, Delta_q, Delta_s, p)

            result = qt.mesolve(H, initRhoState, tlist=tList, c_ops=c_ops, e_ops=e_ops)
            densityList.append(result.states)
            for i in range(len(e_ops)):
                expectList[i].append(result.expect[i])

        expectList = [np.array(x) for x in expectList]

    else:
        raise ValueError(f"Unknown sweep type: {sweep[0]}")

    return densityList, expectList, sweepInfo


def calculateReflectionCompare(density, params):
    """
    Calculates the reflection coefficient using an analytical, but simplified model, and using a density matrix calculated using
    QuTiP and comparing the results by plotting.

    Parameters:
        density (tuple):
            The output of calculateSteadyState: steady state density matrices with the outer qubit frozen in the ground state,
            with it frozen in the excited state, and the sweep information.
        params (dict):
            Parameter dictionary.
    """
    p = withDerived(params)
    beta = p['beta']
    kappac = p['kappac']
    kappa = p['kappa']
    g = p['g']
    gd = p['gd']
    C = p['C']
    S = p['S']
    Gamma_phis = p['Gamma_phis']

    density_Ground = density[0]
    density_Excited = density[1]
    sweepInfo = density[2]

    alpha_ground = np.array([qt.expect(a, rho) for rho in density_Ground])
    alpha_excited = np.array([qt.expect(a, rho) for rho in density_Excited])

    r_groundValues = (beta + np.sqrt(kappac) * alpha_ground) / beta
    r_excitedValues = (beta + np.sqrt(kappac) * alpha_excited) / beta

    r_groundAnalyticalValues = []
    r_excitedAnalyticalValues = []

    for value in sweepInfo[1]:
        Delta_r, Delta_q, _ = getDetunings(p, sweepInfo[0], value)
        r_groundAnalyticalValues.append(1 - kappac / (kappa / 2 + 1j * Delta_r + g**2 / (Gamma_phis + 1j * (Delta_q - 2 * gd * C))))
        r_excitedAnalyticalValues.append(1 - kappac / (kappa / 2 + 1j * Delta_r + g**2 / (Gamma_phis + 1j * (Delta_q + 2 * gd * C))))

    X = p['omega_r'] - sweepInfo[1]
    if sweepInfo[0] == 'omegad':
        Xlabel = r'$\Delta_r = \omega_r - \omega_d$'
    else:
        Xlabel = r'$\omega_r - \omega_q$'

    output_dir = Path("calculateReflectionCompare")
    output_dir.mkdir(parents=True, exist_ok=True)

    writeParameters(output_dir / "parameters.txt",
                    [sweepInfo[0], np.min(sweepInfo[1]), np.max(sweepInfo[1])], p)

    cases = [
        (r_groundValues, r_groundAnalyticalValues, -1, "tauGround.pdf"),
        (r_excitedValues, r_excitedAnalyticalValues, 1, "tauExcited.pdf"),
    ]

    for rQutip, rAnalytical, tauz, fileName in cases:
        fig, ax1 = plt.subplots(1, 1)

        ax1.plot(X, np.abs(rQutip), color='blue', label='QuTiP')
        ax1.plot(X, np.abs(rAnalytical), color='red', label='Analytical')
        ax1.set_xlabel(Xlabel)
        ax1.set_ylabel(r'$|r|$')
        ax1.set_ylim(-0.05, 1.05)

        ax1.set_title(rf'$\Gamma_1^\tau = {p["Gamma_1t"]}, S \approx {round(S, 2)}, C \approx {round(C, 2)}, \tau_z = {tauz}$')

        ax1.legend()

        fig.tight_layout()
        fig.savefig(output_dir / fileName)
        plt.close(fig)


def calculateMeasurementRate(density, params):
    """
    Calculates and plots the measurement rate Gamma_m from the steady states with the outer qubit frozen in the ground and excited state.

    Parameters:
        density (tuple):
            The output of calculateSteadyState.
        params (dict):
            Parameter dictionary.
    """
    p = withDerived(params)

    density_Ground = density[0]
    density_Excited = density[1]
    sweepInfo = density[2]

    alpha_groundValues = np.array([qt.expect(a, rho) for rho in density_Ground])
    alpha_excitedValues = np.array([qt.expect(a, rho) for rho in density_Excited])

    Gamma_m = p['kappa'] / 2 * np.abs(alpha_groundValues - alpha_excitedValues)**2

    output_dir = Path("calculateMeasurementRate")
    output_dir.mkdir(parents=True, exist_ok=True)

    writeParameters(output_dir / "parameters.txt",
                    [sweepInfo[0], np.min(sweepInfo[1]), np.max(sweepInfo[1])], p)

    X = p['omega_r'] - sweepInfo[1]
    if sweepInfo[0] == 'omegad':
        Xlabel = r'$\Delta_r = \omega_r - \omega_d$'
    else:
        Xlabel = r'$\omega_r - \omega_q$'

    fig, ax1 = plt.subplots(1, 1)

    ax1.plot(X, Gamma_m, color='blue')
    ax1.set_xlabel(Xlabel)
    ax1.set_ylabel(r'$\Gamma_m$')

    fig.tight_layout()
    fig.savefig(output_dir / "Gamma_m.pdf")
    plt.close(fig)


def calculateFullPlot(params, rho0='ground', sweep=['none', np.nan, np.nan, np.nan], tMax=25, tRes=100,
                      e_ops=[a, tz, sz, ad * a]):
    """
    Solves the master equation and plots |r|, <tau_z>, <sigma_z> and <a^dagger a>, either as a function of time
    (sweep = ['none']) or as colour plots of time against the swept parameter.

    Parameters:
        params (dict):
            Parameter dictionary.
        rho0 (string):
            Either 'ground' or 'excited', determines the initial state of the outer qubit.
        sweep (list):
            ['none'], or ['omegad' / 'omegaq', minimum, maximum, number of values].
        tMax (float):
            The maximum time.
        tRes (int):
            The number of timesteps.
        e_ops (list):
            Operators for the expectation values. The panels assume the order [a, tz, sz, ad*a].
    """
    p = withDerived(params)
    kappac = p['kappac']
    beta = p['beta']

    densityList, expectList, sweepInfo = MEsolve(p, rho0=rho0, sweep=sweep, tMax=tMax, tRes=tRes, e_ops=e_ops)

    aList = expectList[0]
    rList = 1 + np.sqrt(kappac) * aList / beta
    tList = sweepInfo[2]

    output_dir = Path("calculateFullPlot")
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2)
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

    if sweepInfo[0] == 'none':
        fig.set_size_inches(10, 10)

        for ax, data, label in panels:
            ax.plot(tList, data, color='blue')
            ax.set_xlabel(r'Time')
            ax.set_ylabel(label)

        figFileName = rho0 + "TimeEvolve.pdf"
        parameterFileName = rho0 + "NoneParameters.txt"

    else:
        fig.set_size_inches(15, 10)

        Y = p['omega_r'] - sweepInfo[1]
        if sweepInfo[0] == 'omegad':
            yLabel = r"$\Delta_r$"
            fileTag = "Omegad"
        else:
            yLabel = r"$\omega_r - \omega_q$"
            fileTag = "Omegaq"

        for ax, data, label in panels:
            # data must have shape (len(Y), len(tList))
            mesh = ax.pcolormesh(tList, Y, data, cmap="viridis", shading="auto", rasterized=True)
            cbar = fig.colorbar(mesh, ax=ax, label=label)
            cbar.solids.set_rasterized(True)
            ax.set_xlabel("Time")
            ax.set_ylabel(yLabel)

        figFileName = rho0 + fileTag + "SweepTimeEvolve.pdf"
        parameterFileName = rho0 + fileTag + "Parameters.txt"

    fig.tight_layout()
    fig.savefig(output_dir / figFileName, dpi=300)  # dpi sets the resolution of the rasterized meshes
    plt.close(fig)

    writeParameters(output_dir / parameterFileName,
                    [sweepInfo[0], np.min(sweepInfo[1]), np.max(sweepInfo[1])], p)

def expectationM(tau, expectA, phi, params):
    sqrt_kappac = np.sqrt(params['kappac'])

    aVals = np.asarray(expectA[0])
    tVals = np.asarray(expectA[1])
    tau = np.atleast_1d(tau).astype(float)

    # bounds check
    if np.any(tau < tVals[0]) or np.any(tau > tVals[-1]):
        raise ValueError(
            f"tau must lie within the simulated time range "
            f"[{tVals[0]}, {tVals[-1]}], but got values from "
            f"{tau.min()} to {tau.max()}"
        )

    # integrand on the full grid
    X = 2 * np.real(np.exp(-1j * phi) * aVals)

    # cumulative trapezoid integral: C[i] = integral from tVals[0] to tVals[i]
    C = np.concatenate(([0.0], np.cumsum(0.5 * (X[1:] + X[:-1]) * np.diff(tVals))))

    # index of the last grid point at or before each tau
    i = np.searchsorted(tVals, tau, side='right') - 1

    # add the partial interval from tVals[i] to tau (X linearly interpolated)
    Xtau = np.interp(tau, tVals, X)
    result = C[i] + 0.5 * (X[i] + Xtau) * (tau - tVals[i])

    return sqrt_kappac * result

def CalculateC(params, tList, rho0='ground', sweep='none'):
    p = withDerived(params)
    c_ops = makeCollapseOps(p)

    if rho0 == 'ground':
        initKetState = qt.tensor(qt.basis(N, 0), qt.basis(2, 1), qt.basis(2, 1))  # outer qubit in |1> (ground) eigval -1
    elif rho0 == 'excited':
        initKetState = qt.tensor(qt.basis(N, 0), qt.basis(2, 1), qt.basis(2, 0))  # outer qubit in |0> (excited) eigval 1
    else:
        raise ValueError(f"rho0 must be 'ground' or 'excited', got {rho0}")

    initRhoState = qt.ket2dm(initKetState)

    if sweep == 'none':
        Delta_r, Delta_q, Delta_s = getDetunings(p, sweep, None)

        H = Hamiltonian(Delta_r, Delta_q, Delta_s, p)

        at_atau = qt.correlation_2op_2t(H, initRhoState, tList, tauList, c_ops, a, a)



    

##############################################################################
## Parameters: edit these. Everything below the function definitions uses them.
##############################################################################
# all frequencies are in the RWA
params = {
    "omega_r": 0,    # frequency of the cavity
    "omega_q": 3,    # frequency of the inner qubit
    "omega_s": 3,    # frequency of the outer qubit
    "omega_d": 0,    # frequency of the drive

    "beta": 1,    # coherent drive amplitude

    "g": 2,          # light-matter coupling strength between the cavity and inner qubit
    "gd": 1,         # coupling strength between inner and outer qubit

    "theta_q": 0.1,  # mixing angle of the inner qubit
    "theta_s": 0.1,  # mixing angle of the outer qubit

    "kappac": 0.5,   # cavity coupling rate to the transmission line
    "kappa": 1,      # total cavity decay rate

    "Gamma_1s": 0.1,    # relaxation rate of inner qubit
    "Gamma_phis": 0.2,  # dephasing rate of inner qubit

    "Gamma_1t": 0.0,   # relaxation rate of outer qubit
    "Gamma_phit": 0.2,  # dephasing rate of outer qubit
}
# C, S, Gamma_2s and Gamma_2t are derived automatically (see withDerived),
# and the dissipators are rebuilt from these values inside each function.



## Running calculations
startTime = time.time()

#calculateFullPlot(params, rho0='excited', sweep=['omegad', -10, 10, 100], tMax=15, tRes=60)
#calculateFullPlot(params, rho0='ground', sweep=['omegad', -10, 10, 100], tMax=15, tRes=60)

densityList, expectList, sweepInfo = MEsolve(params, rho0='ground', tMax=30, tRes=200)
groundExpectA = [expectList[0], sweepInfo[2]]

densityList, expectList, sweepInfo = MEsolve(params, rho0='excited', tMax=30, tRes=200)
excitedExpectA = [expectList[0], sweepInfo[2]]

tauList = np.linspace(0, 5, 100)

groundExpectM = expectationM(tauList, groundExpectA, np.pi/4, params)
excitedExpectM = expectationM(tauList, excitedExpectA, np.pi/4, params)

diffExpectM = np.abs(excitedExpectM - groundExpectM)

plt.plot(tauList, groundExpectM, color='blue')
plt.plot(tauList, excitedExpectM, color='red')
plt.plot(tauList, diffExpectM, color='orange')
plt.hlines([0,1], tauList[0], tauList[-1], colors=['k'], linestyles=['--'])
plt.savefig('./test.pdf')


endTime = time.time()
timeDifference = endTime - startTime
minutes = timeDifference // 60
seconds = timeDifference % 60
print(f"Time taken: {int(minutes)} m {int(seconds)} s")

# density = calculateSteadyState(params, sweep=['omegad', -10, 10, 250])
# calculateMeasurementRate(density, params)
# calculateReflectionCompare(density, params)
