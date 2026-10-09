import numpy as np
import qutip as qt
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from pathlib import Path
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import expm_multiply
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


##### Signal-to-noise, measurement rate and efficiency
# These implement the recipe of the measurement rate notes for this code's model:
#   inner qubit = detector D, outer qubit = system S, cavity read out through the kappac port.
# j = 0 is S in its ground state (tz = -1), j = 1 is S in its excited state (tz = +1). D starts in its
# ground state and the cavity starts empty, exactly as in MEsolve.
# Only the kappac port is measured, so kappa -> kappac in the output-field prefactors of the notes
# (and kappa -> eta * kappac with a detection efficiency eta), while the Lindbladian keeps the total kappa.

def cumulativeIntegral(y, dt):
    """
    Cumulative trapezoidal integral of y on a uniform grid with spacing dt. The result starts at 0 and has the same length as y.
    """
    return np.concatenate(([0], np.cumsum((y[1:] + y[:-1]) / 2 * dt)))


def sweepAxis(p, sweepName, values):
    """
    Returns the x-axis values and the x-axis label used when plotting a sweep (same convention as the other plotting functions).
    """
    X = p['omega_r'] - values
    if sweepName == 'omegad':
        Xlabel = r'$\Delta_r = \omega_r - \omega_d$'
    else:
        Xlabel = r'$\omega_r - \omega_q$'
    return X, Xlabel


def heisenbergRows(H, c_ops, phi, tList):
    """
    Calculates the rows w e^{L t} for every t in tList, where w is the vectorised form of the cavity quadrature
    X_phi = exp(-i phi) a + exp(i phi) a^dagger, so that Tr{X_phi e^{L t}[sigma]} = (row t) . vec(sigma).
    This is the Heisenberg-picture form of the quantum regression theorem. It means the propagation is done once for
    all source operators, instead of once for every starting time t'.

    Parameters:
        H (qutip.Qobj):
            Hamiltonian in the rotating frame.
        c_ops (list[qutip.Qobj]):
            Collapse operators.
        phi (float):
            Local oscillator phase of the homodyne detection.
        tList (numpy.ndarray):
            Uniform time grid starting at 0.

    Returns:
        rows (numpy.ndarray):
            Array of shape (len(tList), dimension**2). Vectorisation is column stacking, as in qt.operator_to_vector.
    """
    L = qt.liouvillian(H, c_ops)
    LT = csr_matrix(L.full().T)

    Xphi = np.exp(-1j * phi) * a + np.exp(1j * phi) * ad
    w = qt.operator_to_vector(Xphi.dag()).full().conj().ravel()

    return expm_multiply(LT, w, start=tList[0], stop=tList[-1], num=len(tList), endpoint=True)


def excessNoiseIntegral(rows, densityList, alphaList, phi, tList):
    """
    Calculates the double integral of the excess correlation function,
        I(tau) = int_0^tau int_0^tau C(t, t') dt dt',
    with C(t, t') = <T :X_phi(t) X_phi(t'):> - <X_phi(t)> <X_phi(t')>, for every tau in tList. The normally and time ordered
    correlation is obtained from the quantum regression theorem for t > t',
        <T :X(t) X(t'):> = Tr{ X_phi e^{L (t - t')} [ exp(-i phi) a rho(t') + exp(i phi) rho(t') a^dagger ] },
    since the source operator contains all four orderings of Eq. (15) at once.

    Parameters:
        rows (numpy.ndarray):
            The output of heisenbergRows.
        densityList (list[qutip.Qobj]):
            The density matrices rho(t) on the grid tList.
        alphaList (numpy.ndarray):
            The expectation values <a(t)> on the grid tList.
        phi (float):
            Local oscillator phase of the homodyne detection.
        tList (numpy.ndarray):
            Uniform time grid starting at 0.

    Returns:
        I (numpy.ndarray):
            The double integral for every tau in tList.
    """
    n = len(tList)
    dt = tList[1] - tList[0]

    rho = np.array([r.full() for r in densityList])                  # shape (n, d, d)
    sourceHalf = np.exp(-1j * phi) * np.matmul(a.full(), rho)
    source = sourceHalf + np.conj(np.swapaxes(sourceHalf, 1, 2))     # Hermitian source operators, one per t'
    sourceVec = np.swapaxes(source, 1, 2).reshape(n, -1)             # column stacking

    T = np.real(rows @ sourceVec.T)                                  # T[m, k] = <T :X(t_k + m dt) X(t_k):>
    X = 2 * np.real(np.exp(-1j * phi) * alphaList)

    # C[i, k] = C(t_i, t_k) for i >= k
    C = np.zeros((n, n))
    for k in range(n):
        C[k:, k] = T[:n - k, k] - X[k:] * X[k]

    # I(tau) = 2 int_0^tau dt int_0^t dt' C(t, t'), because C is symmetric in t <-> t'
    G = np.zeros(n)
    for i in range(1, n):
        G[i] = dt * (np.sum(C[i, :i + 1]) - 0.5 * (C[i, 0] + C[i, i]))

    return 2 * cumulativeIntegral(G, dt)


def extractMeasurementRate(tList, SNR, fitRange=(0.5, 1)):
    """
    Extracts the measurement rate and related numbers from SNR(tau).

    Parameters:
        tList (numpy.ndarray):
            The values of tau.
        SNR (numpy.ndarray):
            The signal-to-noise ratio at every tau.
        fitRange (tuple):
            Fraction of the total time (start, end) over which SNR^2 is fitted with a straight line. In the QND-like regime
            SNR^2 grows linearly after the ring-up, and the slope is Gamma_m. Choose a window after the ring-up.

    Returns:
        Gamma_m (float):
            Slope of the linear fit of SNR^2 against tau. Close to zero if SNR^2 has saturated (non-QND regime).
        tau_m (float):
            First tau where SNR = 1 (linearly interpolated), np.nan if SNR never reaches 1.
        SNRMax (float):
            The maximum of SNR(tau).
        tSNRMax (float):
            The tau at which the maximum is reached.
    """
    tMax = tList[-1]
    mask = (tList >= fitRange[0] * tMax) & (tList <= fitRange[1] * tMax)
    Gamma_m = np.polyfit(tList[mask], SNR[mask]**2, 1)[0]

    above = np.where(SNR >= 1)[0]
    if len(above) == 0:
        tau_m = np.nan
    else:
        i = above[0]
        tau_m = tList[i - 1] + (1 - SNR[i - 1]) / (SNR[i] - SNR[i - 1]) * (tList[i] - tList[i - 1])

    iMax = np.argmax(SNR)
    return Gamma_m, tau_m, SNR[iMax], tList[iMax]


def calculateSignalNoise(params, eta=1, phi='auto', tMax=25, tRes=100):
    """
    Calculates the integrated homodyne signal, its noise and the signal-to-noise ratio SNR(tau) for distinguishing the outer
    qubit being in its ground state (j = 0) from its excited state (j = 1), following Eqs. (10), (16) and (18) of the notes.
        Delta M(tau) = sqrt(eta kappac) int_0^tau Delta X(t) dt
        sigma_j^2(tau) = tau + eta kappac int int C_j(t, t') dt dt'
        SNR(tau) = |Delta M(tau)| / (sigma_0(tau) + sigma_1(tau))

    Parameters:
        params (dict):
            Parameter dictionary. omega_d and omega_q are used as they are, a sweep is handled by calculateMeasurementSweep.
        eta (float):
            Detection efficiency of everything after the cavity, between 0 and 1.
        phi (float or string):
            Local oscillator phase. 'auto' chooses phi = arg(int Delta alpha dt) over the full time, which maximises |Delta M(tMax)|.
        tMax (float):
            The maximum integration time tau.
        tRes (int):
            The number of timesteps. This sets the resolution of the integrals, so it should be large enough that the
            curves do not change when it is increased.

    Returns:
        result (dict):
            tList (time grid), alpha0 and alpha1 (<a(t)> for j = 0, 1), phi, eta,
            dX (Delta X(t)), dM (Delta M(tau)), I0 and I1 (the double integrals of C_j),
            sigma0 and sigma1 (noise), SNR, and SNRShot (the SNR if there were only shot noise, C_j = 0).
    """
    p = withDerived(params)
    kappac = p['kappac']
    c_ops = makeCollapseOps(p)

    Delta_r, Delta_q, Delta_s = getDetunings(p, 'none', None)
    H = Hamiltonian(Delta_r, Delta_q, Delta_s, p)

    # Signal: the mean cavity field for both states of the outer qubit.
    # e_ops=[] so that mesolve stores the density matrices (they are needed for the noise), <a> is calculated from them.
    density0, _, sweepInfo = MEsolve(p, rho0='ground', sweep=['none'], tMax=tMax, tRes=tRes, e_ops=[])
    density1, _, _ = MEsolve(p, rho0='excited', sweep=['none'], tMax=tMax, tRes=tRes, e_ops=[])

    tList = sweepInfo[2]
    dt = tList[1] - tList[0]
    alpha0 = np.array([qt.expect(a, rho) for rho in density0])
    alpha1 = np.array([qt.expect(a, rho) for rho in density1])

    if phi == 'auto':
        phi = np.angle(np.sum(((alpha1 - alpha0)[1:] + (alpha1 - alpha0)[:-1]) / 2 * dt))

    dX = 2 * np.real(np.exp(-1j * phi) * (alpha1 - alpha0))
    dM = np.sqrt(eta * kappac) * cumulativeIntegral(dX, dt)

    # Noise: shot noise tau plus the excess noise from the normally and time ordered cavity correlations
    rows = heisenbergRows(H, c_ops, phi, tList)
    I0 = excessNoiseIntegral(rows, density0, alpha0, phi, tList)
    I1 = excessNoiseIntegral(rows, density1, alpha1, phi, tList)

    sigma0 = np.sqrt(np.maximum(tList + eta * kappac * I0, 0))
    sigma1 = np.sqrt(np.maximum(tList + eta * kappac * I1, 0))

    # SNR is 0 at tau = 0 (0/0)
    SNR = np.zeros_like(tList)
    SNRShot = np.zeros_like(tList)
    SNR[1:] = np.abs(dM[1:]) / (sigma0[1:] + sigma1[1:])
    SNRShot[1:] = np.abs(dM[1:]) / (2 * np.sqrt(tList[1:]))

    return {'tList': tList, 'alpha0': alpha0, 'alpha1': alpha1, 'phi': phi, 'eta': eta,
            'dX': dX, 'dM': dM, 'I0': I0, 'I1': I1,
            'sigma0': sigma0, 'sigma1': sigma1, 'SNR': SNR, 'SNRShot': SNRShot}


def calculateDephasingRate(params, tMax=25, tRes=100, fitRange=(0.5, 1)):
    """
    Calculates the measurement-induced dephasing rate Gamma_phi of the outer qubit. The outer qubit is prepared in an equal
    superposition of its ground and excited states (D in its ground state, empty cavity), and the decay rate of its coherence
    <tau_-> is obtained from an exponential fit. The intrinsic dephasing Gamma_2t is subtracted, so only the part caused by
    the cavity (and by the coupling to the inner qubit) remains. This is the Gamma_phi that goes with Gamma_m <= 2 Gamma_phi.

    Parameters:
        params (dict):
            Parameter dictionary.
        tMax (float):
            The maximum time. It should be long enough for the coherence to decay visibly.
        tRes (int):
            The number of timesteps.
        fitRange (tuple):
            Fraction of the total time (start, end) over which log|<tau_->| is fitted. Points where the coherence has dropped
            below 1e-8 of its initial value are discarded. The decay is only a single exponential if the outer qubit is not
            flipped (QND-like regime); otherwise the fitted rate is an effective one.

    Returns:
        Gamma_phi (float):
            Measurement-induced dephasing rate, Gamma_total - Gamma_2t.
        Gamma_total (float):
            Total decay rate of the coherence of the outer qubit.
    """
    p = withDerived(params)
    c_ops = makeCollapseOps(p)

    Delta_r, Delta_q, Delta_s = getDetunings(p, 'none', None)
    H = Hamiltonian(Delta_r, Delta_q, Delta_s, p)

    # outer qubit in (|0> + |1>)/sqrt(2), inner qubit in |1> (ground), empty cavity
    initKetState = qt.tensor(qt.basis(N, 0), qt.basis(2, 1), (qt.basis(2, 0) + qt.basis(2, 1)).unit())
    initRhoState = qt.ket2dm(initKetState)

    tList = np.linspace(0, tMax, tRes)
    result = qt.mesolve(H, initRhoState, tlist=tList, c_ops=c_ops, e_ops=[tm])
    coherence = np.abs(result.expect[0])

    mask = (tList >= fitRange[0] * tMax) & (tList <= fitRange[1] * tMax) & (coherence > 1e-8 * coherence[0])
    if np.sum(mask) < 3:
        return np.nan, np.nan

    Gamma_total = -np.polyfit(tList[mask], np.log(coherence[mask]), 1)[0]
    return Gamma_total - p['Gamma_2t'], Gamma_total


def calculateMeasurementSweep(params, sweep=['omegad', -10, 10, 50], eta=1, phi='auto', tMax=25, tRes=100,
                              fitRange=(0.5, 1), dephasing=True, steady=True):
    """
    Calculates the measurement rate Gamma_m, the measurement time tau_m, the maximum SNR and (optionally) the dephasing rate
    for every value of a sweep, by running calculateSignalNoise at every point. This is the expensive step, so the result is
    meant to be passed to calculateMeasurementRatePlot and calculateEfficiencyPlot.

    Parameters:
        params (dict):
            Parameter dictionary.
        sweep (list):
            ['omegad' / 'omegaq', minimum, maximum, number of values].
        eta (float):
            Detection efficiency of everything after the cavity.
        phi (float or string):
            Local oscillator phase, 'auto' optimises it at every point of the sweep.
        tMax (float):
            The maximum integration time tau.
        tRes (int):
            The number of timesteps.
        fitRange (tuple):
            Fraction of the total time over which the slope of SNR^2 is fitted, see extractMeasurementRate.
        dephasing (bool):
            If True, also calculates Gamma_phi with calculateDephasingRate (needed for the efficiency).
        steady (bool):
            If True, also calculates the steady state result Gamma_m = eta kappac |alpha_1 - alpha_0|^2 with the outer qubit frozen
            in its ground and excited state (the check in step 6 of the recipe: it should agree in the dispersive limit).

    Returns:
        data (dict):
            sweepInfo ([name, values]), eta, Gamma_m, tau_m, SNRMax, tSNRMax, phi, Gamma_phi, Gamma_total, Gamma_steady.
            Entries that were not requested are filled with np.nan.
    """
    p = withDerived(params)
    c_ops = makeCollapseOps(p)

    sweptKeys = {'omegad': 'omega_d', 'omegaq': 'omega_q'}
    if sweep[0] not in sweptKeys:
        raise ValueError(f"Unknown sweep type: {sweep[0]}")

    values = np.linspace(sweep[1], sweep[2], int(sweep[3]))
    sweepInfo = [sweep[0], values]

    names = ['Gamma_m', 'tau_m', 'SNRMax', 'tSNRMax', 'phi', 'Gamma_phi', 'Gamma_total', 'Gamma_steady']
    data = {name: np.full(len(values), np.nan) for name in names}

    for i, value in enumerate(values):
        pValue = {**p, sweptKeys[sweep[0]]: value}

        result = calculateSignalNoise(pValue, eta=eta, phi=phi, tMax=tMax, tRes=tRes)
        rate = extractMeasurementRate(result['tList'], result['SNR'], fitRange=fitRange)
        data['Gamma_m'][i], data['tau_m'][i], data['SNRMax'][i], data['tSNRMax'][i] = rate
        data['phi'][i] = result['phi']

        if dephasing:
            data['Gamma_phi'][i], data['Gamma_total'][i] = calculateDephasingRate(pValue, tMax=tMax, tRes=tRes, fitRange=fitRange)

        if steady:
            Delta_r, Delta_q, Delta_s = getDetunings(pValue, 'none', None)
            rhoGround = qt.steadystate(Hamiltonian(Delta_r, Delta_q, Delta_s, pValue, tz=-1), c_ops)
            rhoExcited = qt.steadystate(Hamiltonian(Delta_r, Delta_q, Delta_s, pValue, tz=1), c_ops)
            data['Gamma_steady'][i] = eta * p['kappac'] * np.abs(qt.expect(a, rhoExcited) - qt.expect(a, rhoGround))**2

    data['sweepInfo'] = sweepInfo
    data['eta'] = eta
    return data


def calculateSNRPlot(params, eta=1, phi='auto', tMax=25, tRes=100, fitRange=(0.5, 1)):
    """
    Calculates and plots the signal and noise for a single set of parameters: the signal Delta X(t), the noise sigma_j^2 / tau
    compared with shot noise, SNR(tau) compared with the shot noise limited SNR, and SNR^2(tau) with the linear fit that gives
    Gamma_m. The numbers (Gamma_m, tau_m, maximum SNR) are also written to a text file.

    Parameters:
        params (dict):
            Parameter dictionary.
        eta (float):
            Detection efficiency of everything after the cavity.
        phi (float or string):
            Local oscillator phase, 'auto' maximises |Delta M(tMax)|.
        tMax (float):
            The maximum integration time tau.
        tRes (int):
            The number of timesteps.
        fitRange (tuple):
            Fraction of the total time over which the slope of SNR^2 is fitted, see extractMeasurementRate.
    """
    p = withDerived(params)

    result = calculateSignalNoise(p, eta=eta, phi=phi, tMax=tMax, tRes=tRes)
    tList = result['tList']
    Gamma_m, tau_m, SNRMax, tSNRMax = extractMeasurementRate(tList, result['SNR'], fitRange=fitRange)

    output_dir = Path("calculateSNRPlot")
    output_dir.mkdir(parents=True, exist_ok=True)

    writeParameters(output_dir / "parameters.txt", ['none', np.nan, np.nan], p)
    with open(output_dir / "results.txt", 'w') as f:
        f.write(f"eta = {eta}\n")
        f.write(f"phi = {result['phi']}\n")
        f.write(f"Gamma_m = {Gamma_m}\n")
        f.write(f"tau_m = {tau_m}\n")
        f.write(f"SNRMax = {SNRMax}\n")
        f.write(f"tSNRMax = {tSNRMax}\n")

    fig, axes = plt.subplots(2, 2)
    ax00: Axes = axes[0, 0]
    ax01: Axes = axes[0, 1]
    ax10: Axes = axes[1, 0]
    ax11: Axes = axes[1, 1]
    fig.set_size_inches(10, 10)

    ax00.plot(tList, result['dX'], color='blue')
    ax00.set_xlabel(r'Time')
    ax00.set_ylabel(r'$\Delta X(t)$')

    ax01.plot(tList[1:], result['sigma0'][1:]**2 / tList[1:], color='blue', label=r'$j = 0$')
    ax01.plot(tList[1:], result['sigma1'][1:]**2 / tList[1:], color='red', label=r'$j = 1$')
    ax01.axhline(1, color='black', linestyle='--', label='Shot noise')
    ax01.set_xlabel(r'$\tau$')
    ax01.set_ylabel(r'$\sigma_j^2 / \tau$')
    ax01.legend()

    ax10.plot(tList, result['SNR'], color='blue', label='SNR')
    ax10.plot(tList, result['SNRShot'], color='red', linestyle=':', label='Shot noise only')
    ax10.axhline(1, color='black', linestyle='--')
    ax10.set_xlabel(r'$\tau$')
    ax10.set_ylabel(r'SNR')
    ax10.legend()

    mask = (tList >= fitRange[0] * tMax) & (tList <= fitRange[1] * tMax)
    offset = np.polyfit(tList[mask], result['SNR'][mask]**2, 1)[1]
    ax11.plot(tList, result['SNR']**2, color='blue', label=r'$\mathrm{SNR}^2$')
    ax11.plot(tList[mask], Gamma_m * tList[mask] + offset, color='red', linestyle='--',
              label=rf'$\Gamma_m = {Gamma_m:.3g}$')
    ax11.set_xlabel(r'$\tau$')
    ax11.set_ylabel(r'$\mathrm{SNR}^2$')
    ax11.legend()

    fig.tight_layout()
    fig.savefig(output_dir / "SNR.pdf")
    plt.close(fig)


def calculateMeasurementRatePlot(data, params):
    """
    Plots the measurement rate Gamma_m and the measurement time tau_m obtained from the SNR, as a function of the sweep.
    If the steady state result is present it is shown as a dashed line, it should agree in the QND-like, dispersive regime.

    Parameters:
        data (dict):
            The output of calculateMeasurementSweep.
        params (dict):
            Parameter dictionary.
    """
    p = withDerived(params)
    sweepInfo = data['sweepInfo']

    output_dir = Path("calculateMeasurementRatePlot")
    output_dir.mkdir(parents=True, exist_ok=True)

    writeParameters(output_dir / "parameters.txt",
                    [sweepInfo[0], np.min(sweepInfo[1]), np.max(sweepInfo[1])], p)

    X, Xlabel = sweepAxis(p, sweepInfo[0], sweepInfo[1])

    fig, axes = plt.subplots(1, 2)
    ax1: Axes = axes[0]
    ax2: Axes = axes[1]
    fig.set_size_inches(15, 6)

    ax1.plot(X, data['Gamma_m'], color='blue', label=r'From SNR')
    if not np.all(np.isnan(data['Gamma_steady'])):
        ax1.plot(X, data['Gamma_steady'], color='red', linestyle='--', label=r'$\eta \kappa_c |\alpha_1 - \alpha_0|^2$')
        ax1.legend()
    ax1.set_xlabel(Xlabel)
    ax1.set_ylabel(r'$\Gamma_m$')

    ax2.plot(X, data['tau_m'], color='blue')
    ax2.set_xlabel(Xlabel)
    ax2.set_ylabel(r'$\tau_m$')

    fig.tight_layout()
    fig.savefig(output_dir / "Gamma_m.pdf")
    plt.close(fig)


def calculateEfficiencyPlot(data, params):
    """
    Calculates and plots the measurement efficiency eta_m = Gamma_m / (2 Gamma_phi) as a function of the sweep, together with
    Gamma_m and 2 Gamma_phi. Gamma_phi is the measurement-induced dephasing rate of the outer qubit. The efficiency with the
    total dephasing rate, Gamma_phi + Gamma_2t, is also shown, since intrinsic dephasing lowers the efficiency if it is
    defined that way. Note that Gamma_m was calculated with the detection efficiency stored in data['eta'], so
    for data['eta'] = 1 this is the efficiency of the setup itself (internal cavity loss kappa - kappac, excess noise).

    Parameters:
        data (dict):
            The output of calculateMeasurementSweep, which must have been run with dephasing=True.
        params (dict):
            Parameter dictionary.

    Returns:
        efficiency (numpy.ndarray):
            Gamma_m / (2 Gamma_phi) for every value of the sweep (np.nan where Gamma_phi is not positive).
        efficiencyTotal (numpy.ndarray):
            Gamma_m / (2 (Gamma_phi + Gamma_2t)).
    """
    p = withDerived(params)
    sweepInfo = data['sweepInfo']

    Gamma_m = data['Gamma_m']
    Gamma_phi = data['Gamma_phi']

    if np.all(np.isnan(Gamma_phi)):
        raise ValueError("data has no dephasing rate, run calculateMeasurementSweep with dephasing=True")

    positive = Gamma_phi > 0
    efficiency = np.where(positive, Gamma_m / (2 * np.where(positive, Gamma_phi, 1)), np.nan)
    efficiencyTotal = np.where(positive, Gamma_m / (2 * (np.where(positive, Gamma_phi, 1) + p['Gamma_2t'])), np.nan)

    output_dir = Path("calculateEfficiencyPlot")
    output_dir.mkdir(parents=True, exist_ok=True)

    writeParameters(output_dir / "parameters.txt",
                    [sweepInfo[0], np.min(sweepInfo[1]), np.max(sweepInfo[1])], p)

    X, Xlabel = sweepAxis(p, sweepInfo[0], sweepInfo[1])

    fig, axes = plt.subplots(1, 2)
    ax1: Axes = axes[0]
    ax2: Axes = axes[1]
    fig.set_size_inches(15, 6)

    ax1.plot(X, Gamma_m, color='blue', label=r'$\Gamma_m$')
    ax1.plot(X, 2 * Gamma_phi, color='red', linestyle='--', label=r'$2 \Gamma_\varphi$')
    ax1.set_xlabel(Xlabel)
    ax1.set_ylabel(r'Rate')
    ax1.legend()

    ax2.plot(X, efficiency, color='blue', label=r'$\Gamma_m / 2 \Gamma_\varphi$')
    ax2.plot(X, efficiencyTotal, color='red', linestyle='--', label=r'$\Gamma_m / 2 (\Gamma_\varphi + \Gamma_2^\tau)$')
    ax2.axhline(1, color='black', linestyle=':')
    ax2.set_xlabel(Xlabel)
    ax2.set_ylabel(r'Efficiency')
    ax2.legend()

    fig.tight_layout()
    fig.savefig(output_dir / "efficiency.pdf")
    plt.close(fig)

    return efficiency, efficiencyTotal


##############################################################################
## Parameters: edit these. Everything below the function definitions uses them.
##############################################################################
# all frequencies are in the RWA
params = {
    "omega_r": 0,    # frequency of the cavity
    "omega_q": 20,    # frequency of the inner qubit
    "omega_s": 20,    # frequency of the outer qubit
    "omega_d": 0,    # frequency of the drive

    "beta": 1,    # coherent drive amplitude

    "g": 10,          # light-matter coupling strength between the cavity and inner qubit
    "gd": 50,         # coupling strength between inner and outer qubit

    "theta_q": 0.000,  # mixing angle of the inner qubit
    "theta_s": 0.5,  # mixing angle of the outer qubit

    "kappac": 1,   # cavity coupling rate to the transmission line
    "kappa": 1,      # total cavity decay rate

    "Gamma_1s": 0.1,    # relaxation rate of inner qubit
    "Gamma_phis": 0.2,  # dephasing rate of inner qubit

    "Gamma_1t": 0.00,   # relaxation rate of outer qubit
    "Gamma_phit": 0.2,  # dephasing rate of outer qubit
}
# C, S, Gamma_2s and Gamma_2t are derived automatically (see withDerived),
# and the dissipators are rebuilt from these values inside each function.

# To try other values without touching the dictionary above:
# params_other = {**params, "beta": 0.1, "theta_q": 0.2}
# calculateFullPlot(params_other, ...)


## Running calculations
startTime = time.time()

#calculateFullPlot(params, rho0='excited', sweep=['omegad', -10, 10, 100], tMax=15, tRes=60)
#calculateFullPlot(params, rho0='ground', sweep=['omegad', -10, 10, 100], tMax=15, tRes=60)

# Signal-to-noise ratio, measurement rate and efficiency
calculateSNRPlot(params, eta=1, tMax=25, tRes=100)
data = calculateMeasurementSweep(params, sweep=['omegad', -5, 5, 80], eta=1, tMax=25, tRes=100)
calculateMeasurementRatePlot(data, params)
calculateEfficiencyPlot(data, params)

endTime = time.time()
timeDifference = endTime - startTime
minutes = timeDifference // 60
seconds = timeDifference % 60
print(f"Time taken: {int(minutes)} m {int(seconds)} s")

# density = calculateSteadyState(params, sweep=['omegad', -10, 10, 250])
# calculateMeasurementRate(density, params)
# calculateReflectionCompare(density, params)

