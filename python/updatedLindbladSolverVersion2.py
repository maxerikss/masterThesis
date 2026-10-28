import numpy as np
import qutip as qt
import scipy.linalg as la
import scipy.sparse as sparse
from scipy.sparse.linalg import expm_multiply
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import LogNorm
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import os
import time
import warnings
from tqdm import tqdm

#  Matplotlib setup
USE_LATEX = True   # set to False if you do not have a LaTeX installation
plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': '18'})
plt.rc('text.latex', preamble=r'\usepackage{amssymb,amsmath,amsfonts,amsthm}')
plt.rcParams['text.usetex'] = USE_LATEX

## QuTiP operators (these do not depend on the physical parameters)
N = 35 # size of cavity Hilbert space (was 20).
# The Liouvillian has dimension (4N)^2 x (4N)^2. With method='eig' it is diagonalized once per parameter point, which
# costs ~(4N)^6 in time and ~(4N)^4 in memory (N=12: seconds, N=16: ~2 min, N=20: many minutes and a few GB).
# The default method='sparse' avoids this and is much less sensitive to N. In both cases use the smallest N that the
# photon number allows: calculateSNR() and the sweeps warn if the top Fock level gets populated (> 1e-4).

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


# Sweep names -> the key in the parameter dictionary they change. Any other key of `params`
# (beta, g, gd, kappac, theta_q, ...) can also be swept by using its own name.
SWEEP_ALIASES = {"omegad": "omega_d", "omegaq": "omega_q"}


def writeParameters(filename, sweep, params, sweep2=None):
    """
    Saves the parameters used in a text file for later reference. The main use is in other functions that generate plots.

    Parameters:
        filename (Path):
            The path to the file where the parameters will be saved.
        sweep (list):
            A list containing the sweep parameter and its range. The first element is a string indicating which parameter is being swept. Second and third elements are the minimum and maximum values of the sweep.
        params (dict):
            Dictionary of the fixed parameters. Entries for the swept parameter are replaced by the sweep range.
        sweep2 (list or None):
            Optional second sweep (same format as `sweep`) for two-dimensional sweeps.
    """
    p = withDerived(params)

    # Map sweep names to the parameter they replace
    sweptKey = SWEEP_ALIASES.get(sweep[0], sweep[0])
    sweptKey2 = None if sweep2 is None else SWEEP_ALIASES.get(sweep2[0], sweep2[0])

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
            elif key == sweptKey2:
                f.write(f"{key} = [{sweep2[1]}, {sweep2[2]}]\n")
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

##############################################################################
## Quantum regression by diagonalizing the Liouvillian (replaces qt.correlation_2op_2t)
##############################################################################
#
# Conventions (notes, Secs. 5-6): the measured quadrature is X_out = X_in + sqrt(kappac) X_phi, with
# X_phi = e^{-i phi} a + e^{+i phi} a^dagger, and for t > t' (Eq. 14-16)
#
#   C_phi(t,t') = < :X_phi(t) X_phi(t'): >_T  -  <X_phi(t)> <X_phi(t')>
#               = 2 Re[ e^{-2 i phi} Caa(t,t') ] + 2 Re[ Cada(t,t') ]
#
# with the connected, normally- and time-ordered correlators
#
#   Caa(t,t')  = <a(t) a(t')>          - <a(t)>      <a(t')>
#   Cada(t,t') = <a^dagger(t) a(t')>   - <a^dagger(t)> <a(t')>        (t > t')
#
# Both are *independent of phi*. They are computed once per parameter point (and per initial state), and
# phi then only enters through the trivial prefactors above, so a whole scan over phi is essentially free.
#
# The quantum regression theorem (Eq. 17) is   <A(t'+s) B(t')> = Tr[ A e^{L s} ( B rho(t') ) ].
# Write L = V diag(lam) V^-1. Then everything is a sum of exponentials:
#
#   rho(t)           = V exp(lam t) c ,                     c = V^-1 vec(rho0)
#   <O>(t)           = w_O . ( exp(lam t) * c ) ,           w_O = vec(O^T)^T V
#   Tr[A e^{L s}(B rho(t_i))] = sum_m w_A[m] exp(lam_m s) [V^-1 vec(B rho(t_i))]_m
#
# so the full two-time table over (t_i, s_k) is a single (n x K)(K x n) matrix product (K = dim^2).
# The only expensive step is the diagonalization of L, which is done once per parameter point and is shared
# by both initial states of the outer qubit.


def initialDensityMatrix(rho0):
    """
    Outer qubit (S) in |j>, inner qubit (D) in its ground state, empty cavity: |j><j|_S x |g><g|_D x |0><0|_cav.
    rho0 is 'ground' (tz = -1) or 'excited' (tz = +1).
    """
    if rho0 == 'ground':
        ket = qt.tensor(qt.basis(N, 0), qt.basis(2, 1), qt.basis(2, 1))  # outer qubit in |1> (ground) eigval -1
    elif rho0 == 'excited':
        ket = qt.tensor(qt.basis(N, 0), qt.basis(2, 1), qt.basis(2, 0))  # outer qubit in |0> (excited) eigval 1
    else:
        raise ValueError(f"rho0 must be 'ground' or 'excited', got {rho0}")
    return qt.ket2dm(ket)


def buildLiouvillian(H, c_ops, dense=False):
    """
    Dense matrix of the Lindbladian  L rho = -i[H, rho] + sum_k D[c_k] rho  acting on vec(rho), using the
    column-stacking convention vec(A rho B) = (B^T kron A) vec(rho), i.e. vec(rho)[r + d*c] = rho[r, c].
    (Built directly with scipy so it does not depend on QuTiP's internal superoperator layout.)

    Parameters:
        H (qutip.Qobj): Hamiltonian.
        c_ops (list[qutip.Qobj]): collapse operators.
        dense (bool): return a dense ndarray instead of a scipy sparse matrix.

    Returns:
        L: complex matrix of shape (d^2, d^2).
    """
    Hs = sparse.csr_matrix(H.full())
    d = Hs.shape[0]
    I = sparse.identity(d, dtype=complex, format='csr')

    L = -1j * (sparse.kron(I, Hs) - sparse.kron(Hs.T, I))
    for c_op in c_ops:
        c = sparse.csr_matrix(c_op.full())
        cdc = c.conj().T @ c
        L = L + sparse.kron(c.conj(), c) - 0.5 * sparse.kron(I, cdc) - 0.5 * sparse.kron(cdc.T, I)
    return L.toarray() if dense else L.tocsr()


class LiouvillianSpectrum:
    """
    Eigendecomposition L = V diag(lam) V^-1 of the Lindbladian, and the quantum-regression machinery built on it.

    Parameters:
        H (qutip.Qobj): Hamiltonian.
        c_ops (list[qutip.Qobj]): collapse operators.
    """

    def __init__(self, H, c_ops):
        self.d = H.shape[0]
        L = buildLiouvillian(H, c_ops, dense=True)
        self.lam, self.V = la.eig(L)
        del L                                      # free the dense Liouvillian before the LU copy of V is made
        self.luV = la.lu_factor(self.V)  # used to apply V^-1 to vectors without forming the inverse

    def coefficients(self, rho0):
        """Expansion coefficients c = V^-1 vec(rho0) of an initial density matrix (ndarray, d x d)."""
        return la.lu_solve(self.luV, np.asarray(rho0).reshape(-1, order='F'))

    def readout(self, op):
        """Row vector w_O with <O>(t) = w_O . (exp(lam t) * c), for an operator O given as a d x d ndarray."""
        return np.asarray(op).reshape(-1) @ self.V  # C-order flatten of O  ==  vec(O^T)

    def _expLam(self, tList):
        return np.exp(np.outer(self.lam, tList))  # shape (K, n)

    def expectation(self, op, c, tList):
        """<O>(t) on the time grid tList, for the initial condition with coefficients c."""
        return self.readout(op) @ (c[:, None] * self._expLam(tList))

    def states(self, c, tList):
        """vec(rho(t_i)) as the columns of a (K, n) array."""
        return self.V @ (c[:, None] * self._expLam(tList))

    def twoTime(self, A_ops, B_op, c, tList):
        """
        Quantum regression theorem on a grid. For every A in A_ops returns the (n x n) array

            G[i, k] = < A(t_i + t_k) B(t_i) > = Tr[ A e^{L t_k} ( B rho(t_i) ) ].

        Entries with t_i + t_k beyond the grid are computed too (they are simply not used later).

        Parameters:
            A_ops (list[ndarray]): the operators A (d x d each).
            B_op (ndarray): the operator B (d x d), applied from the left on rho(t_i).
            c (ndarray): coefficients of the initial state, see `coefficients`.
            tList (ndarray): time grid; t_i and t_k both run over it.
        """
        d, n = self.d, len(tList)
        R = self.states(c, tList)                                           # rho(t_i), shape (K, n)
        BR = np.tensordot(np.asarray(B_op), R.reshape(d, d, n, order='F'), axes=(1, 0))
        Y = la.lu_solve(self.luV, BR.reshape(d * d, n, order='F'))          # eigen-coefficients of B rho(t_i)
        T = self._expLam(tList)                                             # e^{lam_m t_k}, shape (K, n)
        return [(Y * self.readout(A)[:, None]).T @ T for A in A_ops]


def doubleIntegralCumulative(C, dt):
    """Return I[m] = integral of C(t,t') over the full square [0,T_m]^2, for every m,
    assuming C is symmetric under t <-> t'. C[i,k] = C(t'=t_i, t=t_i+t_k). C may be complex."""
    n = C.shape[0]
    A = C[:, ::-1]                      # anti-diagonals of C become diagonals of A
    # h[m] = integral over t' of C(t=T_m, t'),  entries C[i, m-i], i = 0..m  (trapezoid rule)
    h = np.empty(n, dtype=C.dtype)
    for m in range(n):
        diag = np.diagonal(A, offset=n - 1 - m)
        h[m] = dt * (diag.sum() - 0.5 * (diag[0] + diag[-1]))
    # F[m] = running integral of h  (this is the triangle integral)
    F = np.concatenate(([0.0], np.cumsum(0.5 * dt * (h[1:] + h[:-1]))))
    return 2 * F                        # factor 2: the other triangle


def variance(p, C, tList):
    dt = tList[1] - tList[0]
    return tList + p['kappac'] * doubleIntegralCumulative(C, dt)


def _regressionEig(H, c_ops, rho0s, tList, a_m, ad_m, P_top):
    """Backend 1: diagonalize the dense Liouvillian (cost ~ (4N)^6, memory ~ (4N)^4)."""
    spec = LiouvillianSpectrum(H, c_ops)
    identity = np.eye(spec.d)
    out = dict(aExp=[], Gaa=[], Gada=[], topFockPop=0.0, traceError=0.0)
    for rho0 in rho0s:
        c = spec.coefficients(rho0)
        out['aExp'].append(spec.expectation(a_m, c, tList))
        Gaa, Gada = spec.twoTime([a_m, ad_m], a_m, c, tList)   # <a(t'+s) a(t')>, <a^dag(t'+s) a(t')>
        out['Gaa'].append(Gaa)
        out['Gada'].append(Gada)
        out['topFockPop'] = max(out['topFockPop'], np.max(np.real(spec.expectation(P_top, c, tList))))
        out['traceError'] = max(out['traceError'], np.max(np.abs(spec.expectation(identity, c, tList) - 1)))
    out['spectralAbscissa'] = np.max(spec.lam.real)
    return out


def _regressionSparse(H, c_ops, rho0s, tList, a_m, ad_m, P_top):
    """
    Backend 2: no diagonalization. Uses  Tr[A e^{Ls}(X)] = < e^{L^H s} vec(A^dagger) | vec(X) >,  so instead of
    propagating every vector B rho(t_i) one only has to propagate vec(a^dagger) and vec(a) with the adjoint Liouvillian
    L^H (one vector each), and rho0 with L (one vector per initial state). scipy's expm_multiply does this on the whole
    uniform time grid at once with sparse matrix-vector products, so the cost is ~ nnz(L) * ||L|| * tMax instead of
    (4N)^6, and the memory is only ~ (4N)^2 * n.
    """
    d, n = a_m.shape[0], len(tList)
    K, T = d * d, tList[-1]
    L = buildLiouvillian(H, c_ops)
    Lh = L.conj().T.tocsr()

    vec = lambda M: np.asarray(M).reshape(-1, order='F')
    rho_t = expm_multiply(L, np.stack([vec(r) for r in rho0s], axis=1), start=0, stop=T, num=n, endpoint=True)  # (n, K, 2)
    W = expm_multiply(Lh, np.stack([vec(ad_m), vec(a_m)], axis=1), start=0, stop=T, num=n, endpoint=True)       # (n, K, 2)
    # W[k, :, 0] = e^{L^H t_k} vec(a^dag)  -> A = a ;  W[k, :, 1] = e^{L^H t_k} vec(a)  -> A = a^dag

    uA, uP, uI = a_m.reshape(-1), P_top.reshape(-1), np.eye(d).reshape(-1)    # <O> = u_O . vec(rho)
    out = dict(aExp=[], Gaa=[], Gada=[], topFockPop=0.0, traceError=0.0, spectralAbscissa=np.nan)
    for j in range(len(rho0s)):
        R = rho_t[:, :, j]                                                    # vec(rho(t_i)), shape (n, K)
        out['aExp'].append(R @ uA)
        out['topFockPop'] = max(out['topFockPop'], np.max(np.real(R @ uP)))
        out['traceError'] = max(out['traceError'], np.max(np.abs(R @ uI - 1)))
        BR = np.tensordot(a_m, R.T.reshape(d, d, n, order='F'), axes=(1, 0)).reshape(K, n, order='F')  # vec(a rho(t_i))
        out['Gaa'].append((W[:, :, 0].conj() @ BR).T)                         # G[i, k] = <a(t_i + t_k) a(t_i)>
        out['Gada'].append((W[:, :, 1].conj() @ BR).T)                        # G[i, k] = <a^dag(t_i + t_k) a(t_i)>
    return out


def regressionData(params, tMax, tRes, warn=True, method='sparse'):
    """
    Solves the problem for both initial states of the outer qubit and returns everything needed for the SNR.

    Parameters:
        method (str):
            'sparse' (default): adjoint propagation with scipy's expm_multiply, no diagonalization. Fast and light on
                memory, so N = 20 is no problem. Preferred for sweeps.
            'eig': diagonalization of the dense Liouvillian (quantum regression as sums of exponentials).
                Same results (agreement ~1e-10), but ~(4N)^6 in time and ~(4N)^4 in memory per parameter point.

    Returns a dict with
        tList          time grid, shape (n,)
        aExp           <a>(t), shape (2, n)                 [index 0: outer qubit ground, 1: excited]
        Iaa, Iada      double integrals (square [0,tau]^2) of the connected correlators Caa, Cada, shape (2, n)
        Caa, Cada      the connected correlators themselves, shape (2, n, n), Caa[j, i, k] = Caa(t'=t_i, t=t_i+t_k)
        topFockPop     largest population of the top cavity Fock level over the whole evolution (truncation check)
        traceError     largest |Tr rho(t) - 1| (check of the numerics)
        spectralAbscissa   'eig' only (nan otherwise): largest Re(lambda); should be 0 up to rounding
    """
    p = withDerived(params)
    tList = np.linspace(0, tMax, int(tRes))
    n = len(tList)
    dt = tList[1] - tList[0]

    Delta_r, Delta_q, Delta_s = getDetunings(p, 'none', None)
    H = Hamiltonian(Delta_r, Delta_q, Delta_s, p)
    c_ops = makeCollapseOps(p)

    a_m, ad_m = a.full(), ad.full()
    P_top = qt.tensor(qt.fock_dm(N, N - 1), qt.qeye(2), qt.qeye(2)).full()
    rho0s = [initialDensityMatrix(label).full() for label in ('ground', 'excited')]

    if method == 'eig':
        raw = _regressionEig(H, c_ops, rho0s, tList, a_m, ad_m, P_top)
    elif method == 'sparse':
        raw = _regressionSparse(H, c_ops, rho0s, tList, a_m, ad_m, P_top)
    else:
        raise ValueError("method must be 'sparse' or 'eig'")

    ii, kk = np.indices((n, n))
    valid = (ii + kk) < n               # only t' + s <= tMax is on the grid
    idx = np.minimum(ii + kk, n - 1)

    aExp = np.array(raw['aExp'])
    Caa = np.zeros((2, n, n), dtype=complex)
    Cada = np.zeros((2, n, n), dtype=complex)
    Iaa = np.zeros((2, n), dtype=complex)
    Iada = np.zeros((2, n), dtype=complex)
    for j in range(2):
        Caa[j] = np.where(valid, raw['Gaa'][j] - aExp[j][idx] * aExp[j][ii], 0)
        Cada[j] = np.where(valid, raw['Gada'][j] - np.conj(aExp[j][idx]) * aExp[j][ii], 0)
        Iaa[j] = doubleIntegralCumulative(Caa[j], dt)
        Iada[j] = doubleIntegralCumulative(Cada[j], dt)

    topFockPop, traceError, spectralAbscissa = raw['topFockPop'], raw['traceError'], raw['spectralAbscissa']
    if warn:
        if topFockPop > 1e-4:
            warnings.warn(f"Population of the top cavity Fock level reaches {topFockPop:.1e}: "
                          f"the Fock space (N = {N}) may be too small, increase N.")
        if traceError > 1e-6 or spectralAbscissa > 1e-8:
            warnings.warn(f"The numerics look inaccurate (max |Tr rho - 1| = {traceError:.1e}, "
                          f"max Re(lambda) = {spectralAbscissa:.1e}); with method='eig' the Liouvillian may be close "
                          f"to an exceptional point, so V is ill-conditioned.")

    return dict(tList=tList, aExp=aExp, Caa=Caa, Cada=Cada, Iaa=Iaa, Iada=Iada,
                topFockPop=topFockPop, traceError=traceError, spectralAbscissa=spectralAbscissa)


def measurementTime(snr, tList, threshold=1.0):
    """
    First time at which SNR reaches `threshold` (default 1), or np.inf if it never does.
    SNR^2 is interpolated linearly between grid points (it grows linearly in the QND-like regime).
    """
    above = np.nonzero(snr >= threshold)[0]
    if len(above) == 0:
        return np.inf
    k = above[0]
    if k == 0:
        return tList[0]
    s0, s1 = snr[k - 1] ** 2, snr[k] ** 2
    return tList[k - 1] + (threshold ** 2 - s0) / (s1 - s0) * (tList[k] - tList[k - 1])


def snrFromData(params, data, phi='auto', phiRes=64):
    """
    Assembles SNR(tau) = |Delta M| / (sigma_0 + sigma_1) (Eq. 18) from the output of `regressionData`.

    Parameters:
        params (dict): parameter dictionary (uses kappac).
        data (dict): output of regressionData.
        phi (float or 'auto'): homodyne phase. 'auto' scans phi in [0, pi) with `phiRes` points and picks the one
            that gives the shortest measurement time tau_m (or, if SNR = 1 is never reached, the highest SNR peak).
        phiRes (int): number of phases in the scan.

    Returns:
        snr (ndarray): SNR(tau) on the time grid.
        info (dict): phi, tauM (np.inf if never reached), snrMax, tauAtMax, GammaM (slope of SNR^2 over the last
            30% of the time grid; meaningful in the QND-like regime), plus the diagnostics of regressionData.
    """
    p = withDerived(params)
    kc = p['kappac']
    tList = data['tList']
    dt = tList[1] - tList[0]

    # complex integral of the signal difference, D(tau) = int_0^tau (<a>_exc - <a>_gnd) dt
    dA = data['aExp'][1] - data['aExp'][0]
    D = np.concatenate(([0.0], np.cumsum(0.5 * dt * (dA[1:] + dA[:-1]))))

    def curves(phis):
        phis = np.atleast_1d(np.asarray(phis, dtype=float))[:, None]                    # (nphi, 1)
        dM = 2 * np.sqrt(kc) * np.real(np.exp(-1j * phis) * D[None, :])                 # (nphi, n)
        var = tList[None, :] + kc * (2 * np.real(np.exp(-2j * phis) * data['Iaa'][:, None, :])
                                     + 2 * np.real(data['Iada'][:, None, :]))            # (2, nphi, n)
        sigma = np.sqrt(np.maximum(var, 0.0))
        den = sigma[0] + sigma[1]
        snr = np.zeros_like(dM)
        np.divide(np.abs(dM), den, out=snr, where=den > 0)
        return snr

    if isinstance(phi, str):
        if phi != 'auto':
            raise ValueError("phi must be a number or 'auto'")
        phis = np.linspace(0, np.pi, int(phiRes), endpoint=False)   # phi and phi+pi only differ by a sign
        allSnr = curves(phis)
        taus = np.array([measurementTime(s, tList) for s in allSnr])
        best = np.argmin(taus) if np.any(np.isfinite(taus)) else np.argmax(allSnr.max(axis=1))
        phiBest, snr = phis[best], allSnr[best]
    else:
        phiBest, snr = float(phi), curves(phi)[0]

    i0 = int(0.7 * len(tList))
    GammaM = np.polyfit(tList[i0:], snr[i0:] ** 2, 1)[0]
    iMax = int(np.argmax(snr))

    info = dict(phi=phiBest, tauM=measurementTime(snr, tList), snrMax=snr[iMax], tauAtMax=tList[iMax],
                GammaM=GammaM, topFockPop=data['topFockPop'], traceError=data['traceError'],
                spectralAbscissa=data['spectralAbscissa'])
    return snr, info


def calculateSNR(params, phi, tMax, tRes, phiRes=64, returnInfo=False, warn=True, method='sparse'):
    """
    SNR(tau) of the integrated homodyne signal for distinguishing the outer qubit in |ground> from |excited>.
    Uses the exact quantum regression theorem (no QuTiP solvers).

    Parameters:
        params (dict): parameter dictionary.
        phi (float or 'auto'): homodyne phase; 'auto' picks the best one, see snrFromData.
        tMax (float): the SNR is computed for tau in [0, tMax]. Must be well above the measurement time.
        tRes (int): number of time steps. The double integral uses the trapezoid rule, so dt = tMax/(tRes-1) has to
            resolve the fastest oscillation in the correlation functions (roughly the qubit/cavity detunings).
        phiRes (int): number of phases scanned for phi='auto'.
        returnInfo (bool): also return the info dict (phi, tauM, snrMax, ...).
        warn (bool): emit warnings for a too small Fock space or inaccurate numerics.
        method (str): 'sparse' (default, fast) or 'eig' (diagonalization), see regressionData.

    Returns:
        snr, tList  (and info if returnInfo=True)
    """
    p = withDerived(params)
    data = regressionData(p, tMax, tRes, warn=warn, method=method)
    snr, info = snrFromData(p, data, phi=phi, phiRes=phiRes)
    if returnInfo:
        return snr, data['tList'], info
    return snr, data['tList']


##############################################################################
## Sweeps
##############################################################################

PARAM_LABELS = {
    'omega_r': r'$\omega_r$', 'omega_q': r'$\omega_q$', 'omega_s': r'$\omega_s$', 'omega_d': r'$\omega_d$',
    'beta': r'$\beta$', 'g': r'$g$', 'gd': r'$g_d$', 'theta_q': r'$\theta_q$', 'theta_s': r'$\theta_s$',
    'kappac': r'$\kappa_c$', 'kappa': r'$\kappa$',
    'Gamma_1s': r'$\Gamma_1^{s}$', 'Gamma_phis': r'$\Gamma_\phi^{s}$',
    'Gamma_1t': r'$\Gamma_1^{\tau}$', 'Gamma_phit': r'$\Gamma_\phi^{\tau}$',
}


def parseSweep(sweep, params):
    """
    sweep = [name, minimum, maximum, number of values] or [name, minimum, maximum, number, 'log'].
    `name` is 'omegad', 'omegaq' or any key of the parameter dictionary (beta, g, gd, kappac, theta_q, ...).
    Returns (name, key in params, array of values).
    """
    name, lo, hi, num = sweep[0], sweep[1], sweep[2], int(sweep[3])
    scale = sweep[4] if len(sweep) > 4 else 'lin'
    key = SWEEP_ALIASES.get(name, name)
    if key not in params:
        raise KeyError(f"Cannot sweep '{name}': '{key}' is not a parameter. Known: {sorted(params)}")
    if scale == 'log':
        values = np.geomspace(lo, hi, num)
    elif scale == 'lin':
        values = np.linspace(lo, hi, num)
    else:
        raise ValueError(f"Unknown sweep scale '{scale}', use 'lin' or 'log'")
    return name, key, values


def sweepAxis(name, values, params):
    """Values and label to show on a plot axis. omegad -> Delta_r, omegaq -> omega_r - omega_q (as in your other plots)."""
    if name == 'omegad':
        return params['omega_r'] - values, r'$\Delta_r = \omega_r - \omega_d$'
    if name == 'omegaq':
        return params['omega_r'] - values, r'$\omega_r - \omega_q$'
    key = SWEEP_ALIASES.get(name, name)
    fallback = key.replace('_', r'\_') if USE_LATEX else key
    return values, PARAM_LABELS.get(key, fallback)


_threadLimit = None


def initWorker():
    """Runs once in every worker process: restrict BLAS/OpenMP to one thread, otherwise nJobs workers each start
    one thread per core and slow each other down (oversubscription)."""
    global _threadLimit
    try:
        from threadpoolctl import threadpool_limits
        _threadLimit = threadpool_limits(limits=1)
    except ImportError:
        pass


def sweepPoint(args):
    """One point of a sweep (top-level function so that it can be sent to worker processes)."""
    p, phi, tMax, tRes, phiRes, method = args
    try:
        snr, tList, info = calculateSNR(p, phi, tMax, tRes, phiRes=phiRes, returnInfo=True, warn=False,
                                        method=method)
    except np.linalg.LinAlgError:
        return None
    info['snr'] = snr
    return info


def sweepMeasurementTime(params, sweep, sweep2=None, phi='auto', tMax=40, tRes=200, phiRes=64, nJobs=1,
                         method='sparse'):
    """
    Computes the measurement time tau_m (first tau with SNR = 1) on a one- or two-dimensional grid of parameters.

    Parameters:
        params (dict): parameter dictionary (fixed values for everything that is not swept).
        sweep (list): [name, min, max, number(, 'log')], see parseSweep.
        sweep2 (list or None): optional second sweep, same format.
        phi (float or 'auto'): homodyne phase, 'auto' optimizes it separately at every point.
        tMax, tRes: time grid, see calculateSNR. tau_m = inf (stored as nan) if SNR = 1 is not reached before tMax.
        phiRes (int): phases scanned for phi='auto'.
        nJobs (int): number of worker processes (points are independent, so this parallelizes well). Each worker is
            limited to one BLAS thread if `threadpoolctl` is installed (pip install threadpoolctl); otherwise also set
            OMP_NUM_THREADS=1 before starting Python. Needs the `if __name__ == '__main__':` guard on Windows/macOS.
        method (str): 'sparse' (default) or 'eig', see regressionData.

    Returns:
        results (dict) with arrays of shape (len(values2), len(values1)) (or (1, len(values1)) for a 1D sweep):
            tauM, snrMax, tauAtMax, phi, GammaM, topFockPop, and snr of shape (len(values2), len(values1), tRes),
            the sweep definitions (name1, values1, name2, values2), the time grid tList, and `best`: the sweep values
            (and tauM, phi) of the point with the shortest measurement time, or None if no point reached SNR = 1.
    """
    name1, key1, values1 = parseSweep(sweep, params)
    if sweep2 is None:
        name2, key2, values2 = None, None, np.array([np.nan])
    else:
        name2, key2, values2 = parseSweep(sweep2, params)
        if key2 == key1:
            raise ValueError("The two sweeps change the same parameter")

    jobs = []
    for v2 in values2:
        for v1 in values1:
            p = dict(params)
            p[key1] = v1
            if key2 is not None:
                p[key2] = v2
            jobs.append((withDerived(p), phi, tMax, tRes, phiRes, method))

    if nJobs == 1:
        outs = [sweepPoint(job) for job in tqdm(jobs)]
    else:
        with ProcessPoolExecutor(max_workers=nJobs, initializer=initWorker) as pool:
            outs = list(tqdm(pool.map(sweepPoint, jobs), total=len(jobs)))

    shape = (len(values2), len(values1))
    res = {k: np.full(shape, np.nan) for k in ('tauM', 'snrMax', 'tauAtMax', 'phi', 'GammaM', 'topFockPop')}
    res['snr'] = np.full(shape + (int(tRes),), np.nan)
    for flat, out in enumerate(outs):
        j, i = divmod(flat, len(values1))
        if out is None:
            continue
        for k in ('tauM', 'snrMax', 'tauAtMax', 'phi', 'GammaM', 'topFockPop'):
            res[k][j, i] = out[k]
        res['snr'][j, i] = out['snr']
    res['tauM'][~np.isfinite(res['tauM'])] = np.nan   # 'never reached' -> nan

    if np.nanmax(res['topFockPop']) > 1e-4:
        warnings.warn(f"Top Fock level population reaches {np.nanmax(res['topFockPop']):.1e} somewhere in the sweep: "
                      f"increase N.")

    res.update(name1=name1, values1=values1, name2=name2, values2=values2,
               tList=np.linspace(0, tMax, int(tRes)), best=None)
    if np.any(np.isfinite(res['tauM'])):
        j, i = np.unravel_index(np.nanargmin(res['tauM']), shape)
        res['best'] = dict(tauM=res['tauM'][j, i], phi=res['phi'][j, i], snrMax=res['snrMax'][j, i],
                           **{name1: values1[i]}, **({} if name2 is None else {name2: values2[j]}))
    return res


def calculateSNRPlot(params, sweep=['none'], sweep2=None, phi='auto', tMax=40, tRes=200, phiRes=64, nJobs=1,
                     method='sparse'):
    """
    Plots the SNR of the integrated signal and the measurement time tau_m (SNR(tau_m) = 1), optionally as a function of
    one or two swept parameters, and prints the parameters with the shortest tau_m.

        sweep = ['none']                          SNR(tau) for the parameters in `params`.
        sweep = [name, min, max, number]          1D sweep: colour map of SNR(tau) vs the parameter (white line:
                                                  SNR = 1) and tau_m vs the parameter.
        sweep = [...], sweep2 = [...]             2D sweep: maps of tau_m and of the maximal SNR.

    `name` is 'omegad', 'omegaq' or any key of `params` (beta, g, gd, kappac, theta_q, ...). Add 'log' as a fifth
    element for logarithmic spacing, e.g. ['beta', 0.1, 10, 30, 'log'].

    Parameters:
        params (dict): parameter dictionary.
        sweep, sweep2 (list): see above.
        phi (float or 'auto'): homodyne phase; 'auto' optimizes it at every point.
        tMax (float): maximum integration time tau. Choose it well above the measurement time you expect; points that do
            not reach SNR = 1 before tMax are left blank in the tau_m plots.
        tRes (int): number of time steps. dt = tMax/(tRes-1) must resolve the oscillations of the correlation functions.
        phiRes (int): phases scanned for phi='auto'.
        nJobs (int): worker processes for sweeps (see sweepMeasurementTime).
        method (str): 'sparse' (default, fast) or 'eig' (diagonalization), see regressionData.

    Returns:
        For ['none']: dict with snr, tList and the info of calculateSNR. For sweeps: the dict of sweepMeasurementTime.
        The figure, the parameters and the raw numbers (.npz) are written to the folder `calculateSNRPlot`.
    """
    p = withDerived(params)
    output_dir = Path("calculateSNRPlot")
    output_dir.mkdir(parents=True, exist_ok=True)

    ## No sweep: SNR(tau) at one point
    if sweep[0] == 'none':
        snr, tList, info = calculateSNR(p, phi, tMax, tRes, phiRes=phiRes, returnInfo=True, method=method)

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.plot(tList, snr, color='blue')
        ax.hlines([0, 1], tList[0], tList[-1], colors=['k'], linestyles=['--'])
        if np.isfinite(info['tauM']):
            ax.axvline(info['tauM'], color='red', linestyle=':')
        ax.set_xlabel(r'$\tau$')
        ax.set_ylabel(r'SNR')
        tauText = rf"{info['tauM']:.3g}" if np.isfinite(info['tauM']) else r"\infty"
        ax.set_title(rf"$\tau_m = {tauText},\ \phi = {info['phi']:.2f}$")
        fig.tight_layout()
        fig.savefig(output_dir / "SNR.pdf")
        plt.close(fig)

        writeParameters(output_dir / "SNRParameters.txt", ['none', np.nan, np.nan], p)
        print(f"tau_m = {info['tauM']:.4g}, phi = {info['phi']:.3f}, max SNR = {info['snrMax']:.3g} "
              f"at tau = {info['tauAtMax']:.3g}, slope of SNR^2 at late times = {info['GammaM']:.4g}")
        return dict(snr=snr, tList=tList, **info)

    ## Sweeps
    res = sweepMeasurementTime(p, sweep, sweep2=sweep2, phi=phi, tMax=tMax, tRes=tRes, phiRes=phiRes, nJobs=nJobs,
                               method=method)
    name1, name2 = res['name1'], res['name2']
    x, xLabel = sweepAxis(name1, res['values1'], p)
    ox = np.argsort(x)

    if name2 is None:
        base = f"SNR_{name1}"
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        snrMap = res['snr'][0][ox]
        mesh = ax1.pcolormesh(res['tList'], x[ox], snrMap, cmap="viridis", shading="auto", rasterized=True)
        fig.colorbar(mesh, ax=ax1, label=r'SNR')
        if np.nanmin(snrMap) < 1 < np.nanmax(snrMap):
            ax1.contour(res['tList'], x[ox], snrMap, levels=[1], colors='white')
        ax1.set_xlabel(r'$\tau$')
        ax1.set_ylabel(xLabel)

        tauRow = res['tauM'][0][ox]
        reached = np.isfinite(tauRow)
        ax2.plot(x[ox][reached], tauRow[reached], 'o-', color='blue', markersize=4)
        if np.any(~reached):   # SNR = 1 not reached before tMax: show these points on the upper edge
            ax2.plot(x[ox][~reached], np.full(np.count_nonzero(~reached), tMax), 'x', color='gray',
                     label=r'SNR $<1$ up to $\tau = t_{\max}$')
            ax2.legend(loc='best', fontsize=12)
        if res['best'] is not None:
            ax2.plot(x[ox][np.nanargmin(tauRow)], res['best']['tauM'], '*', color='red', markersize=15)
        ax2.set_ylim(0, 1.05 * tMax)
        ax2.set_xlabel(xLabel)
        ax2.set_ylabel(r'$\tau_m$')

    else:
        base = f"SNR_{name1}_{name2}"
        y, yLabel = sweepAxis(name2, res['values2'], p)
        oy = np.argsort(y)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        tauMap = np.ma.masked_invalid(res['tauM'][oy][:, ox])
        if tauMap.count() > 0:
            mesh = ax1.pcolormesh(x[ox], y[oy], tauMap, cmap="viridis_r", shading="auto", norm=LogNorm(),
                                  rasterized=True)
            fig.colorbar(mesh, ax=ax1, label=r'$\tau_m$')
            bj, bi = np.unravel_index(np.nanargmin(res['tauM'][oy][:, ox]), tauMap.shape)
            ax1.plot(x[ox][bi], y[oy][bj], '*', color='red', markersize=15)
        else:
            ax1.text(0.5, 0.5, 'SNR = 1 never reached\n(increase tMax)', ha='center', va='center',
                     transform=ax1.transAxes)

        mesh = ax2.pcolormesh(x[ox], y[oy], res['snrMax'][oy][:, ox], cmap="viridis", shading="auto", rasterized=True)
        fig.colorbar(mesh, ax=ax2, label=r'$\max_\tau$ SNR')

        for ax, title in ((ax1, r'$\tau_m$'), (ax2, r'maximal SNR')):
            ax.set_xlabel(xLabel)
            ax.set_ylabel(yLabel)
            ax.set_title(title)

    fig.tight_layout()
    fig.savefig(output_dir / f"{base}.pdf", dpi=300)
    plt.close(fig)

    def rng(values):
        return [np.min(values), np.max(values)]
    writeParameters(output_dir / f"{base}Parameters.txt", [name1, *rng(res['values1'])], p,
                    sweep2=None if name2 is None else [name2, *rng(res['values2'])])
    np.savez(output_dir / f"{base}.npz", **{k: v for k, v in res.items() if isinstance(v, np.ndarray)})

    if res['best'] is None:
        print("SNR = 1 was not reached anywhere in the sweep before tMax: increase tMax (or change the sweep range).")
    else:
        b = res['best']
        values = ", ".join(f"{k} = {v:.4g}" for k, v in b.items() if k in (name1, name2))
        if 'omegad' in (name1, name2):
            values += f"  (Delta_r = {p['omega_r'] - b['omegad']:.4g})"
        print(f"Shortest measurement time: tau_m = {b['tauM']:.4g} at {values}, phi = {b['phi']:.3f}, "
              f"max SNR = {b['snrMax']:.3g}")
    return res


##############################################################################
## Reference implementation with QuTiP (slow; kept only to validate the code above)
##############################################################################

def CalculateC_qutip(params, tList, phi, rho0='ground', sweep='none'):
    p = withDerived(params)
    c_ops = makeCollapseOps(p)

    initRhoState = initialDensityMatrix(rho0)

    if sweep == 'none':
        Delta_r, Delta_q, Delta_s = getDetunings(p, sweep, None)

        H = Hamiltonian(Delta_r, Delta_q, Delta_s, p)

        aT_at = qt.correlation_2op_2t(H, initRhoState, tList, tList, c_ops, a, a)
        adT_at = qt.correlation_2op_2t(H, initRhoState, tList, tList, c_ops, ad, a)
        #adt_aT = qt.correlation_2op_2t(H, initRhoState, tList, tList, c_ops, ad, a, reverse=True)
        #adt_adT = qt.correlation_2op_2t(H, initRhoState, tList, tList, c_ops, ad, ad, reverse=True)
        adt_aT = np.conj(adT_at)
        adt_adT = np.conj(aT_at)

        aVals = qt.mesolve(H, initRhoState, tList, c_ops, e_ops=[a]).expect[0]

        X = 2 * np.real(np.exp(-1j * phi) * aVals)

        n = len(tList)                                  # Matrix Looks like:    X X X X 
        prod = np.zeros((n, n))                         #                       X X X 0
        for i in range(n):                              #                       X X 0 0
            prod[i, : n - i] = X[i] * X[i:]             #                       X 0 0 0 

        return np.exp(-2j*phi) * aT_at + adT_at + adt_aT + np.exp(2j*phi) * adt_adT - prod


def calculateSNR_qutip(params, phi, tMax, tRes):
    p = withDerived(params)

    tList = np.linspace(0, tMax, tRes)

    densityList, expectList, sweepInfo = MEsolve(p, rho0='ground', tMax=tMax, tRes=tRes, e_ops=[a])
    groundExpectA = [expectList[0], sweepInfo[2]]

    densityList, expectList, sweepInfo = MEsolve(p, rho0='excited', tMax=tMax, tRes=tRes, e_ops=[a])
    excitedExpectA = [expectList[0], sweepInfo[2]]

    groundExpectM = expectationM(tList, groundExpectA, phi, p)
    excitedExpectM = expectationM(tList, excitedExpectA, phi, p)
    diffExpectM = np.abs(excitedExpectM - groundExpectM)

    Cground = CalculateC_qutip(p, tList, phi, rho0='ground')
    Cexcited = CalculateC_qutip(p, tList, phi, rho0='excited')

    stdGround = np.sqrt(variance(p, Cground, tList))
    stdExcited = np.sqrt(variance(p, Cexcited, tList))

    snr = np.zeros_like(tList)
    snr[1:] = np.real(diffExpectM[1:] / (stdGround[1:] + stdExcited[1:]))

    return snr, tList


def compareWithQutip(params, phi=0.3, tMax=8, tRes=20, method='sparse'):
    """
    Checks the new code (method='sparse' or 'eig') against the QuTiP reference on a small grid (this is slow because of
    qt.correlation_2op_2t, so keep tRes small). Prints the largest deviations; they should be at the level of
    the QuTiP solver tolerance (~1e-6 or better).
    """
    p = withDerived(params)
    data = regressionData(p, tMax, tRes, method=method)
    tList = data['tList']
    n = len(tList)
    ii, kk = np.indices((n, n))
    valid = (ii + kk) < n

    for j, label in enumerate(('ground', 'excited')):
        Cq = np.real(CalculateC_qutip(p, tList, phi, rho0=label))
        Cn = 2 * np.real(np.exp(-2j * phi) * data['Caa'][j]) + 2 * np.real(data['Cada'][j])
        print(f"{label}: max |C_qutip - C_eig| = {np.max(np.abs(Cq - Cn)[valid]):.2e}  "
              f"(max |C| = {np.max(np.abs(Cq[valid])):.2e})")

    snrQ, _ = calculateSNR_qutip(p, phi, tMax, tRes)
    snrE, _ = calculateSNR(p, phi, tMax, tRes, method=method)
    print(f"max |SNR_qutip - SNR_eig| = {np.max(np.abs(snrQ - snrE)):.2e}  (max SNR = {np.max(snrE):.3g})")


##############################################################################
## Parameters: edit these. Everything below the function definitions uses them.
##############################################################################
# all frequencies are in the RWA
params = {
    "omega_r": 0,    # frequency of the cavity
    "omega_q": 0,    # frequency of the inner qubit
    "omega_s": 2,    # frequency of the outer qubit
    "omega_d": 0,    # frequency of the drive

    "beta": 4,    # coherent drive amplitude

    "g": 8,          # light-matter coupling strength between the cavity and inner qubit
    "gd": 16,         # coupling strength between inner and outer qubit

    "theta_q": 0.2,  # mixing angle of the inner qubit
    "theta_s": 0.0,  # mixing angle of the outer qubit

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
if __name__ == '__main__':   # the guard is needed for nJobs > 1 on Windows/macOS
    startTime = time.time()
    print("Starting the calculations...")

    #calculateFullPlot(params, rho0='excited', sweep=['omegad', -10, 10, 100], tMax=15, tRes=60)
    #calculateFullPlot(params, rho0='ground', sweep=['omegad', -10, 10, 100], tMax=15, tRes=60)

    # SNR(tau) at one point; phi='auto' optimizes the homodyne phase (use phi=0 to reproduce a fixed phase)
    #calculateSNRPlot(params, sweep=['none'], phi='auto', tMax=20, tRes=100)

    # 1D sweep: where is the measurement time shortest as a function of the drive frequency?
    calculateSNRPlot(params, sweep=['kappac', 0, 1, 25], tMax=10, tRes=100, nJobs=4)

    # 2D sweep: any key of `params` can be swept, e.g. drive frequency and the qubit-qubit coupling
    #calculateSNRPlot(params, sweep=['g', 0.5, 10, 20], sweep2=['gd', 8, 15, 20], tMax=10, tRes=50, nJobs=4)
    # logarithmic spacing: sweep=['beta', 0.1, 10, 30, 'log']

    # Check against the (slow) QuTiP implementation on a small grid:
    #compareWithQutip(params, phi=1.57, tMax=20, tRes=100)

    endTime = time.time()
    print("Finished.")
    timeDifference = endTime - startTime
    minutes = timeDifference // 60
    seconds = timeDifference % 60
    print(f"Time taken: {int(minutes)} m {int(seconds)} s")

    # density = calculateSteadyState(params, sweep=['omegad', -10, 10, 250])
    # calculateMeasurementRate(density, params)
    # calculateReflectionCompare(density, params)
