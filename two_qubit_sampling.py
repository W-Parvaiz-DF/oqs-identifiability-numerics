"""
Two-qubit sampling experiment (controlled OQS).

Drive the two-qubit model with piecewise-constant on/off pulses, sample the
coherence vector on a uniform grid, reconstruct (theta, gamma) via
A_hat -> M^{-1}, and plot the relative error against the number of samples K.
The dashed vertical line marks K*, the first K at which the data matrix of
every pulse level reaches rank n+1 = 16. That is the identifiability condition.

Only needs numpy, scipy and matplotlib.
"""
import itertools
import numpy as np
from scipy.linalg import expm, logm
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, NullFormatter

# ============================================================ 1. settings
DT = 0.5                     # sampling interval
ALPHA = 1.0                  # pulse amplitude
WIDTH = (1, 5)               # pulse width, in samples (min, max)
K_LIST = [20, 30, 40, 50, 60, 70, 80, 100, 150, 200, 300, 500, 800]
SHOTS = [None, 1e6, 1e4]     # shots per expectation value (None = noise-free)
REPEATS = 20                 # random pulse sequences per K
BAND = "iqr"                 # spread around each curve: "iqr" (25-75%) or "logstd" (+-1 std of log10 error)
CONTROLS = {                 # which pulse levels are allowed
    "no control":    [0],        # level 0 = drive off
    "drive qubit 1": [0, 1],     # level 1 = sigma_x on qubit 1
    "drive both":    [0, 1, 2],  # level 2 = sigma_x on qubit 2
}

# ============================================================ 2. model
I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]])
Z = np.diag([1, -1]).astype(complex)
SP = np.array([[0, 1], [0, 0]], dtype=complex)   # sigma_+
SM = SP.T.copy()                                  # sigma_-
on1 = lambda o: np.kron(o, I2)
on2 = lambda o: np.kron(I2, o)

PAULI = [I2, X, Y, Z]
F = [np.kron(PAULI[a], PAULI[b]) / 2               # orthonormal, traceless
     for a, b in itertools.product(range(4), repeat=2) if (a, b) != (0, 0)]
N, n = 4, len(F)                                   # n = 15

w1, w2, J = 1.0, 1.3, 0.2
H0 = w1 / 2 * on1(Z) + w2 / 2 * on2(Z) + J * (on1(SP) @ on2(SM) + on1(SM) @ on2(SP))
JUMPS = []                                         # (rate, jump operator)
for q, (gm, gp, gz) in zip([on1, on2], [(0.05, 0.01, 0.02), (0.03, 0.02, 0.04)]):
    JUMPS += [(2 * gm, q(SM)), (2 * gp, q(SP)), (gz / 2, q(Z))]
H_DRIVE = [on1(X), on2(X)]


def lindblad(rho, H):
    out = -1j * (H @ rho - rho @ H)
    for g, L in JUMPS:
        LdL = L.conj().T @ L
        out += g * (L @ rho @ L.conj().T - 0.5 * (LdL @ rho + rho @ LdL))
    return out


def augmented(H):
    """(n+1)x(n+1) matrix [[A, beta], [0, 0]] acting on z = [x; 1], Eq. (15)."""
    At = np.zeros((n + 1, n + 1))
    for k in range(n):
        At[:n, k] = [np.trace(Fj @ lindblad(F[k], H)).real for Fj in F]
    At[:n, n] = [np.trace(Fj @ lindblad(np.eye(N) / N, H)).real for Fj in F]
    return At


A0 = augmented(H0)
NDRIVE = [augmented(H0 + h) - A0 for h in H_DRIVE]            # bilinear N_j
GEN = [A0] + [A0 + ALPHA * Nj for Nj in NDRIVE]               # one per pulse level
PROP = [expm(G * DT) for G in GEN]                            # exact one-step maps

# ============================================================ 3. M from structure constants (Eq. 25)
f = np.zeros((n, n, n))
g = np.zeros((n, n, n))
for j, k, l in itertools.product(range(n), repeat=3):
    f[j, k, l] = np.real(-1j * np.trace((F[j] @ F[k] - F[k] @ F[j]) @ F[l]))
    g[j, k, l] = np.real(np.trace((F[j] @ F[k] + F[k] @ F[j]) @ F[l]))
zc = f + 1j * g
D = 0.25 * (np.einsum('lrk,jmr->jklm', zc, f) + np.einsum('mrk,jlr->jklm', zc.conj(), f))
T1 = -f.reshape(n * n, n)
T2 = -D.reshape(n * n, n * n)
M = np.block([[T1, T2], [np.zeros((n, n)), -(1j / N) * T1.T]])

theta_true = np.array([np.trace(Fj @ H0).real for Fj in F])
gamma_true = np.zeros((n, n), dtype=complex)
for rate, L in JUMPS:
    c = np.array([np.trace(Fj @ L) for Fj in F])                # L = sum_j c_j F_j
    gamma_true += rate * np.outer(c, c.conj())

# self-check: M maps the true parameters onto the true (A, beta)
y_true = np.concatenate([theta_true, gamma_true.ravel()])
assert np.allclose(M @ y_true, np.concatenate([A0[:n, :n].ravel(), A0[:n, n]]))
alias = max(np.abs(np.linalg.eigvals(G).imag).max() for G in GEN) * DT   # all pulse levels
assert alias < np.pi, f"max|Im eig|*dt = {alias:.2f} >= pi: logm will alias, reduce DT"
print(f"max|Im eig|*dt = {alias:.2f} < pi")

# ============================================================ 4. pulses, simulation, measurement
psi = np.zeros(N)
psi[0] = 1                                                     # |00>
z0 = np.append([np.trace(Fj @ np.outer(psi, psi)).real for Fj in F], 1.0)


def pulse_sequence(K, levels, rng):
    seq = []
    while len(seq) < K:
        seq += [rng.choice(levels)] * rng.integers(WIDTH[0], WIDTH[1] + 1)
    return np.array(seq[:K])


def simulate(seq):
    zs = [z0]
    for c in seq:
        zs.append(PROP[c] @ zs[-1])
    return np.array(zs).T                                      # (n+1) x (K+1)


def measure(Zt, shots, rng):
    """Each x_j = <P_j>/2; estimate <P_j> from `shots` +-1 outcomes (fresh preparation per time point)."""
    if shots is None:
        return Zt
    p_plus = np.clip((1 + 2 * Zt[:n]) / 2, 0, 1)
    Zm = Zt.copy()
    Zm[:n] = (2 * rng.binomial(int(shots), p_plus) / shots - 1) / 2
    return Zm


# ============================================================ 5. estimation and reconstruction
def estimate_A(Zm, seq, levels):
    """Per-level least squares for exp(G_c dt), matrix log, subtract the known drive, average."""
    I = np.eye(n + 1)
    estimates = []
    for c in levels:
        idx = np.flatnonzero(seq == c)
        if len(idx) == 0:
            continue
        Z0, Z1 = Zm[:, idx], Zm[:, idx + 1]
        # Least-squares one-step map. At full rank this equals Z1 @ pinv(Z0) (Eq. example_ls);
        # below full rank it is the minimum-norm update of the identity, i.e. Phi = 1 on
        # directions the data have not reached.
        Phi = I + (Z1 - Z0) @ np.linalg.pinv(Z0)
        # Principal matrix log. With noisy data Phi can have eigenvalues on the negative real
        # axis; logm is then complex and we keep its real part (see Sec. V and
        # log_branch_stats.py for how often this happens).
        estimates.append(logm(Phi).real / DT - (GEN[c] - A0))
    return np.mean(estimates, axis=0)


def reconstruct(Ahat):
    y = np.linalg.solve(M, np.concatenate([Ahat[:n, :n].ravel(), Ahat[:n, n]]))
    gam = y[n:].reshape(n, n)
    return y[:n].real, (gam + gam.conj().T) / 2               # Hermitian part


def rel_err(est, true):
    return np.linalg.norm(est - true) / np.linalg.norm(true)


def first_identifiable_K(levels, rng):
    """K*: first K at which every level's data matrix has rank n+1 (median over REPEATS sequences)."""
    Ks = []
    for _ in range(REPEATS):
        seq = pulse_sequence(max(K_LIST), levels, rng)
        Zt = simulate(seq)

        def ok(K):
            return all(np.linalg.matrix_rank(Zt[:, np.flatnonzero(seq[:K] == c)], tol=1e-9) == n + 1
                       for c in levels if np.any(seq[:K] == c))

        if not ok(len(seq)):
            Ks.append(np.inf)
            continue
        lo, hi = n, len(seq)                                   # rank grows with K, so bisect
        while hi - lo > 1:
            mid = (lo + hi) // 2
            lo, hi = (lo, mid) if ok(mid) else (mid, hi)
        Ks.append(hi)
    K = np.median(Ks)
    return None if np.isinf(K) else int(K)


# ============================================================ 6. sweep
def summarise(errs):
    """Centre, lower and upper edge of the REPEATS errors at one K."""
    e = np.asarray(errs)
    if BAND == "logstd":                                       # geometric mean, x/÷ geometric std
        m, s = np.mean(np.log10(e)), np.std(np.log10(e))
        return 10 ** m, 10 ** (m - s), 10 ** (m + s)
    return np.median(e), np.percentile(e, 25), np.percentile(e, 75)


def sweep(levels, shots, seed=0):
    rng = np.random.default_rng(seed)
    err_th, err_ga = [], []
    for K in K_LIST:
        eth, ega = [], []
        for _ in range(REPEATS):
            seq = pulse_sequence(K, levels, rng)
            th, ga = reconstruct(estimate_A(measure(simulate(seq), shots, rng), seq, levels))
            eth.append(rel_err(th, theta_true))
            ega.append(rel_err(ga, gamma_true))
        err_th.append(summarise(eth))
        err_ga.append(summarise(ega))
    # each error array: one row per K, columns = (centre, lower, upper)
    return np.array(err_th), np.array(err_ga), first_identifiable_K(levels, np.random.default_rng(seed + 1))


if __name__ == "__main__":
    COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]   # fixed order: series 1, 2, 3

    def plot(results, title, fname):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
        for ax, which, name in zip(axes, [0, 1], [r"system $\theta$", r"environment $\gamma$"]):
            same = len({r[2] for r in results.values()}) == 1      # one K* for all curves -> grey line
            for (label, (eth, ega, kstar)), col in zip(results.items(), COLORS):
                e = [eth, ega][which]
                ax.loglog(K_LIST, e[:, 0], "o-", color=col, lw=2, ms=5, label=label)
                ax.fill_between(K_LIST, e[:, 1], e[:, 2], color=col, alpha=0.2, lw=0)
                if kstar is not None:
                    ax.axvline(kstar, color="0.4" if same else col, ls="--", lw=1)
            ax.set_xticks([20, 50, 100, 200, 500])
            ax.xaxis.set_major_formatter(ScalarFormatter())
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.set_title(name)
            ax.set_xlabel("number of samples $K$")
            ax.grid(True, which="major", color="0.9")
        axes[0].set_ylabel("relative error")
        axes[1].legend(frameon=False)
        band = "25-75% of runs" if BAND == "iqr" else r"$\pm 1$ std of $\log_{10}$ error"
        fig.suptitle(f"{title}; band: {band}, {REPEATS} runs per $K$")
        fig.tight_layout()
        fig.savefig(fname, dpi=150)
        print("saved", fname)

    def plot_pulses(K=80, fname="fig_pulses.png"):
        """One example pulse sequence per control type: rows = u_1, u_2; columns = controls."""
        fig, axes = plt.subplots(2, len(CONTROLS), figsize=(12, 3.5), sharex=True, sharey=True)
        t = DT * np.arange(K + 1)
        for col, ((label, levels), c) in enumerate(zip(CONTROLS.items(), COLORS)):
            seq = pulse_sequence(K, levels, np.random.default_rng(0))
            for j in range(2):                                   # u_j = ALPHA when level j+1 is on
                u = ALPHA * (seq == j + 1)
                axes[j, col].step(t, np.append(u, u[-1]), where="post", color=c, lw=1.5)
            axes[0, col].set_title(label)
            axes[1, col].set_xlabel("$t$")
        axes[0, 0].set_ylim(-0.1 * ALPHA, 1.1 * ALPHA)
        axes[0, 0].set_yticks([0, ALPHA])
        axes[0, 0].set_ylabel(r"$u_1$ ($\sigma_x$ on qubit 1)")
        axes[1, 0].set_ylabel(r"$u_2$ ($\sigma_x$ on qubit 2)")
        fig.tight_layout()
        fig.savefig(fname, dpi=150)
        print("saved", fname)

    plot_pulses()

    # (a) noise-free, compare controls -> the identifiability transition
    res_a = {}
    for label, levels in CONTROLS.items():
        res_a[label] = sweep(levels, None)
        print(f"{label:14s} K* = {res_a[label][2]}")
    plot(res_a, "Noise-free: effect of the control (dashed: $K^*$)", "fig_controls.png")

    # (b) drive both qubits, finite shots
    res_b = {("noise-free" if S is None else f"S = {S:.0e}"): sweep(CONTROLS["drive both"], S) for S in SHOTS}
    plot(res_b, "Drive both qubits: effect of shot noise (dashed: $K^*$)", "fig_shots.png")
