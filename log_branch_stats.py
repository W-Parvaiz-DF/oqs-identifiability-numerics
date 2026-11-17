"""
How often the noisy one-step maps have no real principal logarithm.

Reruns the sweeps behind Figs. 1-2 with the same seeds as paper_figures.py and,
for every estimate, counts the pulse settings whose least-squares one-step map
Phi has an eigenvalue on the negative real axis. For those maps the principal
logarithm is complex and the estimator keeps its real part. These are the
numbers quoted in Sec. V.

Output: for each protocol and shot count, the fraction of maps (over settings
and pulse sequences) and of pulse sequences affected, at each K.
"""
import warnings
import numpy as np
import two_qubit_sampling as tqs
import paper_figures as pf

warnings.filterwarnings("ignore", message="logm result may be inaccurate")

TOL = 1e-12        # |Im| below this counts as real
_log = []          # per call of estimate_A: (settings with data, settings affected)
_estimate_A = tqs.estimate_A


def counting_estimate_A(Zm, seq, levels):
    """Same estimator, but record negative real eigenvalues of each Phi."""
    I = np.eye(tqs.n + 1)
    used = bad = 0
    for c in levels:
        idx = np.flatnonzero(seq == c)
        if len(idx) == 0:
            continue
        Phi = I + (Zm[:, idx + 1] - Zm[:, idx]) @ np.linalg.pinv(Zm[:, idx])
        ev = np.linalg.eigvals(Phi)
        used += 1
        bad += bool(np.any((np.abs(ev.imag) < TOL) & (ev.real < 0)))
    _log.append((used, bad))
    return _estimate_A(Zm, seq, levels)


def stats(key, shots):
    _log.clear()
    pf.run(key, shots)
    L = np.array(_log).reshape(len(pf.K_LIST), pf.REPEATS, 2)
    return L[:, :, 1].sum(1) / L[:, :, 0].sum(1), (L[:, :, 1] > 0).mean(1)


if __name__ == "__main__":
    tqs.estimate_A = counting_estimate_A
    tqs.REPEATS, tqs.K_LIST, tqs.BAND = pf.REPEATS, pf.K_LIST, "iqr"
    print("K:", pf.K_LIST)
    cases = [(k, None) for k in pf.PROTOCOLS] + [(k, S) for k in pf.NOISY for S in pf.SHOTS]
    for key, S in cases:
        maps, runs = stats(key, S)
        label = f"{key}, S={'inf' if S is None else f'{S:.0e}'}"
        print(f"{label:18s} maps:", np.round(maps, 2).tolist())
        print(f"{'':18s} runs:", np.round(runs, 2).tolist())
