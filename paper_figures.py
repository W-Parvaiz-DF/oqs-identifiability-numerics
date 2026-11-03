"""
Publication figures for the two-qubit sampling experiment.

Uses the model, pulses and estimator from two_qubit_sampling.py (keep both files
in the same folder). Results are cached in paper_data.npz, so you can restyle the
figures without recomputing; delete that file to recompute.

Outputs: fig1_controls.pdf/.png (noise-free: initial state and control) and
fig2_shots.pdf/.png (shot noise: autonomous |++> vs driven |00>), at PRA double-column width.
"""
import os
import warnings
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, ScalarFormatter, NullFormatter
from matplotlib.lines import Line2D
import two_qubit_sampling as tqs

warnings.filterwarnings("ignore", message="logm result may be inaccurate")   # harmless: errors ~1e-13

# ============================================================ settings
FONT = "serif"        # "serif": Times-like, matches the PRA body text; "sans": Helvetica/Arial
REPEATS = 50          # random pulse sequences per K
K_LIST = [16, 18, 20, 22, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 90, 100, 150, 200, 300, 500, 800]
SIZE = (7.0, 2.6)     # inches; 7.0 in = PRA double column (single column = 3.375 in)
CACHE = "paper_data.npz"
SHOTS = [1e6, 1e4]    # shot counts for Fig. 2

k0, k1 = np.array([1, 0]), np.array([0, 1])
kp = (k0 + k1) / np.sqrt(2)


def zvec(psi):
    """Augmented coherence vector [x; 1] of a pure state."""
    return np.append([np.trace(f @ np.outer(psi, psi.conj())).real for f in tqs.F], 1.0)


STATES = {"00": zvec(np.kron(k0, k0)), "++": zvec(np.kron(kp, kp))}
# key: (legend label, pulse levels, initial state, colour, marker)
PROTOCOLS = {
    "auto00": (r"No control, $|00\rangle$", [0], "00", "0.55", "o"),
    "autopp": (r"No control, $|{+}{+}\rangle$", [0], "++", "#eb6834", "D"),
    "drive1": (r"Drive qubit 1, $|00\rangle$", [0, 1], "00", "#1baf7a", "^"),
    "drive2": (r"Drive both, $|00\rangle$", [0, 1, 2], "00", "#2a78d6", "s"),
}
NOISY = ["autopp", "drive2"]  # protocols compared under shot noise in Fig. 2

# ============================================================ style
mpl.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.linewidth": 0.6, "lines.linewidth": 1.2, "lines.markersize": 3.5,
    "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
    "xtick.major.size": 3, "ytick.major.size": 3, "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
    "legend.frameon": False, "legend.handlelength": 2.2,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300,     # embed fonts as editable TrueType
})
if FONT == "serif":
    mpl.rcParams.update({"font.family": "serif", "mathtext.fontset": "stix",
                         "font.serif": ["Times New Roman", "Times", "TeX Gyre Termes", "STIXGeneral", "DejaVu Serif"]})
else:
    mpl.rcParams.update({"font.family": "sans-serif", "mathtext.fontset": "custom",
                         "mathtext.rm": "sans", "mathtext.it": "sans:italic", "mathtext.bf": "sans:bold",
                         "font.sans-serif": ["Helvetica", "Arial", "Liberation Sans", "TeX Gyre Heros", "DejaVu Sans"]})


# ============================================================ data (computed once, then cached)
def run(key, shots):
    """Sweep one protocol: returns theta errors, gamma errors (rows: median, q25, q75 per K) and K*."""
    _, levels, state, _, _ = PROTOCOLS[key]
    z_saved, tqs.z0 = tqs.z0, STATES[state]
    try:
        th, ga, ks = tqs.sweep(levels, shots)
    finally:
        tqs.z0 = z_saved
    return th, ga, -1 if ks is None else ks


def compute():
    tqs.REPEATS, tqs.K_LIST, tqs.BAND = REPEATS, K_LIST, "iqr"
    data = {"K": np.array(K_LIST), "repeats": REPEATS}
    for key in PROTOCOLS:                                              # Fig. 1: noise-free
        data[f"{key}_th"], data[f"{key}_ga"], data[f"{key}_kstar"] = run(key, None)
    for key in NOISY:                                                  # Fig. 2: shot noise
        for S in SHOTS:
            th, ga, _ = run(key, S)
            data[f"{key}_S{int(S)}_th"], data[f"{key}_S{int(S)}_ga"] = th, ga
    np.savez(CACHE, **data)
    return dict(np.load(CACHE))


def load():
    if os.path.exists(CACHE):
        d = dict(np.load(CACHE))
        if "auto00_th" in d and list(d["K"]) == K_LIST and int(d["repeats"]) == REPEATS:
            return d
    return compute()


# ============================================================ plotting
def error_panels(series, ylim, yticks):
    """(a) theta, (b) gamma side by side. series: dicts with label, th, ga, color, marker, ls, open."""
    fig, axes = plt.subplots(1, 2, figsize=SIZE, sharey=True, layout="constrained")
    fig.get_layout_engine().set(w_pad=0.04, wspace=0.04)
    for ax, j, tag in zip(axes, [0, 1], [r"(a) $\theta$", r"(b) $\gamma$"]):
        for s in series:
            e = [s["th"], s["ga"]][j]
            ax.fill_between(K_LIST, e[:, 1], e[:, 2], color=s["color"], alpha=0.18, lw=0)
            ax.plot(K_LIST, e[:, 0], marker=s["marker"], color=s["color"], ls=s.get("ls", "-"),
                    mfc="white" if s.get("open") else s["color"], label=s["label"])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(14, 900)
        ax.set_ylim(*ylim)
        ax.xaxis.set_major_locator(FixedLocator([20, 50, 100, 200, 500]))
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_major_locator(FixedLocator(yticks))
        ax.set_xlabel(r"Number of samples $K$")
        ax.text(0.03, 0.04, tag, transform=ax.transAxes, ha="left", va="bottom")
    axes[0].set_ylabel("Relative error")
    return fig, axes


def kstar_line(ax, K, col, label=True):
    ax.axvline(K, color=col, ls=":", lw=1)
    if label:
        ax.text(K * 1.03, 0.97, r"$K^*$", color=col, transform=ax.get_xaxis_transform(), ha="left", va="top")


def fig_controls(d):
    series = [dict(label=lab, th=d[f"{k}_th"], ga=d[f"{k}_ga"], color=c, marker=m)
              for k, (lab, _, _, c, m) in PROTOCOLS.items()]
    fig, (a, b) = error_panels(series, (1e-16, 1e2), [10.0 ** p for p in range(-16, 4, 4)])
    for k, (_, _, _, c, _) in PROTOCOLS.items():
        if d[f"{k}_kstar"] > 0:
            for ax in (a, b):
                kstar_line(ax, d[f"{k}_kstar"], c, label=(ax is a))
    b.legend(loc="center right")

    # inset: an example pulse sequence for "drive both"
    ins = a.inset_axes([0.47, 0.25, 0.46, 0.37])
    seq = tqs.pulse_sequence(40, tqs.CONTROLS["drive both"], np.random.default_rng(0))
    t = tqs.DT * np.arange(len(seq) + 1)
    for level, base in [(1, 1.6), (2, 0.0)]:                          # u_1 drawn above u_2
        u = tqs.ALPHA * (seq == level)
        ins.step(t, base + np.append(u, u[-1]), where="post", color=PROTOCOLS["drive2"][3], lw=0.9)
    ins.set_xlim(0, t[-1])
    ins.set_ylim(-0.35, 2.95)
    ins.set_yticks([0.5, 2.1], [r"$u_2$", r"$u_1$"])
    ins.set_xticks([0, 10, 20])
    ins.set_xlabel(r"$\omega_1 t$", labelpad=1)
    ins.tick_params(axis="y", length=0, pad=2)
    ins.tick_params(axis="x", length=2, pad=1.5)
    ins.tick_params(top=False, right=False, which="both")
    fig.savefig("fig1_controls.pdf")
    fig.savefig("fig1_controls.png")
    print("saved fig1_controls.pdf/.png")


def fig_shots(d):
    series = []
    for k in NOISY:
        lab, _, _, c, m = PROTOCOLS[k]
        for S, ls, hollow in zip(SHOTS, ["-", "--"], [False, True]):
            series.append(dict(label=lab, th=d[f"{k}_S{int(S)}_th"], ga=d[f"{k}_S{int(S)}_ga"],
                               color=c, marker=m, ls=ls, open=hollow))
    fig, (a, b) = error_panels(series, (1e-4, 1e4), [10.0 ** p for p in range(-4, 5, 2)])
    for k in NOISY:
        for ax in (a, b):
            kstar_line(ax, d[f"{k}_kstar"], PROTOCOLS[k][3], label=(ax is a))
    # legend 1 (panel a): protocol = colour + marker; legend 2 (panel b): shots = line style
    hp = [Line2D([], [], color=PROTOCOLS[k][3], marker=PROTOCOLS[k][4], label=PROTOCOLS[k][0]) for k in NOISY]
    hs = [Line2D([], [], color="0.3", ls=ls, marker="o", mfc=mf, label=rf"$S = 10^{{{int(round(np.log10(S)))}}}$")
          for S, ls, mf in zip(SHOTS, ["-", "--"], ["0.3", "white"])]
    a.legend(handles=hp, loc="upper right")
    b.legend(handles=hs, loc="upper right")
    fig.savefig("fig2_shots.pdf")
    fig.savefig("fig2_shots.png")
    print("saved fig2_shots.pdf/.png")


if __name__ == "__main__":
    d = load()
    print("K*:", {k: int(d[f"{k}_kstar"]) for k in PROTOCOLS})
    fig_controls(d)
    fig_shots(d)
