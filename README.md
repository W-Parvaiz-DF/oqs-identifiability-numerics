# Identifiability numerics: driven two-qubit system

Code for the numerical example (Sec. V, Figs. 1–2) of
*Identifiability of Autonomous and Controlled Open Quantum Systems*
(W. Parvaiz, J. Aspman, K. Wodecki, G. Korpas, J. Marecek), arXiv:2501.05270.


Two coupled qubits with local decay, excitation and dephasing are probed with σx
pulses and full state tomography at uniformly spaced times. From the samples the
script:

1. estimates the system matrix;
2. reconstructs the Hamiltonian parameters θ and the Kossakowski matrix γ with M⁻¹;
3. plots the relative error against the number of samples K.

## Files

| File | Purpose |
| --- | --- |
| `two_qubit_sampling.py` | Model, structure constants and M (Eq. 25), pulses, simulation, shot noise, estimator, sweep. Running it gives quick-look figures. |
| `paper_figures.py` | Publication figures (Figs. 1 and 2). Imports `two_qubit_sampling.py`. |
| `log_branch_stats.py` | How often the noisy one-step maps have no real principal logarithm (the numbers quoted in Sec. V). |
| `figures/` | The figures as used in the paper. |

## Usage

```bash
pip install -r requirements.txt
python paper_figures.py        # about 1 minute; writes fig1_controls.pdf/.png and fig2_shots.pdf/.png
python log_branch_stats.py     # about 1 minute; prints the log-branch statistics
```

Results are cached in `paper_data.npz`; delete the file to recompute. The settings are at the top of each script:
- pulse amplitude;
- sampling interval;
- shot counts;
- number of runs;
- font.

## Model and protocol

- **Model.** The Hamiltonian is written in a frame rotating at the drive frequency, so ω_k are detunings.
  - Units of ω₁: ω₂ = 1.3, δ = 0.2.
  - Qubit 1: (g⁻, g⁺, gᶻ) = (0.05, 0.01, 0.02). Qubit 2: (0.03, 0.02, 0.04).
- **Initial state and drives.** The initial state is |00⟩, or |++⟩ for the autonomous comparison. The drives are H_k = σx on qubit k, i.e. resonant drives in the rotating-wave approximation.
- **Pulses.**
  - Amplitude 1, with at most one drive on at a time.
  - The setting (off, drive 1 or drive 2) is drawn uniformly at random for consecutive segments of 1–5 sampling intervals.
  - Sampling interval Δt = 0.5.
  - Pulses switch only at sampling times.
- **Measurement.** Each of the 15 Pauli expectation values is estimated from S shots (binomial noise), with a fresh preparation per time point.
- **Estimation**, for each pulse setting:
  1. Least squares for the one-step map. Below full rank this is the minimum-norm update Φ = 1 + (X₊ − X₋)X₋⁺, which acts as the identity on directions not yet reached.
  2. Principal matrix logarithm. If Φ has eigenvalues on the negative real axis, the logarithm is complex and its real part is used.
  3. Subtract the known drive term.
  
  Then average over settings and compute (θ, γ) = M⁻¹(A, β).
- **K\*.** The first K at which the data of every setting have rank 16, with singular values below 1e-9 treated as zero (absolute tolerance). We report the median over pulse sequences.
- **Statistics.** Median and interquartile range over 50 random pulse sequences per K.

## Reproducibility

All randomness comes from `numpy.random.default_rng`:
- each sweep (one protocol, one shot count) uses seed 0;
- the K\* computation uses seed 1.

Rerunning the scripts therefore reproduces the figures and the statistics exactly, given the same package versions.

Tested with Python 3.13, numpy 2.5.3, scipy 1.18.1, matplotlib 3.11.2.

## Citation

If you use this code, please cite the paper (arXiv:2501.05270).
