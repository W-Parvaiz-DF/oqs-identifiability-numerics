# Identifiability numerics: driven two-qubit system

Code for the numerical example (Sec. V, Figs. 1–2) of
*Identifiability of Autonomous and Controlled Open Quantum Systems*
(W. Parvaiz, J. Aspman, K. Wodecki, G. Korpas, J. Marecek), arXiv:2501.05270.

Two coupled qubits with local decay, excitation and dephasing are probed with
on/off σx pulses and full state tomography at uniformly spaced times. The script
estimates the system matrix from the samples, reconstructs the Hamiltonian
parameters θ and the Kossakowski matrix γ with M⁻¹, and plots the relative error
against the number of samples K.

## Files

| File | Purpose |
| --- | --- |
| `two_qubit_sampling.py` | Model, structure constants and M (Eq. 25), pulses, simulation, shot noise, estimator, sweep. Running it gives quick-look figures. |
| `paper_figures.py` | Publication figures (Figs. 1 and 2). Imports `two_qubit_sampling.py`. |
| `figures/` | The figures as used in the paper. |

## Usage

```bash
pip install -r requirements.txt
python paper_figures.py        # about 1 minute; writes fig1_controls.pdf/.png and fig2_shots.pdf/.png
```

Results are cached in `paper_data.npz`; delete it to recompute. Settings
(pulse amplitude, sampling interval, shot counts, number of runs, font) are at
the top of each script.

## Model and protocol

- Units of ω₁; ω₂ = 1.3, δ = 0.2; (g⁻, g⁺, gᶻ) = (0.05, 0.01, 0.02) for qubit 1 and (0.03, 0.02, 0.04) for qubit 2.
- Initial state |00⟩ (or |++⟩ for the autonomous comparison); drives H_k = σx on qubit k.
- Pulses: amplitude 1, one drive on at a time, random widths of 1–5 sampling intervals, Δt = 0.5.
- Measurement: each of the 15 Pauli expectation values estimated from S shots (binomial noise), fresh preparation per time point.
- Estimation: per pulse setting, least squares for the one-step map, principal matrix log, subtract the known drive term, average; then (θ, γ) = M⁻¹(A, β).
- Statistics: median and interquartile range over 50 random pulse sequences per K.

Tested with Python 3.13, numpy 2.5.3, scipy 1.18.1, matplotlib 3.11.2.
