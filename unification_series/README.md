# Unification Series — Computational Scripts

**Author: Lucian Randolph**

Computational scripts supporting the Unification Series:

> *"The Fractal Geometric Classification of the Fundamental Equations of Physics"*
> (Paper I — submitted to Physical Review E)

> *"The Universal Cascade Across the Quantum-Classical Boundary"*
> (Paper II — submitted to Physical Review Letters / Physical Review A)

These scripts produce the N-convergence tables and Whisper exponent
quantification cited in Papers I and II.

---

## Scripts

| Script | Description | Key Result |
|--------|-------------|------------|
| `82_n_convergence.py` | N-convergence of the quantum phase transition | Proves γˣ = 1.1838 is stable: 0.000% variation from N=30 to N=50 |
| `83_semiclassical_correspondence.py` | Semiclassical emergence threshold analysis | Identifies the quantum-to-classical boundary region |
| `84_whisper_quantified.py` | Whisper exponent quantification | β = −δ = −4.66920…; 0.26% agreement with cascade prediction |

---

## Key Numerical Results Produced

### N-Convergence (Script 82)
- Quantum phase transition location γˣ = χ/κ = **1.1838**
- Quantum tunneling correction **Δγ = 0.738**
- N-convergence from N=30 to N=50: **0.000%** — fully converged, not a Hilbert space truncation artifact
- Hilbert space dimensions swept: N = 15, 20, 25, 30, 35, 40, 50

### Whisper Exponent (Script 84)
- Stroboscopic spread at classical bifurcation points scales as β = **−δ = −4.66920…**
- Drummond-Walls (1980) prediction: β = −0.5 (factor 9.3 error)
- Agreement with cascade prediction: **0.26%**
- The Whisper is the quantum imprint of the cascade's universal constant

---

## System Parameters

All scripts use the rotating-frame driven-dissipative Kerr oscillator:

```
H = Δ a†a + (K/2) a†a(a†a − 1) + F(a† + a)
L = √γ · a  (photon loss)

Δ = −3.0  (detuning)
K =  0.3  (Kerr nonlinearity — C₂ condition)
F =  3.0  (coherent drive amplitude)
```

UCT conditions satisfied:
- **C₁** — Trace-preserving density matrix (dissipative boundedness)
- **C₂** — Kerr nonlinearity (non-degenerate quadratic fold)
- **C₃** — Photon loss to environment (transversal spectral crossing)

---

## Requirements

```bash
pip install numpy scipy matplotlib qutip
```

QuTiP is required for the Lindblad steady-state solver.

---

## Related Scripts (in `paper_birth_of_structure/`)

Scripts 73–80 provide the foundational quantum Kerr computations that
Scripts 82–84 build upon:

- `79_quantum_kerr_cascade.py` — Quantum Kerr bifurcation diagram
- `80_quantum_kerr_parametric.py` — Quantum vs. classical parametric drive
