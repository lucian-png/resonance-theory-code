# Resonance Theory — Computational Code & Visualizations

**Author: Lucian Randolph**

20 papers in 5 days. This repository contains the Python code used to generate all computational visualizations, formatted documents, and control group validation for the Resonance Theory paper series and the Lucian Method.

---

## The Lucian Method

The Lucian Method (formally: Mono-Variable Extreme Scale Analysis, MESA) is a mathematical methodology for revealing the geometric structure of nonlinear coupled equation systems. The method isolates a single driving variable, holds all other parameters fixed, extends the driving variable across extreme orders of magnitude, and observes the geometric morphology of coupled variables as they respond.

**Calibration**: The method was validated against Mandelbrot's equation z → z² + c — a known fractal. All five fractal criteria were confirmed. The instrument is calibrated.

**Application**: The calibrated method was applied to Einstein's field equations and the Yang-Mills equations of the Standard Model. Both classified as fractal geometric.

**Full method paper and results**: [lucian.co](https://lucian.co)

---

## The Papers

| # | Title | Field | DOI |
|---|-------|-------|-----|
| 0 | **The Field That Forgot Itself** | Philosophy of Science | [10.5281/zenodo.18764176](https://doi.org/10.5281/zenodo.18764176) |
| I | **The Bridge Was Already Built** | Physics | [10.5281/zenodo.18716086](https://doi.org/10.5281/zenodo.18716086) |
| II | **One Light, Every Scale** | Physics | [10.5281/zenodo.18723787](https://doi.org/10.5281/zenodo.18723787) |
| III | **Seven Problems, One Framework** | Physics | [10.5281/zenodo.18724585](https://doi.org/10.5281/zenodo.18724585) |
| IV | **The Resonance of Reality** | Consciousness | [10.5281/zenodo.18725698](https://doi.org/10.5281/zenodo.18725698) |
| V | **The Lucian Method** | Methodology | [10.5281/zenodo.18764623](https://doi.org/10.5281/zenodo.18764623) |
| VI | **How to Break Resonance Theory** | Falsifiability | [10.5281/zenodo.18750736](https://doi.org/10.5281/zenodo.18750736) |
| VII | **Solve These Better** | Open Problems | [10.5281/zenodo.18756292](https://doi.org/10.5281/zenodo.18756292) |
| VIII | **Why Does Time Have a Direction?** | Physics | [10.5281/zenodo.18764576](https://doi.org/10.5281/zenodo.18764576) |
| IX | **Cancer as Fractal Emergence** | Medicine | [10.5281/zenodo.18735634](https://doi.org/10.5281/zenodo.18735634) |
| X | **The Fractal Genome** | Genomics | [10.5281/zenodo.18735638](https://doi.org/10.5281/zenodo.18735638) |
| XI | **The Millennium Problem** | Mathematics | [10.5281/zenodo.18759223](https://doi.org/10.5281/zenodo.18759223) |
| XII | **The Boltzmann Fractal** | Thermodynamics | [10.5281/zenodo.18772441](https://doi.org/10.5281/zenodo.18772441) |
| XIII | **Why You Will Resist This** | Psychology of Science | [10.5281/zenodo.18757190](https://doi.org/10.5281/zenodo.18757190) |
| XIV | **The Universal Diagnostic** | Psychology | [10.5281/zenodo.18725703](https://doi.org/10.5281/zenodo.18725703) |
| XV | **The Interior Method** | Psychology | [10.5281/zenodo.18733515](https://doi.org/10.5281/zenodo.18733515) |
| XVI | **One Geometry — Resonance Unification** | Unification | [10.5281/zenodo.18776715](https://doi.org/10.5281/zenodo.18776715) |

Papers XVII, XIX, and XX are completed but withheld pending IP protection.

---

## Repository Structure

```
resonance-theory-code/
├── paper_I_the_bridge/              # Paper I: The Bridge Was Already Built
│   ├── 01_schwarzschild_across_scales.py
│   ├── 02_bkl_harmonic_dynamics.py
│   ├── 03_harmonic_evolution.py
│   ├── 04_quantum_bridge.py
│   ├── 05_generate_word_doc.py
│   └── figures/                     # 13 generated figures
│
├── paper_II_one_light/              # Paper II: One Light, Every Scale
│   ├── 06_standard_model_part1.py
│   ├── 07_standard_model_part2.py
│   ├── 08_generate_paper_two_doc.py
│   ├── 09_cosmos_part1.py
│   ├── 10_cosmos_part2.py
│   └── figures/                     # 20 generated figures
│
├── paper_III_the_room/              # Paper III: Seven Problems, One Framework
│   ├── 11_quantum_foundations.py
│   ├── 12_time_blackholes.py
│   ├── 13_particles_masses.py
│   ├── 14_graveyard.py
│   ├── 15_generate_paper_three_doc.py
│   └── figures/                     # 10 generated figures
│
├── paper_IV_resonance_of_reality/   # Paper IV: The Resonance of Reality
│   └── 17_generate_paper_iv_doc.py
│
├── paper_IX_cancer_fractal_emergence/  # Paper IX: Cancer as Fractal Emergence
│   └── 18_generate_paper_ix_doc.py
│
├── paper_X_fractal_genome/          # Paper X: The Fractal Genome
│   └── 19_generate_paper_x_doc.py
│
├── paper_XIV_universal_diagnostic/  # Paper XIV: The Universal Diagnostic
│   └── 16_generate_paper_xiv_doc.py
│
└── lucian_method/                   # The Lucian Method — Calibration
    ├── 20_mandelbrot_control_group.py
    ├── 21_generate_mandelbrot_control_paper.py
    ├── 22_generate_lucian_method_v2.py
    └── figures/
        ├── fig20_mandelbrot_control_group.png
        └── fig21_mandelbrot_extreme_range.png
```

---

## What the Code Does

### The Lucian Method — Calibration & Control Group

The `lucian_method/` directory contains the foundational validation of the Lucian Method:

- **`20_mandelbrot_control_group.py`** — Applies the Lucian Method to Mandelbrot's equation z → z² + c as a control group. Generates two 6-panel figures (12 panels total) testing all five fractal criteria: self-similarity, power-law scaling, fractal dimension, Feigenbaum bifurcation, and Lyapunov structure. The method correctly identifies a known fractal as fractal geometric. The instrument is calibrated.

- **`21_generate_mandelbrot_control_paper.py`** — Generates the formal control group paper (.docx) with embedded figures explaining the calibration results.

- **`22_generate_lucian_method_v2.py`** — Generates the complete Lucian Method paper v2 (.docx) with the full narrative: method → calibration → application to Einstein and Yang-Mills → proof.

### Visualization Scripts (Papers I–III)

The physics papers include computational visualizations demonstrating that Einstein's field equations and the Yang-Mills gauge field equations satisfy the five classification criteria for fractal geometric equations. The scripts generate figures showing:

- **Paper I:** Schwarzschild metric self-similarity across 40+ orders of magnitude, BKL/Kasner fractal dynamics near singularities, harmonic structure of gravitational solutions, quantum-classical bridge visualizations
- **Paper II:** Standard Model classification against all five fractal criteria, cosmological implications (dark matter, dark energy, vacuum energy, cosmic web, BAO), grand unification landscape
- **Paper III:** Quantum measurement problem, entanglement, Bell inequality, arrow of time, black hole information paradox, matter-antimatter asymmetry, Strong CP problem, neutrino masses

### Document Generators

Each paper with code in this repository has a Python script that generates a formatted Word document (.docx) using the `python-docx` library. These produce publication-ready documents with consistent formatting: A4 page, Times New Roman 11pt, 1.15 line spacing, formal academic structure.

---

## Requirements

```
numpy
matplotlib
python-docx
```

Install with:
```bash
pip install numpy matplotlib python-docx
```

---

## Running the Code

Each script is self-contained. To generate the Mandelbrot control group figures:

```bash
cd lucian_method
python 20_mandelbrot_control_group.py
```

This produces:
- `fig20_mandelbrot_control_group.png` — 6-panel control group analysis
- `fig21_mandelbrot_extreme_range.png` — 6-panel extreme range validation

To generate figures for Papers I–III:
```bash
cd paper_I_the_bridge
python 01_schwarzschild_across_scales.py
python 02_bkl_harmonic_dynamics.py
python 03_harmonic_evolution.py
python 04_quantum_bridge.py
```

Figures are saved to the same directory (or a `figures/` subfolder) as PNG files.

---

## Framework Summary

Resonance Theory proposes that the fundamental equations of physics — Einstein's field equations and the Yang-Mills gauge field equations — are fractal geometric equations, classifiable under the mathematical taxonomy developed in the 1970s. The Lucian Method was created to reveal this structure, calibrated against Mandelbrot's equation as a control group, and applied to both Einstein's equations and the Standard Model. All three systems produce the same five fractal signatures.

This classification resolves the apparent incompatibility between quantum mechanics and general relativity by revealing them as different harmonic scales of a single continuous fractal geometric structure.

The framework extends to psychology, consciousness, cancer biology, genomics, thermodynamics, and mathematics — each domain analyzed through the same fractal geometric lens, each producing consistent results.

---

## License

All code is provided under Creative Commons Attribution 4.0 International (CC BY 4.0). The papers are published on Zenodo with DOIs listed above.

---

## Citation

If referencing this work, please cite the relevant paper(s) by DOI. For the code specifically:

```
Randolph, L. (2026). Resonance Theory — Computational Code & Visualizations.
GitHub: https://github.com/lucian-png/resonance-theory-code
```
