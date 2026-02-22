# Resonance Theory — Computational Code & Visualizations

**Author: Lucian Randolph**

This repository contains the Python code used to generate all computational visualizations and formatted documents for the Resonance Theory paper series.

---

## The Papers

| # | Title | Field | DOI | Figures |
|---|-------|-------|-----|---------|
| I | **The Bridge Was Already Built** | Physics | [10.5281/zenodo.18716086](https://doi.org/10.5281/zenodo.18716086) | 13 |
| II | **One Light, Every Scale** | Physics | [10.5281/zenodo.18723787](https://doi.org/10.5281/zenodo.18723787) | 20 |
| III | **The Room Is Larger Than We Thought** | Physics | [10.5281/zenodo.18724585](https://doi.org/10.5281/zenodo.18724585) | 10 |
| XXIX | **The Universal Diagnostic** | Psychology | [10.5281/zenodo.18725698](https://doi.org/10.5281/zenodo.18725698) | 0 |
| XXX | **The Resonance of Reality** | Consciousness | [10.5281/zenodo.18725703](https://doi.org/10.5281/zenodo.18725703) | 0 |
| XXXII | **Cancer as Fractal Emergence** | Medicine | [DOI pending] | 0 |
| XXXIII | **The Fractal Genome** | Genomics | [DOI pending] | 0 |

---

## Repository Structure

```
resonance-theory-code/
├── paper_I_the_bridge/           # Resonance Theory I
│   ├── 01_schwarzschild_across_scales.py
│   ├── 02_bkl_harmonic_dynamics.py
│   ├── 03_harmonic_evolution.py
│   ├── 04_quantum_bridge.py
│   ├── 05_generate_word_doc.py
│   └── figures/                  # 13 generated figures
│
├── paper_II_one_light/           # Resonance Theory II
│   ├── 06_standard_model_part1.py
│   ├── 07_standard_model_part2.py
│   ├── 08_generate_paper_two_doc.py
│   ├── 09_cosmos_part1.py
│   ├── 10_cosmos_part2.py
│   └── figures/                  # 20 generated figures
│
├── paper_III_the_room/           # Resonance Theory III
│   ├── 11_quantum_foundations.py
│   ├── 12_time_blackholes.py
│   ├── 13_particles_masses.py
│   ├── 14_graveyard.py
│   ├── 15_generate_paper_three_doc.py
│   └── figures/                  # 10 generated figures
│
├── paper_XXIX_universal_diagnostic/
│   └── 16_generate_paper_xxix_doc.py
│
├── paper_XXX_resonance_of_reality/
│   └── 17_generate_paper_xxx_doc.py
│
├── paper_XXXII_cancer_fractal_emergence/
│   └── 18_generate_paper_xxxii_doc.py
│
└── paper_XXXIII_fractal_genome/
    └── 19_generate_paper_xxxiii_doc.py
```

---

## What the Code Does

### Visualization Scripts (Papers I-III)

The physics papers include computational visualizations demonstrating that Einstein's field equations and the Yang-Mills gauge field equations satisfy the five classification criteria for fractal geometric equations. The scripts generate figures showing:

- **Paper I:** Schwarzschild metric self-similarity across 40+ orders of magnitude, BKL/Kasner fractal dynamics near singularities, harmonic structure of gravitational solutions, quantum-classical bridge visualizations
- **Paper II:** Standard Model classification against all five fractal criteria, cosmological implications (dark matter, dark energy, vacuum energy, cosmic web, BAO), grand unification landscape
- **Paper III:** Quantum measurement problem, entanglement, Bell inequality, arrow of time, black hole information paradox, matter-antimatter asymmetry, Strong CP problem, neutrino masses

### Document Generators (All Papers)

Each paper has a Python script that generates a formatted Word document (.docx) using the `python-docx` library. These produce publication-ready documents with consistent formatting: A4 page, Times New Roman 11pt, 1.15 line spacing, formal academic structure.

---

## Requirements

```
numpy
matplotlib
scipy
python-docx
```

Install with:
```bash
pip install numpy matplotlib scipy python-docx
```

---

## Running the Code

Each script is self-contained. To generate figures for Paper I:

```bash
cd paper_I_the_bridge
python 01_schwarzschild_across_scales.py
python 02_bkl_harmonic_dynamics.py
python 03_harmonic_evolution.py
python 04_quantum_bridge.py
```

To generate a Word document:
```bash
python 05_generate_word_doc.py
```

Figures are saved to the same directory (or a `figures/` subfolder) as PNG files.

---

## Framework Summary

Resonance Theory proposes that the fundamental equations of physics — Einstein's field equations and the Yang-Mills gauge field equations — are fractal geometric equations, classifiable under the mathematical taxonomy developed in the 1970s. This classification resolves the apparent incompatibility between quantum mechanics and general relativity by revealing them as different harmonic scales of a single continuous fractal geometric structure.

The framework extends to psychology (the Dunning-Kruger effect as a fractal observational artifact), consciousness (as a resonant harmonic peak in a fractal information system), cancer (as a fractal emergent phase transition detectable before manifestation), and genomics (non-coding DNA as the fractal architecture of the genome).

---

## License

This code is provided for academic and research purposes. The papers are published on Zenodo with DOIs listed above.

---

## Citation

If referencing this work, please cite the relevant paper(s) by DOI. For the code specifically:

```
Randolph, L. (2026). Resonance Theory — Computational Code & Visualizations.
GitHub: https://github.com/lucian-png/resonance-theory-code
```
