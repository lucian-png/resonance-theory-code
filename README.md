# Resonance Theory — Computational Code & Visualizations

**Author: Lucian Randolph**

28 papers. One law. This repository contains the Python code used to generate all computational visualizations, formatted documents, control group validation, and the Lucian Law falsification protocol for the Resonance Theory paper series.

**Full papers and results**: [lucian.co](https://lucian.co)

---

## The Papers

### 0 — The Lucian Law

The foundation. Four papers establishing the law and its consequences.

| Paper | Title | Field | DOI |
|-------|-------|-------|-----|
| 1 | **The Lucian Law** | Framework | [10.5281/zenodo.14908930](https://doi.org/10.5281/zenodo.14908930) |
| 2 | **The Geometric Necessity of Feigenbaum's Constant** | Mathematics / Physics | [10.5281/zenodo.14908962](https://doi.org/10.5281/zenodo.14908962) |
| 3 | **The Full Extent of the Lucian Law** | Physics / Cosmology | [10.5281/zenodo.14908978](https://doi.org/10.5281/zenodo.14908978) |
| 4 | **The Inflationary Parameters** | Astrophysics / Cosmology | [10.5281/zenodo.14909742](https://doi.org/10.5281/zenodo.14909742) |

### 1 — The Kill

| Paper | Title | Field | DOI |
|-------|-------|-------|-----|
| 5 | **Slaying the Twin Dragons** | Astrophysics / Cosmology | — |

### 2 — The Evidence Base

| Paper | Title | Field | DOI |
|-------|-------|-------|-----|
| 6 | **One Geometry — Resonance Unification** | Unification | [10.5281/zenodo.18776715](https://doi.org/10.5281/zenodo.18776715) |
| 7 | **The Lucian Universe** | Astrophysics / GR | [10.5281/zenodo.18791921](https://doi.org/10.5281/zenodo.18791921) |
| 8 | **Dual Attractor Basins** | Astrophysics | — |
| 9 | **Validating the Method** | Methodology | [10.5281/zenodo.18764623](https://doi.org/10.5281/zenodo.18764623) |

### 3 — The Hard Sciences

Thirteen evidence papers (original numbering).

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

### 4 — The Soft Sciences

| # | Title | Field | DOI |
|---|-------|-------|-----|
| XIII | **Why You Will Resist This** | Psychology of Science | [10.5281/zenodo.18757190](https://doi.org/10.5281/zenodo.18757190) |
| XIV | **The Universal Diagnostic** | Psychology | [10.5281/zenodo.18725703](https://doi.org/10.5281/zenodo.18725703) |
| XV | **The Interior Method** | Psychology | [10.5281/zenodo.18733515](https://doi.org/10.5281/zenodo.18733515) |

### 5 — Withheld for IP

Three papers, unlisted.

---

## Repository Structure

```
resonance-theory-code/
│
├── lucian_law_trilogy/              # Papers 1–3: The Lucian Law
│   ├── generate_lucian_law_paper.py
│   ├── generate_feigenbaum_paper.py
│   ├── generate_full_extent_paper.py
│   ├── make_paper_figures.py
│   ├── make_full_extent_figures.py
│   └── paper_figures/
│
├── feigenbaum_derivation/           # Paper 2: Feigenbaum Constant Derivation
│   ├── 42_feigenbaum_derivation.py
│   ├── figures/                     # 10 panels + composite
│   └── paper_figures/
│
├── falsification_protocol/          # Paper 1: Lucian Law Falsification Protocol
│   ├── 36–41 scripts               # 6 independent tests
│   ├── LUCIAN_LAW_FALSIFICATION_PROTOCOL.md
│   └── figures/
│
├── paper_IV_inflationary_parameters/ # Paper 4: The Inflationary Parameters
│   ├── 43_inflation_derivation.py
│   ├── 44_generate_paper_four_doc.py
│   └── figures/
│
├── paper_V_twin_dragons/            # Paper 5: Slaying the Twin Dragons
│   ├── 44_dark_matter_dragon.py     # 175 SPARC galaxies, 7 panels
│   ├── 45_dark_energy_twin.py       # 1,701 Pantheon+ SNe, 3 panels
│   ├── 46_generate_twin_dragons_doc.py
│   └── figures/                     # 12 panels + 2 composites
│
├── paper_XXI_chladni_universe/      # Paper 7: The Lucian Universe
│   ├── 28–34 scripts               # Analysis, figures, v1/v2/v3 generators
│   └── figures/
│
├── paper_XXII_dual_attractor/       # Paper 8: Dual Attractor Basins
│   ├── 34_dual_attractor_basins.py
│   ├── 35_generate_paper_xxii_doc.py
│   └── figures/
│
├── gaia_confirmation/               # Gaia DR3 Empirical Confirmation
│   ├── 36_gaia_dr3_confirmation.py  # 50,000 stars, p < 10⁻³⁰⁰
│   └── figures/
│
├── lucian_method/                   # Paper 9: Validating the Method
│   ├── 20_mandelbrot_control_group.py
│   ├── 21_generate_mandelbrot_control_paper.py
│   ├── 22_generate_lucian_method_v2.py
│   └── figures/
│
├── paper_I_the_bridge/              # Paper I: The Bridge Was Already Built
│   ├── 01–05 scripts
│   └── figures/
│
├── paper_II_one_light/              # Paper II: One Light, Every Scale
│   ├── 06–10 scripts
│   └── figures/
│
├── paper_III_the_room/              # Paper III: Seven Problems, One Framework
│   ├── 11–15 scripts
│   └── figures/
│
├── paper_IV_resonance_of_reality/   # Paper IV: The Resonance of Reality
│   └── 17_generate_paper_iv_doc.py
│
├── paper_IX_cancer_fractal_emergence/ # Paper IX: Cancer as Fractal Emergence
│   └── 18_generate_paper_ix_doc.py
│
├── paper_X_fractal_genome/          # Paper X: The Fractal Genome
│   └── 19_generate_paper_x_doc.py
│
└── paper_XIV_universal_diagnostic/  # Paper XIV: The Universal Diagnostic
    └── 16_generate_paper_xiv_doc.py
```

---

## Key Results

### The Lucian Law (Papers 1–4)

The Lucian Law states that any nonlinear system driven across extreme dynamic range produces dual attractor basins separated by depleted transition zones, with basin boundaries spaced by the Feigenbaum constant.

- **Paper 1:** Nineteen equation systems. Zero refutations. Six falsification tests, all passed.
- **Paper 2:** Derives δ = 4.669201609... from three geometric constraints. No dynamical iteration required.
- **Paper 3:** The Big Bang as a dual attractor threshold event. Inflation as the basin transition curve.
- **Paper 4:** Stellar geometry and cosmological constants. Gaia DR3: 50,000 stars, p = 10⁻³¹⁰. Three inflationary parameters confirmed.

### Slaying the Twin Dragons (Paper 5)

Dark matter and dark energy as twin artifacts of assuming time is fully expressed everywhere.

- **Dark Matter:** 175 SPARC galaxies. Feigenbaum z=6 in acceleration space at p = 3 × 10⁻¹¹. Time emergence τ < 0.5 at edges of 117/175 galaxies. a₀ = c·H₀/(δ+α) within 3.8% of fitted value.
- **Dark Energy:** 1,701 Pantheon+ Type Ia supernovae. Zero-parameter τ(z) model fits within 3.2% of ΛCDM. Fitted z_transition = 1.449, ln(δ) prediction = 1.541 (6.4% error).

### The Lucian Universe (Paper 7)

Five-cascade harmonic structure in the interior Schwarzschild metric. Feigenbaum sub-harmonic spectrum. Dual attractor basins in astrophysical densities — active cores at 0.53–0.66×, passive objects at 1.05–1.66×. Gaia prediction confirmed at p = 10⁻³¹⁰.

---

## Requirements

```
numpy
scipy
matplotlib
python-docx
Pillow
astropy        # For Gaia DR3 queries (optional)
astroquery     # For Gaia DR3 queries (optional)
```

Install with:
```bash
pip install numpy scipy matplotlib python-docx Pillow
# Optional, for Gaia DR3 data retrieval:
pip install astropy astroquery
```

---

## Running the Code

Each script is self-contained. Examples:

```bash
# Paper 2: Feigenbaum derivation (10 panels + composite)
cd feigenbaum_derivation
python 42_feigenbaum_derivation.py

# Paper 4: Inflationary parameters
cd paper_IV_inflationary_parameters
python 43_inflation_derivation.py

# Paper 5: Twin Dragons — Dark Matter
cd paper_V_twin_dragons
python 44_dark_matter_dragon.py

# Paper 5: Twin Dragons — Dark Energy
python 45_dark_energy_twin.py

# Paper 5: Generate .docx
python 46_generate_twin_dragons_doc.py

# Paper 7: Lucian Universe figures
cd paper_XXI_chladni_universe
python 28_paper_xxi_figures.py

# Falsification protocol (all six tests)
cd falsification_protocol
python 36_negative_control_linear.py
python 37_nonlinearity_threshold.py
python 38_coupling_topology.py
python 39_counterexample_attempt.py
python 40_blind_prediction.py
python 41_dimensionality_test.py

# Lucian Law trilogy (.docx)
cd lucian_law_trilogy
python generate_lucian_law_paper.py
python generate_feigenbaum_paper.py
python generate_full_extent_paper.py
```

Figures are saved to the same directory (or a `figures/` subfolder) as PNG files.

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
