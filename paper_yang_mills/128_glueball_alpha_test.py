#!/usr/bin/env python3
"""
Script 128 — Glueball Mass Ratios: Hunting for α
==================================================
Test whether the Feigenbaum spatial scaling constant α = 2.503
appears in the SU(3) glueball mass spectrum from lattice QCD.

The mass gap paper claims the glueball spectrum should follow
cascade level spacing related to α. If successive glueball
masses relate to α, that's immediate evidence for the cascade
interpretation. If not, we need to know before making the claim.

Data sources:
  - Morningstar & Peardon (1999): "The glueball spectrum from
    an anisotropic lattice study" PRD 60, 034509
  - Chen et al. (2006): "Glueball spectrum and matrix elements
    on anisotropic lattices" PRD 73, 014516
  - Meyer (2005): "Glueball regge trajectories" hep-lat/0508002
  - Athenodorou & Teper (2020): updated SU(3) spectrum

Author: Lucian Randolph & Claude Anthro Randolph
Date: April 6, 2026
"""

import numpy as np
import os

DELTA = 4.669201609
ALPHA = 2.502907875
LAMBDA_R = DELTA / ALPHA
LN_DELTA = np.log(DELTA)
LN_ALPHA = np.log(ALPHA)

print('=' * 72)
print('  Script 128 — Glueball Mass Ratios: Hunting for α')
print(f'  α = {ALPHA}   δ = {DELTA}   λᵣ = {LAMBDA_R:.6f}')
print(f'  α² = {ALPHA**2:.4f}   √α = {np.sqrt(ALPHA):.4f}')
print(f'  δ/α = λᵣ = {LAMBDA_R:.4f}')
print('=' * 72)

# ================================================================
# GLUEBALL SPECTRUM DATA — SU(3) Pure Gauge
# ================================================================
# From Morningstar & Peardon (1999) and Chen et al. (2006)
# Masses in units of the string tension √σ ≈ 440 MeV
# and converted to MeV using r₀ scale setting
#
# The key result is from Morningstar & Peardon (1999) Table XIV
# and Chen et al. (2006) Table IV, using r₀ = 0.5 fm scale.
#
# J^PC  Mass (MeV)  Error (MeV)  Source
# 0++   1710        50-80        Morningstar (1999), Chen (2006)
# 2++   2390        100-120      Morningstar (1999), Chen (2006)
# 0-+   2560        120-150      Morningstar (1999), Chen (2006)
# 0++*  2670        180          Morningstar (1999) (first excited 0++)
# 2-+   3040        150          Morningstar (1999)
# 3++   3550        170          Morningstar (1999)
# 1+-   2940        140          Morningstar (1999)
# 2--   3100        200          Meyer (2005)
# 0--   3640        200          Meyer (2005)

# Using Morningstar & Peardon (1999) values in units of r₀⁻¹
# r₀ ≈ 0.5 fm → r₀⁻¹ ≈ 395 MeV
# Their Table XIV gives masses in units of r₀⁻¹:
# 0++: 4.329 ± 0.041  → 1710 MeV
# 2++: 6.065 ± 0.058  → 2396 MeV
# 0-+: 6.326 ± 0.065  → 2499 MeV
# 0++*: 6.771 ± 0.092 → 2675 MeV
# 2-+: 7.534 ± 0.083  → 2976 MeV
# 3++: 8.842 ± 0.089  → 3493 MeV
# 1+-: 7.314 ± 0.090  → 2889 MeV

# Using r₀⁻¹ units (dimensionless, cleaner for ratio analysis)
glueballs_r0 = [
    ('0++',   4.329, 0.041, 'Ground state scalar'),
    ('2++',   6.065, 0.058, 'Tensor'),
    ('0-+',   6.326, 0.065, 'Pseudoscalar'),
    ('0++*',  6.771, 0.092, 'Excited scalar'),
    ('1+-',   7.314, 0.090, 'Axial vector'),
    ('2-+',   7.534, 0.083, 'Pseudo-tensor'),
    ('3++',   8.842, 0.089, 'Spin-3'),
]

# Also: masses from Chen et al. (2006) in MeV (using different scale setting)
glueballs_MeV = [
    ('0++',   1710,  50, 'Ground state'),
    ('2++',   2390, 120, 'Tensor'),
    ('0-+',   2560, 120, 'Pseudoscalar'),
    ('0++*',  2670, 180, 'Excited scalar'),
    ('1+-',   2940, 140, 'Axial vector'),
    ('2-+',   3040, 150, 'Pseudo-tensor'),
    ('3++',   3550, 170, 'Spin-3'),
]

print('\n--- SU(3) Glueball Spectrum (Morningstar & Peardon 1999) ---')
print(f'  {"State":8s}  {"Mass/r₀⁻¹":>10s}  {"Mass (MeV)":>12s}  Description')
print(f'  {"-"*8}  {"-"*10}  {"-"*12}  {"-"*20}')
for (st1, m1, e1, d1), (st2, m2, e2, d2) in zip(glueballs_r0, glueballs_MeV):
    print(f'  {st1:8s}  {m1:10.3f}  {m2:12d}  {d1}')

# ================================================================
# TEST 1: SUCCESSIVE MASS RATIOS
# ================================================================
print('\n' + '=' * 60)
print('  TEST 1: Successive Mass Ratios')
print('=' * 60)

masses_r0 = [m for _, m, _, _ in glueballs_r0]
masses_MeV = [m for _, m, _, _ in glueballs_MeV]
names = [s for s, _, _, _ in glueballs_r0]

print('\n  Ratios of successive masses (M_[n+1]/M_[n]):')
print(f'  {"Ratio":20s}  {"Value":>8s}  {"vs α":>8s}  {"vs √α":>8s}  '
      f'{"vs δ/α":>8s}  {"vs δ^(1/3)":>10s}')
print(f'  {"-"*20}  {"-"*8}  {"-"*8}  {"-"*8}  {"-"*8}  {"-"*10}')

for i in range(len(masses_r0) - 1):
    ratio = masses_r0[i+1] / masses_r0[i]
    dev_alpha = (ratio - ALPHA) / ALPHA * 100
    dev_sqrt = (ratio - np.sqrt(ALPHA)) / np.sqrt(ALPHA) * 100
    dev_lr = (ratio - LAMBDA_R) / LAMBDA_R * 100
    dev_d13 = (ratio - DELTA**(1/3)) / DELTA**(1/3) * 100
    print(f'  {names[i]}→{names[i+1]:8s}  {ratio:8.4f}  '
          f'{dev_alpha:+7.1f}%  {dev_sqrt:+7.1f}%  '
          f'{dev_lr:+7.1f}%  {dev_d13:+9.1f}%')

# ================================================================
# TEST 2: ALL MASSES RELATIVE TO GROUND STATE
# ================================================================
print('\n' + '=' * 60)
print('  TEST 2: All Masses Relative to Ground State (0++)')
print('=' * 60)

m0 = masses_r0[0]  # ground state mass

print(f'  Ground state 0++: {m0:.3f} r₀⁻¹')
print(f'\n  {"State":8s}  {"M/M₀":>8s}  {"log_α(M/M₀)":>12s}  '
      f'{"Nearest n":>10s}  {"α^n":>8s}  {"Dev":>8s}')
print(f'  {"-"*8}  {"-"*8}  {"-"*12}  {"-"*10}  {"-"*8}  {"-"*8}')

for st, m, e, d in glueballs_r0:
    ratio = m / m0
    if ratio > 1:
        log_alpha = np.log(ratio) / LN_ALPHA
    else:
        log_alpha = 0
    n_nearest = round(log_alpha * 2) / 2  # nearest half-integer
    alpha_n = ALPHA**n_nearest if n_nearest > 0 else 1.0
    dev = (ratio - alpha_n) / alpha_n * 100 if alpha_n > 0 else 0
    print(f'  {st:8s}  {ratio:8.4f}  {log_alpha:12.4f}  '
          f'{n_nearest:10.1f}  {alpha_n:8.4f}  {dev:+7.1f}%')

# ================================================================
# TEST 3: MASSES RELATIVE TO ΛQCD
# ================================================================
print('\n' + '=' * 60)
print('  TEST 3: Masses as Cascade Levels Above ΛQCD')
print('=' * 60)

# ΛQCD ≈ 200 MeV (MSbar scheme) or ~330 MeV (lattice definition)
LAMBDA_QCD = 200  # MeV (MSbar)
LAMBDA_QCD_LAT = 330  # MeV (lattice)

print(f'  ΛQCD = {LAMBDA_QCD} MeV (MSbar), {LAMBDA_QCD_LAT} MeV (lattice)')

for lqcd, label in [(LAMBDA_QCD, 'MSbar'), (LAMBDA_QCD_LAT, 'lattice')]:
    print(f'\n  Using ΛQCD = {lqcd} MeV ({label}):')
    print(f'  {"State":8s}  {"M (MeV)":>10s}  {"M/ΛQCD":>8s}  '
          f'{"log_α":>8s}  {"log_δ":>8s}  {"n (α)":>6s}  {"n (δ)":>6s}')
    for st, m, e, d in glueballs_MeV:
        ratio = m / lqcd
        la = np.log(ratio) / LN_ALPHA
        ld = np.log(ratio) / LN_DELTA
        print(f'  {st:8s}  {m:10d}  {ratio:8.3f}  '
              f'{la:8.3f}  {ld:8.3f}  {la:6.2f}  {ld:6.2f}')

# ================================================================
# TEST 4: SEARCH FOR ANY FEIGENBAUM COMBINATION
# ================================================================
print('\n' + '=' * 60)
print('  TEST 4: Systematic Search for Feigenbaum Combinations')
print('=' * 60)

# For each mass ratio M_i/M_0, check all combinations of α and δ
# up to power 4

feig_combos = []
for pa in np.arange(-2, 4, 0.5):
    for pd in np.arange(-2, 4, 0.5):
        val = ALPHA**pa * DELTA**pd
        if 0.5 < val < 10:
            label = f'α^{pa:.1f}·δ^{pd:.1f}'
            feig_combos.append((val, label))

# Also add pure powers
for p in np.arange(0, 4, 0.25):
    feig_combos.append((ALPHA**p, f'α^{p:.2f}'))
    feig_combos.append((DELTA**p, f'δ^{p:.2f}'))
    if p > 0:
        feig_combos.append((LAMBDA_R**p, f'λᵣ^{p:.2f}'))

print(f'\n  For each glueball ratio M/M₀, find closest Feigenbaum combo:')
print(f'  {"State":8s}  {"M/M₀":>8s}  {"Best match":>20s}  {"Value":>8s}  {"Dev":>8s}')
print(f'  {"-"*8}  {"-"*8}  {"-"*20}  {"-"*8}  {"-"*8}')

for st, m, e, d in glueballs_r0:
    ratio = m / m0
    if ratio <= 1.001:
        continue
    best_dev = 999
    best_label = ''
    best_val = 0
    for val, label in feig_combos:
        dev = abs(ratio - val) / ratio * 100
        if dev < best_dev:
            best_dev = dev
            best_label = label
            best_val = val
    print(f'  {st:8s}  {ratio:8.4f}  {best_label:>20s}  '
          f'{best_val:8.4f}  {best_dev:+7.2f}%')

# ================================================================
# TEST 5: SU(N) SCALING — DOES THE GAP SCALE WITH N?
# ================================================================
print('\n' + '=' * 60)
print('  TEST 5: SU(N) Glueball Ground State Scaling')
print('=' * 60)

# From Lucini, Teper & Wenger (2004) and Athenodorou & Teper (2020)
# 0++ glueball mass in units of string tension √σ
# SU(N):
su_n_data = [
    (2, 3.55, 0.07, 'Teper (1998)'),
    (3, 4.329, 0.041, 'Morningstar (1999)'),
    (4, 4.14, 0.06, 'Lucini (2004)'),
    (5, 4.10, 0.07, 'Lucini (2004)'),
    (6, 4.06, 0.10, 'Lucini (2004)'),
    (8, 4.05, 0.12, 'Lucini (2004)'),
]

print(f'  0++ glueball mass in units of √σ:')
print(f'  {"SU(N)":>6s}  {"M/√σ":>8s}  {"Error":>8s}  Source')
for N, m, e, src in su_n_data:
    print(f'  SU({N})  {m:8.3f}  {e:8.3f}  {src}')

# Check ratios
print(f'\n  SU(N)/SU(3) ratios:')
m_su3 = 4.329
for N, m, e, src in su_n_data:
    ratio = m / m_su3
    print(f'  SU({N})/SU(3) = {ratio:.4f}')

# Large-N limit: mass should approach a constant (Casimir scaling)
# Check if the approach to the limit involves α
print(f'\n  Approach to large-N limit:')
print(f'  M(N)/M(∞) ≈ 1 + c/N² for Casimir scaling')
for N, m, e, src in su_n_data:
    if N >= 3:
        correction = m / 4.05 - 1  # using SU(8) as proxy for N→∞
        print(f'  SU({N}): correction = {correction:.4f}, '
              f'1/N² = {1/N**2:.4f}, '
              f'ratio = {correction * N**2:.3f}')

# ================================================================
# SUMMARY
# ================================================================
print('\n' + '=' * 72)
print('  SUMMARY: Where Is α?')
print('=' * 72)
print(f'')
print(f'  Successive mass ratios (Test 1):')
print(f'    Range: {min(masses_r0[i+1]/masses_r0[i] for i in range(len(masses_r0)-1)):.3f} '
      f'to {max(masses_r0[i+1]/masses_r0[i] for i in range(len(masses_r0)-1)):.3f}')
print(f'    α = {ALPHA:.3f} — NOT in the successive ratios')
print(f'    The ratios are 1.03 to 1.40 — too small for α')
print(f'')
print(f'  Masses relative to ground state (Test 2):')
print(f'    Range: 1.0 to {max(m/m0 for _, m, _, _ in glueballs_r0):.3f}')
print(f'    α = {ALPHA:.3f} — would predict M/M₀ = 2.503 for level 1')
print(f'    Actual second state: M/M₀ = {masses_r0[1]/masses_r0[0]:.3f}')
print(f'')
print(f'  SU(N) scaling (Test 5):')
print(f'    Large-N behavior: Casimir scaling (1/N²)')
print(f'    No obvious α dependence on N')
print(f'')
print(f'  HONEST ASSESSMENT:')
print(f'    The glueball spectrum does NOT show clean α-scaling')
print(f'    in the obvious places (successive ratios, M/M₀).')
print(f'    The spectrum is more complex than simple cascade levels.')
print(f'    This does NOT invalidate the mass gap argument —')
print(f'    the cascade amplification of the instanton mass parameter')
print(f'    is about EXISTENCE and POSITIVITY, not about the specific')
print(f'    spectrum of states above the gap.')
print(f'    But the prediction that mass ratios follow α needs')
print(f'    to be REMOVED or QUALIFIED in the paper.')
print('=' * 72)
