#!/usr/bin/env python3
"""
Script 129 — δ^(1/4) Sensitivity Test
========================================
Does M(0⁻⁺)/M(0⁺⁺) = δ^(1/4) survive across multiple
independent lattice QCD determinations?

The Feynman test: if the ratio shifts significantly across
different lattice calculations, it's a coincidence.

Data from:
  - Morningstar & Peardon (1999) PRD 60, 034509
  - Chen et al. (2006) PRD 73, 014516
  - Meyer & Teper (2005) PLB 605, 344
  - Lucini, Teper & Wenger (2004) JHEP 0406:012
  - Athenodorou & Teper (2020) JHEP 2020, 172

Then: systematic test of ALL simple Feigenbaum combinations
against ALL glueball mass ratios from ALL sources.

Author: Lucian Randolph & Claude Anthro Randolph
Date: April 6, 2026
"""

import numpy as np

DELTA = 4.669201609
ALPHA = 2.502907875
LAMBDA_R = DELTA / ALPHA
LN_DELTA = np.log(DELTA)
LN_ALPHA = np.log(ALPHA)

print('=' * 72)
print('  Script 129 — δ^(1/4) Sensitivity Test')
print(f'  δ^(1/4) = {DELTA**0.25:.6f}')
print(f'  α^(1/2) = {ALPHA**0.5:.6f}')
print('=' * 72)

# ================================================================
# MULTI-SOURCE GLUEBALL DATA
# ================================================================
# Masses in units of r₀⁻¹ (Sommer parameter, r₀ ≈ 0.5 fm)
# This is the standard dimensionless unit for lattice comparison.
#
# Different groups report in different units. We normalize to
# r₀⁻¹ where possible, or use mass RATIOS which are unit-independent.

# Source 1: Morningstar & Peardon (1999) — quenched, anisotropic
# Table XIV of PRD 60, 034509
# Masses in units of r₀⁻¹:
MP99 = {
    '0++':  (4.329, 0.041),
    '2++':  (6.065, 0.058),
    '0-+':  (6.326, 0.065),
    '0++*': (6.771, 0.092),
    '1+-':  (7.314, 0.090),
    '2-+':  (7.534, 0.083),
    '3++':  (8.842, 0.089),
}

# Source 2: Chen et al. (2006) — improved anisotropic action
# PRD 73, 014516
# Masses in units of r₀⁻¹ (from their Table IV):
Chen06 = {
    '0++':  (4.16, 0.11),
    '2++':  (5.85, 0.10),
    '0-+':  (6.33, 0.13),
    '0++*': (6.52, 0.17),
}

# Source 3: Meyer & Teper (2005) — quenched SU(3)
# Reported masses in units of string tension √σ.
# Convert using √σ ≈ 0.44 GeV, r₀ √σ ≈ 1.195
# So M/r₀⁻¹ = M/√σ × 1.195
MT05_sigma = {
    '0++':  (3.55, 0.07),  # in units of √σ
    '2++':  (5.02, 0.09),
    '0-+':  (5.18, 0.15),
}
# Convert to r₀⁻¹ units
r0_sigma = 1.195
MT05 = {k: (v[0]*r0_sigma, v[1]*r0_sigma) for k, v in MT05_sigma.items()}

# Source 4: Lucini, Teper & Wenger (2004) — SU(3) from their multi-N study
# JHEP 0406:012 — reported in units of √σ
LTW04_sigma = {
    '0++':  (4.329/r0_sigma, 0.041/r0_sigma),  # Using MP99 as SU(3) reference
}
# Actually Lucini et al. report SU(3) masses in √σ units:
# 0++: 4.21 ± 0.11 (from their Table 1)
LTW04_sigma_direct = {
    '0++': (4.21, 0.11),
}
LTW04 = {k: (v[0]*r0_sigma, v[1]*r0_sigma)
          for k, v in LTW04_sigma_direct.items()}

print('\n--- Multi-Source Glueball Masses (r₀⁻¹ units) ---')
print(f'  {"State":8s}  {"MP99":>12s}  {"Chen06":>12s}  {"MT05":>12s}')
print(f'  {"-"*8}  {"-"*12}  {"-"*12}  {"-"*12}')
for state in ['0++', '2++', '0-+', '0++*']:
    mp = f'{MP99[state][0]:.3f}±{MP99[state][1]:.3f}' if state in MP99 else '—'
    ch = f'{Chen06[state][0]:.2f}±{Chen06[state][1]:.2f}' if state in Chen06 else '—'
    mt = f'{MT05[state][0]:.3f}±{MT05[state][1]:.3f}' if state in MT05 else '—'
    print(f'  {state:8s}  {mp:>12s}  {ch:>12s}  {mt:>12s}')

# ================================================================
# TEST A: δ^(1/4) SENSITIVITY
# ================================================================
print('\n' + '=' * 60)
print('  TEST A: M(0⁻⁺)/M(0⁺⁺) vs δ^(1/4)')
print(f'  Target: δ^(1/4) = {DELTA**0.25:.6f}')
print('=' * 60)

target = DELTA**0.25

# Compute ratio from each source
ratios_0mp_0pp = []

for label, data in [('Morningstar & Peardon (1999)', MP99),
                     ('Chen et al. (2006)', Chen06),
                     ('Meyer & Teper (2005)', MT05)]:
    if '0-+' in data and '0++' in data:
        m_ps, e_ps = data['0-+']
        m_sc, e_sc = data['0++']
        ratio = m_ps / m_sc
        # Error propagation
        err = ratio * np.sqrt((e_ps/m_ps)**2 + (e_sc/m_sc)**2)
        dev = (ratio - target) / target * 100
        ratios_0mp_0pp.append((ratio, err))
        print(f'  {label}:')
        print(f'    M(0⁻⁺)/M(0⁺⁺) = {ratio:.4f} ± {err:.4f}')
        print(f'    δ^(1/4) = {target:.4f}')
        print(f'    Deviation: {dev:+.2f}%')
        print(f'    Within 1σ? {"YES" if abs(ratio - target) < err else "NO"}')
        print()

# Spread analysis
if len(ratios_0mp_0pp) >= 2:
    vals = [r[0] for r in ratios_0mp_0pp]
    spread = max(vals) - min(vals)
    print(f'  SPREAD ANALYSIS:')
    print(f'    Ratios: {[f"{v:.4f}" for v in vals]}')
    print(f'    Spread: {spread:.4f}')
    print(f'    δ^(1/4) = {target:.4f}')
    print(f'    Stability threshold: 0.01')
    if spread > 0.01:
        print(f'    ✗ SPREAD > 0.01 — ratio is NOT stable across sources')
    else:
        print(f'    ✓ SPREAD < 0.01 — ratio is stable')

# ================================================================
# TEST B: ALL FEIGENBAUM COMBOS vs ALL RATIOS
# ================================================================
print('\n' + '=' * 60)
print('  TEST B: Systematic Feigenbaum Combination Search')
print('=' * 60)

# Generate all simple Feigenbaum combinations
combos = {}
for n in [0.2, 0.25, 1/3, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
    combos[f'δ^({n:.3f})'] = DELTA**n
    combos[f'α^({n:.3f})'] = ALPHA**n
# Special
combos['δ/α (λᵣ)'] = LAMBDA_R
combos['ln(δ)'] = LN_DELTA
combos['ln(α)'] = LN_ALPHA
combos['α/δ^(1/2)'] = ALPHA / DELTA**0.5
combos['δ^(1/2)/α'] = DELTA**0.5 / ALPHA
combos['(δ/α)^(1/2)'] = LAMBDA_R**0.5
combos['α·ln(α)'] = ALPHA * LN_ALPHA

# All unique mass ratios from MP99
states = list(MP99.keys())
masses = {s: MP99[s][0] for s in states}
errors = {s: MP99[s][1] for s in states}

print(f'\n  All mass ratios M_i/M_j with sub-2% Feigenbaum matches:')
print(f'  {"Ratio":20s}  {"Value":>8s}  {"Best combo":>20s}  '
      f'{"Combo val":>10s}  {"Dev":>8s}')
print(f'  {"-"*20}  {"-"*8}  {"-"*20}  {"-"*10}  {"-"*8}')

matches_sub2 = []

for i, si in enumerate(states):
    for j, sj in enumerate(states):
        if i >= j:
            continue
        ratio = masses[si] / masses[sj] if masses[sj] != 0 else 0
        if ratio < 1:
            ratio = 1 / ratio
            label = f'{sj}/{si}'
        else:
            label = f'{si}/{sj}'

        if ratio <= 1.001 or ratio > 5:
            continue

        # Find best Feigenbaum match
        best_dev = 999
        best_name = ''
        best_val = 0
        for name, val in combos.items():
            if val <= 0.5 or val > 5:
                continue
            dev = abs(ratio - val) / val * 100
            if dev < best_dev:
                best_dev = dev
                best_name = name
                best_val = val

        if best_dev < 2.0:
            matches_sub2.append((label, ratio, best_name, best_val, best_dev))
            print(f'  {label:20s}  {ratio:8.4f}  {best_name:>20s}  '
                  f'{best_val:10.4f}  {best_dev:7.2f}%')

print(f'\n  Total sub-2% matches: {len(matches_sub2)}')

# ================================================================
# TEST C: WHICH MATCHES SURVIVE ACROSS SOURCES?
# ================================================================
print('\n' + '=' * 60)
print('  TEST C: Cross-Source Stability of Sub-2% Matches')
print('=' * 60)

# For each sub-2% match from MP99, check if the same ratio
# is within 2% in Chen06 and MT05

for label, ratio_mp, combo_name, combo_val, dev_mp in matches_sub2:
    states_in_ratio = label.split('/')
    if len(states_in_ratio) != 2:
        continue
    s1, s2 = states_in_ratio

    print(f'\n  {label} = {ratio_mp:.4f} ≈ {combo_name} = {combo_val:.4f}')

    for src_name, src_data in [('Chen06', Chen06), ('MT05', MT05)]:
        if s1 in src_data and s2 in src_data:
            m1 = src_data[s1][0]
            m2 = src_data[s2][0]
            r = m1/m2 if m1 > m2 else m2/m1
            dev = (r - combo_val) / combo_val * 100
            stable = abs(dev) < 2.0
            print(f'    {src_name}: ratio = {r:.4f}, '
                  f'dev from {combo_name} = {dev:+.2f}% '
                  f'{"✓" if stable else "✗"}')
        else:
            print(f'    {src_name}: states not available')

# ================================================================
# TEST D: GROUND STATE vs ΛQCD — CASCADE LEVEL
# ================================================================
print('\n' + '=' * 60)
print('  TEST D: Ground State as Cascade Level')
print('=' * 60)

# M(0++) / ΛQCD across different ΛQCD definitions
for lqcd, label in [(200, 'MSbar'), (250, 'MOM'), (330, 'lattice')]:
    ratio = 1710 / lqcd  # using MeV values
    la = np.log(ratio) / LN_ALPHA
    ld = np.log(ratio) / LN_DELTA

    # Check if the cascade level is close to an integer or half-integer
    la_near = round(la * 2) / 2
    ld_near = round(ld * 2) / 2
    dev_a = abs(la - la_near)
    dev_d = abs(ld - ld_near)

    print(f'  ΛQCD = {lqcd} MeV ({label}):')
    print(f'    M(0++)/ΛQCD = {ratio:.3f}')
    print(f'    log_α = {la:.4f} (nearest half-int: {la_near}, dev: {dev_a:.3f})')
    print(f'    log_δ = {ld:.4f} (nearest half-int: {ld_near}, dev: {dev_d:.3f})')
    print()

# ================================================================
# SUMMARY
# ================================================================
print('=' * 72)
print('  RESULTS')
print('=' * 72)
print()
print(f'  TEST A (δ^(1/4) stability):')
if len(ratios_0mp_0pp) >= 2:
    vals = [r[0] for r in ratios_0mp_0pp]
    print(f'    Ratios across sources: {[f"{v:.3f}" for v in vals]}')
    print(f'    Spread: {max(vals)-min(vals):.3f}')
    if max(vals) - min(vals) > 0.05:
        print(f'    VERDICT: LARGE SPREAD — δ^(1/4) is source-dependent')
    elif max(vals) - min(vals) > 0.01:
        print(f'    VERDICT: MODERATE SPREAD — δ^(1/4) uncertain')
    else:
        print(f'    VERDICT: STABLE — δ^(1/4) survives')

print(f'\n  TEST B (sub-2% Feigenbaum matches):')
print(f'    Found {len(matches_sub2)} matches at sub-2%')
for label, ratio, combo, val, dev in matches_sub2:
    print(f'      {label} = {ratio:.4f} ≈ {combo} ({dev:.2f}%)')

print(f'\n  NEXT STEP:')
print(f'    Proceed to instanton mass parameter calculation (Link 2)')
print(f'    regardless of spectral results. The mass gap proof is about')
print(f'    EXISTENCE and POSITIVITY, not about the specific spectrum.')
print('=' * 72)
