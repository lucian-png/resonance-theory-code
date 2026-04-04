#!/usr/bin/env python3
"""
Script 125 — Verification: Is 1-δ⁻² Real or Coincidence?
==========================================================
The Feynman Trap Test

H₀(θ*)/H₀(cascade) = 1/(1-δ⁻²) at 0.017%?
Or is it a beautiful accident?

Test 1: Sensitivity to Ωbh² (THE critical test)
Test 2: Physical derivation — why n=2?
Test 3: Does 0.9541 appear elsewhere?
Test 5: Mathematical consistency — t_age = 14.5 Gyr?
Test 6: Numerical precision — is 0.017% real?

Author: Lucian Randolph & Claude Anthro Randolph
Date: April 4, 2026
"""

import numpy as np
import camb
import os
import warnings
warnings.filterwarnings('ignore')

DELTA = 4.669201609
ALPHA = 2.502907875
LN_DELTA = np.log(DELTA)
LN_ALPHA = np.log(ALPHA)

T_CMB = 2.7255
N_EFF = 3.046
NS = 0.9649
AS = 2.1e-9
TAU_REION = 0.0544
MPC_TO_M = 3.0857e22
GYR_TO_S = 3.156e16

# The target: 1 - δ⁻²
DELTA_INV_SQ = 1.0 / DELTA**2
TARGET = 1.0 - DELTA_INV_SQ

print('=' * 72)
print('  Script 125 — Is 1-δ⁻² Real or Coincidence?')
print(f'  δ = {DELTA}')
print(f'  δ⁻² = {DELTA_INV_SQ:.10f}')
print(f'  1-δ⁻² = {TARGET:.10f}')
print('=' * 72)


def run_ghost_theta(ombh2_val, label=''):
    """Run CAMB with Ghost parameters, solve for H₀ from θ*.
    Returns (H₀_theta, H₀_cascade, ratio, age)."""
    theta_planck = 1.04110 / 100

    omch2_val = ombh2_val * (ALPHA**2 - 1)

    pars = camb.CAMBparams()
    pars.set_cosmology(
        cosmomc_theta=theta_planck,
        ombh2=ombh2_val, omch2=omch2_val,
        mnu=0.06, omk=0, tau=TAU_REION, TCMB=T_CMB, nnu=N_EFF)
    pars.InitPower.set_params(As=AS, ns=NS)
    pars.set_for_lmax(2500, lens_potential_accuracy=0)

    try:
        results = camb.get_results(pars)
        derived = results.get_derived_params()

        H0_theta = pars.H0
        age = derived['age']
        rdrag = derived['rdrag']

        H0_cascade = LN_ALPHA / (age * GYR_TO_S) * MPC_TO_M / 1e3

        ratio = H0_cascade / H0_theta

        return {
            'label': label,
            'ombh2': ombh2_val,
            'omch2': omch2_val,
            'H0_theta': H0_theta,
            'H0_cascade': H0_cascade,
            'ratio': ratio,
            'age': age,
            'rdrag': rdrag,
            'deviation_from_target': (ratio - TARGET) / TARGET * 100,
        }
    except Exception as e:
        return {'label': label, 'error': str(e)}


# ================================================================
# TEST 1: SENSITIVITY TO Ωbh²
# ================================================================
print('\n' + '=' * 60)
print('  TEST 1: Sensitivity to Ωbh²')
print('  (THE critical test — is the ratio ROBUST?)')
print('=' * 60)

# Planck 2018: Ωbh² = 0.02237 ± 0.00015
ombh2_central = 0.02237
ombh2_sigma = 0.00015

# Test at -2σ, -1σ, central, +1σ, +2σ
test_values = [
    (ombh2_central - 2*ombh2_sigma, '-2σ'),
    (ombh2_central - 1*ombh2_sigma, '-1σ'),
    (ombh2_central, 'central'),
    (ombh2_central + 1*ombh2_sigma, '+1σ'),
    (ombh2_central + 2*ombh2_sigma, '+2σ'),
]

# Also test at wider range
test_values.extend([
    (0.02100, 'low extreme'),
    (0.02400, 'high extreme'),
])

print(f'\n  {"Label":>14s}  {"Ωbh²":>10s}  {"H₀(θ*)":>10s}  '
      f'{"H₀(casc)":>10s}  {"Ratio":>10s}  {"1-δ⁻²":>10s}  {"Dev":>8s}')
print(f'  {"-"*14}  {"-"*10}  {"-"*10}  {"-"*10}  {"-"*10}  {"-"*10}  {"-"*8}')

results_t1 = []
for ombh2_val, label in test_values:
    r = run_ghost_theta(ombh2_val, label)
    results_t1.append(r)
    if 'error' not in r:
        print(f'  {label:>14s}  {r["ombh2"]:.5f}  {r["H0_theta"]:10.4f}  '
              f'{r["H0_cascade"]:10.4f}  {r["ratio"]:10.6f}  '
              f'{TARGET:10.6f}  {r["deviation_from_target"]:+7.3f}%')
    else:
        print(f'  {label:>14s}  {ombh2_val:.5f}  FAILED: {r["error"]}')

# Compute the RANGE of the ratio
valid_results = [r for r in results_t1 if 'error' not in r]
if valid_results:
    ratios = [r['ratio'] for r in valid_results]
    ratio_mean = np.mean(ratios)
    ratio_std = np.std(ratios)
    ratio_range = max(ratios) - min(ratios)

    # Just the ±1σ range
    sigma_results = [r for r in valid_results
                     if abs(r['ombh2'] - ombh2_central) <= 1.1 * ombh2_sigma]
    sigma_ratios = [r['ratio'] for r in sigma_results]
    sigma_range = max(sigma_ratios) - min(sigma_ratios)

    print(f'\n  STABILITY ANALYSIS:')
    print(f'    Ratio mean (all): {ratio_mean:.6f}')
    print(f'    Ratio std (all): {ratio_std:.6f}')
    print(f'    Ratio range (full): {ratio_range:.6f}')
    print(f'    Ratio range (±1σ): {sigma_range:.6f}')
    print(f'    Target (1-δ⁻²): {TARGET:.6f}')
    print(f'')

    # THE VERDICT
    if sigma_range < 0.001:
        print(f'    ✓ RATIO IS STABLE within ±1σ of Ωbh²')
        print(f'    The range {sigma_range:.6f} is less than 0.001')
        print(f'    The δ⁻² relationship is ROBUST under parameter variation.')
    else:
        print(f'    ✗ RATIO SHIFTS significantly within ±1σ of Ωbh²')
        print(f'    The range {sigma_range:.6f} exceeds 0.001')
        print(f'    The 0.954 match may be coincidental.')

# ================================================================
# TEST 2: PHYSICAL DERIVATION — WHY n=2?
# ================================================================
print('\n' + '=' * 60)
print('  TEST 2: Physical Derivation — Why n=2?')
print('=' * 60)

print(f'  Paper 37 gravitational stabilization:')
print(f'    Δn_g = 0.90 cascade levels')
print(f'    90% stabilization at n ≈ 2')
print(f'    99% stabilization at n ≈ 3.5')
print(f'')
print(f'  Testing 1 - δ⁻ⁿ for various n:')
for n in [0.5, 0.9, 1.0, 1.5, 1.9, 2.0, 2.1, 2.5, 3.0, 3.5]:
    val = 1.0 - DELTA**(-n)
    dev = (val - TARGET) / TARGET * 100
    marker = ' ← MATCH' if abs(dev) < 0.1 else ''
    print(f'    n = {n:4.1f}: 1-δ⁻ⁿ = {val:.6f}  '
          f'(dev from target: {dev:+.3f}%){marker}')

print(f'\n  n must be EXACTLY 2.0 for the match.')
print(f'  Paper 37 tier completion: Tier 3 (gravity) at level 2.')
print(f'  Structurally motivated? Yes — gravity is the 3rd tier,')
print(f'  completing at cascade level 2.')
print(f'  But this is a POST-HOC connection.')

# ================================================================
# TEST 3: DOES 0.9541 APPEAR ELSEWHERE?
# ================================================================
print('\n' + '=' * 60)
print('  TEST 3: Does 1-δ⁻² Appear Elsewhere?')
print('=' * 60)

# Check various cosmological ratios
comparisons = [
    ('Ghost ΩΛ / ΛCDM ΩΛ', 0.691 / 0.686),
    ('Ghost age / ΛCDM age', 13.858 / 13.797),
    ('Ghost Ωmh² / ΛCDM Ωmh²', 0.1400 / 0.1424),
    ('Planck H₀ / SH0ES H₀', 67.36 / 73.04),
    ('Paper 9 τ_floor', 0.937),
    ('R₁₂(Ghost) / R₁₂(ΛCDM)', 2.2118 / 2.2099),
    ('1 - α⁻²', 1 - 1/ALPHA**2),
    ('1 - (δ/α²)⁻¹', 1 - ALPHA**2/DELTA),
    ('δ/(δ+1)', DELTA/(DELTA+1)),
    ('α²/(α²+1)', ALPHA**2/(ALPHA**2+1)),
]

print(f'  {"Quantity":35s}  {"Value":>10s}  {"1-δ⁻²":>10s}  {"Dev":>8s}')
print(f'  {"-"*35}  {"-"*10}  {"-"*10}  {"-"*8}')
for name, val in comparisons:
    dev = (val - TARGET) / TARGET * 100
    marker = ' ←' if abs(dev) < 2 else ''
    print(f'  {name:35s}  {val:10.6f}  {TARGET:10.6f}  {dev:+7.2f}%{marker}')

# ================================================================
# TEST 5: MATHEMATICAL CONSISTENCY — t_age = 14.5 Gyr?
# ================================================================
print('\n' + '=' * 60)
print('  TEST 5: Mathematical Consistency')
print('=' * 60)

# If H₀(measured) = H₀(cascade)/(1-δ⁻²):
# t_age(cascade_structural) = t_age(measured)/(1-δ⁻²)
t_age_measured = 13.858  # Ghost CAMB age
t_age_structural = t_age_measured / TARGET
print(f'  t_age(measured) = {t_age_measured:.3f} Gyr')
print(f'  t_age(structural) = t_age/(1-δ⁻²) = {t_age_structural:.3f} Gyr')
print(f'  Oldest globular clusters: 12.5 - 14.5 Gyr')
print(f'  {t_age_structural:.1f} Gyr is at the TOP of the stellar range.')
print(f'  If JWST pushes ages above 14 Gyr, this would be supporting.')

# ================================================================
# TEST 6: NUMERICAL PRECISION
# ================================================================
print('\n' + '=' * 60)
print('  TEST 6: Numerical Precision Check')
print('=' * 60)

# Run CAMB with highest precision for central value
pars_hires = camb.CAMBparams()
pars_hires.set_cosmology(
    cosmomc_theta=1.04110/100,
    ombh2=0.02237, omch2=0.02237*(ALPHA**2-1),
    mnu=0.06, omk=0, tau=TAU_REION, TCMB=T_CMB, nnu=N_EFF)
pars_hires.InitPower.set_params(As=AS, ns=NS)
pars_hires.set_for_lmax(3000, lens_potential_accuracy=1)
pars_hires.set_accuracy(AccuracyBoost=2, lSampleBoost=2, lAccuracyBoost=2)

try:
    res_hires = camb.get_results(pars_hires)
    der_hires = res_hires.get_derived_params()

    H0_hi = pars_hires.H0
    age_hi = der_hires['age']
    H0_casc_hi = LN_ALPHA / (age_hi * 1e9 * GYR_TO_S) * MPC_TO_M / 1e3
    ratio_hi = H0_casc_hi / H0_hi

    print(f'  High-precision CAMB run:')
    print(f'    H₀(θ*) = {H0_hi:.6f} km/s/Mpc')
    print(f'    Age = {age_hi:.6f} Gyr')
    print(f'    H₀(cascade) = {H0_casc_hi:.6f} km/s/Mpc')
    print(f'    Ratio = {ratio_hi:.8f}')
    print(f'    Target (1-δ⁻²) = {TARGET:.8f}')
    print(f'    Deviation = {(ratio_hi - TARGET)/TARGET*100:.4f}%')
    print(f'')
    print(f'  Standard precision:')
    central = [r for r in results_t1 if r.get('label') == 'central']
    if central:
        r = central[0]
        print(f'    Ratio = {r["ratio"]:.8f}')
        print(f'    Deviation = {r["deviation_from_target"]:.4f}%')
    print(f'    Precision improvement: '
          f'{abs(ratio_hi-TARGET)/abs(central[0]["ratio"]-TARGET):.2f}×'
          if central and abs(central[0]["ratio"]-TARGET) > 0 else '')
except Exception as e:
    print(f'  High-precision run failed: {e}')

# ================================================================
# COMPLETE VERDICT
# ================================================================
print('\n' + '=' * 72)
print('  VERDICT: REAL OR COINCIDENCE?')
print('=' * 72)

if valid_results:
    print(f'')
    print(f'  TEST 1 (Sensitivity):')
    print(f'    Ratio range within ±1σ of Ωbh²: {sigma_range:.6f}')
    if sigma_range < 0.001:
        print(f'    RESULT: STABLE ✓ (range < 0.001)')
    else:
        print(f'    RESULT: UNSTABLE ✗ (range ≥ 0.001)')
    print(f'')
    print(f'  TEST 2 (Physical derivation):')
    print(f'    n = 2 motivated by gravitational tier completion')
    print(f'    RESULT: PLAUSIBLE but POST-HOC')
    print(f'')
    print(f'  TEST 3 (Appears elsewhere):')
    matches = [(n, v) for n, v in comparisons
               if abs((v-TARGET)/TARGET*100) < 2]
    print(f'    Matches within 2%: {len(matches)}')
    for n, v in matches:
        print(f'      {n}: {v:.6f}')
    print(f'    RESULT: {"CONFIRMED" if len(matches) >= 2 else "NOT CONFIRMED"} '
          f'in other ratios')
    print(f'')
    print(f'  TEST 6 (Precision):')
    print(f'    High-res ratio: {ratio_hi:.8f}')
    print(f'    Target: {TARGET:.8f}')
    print(f'    RESULT: Match at {abs((ratio_hi-TARGET)/TARGET*100):.3f}%')

    print(f'')
    print(f'  ═══════════════════════════════════════════')
    if sigma_range < 0.001:
        print(f'  THE δ⁻² RELATIONSHIP IS ROBUST.')
        print(f'  The ratio H₀(cascade)/H₀(θ*) = 1-δ⁻² is stable')
        print(f'  under parameter variation. Include in paper.')
    else:
        print(f'  THE δ⁻² MATCH IS A COINCIDENCE.')
        print(f'  The ratio shifts with Ωbh². Remove ln(α)/t_age')
        print(f'  from the paper. Report H₀(θ*) only.')
    print(f'  ═══════════════════════════════════════════')
print('=' * 72)
