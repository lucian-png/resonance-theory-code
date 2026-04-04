#!/usr/bin/env python3
"""
Script 121 — CAMB Geometry Verification
=========================================
Is the universe open or flat?

Test 1: Explicit open universe in CAMB
Test 2: What did Friday's run actually compute?
Test 3: Three-model comparison (F1 ΛCDM, F2 Ghost Flat, F3 Ghost Open)
Test 4: Curvature degeneracy across observables
Test 5: τ-modified open (if needed)

The fork in the road. The most important calculation.

Author: Lucian Randolph & Claude Anthro Randolph
Date: April 4, 2026
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import camb
import os
import warnings
warnings.filterwarnings('ignore')

DELTA = 4.669201609
ALPHA = 2.502907875
LN_DELTA = np.log(DELTA)

OMEGA_B_H2 = 0.02237
OMEGA_C_H2_LCDM = 0.1200
OMEGA_C_H2_GHOST = 0.116910   # α²×Ωb×h² - Ωbh²
H0 = 67.36
T_CMB = 2.7255
N_EFF = 3.046
NS = 0.9649
AS = 2.1e-9
TAU_REION = 0.0544

OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')

print('=' * 72)
print('  Script 121 — CAMB Geometry Verification')
print('  Is the Universe Open or Flat?')
print(f'  α²×Ωb h² = {OMEGA_B_H2 + OMEGA_C_H2_GHOST:.6f}')
print(f'  Ghost Ωch² = {OMEGA_C_H2_GHOST:.6f}')
print(f'  ΛCDM Ωch² = {OMEGA_C_H2_LCDM}')
print('=' * 72)


def run_camb_model(label, ombh2, omch2, omk, H0_val=H0,
                   print_derived=True):
    """Run CAMB with explicit parameters and return results."""
    print(f'\n  --- {label} ---')
    print(f'  Input: ombh2={ombh2}, omch2={omch2}, omk={omk}, H0={H0_val}')

    pars = camb.CAMBparams()
    pars.set_cosmology(
        H0=H0_val, ombh2=ombh2, omch2=omch2,
        mnu=0.06, omk=omk, tau=TAU_REION,
        TCMB=T_CMB, nnu=N_EFF)
    pars.InitPower.set_params(As=AS, ns=NS)
    pars.set_for_lmax(2500, lens_potential_accuracy=0)

    try:
        results = camb.get_results(pars)
        derived = results.get_derived_params()

        # Get the power spectrum
        powers = results.get_cmb_power_spectra(pars, CMB_unit='muK')
        cl_tt = powers['total'][:, 0]

        # Extract what CAMB actually used internally
        cp = pars  # the params object has the computed values
        # Compute Ωm, ΩΛ, Ωk from CAMB's perspective
        h2 = (H0_val/100)**2
        Om = (ombh2 + omch2 + 0.06 * 0.0006/0.06) / h2  # approximate
        # Actually use derived params
        age = derived.get('age', 0)
        rdrag = derived.get('rdrag', 0)
        theta_star = derived.get('thetastar', 0)
        DA_star = derived.get('DAstar', 0)  # comoving angular diameter dist

        if print_derived:
            print(f'  CAMB computed:')
            print(f'    Age: {age:.3f} Gyr')
            print(f'    r_drag: {rdrag:.2f} Mpc')
            print(f'    theta* (100×): {theta_star:.6f}')
            if DA_star:
                print(f'    D_A*: {DA_star:.2f} Mpc')
            print(f'    Ωk (input): {omk}')

            # Compute what ΩΛ CAMB used
            # In CAMB: ΩΛ = 1 - Ωm - Ωr - Ωk (for flat: Ωk=0)
            Om_total = (ombh2 + omch2) / h2 + 0.06/(93.14 * h2)
            Or_total = 4.15e-5 / h2 * (T_CMB/2.7255)**4 * (1 + N_EFF * 7/8 * (4/11)**(4/3))
            OL_computed = 1.0 - Om_total - Or_total - omk
            print(f'    Ωm (computed): {Om_total:.6f}')
            print(f'    Ωr (computed): {Or_total:.6f}')
            print(f'    ΩΛ (= 1-Ωm-Ωr-Ωk): {OL_computed:.6f}')

        # Find peaks
        peaks = []
        for i in range(100, min(len(cl_tt)-1, 2000)):
            if cl_tt[i] > cl_tt[i-1] and cl_tt[i] > cl_tt[i+1] and cl_tt[i] > 50:
                peaks.append((i, cl_tt[i]))

        if print_derived and peaks:
            print(f'  Peaks:')
            for j, (ell, ht) in enumerate(peaks[:6]):
                print(f'    Peak {j+1}: ℓ = {ell}, height = {ht:.1f} μK²')

        return {
            'label': label,
            'cl_tt': cl_tt,
            'peaks': peaks,
            'age': age,
            'rdrag': rdrag,
            'theta_star': theta_star,
            'OL_computed': 1.0 - (ombh2+omch2)/h2 - omk,  # simplified
            'derived': derived,
        }

    except Exception as e:
        print(f'  FAILED: {e}')
        return {'label': label, 'error': str(e)}


# ================================================================
# TEST 1: EXPLICIT OPEN UNIVERSE
# ================================================================
print('\n' + '=' * 60)
print('  TEST 1: Explicit Open Universe in CAMB')
print('=' * 60)

# Ghost matter content, EXPLICIT open curvature, NO dark energy
h2 = (H0/100)**2
Om_ghost = (OMEGA_B_H2 + OMEGA_C_H2_GHOST) / h2
Ok_ghost = 1.0 - Om_ghost  # curvature fills the rest (ignoring radiation)
print(f'  Ωm(ghost) = {Om_ghost:.6f}')
print(f'  Ωk(ghost) = 1 - Ωm = {Ok_ghost:.6f}')
print(f'  (This should force ΩΛ ≈ 0)')

R1 = run_camb_model(
    'TEST 1: Ghost Open (explicit Ωk)',
    ombh2=OMEGA_B_H2, omch2=OMEGA_C_H2_GHOST,
    omk=Ok_ghost)

# ================================================================
# TEST 2: WHAT DID FRIDAY'S RUN COMPUTE?
# ================================================================
print('\n' + '=' * 60)
print('  TEST 2: Friday\'s CAMB Run (What Actually Happened)')
print('=' * 60)

# Friday's run: we specified ombh2, omch2, H0, omk=0
# This means CAMB assumed FLAT and computed ΩΛ as closure
R2 = run_camb_model(
    'TEST 2: Friday\'s Run (Ghost Flat, Ωk=0)',
    ombh2=OMEGA_B_H2, omch2=OMEGA_C_H2_GHOST,
    omk=0.0)

# ================================================================
# TEST 3: THREE-MODEL COMPARISON
# ================================================================
print('\n' + '=' * 60)
print('  TEST 3: Three-Model Comparison')
print('=' * 60)

F1 = run_camb_model(
    'F1: Standard ΛCDM',
    ombh2=OMEGA_B_H2, omch2=OMEGA_C_H2_LCDM,
    omk=0.0)

F2 = run_camb_model(
    'F2: Ghost Flat (α²×Ωb, flat, with Λ)',
    ombh2=OMEGA_B_H2, omch2=OMEGA_C_H2_GHOST,
    omk=0.0)

F3 = run_camb_model(
    'F3: Ghost Open (α²×Ωb, open, NO Λ)',
    ombh2=OMEGA_B_H2, omch2=OMEGA_C_H2_GHOST,
    omk=Ok_ghost)

# ================================================================
# TEST 4: COMPARISON TABLE
# ================================================================
print('\n' + '=' * 60)
print('  TEST 4: Complete Comparison')
print('=' * 60)

models = [F1, F2, F3]
labels = ['F1 (ΛCDM)', 'F2 (Ghost Flat)', 'F3 (Ghost Open)']

print(f'\n  {"Observable":30s}', end='')
for lab in labels:
    print(f'  {lab:>18s}', end='')
print()
print(f'  {"-"*30}', end='')
for _ in labels:
    print(f'  {"-"*18}', end='')
print()

# Peak 1 position
print(f'  {"ℓ₁ (first peak)":30s}', end='')
for m in models:
    if 'peaks' in m and m['peaks']:
        print(f'  {m["peaks"][0][0]:18d}', end='')
    else:
        print(f'  {"FAILED":>18s}', end='')
print()

# Peak 1 height
print(f'  {"Peak 1 height [μK²]":30s}', end='')
for m in models:
    if 'peaks' in m and m['peaks']:
        print(f'  {m["peaks"][0][1]:18.1f}', end='')
    else:
        print(f'  {"—":>18s}', end='')
print()

# R₁₂
print(f'  {"R₁₂ (peak 1/peak 2)":30s}', end='')
for m in models:
    if 'peaks' in m and len(m['peaks']) >= 2:
        r12 = m['peaks'][0][1] / m['peaks'][1][1]
        print(f'  {r12:18.4f}', end='')
    else:
        print(f'  {"—":>18s}', end='')
print()

# Sound horizon
print(f'  {"r_drag [Mpc]":30s}', end='')
for m in models:
    if 'rdrag' in m:
        print(f'  {m["rdrag"]:18.2f}', end='')
    else:
        print(f'  {"—":>18s}', end='')
print()

# Age
print(f'  {"Age [Gyr]":30s}', end='')
for m in models:
    if 'age' in m:
        print(f'  {m["age"]:18.3f}', end='')
    else:
        print(f'  {"—":>18s}', end='')
print()

# theta*
print(f'  {"100×θ*":30s}', end='')
for m in models:
    if 'theta_star' in m:
        print(f'  {m["theta_star"]:18.6f}', end='')
    else:
        print(f'  {"—":>18s}', end='')
print()

# ΩΛ
print(f'  {"ΩΛ (computed)":30s}', end='')
for m in models:
    if 'OL_computed' in m:
        print(f'  {m["OL_computed"]:18.6f}', end='')
    else:
        print(f'  {"—":>18s}', end='')
print()

# F2 vs F3 comparison
if ('peaks' in F2 and F2['peaks'] and
    'peaks' in F3 and F3['peaks']):
    print(f'\n  F2 vs F3 COMPARISON:')
    ell1_F2 = F2['peaks'][0][0]
    ell1_F3 = F3['peaks'][0][0]
    print(f'    ℓ₁: F2 = {ell1_F2}, F3 = {ell1_F3}, '
          f'difference = {ell1_F3 - ell1_F2} '
          f'({(ell1_F3 - ell1_F2)/ell1_F2 * 100:+.1f}%)')

    if len(F2['peaks']) >= 3 and len(F3['peaks']) >= 3:
        for i in range(min(len(F2['peaks']), len(F3['peaks']), 6)):
            l2, h2_p = F2['peaks'][i]
            l3, h3_p = F3['peaks'][i]
            print(f'    Peak {i+1}: F2 ℓ={l2} ht={h2_p:.0f}, '
                  f'F3 ℓ={l3} ht={h3_p:.0f}, '
                  f'Δℓ={l3-l2} ({(l3-l2)/l2*100:+.1f}%), '
                  f'Δht={h3_p-h2_p:.0f} ({(h3_p-h2_p)/h2_p*100:+.1f}%)')

# ================================================================
# FIGURES
# ================================================================
print('\n--- Generating Figures ---')

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# (a) CMB TT power spectra — all three models
ax = axes[0, 0]
ell = np.arange(2, 2500)
colors = {'F1': 'b', 'F2': 'r', 'F3': 'g'}
for m, lab, c, ls in [(F1, 'F1: ΛCDM', 'b', '-'),
                       (F2, 'F2: Ghost Flat', 'r', '--'),
                       (F3, 'F3: Ghost Open', 'g', ':')]:
    if 'cl_tt' in m and len(m['cl_tt']) > 2500:
        Dl = ell * (ell+1) * m['cl_tt'][2:2500] / (2*np.pi)
        ax.plot(ell, Dl, c+ls, linewidth=2, label=lab, alpha=0.8)

ax.set_xlabel('Multipole ℓ', fontsize=11)
ax.set_ylabel('D_ℓ [μK²]', fontsize=11)
ax.set_title('(a) CMB TT: ΛCDM vs Ghost Flat vs Ghost Open', fontsize=12)
ax.legend(fontsize=10)
ax.set_xlim(2, 2500)
ax.grid(True, alpha=0.3)

# (b) Ratio to ΛCDM
ax = axes[0, 1]
if 'cl_tt' in F1 and 'cl_tt' in F2 and 'cl_tt' in F3:
    cl_F1 = F1['cl_tt'][2:2500]
    cl_F2 = F2['cl_tt'][2:2500]
    cl_F3 = F3['cl_tt'][2:2500]
    safe = np.maximum(cl_F1, 1e-10)
    ax.plot(ell, cl_F2/safe, 'r-', linewidth=1.5,
            label='F2/F1 (Ghost Flat / ΛCDM)')
    ax.plot(ell, cl_F3/safe, 'g-', linewidth=1.5,
            label='F3/F1 (Ghost Open / ΛCDM)')
    ax.axhline(y=1.0, color='blue', linestyle='--', alpha=0.5)
    ax.set_xlabel('Multipole ℓ', fontsize=11)
    ax.set_ylabel('C_ℓ / C_ℓ(ΛCDM)', fontsize=11)
    ax.set_title('(b) Spectral Ratios', fontsize=12)
    ax.legend(fontsize=10)
    ax.set_xlim(2, 2500)
    ax.set_ylim(0, 3)
    ax.grid(True, alpha=0.3)

# (c) Low-ℓ zoom
ax = axes[1, 0]
ell_lo = np.arange(2, 500)
for m, lab, c, ls in [(F1, 'F1: ΛCDM', 'b', '-'),
                       (F2, 'F2: Ghost Flat', 'r', '--'),
                       (F3, 'F3: Ghost Open', 'g', ':')]:
    if 'cl_tt' in m and len(m['cl_tt']) > 500:
        Dl = ell_lo * (ell_lo+1) * m['cl_tt'][2:500] / (2*np.pi)
        ax.plot(ell_lo, Dl, c+ls, linewidth=2, label=lab, alpha=0.8)

ax.set_xlabel('Multipole ℓ', fontsize=11)
ax.set_ylabel('D_ℓ [μK²]', fontsize=11)
ax.set_title('(c) Low-ℓ Zoom: First Three Peaks', fontsize=12)
ax.legend(fontsize=10)
ax.set_xlim(2, 500)
ax.grid(True, alpha=0.3)

# (d) Decision summary
ax = axes[1, 1]
ax.axis('off')
ax.set_title('(d) The Verdict', fontsize=14, fontweight='bold')

if 'peaks' in F3 and F3['peaks']:
    ell1_F3 = F3['peaks'][0][0]
    if abs(ell1_F3 - 220) < 20:
        verdict = (
            f'F3 (Ghost Open): ℓ₁ = {ell1_F3}\n'
            f'F2 (Ghost Flat): ℓ₁ = {F2["peaks"][0][0] if F2.get("peaks") else "?"}\n'
            f'F1 (ΛCDM):       ℓ₁ = {F1["peaks"][0][0] if F1.get("peaks") else "?"}\n\n'
            f'VERDICT: OPEN UNIVERSE WORKS\n'
            f'Dark energy IS curvature.'
        )
    elif ell1_F3 > 280:
        verdict = (
            f'F3 (Ghost Open): ℓ₁ = {ell1_F3}\n'
            f'F2 (Ghost Flat): ℓ₁ = {F2["peaks"][0][0] if F2.get("peaks") else "?"}\n'
            f'F1 (ΛCDM):       ℓ₁ = {F1["peaks"][0][0] if F1.get("peaks") else "?"}\n\n'
            f'VERDICT: CMB RULES OUT OPEN\n'
            f'Ghost model is FLAT with τ(z).'
        )
    else:
        verdict = (
            f'F3 (Ghost Open): ℓ₁ = {ell1_F3}\n'
            f'F2 (Ghost Flat): ℓ₁ = {F2["peaks"][0][0] if F2.get("peaks") else "?"}\n'
            f'F1 (ΛCDM):       ℓ₁ = {F1["peaks"][0][0] if F1.get("peaks") else "?"}\n\n'
            f'VERDICT: INTERMEDIATE\n'
            f'Further analysis needed.'
        )
else:
    verdict = 'F3 computation failed.\nCheck CAMB parameters.'

ax.text(0.1, 0.5, verdict, fontsize=13, fontfamily='monospace',
        verticalalignment='center',
        bbox=dict(boxstyle='round', facecolor='lightyellow',
                  edgecolor='red', linewidth=2))

plt.tight_layout()
fig_path = os.path.join(OUTDIR, 'fig11_geometry_verification.png')
plt.savefig(fig_path, dpi=200, bbox_inches='tight')
plt.close()
print(f'  Saved: {fig_path}')

# ================================================================
# FINAL VERDICT
# ================================================================
print('\n' + '=' * 72)
print('  THE VERDICT')
print('=' * 72)

if 'peaks' in F2 and F2['peaks'] and 'peaks' in F3 and F3['peaks']:
    ell1_F1 = F1['peaks'][0][0] if F1.get('peaks') else None
    ell1_F2 = F2['peaks'][0][0]
    ell1_F3 = F3['peaks'][0][0]

    print(f'\n  F1 (ΛCDM):       ℓ₁ = {ell1_F1}')
    print(f'  F2 (Ghost Flat):  ℓ₁ = {ell1_F2}')
    print(f'  F3 (Ghost Open):  ℓ₁ = {ell1_F3}')
    print()

    if ell1_F3 and abs(ell1_F3 - 220) < 20:
        print(f'  ═══════════════════════════════════════════')
        print(f'  THE UNIVERSE IS OPEN.')
        print(f'  Dark energy is curvature.')
        print(f'  Ωk = 1 - α²×Ωb ≈ {Ok_ghost:.4f}')
        print(f'  ΩΛ = 0.')
        print(f'  ═══════════════════════════════════════════')
    elif ell1_F3 and ell1_F3 > 280:
        print(f'  ═══════════════════════════════════════════')
        print(f'  THE CMB RULES OUT THE OPEN UNIVERSE.')
        print(f'  The Ghost model is FLAT with τ(z).')
        print(f'  Dark energy is replaced by time emergence,')
        print(f'  not by curvature.')
        print(f'  ═══════════════════════════════════════════')
    else:
        print(f'  Result is intermediate. Further analysis needed.')

print()
print(f'  TEST 2 ANSWER: Friday\'s run used Ωk = 0 (flat).')
print(f'  CAMB computed ΩΛ as closure. Friday\'s result was')
print(f'  Ghost Flat (F2), not Ghost Open (F3).')
print(f'  The peak confirmation (sub-1.3%) is for the FLAT model.')
print('=' * 72)
