#!/usr/bin/env python3
"""
Script 116 — Modified Friedmann Equation Feasibility Test
==========================================================
Can Baryons + Radiation + τ(z) Replace ΛCDM?

Four models tested against 1,701 Pantheon+ supernovae:
  A: Standard ΛCDM (benchmark)
  B: Baryons + radiation only (baseline failure)
  C: Paper 9 form (Ωm + τ(z), already validated)
  D: Full replacement (Ωb + τ(z), THE TEST)

Plus: Ghost of α cross-check, CMB first peak, z_t refit.

Author: Lucian Randolph & Claude Anthro Randolph
Date: April 1, 2026
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import minimize, minimize_scalar
import os
import warnings
warnings.filterwarnings('ignore')

# ================================================================
# CONSTANTS
# ================================================================
DELTA = 4.669201609
ALPHA = 2.502907875
LN_DELTA = np.log(DELTA)
LN_ALPHA = np.log(ALPHA)
LAMBDA_R = DELTA / ALPHA

C_LIGHT = 299792.458     # km/s
H0_PLANCK = 67.4         # km/s/Mpc
H0_SHOES = 73.04
OMEGA_M = 0.315
OMEGA_B = 0.049
OMEGA_R = 9.15e-5
OMEGA_L = 0.685

OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')

print('=' * 72)
print('  Script 116 — Modified Friedmann Equation Test')
print('  Can Baryons + Radiation + τ(z) Replace ΛCDM?')
print(f'  δ = {DELTA}   α = {ALPHA}   ln(δ) = {LN_DELTA:.4f}')
print('=' * 72)

# ================================================================
# LOAD PANTHEON+ DATA
# ================================================================
print('\n--- Loading Pantheon+ Data ---')

data_file = '/tmp/pantheon_data/Pantheon+SH0ES.dat'
with open(data_file, 'r') as f:
    header = f.readline().strip().split()
    rows = []
    for line in f:
        parts = line.strip().split()
        if len(parts) >= len(header):
            rows.append(parts)

col_idx = {name: i for i, name in enumerate(header)}
zHD = np.array([float(row[col_idx['zHD']]) for row in rows])
MU_SH0ES = np.array([float(row[col_idx['MU_SH0ES']]) for row in rows])
MU_err = np.array([float(row[col_idx['MU_SH0ES_ERR_DIAG']]) for row in rows])
is_cal = np.array([int(row[col_idx['IS_CALIBRATOR']]) for row in rows])

# Hubble flow selection
mask = (zHD > 0.01) & (is_cal == 0)
z_data = zHD[mask]
mu_obs = MU_SH0ES[mask]
mu_err_data = MU_err[mask]
N_data = len(z_data)

print(f'  Total: {len(zHD)} supernovae')
print(f'  Hubble flow (z>0.01, non-cal): {N_data}')
print(f'  z range: [{z_data.min():.4f}, {z_data.max():.4f}]')

# ================================================================
# MODEL DEFINITIONS
# ================================================================

def E_LCDM(z):
    return np.sqrt(OMEGA_M * (1+z)**3 + OMEGA_R * (1+z)**4 + OMEGA_L)

def E_baryons(z):
    """Baryons + radiation. Open universe (curvature fills rest)."""
    Ok = 1 - OMEGA_B - OMEGA_R
    return np.sqrt(OMEGA_B * (1+z)**3 + OMEGA_R * (1+z)**4 + Ok * (1+z)**2)

def E_matter(z, Om=OMEGA_M):
    """Matter-only, open universe."""
    Ok = 1 - Om
    return np.sqrt(Om * (1+z)**3 + Ok * (1+z)**2)

def tau_func(z, z_t=1.449, beta=LN_DELTA):
    """Time emergence function τ(z).
    τ(0) = 1 (fully emerged now)
    τ(∞) → 0 (not yet emerged)"""
    return 1.0 / (1.0 + (z / z_t)**beta)

def luminosity_distance(z, H0, E_func, tau=None, z_t=1.449):
    """Luminosity distance in Mpc.
    If tau is provided, it modifies the integrand (Paper 9 form)."""
    def integrand(zp):
        E = E_func(zp)
        if tau is not None:
            t = tau(zp, z_t=z_t)
            return 1.0 / (E * t) if t > 0.001 else 1.0 / (E * 0.001)
        return 1.0 / E
    comoving, _ = quad(integrand, 0, z, limit=200)
    return (1 + z) * (C_LIGHT / H0) * comoving

def dist_modulus(d_L_Mpc):
    """Distance modulus from luminosity distance in Mpc."""
    if d_L_Mpc <= 0:
        return 0
    return 5.0 * np.log10(d_L_Mpc * 1e6 / 10.0)

# ================================================================
# PHASE 1: COMPUTE ALL FOUR MODELS
# ================================================================
print('\n' + '=' * 60)
print('  PHASE 1: Four Expansion Models')
print('=' * 60)

H0 = H0_PLANCK

# Pre-compute for all data points
mu_A = np.zeros(N_data)  # ΛCDM
mu_B = np.zeros(N_data)  # Baryons only
mu_C = np.zeros(N_data)  # Paper 9 (Ωm + τ)
mu_D = np.zeros(N_data)  # Full replacement (Ωb + τ)

print('  Computing Model A (ΛCDM)...', flush=True)
for i, z in enumerate(z_data):
    d = luminosity_distance(z, H0, E_LCDM)
    mu_A[i] = dist_modulus(d)
    if (i+1) % 300 == 0:
        print(f'    {i+1}/{N_data}', flush=True)

print('  Computing Model B (Baryons only)...', flush=True)
for i, z in enumerate(z_data):
    d = luminosity_distance(z, H0, E_baryons)
    mu_B[i] = dist_modulus(d)

print('  Computing Model C (Ωm + τ, Paper 9)...', flush=True)
for i, z in enumerate(z_data):
    d = luminosity_distance(z, H0, lambda zp: E_matter(zp, Om=OMEGA_M),
                           tau=tau_func, z_t=1.449)
    mu_C[i] = dist_modulus(d)

print('  Computing Model D (Ωb + τ, FULL REPLACEMENT)...', flush=True)
for i, z in enumerate(z_data):
    d = luminosity_distance(z, H0, E_baryons, tau=tau_func, z_t=1.449)
    mu_D[i] = dist_modulus(d)

print('  All models computed.', flush=True)

# ================================================================
# PHASE 2: SUPERNOVA TEST — χ²/dof
# ================================================================
print('\n' + '=' * 60)
print('  PHASE 2: Supernova Test')
print('=' * 60)

def compute_chi2(mu_model, mu_obs, mu_err):
    """Compute χ²/dof with fitted offset ΔM."""
    DM = np.median(mu_obs - mu_model)
    mu_shifted = mu_model + DM
    chi2 = np.sum((mu_obs - mu_shifted)**2 / mu_err**2)
    dof = len(mu_obs) - 2  # 2 parameters: H₀ and ΔM
    return chi2, chi2/dof, DM

chi2_A, chi2dof_A, DM_A = compute_chi2(mu_A, mu_obs, mu_err_data)
chi2_B, chi2dof_B, DM_B = compute_chi2(mu_B, mu_obs, mu_err_data)
chi2_C, chi2dof_C, DM_C = compute_chi2(mu_C, mu_obs, mu_err_data)
chi2_D, chi2dof_D, DM_D = compute_chi2(mu_D, mu_obs, mu_err_data)

print(f'  {"Model":30s}  {"χ²":>10s}  {"χ²/dof":>8s}  {"ΔM":>8s}')
print(f'  {"-"*30}  {"-"*10}  {"-"*8}  {"-"*8}')
print(f'  {"A: ΛCDM (benchmark)":30s}  {chi2_A:10.1f}  {chi2dof_A:8.3f}  {DM_A:8.3f}')
print(f'  {"B: Baryons only":30s}  {chi2_B:10.1f}  {chi2dof_B:8.3f}  {DM_B:8.3f}')
print(f'  {"C: Ωm + τ(z) (Paper 9)":30s}  {chi2_C:10.1f}  {chi2dof_C:8.3f}  {DM_C:8.3f}')
print(f'  {"D: Ωb + τ(z) (FULL REPLACE)":30s}  {chi2_D:10.1f}  {chi2dof_D:8.3f}  {DM_D:8.3f}')

print(f'\n  Model D / Model A ratio: {chi2dof_D/chi2dof_A:.3f}')

# ================================================================
# PHASE 3: REFIT z_t FOR MODEL D
# ================================================================
print('\n' + '=' * 60)
print('  PHASE 3: Refit z_t for Baryons-Only + τ(z)')
print('=' * 60)

def chi2_for_zt(z_t_val):
    """Compute χ² for Model D with a specific z_t."""
    mu_test = np.zeros(N_data)
    for i, z in enumerate(z_data):
        d = luminosity_distance(z, H0, E_baryons, tau=tau_func, z_t=z_t_val)
        mu_test[i] = dist_modulus(d)
    DM = np.median(mu_obs - mu_test)
    chi2 = np.sum((mu_obs - mu_test - DM)**2 / mu_err_data**2)
    return chi2

# Scan z_t from 0.5 to 5.0
print('  Scanning z_t from 0.5 to 5.0...', flush=True)
zt_range = np.linspace(0.3, 5.0, 30)
chi2_scan = []
for zt_test in zt_range:
    c2 = chi2_for_zt(zt_test)
    chi2_scan.append(c2)
    if len(chi2_scan) % 5 == 0:
        print(f'    z_t = {zt_test:.2f}: χ² = {c2:.1f} '
              f'(χ²/dof = {c2/(N_data-3):.3f})', flush=True)

chi2_scan = np.array(chi2_scan)
best_idx = np.argmin(chi2_scan)
zt_best_approx = zt_range[best_idx]
print(f'\n  Best z_t (coarse): {zt_best_approx:.2f} '
      f'(χ²/dof = {chi2_scan[best_idx]/(N_data-3):.4f})')

# Refine with scipy
result = minimize_scalar(chi2_for_zt,
                         bounds=(max(0.1, zt_best_approx-0.5),
                                 zt_best_approx+0.5),
                         method='bounded')
zt_best = result.x
chi2_best = result.fun
print(f'  Best z_t (refined): {zt_best:.4f} '
      f'(χ²/dof = {chi2_best/(N_data-3):.4f})')
print(f'  Original Paper 9 z_t: 1.449')
print(f'  Shift: {zt_best - 1.449:+.3f}')

# Recompute Model D with best z_t
print(f'\n  Recomputing Model D with z_t = {zt_best:.4f}...', flush=True)
mu_D_best = np.zeros(N_data)
for i, z in enumerate(z_data):
    d = luminosity_distance(z, H0, E_baryons, tau=tau_func, z_t=zt_best)
    mu_D_best[i] = dist_modulus(d)

chi2_D2, chi2dof_D2, DM_D2 = compute_chi2(mu_D_best, mu_obs, mu_err_data)
print(f'  Model D (refitted): χ²/dof = {chi2dof_D2:.4f}')

# ================================================================
# PHASE 4: CMB FIRST PEAK CHECK
# ================================================================
print('\n' + '=' * 60)
print('  PHASE 4: CMB First Peak Angular Diameter Distance')
print('=' * 60)

z_CMB = 1089.9

def angular_diameter_distance(z_target, H0, E_func, tau=None, z_t=1.449):
    """d_A = d_L / (1+z)² = (c/H₀) × ∫dz/E(z) / (1+z)."""
    def integrand(zp):
        E = E_func(zp)
        if tau is not None:
            t = tau(zp, z_t=z_t)
            return 1.0 / (E * t) if t > 0.001 else 1.0 / (E * 0.001)
        return 1.0 / E
    comoving, _ = quad(integrand, 0, z_target, limit=500)
    return (C_LIGHT / H0) * comoving / (1 + z_target)  # Mpc

dA_LCDM = angular_diameter_distance(z_CMB, H0, E_LCDM)
dA_baryons = angular_diameter_distance(z_CMB, H0, E_baryons)
dA_D = angular_diameter_distance(z_CMB, H0, E_baryons,
                                  tau=tau_func, z_t=zt_best)

# The sound horizon at recombination (simplified Eisenstein & Hu 1998)
# r_s ≈ 147 Mpc (Planck 2018 value)
r_s_planck = 147.09  # Mpc

# Angular scale of sound horizon
theta_LCDM = r_s_planck / dA_LCDM  # radians
theta_D = r_s_planck / dA_D

# Multipole of first peak
ell_LCDM = np.pi / theta_LCDM
ell_D = np.pi / theta_D
ell_observed = 220.0  # Planck measured

print(f'  Angular diameter distance to CMB (z = {z_CMB}):')
print(f'    ΛCDM:      d_A = {dA_LCDM:.2f} Mpc')
print(f'    Baryons:   d_A = {dA_baryons:.2f} Mpc')
print(f'    Model D:   d_A = {dA_D:.2f} Mpc')
print(f'')
print(f'  First peak multipole (using r_s = {r_s_planck} Mpc):')
print(f'    ΛCDM:      ℓ = {ell_LCDM:.1f}')
print(f'    Model D:   ℓ = {ell_D:.1f}')
print(f'    Observed:  ℓ = {ell_observed}')
print(f'    D/observed: {ell_D/ell_observed:.4f}')

# ================================================================
# PHASE 5: GHOST OF α CROSS-CHECK
# ================================================================
print('\n' + '=' * 60)
print('  PHASE 5: Ghost of α Cross-Check')
print('=' * 60)

# If an ΛCDM observer fits Model D's expansion history,
# what Ωm would they derive?
# Method: find Ωm such that ΛCDM best matches Model D's d_L(z)

def chi2_ghost(Om_test):
    """χ² between Model D distances and ΛCDM with Om_test."""
    z_test = np.linspace(0.02, 2.0, 100)
    chi2 = 0
    for z in z_test:
        d_D = luminosity_distance(z, H0, E_baryons, tau=tau_func, z_t=zt_best)
        d_LCDM = luminosity_distance(z, H0,
                     lambda zp: np.sqrt(Om_test*(1+zp)**3 + (1-Om_test)))
        if d_D > 0 and d_LCDM > 0:
            chi2 += (dist_modulus(d_D) - dist_modulus(d_LCDM))**2
    return chi2

print('  Finding effective Ωm that ΛCDM observer would measure...')
result_ghost = minimize_scalar(chi2_ghost, bounds=(0.05, 0.8), method='bounded')
Om_effective = result_ghost.x
Om_predicted = ALPHA**2 * OMEGA_B

print(f'  Effective Ωm (from fitting ΛCDM to Model D):  {Om_effective:.4f}')
print(f'  α² × Ωb = {ALPHA**2:.4f} × {OMEGA_B} = {Om_predicted:.4f}')
print(f'  Actual ΛCDM Ωm:                               {OMEGA_M}')
print(f'  Deviation (effective vs α²×Ωb): '
      f'{(Om_effective - Om_predicted)/Om_predicted * 100:+.1f}%')
print(f'  Deviation (effective vs ΛCDM Ωm): '
      f'{(Om_effective - OMEGA_M)/OMEGA_M * 100:+.1f}%')

# ================================================================
# PHASE 6: FIGURES
# ================================================================
print('\n--- Generating Figures ---')

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel (a): Hubble diagram — all four models
ax = axes[0, 0]
# Plot data
ax.scatter(z_data, mu_obs, s=1, c='gray', alpha=0.3, label='Pantheon+ data')
# Plot models
z_smooth = np.linspace(0.01, 2.3, 200)
mu_smooth_A = np.array([dist_modulus(luminosity_distance(z, H0, E_LCDM))
                        for z in z_smooth]) + DM_A
mu_smooth_D = np.array([dist_modulus(luminosity_distance(z, H0, E_baryons,
                        tau=tau_func, z_t=zt_best)) for z in z_smooth]) + DM_D2
mu_smooth_B = np.array([dist_modulus(luminosity_distance(z, H0, E_baryons))
                        for z in z_smooth]) + DM_B

ax.plot(z_smooth, mu_smooth_A, 'b-', linewidth=2, label='A: ΛCDM')
ax.plot(z_smooth, mu_smooth_D, 'r--', linewidth=2,
        label=f'D: Ωb + τ(z_t={zt_best:.2f})')
ax.plot(z_smooth, mu_smooth_B, 'g:', linewidth=1.5, label='B: Baryons only')
ax.set_xlabel('Redshift z', fontsize=11)
ax.set_ylabel('Distance Modulus μ', fontsize=11)
ax.set_title('(a) Hubble Diagram: Four Models', fontsize=12)
ax.legend(fontsize=8, loc='lower right')
ax.set_xlim(0, 2.3)
ax.grid(True, alpha=0.3)

# Panel (b): Residuals (Model D - ΛCDM)
ax = axes[0, 1]
resid_D = (mu_D_best + DM_D2) - (mu_A + DM_A)
ax.scatter(z_data, mu_obs - (mu_A + DM_A), s=1, c='gray', alpha=0.3,
           label='Data - ΛCDM')
ax.plot(z_smooth,
        np.array([dist_modulus(luminosity_distance(z, H0, E_baryons,
                  tau=tau_func, z_t=zt_best)) for z in z_smooth]) + DM_D2 -
        np.array([dist_modulus(luminosity_distance(z, H0, E_LCDM))
                  for z in z_smooth]) - DM_A,
        'r-', linewidth=2, label='Model D - ΛCDM')
ax.axhline(y=0, color='blue', linestyle='--', alpha=0.5)
ax.set_xlabel('Redshift z', fontsize=11)
ax.set_ylabel('Δμ (mag)', fontsize=11)
ax.set_title('(b) Residuals Relative to ΛCDM', fontsize=12)
ax.legend(fontsize=9)
ax.set_xlim(0, 2.3)
ax.set_ylim(-1, 1)
ax.grid(True, alpha=0.3)

# Panel (c): z_t scan
ax = axes[1, 0]
ax.plot(zt_range, chi2_scan / (N_data - 3), 'k-', linewidth=2)
ax.axvline(x=1.449, color='blue', linestyle='--', label='Paper 9: z_t=1.449')
ax.axvline(x=zt_best, color='red', linestyle='--',
           label=f'Best fit: z_t={zt_best:.3f}')
ax.axhline(y=chi2dof_A, color='green', linestyle=':', alpha=0.5,
           label=f'ΛCDM χ²/dof={chi2dof_A:.3f}')
ax.set_xlabel('z_t', fontsize=11)
ax.set_ylabel('χ²/dof', fontsize=11)
ax.set_title('(c) z_t Optimization for Model D', fontsize=12)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Panel (d): Summary table
ax = axes[1, 1]
ax.axis('off')
ax.set_title('(d) Results Summary', fontsize=12)

summary = (
    f'MODEL COMPARISON (Pantheon+ {N_data} SNe)\n'
    f'{"─"*45}\n'
    f'A: ΛCDM           χ²/dof = {chi2dof_A:.3f}\n'
    f'B: Baryons only    χ²/dof = {chi2dof_B:.3f}\n'
    f'C: Ωm + τ (Paper9) χ²/dof = {chi2dof_C:.3f}\n'
    f'D: Ωb + τ (z_t=1.449) χ²/dof = {chi2dof_D:.3f}\n'
    f'D: Ωb + τ (z_t={zt_best:.3f}) χ²/dof = {chi2dof_D2:.3f}\n'
    f'{"─"*45}\n'
    f'Best z_t: {zt_best:.4f} (was 1.449)\n'
    f'CMB ℓ₁: {ell_D:.0f} (observed: 220)\n'
    f'Ghost Ωm_eff: {Om_effective:.4f}\n'
    f'α²×Ωb: {Om_predicted:.4f}\n'
    f'ΛCDM Ωm: {OMEGA_M}'
)
ax.text(0.1, 0.5, summary, fontsize=11, fontfamily='monospace',
        verticalalignment='center',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
fig_path = os.path.join(OUTDIR, 'fig7_friedmann_test.png')
plt.savefig(fig_path, dpi=200, bbox_inches='tight')
plt.close()
print(f'  Saved: {fig_path}')

# ================================================================
# COMPLETE SUMMARY
# ================================================================
print('\n' + '=' * 72)
print('  COMPLETE RESULTS SUMMARY')
print('=' * 72)
print(f'')
print(f'  PHASE 2 — SUPERNOVA TEST:')
print(f'    A: ΛCDM                    χ²/dof = {chi2dof_A:.4f}')
print(f'    B: Baryons only            χ²/dof = {chi2dof_B:.4f}')
print(f'    C: Ωm + τ (Paper 9)        χ²/dof = {chi2dof_C:.4f}')
print(f'    D: Ωb + τ (z_t=1.449)      χ²/dof = {chi2dof_D:.4f}')
print(f'    D: Ωb + τ (z_t={zt_best:.3f})    χ²/dof = {chi2dof_D2:.4f}')
print(f'')
print(f'  PHASE 3 — z_t REFIT:')
print(f'    Original: z_t = 1.449')
print(f'    Refitted: z_t = {zt_best:.4f}')
print(f'    Shift: {zt_best - 1.449:+.4f}')
print(f'')
print(f'  PHASE 4 — CMB FIRST PEAK:')
print(f'    ℓ₁ (ΛCDM): {ell_LCDM:.1f}')
print(f'    ℓ₁ (Model D): {ell_D:.1f}')
print(f'    ℓ₁ (observed): {ell_observed}')
print(f'')
print(f'  PHASE 5 — GHOST OF α:')
print(f'    Effective Ωm from ΛCDM fit to Model D: {Om_effective:.4f}')
print(f'    α² × Ωb = {Om_predicted:.4f}')
print(f'    ΛCDM Ωm = {OMEGA_M}')
print(f'    Deviation: {(Om_effective - Om_predicted)/Om_predicted * 100:+.1f}%')
print(f'')

# Determine outcome
if chi2dof_D2 < chi2dof_A * 1.1:
    outcome = 'OUTCOME 1: Model D matches ΛCDM within 10%'
elif chi2dof_D2 > chi2dof_A * 2.0:
    if chi2dof_D2 > chi2dof_B:
        outcome = 'OUTCOME 3: Model D undershoots — τ not strong enough'
    else:
        outcome = 'OUTCOME 2: Model D overshoots — τ overcompensates'
else:
    outcome = 'INTERMEDIATE: Model D is between 10% and 2× of ΛCDM'

print(f'  VERDICT: {outcome}')
print('=' * 72)
