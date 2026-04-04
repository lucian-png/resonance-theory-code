#!/usr/bin/env python3
"""
Script 117 — Paper 39: The Modified Friedmann Equation
=======================================================
"The Answer is 42"

Phase 1: Two-regime τ(z) — recover CMB first peak ℓ₁ = 220
Phase 2: Sound horizon calculation
Phase 3: Peak heights through CAMB (three approaches)
Phase 4: BAO cross-check
Phase 5: Hubble tension check
Phase 6: Assessment

THE EQUATION:
  H²(z) = (8πG/3) × [ρ_b(1+z)³ + ρ_r(1+z)⁴] / τ(z)²

where τ(z) has two regimes:
  z < z_activation: τ = 1/(1 + (z/z_t)^β), β = ln(δ), z_t = 2.046
  z > z_activation: τ = τ_floor

Paper 42 (counting classified). The Answer to Life, the Universe,
and Everything. For Cuz. For Douglas Adams.

Author: Lucian Randolph & Claude Anthro Randolph
Date: April 3, 2026
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import brentq, minimize_scalar
import camb
from camb import model as camb_model
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
C_SI    = 2.99792458e8   # m/s
G_N     = 6.67430e-11    # m³ kg⁻¹ s⁻²
HBAR    = 1.054571817e-34
K_B     = 1.380649e-23
M_P     = 1.67262192e-27  # proton mass kg

# Planck 2018 ΛCDM parameters
H0_PLANCK = 67.36
OMEGA_M   = 0.315
OMEGA_B   = 0.049
OMEGA_B_H2 = 0.02237
OMEGA_C_H2 = 0.1200
OMEGA_R   = 9.15e-5
OMEGA_L   = 0.685
T_CMB     = 2.7255
N_EFF     = 3.046
NS_PLANCK = 0.9649
AS_PLANCK = 2.1e-9
TAU_REION = 0.0544

# Model D parameters (from Script 116)
Z_T = 2.046     # refitted transition redshift
BETA = LN_DELTA  # = 1.5410

# Conversions
MPC_TO_M = 3.0857e22
GYR_TO_S = 3.156e16

OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')

print('=' * 72)
print('  Script 117 — Paper 39: The Modified Friedmann Equation')
print('  "The Answer is 42"')
print(f'  δ = {DELTA}   α = {ALPHA}   ln(δ) = {LN_DELTA:.4f}')
print(f'  z_t = {Z_T}   β = ln(δ) = {BETA:.4f}')
print('=' * 72)

# ================================================================
# τ(z) FUNCTION — TWO-REGIME
# ================================================================

def tau_two_regime(z, z_t=Z_T, beta=BETA, z_act=50.0, tau_floor=0.937):
    """Two-regime time emergence function.
    z < z_act: τ = 1/(1 + (z/z_t)^β) — active regime
    z > z_act: τ = τ_floor — floor regime (pre-structure)

    Smooth transition between regimes using a sigmoid blend.
    """
    tau_active = 1.0 / (1.0 + (z / z_t)**beta)
    # Smooth blend over ~5 in z centered at z_act
    blend = 1.0 / (1.0 + np.exp(-(z - z_act) / (z_act * 0.1)))
    return tau_active * (1 - blend) + tau_floor * blend

def E_model_D(z, Ob=OMEGA_B, Or=OMEGA_R, tau_func=None, **tau_kwargs):
    """Model D expansion function: baryons + radiation + curvature,
    with τ(z) correction on the expansion rate."""
    Ok = 1 - Ob - Or
    E2_base = Ob * (1+z)**3 + Or * (1+z)**4 + Ok * (1+z)**2
    if tau_func is not None:
        tau = tau_func(z, **tau_kwargs)
        tau = max(tau, 0.001)  # prevent division by zero
        return np.sqrt(E2_base) / tau
    return np.sqrt(E2_base)

def luminosity_distance(z, H0, E_func, **kwargs):
    """Luminosity distance in Mpc."""
    comoving, _ = quad(lambda zp: 1.0 / E_func(zp, **kwargs),
                       0, z, limit=500)
    return (1 + z) * (C_LIGHT / H0) * comoving

def angular_diameter_distance(z, H0, E_func, **kwargs):
    """Angular diameter distance in Mpc."""
    comoving, _ = quad(lambda zp: 1.0 / E_func(zp, **kwargs),
                       0, z, limit=500)
    return (C_LIGHT / H0) * comoving / (1 + z)

def dist_modulus(d_L_Mpc):
    if d_L_Mpc <= 0:
        return 0
    return 5.0 * np.log10(d_L_Mpc * 1e6 / 10.0)

# ================================================================
# PHASE 1: TWO-REGIME τ — RECOVER ℓ₁ = 220
# ================================================================
print('\n' + '=' * 60)
print('  PHASE 1: Recover CMB First Peak Position')
print('=' * 60)

# Sound horizon at recombination (Planck value for initial calibration)
R_S_PLANCK = 147.09  # Mpc
Z_REC = 1089.9

# Step 1.1: Test z_activation values
print('\n  Step 1.1: Scanning z_activation...')
print(f'  {"z_act":>8s}  {"τ_floor":>8s}  {"d_A (Mpc)":>12s}  {"ℓ₁":>8s}')
print(f'  {"-"*8}  {"-"*8}  {"-"*12}  {"-"*8}')

# For each z_act, find the τ_floor that gives ℓ₁ = 220
results_phase1 = []

for z_act in [10, 20, 30, 50, 100, 200, 500, 1000]:
    for tau_fl in np.arange(0.50, 1.00, 0.01):
        try:
            dA = angular_diameter_distance(
                Z_REC, H0_PLANCK,
                lambda z: E_model_D(z, tau_func=tau_two_regime,
                                     z_act=z_act, tau_floor=tau_fl))
            if dA > 0:
                ell1 = np.pi * dA / R_S_PLANCK * (1 + Z_REC)
                # Actually: ℓ = π × d_A(z_rec) / r_s
                # But d_A already has the (1+z) factor divided out
                ell1 = np.pi * dA / R_S_PLANCK
                if abs(ell1 - 220) < 5:
                    results_phase1.append((z_act, tau_fl, dA, ell1))
                    print(f'  {z_act:8.0f}  {tau_fl:8.3f}  {dA:12.2f}  {ell1:8.1f}')
        except Exception:
            pass

if not results_phase1:
    print('  No matches found in coarse scan. Trying wider range...')
    # Try with a wider τ_floor range and different E_model
    for z_act in [10, 30, 50, 100, 500]:
        # Compute d_A for a range of tau_floor values
        for tau_fl in np.arange(0.01, 1.00, 0.005):
            try:
                dA = angular_diameter_distance(
                    Z_REC, H0_PLANCK,
                    lambda z, _tf=tau_fl, _za=z_act:
                        E_model_D(z, tau_func=tau_two_regime,
                                   z_act=_za, tau_floor=_tf))
                if dA > 0:
                    ell1 = np.pi * dA / R_S_PLANCK
                    if abs(ell1 - 220) < 20:
                        results_phase1.append((z_act, tau_fl, dA, ell1))
                        if abs(ell1 - 220) < 5:
                            print(f'  {z_act:8.0f}  {tau_fl:8.4f}  '
                                  f'{dA:12.2f}  {ell1:8.1f}  ← MATCH')
            except Exception:
                pass

# If still no matches, let's compute what d_A we NEED and work backward
target_dA = R_S_PLANCK * 220 / np.pi
print(f'\n  Target d_A for ℓ₁=220: {target_dA:.2f} Mpc')
print(f'  (using r_s = {R_S_PLANCK} Mpc)')

# Compute d_A for ΛCDM as sanity check
dA_LCDM = angular_diameter_distance(
    Z_REC, H0_PLANCK,
    lambda z: np.sqrt(OMEGA_M*(1+z)**3 + OMEGA_R*(1+z)**4 + OMEGA_L))
ell_LCDM = np.pi * dA_LCDM / R_S_PLANCK
print(f'  ΛCDM d_A: {dA_LCDM:.2f} Mpc → ℓ₁ = {ell_LCDM:.1f}')

# Scan d_A as function of τ_floor for a few z_act values
print(f'\n  Detailed d_A scan:')
print(f'  {"z_act":>6s}  {"τ_floor":>8s}  {"d_A":>10s}  {"ℓ₁":>8s}')
print(f'  {"-"*6}  {"-"*8}  {"-"*10}  {"-"*8}')

best_match = None
best_deviation = 999

for z_act in [30, 50, 100]:
    for tau_fl in np.arange(0.01, 1.00, 0.02):
        try:
            dA = angular_diameter_distance(
                Z_REC, H0_PLANCK,
                lambda z, _tf=tau_fl, _za=z_act:
                    E_model_D(z, tau_func=tau_two_regime,
                               z_act=_za, tau_floor=_tf))
            ell1 = np.pi * dA / R_S_PLANCK
            dev = abs(ell1 - 220)

            if dev < 30:
                print(f'  {z_act:6.0f}  {tau_fl:8.3f}  {dA:10.2f}  {ell1:8.1f}')

            if dev < best_deviation:
                best_deviation = dev
                best_match = (z_act, tau_fl, dA, ell1)
        except Exception:
            pass

if best_match:
    print(f'\n  Best match: z_act={best_match[0]}, τ_floor={best_match[1]:.3f}, '
          f'ℓ₁={best_match[3]:.1f} (dev={best_deviation:.1f})')
else:
    print(f'\n  No close match found. Need to investigate further.')

# Let's also try: what if τ doesn't modify H(z) directly but only
# modifies the LUMINOSITY DISTANCE integral (as in Paper 9)?
# For the angular diameter distance:
# d_A = (c/H₀) × 1/(1+z) × ∫₀ᶻ dz'/[E(z') × τ(z')]
# where E(z) is the STANDARD baryons-only expansion.

print(f'\n  Alternative: τ on propagation integral only (Paper 9 form)...')

def dA_paper9_form(z_target, H0, z_act=50, tau_fl=0.937):
    """d_A with τ applied to the propagation integral, not to H(z)."""
    Ob = OMEGA_B
    Or = OMEGA_R
    Ok = 1 - Ob - Or

    def integrand(z):
        E = np.sqrt(Ob*(1+z)**3 + Or*(1+z)**4 + Ok*(1+z)**2)
        tau = tau_two_regime(z, z_act=z_act, tau_floor=tau_fl)
        tau = max(tau, 0.001)
        return 1.0 / (E * tau)

    comoving, _ = quad(integrand, 0, z_target, limit=500)
    return (C_LIGHT / H0) * comoving / (1 + z_target)

print(f'  {"z_act":>6s}  {"τ_floor":>8s}  {"d_A":>10s}  {"ℓ₁":>8s}')
print(f'  {"-"*6}  {"-"*8}  {"-"*10}  {"-"*8}')

best_p9 = None
best_p9_dev = 999

for z_act in [10, 20, 30, 50, 100, 500]:
    for tau_fl in np.arange(0.80, 1.00, 0.005):
        try:
            dA = dA_paper9_form(Z_REC, H0_PLANCK, z_act=z_act, tau_fl=tau_fl)
            ell1 = np.pi * dA / R_S_PLANCK
            dev = abs(ell1 - 220)

            if dev < 30:
                print(f'  {z_act:6.0f}  {tau_fl:8.4f}  {dA:10.2f}  {ell1:8.1f}')

            if dev < best_p9_dev:
                best_p9_dev = dev
                best_p9 = (z_act, tau_fl, dA, ell1)
        except Exception:
            pass

if best_p9:
    print(f'\n  Best Paper 9 form: z_act={best_p9[0]}, τ_floor={best_p9[1]:.4f}, '
          f'ℓ₁={best_p9[3]:.1f} (dev={best_p9_dev:.1f})')

# ================================================================
# PHASE 2: SOUND HORIZON
# ================================================================
print('\n' + '=' * 60)
print('  PHASE 2: Sound Horizon Calculation')
print('=' * 60)

# Sound speed: c_s = c / √(3(1+R))
# R = 3ρ_b / (4ρ_γ) = (3 Ωb h²) / (4 Ωγ h²) × a
# At recombination:

h = H0_PLANCK / 100
Omega_gamma_h2 = 2.469e-5 * (T_CMB / 2.7255)**4
R_rec = 3 * OMEGA_B_H2 / (4 * Omega_gamma_h2) * (1 / (1 + Z_REC))

# Actually R(a) = 3Ωb/(4Ωγ) × a
R_factor = 3 * OMEGA_B_H2 / (4 * Omega_gamma_h2)
R_at_rec = R_factor / (1 + Z_REC)

print(f'  Baryon loading R = 3Ωb/(4Ωγ) = {R_factor:.4f}')
print(f'  R at recombination: {R_at_rec:.6f}')
print(f'  Sound speed at rec: c_s = c/{np.sqrt(3*(1+R_at_rec)):.4f} '
      f'= {1/np.sqrt(3*(1+R_at_rec)):.6f} c')
print(f'')
print(f'  KEY: R depends on Ωbh² and Ωγh² ONLY.')
print(f'  Dark matter does NOT affect the sound speed.')
print(f'  c_s is IDENTICAL in Model D and ΛCDM.')

# Compute sound horizon using CAMB for standard ΛCDM
print(f'\n  Computing sound horizon via CAMB...')
pars_lcdm = camb.CAMBparams()
pars_lcdm.set_cosmology(
    H0=H0_PLANCK, ombh2=OMEGA_B_H2, omch2=OMEGA_C_H2,
    mnu=0.06, omk=0, tau=TAU_REION,
    TCMB=T_CMB, nnu=N_EFF)
pars_lcdm.InitPower.set_params(As=AS_PLANCK, ns=NS_PLANCK)
pars_lcdm.set_for_lmax(2500, lens_potential_accuracy=0)

results_lcdm = camb.get_results(pars_lcdm)
rs_lcdm = results_lcdm.get_derived_params()['rdrag']
print(f'  ΛCDM sound horizon (drag epoch): r_d = {rs_lcdm:.2f} Mpc')

# For baryons-only (no dark matter):
pars_baryons = camb.CAMBparams()
pars_baryons.set_cosmology(
    H0=H0_PLANCK, ombh2=OMEGA_B_H2, omch2=0.0,  # NO dark matter
    mnu=0.0, omk=0, tau=TAU_REION,
    TCMB=T_CMB, nnu=N_EFF)
pars_baryons.InitPower.set_params(As=AS_PLANCK, ns=NS_PLANCK)
pars_baryons.set_for_lmax(2500, lens_potential_accuracy=0)

try:
    results_baryons = camb.get_results(pars_baryons)
    rs_baryons = results_baryons.get_derived_params()['rdrag']
    print(f'  Baryons-only sound horizon: r_d = {rs_baryons:.2f} Mpc')
    print(f'  Ratio: {rs_baryons/rs_lcdm:.4f}')
except Exception as e:
    print(f'  Baryons-only CAMB failed: {e}')
    rs_baryons = None

# ================================================================
# PHASE 3: PEAK HEIGHTS THROUGH CAMB
# ================================================================
print('\n' + '=' * 60)
print('  PHASE 3: CMB Power Spectrum — Three Approaches')
print('=' * 60)

# APPROACH A: Pure baryons, standard cosmology
print('\n  --- Approach A: Pure baryons, no dark matter ---')
pars_A = camb.CAMBparams()
pars_A.set_cosmology(
    H0=H0_PLANCK, ombh2=OMEGA_B_H2, omch2=0.0,
    mnu=0.0, omk=0, tau=TAU_REION,
    TCMB=T_CMB, nnu=N_EFF)
pars_A.InitPower.set_params(As=AS_PLANCK, ns=NS_PLANCK)
pars_A.set_for_lmax(2500, lens_potential_accuracy=0)

try:
    results_A = camb.get_results(pars_A)
    powers_A = results_A.get_cmb_power_spectra(pars_A, CMB_unit='muK')
    cl_A = powers_A['total'][:, 0]  # TT spectrum
    ell_A = np.arange(len(cl_A))
    derived_A = results_A.get_derived_params()
    print(f'  Approach A computed successfully.')
    print(f'  Derived params: H0={derived_A.get("H0",0):.2f}, '
          f'age={derived_A.get("age",0):.3f} Gyr')
except Exception as e:
    print(f'  Approach A failed: {e}')
    cl_A = None

# APPROACH B: Standard ΛCDM (reference)
print('\n  --- Approach B (Reference): Standard ΛCDM ---')
try:
    powers_LCDM = results_lcdm.get_cmb_power_spectra(pars_lcdm, CMB_unit='muK')
    cl_LCDM = powers_LCDM['total'][:, 0]
    ell_LCDM = np.arange(len(cl_LCDM))
    derived_LCDM = results_lcdm.get_derived_params()
    print(f'  ΛCDM computed successfully.')
    print(f'  Derived: age={derived_LCDM.get("age",0):.3f} Gyr, '
          f'r_drag={derived_LCDM.get("rdrag",0):.2f} Mpc')
except Exception as e:
    print(f'  ΛCDM reference failed: {e}')
    cl_LCDM = None

# APPROACH C: Baryons with α² gravitational enhancement
# Use omch2 that gives total Ωm = α² × Ωb for PERTURBATION growth
# while the background is different
print('\n  --- Approach C: α² gravitational enhancement ---')
# Effective dark matter for perturbation growth only:
# Ωm_eff = α² × Ωb → Ωch²_eff = Ωm_eff × h² - Ωbh²
Om_eff = ALPHA**2 * OMEGA_B
Oc_h2_eff = Om_eff * h**2 - OMEGA_B_H2
print(f'  α² × Ωb = {Om_eff:.4f}')
print(f'  Effective Ωch² for perturbations: {Oc_h2_eff:.6f}')
print(f'  ΛCDM Ωch²: {OMEGA_C_H2}')

pars_C = camb.CAMBparams()
pars_C.set_cosmology(
    H0=H0_PLANCK, ombh2=OMEGA_B_H2, omch2=Oc_h2_eff,
    mnu=0.0, omk=0, tau=TAU_REION,
    TCMB=T_CMB, nnu=N_EFF)
pars_C.InitPower.set_params(As=AS_PLANCK, ns=NS_PLANCK)
pars_C.set_for_lmax(2500, lens_potential_accuracy=0)

try:
    results_C = camb.get_results(pars_C)
    powers_C = results_C.get_cmb_power_spectra(pars_C, CMB_unit='muK')
    cl_C = powers_C['total'][:, 0]
    ell_C = np.arange(len(cl_C))
    derived_C = results_C.get_derived_params()
    print(f'  Approach C computed successfully.')
    print(f'  Derived: age={derived_C.get("age",0):.3f} Gyr, '
          f'r_drag={derived_C.get("rdrag",0):.2f} Mpc')
except Exception as e:
    print(f'  Approach C failed: {e}')
    cl_C = None

# ================================================================
# PEAK ANALYSIS
# ================================================================
print('\n  --- Peak Height Analysis ---')

def find_peaks(cl, ell, ell_min=50, ell_max=2000):
    """Find the positions and heights of CMB peaks."""
    peaks = []
    for i in range(max(2, ell_min), min(len(cl)-1, ell_max)):
        if cl[i] > cl[i-1] and cl[i] > cl[i+1] and cl[i] > 100:
            peaks.append((i, cl[i]))
    return peaks

if cl_LCDM is not None:
    peaks_LCDM = find_peaks(cl_LCDM, ell_LCDM)
    print(f'\n  ΛCDM peaks:')
    for i, (ell_p, height) in enumerate(peaks_LCDM[:7]):
        ptype = 'compression' if i % 2 == 0 else 'rarefaction'
        print(f'    Peak {i+1}: ℓ = {ell_p}, height = {height:.1f} μK² ({ptype})')

if cl_A is not None:
    peaks_A = find_peaks(cl_A, ell_A)
    print(f'\n  Approach A (baryons only) peaks:')
    for i, (ell_p, height) in enumerate(peaks_A[:7]):
        ptype = 'compression' if i % 2 == 0 else 'rarefaction'
        print(f'    Peak {i+1}: ℓ = {ell_p}, height = {height:.1f} μK² ({ptype})')

if cl_C is not None:
    peaks_C = find_peaks(cl_C, ell_C)
    print(f'\n  Approach C (α² enhancement) peaks:')
    for i, (ell_p, height) in enumerate(peaks_C[:7]):
        ptype = 'compression' if i % 2 == 0 else 'rarefaction'
        print(f'    Peak {i+1}: ℓ = {ell_p}, height = {height:.1f} μK² ({ptype})')

# Peak height ratios (odd/even)
if cl_LCDM is not None and len(peaks_LCDM) >= 3:
    R12_LCDM = peaks_LCDM[0][1] / peaks_LCDM[1][1]
    R13_LCDM = peaks_LCDM[0][1] / peaks_LCDM[2][1]
    print(f'\n  Peak height ratios:')
    print(f'    {"Model":30s}  {"R₁₂":>8s}  {"R₁₃":>8s}')
    print(f'    {"-"*30}  {"-"*8}  {"-"*8}')
    print(f'    {"ΛCDM":30s}  {R12_LCDM:8.4f}  {R13_LCDM:8.4f}')

    if cl_A is not None and len(peaks_A) >= 3:
        R12_A = peaks_A[0][1] / peaks_A[1][1]
        R13_A = peaks_A[0][1] / peaks_A[2][1]
        print(f'    {"A: Baryons only":30s}  {R12_A:8.4f}  {R13_A:8.4f}')

    if cl_C is not None and len(peaks_C) >= 3:
        R12_C = peaks_C[0][1] / peaks_C[1][1]
        R13_C = peaks_C[0][1] / peaks_C[2][1]
        print(f'    {"C: α² enhancement":30s}  {R12_C:8.4f}  {R13_C:8.4f}')

# ================================================================
# PHASE 4: BAO CROSS-CHECK
# ================================================================
print('\n' + '=' * 60)
print('  PHASE 4: BAO Scale Cross-Check')
print('=' * 60)

# BAO scale = r_d (sound horizon at drag epoch)
# DESI 2024 measurement: r_d = 147.09 ± 0.26 Mpc (from ΛCDM fit)
R_D_DESI = 147.09

print(f'  ΛCDM r_d (CAMB): {rs_lcdm:.2f} Mpc')
if rs_baryons is not None:
    print(f'  Baryons-only r_d: {rs_baryons:.2f} Mpc')
    print(f'  DESI measurement: {R_D_DESI:.2f} Mpc')
    print(f'  Baryons / ΛCDM: {rs_baryons/rs_lcdm:.4f}')
    print(f'  Baryons / DESI: {rs_baryons/R_D_DESI:.4f}')

# ================================================================
# PHASE 5: HUBBLE TENSION CHECK
# ================================================================
print('\n' + '=' * 60)
print('  PHASE 5: Hubble Tension Check')
print('=' * 60)

# Model D age of universe
def age_integral_modelD(z_act=50, tau_fl=0.937):
    """Age of universe in Model D."""
    H0_si = H0_PLANCK * 1e3 / MPC_TO_M
    def integrand(z):
        E = E_model_D(z, tau_func=tau_two_regime,
                       z_act=z_act, tau_floor=tau_fl)
        return 1.0 / ((1 + z) * E)
    result, _ = quad(integrand, 0, 1e5, limit=500)
    return result / H0_si / GYR_TO_S

# Use best-fit parameters from Phase 1
if best_p9:
    z_act_use, tau_fl_use = best_p9[0], best_p9[1]
elif best_match:
    z_act_use, tau_fl_use = best_match[0], best_match[1]
else:
    z_act_use, tau_fl_use = 50, 0.937

age_D = age_integral_modelD(z_act=z_act_use, tau_fl=tau_fl_use)
H0_cascade = LN_ALPHA / (age_D * GYR_TO_S) * MPC_TO_M / 1e3

print(f'  Model D age: {age_D:.3f} Gyr')
print(f'  H₀ = ln(α)/t_age = {H0_cascade:.2f} km/s/Mpc')
print(f'  Planck H₀: {H0_PLANCK}')
print(f'  SH0ES H₀: 73.04')

# ΛCDM derived age
if derived_LCDM:
    print(f'  ΛCDM age (CAMB): {derived_LCDM.get("age", 0):.3f} Gyr')

# ================================================================
# PHASE 6: FIGURES
# ================================================================
print('\n--- Generating Figures ---')

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel (a): CMB power spectra comparison
ax = axes[0, 0]
if cl_LCDM is not None:
    ell_plot = np.arange(2, 2500)
    Dl_LCDM = ell_plot * (ell_plot + 1) * cl_LCDM[2:2500] / (2 * np.pi)
    ax.plot(ell_plot, Dl_LCDM, 'b-', linewidth=2, label='ΛCDM', alpha=0.8)

if cl_A is not None:
    Dl_A = ell_plot * (ell_plot + 1) * cl_A[2:2500] / (2 * np.pi)
    ax.plot(ell_plot, Dl_A, 'r--', linewidth=1.5, label='A: Baryons only')

if cl_C is not None:
    Dl_C = ell_plot * (ell_plot + 1) * cl_C[2:2500] / (2 * np.pi)
    ax.plot(ell_plot, Dl_C, 'g-', linewidth=1.5, label=f'C: α² enhancement')

ax.set_xlabel('Multipole ℓ', fontsize=11)
ax.set_ylabel('D_ℓ = ℓ(ℓ+1)C_ℓ/2π [μK²]', fontsize=11)
ax.set_title('(a) CMB TT Power Spectrum', fontsize=12)
ax.legend(fontsize=9)
ax.set_xlim(2, 2500)
ax.set_ylim(0, None)
ax.grid(True, alpha=0.3)

# Panel (b): τ(z) two-regime function
ax = axes[0, 1]
z_arr = np.logspace(-2, 3.5, 1000)
for z_act, ls, lbl in [(30, '-', 'z_act=30'), (50, '--', 'z_act=50'),
                         (100, ':', 'z_act=100')]:
    tau_arr = [tau_two_regime(z, z_act=z_act, tau_floor=0.937) for z in z_arr]
    ax.semilogx(z_arr, tau_arr, ls, linewidth=2, label=lbl)

ax.axhline(y=0.937, color='orange', linestyle=':', alpha=0.7,
           label='τ_floor = 0.937')
ax.axvline(x=Z_REC, color='gray', linestyle='--', alpha=0.5,
           label=f'z_rec = {Z_REC}')
ax.set_xlabel('Redshift z', fontsize=11)
ax.set_ylabel('τ(z)', fontsize=11)
ax.set_title('(b) Two-Regime τ(z) Function', fontsize=12)
ax.legend(fontsize=9)
ax.set_ylim(0, 1.1)
ax.grid(True, alpha=0.3)

# Panel (c): Ratio of spectra (Approach A / ΛCDM and C / ΛCDM)
ax = axes[1, 0]
if cl_LCDM is not None and cl_A is not None:
    ratio_A = cl_A[2:2500] / np.maximum(cl_LCDM[2:2500], 1e-10)
    ax.plot(ell_plot, ratio_A, 'r-', linewidth=1.5, alpha=0.7,
            label='A / ΛCDM')
if cl_LCDM is not None and cl_C is not None:
    ratio_C = cl_C[2:2500] / np.maximum(cl_LCDM[2:2500], 1e-10)
    ax.plot(ell_plot, ratio_C, 'g-', linewidth=1.5, alpha=0.7,
            label='C / ΛCDM')
ax.axhline(y=1.0, color='blue', linestyle='--', alpha=0.5)
ax.set_xlabel('Multipole ℓ', fontsize=11)
ax.set_ylabel('C_ℓ / C_ℓ(ΛCDM)', fontsize=11)
ax.set_title('(c) Spectral Ratios vs ΛCDM', fontsize=12)
ax.legend(fontsize=10)
ax.set_xlim(2, 2500)
ax.set_ylim(0, 3)
ax.grid(True, alpha=0.3)

# Panel (d): Summary
ax = axes[1, 1]
ax.axis('off')
ax.set_title('(d) Results Summary', fontsize=12)

summary_lines = [
    f'PHASE 1: CMB First Peak',
    f'  ΛCDM ell_1 computed' if cl_LCDM is not None else '',
    f'  Target: ℓ₁ = 220.0',
    f'',
    f'PHASE 2: Sound Horizon',
    f'  ΛCDM r_d = {rs_lcdm:.2f} Mpc',
    f'  Baryons r_d = {rs_baryons:.2f} Mpc' if rs_baryons else '',
    f'',
    f'PHASE 3: Peak Heights',
]

if cl_LCDM is not None and len(peaks_LCDM) >= 2:
    summary_lines.append(f'  ΛCDM R₁₂ = {R12_LCDM:.3f}')
if cl_A is not None and len(peaks_A) >= 2:
    summary_lines.append(f'  Baryons R₁₂ = {R12_A:.3f}')
if cl_C is not None and len(peaks_C) >= 2:
    summary_lines.append(f'  α² enh. R₁₂ = {R12_C:.3f}')

summary_lines.extend([
    f'',
    f'PHASE 5: Hubble',
    f'  Model D age = {age_D:.2f} Gyr',
    f'  H₀(cascade) = {H0_cascade:.1f} km/s/Mpc',
])

ax.text(0.05, 0.95, '\n'.join(summary_lines), fontsize=10,
        fontfamily='monospace', verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
fig_path = os.path.join(OUTDIR, 'fig8_modified_friedmann_cmb.png')
plt.savefig(fig_path, dpi=200, bbox_inches='tight')
plt.close()
print(f'  Saved: {fig_path}')

# ================================================================
# COMPLETE SUMMARY
# ================================================================
print('\n' + '=' * 72)
print('  COMPLETE RESULTS — THE ANSWER IS 42')
print('=' * 72)
print(f'')
print(f'  PHASE 1 — CMB First Peak:')
if best_match:
    print(f'    Best match: z_act={best_match[0]}, τ_floor={best_match[1]:.3f}')
    print(f'    ℓ₁ = {best_match[3]:.1f} (target: 220)')
if best_p9:
    print(f'    Paper 9 form: z_act={best_p9[0]}, τ_floor={best_p9[1]:.4f}')
    print(f'    ℓ₁ = {best_p9[3]:.1f}')
print(f'')
print(f'  PHASE 2 — Sound Horizon:')
print(f'    ΛCDM: r_d = {rs_lcdm:.2f} Mpc')
if rs_baryons:
    print(f'    Baryons: r_d = {rs_baryons:.2f} Mpc ({rs_baryons/rs_lcdm:.3f}× ΛCDM)')
print(f'')
print(f'  PHASE 3 — Peak Heights:')
if cl_LCDM is not None and cl_A is not None and cl_C is not None:
    if len(peaks_LCDM) >= 3 and len(peaks_A) >= 3 and len(peaks_C) >= 3:
        print(f'    R₁₂: ΛCDM={R12_LCDM:.3f}, Baryons={R12_A:.3f}, '
              f'α²={R12_C:.3f}')
        print(f'    R₁₃: ΛCDM={R13_LCDM:.3f}, Baryons={R13_A:.3f}, '
              f'α²={R13_C:.3f}')
print(f'')
print(f'  PHASE 4 — BAO:')
if rs_baryons:
    print(f'    Baryons r_d / DESI r_d = {rs_baryons/R_D_DESI:.4f}')
print(f'')
print(f'  PHASE 5 — Hubble:')
print(f'    Model D age: {age_D:.3f} Gyr')
print(f'    H₀(cascade): {H0_cascade:.2f} km/s/Mpc')
print(f'    Planck: {H0_PLANCK}, SH0ES: 73.04')
print('=' * 72)
