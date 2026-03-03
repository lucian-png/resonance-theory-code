#!/usr/bin/env python3
"""
Script 58: CMB PEAK HEIGHT RATIO
==================================
The one number that tests whether time emergence can explain dark matter.

The CMB peak POSITIONS (Script 48) depend on the sound horizon and angular
diameter distance — both spatial. tau modifies them minimally. CONFIRMED.

The CMB peak HEIGHTS encode the matter content of the universe:
  - Dark matter deepens gravitational potential wells
  - Deeper wells → stronger baryon compression → enhanced odd peaks
  - The first-to-second peak height ratio tells you Omega_m / Omega_b

ΛCDM: Omega_m = 0.315, Omega_b = 0.049 → h1/h2 ~ 2.0-2.5
Baryon-only: Omega_m = Omega_b → h1/h2 ~ ???
tau-model: baryons + time emergence → h1/h2 ~ ???

The key physics is the gravitational potential transfer function T(k).
Without dark matter, potentials decay during radiation domination.
Dark matter sustains them. Time emergence must either:
  (a) Sustain potentials through enhanced coupling, or
  (b) Produce a different observable signature

This script computes the analytical peak height ratio for all three cases.

Author: Lucian Randolph
Date: March 2, 2026
"""

import numpy as np
from scipy.integrate import solve_ivp, quad
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
import warnings
warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.abspath(__file__))

# ==================================================================
# CONSTANTS
# ==================================================================
DELTA = 4.669201609102990
C_LIGHT = 299792.458      # km/s
H0 = 67.4                 # km/s/Mpc
h = H0 / 100.0            # 0.674
OMEGA_M = 0.315            # Total matter (ΛCDM)
OMEGA_B = 0.0493           # Baryonic matter
OMEGA_R = 9.15e-5          # Radiation
OMEGA_L = 0.685            # Dark energy (ΛCDM)
T_CMB = 2.7255             # K
Z_STAR = 1089.92
Z_EQ_LCDM = OMEGA_M / OMEGA_R   # ~ 3442
Z_EQ_BARY = OMEGA_B / OMEGA_R   # ~ 539

# Photon density
OMEGA_GAMMA = 2.469e-5 / h**2

# Time emergence
Z_T = 1.449
BETA = np.log(DELTA)
TAU_FLOOR = 0.936847       # From Script 48

print("=" * 70)
print("SCRIPT 58: CMB PEAK HEIGHT RATIO")
print("Can time emergence replace dark matter in the CMB?")
print("=" * 70)
print(f"  Ω_m = {OMEGA_M},  Ω_b = {OMEGA_B}")
print(f"  z_eq(ΛCDM) = {Z_EQ_LCDM:.0f}")
print(f"  z_eq(baryon-only) = {Z_EQ_BARY:.0f}")
print(f"  z* = {Z_STAR}")
print(f"  τ_floor = {TAU_FLOOR}")


# ==================================================================
# PART 1: THE BARYON LOADING — SAME FOR ALL MODELS
# ==================================================================
print("\n" + "=" * 70)
print("PART 1: BARYON LOADING AT RECOMBINATION")
print("=" * 70)

# R = 3ρ_b / (4ρ_γ) = (3Ω_b) / (4Ω_γ) / (1+z)
R_star = 3.0 * OMEGA_B / (4.0 * OMEGA_GAMMA) / (1.0 + Z_STAR)
print(f"\n  Baryon-photon momentum ratio at z* = {Z_STAR}:")
print(f"  R* = 3Ω_b/(4Ω_γ)/(1+z*) = {R_star:.4f}")
print(f"\n  Sound speed: c_s = c/√(3(1+R*)) = c × {1.0/np.sqrt(3*(1+R_star)):.4f}")
print(f"\n  R* is the SAME for ΛCDM and baryon-only models")
print(f"  (both have the same Ω_b and Ω_γ)")


# ==================================================================
# PART 2: MATTER-RADIATION EQUALITY AND POTENTIAL DECAY
# ==================================================================
print("\n" + "=" * 70)
print("PART 2: THE POTENTIAL TRANSFER FUNCTION")
print("=" * 70)

# The gravitational potential Ψ after entering the horizon:
# In matter era: Ψ = constant (Ψ_0 × 9/10)
# In radiation era: Ψ decays as [sin(x)-x*cos(x)]/x³ where x = kη/√3
# The transition depends on when the mode enters vs z_eq

# Eisenstein & Hu (1998) transfer function:
# This is the ratio of the sub-horizon matter perturbation δ(k,z)
# to the primordial perturbation, normalized at large scales.
# T(k) = δ(k)/δ(k→0) as a function of k

def transfer_EH98(k_mpc, omega_m, omega_b):
    """
    Eisenstein & Hu (1998) fitting formula for the matter transfer function.
    k_mpc: wavenumber in h/Mpc
    omega_m, omega_b: physical density parameters Ωh²
    Returns T(k) normalized to 1 at k→0.
    """
    theta_27 = T_CMB / 2.7  # 1.0094

    # Equality scale
    z_eq = 2.5e4 * omega_m * theta_27**(-4)
    k_eq = 7.46e-2 * omega_m * theta_27**(-2)  # Mpc^-1

    # Sound horizon
    b1 = 0.313 * omega_m**(-0.419) * (1 + 0.607 * omega_m**0.674)
    b2 = 0.238 * omega_m**0.223
    z_d = 1291.0 * omega_m**0.251 / (1 + 0.659 * omega_m**0.828) * \
          (1 + b1 * omega_b**b2)

    R_eq = 31.5e3 * omega_b * theta_27**(-4) / z_eq
    R_d = 31.5e3 * omega_b * theta_27**(-4) / z_d
    s = 2.0 / (3.0 * k_eq) * np.sqrt(6.0 / R_eq) * \
        np.log((np.sqrt(1 + R_d) + np.sqrt(R_d + R_eq)) / (1 + np.sqrt(R_eq)))

    # CDM transfer function (zero baryon)
    f_b = omega_b / omega_m     # baryon fraction
    f_c = 1.0 - f_b             # CDM fraction

    a1 = (46.9 * omega_m)**0.670 * (1 + (32.1 * omega_m)**(-0.532))
    a2 = (12.0 * omega_m)**0.424 * (1 + (45.0 * omega_m)**(-0.582))
    alpha_c = a1**(-f_b) * a2**(-(f_b)**3)

    b1_c = 0.944 / (1 + (458 * omega_m)**(-0.708))
    b2_c = (0.395 * omega_m)**(-0.0266)
    # Guard: when f_c → 0 (baryon-only), beta_c is irrelevant (Tc gets weight 0)
    if f_c > 1e-10:
        beta_c = 1.0 / (1 + b1_c * f_c**b2_c - 1)
    else:
        beta_c = 1.0  # placeholder — Tc multiplied by f_c ≈ 0 anyway

    q = k_mpc / (13.41 * k_eq)

    def T0_tilde(k_mpc, alpha, beta_node):
        q_eff = k_mpc / (13.41 * k_eq * alpha)
        C = 14.2 / alpha + 386.0 / (1 + 69.9 * q_eff**1.08)
        T0 = np.log(np.e + 1.8 * beta_node * q_eff) / \
             (np.log(np.e + 1.8 * beta_node * q_eff) + C * q_eff**2)
        return T0

    f = 1.0 / (1 + (k_mpc * s / 5.4)**4)
    Tc = f * T0_tilde(k_mpc, 1, beta_c) + (1 - f) * T0_tilde(k_mpc, alpha_c, 1)

    # Baryon transfer function
    y = z_eq / z_d
    Gy = y * (-6 * np.sqrt(1 + y) + (2 + 3 * y) *
              np.log((np.sqrt(1 + y) + 1) / (np.sqrt(1 + y) - 1)))
    alpha_b = 2.07 * k_eq * s * (1 + R_d)**(-0.75) * Gy

    beta_node = 8.41 * omega_m**0.435
    beta_b = 0.5 + f_b + \
             (3 - 2 * f_b) * np.sqrt((17.2 * omega_m)**2 + 1)

    s_tilde = s / (1 + (beta_node / (k_mpc * s))**3)**(1.0/3.0)

    Tb = T0_tilde(k_mpc, 1, 1) / (1 + (k_mpc * s / 5.2)**2) + \
         alpha_b / (1 + (beta_b / (k_mpc * s))**3) * \
         np.exp(-(k_mpc / (7.68e-2 * np.sqrt(omega_b * s**(-1))))**1.18) * \
         np.sinc(k_mpc * s_tilde / np.pi)  # sinc = sin(x)/x

    T_full = f_b * Tb + f_c * Tc

    return T_full


# Compute k values for first 5 peaks
# Peak n corresponds to k_n ≈ n × π / r_s
# Sound horizon from Script 48: r_s ≈ 145 Mpc (comoving)
R_S_COMOVING = 145.0  # Mpc (approximate)

k_peaks = np.array([n * np.pi / R_S_COMOVING for n in range(1, 8)])  # h⁻¹ Mpc... actually just Mpc⁻¹

print(f"\n  Acoustic peak wavenumbers (k ≈ nπ/r_s):")
print(f"  r_s ≈ {R_S_COMOVING} Mpc (comoving)")
for n in range(1, 8):
    print(f"    Peak {n}: k ≈ {k_peaks[n-1]:.4f} Mpc⁻¹")

# Physical density parameters
omega_m_LCDM = OMEGA_M * h**2    # 0.315 × 0.674² = 0.143
omega_b_phys = OMEGA_B * h**2     # 0.049 × 0.674² = 0.022
omega_m_bary = OMEGA_B * h**2     # Baryon-only: Ω_m = Ω_b

print(f"\n  Physical density parameters (Ωh²):")
print(f"    ΛCDM:  ω_m = {omega_m_LCDM:.4f},  ω_b = {omega_b_phys:.4f}")
print(f"    Bary:  ω_m = {omega_m_bary:.4f},  ω_b = {omega_b_phys:.4f}")

# Transfer functions
print(f"\n  Transfer function T(k) at peak wavenumbers:")
print(f"  {'Peak':>5s}  {'k (Mpc⁻¹)':>10s}  {'T(ΛCDM)':>10s}  {'T(bary)':>10s}  {'Ratio':>8s}")
print(f"  {'---':>5s}  {'---':>10s}  {'---':>10s}  {'---':>10s}  {'---':>8s}")

T_LCDM = []
T_bary = []
for n in range(7):
    k = k_peaks[n]
    t_l = transfer_EH98(k, omega_m_LCDM, omega_b_phys)
    t_b = transfer_EH98(k, omega_m_bary, omega_b_phys)
    T_LCDM.append(t_l)
    T_bary.append(t_b)
    ratio = t_b / t_l if t_l > 0 else 0
    print(f"  {n+1:5d}  {k:10.4f}  {t_l:10.4f}  {t_b:10.4f}  {ratio:8.3f}")


# ==================================================================
# PART 3: PEAK HEIGHTS FROM OSCILLATION EQUATION
# ==================================================================
print("\n" + "=" * 70)
print("PART 3: PEAK HEIGHTS FROM ANALYTICAL OSCILLATION SOLUTION")
print("=" * 70)

# The effective temperature at the surface of last scattering:
# (Θ₀ + Ψ)_n ≈ [(1/3 + R_*)(−1)^(n+1) − R_*] × Ψ(k_n, η_*)
#
# The gravitational potential at recombination depends on the transfer function:
# Ψ(k, η_*) = T(k) × Ψ_primordial(k) × (potential normalization)
#
# For the peak height RATIO, Ψ_primordial cancels if we assume
# a scale-invariant primordial spectrum.
#
# The Silk damping suppresses high-k modes:
# D(k) = exp(−(k/k_D)²) with k_D ≈ 0.10 Mpc⁻¹ (approximate)

k_D = 0.10  # Silk damping scale (approximate), Mpc⁻¹

def peak_height_squared(n, T_k, R_star, k, k_D):
    """
    Compute (Θ₀ + Ψ)² at the nth peak.

    The oscillation solution with baryon loading:
    (Θ₀ + Ψ) = A_n × T(k) × exp(−k²/k_D²)

    where A_n depends on whether it's a compression or rarefaction.

    For compression (odd n):
      A_odd = (1/3 + 2R_*) [shifted zero-point adds constructively]

    For rarefaction (even n):
      A_even = (1/3) [shifted zero-point subtracts]

    This is the Hu & Sugiyama formulation for the zero-point offset.
    """
    damping = np.exp(-2.0 * (k / k_D)**2)

    if n % 2 == 1:  # Compression (odd peak)
        A = (1.0/3.0 + 2.0 * R_star)
    else:  # Rarefaction (even peak)
        A = 1.0/3.0

    return A**2 * T_k**2 * damping


# But the above is the SIMPLEST model. The real enhancement from
# dark matter comes through how T(k) is computed — with dark matter,
# the potential is sustained and T(k) is larger.

# Additionally, there's the "radiation driving" effect:
# Modes entering the horizon during radiation domination experience
# a boost because the decaying potential drives the oscillation.
# This boost is ABSENT in the matter era (where Ψ = const).
# Without dark matter, z_eq is lower, so MORE modes experience
# radiation driving — this actually BOOSTS the baryon-only peaks!

# The radiation driving enhancement:
# For modes entering during radiation domination (k >> k_eq):
#   The peak height gets an extra factor ~ 5 enhancement from the
#   resonant driving as the potential decays in phase with the oscillation.
# For modes entering during matter domination (k << k_eq):
#   No enhancement.

# Equality wavenumber
k_eq_LCDM = 0.073 * omega_m_LCDM  # Mpc⁻¹ (approximate)
k_eq_bary = 0.073 * omega_m_bary   # Mpc⁻¹

print(f"\n  Equality wavenumber:")
print(f"    ΛCDM:  k_eq ≈ {k_eq_LCDM:.4f} Mpc⁻¹")
print(f"    Bary:  k_eq ≈ {k_eq_bary:.4f} Mpc⁻¹")

def radiation_driving_boost(k, k_eq):
    """
    Enhancement factor from radiation driving.
    Modes with k >> k_eq enter during radiation era and get boosted.
    Approximate as a smooth transition.
    Based on Hu & Sugiyama (1996) Fig. 1.
    """
    x = k / k_eq
    if x < 0.5:
        return 1.0  # matter era, no boost
    elif x > 5.0:
        return 5.0  # full radiation driving boost (approximate)
    else:
        # Smooth interpolation
        return 1.0 + 4.0 * (1 - np.exp(-(x - 0.5)))


print(f"\n  Radiation driving boost at each peak:")
print(f"  {'Peak':>5s}  {'k':>8s}  {'Boost(ΛCDM)':>12s}  {'Boost(bary)':>12s}")
print(f"  {'---':>5s}  {'---':>8s}  {'---':>12s}  {'---':>12s}")
for n in range(7):
    k = k_peaks[n]
    b_l = radiation_driving_boost(k, k_eq_LCDM)
    b_b = radiation_driving_boost(k, k_eq_bary)
    print(f"  {n+1:5d}  {k:8.4f}  {b_l:12.2f}  {b_b:12.2f}")


# ==================================================================
# PART 4: COMPUTE PEAK HEIGHT RATIOS
# ==================================================================
print("\n" + "=" * 70)
print("PART 4: THE MONEY — PEAK HEIGHT RATIOS")
print("=" * 70)

# Full peak height: amplitude² × transfer² × damping × radiation_boost²
def compute_heights(T_func_vals, R_star, k_peaks, k_eq, k_D, model_name):
    """Compute peak heights for a given model."""
    heights = []
    for n in range(len(k_peaks)):
        k = k_peaks[n]
        T_k = T_func_vals[n]
        damping = np.exp(-2.0 * (k / k_D)**2)
        rad_boost = radiation_driving_boost(k, k_eq)

        if (n + 1) % 2 == 1:  # Odd peak (compression)
            A = (1.0/3.0 + 2.0 * R_star)
        else:  # Even peak (rarefaction)
            A = 1.0/3.0

        h_sq = A**2 * T_k**2 * damping * rad_boost**2
        heights.append(np.sqrt(h_sq))
    return np.array(heights)

h_LCDM = compute_heights(T_LCDM, R_star, k_peaks, k_eq_LCDM, k_D, "ΛCDM")
h_bary = compute_heights(T_bary, R_star, k_peaks, k_eq_bary, k_D, "Bary")

# Normalize to first peak
h_LCDM_norm = h_LCDM / h_LCDM[0]
h_bary_norm = h_bary / h_bary[0]

print(f"\n  Peak heights (normalized to peak 1):")
print(f"  {'Peak':>5s}  {'ΛCDM':>10s}  {'Bary-only':>10s}  {'Planck~':>10s}")
print(f"  {'---':>5s}  {'---':>10s}  {'---':>10s}  {'---':>10s}")

# Approximate Planck peak heights (from TT spectrum, relative to peak 1)
planck_heights_approx = [1.00, 0.45, 0.49, 0.25, 0.19, 0.10, 0.06]

for n in range(7):
    print(f"  {n+1:5d}  {h_LCDM_norm[n]:10.4f}  {h_bary_norm[n]:10.4f}  "
          f"{planck_heights_approx[n]:10.2f}")

print(f"\n  KEY RATIOS:")
ratio_12_LCDM = h_LCDM[0] / h_LCDM[1]
ratio_12_bary = h_bary[0] / h_bary[1]
ratio_13_LCDM = h_LCDM[0] / h_LCDM[2]
ratio_13_bary = h_bary[0] / h_bary[2]

print(f"    h₁/h₂ (ΛCDM):      {ratio_12_LCDM:.3f}")
print(f"    h₁/h₂ (bary-only):  {ratio_12_bary:.3f}")
print(f"    h₁/h₂ (Planck):     ~{1/0.45:.1f}")
print(f"\n    h₁/h₃ (ΛCDM):      {ratio_13_LCDM:.3f}")
print(f"    h₁/h₃ (bary-only):  {ratio_13_bary:.3f}")


# ==================================================================
# PART 5: tau-MODIFIED MODEL
# ==================================================================
print("\n" + "=" * 70)
print("PART 5: τ-MODIFIED BARYON-ONLY MODEL")
print("=" * 70)

# The hypothesis: τ(z) modifies the effective gravitational coupling.
# If the effective matter density for perturbation growth is:
#   Ω_m,eff = Ω_b / τ²
# Then the effective transfer function uses Ω_m,eff instead of Ω_b.

# Test multiple τ values to see what's needed
print(f"\n  Testing: what τ value gives ΛCDM-like peak ratios?")
print(f"  (Using Ω_m,eff = Ω_b / τ²)")
print(f"\n  {'τ':>8s}  {'Ω_m,eff':>10s}  {'z_eq,eff':>10s}  "
      f"{'h₁/h₂':>8s}  {'Match?':>8s}")
print(f"  {'---':>8s}  {'---':>10s}  {'---':>10s}  {'---':>8s}  {'---':>8s}")

tau_test_values = [0.937, 0.8, 0.6, 0.5, 0.4, 0.39, 0.35, 0.3, 0.2]

for tau_val in tau_test_values:
    omega_m_eff = OMEGA_B / tau_val**2
    omega_m_eff_h2 = omega_m_eff * h**2
    z_eq_eff = omega_m_eff / OMEGA_R

    # Compute transfer function with effective Ω_m
    T_tau = []
    for k in k_peaks:
        t = transfer_EH98(k, omega_m_eff_h2, omega_b_phys)
        T_tau.append(t)

    k_eq_tau = 0.073 * omega_m_eff_h2
    h_tau = compute_heights(T_tau, R_star, k_peaks, k_eq_tau, k_D, f"τ={tau_val}")
    ratio_12_tau = h_tau[0] / h_tau[1]

    match = "✓" if abs(ratio_12_tau - ratio_12_LCDM) / ratio_12_LCDM < 0.1 else ""
    print(f"  {tau_val:8.3f}  {omega_m_eff:10.4f}  {z_eq_eff:10.0f}  "
          f"{ratio_12_tau:8.3f}  {match:>8s}")


# ==================================================================
# PART 6: THE HONEST ASSESSMENT
# ==================================================================
print("\n" + "=" * 70)
print("PART 6: THE HONEST ASSESSMENT")
print("=" * 70)

# What τ value would be needed?
from scipy.optimize import brentq

def ratio_residual(tau_val):
    omega_m_eff = OMEGA_B / tau_val**2
    omega_m_eff_h2 = omega_m_eff * h**2
    T_tau = [transfer_EH98(k, omega_m_eff_h2, omega_b_phys) for k in k_peaks]
    k_eq_tau = 0.073 * omega_m_eff_h2
    h_tau = compute_heights(T_tau, R_star, k_peaks, k_eq_tau, k_D, "")
    return h_tau[0] / h_tau[1] - ratio_12_LCDM

try:
    tau_needed = brentq(ratio_residual, 0.1, 0.95)
    omega_needed = OMEGA_B / tau_needed**2
    z_eq_needed = omega_needed / OMEGA_R
    print(f"\n  To match ΛCDM peak ratio h₁/h₂ = {ratio_12_LCDM:.3f}:")
    print(f"    τ needed:     {tau_needed:.4f}")
    print(f"    Ω_m,eff:      {omega_needed:.4f}  (vs ΛCDM's {OMEGA_M})")
    print(f"    z_eq,eff:      {z_eq_needed:.0f}  (vs ΛCDM's {Z_EQ_LCDM:.0f})")
except Exception:
    tau_needed = None
    print(f"\n  Could not find τ that matches ΛCDM peak ratio")

print(f"\n  COMPARISON:")
print(f"    τ_floor (from peak positions):  {TAU_FLOOR:.4f}")
if tau_needed:
    print(f"    τ needed (for peak heights):    {tau_needed:.4f}")
    print(f"    Ratio τ_floor / τ_needed:       {TAU_FLOOR / tau_needed:.2f}")

    if abs(TAU_FLOOR - tau_needed) / TAU_FLOOR < 0.1:
        print(f"\n  → CONSISTENT! The same τ_floor works for both positions and heights.")
        verdict = "CONSISTENT"
    elif TAU_FLOOR > tau_needed:
        print(f"\n  → TENSION. Peak positions require τ = {TAU_FLOOR:.3f} (nearly emerged),")
        print(f"    but peak heights require τ = {tau_needed:.3f} (much less emerged).")
        print(f"    The simple G_eff = G/τ² prescription cannot satisfy BOTH constraints")
        print(f"    with a single τ value at recombination.")
        verdict = "TENSION"
    else:
        print(f"\n  → τ_heights < τ_positions: modest tension.")
        verdict = "MILD TENSION"

    # What does G_eff = G/τ_floor² give?
    print(f"\n  With τ_floor = {TAU_FLOOR}:")
    omega_m_floor = OMEGA_B / TAU_FLOOR**2
    print(f"    Ω_m,eff = Ω_b/τ² = {omega_m_floor:.4f}")
    print(f"    Compare: Ω_m(ΛCDM) = {OMEGA_M}")
    print(f"    Ratio: {omega_m_floor / OMEGA_M:.3f}")
    print(f"    The τ_floor only boosts effective Ω_m by {(omega_m_floor/OMEGA_B - 1)*100:.1f}%")
    print(f"    Dark matter provides {(OMEGA_M/OMEGA_B - 1)*100:.0f}% more than baryons alone")
else:
    verdict = "UNKNOWN"


# ==================================================================
# PART 7: WHAT THE PROFILE ACTUALLY LOOKS LIKE
# ==================================================================
print("\n" + "=" * 70)
print("PART 7: PREDICTED C_ℓ SHAPE")
print("=" * 70)

# Compute a simplified C_ℓ for the three models
ell_range = np.arange(2, 2500)
k_range = ell_range * np.pi / (R_S_COMOVING * 301.76)  # approximate k(ℓ) scaling

def compute_Cl_simplified(ell_range, omega_m_h2, omega_b_h2, R_star, k_D, label):
    """Simplified C_ℓ from analytical oscillation + transfer function."""
    k_eq = 0.073 * omega_m_h2
    ell_A = 301.76  # acoustic scale

    Cl = np.zeros(len(ell_range))
    for i, ell in enumerate(ell_range):
        # Approximate k for this ℓ
        k = ell / (R_S_COMOVING * ell_A / np.pi)

        # Transfer function
        T_k = transfer_EH98(k, omega_m_h2, omega_b_h2)

        # Sound horizon integral gives phase: krs = ℓ × π/ℓ_A
        phase = ell * np.pi / ell_A

        # Oscillation with baryon loading
        theta_plus_psi = ((1.0/3.0 + R_star) * np.cos(phase) - R_star)

        # Silk damping
        damping = np.exp(-(k / k_D)**2)

        # Radiation driving
        rad_boost = radiation_driving_boost(k, k_eq)

        # Power spectrum
        Cl[i] = (theta_plus_psi * T_k * damping * rad_boost)**2

    # Normalize
    Cl_max = np.max(Cl)
    if Cl_max > 0:
        Cl /= Cl_max
    return Cl

Cl_LCDM = compute_Cl_simplified(ell_range, omega_m_LCDM, omega_b_phys,
                                  R_star, k_D, "ΛCDM")
Cl_bary = compute_Cl_simplified(ell_range, omega_m_bary, omega_b_phys,
                                  R_star, k_D, "Bary")

if tau_needed:
    omega_tau = OMEGA_B / tau_needed**2 * h**2
    Cl_tau_match = compute_Cl_simplified(ell_range, omega_tau, omega_b_phys,
                                          R_star, k_D, "τ-match")
    omega_floor = OMEGA_B / TAU_FLOOR**2 * h**2
    Cl_tau_floor = compute_Cl_simplified(ell_range, omega_floor, omega_b_phys,
                                          R_star, k_D, "τ-floor")


# ==================================================================
# GENERATE FIGURES
# ==================================================================
print("\n" + "=" * 70)
print("GENERATING FIGURES")
print("=" * 70)

fig = plt.figure(figsize=(22, 16))
gs = GridSpec(2, 2, figure=fig, hspace=0.32, wspace=0.28)
fig.suptitle('Script 58: CMB Peak Height Ratio\n'
             'Can Time Emergence Replace Dark Matter?',
             fontsize=16, fontweight='bold', y=0.99)

# ── Panel A: Transfer functions ──
ax1 = fig.add_subplot(gs[0, 0])
k_plot = np.logspace(-3, 0, 500)
T_plot_LCDM = [transfer_EH98(k, omega_m_LCDM, omega_b_phys) for k in k_plot]
T_plot_bary = [transfer_EH98(k, omega_m_bary, omega_b_phys) for k in k_plot]
ax1.loglog(k_plot, T_plot_LCDM, 'b-', linewidth=2.5, label='ΛCDM (Ωm=0.315)')
ax1.loglog(k_plot, T_plot_bary, 'r-', linewidth=2.5, label='Baryons only (Ωm=0.049)')
# Mark peak positions
for n in range(5):
    ax1.axvline(k_peaks[n], color='gray', linestyle=':', alpha=0.3)
    ax1.text(k_peaks[n], 1.05, f'P{n+1}', fontsize=8, ha='center',
             transform=ax1.get_xaxis_transform())
ax1.set_xlabel('Wavenumber k (Mpc⁻¹)', fontsize=12)
ax1.set_ylabel('Transfer Function T(k)', fontsize=12)
ax1.set_title('Panel A: Matter Transfer Function\n'
              'Dark Matter Sustains Small-Scale Potential',
              fontsize=13, fontweight='bold')
ax1.legend(fontsize=10)
ax1.set_xlim(1e-3, 1)
ax1.set_ylim(1e-3, 1.5)
ax1.grid(True, alpha=0.3)

# ── Panel B: Peak heights ──
ax2 = fig.add_subplot(gs[0, 1])
peak_nums = np.arange(1, 8)
ax2.bar(peak_nums - 0.2, h_LCDM_norm, 0.35, color='#2980b9', label='ΛCDM', alpha=0.8)
ax2.bar(peak_nums + 0.2, h_bary_norm, 0.35, color='#e74c3c', label='Bary-only', alpha=0.8)
ax2.plot(peak_nums, planck_heights_approx, 'ko', markersize=10,
         label='Planck (approx)', zorder=10)
ax2.set_xlabel('Peak Number', fontsize=12)
ax2.set_ylabel('Height (normalized to peak 1)', fontsize=12)
ax2.set_title('Panel B: Peak Heights Comparison\n'
              'Even-Odd Asymmetry from Baryon Loading',
              fontsize=13, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3, axis='y')

# ── Panel C: Simplified Cℓ spectra ──
ax3 = fig.add_subplot(gs[1, 0])
ax3.plot(ell_range, Cl_LCDM * ell_range * (ell_range + 1),
         'b-', linewidth=1.5, alpha=0.8, label='ΛCDM')
ax3.plot(ell_range, Cl_bary * ell_range * (ell_range + 1),
         'r-', linewidth=1.5, alpha=0.8, label='Baryons only')
if tau_needed:
    ax3.plot(ell_range, Cl_tau_floor * ell_range * (ell_range + 1),
             'g--', linewidth=1.5, alpha=0.8, label=f'τ_floor = {TAU_FLOOR:.3f}')
ax3.set_xlabel('Multipole ℓ', fontsize=12)
ax3.set_ylabel('ℓ(ℓ+1)C_ℓ (arb.)', fontsize=12)
ax3.set_title('Panel C: Simplified CMB Power Spectrum\n'
              'ℓ(ℓ+1)C_ℓ Shape',
              fontsize=13, fontweight='bold')
ax3.legend(fontsize=9)
ax3.set_xlim(0, 2000)
ax3.grid(True, alpha=0.3)

# ── Panel D: Summary ──
ax4 = fig.add_subplot(gs[1, 1])
ax4.axis('off')

tau_str = f"{tau_needed:.4f}" if tau_needed else "N/A"
summary = (
    f"CMB PEAK HEIGHT RATIO — RESULTS\n"
    f"{'='*44}\n\n"
    f"h₁/h₂ (first-to-second peak):\n"
    f"  ΛCDM:       {ratio_12_LCDM:.3f}\n"
    f"  Bary-only:  {ratio_12_bary:.3f}\n"
    f"  Planck:     ~{1/0.45:.1f}\n\n"
    f"THE GAP:\n"
    f"  Dark matter provides {OMEGA_M/OMEGA_B:.1f}× more mass\n"
    f"  τ_floor = {TAU_FLOOR} gives only\n"
    f"  {(1/TAU_FLOOR**2 - 1)*100:.0f}% boost to effective Ω_m\n\n"
    f"τ NEEDED for peak heights: {tau_str}\n"
    f"τ REQUIRED for peak positions: {TAU_FLOOR:.4f}\n\n"
    f"VERDICT: {verdict}"
)
ax4.text(0.5, 0.5, summary, transform=ax4.transAxes,
         fontsize=10, va='center', ha='center', fontfamily='monospace',
         bbox=dict(boxstyle='round,pad=0.6', facecolor='#fffff0',
                   edgecolor='#d4a843', linewidth=2.5, alpha=0.95))

fig.savefig(os.path.join(BASE, 'fig58_peak_heights.png'),
            dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print(f"\n  Figure saved: fig58_peak_heights.png")


# ==================================================================
# FINAL REPORT
# ==================================================================
print("\n" + "=" * 70)
print("FINAL REPORT")
print("=" * 70)

print(f"""
  THE QUESTION:
    Can τ(z) with τ_floor = {TAU_FLOOR} replace dark matter
    in explaining CMB peak heights?

  THE ANSWER:
    Peak height ratio h₁/h₂:
      ΛCDM:       {ratio_12_LCDM:.3f}  (with dark matter)
      Bary-only:  {ratio_12_bary:.3f}  (without dark matter)
      Planck:     ~{1/0.45:.1f}

    The τ_floor effect on peak heights:
      Ω_m,eff = Ω_b/τ² = {OMEGA_B/TAU_FLOOR**2:.4f}
      This is {OMEGA_B/TAU_FLOOR**2 / OMEGA_M * 100:.1f}% of ΛCDM's Ω_m.
      The 6.3% temporal modification cannot replace
      a 6.4× increase in matter content.
""")

if tau_needed and tau_needed < TAU_FLOOR:
    print(f"  To match peak heights, τ would need to be {tau_needed:.3f}")
    print(f"  at recombination — but peak positions require τ = {TAU_FLOOR:.3f}.")
    print(f"  The simple G_eff = G/τ² model has an internal inconsistency.")
    print(f"")
    print(f"  THIS IS AN HONEST RESULT.")
    print(f"  The paper should:")
    print(f"    1. KEEP the peak position result (confirmed, Section 4)")
    print(f"    2. REPORT the peak height tension (new, Section 6)")
    print(f"    3. Note that a simple G_eff = G/τ² is insufficient")
    print(f"    4. Flag that a more sophisticated mechanism is needed")
    print(f"    5. The S₈ tension IS predicted by τ — partial support")

print(f"\n" + "=" * 70)
print("Script 58 complete.")
print("=" * 70)
