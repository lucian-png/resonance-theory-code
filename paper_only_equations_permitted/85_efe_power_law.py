#!/usr/bin/env python3
"""
Script 85 — The Lucian Law Applied to Metric Gravity
======================================================
Derivation of the EFE Power-Law Family from the Schwarzschild metric.

The Question: Are the integer power-law exponents of general relativity
(r_s ∝ M¹, S ∝ M², T_H ∝ M⁻¹, t_evap ∝ M³, K ∝ M⁻⁴, α_g ∝ M²)
a CONSEQUENCE of the Lucian Law's cascade architecture operating on
the Schwarzschild metric?

The System: Schwarzschild metric in dimensionless form:
    ds² = −(1 − 1/ξ)c²dt² + (1 − 1/ξ)⁻¹dr² + r²dΩ²
    where ξ = r/r_s is the dimensionless radial coordinate.

Lucian Law Diagnostic:
    C₁ — Boundedness: Metric confined between horizon (ξ=1) and flat (ξ→∞). ✓
    C₂ — Nonlinear fold: 1/ξ term creates rational nonlinearity + pole. ✓
    C₃ — Coupling: M enters through r_s = 2GM/c², drives all scaling. ✓

Four Parts:
    A: Power-Law Family — 6 observables across 83 orders of M
    B: Self-Similarity — f(ξ) = 1−1/ξ and its derivative ratios
    C: Geodesic Cascade — Effective potential, frequency ratios near ISCO
    D: Exponent Architecture — Ratios vs Feigenbaum quantities

Four Figures:
    fig85a_power_law_family.png      — 6-panel power-law confirmation
    fig85b_metric_self_similarity.png — 4-panel self-similarity analysis
    fig85c_geodesic_cascade.png      — 4-panel geodesic dynamics
    fig85d_exponent_architecture.png  — 2-panel exponent analysis

Uses NO approximations. Every formula is exact Schwarzschild.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.optimize import brentq
import time


# ==============================================================================
# Physical Constants (SI)
# ==============================================================================

G = 6.67430e-11       # m³/(kg·s²)
c = 2.99792458e8      # m/s
hbar = 1.054571817e-34 # J·s
k_B = 1.380649e-23    # J/K
M_sun = 1.98892e30    # kg

FEIGENBAUM_DELTA = 4.669201609102990671853203820466
FEIGENBAUM_ALPHA = 2.502907875095892822283902873218
LN_DELTA = np.log(FEIGENBAUM_DELTA)  # 1.5410...


# ==============================================================================
# Standard Colors and Style
# ==============================================================================

COLORS = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#e67e22', '#1abc9c']


# ==============================================================================
# PART A: The Power-Law Family
# ==============================================================================

def compute_observables(M):
    """
    Compute six observables from the Schwarzschild metric.
    All exact — no approximations.

    Returns: (r_s, S, T_H, t_evap, K_kretschner, alpha_g)
    """
    # 1. Schwarzschild radius: r_s = 2GM/c²
    r_s = 2.0 * G * M / (c * c)

    # 2. Bekenstein-Hawking entropy: S = 4πG²M²/(ℏc) [in units of k_B]
    #    S = A/(4l_P²) = 4π(2GM/c²)²/(4ℏG/c³) = 4πG²M²/(ℏc³) · c²
    #    More precisely: S/k_B = 4πG M² / (ℏc)
    #    Let me use the standard form: S = (4π G M²) / (ℏ c)
    #    Actually: S = k_B c³ A / (4Gℏ), A = 4π r_s² = 16πG²M²/c⁴
    #    S/k_B = c³ · 16πG²M²/(c⁴) / (4Gℏ) = 4πGM²/(ℏc)
    S = 4.0 * np.pi * G * M * M / (hbar * c)

    # 3. Hawking temperature: T_H = ℏc³/(8πGMk_B)
    T_H = hbar * c**3 / (8.0 * np.pi * G * M * k_B)

    # 4. Evaporation time: t_evap = 5120π G²M³/(ℏc⁴)
    t_evap = 5120.0 * np.pi * G**2 * M**3 / (hbar * c**4)

    # 5. Kretschner scalar at r = 2r_s:
    #    K = 48G²M²/(c⁴r⁶), r = 2r_s = 4GM/c²
    #    K = 48G²M² / (c⁴ · (4GM/c²)⁶)
    #      = 48G²M² / (c⁴ · 4⁶G⁶M⁶/c¹²)
    #      = 48 c⁸ / (4096 G⁴ M⁴)
    K = 48.0 * c**8 / (4096.0 * G**4 * M**4)

    # 6. Gravitational coupling: α_g = GM²/(ℏc) for particle of mass M
    alpha_g = G * M * M / (hbar * c)

    return r_s, S, T_H, t_evap, K, alpha_g


def part_A(M_array):
    """Compute power-law family across all masses."""
    N = len(M_array)
    r_s = np.zeros(N)
    S = np.zeros(N)
    T_H = np.zeros(N)
    t_evap = np.zeros(N)
    K = np.zeros(N)
    alpha_g = np.zeros(N)

    for i, M in enumerate(M_array):
        r_s[i], S[i], T_H[i], t_evap[i], K[i], alpha_g[i] = compute_observables(M)

    return r_s, S, T_H, t_evap, K, alpha_g


# ==============================================================================
# PART B: Self-Similarity of the Metric
# ==============================================================================

def metric_function(xi):
    """f(ξ) = 1 − 1/ξ, the Schwarzschild metric component."""
    return 1.0 - 1.0 / xi


def metric_derivative_1(xi):
    """f'(ξ) = 1/ξ²"""
    return 1.0 / (xi * xi)


def metric_derivative_2(xi):
    """f''(ξ) = −2/ξ³"""
    return -2.0 / (xi * xi * xi)


def part_B(xi_array):
    """Self-similarity analysis of f(ξ)."""
    f = metric_function(xi_array)
    fp = metric_derivative_1(xi_array)
    fpp = metric_derivative_2(xi_array)

    # Ratio functions
    ratio_1 = fp / f       # f'/f = 1/(ξ²(1-1/ξ)) = 1/(ξ²-ξ) = 1/(ξ(ξ-1))
    ratio_2 = fpp / fp     # f''/f' = -2/ξ

    return f, fp, fpp, ratio_1, ratio_2


# ==============================================================================
# PART C: Geodesic Cascade
# ==============================================================================

def effective_potential(xi, L):
    """
    Effective potential for massive particle geodesic in Schwarzschild.
    V_eff(ξ) = -1/(2ξ) + L²/(2ξ²) - L²/(2ξ³)
    where ξ = r/r_s, L = dimensionless angular momentum.
    """
    return -0.5 / xi + 0.5 * L**2 / xi**2 - 0.5 * L**2 / xi**3


def dVeff_dxi(xi, L):
    """dV_eff/dξ"""
    return 0.5 / xi**2 - L**2 / xi**3 + 1.5 * L**2 / xi**4


def d2Veff_dxi2(xi, L):
    """d²V_eff/dξ²"""
    return -1.0 / xi**3 + 3.0 * L**2 / xi**4 - 6.0 * L**2 / xi**5


def circular_orbit_radius(L):
    """
    Radii of circular orbits for given L.
    From dV/dξ = 0: ξ² - 2L²ξ + 3L² = 0
    ξ = L² ± √(L⁴ - 3L²)

    Returns: (xi_stable, xi_unstable) or (None, None) if no circular orbits.
    """
    disc = L**4 - 3.0 * L**2
    if disc < 0:
        return None, None

    sqrt_disc = np.sqrt(disc)
    xi_stable = L**2 + sqrt_disc     # outer (stable)
    xi_unstable = L**2 - sqrt_disc   # inner (unstable)

    return xi_stable, xi_unstable


def frequency_ratio(xi_0):
    """
    Ratio ω_r/ω_φ for circular orbit at ξ₀.
    ω_r/ω_φ = √(1 − 3/ξ₀)

    This is the epicyclic-to-orbital frequency ratio.
    At ISCO (ξ₀ = 3): ratio = 0
    At ξ₀ → ∞: ratio → 1 (Keplerian)
    """
    if xi_0 <= 3.0:
        return 0.0
    return np.sqrt(1.0 - 3.0 / xi_0)


def precession_per_orbit(xi_0):
    """
    Perihelion advance per orbit: Δφ = 2π(1 − ω_r/ω_φ)
    For weak field: Δφ ≈ 3π/ξ₀ = 6πGM/(rc²) (Einstein result)
    """
    fr = frequency_ratio(xi_0)
    if fr == 0:
        return 2.0 * np.pi  # one full extra revolution at ISCO
    return 2.0 * np.pi * (1.0 - fr)


def L_from_xi(xi_0):
    """Angular momentum for circular orbit at ξ₀."""
    return np.sqrt(xi_0**2 / (2.0 * xi_0 - 3.0))


def part_C():
    """Geodesic cascade analysis."""

    # L_ISCO = √3
    L_ISCO = np.sqrt(3.0)

    # Five L values spanning from far field to ISCO
    L_values = [10.0, 5.0, 3.0, 2.0, L_ISCO * 1.001]

    # Frequency ratio as function of ξ₀ (stable circular orbit radius)
    xi_circ = np.linspace(3.01, 100.0, 1000)
    freq_ratios = np.array([frequency_ratio(x) for x in xi_circ])
    L_of_xi = np.array([L_from_xi(x) for x in xi_circ])
    prec_of_xi = np.array([precession_per_orbit(x) for x in xi_circ])

    # Approach to ISCO: log scaling of properties
    # As L → L_ISCO, the stable and unstable orbits merge
    eps_values = np.logspace(-6, 0, 200)  # L - L_ISCO
    L_approach = L_ISCO + eps_values

    xi_stable_approach = []
    xi_unstable_approach = []
    gap_approach = []
    freq_ratio_approach = []

    for L in L_approach:
        xi_s, xi_u = circular_orbit_radius(L)
        if xi_s is not None:
            xi_stable_approach.append(xi_s)
            xi_unstable_approach.append(xi_u)
            gap_approach.append(xi_s - xi_u)
            freq_ratio_approach.append(frequency_ratio(xi_s))
        else:
            xi_stable_approach.append(np.nan)
            xi_unstable_approach.append(np.nan)
            gap_approach.append(np.nan)
            freq_ratio_approach.append(np.nan)

    return {
        'L_ISCO': L_ISCO,
        'L_values': L_values,
        'xi_circ': xi_circ,
        'freq_ratios': freq_ratios,
        'L_of_xi': L_of_xi,
        'prec_of_xi': prec_of_xi,
        'eps_values': eps_values,
        'L_approach': L_approach,
        'xi_stable': np.array(xi_stable_approach),
        'xi_unstable': np.array(xi_unstable_approach),
        'gap': np.array(gap_approach),
        'freq_ratio_approach': np.array(freq_ratio_approach),
    }


# ==============================================================================
# PART D: Exponent Architecture
# ==============================================================================

def part_D():
    """Analyze exponent ratios vs Feigenbaum quantities."""

    exponents = {
        r'$r_s$': 1,
        r'$S$': 2,
        r'$T_H$': -1,
        r'$t_{evap}$': 3,
        r'$K$': -4,
        r'$\alpha_g$': 2,
    }

    # All unique exponents
    unique_exp = sorted(set(exponents.values()))

    # Feigenbaum-derived quantities
    feig_quantities = {
        r'$\delta$': FEIGENBAUM_DELTA,
        r'$\alpha$': FEIGENBAUM_ALPHA,
        r'$\ln(\delta)$': LN_DELTA,
        r'$\delta/\alpha$': FEIGENBAUM_DELTA / FEIGENBAUM_ALPHA,
        r'$1/\delta$': 1.0 / FEIGENBAUM_DELTA,
        r'$1/\alpha$': 1.0 / FEIGENBAUM_ALPHA,
        r'$\ln(\alpha)$': np.log(FEIGENBAUM_ALPHA),
        r'$2\ln(\delta)$': 2 * LN_DELTA,
    }

    # All pairwise ratios of exponents
    exp_names = list(exponents.keys())
    exp_vals = list(exponents.values())
    ratios = {}
    for i in range(len(exp_vals)):
        for j in range(len(exp_vals)):
            if i != j and exp_vals[j] != 0:
                ratio = exp_vals[i] / exp_vals[j]
                key = f'{exp_names[i]}/{exp_names[j]}'
                ratios[key] = ratio

    # Find closest Feigenbaum match for each ratio
    matches = []
    for rname, rval in ratios.items():
        best_match = None
        best_err = float('inf')
        for fname, fval in feig_quantities.items():
            if fval != 0:
                err = abs(rval - fval) / abs(fval) * 100
                if err < best_err:
                    best_err = err
                    best_match = fname
        if best_err < 10.0:  # within 10%
            matches.append((rname, rval, best_match,
                          feig_quantities[best_match], best_err))

    return exponents, unique_exp, feig_quantities, ratios, matches


# ==============================================================================
# Main
# ==============================================================================

def main():
    print("=" * 70)
    print("Script 85 — The Lucian Law Applied to Metric Gravity")
    print("Derivation of the EFE Power-Law Family")
    print("=" * 70)
    t0 = time.time()

    # ══════════════════════════════════════════════════════════════════════
    # PART A: Power-Law Family
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 50)
    print("  PART A: Power-Law Family")
    print("─" * 50)

    # 1000 mass values, log-spaced across 83 orders
    M_array = np.logspace(-30, 53, 1000)
    log_M = np.log10(M_array)

    r_s, S, T_H, t_evap, K, alpha_g = part_A(M_array)

    # Linear regression on log-log to extract exponents
    obs_names = [r'$r_s$', r'$S$', r'$T_H$', r'$t_{evap}$', r'$K$', r'$\alpha_g$']
    obs_arrays = [r_s, S, T_H, t_evap, K, alpha_g]
    obs_expected = [1, 2, -1, 3, -4, 2]
    obs_units = ['m', r'$k_B$', 'K', 's', r'm$^{-4}$', '1']

    print(f"\n  {'Observable':>12s}  {'Slope':>12s}  {'Expected':>10s}  "
          f"{'Error %':>10s}  {'R²':>12s}")
    print(f"  {'─' * 12}  {'─' * 12}  {'─' * 10}  {'─' * 10}  {'─' * 12}")

    slopes = []
    r_squared = []
    for name, arr, expected in zip(obs_names, obs_arrays, obs_expected):
        log_obs = np.log10(arr)
        # Linear regression
        coeffs = np.polyfit(log_M, log_obs, 1)
        slope = coeffs[0]
        slopes.append(slope)

        # R²
        predicted = np.polyval(coeffs, log_M)
        ss_res = np.sum((log_obs - predicted) ** 2)
        ss_tot = np.sum((log_obs - np.mean(log_obs)) ** 2)
        r2 = 1.0 - ss_res / ss_tot
        r_squared.append(r2)

        err = abs(slope - expected) / abs(expected) * 100
        print(f"  {name:>12s}  {slope:>12.6f}  {expected:>10d}  "
              f"{err:>10.4f}%  {r2:>12.10f}")

    # ══════════════════════════════════════════════════════════════════════
    # FIGURE 85a: Power-Law Family (6 panels)
    # ══════════════════════════════════════════════════════════════════════
    print("\n  Generating fig85a...")

    fig = plt.figure(figsize=(18, 14), facecolor='white')
    gs = GridSpec(2, 3, hspace=0.35, wspace=0.3)

    for idx, (name, arr, expected, color, unit) in enumerate(
            zip(obs_names, obs_arrays, obs_expected, COLORS, obs_units)):
        ax = fig.add_subplot(gs[idx // 3, idx % 3])

        ax.loglog(M_array, arr, '-', color=color, lw=1.5)
        ax.set_xlabel(r'Mass $M$ [kg]', fontsize=10)
        ylabel = f'{name} [{unit}]'
        ax.set_ylabel(ylabel, fontsize=10)

        panel_label = chr(65 + idx)  # A, B, C, D, E, F
        ax.set_title(f'{panel_label}. {name} '
                     f'(slope = {slopes[idx]:.4f}, '
                     f'expected = {expected}, '
                     f'R² = {r_squared[idx]:.10f})',
                     fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.2, which='both')

        # Annotate the power law
        ax.annotate(f'{name} ∝ M^{{{expected}}}',
                   xy=(0.05, 0.92), xycoords='axes fraction',
                   fontsize=9, fontweight='bold', color=color,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                            alpha=0.8))

    fig.suptitle(
        'The EFE Power-Law Family: Six Observables Across 83 Orders of Mass\n'
        r'Schwarzschild metric: $r_s = 2GM/c^2$, exact computation, no approximations',
        fontsize=15, fontweight='bold', y=0.98
    )

    plt.savefig('fig85a_power_law_family.png', dpi=200, bbox_inches='tight',
                facecolor='white')
    plt.close()
    print("  Saved: fig85a_power_law_family.png")

    # ══════════════════════════════════════════════════════════════════════
    # PART B: Self-Similarity of the Metric
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 50)
    print("  PART B: Self-Similarity of the Metric")
    print("─" * 50)

    xi_array = np.logspace(np.log10(1.001), 6, 2000)
    f, fp, fpp, ratio_1, ratio_2 = part_B(xi_array)

    # Analytical forms of the ratios:
    # f'/f = 1/(ξ(ξ-1))
    # f''/f' = -2/ξ
    print(f"\n  Metric function f(ξ) = 1 − 1/ξ")
    print(f"  f'(ξ) = 1/ξ²")
    print(f"  f''(ξ) = −2/ξ³")
    print(f"  f'/f = 1/(ξ(ξ−1))  — has pole at ξ=1 (horizon)")
    print(f"  f''/f' = −2/ξ  — exact power law with exponent −1")

    # The ratio f''/f' = -2/ξ is a PURE power law
    # This means the metric derivatives are exactly self-similar
    # with scaling exponent -1
    print(f"\n  KEY RESULT: f''/f' = −2/ξ")
    print(f"  This is a pure power law. The derivative ratio is EXACTLY")
    print(f"  self-similar with exponent −1. No cascade structure — the")
    print(f"  metric is analytically scale-free.")

    # Self-similarity check: f at different scales
    # Compare f(ξ) and f(aξ) for various scaling factors a
    scale_factors = [2, 5, 10, 100]
    xi_base = np.linspace(1.01, 5.0, 500)

    # ══════════════════════════════════════════════════════════════════════
    # FIGURE 85b: Self-Similarity of the Metric (4 panels)
    # ══════════════════════════════════════════════════════════════════════
    print("\n  Generating fig85b...")

    fig = plt.figure(figsize=(16, 12), facecolor='white')
    gs = GridSpec(2, 2, hspace=0.35, wspace=0.3)

    # Panel A: f(ξ) across extreme range
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.semilogx(xi_array, f, '-', color='#e74c3c', lw=2)
    ax1.axhline(y=0, color='gray', ls=':', alpha=0.5)
    ax1.axhline(y=1, color='gray', ls=':', alpha=0.5, label='Flat space')
    ax1.axvline(x=1, color='black', ls='--', alpha=0.3, label='Horizon (ξ=1)')
    ax1.set_xlabel(r'$\xi = r/r_s$', fontsize=10)
    ax1.set_ylabel(r'$f(\xi) = 1 - 1/\xi$', fontsize=10)
    ax1.set_title('A. Metric Function Across Extreme Range', fontsize=12,
                  fontweight='bold')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.2)

    # Panel B: f'/f ratio
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.loglog(xi_array, np.abs(ratio_1), '-', color='#3498db', lw=2,
               label=r"$|f'/f| = 1/(\xi(\xi-1))$")
    # Overlay 1/ξ² for comparison
    ax2.loglog(xi_array, 1.0 / xi_array**2, '--', color='gray', alpha=0.5,
               label=r'$1/\xi^2$ (for comparison)')
    ax2.set_xlabel(r'$\xi = r/r_s$', fontsize=10)
    ax2.set_ylabel(r"$|f'(\xi)/f(\xi)|$", fontsize=10)
    ax2.set_title("B. Derivative Ratio — Tidal-to-Metric", fontsize=12,
                  fontweight='bold')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.2, which='both')

    # Panel C: f''/f' ratio
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.loglog(xi_array, np.abs(ratio_2), '-', color='#2ecc71', lw=2,
               label=r"$|f''/f'| = 2/\xi$")
    # This is a pure power law — slope should be exactly -1
    ax3.loglog(xi_array, 2.0 / xi_array, '--', color='red', alpha=0.5, lw=1,
               label=r'$2/\xi$ (exact)')
    ax3.set_xlabel(r'$\xi = r/r_s$', fontsize=10)
    ax3.set_ylabel(r"$|f''(\xi)/f'(\xi)|$", fontsize=10)
    ax3.set_title("C. Ratio of Ratios — Pure Power Law (slope = −1)",
                  fontsize=12, fontweight='bold')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.2, which='both')

    # Measure the slope to confirm
    log_xi = np.log10(xi_array[100:])
    log_ratio2 = np.log10(np.abs(ratio_2[100:]))
    slope_ratio = np.polyfit(log_xi, log_ratio2, 1)[0]
    ax3.annotate(f'Measured slope: {slope_ratio:.6f}\nExact: −1.000000',
                xy=(0.60, 0.85), xycoords='axes fraction', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                         alpha=0.9))

    # Panel D: Self-similarity — overlay f(ξ) at different scales
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(xi_base, metric_function(xi_base), '-', color='#e74c3c', lw=2,
             label=r'$f(\xi)$')
    for a, col in zip(scale_factors, ['#3498db', '#2ecc71', '#9b59b6', '#e67e22']):
        # Rescale: plot f(aξ) vs ξ, normalized to [0,1]
        f_scaled = metric_function(a * xi_base)
        ax4.plot(xi_base, f_scaled, '--', color=col, lw=1.5,
                 label=f'$f({a}\\xi)$')

    ax4.set_xlabel(r'$\xi$', fontsize=10)
    ax4.set_ylabel(r'$f(\xi)$', fontsize=10)
    ax4.set_title('D. Metric at Different Scales (NOT self-similar)',
                  fontsize=12, fontweight='bold')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.2)
    ax4.annotate(r'$f(a\xi) \neq g(a) \cdot f(\xi)$ — metric is NOT self-similar'
                 '\nbut its derivative RATIOS are exact power laws',
                xy=(0.03, 0.05), xycoords='axes fraction', fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                         alpha=0.9))

    fig.suptitle(
        r'Self-Similarity Structure of the Schwarzschild Metric $f(\xi) = 1 - 1/\xi$'
        '\nDerivative ratios are exact power laws — analytically scale-free',
        fontsize=15, fontweight='bold', y=0.98
    )

    plt.savefig('fig85b_metric_self_similarity.png', dpi=200, bbox_inches='tight',
                facecolor='white')
    plt.close()
    print("  Saved: fig85b_metric_self_similarity.png")

    # ══════════════════════════════════════════════════════════════════════
    # PART C: Geodesic Cascade
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 50)
    print("  PART C: Geodesic Cascade — Dynamics Near ISCO")
    print("─" * 50)

    geodesic_data = part_C()
    L_ISCO = geodesic_data['L_ISCO']

    print(f"\n  L_ISCO = √3 = {L_ISCO:.6f}")
    print(f"  ξ_ISCO = 3 (r = 6GM/c² = 3r_s)")

    # Check frequency ratio at specific ξ values
    test_xi = [3.5, 4.0, 5.0, 6.0, 10.0, 20.0, 50.0, 100.0]
    print(f"\n  {'ξ₀':>8s}  {'ω_r/ω_φ':>12s}  {'Δφ (rad)':>12s}  {'L':>10s}")
    print(f"  {'─' * 8}  {'─' * 12}  {'─' * 12}  {'─' * 10}")
    for xi in test_xi:
        fr = frequency_ratio(xi)
        dp = precession_per_orbit(xi)
        L = L_from_xi(xi)
        print(f"  {xi:>8.1f}  {fr:>12.6f}  {dp:>12.6f}  {L:>10.4f}")

    # Approach to ISCO: scaling analysis
    # Gap between stable and unstable orbits: Δξ = 2√(L⁴ - 3L²)
    # Near ISCO (L → √3):
    #   L = √3 + ε, L² = 3 + 2√3ε + ε²
    #   L⁴ - 3L² ≈ 12√3ε → Δξ ≈ 2(12√3ε)^(1/2) ∝ ε^(1/2)
    # Critical exponent = 1/2 (saddle-node bifurcation)

    valid = ~np.isnan(geodesic_data['gap'])
    if np.any(valid):
        log_eps = np.log10(geodesic_data['eps_values'][valid])
        log_gap = np.log10(geodesic_data['gap'][valid])
        gap_slope = np.polyfit(log_eps, log_gap, 1)[0]
        print(f"\n  Gap scaling: Δξ ∝ (L − L_ISCO)^{gap_slope:.4f}")
        print(f"  Expected: 0.5 (saddle-node bifurcation)")
        print(f"  This is NOT Feigenbaum — it's a standard saddle-node merger.")

    # Check: does ω_r/ω_φ pass through specific rational values?
    fr_data = geodesic_data['freq_ratios']
    xi_data = geodesic_data['xi_circ']
    print(f"\n  Rational frequency ratios (resonances):")
    for target_ratio in [0.5, 1.0/3, 0.25, 0.2, 1.0/6, 0.125]:
        idx = np.argmin(np.abs(fr_data - target_ratio))
        xi_at = xi_data[idx]
        L_at = L_from_xi(xi_at)
        print(f"    ω_r/ω_φ = {target_ratio:.4f} at ξ₀ = {xi_at:.3f} "
              f"(L = {L_at:.4f})")

    # ══════════════════════════════════════════════════════════════════════
    # FIGURE 85c: Geodesic Cascade (4 panels)
    # ══════════════════════════════════════════════════════════════════════
    print("\n  Generating fig85c...")

    fig = plt.figure(figsize=(16, 12), facecolor='white')
    gs = GridSpec(2, 2, hspace=0.35, wspace=0.3)

    # Panel A: Effective potential at five L values
    ax1 = fig.add_subplot(gs[0, 0])
    xi_pot = np.linspace(1.5, 30.0, 1000)
    L_plot_values = [10.0, 5.0, 3.0, 2.0, L_ISCO * 1.001]
    L_labels = ['10.0', '5.0', '3.0', '2.0', f'{L_ISCO:.3f} (ISCO)']

    for L, label, col in zip(L_plot_values, L_labels,
                              ['#3498db', '#2ecc71', '#e67e22', '#9b59b6', '#e74c3c']):
        V = effective_potential(xi_pot, L)
        ax1.plot(xi_pot, V, '-', color=col, lw=1.5, label=f'L = {label}')

    ax1.axhline(y=0, color='gray', ls=':', alpha=0.3)
    ax1.set_xlabel(r'$\xi = r/r_s$', fontsize=10)
    ax1.set_ylabel(r'$V_{eff}(\xi)$', fontsize=10)
    ax1.set_title('A. Effective Potential at Five Angular Momenta', fontsize=12,
                  fontweight='bold')
    ax1.legend(fontsize=8)
    ax1.set_ylim(-0.15, 0.15)
    ax1.grid(True, alpha=0.2)

    # Panel B: Frequency ratio vs ξ₀
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(xi_data, fr_data, '-', color='#3498db', lw=2)
    ax2.axhline(y=1.0, color='gray', ls=':', alpha=0.5, label='Keplerian')
    ax2.axvline(x=3.0, color='#e74c3c', ls='--', alpha=0.5, label='ISCO (ξ=3)')

    # Mark rational resonances
    for target, label_text in [(0.5, '1:2'), (1.0/3, '1:3'),
                                (0.25, '1:4'), (0.125, '1:8')]:
        idx = np.argmin(np.abs(fr_data - target))
        ax2.plot(xi_data[idx], fr_data[idx], 'o', color='#e74c3c',
                ms=6, mfc='white', mew=1.5, zorder=5)
        ax2.annotate(label_text, xy=(xi_data[idx], fr_data[idx]),
                    xytext=(5, 5), textcoords='offset points', fontsize=7,
                    color='#e74c3c')

    ax2.set_xlabel(r'$\xi_0$ (stable circular orbit radius / $r_s$)', fontsize=10)
    ax2.set_ylabel(r'$\omega_r / \omega_\varphi$', fontsize=10)
    ax2.set_title(r'B. Frequency Ratio $\omega_r/\omega_\varphi = \sqrt{1-3/\xi_0}$',
                  fontsize=12, fontweight='bold')
    ax2.legend(fontsize=8)
    ax2.set_xlim(3, 50)
    ax2.grid(True, alpha=0.2)

    # Panel C: Gap between stable/unstable orbits vs L-L_ISCO
    ax3 = fig.add_subplot(gs[1, 0])
    valid_mask = ~np.isnan(geodesic_data['gap']) & (geodesic_data['gap'] > 0)
    ax3.loglog(geodesic_data['eps_values'][valid_mask],
               geodesic_data['gap'][valid_mask],
               '-', color='#9b59b6', lw=2)

    # Fit line
    if np.any(valid_mask):
        log_e = np.log10(geodesic_data['eps_values'][valid_mask])
        log_g = np.log10(geodesic_data['gap'][valid_mask])
        gap_coeffs = np.polyfit(log_e, log_g, 1)
        fit_gap = 10 ** np.polyval(gap_coeffs, log_e)
        ax3.loglog(geodesic_data['eps_values'][valid_mask], fit_gap,
                   '--', color='red', alpha=0.5,
                   label=f'slope = {gap_coeffs[0]:.4f} (expected 0.5)')

    ax3.set_xlabel(r'$L - L_{ISCO}$', fontsize=10)
    ax3.set_ylabel(r'$\Delta\xi = \xi_{stable} - \xi_{unstable}$', fontsize=10)
    ax3.set_title('C. Orbit Gap Scaling Near ISCO (Saddle-Node)', fontsize=12,
                  fontweight='bold')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.2, which='both')

    # Panel D: Precession per orbit vs ξ₀
    ax4 = fig.add_subplot(gs[1, 1])
    # Convert to degrees for readability
    prec_deg = geodesic_data['prec_of_xi'] * 180 / np.pi
    ax4.semilogy(xi_data, prec_deg, '-', color='#e67e22', lw=2)
    ax4.axvline(x=3.0, color='#e74c3c', ls='--', alpha=0.5, label='ISCO')

    # Mark Mercury (ξ ≈ 1.5e7 for solar mass, but relative to r_s)
    # Mercury: r ≈ 5.79e10 m, r_s(Sun) = 2.95 km → ξ ≈ 1.96e7
    # Precession ≈ 0.103 arcsec/orbit = 5e-7 rad/orbit
    # Let's annotate the weak-field regime
    ax4.annotate('Weak field:\n' + r'$\Delta\varphi \approx 3\pi/\xi_0$',
                xy=(0.65, 0.80), xycoords='axes fraction', fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                         alpha=0.9))
    ax4.annotate(r'Strong field: $\Delta\varphi \to 2\pi$' + '\n(zoom-whirl)',
                xy=(0.05, 0.80), xycoords='axes fraction', fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                         alpha=0.9))

    ax4.set_xlabel(r'$\xi_0$ (orbit radius / $r_s$)', fontsize=10)
    ax4.set_ylabel(r'Precession per orbit $\Delta\varphi$ [degrees]', fontsize=10)
    ax4.set_title('D. Perihelion Advance — Smooth, No Period-Doubling',
                  fontsize=12, fontweight='bold')
    ax4.legend(fontsize=8)
    ax4.set_xlim(3, 100)
    ax4.grid(True, alpha=0.2)

    fig.suptitle(
        'Geodesic Dynamics in the Schwarzschild Metric\n'
        r'ISCO merger is saddle-node ($\propto \sqrt{\epsilon}$), '
        r'NOT Feigenbaum — integrable system, no period-doubling',
        fontsize=15, fontweight='bold', y=0.98
    )

    plt.savefig('fig85c_geodesic_cascade.png', dpi=200, bbox_inches='tight',
                facecolor='white')
    plt.close()
    print("  Saved: fig85c_geodesic_cascade.png")

    # ══════════════════════════════════════════════════════════════════════
    # PART D: Exponent Architecture
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 50)
    print("  PART D: Exponent Architecture")
    print("─" * 50)

    exponents, unique_exp, feig_quantities, ratios, matches = part_D()

    print(f"\n  Power-law exponents: {unique_exp}")
    print(f"  Feigenbaum δ = {FEIGENBAUM_DELTA:.6f}")
    print(f"  ln(δ) = {LN_DELTA:.6f}")
    print(f"  Feigenbaum α = {FEIGENBAUM_ALPHA:.6f}")

    print(f"\n  Exponent ratios within 10% of Feigenbaum quantities:")
    if matches:
        for rname, rval, fname, fval, err in matches:
            print(f"    {rname:>20s} = {rval:>8.4f}  ≈  "
                  f"{fname} = {fval:.4f}  (Δ = {err:.2f}%)")
    else:
        print("    None found.")

    # The key comparison: 3/2 vs ln(δ)
    print(f"\n  KEY COMPARISON:")
    print(f"    t_evap/S exponent ratio: 3/2 = 1.5000")
    print(f"    ln(δ) = {LN_DELTA:.6f}")
    print(f"    Difference: {abs(1.5 - LN_DELTA) / LN_DELTA * 100:.2f}%")

    # Additional: spacing between consecutive unique exponents
    exp_sorted = sorted(unique_exp)
    print(f"\n  Exponent spacings:")
    spacings = []
    for i in range(1, len(exp_sorted)):
        sp = exp_sorted[i] - exp_sorted[i - 1]
        spacings.append(sp)
        print(f"    {exp_sorted[i-1]} → {exp_sorted[i]}: spacing = {sp}")
    print(f"  Spacings: {spacings}")

    # Check if spacings are related to Feigenbaum
    print(f"\n  Spacing ratios:")
    for i in range(1, len(spacings)):
        if spacings[i - 1] != 0:
            ratio = spacings[i] / spacings[i - 1]
            print(f"    spacing[{i}]/spacing[{i-1}] = "
                  f"{spacings[i]}/{spacings[i-1]} = {ratio:.4f}")

    # ══════════════════════════════════════════════════════════════════════
    # FIGURE 85d: Exponent Architecture (2 panels)
    # ══════════════════════════════════════════════════════════════════════
    print("\n  Generating fig85d...")

    fig = plt.figure(figsize=(16, 7), facecolor='white')
    gs = GridSpec(1, 2, wspace=0.3)

    # Panel A: Exponents on number line
    ax1 = fig.add_subplot(gs[0, 0])

    exp_labels_sorted = [r'$K$', r'$T_H$', r'$r_s$', r'$S, \alpha_g$', r'$t_{evap}$']
    exp_vals_sorted = [-4, -1, 1, 2, 3]

    for i, (label, val) in enumerate(zip(exp_labels_sorted, exp_vals_sorted)):
        ax1.plot(val, 0, 'o', color=COLORS[i % len(COLORS)], ms=15, zorder=5)
        ax1.annotate(label, xy=(val, 0), xytext=(0, 20),
                    textcoords='offset points', ha='center', fontsize=11,
                    fontweight='bold', color=COLORS[i % len(COLORS)])
        ax1.annotate(f'{val}', xy=(val, 0), xytext=(0, -25),
                    textcoords='offset points', ha='center', fontsize=10)

    # Draw spacings
    for i in range(1, len(exp_vals_sorted)):
        x1, x2 = exp_vals_sorted[i - 1], exp_vals_sorted[i]
        mid = (x1 + x2) / 2
        sp = x2 - x1
        ax1.annotate('', xy=(x2, -0.015), xytext=(x1, -0.015),
                    arrowprops=dict(arrowstyle='<->', color='gray', lw=1.5))
        ax1.annotate(f'Δ={sp}', xy=(mid, -0.015), xytext=(0, -15),
                    textcoords='offset points', ha='center', fontsize=8,
                    color='gray')

    ax1.axhline(y=0, color='black', lw=0.5)
    ax1.set_xlim(-5.5, 4.5)
    ax1.set_ylim(-0.08, 0.08)
    ax1.set_xlabel('Power-Law Exponent', fontsize=12)
    ax1.set_yticks([])
    ax1.set_title('A. Exponent Number Line — Integer Positions',
                  fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.2, axis='x')

    # Panel B: Exponent ratios vs Feigenbaum quantities
    ax2 = fig.add_subplot(gs[0, 1])

    # Plot all pairwise ratios
    ratio_vals = list(ratios.values())
    ratio_names_short = list(ratios.keys())
    unique_ratios = sorted(set([abs(r) for r in ratio_vals]))

    # Mark Feigenbaum quantities as horizontal bands
    feig_plot = {
        r'$\ln(\delta)$ = 1.541': LN_DELTA,
        r'$1/\alpha$ = 0.400': 1.0 / FEIGENBAUM_ALPHA,
        r'$\delta/\alpha$ = 1.866': FEIGENBAUM_DELTA / FEIGENBAUM_ALPHA,
        r'$\ln(\alpha)$ = 0.917': np.log(FEIGENBAUM_ALPHA),
    }

    y_pos = np.arange(len(unique_ratios))
    ax2.barh(y_pos, unique_ratios, height=0.6, color='#3498db', alpha=0.7)

    for label, val in feig_plot.items():
        ax2.axvline(x=val, color='#e74c3c', ls='--', alpha=0.5, lw=1.5,
                    label=label)

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([f'{r:.3f}' for r in unique_ratios], fontsize=8)
    ax2.set_xlabel('|Exponent Ratio|', fontsize=10)
    ax2.set_title('B. Exponent Ratios vs Feigenbaum Quantities',
                  fontsize=12, fontweight='bold')
    ax2.legend(fontsize=7, loc='lower right')
    ax2.grid(True, alpha=0.2, axis='x')

    # Highlight the 3/2 ≈ ln(δ) match
    idx_15 = unique_ratios.index(1.5) if 1.5 in unique_ratios else None
    if idx_15 is not None:
        ax2.barh(idx_15, 1.5, height=0.6, color='#e74c3c', alpha=0.7)
        ax2.annotate(f'3/2 = 1.500 ≈ ln(δ) = {LN_DELTA:.4f}\n'
                     f'(Δ = {abs(1.5 - LN_DELTA)/LN_DELTA*100:.1f}%)',
                    xy=(1.5, idx_15), xytext=(50, 20),
                    textcoords='offset points', fontsize=8,
                    arrowprops=dict(arrowstyle='->', color='#e74c3c'),
                    color='#e74c3c', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                             alpha=0.9))

    fig.suptitle(
        'The Exponent Architecture of General Relativity\n'
        'Integer exponents (1, 2, −1, 3, −4, 2) — are they locked by '
        r'Feigenbaum $\delta$?',
        fontsize=15, fontweight='bold', y=1.02
    )

    plt.savefig('fig85d_exponent_architecture.png', dpi=200, bbox_inches='tight',
                facecolor='white')
    plt.close()
    print("  Saved: fig85d_exponent_architecture.png")

    # ══════════════════════════════════════════════════════════════════════
    # FINAL VERDICT
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("FINAL VERDICT — Script 85")
    print(f"{'=' * 70}")

    print(f"""
  PART A — Power-Law Family: CONFIRMED
    All six observables are exact power laws (R² = 1.000000000x)
    across 83 orders of magnitude. The exponents are exactly
    (1, 2, -1, 3, -4, 2) — integers, no deviations.

  PART B — Self-Similarity: CONFIRMED (but trivial)
    The metric derivative ratios are exact power laws:
    f''/f' = -2/ξ (exponent -1, exactly).
    This is ANALYTIC self-similarity — the Schwarzschild metric
    is scale-free by construction (rational function of ξ).
    No cascade fingerprint — the self-similarity is too clean.

  PART C — Geodesic Cascade: NOT FOUND
    The Schwarzschild geodesic is INTEGRABLE. The ISCO merger
    is a standard saddle-node bifurcation (gap ∝ ε^0.5), NOT
    Feigenbaum (which requires non-integrability for period-doubling).
    The frequency ratio ω_r/ω_φ = √(1 - 3/ξ₀) is smooth and
    monotonic — no cascade structure.
    Honest result: no period-doubling in pure Schwarzschild.

  PART D — Exponent Architecture: ONE NEAR-MATCH
    The ratio t_evap/S = 3/2 = 1.500 is within 2.7% of ln(δ) = {LN_DELTA:.4f}.
    This is suggestive but not conclusive. With only six exponents
    and many possible Feigenbaum-derived quantities, a ~3% match
    could be coincidental.

  CLASSIFICATION:
    The Schwarzschild metric SATISFIES the Lucian Law prerequisites
    (C₁ bounded, C₂ nonlinear fold, C₃ coupling through M).
    The power-law family is confirmed exactly.
    But the geodesic dynamics are integrable — no Feigenbaum cascade.
    The law classifies gravity. It does not (yet) produce it.
    The cascade may appear in NON-integrable extensions: Kerr geodesics
    in external fields, self-force corrections, or full numerical GR.
""")

    total_time = time.time() - t0
    print(f"  Total computation time: {total_time:.1f}s")
    print(f"\n{'=' * 70}")
    print("Script 85 complete.")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
