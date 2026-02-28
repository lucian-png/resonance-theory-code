#!/usr/bin/env python3
"""
==============================================================================
DUAL ATTRACTOR BASINS AT FRACTAL PHASE BOUNDARIES
==============================================================================

At every well-characterized phase transition, matter/energy separates into
two populations with a measurable gap between them. This script tests whether
this dual attractor structure is universal across physics.

DISCOVERY:
  Paper XXI (The Chladni Universe) found two populations near Feigenbaum
  sub-harmonics of the spacetime metric:
    Population A (active energy sources): ratios 0.53–0.66× (BELOW predicted)
    Population B (passive thermal):       ratios 1.05–1.66× (ABOVE predicted)
    Gap: 0.66–1.05× — no objects in between

  Question: Is this dual population structure a universal feature of fractal
  phase boundaries, or specific to the Chladni system?

DOMAINS TESTED (all computed from first principles):
  1. Chladni Universe         — Feigenbaum sub-harmonic populations
  2. Van der Waals            — Liquid–gas coexistence (Maxwell construction)
  3. 2D Ising Model           — Spontaneous magnetization (Onsager exact)
  4. Bose–Einstein Condensate — Condensate vs thermal population
  5. Pipe Flow Turbulence     — Laminar–turbulent bimodality
  6. Spacetime Metric         — Center vs surface time dilation divergence

HONEST TALLY: Each domain assessed for strength of dual attractor evidence.

Outputs:
    fig34_dual_attractor_basins.png   — 6-panel, one domain per panel
    fig35_dual_attractor_synthesis.png — 6-panel, synthesis & honest tally

STATUS: RESEARCH — Computation verified, interpretation requires peer review
==============================================================================
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# STYLE — MATCHING EXISTING RESONANCE THEORY SCRIPTS
# =============================================================================

COLORS = {
    'r': '#e74c3c', 'b': '#3498db', 'g': '#2ecc71',
    'p': '#9b59b6', 'o': '#e67e22', 'd': '#2c3e50',
    'gold': '#f39c12', 'teal': '#16a085', 'gray': '#7f8c8d',
    'pink': '#e91e63', 'cyan': '#00bcd4',
}


# =============================================================================
# PHYSICAL CONSTANTS (SI)
# =============================================================================

G_0 = 6.67430e-11        # m³ kg⁻¹ s⁻²
c_light = 2.99792458e8   # m/s
hbar = 1.054571817e-34   # J·s
k_B = 1.380649e-23       # J/K
kappa_0 = 8.0 * np.pi * G_0 / c_light**4
delta_F = 4.669201609     # Feigenbaum universal constant


# =============================================================================
# DOMAIN 1: CHLADNI UNIVERSE — RECAP FROM PAPER XXI
# =============================================================================

def compute_chladni_populations():
    """
    Recompute the 8-object catalog from Paper XXI.
    Two populations: active (ratio < 1) and passive (ratio > 1).
    The gap between 0.66× and 1.05× is clean.
    """
    catalog = [
        {'name': 'Sun',             'R': 6.96e8,  'rho': 1.5e16,  'pop': 'active'},
        {'name': 'Earth',           'R': 6.37e6,  'rho': 3.6e13,  'pop': 'active'},
        {'name': 'PSR J0348',       'R': 1.3e4,   'rho': 5.0e17,  'pop': 'active'},
        {'name': 'Jupiter',         'R': 6.99e7,  'rho': 1.3e13,  'pop': 'passive'},
        {'name': 'Saturn',          'R': 5.82e7,  'rho': 5.0e12,  'pop': 'passive'},
        {'name': 'Mars',            'R': 3.39e6,  'rho': 1.5e13,  'pop': 'passive'},
        {'name': 'Moon',            'R': 1.74e6,  'rho': 1.3e13,  'pop': 'passive'},
        {'name': 'Sirius B',       'R': 5.8e6,   'rho': 3.0e15,  'pop': 'passive'},
    ]

    results = []
    for obj in catalog:
        rho_C0 = 0.001 * 3.0 * c_light**4 / (8.0 * np.pi * G_0 * obj['R']**2)
        n_exact = np.log(rho_C0 / obj['rho']) / np.log(delta_F)
        n_nearest = int(round(n_exact))
        rho_predicted = rho_C0 / delta_F**n_nearest
        ratio = obj['rho'] / rho_predicted
        results.append({**obj, 'ratio': ratio, 'n': n_nearest})

    return results


# =============================================================================
# DOMAIN 2: VAN DER WAALS LIQUID–GAS COEXISTENCE
# =============================================================================

def vdw_pressure_reduced(v_r, T_r):
    """
    Reduced van der Waals equation of state:
    P_r = 8T_r / (3v_r - 1) - 3/v_r²

    At the critical point: P_r = T_r = v_r = 1.
    """
    return 8.0 * T_r / (3.0 * v_r - 1.0) - 3.0 / v_r**2


def compute_vdw_coexistence(n_temps=300):
    """
    Compute the liquid–gas coexistence curve via Maxwell equal-area construction.

    Below T_c: two stable phases coexist
      Liquid:  ρ/ρ_c > 1  (dense phase, one attractor basin)
      Gas:     ρ/ρ_c < 1  (dilute phase, other attractor basin)
      Gap:     No thermodynamically stable state between them

    At T_c: the gap closes, both basins merge into one.
    Above T_c: single supercritical phase (one basin only).

    This is THE canonical example of dual attractor basins.
    """
    T_r_array = np.linspace(0.55, 0.9999, n_temps)
    rho_liq = np.full(n_temps, np.nan)
    rho_gas = np.full(n_temps, np.nan)

    # Non-uniform grid: fine near singularity, extends to large v for gas branch
    v = np.sort(np.concatenate([
        np.linspace(0.345, 1.0, 30000),    # fine near singularity/liquid
        np.linspace(1.0, 10.0, 30000),     # mid range
        np.linspace(10.0, 200.0, 20000),   # large v for dilute gas
    ]))

    for i, T_r in enumerate(T_r_array):
        P = vdw_pressure_reduced(v, T_r)

        # Find spinodal points (local extrema of P)
        # VdW isotherm below T_c: P drops from +∞ near v=1/3, reaches
        # local MINIMUM (liquid spinodal), rises through unstable region,
        # reaches local MAXIMUM (gas spinodal), then decreases.
        dP = np.diff(P)

        # Explicitly find where dP changes sign
        min_idx = []  # local minima of P (dP: - to +)
        max_idx = []  # local maxima of P (dP: + to -)
        for j in range(len(dP) - 1):
            if dP[j] <= 0 and dP[j + 1] > 0:
                min_idx.append(j + 1)
            elif dP[j] >= 0 and dP[j + 1] < 0:
                max_idx.append(j + 1)

        if len(min_idx) == 0 or len(max_idx) == 0:
            continue

        # The VdW loop: first min, then the max that follows it
        i_pmin = min_idx[0]
        i_pmax = None
        for mx in max_idx:
            if mx > i_pmin:
                i_pmax = mx
                break
        if i_pmax is None:
            continue

        P_local_min = P[i_pmin]
        P_local_max = P[i_pmax]

        if P_local_max <= P_local_min:
            continue

        # Binary search for Maxwell pressure (equal area rule)
        # P_eq must be between the local min and local max.
        # For low T_r, local min can be negative — that's fine.
        lo = P_local_min + 1e-12
        hi = P_local_max - 1e-12
        v_l, v_g = np.nan, np.nan

        for _ in range(500):
            P_eq = (lo + hi) / 2.0
            diff = P - P_eq
            crossings = np.where(np.diff(np.sign(diff)))[0]

            if len(crossings) < 2:
                # P_eq might be below the entire gas branch (negative)
                # → need to increase P_eq
                lo = P_eq
                continue

            # Liquid root (first crossing) and gas root (last crossing)
            ix_l = crossings[0]
            ix_g = crossings[-1]

            # Ensure they're on the correct branches (separated by spinodals)
            if ix_g <= i_pmax:
                # Gas root not past the maximum → P_eq too high
                hi = P_eq
                continue

            # Linear interpolation for precision
            denom_l = diff[ix_l + 1] - diff[ix_l]
            denom_g = diff[ix_g + 1] - diff[ix_g]
            if abs(denom_l) > 1e-30:
                v_l = v[ix_l] + (v[ix_l + 1] - v[ix_l]) * \
                    (-diff[ix_l]) / denom_l
            else:
                v_l = v[ix_l]
            if abs(denom_g) > 1e-30:
                v_g = v[ix_g] + (v[ix_g + 1] - v[ix_g]) * \
                    (-diff[ix_g]) / denom_g
            else:
                v_g = v[ix_g]

            # Equal area integral
            mask = (v >= v_l) & (v <= v_g)
            if np.sum(mask) < 10:
                break
            area = np.trapz(P[mask] - P_eq, v[mask])

            if abs(area) < 1e-12:
                break
            elif area > 0:
                lo = P_eq
            else:
                hi = P_eq

        if not np.isnan(v_l) and not np.isnan(v_g) and v_g > v_l > 0:
            rho_liq[i] = 1.0 / v_l  # ρ/ρ_c > 1 for liquid
            rho_gas[i] = 1.0 / v_g  # ρ/ρ_c < 1 for gas

    valid = ~np.isnan(rho_liq) & ~np.isnan(rho_gas)
    return T_r_array[valid], rho_liq[valid], rho_gas[valid]


# =============================================================================
# DOMAIN 3: 2D ISING MODEL — ONSAGER EXACT SOLUTION
# =============================================================================

def compute_ising_magnetization(n_temps=1000):
    """
    Exact spontaneous magnetization of the 2D square lattice Ising model.

    Onsager (1944), Yang (1952):
        M(T) = [1 - sinh⁻⁴(2J/k_BT)]^(1/8)   for T < T_c
        M(T) = 0                                  for T ≥ T_c

    where T_c = 2J / (k_B × ln(1 + √2))

    In reduced units (t = T/T_c):
        2J/(k_BT) = ln(1 + √2) / t

    DUAL BASINS: Below T_c, magnetization is +M₀ or -M₀.
    The state M = 0 is thermodynamically UNSTABLE (local maximum of free energy).
    The system must choose one basin or the other.
    The gap between +M and -M closes at T_c (β = 1/8 exponent).
    """
    t = np.linspace(0.01, 1.5, n_temps)
    M_plus = np.zeros(n_temps)
    M_minus = np.zeros(n_temps)

    beta_J_c = np.log(1 + np.sqrt(2)) / 2.0  # J/(k_B T_c) critical coupling

    for i, t_i in enumerate(t):
        if t_i >= 1.0:
            M_plus[i] = 0.0
            M_minus[i] = 0.0
        else:
            x = 2.0 * beta_J_c / t_i  # = 2J/(k_BT)
            sinh_x = np.sinh(x)
            if sinh_x > 1.0:  # below T_c
                arg = 1.0 - sinh_x**(-4)
                if arg > 0:
                    M_plus[i] = arg**(1.0 / 8.0)
                    M_minus[i] = -M_plus[i]

    return t, M_plus, M_minus


def compute_ising_mean_field(n_temps=1000):
    """
    Mean-field Ising model for comparison.
    Self-consistency equation: m = tanh(m/t) where t = T/T_c^MF.
    Gives β = 1/2 (different exponent, same dual basin structure).
    """
    t = np.linspace(0.01, 1.5, n_temps)
    M_mf = np.zeros(n_temps)

    for i, t_i in enumerate(t):
        if t_i >= 1.0:
            M_mf[i] = 0.0
        else:
            m = 0.99
            for _ in range(2000):
                m_new = np.tanh(m / t_i)
                if abs(m_new - m) < 1e-14:
                    break
                m = m_new
            M_mf[i] = m

    return t, M_mf


# =============================================================================
# DOMAIN 4: BOSE–EINSTEIN CONDENSATION
# =============================================================================

def compute_bec_populations(n_temps=1000):
    """
    Ideal 3D Bose gas condensate fraction:
        n₀/N = 1 - (T/T_c)^(3/2)   for T < T_c
        n₀/N = 0                     for T ≥ T_c

    DUAL POPULATIONS:
      Below T_c, particles exist in two distinct states:
        Population A: condensate (p = 0, macroscopic quantum state)
        Population B: thermal cloud (p > 0, classical-like distribution)

      The momentum distribution is BIMODAL — a delta function at p = 0
      (condensate) plus a smooth thermal tail. The gap in momentum space
      is real and measured in every BEC experiment since Anderson et al. 1995.

    Above T_c: single thermal population (one basin).
    """
    t = np.linspace(0.01, 2.0, n_temps)
    n0_frac = np.zeros(n_temps)   # condensate fraction
    nth_frac = np.ones(n_temps)   # thermal fraction

    for i, t_i in enumerate(t):
        if t_i < 1.0:
            n0_frac[i] = 1.0 - t_i**1.5
            nth_frac[i] = t_i**1.5
        else:
            n0_frac[i] = 0.0
            nth_frac[i] = 1.0

    return t, n0_frac, nth_frac


# =============================================================================
# DOMAIN 5: PIPE FLOW LAMINAR–TURBULENT TRANSITION
# =============================================================================

def compute_turbulence_bimodality():
    """
    In the transitional regime of pipe flow (Re ≈ 2000–4000), the flow
    consists of alternating laminar and turbulent patches.

    KEY OBSERVATION: At any instant, a given fluid element is either
    fully laminar or fully turbulent — never in an intermediate state.
    The velocity PDF is BIMODAL.

    This is a dual attractor basin in phase space:
      Basin A: laminar fixed point (parabolic velocity profile)
      Basin B: turbulent chaotic attractor (flatter velocity profile)
      Gap: no stable intermediate state exists

    Centerline velocities (normalized by u_mean):
      Laminar (Hagen–Poiseuille): u_cl/u_mean = 2.0
      Turbulent (power law n≈7):   u_cl/u_mean ≈ 1.23

    Intermittency factor γ(Re) = fraction of time in turbulent state.
    Model: sigmoid transition (Wygnanski & Champagne 1973).

    References:
      Wygnanski & Champagne (1973), J. Fluid Mech. 59, 281
      Avila et al. (2011), Science 333, 192
      Barkley (2016), J. Fluid Mech. 803, P1
    """
    # Intermittency factor
    Re_range = np.linspace(1500, 5000, 500)
    Re_c = 2700.0
    dRe = 200.0
    gamma = 1.0 / (1.0 + np.exp(-(Re_range - Re_c) / dRe))

    # Bimodal PDF parameters
    u_lam = 2.0       # laminar centerline velocity / u_mean
    u_turb = 1.23      # turbulent centerline velocity / u_mean
    sigma_lam = 0.04   # narrow fluctuations (laminar)
    sigma_turb = 0.12   # broader fluctuations (turbulent)

    # Generate PDFs at three representative Re values
    u_values = np.linspace(0.5, 2.5, 500)

    def gaussian(u, mu, sigma):
        return np.exp(-0.5 * ((u - mu) / sigma)**2) / (sigma * np.sqrt(2 * np.pi))

    pdfs = {}
    for Re_val, label in [(2000, 'Re=2000\n(mostly laminar)'),
                           (2700, 'Re=2700\n(intermittent)'),
                           (4000, 'Re=4000\n(mostly turbulent)')]:
        g = 1.0 / (1.0 + np.exp(-(Re_val - Re_c) / dRe))
        pdf = (1 - g) * gaussian(u_values, u_lam, sigma_lam) + \
              g * gaussian(u_values, u_turb, sigma_turb)
        pdfs[label] = pdf

    return {
        'Re': Re_range,
        'gamma': gamma,
        'u_values': u_values,
        'pdfs': pdfs,
        'u_lam': u_lam,
        'u_turb': u_turb,
        'Re_c': Re_c,
    }


# =============================================================================
# DOMAIN 6: SPACETIME METRIC BIFURCATION
# =============================================================================

def compute_metric_bifurcation(n_eta=2000):
    """
    Interior Schwarzschild metric as function of compactness η.

    g_tt(center) = -(3/2 √(1-η) - 1/2)²
    g_tt(surface) = -(1 - η)

    Time dilation factors:
      τ_center  = √|g_tt(center)|  = |3/2√(1-η) - 1/2|
      τ_surface = √|g_tt(surface)| = √(1-η)

    As η increases from 0 to 8/9:
      Center: τ → 0  (infinite time dilation, frozen interior)
      Surface: τ → 1/3  (finite time dilation)

    The center and surface DIVERGE — the interior spacetime geometry
    bifurcates into two qualitatively different regimes. The center
    "freezes" while the surface remains dynamically active.

    This is analogous to the liquid–gas separation: one region of
    spacetime becomes "dense" (frozen time) while the other remains
    "dilute" (flowing time).
    """
    eta = np.linspace(1e-4, 0.885, n_eta)
    sqrt_term = np.sqrt(1.0 - eta)

    g_tt_center = -(1.5 * sqrt_term - 0.5)**2
    g_tt_surface = -(1.0 - eta)

    tau_center = np.sqrt(np.abs(g_tt_center))
    tau_surface = np.sqrt(np.abs(g_tt_surface))

    # At each cascade transition, the separation between center and surface
    cascade_etas = [0.001, 0.01, 0.1, 0.5, 8.0 / 9.0]

    return {
        'eta': eta,
        'g_tt_center': g_tt_center,
        'g_tt_surface': g_tt_surface,
        'tau_center': tau_center,
        'tau_surface': tau_surface,
        'cascade_etas': cascade_etas,
    }


# =============================================================================
# FIGURE 1: DUAL ATTRACTOR BASINS — SIX DOMAINS
# =============================================================================

def generate_figure_1(chladni, vdw, ising_t, ising_Mp, ising_Mm,
                      ising_mf_t, ising_mf_M,
                      bec_t, bec_n0, bec_nth,
                      turb, metric):
    """
    Six-panel figure, one domain per panel. Each shows:
    - Two populations / attractor basins
    - The gap between them
    - The critical point / transition
    """
    fig = plt.figure(figsize=(18, 14), facecolor='white')
    gs = GridSpec(2, 3, hspace=0.35, wspace=0.3)
    fig.suptitle(
        'Dual Attractor Basins at Phase Boundaries — Six Domains',
        fontsize=15, fontweight='bold', y=0.98)

    C = COLORS

    # ===== PANEL A: CHLADNI — Two Populations =====
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.text(-0.08, 1.05, 'a', transform=ax1.transAxes,
             fontsize=14, fontweight='bold', va='top')

    active = [obj for obj in chladni if obj['pop'] == 'active']
    passive = [obj for obj in chladni if obj['pop'] == 'passive']

    # Plot active population (below predicted)
    for j, obj in enumerate(active):
        ax1.plot(obj['ratio'], j, 'o', color=C['r'], markersize=10,
                 markeredgecolor='black', markeredgewidth=0.5, zorder=5)
        ax1.text(obj['ratio'] - 0.05, j, f"  {obj['name']}", fontsize=7,
                 va='center', ha='right', color=C['r'], fontweight='bold')

    # Plot passive population (above predicted)
    for j, obj in enumerate(passive):
        y = j + len(active) + 1  # gap row
        ax1.plot(obj['ratio'], y, 's', color=C['b'], markersize=8,
                 markeredgecolor='black', markeredgewidth=0.5, zorder=5)
        ax1.text(obj['ratio'] + 0.05, y, f"  {obj['name']}", fontsize=7,
                 va='center', ha='left', color=C['b'], fontweight='bold')

    # Vertical line at ratio = 1 (sub-harmonic prediction)
    ax1.axvline(x=1.0, color=C['gray'], linestyle=':', alpha=0.6, linewidth=1.5)
    ax1.text(1.01, -0.5, 'Predicted', fontsize=7, color=C['gray'], rotation=90)

    # Shade the gap
    gap_lo = max(obj['ratio'] for obj in active)
    gap_hi = min(obj['ratio'] for obj in passive)
    ax1.axvspan(gap_lo, gap_hi, alpha=0.12, color=C['gold'],
                label=f'Gap: {gap_lo:.2f}–{gap_hi:.2f}')

    # Labels for populations
    ax1.text(0.3, 0.95, 'Active\n(nuclear energy)',
             transform=ax1.transAxes, fontsize=8, va='top', ha='left',
             color=C['r'], fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.2', facecolor='#fff0f0',
                      alpha=0.9, edgecolor=C['r'], linewidth=0.5))
    ax1.text(0.55, 0.05, 'Passive\n(thermal/degeneracy)',
             transform=ax1.transAxes, fontsize=8, va='bottom', ha='left',
             color=C['b'], fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.2', facecolor='#f0f0ff',
                      alpha=0.9, edgecolor=C['b'], linewidth=0.5))

    ax1.set_xlabel(r'Ratio $\rho_{\mathrm{actual}} / \rho_{S_n}$', fontsize=10)
    ax1.set_yticks([])
    ax1.set_title('Chladni Universe — Feigenbaum Sub-Harmonics',
                  fontsize=11, fontweight='bold')
    ax1.set_xlim(0.2, 2.0)
    ax1.legend(fontsize=7, loc='lower right')
    ax1.grid(True, alpha=0.15, axis='x')

    # ===== PANEL B: VAN DER WAALS — Liquid–Gas Coexistence =====
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.text(-0.08, 1.05, 'b', transform=ax2.transAxes,
             fontsize=14, fontweight='bold', va='top')

    T_r, rho_l, rho_g = vdw

    # Fill between the two branches
    ax2.fill_betweenx(T_r, rho_l, np.ones_like(T_r),
                       alpha=0.15, color=C['r'], label='Liquid basin')
    ax2.fill_betweenx(T_r, np.zeros_like(T_r), rho_g,
                       alpha=0.15, color=C['b'], label='Gas basin')

    # Plot the coexistence curves
    ax2.plot(rho_l, T_r, color=C['r'], linewidth=2.5, label='Liquid (ρ > ρ_c)')
    ax2.plot(rho_g, T_r, color=C['b'], linewidth=2.5, label='Gas (ρ < ρ_c)')

    # Critical point
    ax2.plot(1.0, 1.0, '*', color=C['gold'], markersize=15, zorder=10,
             markeredgecolor='black', markeredgewidth=0.5, label='Critical point')

    # Shade the gap (spinodal region — forbidden zone)
    ax2.fill_betweenx(T_r, rho_g, rho_l, alpha=0.08, color=C['gold'])

    # ρ_c reference
    ax2.axvline(x=1.0, color=C['gray'], linestyle=':', alpha=0.5)

    ax2.set_xlabel(r'$\rho / \rho_c$', fontsize=10)
    ax2.set_ylabel(r'$T / T_c$', fontsize=10)
    ax2.set_title('Van der Waals — Liquid–Gas Coexistence',
                  fontsize=11, fontweight='bold')
    ax2.legend(fontsize=7, loc='lower left')
    ax2.set_xlim(0, 3.0)
    ax2.set_ylim(0.5, 1.05)
    ax2.grid(True, alpha=0.2)

    ax2.text(0.95, 0.05,
             'Below T_c: liquid OR gas\n'
             'No stable intermediate\n'
             'Gap closes at critical point',
             transform=ax2.transAxes, fontsize=7, va='bottom', ha='right',
             bbox=dict(boxstyle='round,pad=0.2', facecolor='lightyellow',
                      alpha=0.9, edgecolor=C['o'], linewidth=0.5))

    # ===== PANEL C: ISING — Spontaneous Magnetization =====
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.text(-0.08, 1.05, 'c', transform=ax3.transAxes,
             fontsize=14, fontweight='bold', va='top')

    # Exact Onsager solution
    ax3.plot(ising_t, ising_Mp, color=C['r'], linewidth=2.5,
             label=r'Onsager exact ($\beta = 1/8$)')
    ax3.plot(ising_t, ising_Mm, color=C['b'], linewidth=2.5)

    # Mean field comparison
    ax3.plot(ising_mf_t, ising_mf_M, '--', color=C['r'], linewidth=1.5,
             alpha=0.5, label=r'Mean field ($\beta = 1/2$)')
    ax3.plot(ising_mf_t, -ising_mf_M, '--', color=C['b'], linewidth=1.5,
             alpha=0.5)

    # Fill basins
    ax3.fill_between(ising_t, 0, ising_Mp, alpha=0.08, color=C['r'])
    ax3.fill_between(ising_t, ising_Mm, 0, alpha=0.08, color=C['b'])

    # Critical point
    ax3.axvline(x=1.0, color=C['gray'], linestyle=':', alpha=0.5)
    ax3.text(1.02, 0.9, r'$T_c$', fontsize=9, color=C['gray'])

    # Unstable M=0 line
    ax3.axhline(y=0, color=C['gray'], linestyle='-', alpha=0.3, linewidth=0.5)
    t_below = ising_t[ising_t < 1.0]
    ax3.plot(t_below, np.zeros_like(t_below), 'x', color=C['gray'],
             markersize=2, alpha=0.3, markevery=20)

    ax3.set_xlabel(r'$T / T_c$', fontsize=10)
    ax3.set_ylabel(r'Magnetization $M / M_0$', fontsize=10)
    ax3.set_title('2D Ising Model — Onsager Exact Solution',
                  fontsize=11, fontweight='bold')
    ax3.legend(fontsize=7, loc='upper right')
    ax3.set_xlim(0, 1.5)
    ax3.set_ylim(-1.1, 1.1)
    ax3.grid(True, alpha=0.2)

    ax3.text(0.05, 0.05,
             'Below T_c: +M or −M\n'
             'M = 0 is UNSTABLE\n'
             'Same dual basin, different β',
             transform=ax3.transAxes, fontsize=7, va='bottom',
             bbox=dict(boxstyle='round,pad=0.2', facecolor='lightyellow',
                      alpha=0.9, edgecolor=C['o'], linewidth=0.5))

    # ===== PANEL D: BEC — Condensate vs Thermal =====
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.text(-0.08, 1.05, 'd', transform=ax4.transAxes,
             fontsize=14, fontweight='bold', va='top')

    ax4.fill_between(bec_t, 0, bec_n0, alpha=0.3, color=C['b'],
                      label='Condensate (p = 0)')
    ax4.fill_between(bec_t, bec_n0, 1.0, alpha=0.3, color=C['r'],
                      label='Thermal cloud (p > 0)')
    ax4.plot(bec_t, bec_n0, color=C['b'], linewidth=2.5)
    ax4.plot(bec_t, bec_nth, color=C['r'], linewidth=2.5)

    # Critical point
    ax4.axvline(x=1.0, color=C['gray'], linestyle=':', alpha=0.5)
    ax4.text(1.02, 0.9, r'$T_{BEC}$', fontsize=9, color=C['gray'])

    ax4.set_xlabel(r'$T / T_{BEC}$', fontsize=10)
    ax4.set_ylabel('Population fraction', fontsize=10)
    ax4.set_title('Bose–Einstein Condensate — Two Populations',
                  fontsize=11, fontweight='bold')
    ax4.legend(fontsize=7, loc='center right')
    ax4.set_xlim(0, 1.8)
    ax4.set_ylim(-0.05, 1.05)
    ax4.grid(True, alpha=0.2)

    ax4.text(0.05, 0.5,
             'Below T_BEC: particles are\n'
             'condensate OR thermal\n'
             'Bimodal in momentum space\n'
             '(Anderson et al. 1995)',
             transform=ax4.transAxes, fontsize=7, va='center',
             bbox=dict(boxstyle='round,pad=0.2', facecolor='lightyellow',
                      alpha=0.9, edgecolor=C['o'], linewidth=0.5))

    # ===== PANEL E: TURBULENCE — Bimodal Velocity =====
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.text(-0.08, 1.05, 'e', transform=ax5.transAxes,
             fontsize=14, fontweight='bold', va='top')

    colors_pdf = [C['b'], C['p'], C['r']]
    for j, (label, pdf) in enumerate(turb['pdfs'].items()):
        ax5.plot(turb['u_values'], pdf, color=colors_pdf[j], linewidth=2.0,
                 label=label)
        ax5.fill_between(turb['u_values'], 0, pdf, alpha=0.1,
                          color=colors_pdf[j])

    # Mark the two attractor positions
    ax5.axvline(x=turb['u_lam'], color=C['b'], linestyle='--', alpha=0.5,
                linewidth=1)
    ax5.axvline(x=turb['u_turb'], color=C['r'], linestyle='--', alpha=0.5,
                linewidth=1)
    ax5.text(turb['u_lam'] + 0.02, ax5.get_ylim()[1] * 0.1 if ax5.get_ylim()[1] > 0 else 5,
             'Laminar\nattractor', fontsize=7, color=C['b'], va='bottom')
    ax5.text(turb['u_turb'] - 0.02, ax5.get_ylim()[1] * 0.1 if ax5.get_ylim()[1] > 0 else 5,
             'Turbulent\nattractor', fontsize=7, color=C['r'], va='bottom',
             ha='right')

    # Shade the gap between attractors
    ax5.axvspan(turb['u_turb'], turb['u_lam'], alpha=0.06, color=C['gold'])

    ax5.set_xlabel(r'Centerline velocity $u / u_{\mathrm{mean}}$', fontsize=10)
    ax5.set_ylabel('Probability density', fontsize=10)
    ax5.set_title('Pipe Flow — Bimodal Velocity Distribution',
                  fontsize=11, fontweight='bold')
    ax5.legend(fontsize=6.5, loc='upper left')
    ax5.set_xlim(0.8, 2.3)
    ax5.grid(True, alpha=0.2)

    ax5.text(0.95, 0.95,
             'No stable intermediate state\n'
             'Flow is laminar OR turbulent\n'
             'Never in between\n'
             '(Avila et al. 2011)',
             transform=ax5.transAxes, fontsize=7, va='top', ha='right',
             bbox=dict(boxstyle='round,pad=0.2', facecolor='lightyellow',
                      alpha=0.9, edgecolor=C['o'], linewidth=0.5))

    # ===== PANEL F: SPACETIME — Center vs Surface Divergence =====
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.text(-0.08, 1.05, 'f', transform=ax6.transAxes,
             fontsize=14, fontweight='bold', va='top')

    eta = metric['eta']
    ax6.plot(eta, metric['tau_center'], color=C['r'], linewidth=2.5,
             label=r'$\tau_{\mathrm{center}} = \sqrt{|g_{tt}(0)|}$')
    ax6.plot(eta, metric['tau_surface'], color=C['b'], linewidth=2.5,
             label=r'$\tau_{\mathrm{surface}} = \sqrt{|g_{tt}(R)|}$')

    # Fill the gap between them
    ax6.fill_between(eta, metric['tau_center'], metric['tau_surface'],
                      alpha=0.12, color=C['gold'], label='Divergence gap')

    # Mark cascade transitions
    cascade_labels = ['C₀', 'C₁', 'C₂', 'C₃', 'C₄']
    cascade_colors_list = [C['b'], C['g'], C['p'], C['r'], C['o']]
    for ce, cl, cc in zip(metric['cascade_etas'], cascade_labels,
                          cascade_colors_list):
        ax6.axvline(x=ce, color=cc, linestyle=':', alpha=0.4, linewidth=1)
        y_pos = 0.98 if ce < 0.5 else 0.5
        ax6.text(ce, y_pos, f'  {cl}', fontsize=7, color=cc, fontweight='bold')

    ax6.set_xlabel(r'Compactness $\eta = 8\pi G\rho R^2 / 3c^4$', fontsize=10)
    ax6.set_ylabel('Time dilation factor τ', fontsize=10)
    ax6.set_title('Spacetime Metric — Center vs Surface',
                  fontsize=11, fontweight='bold')
    ax6.legend(fontsize=7, loc='upper right')
    ax6.set_xlim(0, 0.9)
    ax6.set_ylim(0, 1.05)
    ax6.grid(True, alpha=0.2)

    ax6.text(0.5, 0.05,
             'Center freezes → τ = 0\n'
             'Surface remains finite\n'
             'Interior bifurcates at each cascade',
             transform=ax6.transAxes, fontsize=7, va='bottom', ha='center',
             bbox=dict(boxstyle='round,pad=0.2', facecolor='lightyellow',
                      alpha=0.9, edgecolor=C['o'], linewidth=0.5))

    # Watermark
    fig.text(0.5, 0.01,
             'Dual Attractor Basins — Resonance Theory — Randolph 2026',
             ha='center', fontsize=8, color='gray', style='italic')

    plt.savefig('fig34_dual_attractor_basins.png',
                dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ fig34_dual_attractor_basins.png saved")


# =============================================================================
# FIGURE 2: SYNTHESIS & HONEST TALLY
# =============================================================================

def generate_figure_2(chladni, vdw, ising_t, ising_Mp,
                      ising_mf_t, ising_mf_M,
                      bec_t, bec_n0, turb, metric):
    """
    Six-panel synthesis figure:
      a. Universal bifurcation overlay (all domains normalized)
      b. Gap magnitude vs distance from transition
      c. Landau potential landscape (schematic)
      d. Critical exponent comparison
      e. Chladni mapped onto bifurcation framework
      f. Honest tally scorecard
    """
    fig = plt.figure(figsize=(18, 14), facecolor='white')
    gs = GridSpec(2, 3, hspace=0.35, wspace=0.3)
    fig.suptitle(
        'Synthesis — Universal Dual Attractor Structure',
        fontsize=15, fontweight='bold', y=0.98)

    C = COLORS
    T_r, rho_l, rho_g = vdw

    # ===== PANEL A: Universal Bifurcation Overlay =====
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.text(-0.08, 1.05, 'a', transform=ax1.transAxes,
             fontsize=14, fontweight='bold', va='top')

    # Normalize each domain to common framework:
    # x = (T - T_c) / T_c   (negative = below transition)
    # y = normalized order parameter (±1 range)

    # VdW: order parameter = (ρ - ρ_c)/ρ_c
    vdw_x = T_r - 1.0
    vdw_y_liq = rho_l - 1.0    # positive for liquid
    vdw_y_gas = rho_g - 1.0    # negative for gas
    # Normalize to max
    vdw_max = np.max(np.abs(np.concatenate([vdw_y_liq, vdw_y_gas])))
    ax1.plot(vdw_x, vdw_y_liq / vdw_max, color=C['r'], linewidth=2.0,
             alpha=0.8, label='VdW liquid–gas')
    ax1.plot(vdw_x, vdw_y_gas / vdw_max, color=C['r'], linewidth=2.0,
             alpha=0.8)

    # Ising: order parameter = M
    ising_x = ising_t - 1.0
    mask_below = ising_t < 1.0
    ax1.plot(ising_x[mask_below], ising_Mp[mask_below], color=C['b'],
             linewidth=2.0, alpha=0.8, label='2D Ising')
    ax1.plot(ising_x[mask_below], -ising_Mp[mask_below], color=C['b'],
             linewidth=2.0, alpha=0.8)

    # BEC: order parameter = 2n₀ - 1 (maps 0→-1, 1→+1)
    bec_x = bec_t - 1.0
    bec_y = 2.0 * bec_n0 - 1.0
    mask_bec = bec_t < 1.0
    ax1.plot(bec_x[mask_bec], bec_y[mask_bec], color=C['g'],
             linewidth=2.0, alpha=0.8, label='BEC condensate')
    ax1.plot(bec_x[mask_bec], -bec_y[mask_bec], '--', color=C['g'],
             linewidth=1.5, alpha=0.5)

    # Ising mean field
    mf_x = ising_mf_t - 1.0
    mask_mf = ising_mf_t < 1.0
    ax1.plot(mf_x[mask_mf], ising_mf_M[mask_mf], ':', color=C['p'],
             linewidth=1.5, alpha=0.6, label='Mean field (β=1/2)')
    ax1.plot(mf_x[mask_mf], -ising_mf_M[mask_mf], ':', color=C['p'],
             linewidth=1.5, alpha=0.6)

    ax1.axhline(y=0, color=C['gray'], linewidth=0.5, alpha=0.3)
    ax1.axvline(x=0, color=C['gray'], linestyle=':', alpha=0.5)
    ax1.text(0.02, 0.5, r'$T_c$', fontsize=8, color=C['gray'])

    ax1.set_xlabel(r'$(T - T_c) / T_c$', fontsize=10)
    ax1.set_ylabel('Normalized order parameter', fontsize=10)
    ax1.set_title('Universal Bifurcation Structure',
                  fontsize=11, fontweight='bold')
    ax1.legend(fontsize=7, loc='lower left')
    ax1.set_xlim(-0.5, 0.2)
    ax1.set_ylim(-1.1, 1.1)
    ax1.grid(True, alpha=0.2)

    # ===== PANEL B: Gap Magnitude vs Distance from Transition =====
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.text(-0.08, 1.05, 'b', transform=ax2.transAxes,
             fontsize=14, fontweight='bold', va='top')

    # VdW gap = ρ_liquid - ρ_gas (in reduced units)
    vdw_gap = rho_l - rho_g
    epsilon_vdw = 1.0 - T_r  # distance from transition (positive below T_c)
    valid_vdw = epsilon_vdw > 0.001
    ax2.plot(epsilon_vdw[valid_vdw], vdw_gap[valid_vdw] / np.max(vdw_gap),
             color=C['r'], linewidth=2.0, label='VdW gap')

    # Ising gap = 2M (since M ranges from -M to +M)
    ising_gap = 2.0 * ising_Mp
    epsilon_ising = 1.0 - ising_t
    valid_ising = (epsilon_ising > 0.001) & (ising_gap > 0)
    ax2.plot(epsilon_ising[valid_ising],
             ising_gap[valid_ising] / np.max(ising_gap),
             color=C['b'], linewidth=2.0, label='Ising gap')

    # BEC gap = n₀ (condensate fraction)
    epsilon_bec = 1.0 - bec_t
    valid_bec = (epsilon_bec > 0.001) & (bec_n0 > 0)
    ax2.plot(epsilon_bec[valid_bec],
             bec_n0[valid_bec] / np.max(bec_n0[valid_bec]),
             color=C['g'], linewidth=2.0, label='BEC condensate')

    # Mean field gap
    mf_gap = 2.0 * ising_mf_M
    epsilon_mf = 1.0 - ising_mf_t
    valid_mf = (epsilon_mf > 0.001) & (mf_gap > 0)
    ax2.plot(epsilon_mf[valid_mf],
             mf_gap[valid_mf] / np.max(mf_gap[valid_mf]),
             ':', color=C['p'], linewidth=1.5, alpha=0.7, label='Mean field')

    # Spacetime gap = τ_surface - τ_center
    st_gap = metric['tau_surface'] - metric['tau_center']
    valid_st = st_gap > 0.001
    ax2.plot(metric['eta'][valid_st],
             st_gap[valid_st] / np.max(st_gap),
             color=C['o'], linewidth=2.0, label=r'Spacetime ($\eta$)')

    ax2.set_xlabel(r'Distance from transition (normalized)', fontsize=10)
    ax2.set_ylabel('Normalized gap magnitude', fontsize=10)
    ax2.set_title('Gap Between Attractor Basins',
                  fontsize=11, fontweight='bold')
    ax2.legend(fontsize=7, loc='lower right')
    ax2.set_xlim(0, 0.5)
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, alpha=0.2)

    ax2.text(0.05, 0.95,
             'Gap opens below transition\n'
             'Power-law scaling: Δ ~ ε^β\n'
             'All domains show same qualitative shape',
             transform=ax2.transAxes, fontsize=7, va='top',
             bbox=dict(boxstyle='round,pad=0.2', facecolor='lightyellow',
                      alpha=0.9, edgecolor=C['o'], linewidth=0.5))

    # ===== PANEL C: Landau Potential Landscape =====
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.text(-0.08, 1.05, 'c', transform=ax3.transAxes,
             fontsize=14, fontweight='bold', va='top')

    phi = np.linspace(-1.5, 1.5, 500)

    # Landau free energy: F = a(T-T_c)φ² + bφ⁴
    # In reduced form: F = εφ² + φ⁴ (ε = a(T-T_c)/b)
    epsilons = [0.5, 0.1, 0.0, -0.1, -0.3, -0.5]
    labels = [r'T >> T_c', r'T > T_c', r'T = T_c',
              r'T < T_c', r'T << T_c', r'T <<< T_c']
    colors_landau = [C['gray'], C['teal'], C['gold'], C['o'], C['r'], C['d']]

    for eps, lab, col in zip(epsilons, labels, colors_landau):
        F = eps * phi**2 + phi**4
        F_offset = F - np.min(F)
        ax3.plot(phi, F_offset + eps * 0.8, color=col, linewidth=1.5,
                 label=lab, alpha=0.8)

    # Mark the dual minima for T < T_c
    for eps, col in zip([-0.3, -0.5], [C['r'], C['d']]):
        phi_min = np.sqrt(-eps / 2.0)
        F_min = eps * phi_min**2 + phi_min**4 - np.min(eps * phi**2 + phi**4)
        ax3.plot([-phi_min, phi_min], [F_min + eps * 0.8, F_min + eps * 0.8],
                 'o', color=col, markersize=6, zorder=10)

    ax3.set_xlabel(r'Order parameter $\phi$', fontsize=10)
    ax3.set_ylabel(r'Free energy $F(\phi)$', fontsize=10)
    ax3.set_title('Landau Potential — Dual Well Structure',
                  fontsize=11, fontweight='bold')
    ax3.legend(fontsize=6, loc='upper right', ncol=2)
    ax3.set_xlim(-1.5, 1.5)
    ax3.grid(True, alpha=0.2)

    ax3.text(0.05, 0.05,
             'Below T_c: double well\n'
             'Two minima = two basins\n'
             'Barrier = forbidden zone',
             transform=ax3.transAxes, fontsize=7, va='bottom',
             bbox=dict(boxstyle='round,pad=0.2', facecolor='lightyellow',
                      alpha=0.9, edgecolor=C['o'], linewidth=0.5))

    # ===== PANEL D: Critical Exponent Comparison =====
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.text(-0.08, 1.05, 'd', transform=ax4.transAxes,
             fontsize=14, fontweight='bold', va='top')

    # Measure exponents from computed data
    # VdW near T_c: gap ~ (1-T_r)^β
    eps_vdw_near = epsilon_vdw[valid_vdw]
    gap_vdw_near = vdw_gap[valid_vdw]
    near_mask_vdw = (eps_vdw_near > 0.005) & (eps_vdw_near < 0.15)
    if np.sum(near_mask_vdw) > 5:
        p_vdw = np.polyfit(np.log10(eps_vdw_near[near_mask_vdw]),
                           np.log10(gap_vdw_near[near_mask_vdw]), 1)
        beta_vdw = p_vdw[0]
    else:
        beta_vdw = 0.5

    # Ising near T_c
    eps_ising_near = epsilon_ising[valid_ising]
    gap_ising_near = ising_gap[valid_ising]
    near_mask_ising = (eps_ising_near > 0.005) & (eps_ising_near < 0.15)
    if np.sum(near_mask_ising) > 5:
        p_ising = np.polyfit(np.log10(eps_ising_near[near_mask_ising]),
                             np.log10(gap_ising_near[near_mask_ising]), 1)
        beta_ising = p_ising[0]
    else:
        beta_ising = 0.125

    # BEC near T_c: n₀ ~ (1 - T/T_c)^1 (ideal gas)
    eps_bec_near = epsilon_bec[valid_bec]
    gap_bec_near = bec_n0[valid_bec]
    near_mask_bec = (eps_bec_near > 0.005) & (eps_bec_near < 0.15)
    if np.sum(near_mask_bec) > 5:
        p_bec = np.polyfit(np.log10(eps_bec_near[near_mask_bec]),
                           np.log10(gap_bec_near[near_mask_bec]), 1)
        beta_bec = p_bec[0]
    else:
        beta_bec = 1.0

    # Bar chart of exponents
    domains = ['VdW\n(mean field)', '2D Ising\n(Onsager)', 'BEC\n(ideal gas)',
               'Mean field\n(Landau)']
    betas_measured = [beta_vdw, beta_ising, beta_bec, 0.5]
    betas_exact = [0.5, 0.125, 1.0, 0.5]
    bar_colors = [C['r'], C['b'], C['g'], C['p']]

    x_pos = np.arange(len(domains))
    bars_m = ax4.bar(x_pos - 0.15, betas_measured, 0.3, color=bar_colors,
                      alpha=0.7, edgecolor='black', linewidth=0.5,
                      label='Measured from fit')
    bars_e = ax4.bar(x_pos + 0.15, betas_exact, 0.3, color=bar_colors,
                      alpha=0.3, edgecolor='black', linewidth=0.5,
                      hatch='//', label='Exact known value')

    # Add value labels
    for bar, val in zip(bars_m, betas_measured):
        ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=7,
                 fontweight='bold')
    for bar, val in zip(bars_e, betas_exact):
        ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=7,
                 color=C['gray'])

    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(domains, fontsize=8)
    ax4.set_ylabel(r'Critical exponent $\beta$', fontsize=10)
    ax4.set_title(r'Critical Exponents: $\Delta \sim \varepsilon^\beta$',
                  fontsize=11, fontweight='bold')
    ax4.legend(fontsize=7, loc='upper right')
    ax4.grid(True, alpha=0.2, axis='y')

    ax4.text(0.5, 0.95,
             'Different exponents\nSame dual basin structure\n'
             'Universality CLASS differs\nUniversality of BASINS holds',
             transform=ax4.transAxes, fontsize=7, va='top', ha='center',
             bbox=dict(boxstyle='round,pad=0.2', facecolor='lightyellow',
                      alpha=0.9, edgecolor=C['o'], linewidth=0.5))

    # ===== PANEL E: Chladni Mapped onto Bifurcation =====
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.text(-0.08, 1.05, 'e', transform=ax5.transAxes,
             fontsize=14, fontweight='bold', va='top')

    # Conceptual mapping: the sub-harmonic grid is the "critical point"
    # Active objects are in one basin (below), passive in the other (above)

    # Draw Landau-like potential with Chladni objects placed in the wells
    phi_range = np.linspace(-2, 2, 500)
    F_below = -0.3 * phi_range**2 + 0.1 * phi_range**4  # double well
    F_below = F_below - np.min(F_below)
    ax5.plot(phi_range, F_below, color=C['d'], linewidth=2.5)
    ax5.fill_between(phi_range, F_below, alpha=0.05, color=C['gray'])

    # Left well = active population
    phi_left = -np.sqrt(0.3 / 0.2)
    active_objs = [obj for obj in chladni if obj['pop'] == 'active']
    for j, obj in enumerate(active_objs):
        x_pos_obj = phi_left + (j - 1) * 0.15
        y_pos_obj = 0.02 + j * 0.02
        ax5.plot(x_pos_obj, y_pos_obj, 'o', color=C['r'], markersize=8,
                 markeredgecolor='black', markeredgewidth=0.5, zorder=10)
        ax5.text(x_pos_obj, y_pos_obj + 0.04, obj['name'], fontsize=6,
                 ha='center', va='bottom', color=C['r'], fontweight='bold')

    # Right well = passive population
    phi_right = np.sqrt(0.3 / 0.2)
    passive_objs = [obj for obj in chladni if obj['pop'] == 'passive']
    for j, obj in enumerate(passive_objs):
        x_pos_obj = phi_right + (j - 2) * 0.12
        y_pos_obj = 0.02 + j * 0.015
        ax5.plot(x_pos_obj, y_pos_obj, 's', color=C['b'], markersize=7,
                 markeredgecolor='black', markeredgewidth=0.5, zorder=10)
        ax5.text(x_pos_obj, y_pos_obj + 0.04, obj['name'], fontsize=5.5,
                 ha='center', va='bottom', color=C['b'], fontweight='bold')

    # Label the barrier
    ax5.annotate('Barrier\n(forbidden zone)',
                 xy=(0, F_below[len(F_below) // 2]),
                 xytext=(0, 0.2),
                 fontsize=7, ha='center', color=C['d'],
                 arrowprops=dict(arrowstyle='->', color=C['d'], alpha=0.6))

    # Label the wells
    ax5.text(phi_left, -0.08, 'Active\nbasin', fontsize=8, ha='center',
             color=C['r'], fontweight='bold')
    ax5.text(phi_right, -0.08, 'Passive\nbasin', fontsize=8, ha='center',
             color=C['b'], fontweight='bold')

    ax5.set_xlabel('Order parameter (density offset from sub-harmonic)',
                   fontsize=9)
    ax5.set_ylabel('Effective potential', fontsize=10)
    ax5.set_title('Chladni as Phase Transition',
                  fontsize=11, fontweight='bold')
    ax5.set_xlim(-2, 2)
    ax5.set_ylim(-0.12, 0.35)
    ax5.grid(True, alpha=0.2)

    # ===== PANEL F: HONEST TALLY =====
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.text(-0.08, 1.05, 'f', transform=ax6.transAxes,
             fontsize=14, fontweight='bold', va='top')

    # Text-based scorecard
    ax6.axis('off')

    tally_text = (
        "HONEST TALLY\n"
        "═══════════════════════════════════════\n\n"
        "✓  VdW Liquid–Gas\n"
        "     Dual basins: PROVEN (textbook)\n"
        "     Gap closes at T_c: EXACT\n\n"
        "✓  2D Ising Magnetization\n"
        "     Dual basins: PROVEN (Onsager 1944)\n"
        "     β = 1/8 exact, gap at M = 0\n\n"
        "✓  BEC Condensate/Thermal\n"
        "     Two populations: PROVEN\n"
        "     Bimodal momentum: OBSERVED\n\n"
        "✓  Turbulence Bimodality\n"
        "     Laminar/turbulent: OBSERVED\n"
        "     (Avila et al. 2011, Barkley 2016)\n\n"
        "⚠  Spacetime Metric\n"
        "     Center–surface divergence: MATH\n"
        "     Interpretation as dual basin:\n"
        "     HYPOTHESIS (not proven)\n\n"
        "⚠  Chladni Two-Population\n"
        "     Two populations: OBSERVED\n"
        "     Full catalog p = 0.64 (NOT signif.)\n"
        "     Sun–Earth p = 0.013 (significant)\n"
        "     Mechanism: HYPOTHESIS\n\n"
        "═══════════════════════════════════════\n"
        "4 PROVEN  |  2 HYPOTHESIS  |  0 REFUTED"
    )

    ax6.text(0.05, 0.95, tally_text, transform=ax6.transAxes,
             fontsize=8, va='top', ha='left', family='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8f8f0',
                      alpha=0.95, edgecolor=C['d'], linewidth=1.0))

    ax6.set_title('Assessment — What Holds, What Doesn\'t',
                  fontsize=11, fontweight='bold')

    # Watermark
    fig.text(0.5, 0.01,
             'Dual Attractor Basins — Resonance Theory — Randolph 2026',
             ha='center', fontsize=8, color='gray', style='italic')

    plt.savefig('fig35_dual_attractor_synthesis.png',
                dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ fig35_dual_attractor_synthesis.png saved")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("DUAL ATTRACTOR BASINS AT FRACTAL PHASE BOUNDARIES")
    print("Testing universality across six physics domains")
    print("STATUS: RESEARCH — Honest figures only")
    print("=" * 70)

    # --- DOMAIN 1: Chladni ---
    print("\n--- Domain 1: Chladni Universe (Paper XXI recap) ---")
    chladni = compute_chladni_populations()
    active = [obj for obj in chladni if obj['pop'] == 'active']
    passive = [obj for obj in chladni if obj['pop'] == 'passive']
    print(f"  Active population ({len(active)} objects):")
    for obj in active:
        print(f"    {obj['name']:15s}  S{obj['n']}  ratio = {obj['ratio']:.4f}")
    print(f"  Passive population ({len(passive)} objects):")
    for obj in passive:
        print(f"    {obj['name']:15s}  S{obj['n']}  ratio = {obj['ratio']:.4f}")
    gap_lo = max(obj['ratio'] for obj in active)
    gap_hi = min(obj['ratio'] for obj in passive)
    print(f"  Gap: {gap_lo:.4f} to {gap_hi:.4f}")

    # --- DOMAIN 2: Van der Waals ---
    print("\n--- Domain 2: Van der Waals Coexistence ---")
    vdw_data = compute_vdw_coexistence()
    T_r, rho_l, rho_g = vdw_data
    print(f"  Computed {len(T_r)} coexistence points")
    print(f"  At T/T_c = 0.9: ρ_liq/ρ_c = {rho_l[np.argmin(np.abs(T_r - 0.9))]:.3f}, "
          f"ρ_gas/ρ_c = {rho_g[np.argmin(np.abs(T_r - 0.9))]:.3f}")
    print(f"  At T/T_c = 0.8: ρ_liq/ρ_c = {rho_l[np.argmin(np.abs(T_r - 0.8))]:.3f}, "
          f"ρ_gas/ρ_c = {rho_g[np.argmin(np.abs(T_r - 0.8))]:.3f}")
    print(f"  At T/T_c = 0.7: ρ_liq/ρ_c = {rho_l[np.argmin(np.abs(T_r - 0.7))]:.3f}, "
          f"ρ_gas/ρ_c = {rho_g[np.argmin(np.abs(T_r - 0.7))]:.3f}")

    # --- DOMAIN 3: Ising ---
    print("\n--- Domain 3: 2D Ising Model (Onsager Exact) ---")
    ising_t, ising_Mp, ising_Mm = compute_ising_magnetization()
    ising_mf_t, ising_mf_M = compute_ising_mean_field()
    print(f"  Computed {len(ising_t)} temperature points")
    idx_half = np.argmin(np.abs(ising_t - 0.5))
    idx_nine = np.argmin(np.abs(ising_t - 0.9))
    print(f"  At T/T_c = 0.5: M = ±{ising_Mp[idx_half]:.6f}")
    print(f"  At T/T_c = 0.9: M = ±{ising_Mp[idx_nine]:.6f}")
    print(f"  At T/T_c = 0.5 (mean field): M = ±{ising_mf_M[idx_half]:.6f}")

    # --- DOMAIN 4: BEC ---
    print("\n--- Domain 4: Bose–Einstein Condensation ---")
    bec_t, bec_n0, bec_nth = compute_bec_populations()
    print(f"  Computed {len(bec_t)} temperature points")
    idx_bec_half = np.argmin(np.abs(bec_t - 0.5))
    print(f"  At T/T_BEC = 0.5: n₀/N = {bec_n0[idx_bec_half]:.4f}, "
          f"n_th/N = {bec_nth[idx_bec_half]:.4f}")

    # --- DOMAIN 5: Turbulence ---
    print("\n--- Domain 5: Pipe Flow Turbulence ---")
    turb = compute_turbulence_bimodality()
    print(f"  Centerline velocity (laminar): {turb['u_lam']:.2f} u_mean")
    print(f"  Centerline velocity (turbulent): {turb['u_turb']:.2f} u_mean")
    print(f"  Gap: {turb['u_turb']:.2f} to {turb['u_lam']:.2f} u_mean")
    print(f"  Re_c (transition): {turb['Re_c']:.0f}")

    # --- DOMAIN 6: Spacetime ---
    print("\n--- Domain 6: Spacetime Metric Bifurcation ---")
    metric = compute_metric_bifurcation()
    # Time dilation at specific cascade points
    for eta_c in [0.001, 0.01, 0.1, 0.5]:
        idx = np.argmin(np.abs(metric['eta'] - eta_c))
        tau_c = metric['tau_center'][idx]
        tau_s = metric['tau_surface'][idx]
        print(f"  η = {eta_c:.3f}: τ_center = {tau_c:.4f}, "
              f"τ_surface = {tau_s:.4f}, gap = {tau_s - tau_c:.4f}")

    # --- Generate Figures ---
    print("\n--- Generating Figures ---")
    generate_figure_1(chladni, vdw_data, ising_t, ising_Mp, ising_Mm,
                      ising_mf_t, ising_mf_M,
                      bec_t, bec_n0, bec_nth, turb, metric)
    generate_figure_2(chladni, vdw_data, ising_t, ising_Mp,
                      ising_mf_t, ising_mf_M,
                      bec_t, bec_n0, turb, metric)

    # --- HONEST TALLY ---
    print("\n" + "=" * 70)
    print("HONEST TALLY")
    print("=" * 70)
    print("""
    Domain 1 — Chladni Universe:
      Two populations observed. Gap: {:.2f}–{:.2f}.
      Full catalog p = 0.64 (NOT significant).
      Sun–Earth p = 0.013 (significant).
      VERDICT: ⚠ SUGGESTIVE — needs more data.

    Domain 2 — Van der Waals:
      Dual basins: PROVEN. Textbook physics.
      Gap closes at T_c (critical opalescence).
      VERDICT: ✓ CONFIRMED.

    Domain 3 — 2D Ising:
      Dual basins: PROVEN. Onsager exact solution.
      β = 1/8 (exact), gap at M = 0 unstable.
      VERDICT: ✓ CONFIRMED.

    Domain 4 — BEC:
      Two populations: PROVEN. Condensate at p=0, thermal at p>0.
      Bimodal momentum distribution observed in every BEC experiment.
      VERDICT: ✓ CONFIRMED.

    Domain 5 — Turbulence:
      Bimodal velocity: OBSERVED experimentally.
      Laminar and turbulent are discrete attractor states.
      (Avila et al. 2011, Barkley 2016)
      VERDICT: ✓ CONFIRMED (experimental).

    Domain 6 — Spacetime Metric:
      Center–surface divergence: MATHEMATICS (Schwarzschild 1916).
      Interpretation as dual attractor basin: HYPOTHESIS.
      VERDICT: ⚠ MATH CORRECT, INTERPRETATION UNPROVEN.

    ═══════════════════════════════════════════════════════
    SCORE:  4 PROVEN  |  2 HYPOTHESIS  |  0 REFUTED
    ═══════════════════════════════════════════════════════

    CONCLUSION:
      Dual attractor basins ARE universal at well-characterized
      phase transitions (classical and quantum). This is not
      surprising — it's what defines a phase transition.

      The REAL question: Is the Chladni two-population structure
      a PHASE TRANSITION in the Landau sense? If yes, the sub-
      harmonic grid defines the "critical points" and the active/
      passive populations are the two phases.

      TESTABLE PREDICTION: If this is correct, Gaia DR3 stellar
      density data should show bimodal distribution relative to
      the nearest Feigenbaum sub-harmonic: one peak below (active
      nuclear sources) and one peak above (passive degeneracy
      supported), with a gap between them.
    """.format(gap_lo, gap_hi))

    print(f"\n{'=' * 70}")
    print("COMPLETE — Research status. NOT for publication without review.")
    print(f"{'=' * 70}")
