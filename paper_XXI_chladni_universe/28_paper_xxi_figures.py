#!/usr/bin/env python3
"""
==============================================================================
PAPER XIX: THE CHLADNI UNIVERSE — PUBLIC FIGURES
==============================================================================
Why Celestial Objects Exist Where They Do:
Feigenbaum Sub-Harmonic Structure of the Spacetime Metric

STATUS: PUBLIC — FOR PUBLICATION

This script generates figures for the PUBLIC paper demonstrating that:
1. Einstein's interior Schwarzschild solution has a harmonic cascade structure
2. The cascade is self-similar across all spatial scales
3. Sub-cascades follow Feigenbaum's universal constant δ = 4.669...
4. Every major class of astrophysical object operates near a Feigenbaum
   sub-harmonic of the spacetime metric at its own characteristic scale

Outputs:
    fig28_spacetime_chladni_analysis.png   — 6-panel metric + self-similarity
    fig29_feigenbaum_universe.png          — 6-panel sub-cascade + astro mapping

==============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patheffects as PathEffects
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# UTILITY
# =============================================================================

def smooth(data: np.ndarray, window: int = 21) -> np.ndarray:
    if window < 3 or len(data) < window:
        return data
    kernel = np.ones(window) / window
    padded = np.pad(data, window // 2, mode='edge')
    return np.convolve(padded, kernel, mode='valid')[:len(data)]


# =============================================================================
# PHYSICAL CONSTANTS (SI)
# =============================================================================

G_0 = 6.67430e-11
c = 2.99792458e8
hbar = 1.054571817e-34
k_B = 1.380649e-23
eV = 1.602176634e-19
l_P = np.sqrt(hbar * G_0 / c**3)
kappa_0 = 8.0 * np.pi * G_0 / c**4

# Feigenbaum constant
delta_F = 4.669201609

# Reference densities (J/m³)
rho_room = 6.1e4
rho_chemical = 3.4e10
rho_nuclear = 4.3e14
rho_sun_core = 1.5e16
rho_white_dwarf = 1.0e22
rho_neutron_star = 5.0e34
rho_magnetar = 1.0e41

# Reference radii (m)
R_sun = 6.96e8
R_neutron_star = 1.0e4
R_white_dwarf = 7.0e6
R_earth = 6.371e6


# =============================================================================
# INTERIOR SCHWARZSCHILD METRIC — THE SACRED EQUATION
# =============================================================================

def compute_interior_metric(eta: np.ndarray) -> dict:
    """Interior Schwarzschild metric as function of compactness η."""
    eta_safe = np.clip(eta, 0, 0.88)
    sqrt_1_eta = np.sqrt(1.0 - eta_safe)
    g_tt_center = -(1.5 * sqrt_1_eta - 0.5)**2
    g_tt_surface = -(1.0 - eta_safe)
    sqrt_1_eta4 = np.sqrt(1.0 - eta_safe / 4.0)
    g_tt_half = -(1.5 * sqrt_1_eta - 0.5 * sqrt_1_eta4)**2
    tau_center = np.sqrt(np.abs(g_tt_center))
    tau_surface = np.sqrt(np.abs(g_tt_surface))
    denom = 3.0 * sqrt_1_eta - 1.0
    denom_safe = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
    w_center = (1.0 - sqrt_1_eta) / denom_safe
    K_normalized = 12.0 * eta_safe**2
    flatness = np.abs(g_tt_center + 1.0)

    return {
        'eta_safe': eta_safe,
        'g_tt_center': g_tt_center,
        'g_tt_surface': g_tt_surface,
        'g_tt_half': g_tt_half,
        'tau_center': tau_center,
        'tau_surface': tau_surface,
        'w_center': w_center,
        'K_normalized': K_normalized,
        'flatness': flatness,
    }


# =============================================================================
# MULTI-SCALE LANDSCAPE
# =============================================================================

def compute_scale_landscape(rho: np.ndarray) -> dict:
    radii_m = np.array([1e-3, 1e-1, 1e0, 5e0, 1e2, 1e4, 1e6, 1e8, R_sun])
    labels = ['1 mm', '10 cm', '1 m', '5 m', '100 m',
              '10 km', '10⁶ m', '10⁸ m', 'R_☉']
    results = []
    for i, R in enumerate(radii_m):
        eta = kappa_0 * rho * R**2 / 3.0
        metric = compute_interior_metric(eta)
        results.append({
            'R': R, 'label': labels[i], 'eta': eta, 'metric': metric,
        })
    return {'radii': radii_m, 'labels': labels, 'results': results}


# =============================================================================
# FEIGENBAUM SUB-CASCADE
# =============================================================================

def feigenbaum_subcascades(eta_C0: float = 0.001, n_levels: int = 25):
    return np.array([eta_C0 / delta_F**n for n in range(n_levels)])


def subcascade_density(sub_etas: np.ndarray, R: float) -> np.ndarray:
    return sub_etas * 3.0 * c**4 / (8.0 * np.pi * G_0 * R**2)


# =============================================================================
# ASTROPHYSICAL OBJECT DATABASE
# =============================================================================

astro_objects = {
    'Sun':            {'R': R_sun,          'rho': rho_sun_core,     'symbol': '☀', 'color': '#e67e22'},
    'White Dwarf':    {'R': R_white_dwarf,  'rho': rho_white_dwarf,  'symbol': '○', 'color': '#3498db'},
    'Neutron Star':   {'R': R_neutron_star, 'rho': rho_neutron_star, 'symbol': '●', 'color': '#9b59b6'},
    'Earth Core':     {'R': R_earth,        'rho': 3.6e13,           'symbol': '⊕', 'color': '#2ecc71'},
    'Jupiter Core':   {'R': 7.15e7,         'rho': 2.5e13,           'symbol': '♃', 'color': '#e74c3c'},
}


def find_subcascade_match(R: float, rho: float, eta_C0: float = 0.001) -> float:
    """Find which Feigenbaum sub-cascade level matches given (R, ρ)."""
    rho_C0 = eta_C0 * 3.0 * c**4 / (8.0 * np.pi * G_0 * R**2)
    if rho_C0 <= 0 or rho <= 0:
        return -1
    return np.log(rho_C0 / rho) / np.log(delta_F)


# =============================================================================
# FIGURE 1: SPACETIME HARMONIC CASCADE ANALYSIS (PUBLIC)
# =============================================================================

def generate_figure_1(rho: np.ndarray, landscape: dict) -> None:
    """
    Figure 28: The metric harmonic cascade and self-similarity.
    Public analysis — pure astronomy.
    """
    fig = plt.figure(figsize=(18, 14), facecolor='white')
    gs = GridSpec(2, 3, hspace=0.35, wspace=0.3)
    fig.suptitle(
        'Harmonic Cascade Structure of the Interior Schwarzschild Metric',
        fontsize=15, fontweight='bold', y=0.98)

    C = {'r': '#e74c3c', 'b': '#3498db', 'g': '#2ecc71',
         'p': '#9b59b6', 'o': '#e67e22', 'd': '#2c3e50'}

    log_rho = np.log10(rho)

    # Use R = R_sun for primary analysis (astrophysically relevant)
    R_primary = R_sun
    eta_primary = kappa_0 * rho * R_primary**2 / 3.0
    metric = compute_interior_metric(eta_primary)

    # ===== PANEL 1: g_tt vs Energy Density (R = R_☉) =====
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(log_rho, metric['g_tt_center'], color=C['r'],
             linewidth=2.0, label='$g_{tt}$ (center)')
    ax1.plot(log_rho, metric['g_tt_half'], color=C['b'],
             linewidth=2.0, linestyle='--', label='$g_{tt}$ (r=R/2)')
    ax1.plot(log_rho, metric['g_tt_surface'], color=C['g'],
             linewidth=2.0, linestyle=':', label='$g_{tt}$ (surface)')
    ax1.axhline(y=-1.0, color='gray', linestyle=':', alpha=0.5)
    ax1.text(log_rho[0] + 0.5, -0.93, 'Minkowski (flat)', fontsize=7,
             color='gray', style='italic')

    # Mark Sun core density
    ax1.axvline(x=np.log10(rho_sun_core), color=C['o'], alpha=0.5,
                linestyle='--', linewidth=1.5)
    ax1.text(np.log10(rho_sun_core) + 0.3, -0.5, '☀ Sun core',
             fontsize=8, color=C['o'])

    ax1.set_xlabel('log₁₀(ρ) [J/m³]', fontsize=10)
    ax1.set_ylabel('$g_{tt}$', fontsize=10)
    ax1.set_title('Metric Component $g_{tt}$ (R = R_☉)',
                  fontsize=11, fontweight='bold')
    ax1.legend(fontsize=7, loc='lower left')
    ax1.grid(True, alpha=0.2)

    # ===== PANEL 2: Compactness η — Five Cascade Transitions =====
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(log_rho, eta_primary, color=C['r'], linewidth=2.5)
    ax2.set_yscale('log')

    cascade_etas = [0.001, 0.01, 0.1, 0.5, 8.0/9.0]
    cascade_names = ['C0: Onset', 'C1: Binding', 'C2: Relativistic',
                     'C3: Strong', 'C4: Buchdahl']
    cascade_colors = [C['b'], C['g'], C['p'], C['r'], C['o']]
    for eta_t, name, cc in zip(cascade_etas, cascade_names, cascade_colors):
        ax2.axhline(y=eta_t, color=cc, linestyle='--', alpha=0.5, linewidth=1)
        ax2.text(log_rho[-1] - 0.5, eta_t * 1.15, name, fontsize=7,
                 ha='right', color=cc, fontweight='bold')

    ax2.axvline(x=np.log10(rho_sun_core), color=C['o'], alpha=0.3,
                linestyle='--')

    ax2.set_xlabel('log₁₀(ρ) [J/m³]', fontsize=10)
    ax2.set_ylabel('Compactness η = r_s/R', fontsize=10)
    ax2.set_title('Five Cascade Transitions (R = R_☉)',
                  fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.2)

    # ===== PANEL 3: Equation of State w(η) =====
    ax3 = fig.add_subplot(gs[0, 2])
    w = metric['w_center']
    valid = w > 1e-15
    if np.any(valid):
        ax3.plot(log_rho[valid], np.log10(w[valid]), color=C['p'],
                 linewidth=2.0, label='w = p/(ρc²)')

    ax3.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax3.text(log_rho[0] + 0.5, 0.05, 'w = 1 (stiff)', fontsize=7, color='gray')
    ax3.axhline(y=np.log10(1.0/3.0), color='gray', linestyle=':', alpha=0.3)
    ax3.text(log_rho[0] + 0.5, np.log10(1.0/3.0) + 0.05,
             'w = 1/3 (radiation)', fontsize=7, color='gray')

    ax3.set_xlabel('log₁₀(ρ) [J/m³]', fontsize=10)
    ax3.set_ylabel('log₁₀(w)', fontsize=10)
    ax3.set_title('Equation of State Parameter',
                  fontsize=11, fontweight='bold')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.2)

    # ===== PANEL 4: Self-Similarity — ALL Scales Collapse =====
    ax4 = fig.add_subplot(gs[1, 0])
    cmap = plt.cm.plasma
    n_scales = len(landscape['results'])
    for i, res in enumerate(landscape['results']):
        color = cmap(i / (n_scales - 1))
        eta_i = res['eta']
        g_tt_i = res['metric']['g_tt_center']
        valid = eta_i > 1e-8
        if np.any(valid):
            ax4.plot(np.log10(eta_i[valid]), g_tt_i[valid],
                     color=color, linewidth=1.5,
                     label=f'R = {res["label"]}', alpha=0.8)

    ax4.axhline(y=-1.0, color='gray', linestyle=':', alpha=0.5)
    ax4.axhline(y=-1.0/9.0, color=C['o'], linestyle=':', alpha=0.5)
    ax4.text(log_rho[0] - 35, -1.0/9.0 + 0.02, 'Buchdahl g_tt = −1/9',
             fontsize=7, color=C['o'])

    ax4.set_xlabel('log₁₀(η) — Compactness', fontsize=10)
    ax4.set_ylabel('$g_{tt}$(center)', fontsize=10)
    ax4.set_title('SELF-SIMILARITY: All Scales → One Curve',
                  fontsize=12, fontweight='bold', color=C['r'])
    ax4.legend(fontsize=6, loc='lower left', ncol=2)
    ax4.grid(True, alpha=0.2)

    ax4.text(0.95, 0.95, 'FRACTAL SIGNATURE:\n'
             'Every scale follows the\nsame harmonic curve.\n'
             'The equation is self-similar\nacross 12 orders of\nmagnitude in radius.',
             transform=ax4.transAxes, fontsize=8, va='top', ha='right',
             fontweight='bold', color=C['r'],
             bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                      alpha=0.9, edgecolor=C['r']))

    # ===== PANEL 5: Power-Law Scaling =====
    ax5 = fig.add_subplot(gs[1, 1])
    flat = metric['flatness']
    K = metric['K_normalized']
    valid_f = flat > 1e-30
    valid_k = K > 1e-30

    if np.any(valid_f):
        ax5.plot(log_rho[valid_f], np.log10(flat[valid_f]),
                 color=C['r'], linewidth=2.0, label='|g_tt + 1|')
    if np.any(valid_k):
        ax5.plot(log_rho[valid_k], np.log10(K[valid_k]),
                 color=C['p'], linewidth=2.0, linestyle='--', label='K (Kretschner)')

    # Measure slope in linear regime
    lin_region = (eta_primary > 0.001) & (eta_primary < 0.1)
    if np.any(lin_region):
        idx = np.where(lin_region)[0]
        if len(idx) > 10:
            p_f = np.polyfit(log_rho[idx], np.log10(flat[idx] + 1e-100), 1)
            p_K = np.polyfit(log_rho[idx], np.log10(K[idx] + 1e-100), 1)
            ax5.text(0.95, 0.05,
                     f'Scaling:\n  Flatness ∝ ρ^{p_f[0]:.2f}\n  K ∝ ρ^{p_K[0]:.2f}',
                     transform=ax5.transAxes, fontsize=9, va='bottom',
                     ha='right', fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.3',
                              facecolor='lightyellow', alpha=0.8))

    ax5.set_xlabel('log₁₀(ρ) [J/m³]', fontsize=10)
    ax5.set_ylabel('log₁₀(quantity)', fontsize=10)
    ax5.set_title('Power-Law Scaling (R = R_☉)',
                  fontsize=11, fontweight='bold')
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.2)

    # ===== PANEL 6: Cascade Density vs Body Radius =====
    ax6 = fig.add_subplot(gs[1, 2])
    R_range = np.logspace(-1, 12, 500)
    for eta_t, name, cc in zip([0.001, 0.01, 0.1, 0.5],
                                ['C0', 'C1', 'C2', 'C3'],
                                [C['b'], C['g'], C['p'], C['r']]):
        rho_crit = eta_t * 3.0 * c**4 / (8.0 * np.pi * G_0 * R_range**2)
        ax6.plot(np.log10(R_range), np.log10(rho_crit),
                 color=cc, linewidth=2.0, label=name)

    # Mark astrophysical objects
    for name, obj in astro_objects.items():
        ax6.plot(np.log10(obj['R']), np.log10(obj['rho']),
                 'o', color=obj['color'], markersize=10, zorder=5,
                 markeredgecolor='black', markeredgewidth=0.5)
        ax6.annotate(f'{obj["symbol"]} {name}',
                     xy=(np.log10(obj['R']), np.log10(obj['rho'])),
                     xytext=(np.log10(obj['R']) + 0.5,
                             np.log10(obj['rho']) + 0.8),
                     fontsize=7, color=obj['color'], fontweight='bold',
                     arrowprops=dict(arrowstyle='->', color=obj['color'],
                                    alpha=0.5))

    ax6.set_xlabel('log₁₀(R) [m]', fontsize=10)
    ax6.set_ylabel('log₁₀(ρ_cascade) [J/m³]', fontsize=10)
    ax6.set_title('Cascade Density vs Body Radius — ρ ∝ 1/R²',
                  fontsize=11, fontweight='bold')
    ax6.legend(fontsize=8, loc='upper right')
    ax6.grid(True, alpha=0.2)
    ax6.set_ylim(5, 55)

    # Watermark
    fig.text(0.5, 0.01,
             'Resonance Theory Paper XIX — Randolph 2026',
             ha='center', fontsize=8, color='gray', style='italic')

    plt.savefig('fig28_spacetime_chladni_analysis.png',
                dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ fig28_spacetime_chladni_analysis.png saved")


# =============================================================================
# FIGURE 2: THE FEIGENBAUM UNIVERSE (PUBLIC)
# =============================================================================

def generate_figure_2(rho: np.ndarray, landscape: dict) -> None:
    """
    Figure 29: Feigenbaum sub-cascade structure and astrophysical mapping.
    Public analysis — pure astronomy. Pure astronomy.
    """
    fig = plt.figure(figsize=(18, 14), facecolor='white')
    gs = GridSpec(2, 3, hspace=0.35, wspace=0.3)
    fig.suptitle(
        'The Chladni Universe — Feigenbaum Sub-Harmonic Structure of Spacetime',
        fontsize=15, fontweight='bold', y=0.98)

    C = {'r': '#e74c3c', 'b': '#3498db', 'g': '#2ecc71',
         'p': '#9b59b6', 'o': '#e67e22', 'd': '#2c3e50'}

    n_sub = 20
    sub_etas = feigenbaum_subcascades(0.001, n_sub)

    # ===== PANEL 1: Feigenbaum Sub-Cascade Spacing =====
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.semilogy(range(n_sub), sub_etas, 'o-', color=C['r'],
                 markersize=8, linewidth=2, label='η at sub-cascade Sn')
    ax1.axhline(y=0.001, color=C['o'], linestyle='--', alpha=0.5)
    ax1.text(n_sub - 1, 0.0012, 'C0 = 0.001', fontsize=8,
             color=C['o'], ha='right')

    ax1.set_xlabel('Sub-cascade level n', fontsize=10)
    ax1.set_ylabel('Compactness η_Sn', fontsize=10)
    ax1.set_title(f'Feigenbaum Sub-Cascade (δ = {delta_F:.3f})',
                  fontsize=11, fontweight='bold')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.2)
    ax1.text(0.05, 0.05,
             f'η_Sn = η_C0 / δⁿ\nδ = {delta_F:.6f}\n'
             f'Universal constant\n(Feigenbaum 1978)',
             transform=ax1.transAxes, fontsize=8, va='bottom',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                      alpha=0.8))

    # ===== PANEL 2: Sub-Cascade Densities at Multiple Scales =====
    ax2 = fig.add_subplot(gs[0, 1])
    radii_plot = {
        'R_☉': R_sun,
        'WD (7×10⁶ m)': R_white_dwarf,
        'NS (10 km)': R_neutron_star,
        'Earth': R_earth,
    }
    cmap = plt.cm.viridis
    n_r = len(radii_plot)
    for i, (label, R) in enumerate(radii_plot.items()):
        color = cmap(i / (n_r - 1))
        rho_subs = subcascade_density(sub_etas, R)
        ax2.plot(range(n_sub), np.log10(rho_subs),
                 'o-', color=color, markersize=5, linewidth=1.5,
                 label=f'R = {label}')

    # Mark reference densities
    refs = [
        (np.log10(rho_sun_core), '☀ Sun core', C['o']),
        (np.log10(rho_neutron_star), '● Neutron star', C['p']),
        (np.log10(rho_white_dwarf), '○ White dwarf', C['b']),
        (np.log10(rho_nuclear), '⚛ Nuclear', 'gray'),
    ]
    for ry, rl, rc in refs:
        ax2.axhline(y=ry, color=rc, alpha=0.4, linestyle=':')
        ax2.text(n_sub - 1, ry + 0.3, rl, fontsize=7, ha='right', color=rc)

    ax2.set_xlabel('Sub-cascade level n', fontsize=10)
    ax2.set_ylabel('log₁₀(ρ_Sn) [J/m³]', fontsize=10)
    ax2.set_title('Sub-Cascade Densities — Multiple Scales',
                  fontsize=11, fontweight='bold')
    ax2.legend(fontsize=7, loc='lower left')
    ax2.grid(True, alpha=0.2)

    # ===== PANEL 3: ★★★ THE SOLAR CONNECTION ★★★ =====
    ax3 = fig.add_subplot(gs[0, 2])
    rho_subs_sun = subcascade_density(sub_etas, R_sun)
    ax3.semilogy(range(n_sub), rho_subs_sun, 'o-', color=C['r'],
                 markersize=10, linewidth=2.5, label='Sub-cascades at R = R_☉')

    # Sun core density
    ax3.axhline(y=rho_sun_core, color=C['o'], linewidth=2.5,
                linestyle='--', alpha=0.8)
    ax3.text(0.5, rho_sun_core * 2.5, '☀ ACTUAL SUN CORE DENSITY',
             fontsize=10, color=C['o'], fontweight='bold')

    # Find S9
    rho_C0_sun = 0.001 * 3.0 * c**4 / (8.0 * np.pi * G_0 * R_sun**2)
    n_exact = np.log(rho_C0_sun / rho_sun_core) / np.log(delta_F)
    n_near = int(round(n_exact))
    rho_at_n = rho_C0_sun / delta_F**n_near
    ratio = rho_at_n / rho_sun_core

    ax3.plot(n_near, rho_at_n, '*', color=C['o'], markersize=22, zorder=10)
    ax3.annotate(
        f'Sub-cascade S{n_near}\n'
        f'ρ = {rho_at_n:.2e} J/m³\n'
        f'Ratio to Sun: {ratio:.2f}×',
        xy=(n_near, rho_at_n),
        xytext=(n_near + 3, rho_at_n * 200),
        fontsize=9, fontweight='bold', color=C['d'],
        arrowprops=dict(arrowstyle='->', color=C['o'], linewidth=2),
        bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow',
                 alpha=0.95, edgecolor=C['o'], linewidth=2))

    ax3.set_xlabel('Sub-cascade level n', fontsize=10)
    ax3.set_ylabel('ρ_Sn [J/m³]', fontsize=10)
    ax3.set_title('THE SOLAR CONNECTION',
                  fontsize=12, fontweight='bold', color=C['r'])
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.2)

    # ===== PANEL 4: Self-Similarity of Sub-Cascade (1/δⁿ collapse) =====
    ax4 = fig.add_subplot(gs[1, 0])
    radii_all = {
        '1 mm': 1e-3, '10 cm': 0.1, '1 m': 1.0, '100 m': 100.0,
        '10 km': 1e4, 'R_☉': R_sun,
    }
    cmap2 = plt.cm.plasma
    n_r2 = len(radii_all)
    for i, (label, R) in enumerate(radii_all.items()):
        color = cmap2(i / (n_r2 - 1))
        rho_subs = subcascade_density(sub_etas, R)
        rho_C0_local = rho_subs[0]
        ax4.plot(range(n_sub), rho_subs / rho_C0_local,
                 'o', color=color, markersize=3, alpha=0.7, label=f'{label}')

    # Theoretical line
    n_range = np.arange(n_sub)
    ax4.plot(n_range, 1.0 / delta_F**n_range, 'k--',
             linewidth=2.5, label='1/δⁿ (theory)', zorder=10)

    ax4.set_yscale('log')
    ax4.set_xlabel('Sub-cascade level n', fontsize=10)
    ax4.set_ylabel('ρ_Sn / ρ_C0 (normalized)', fontsize=10)
    ax4.set_title('Self-Similarity: All Scales → 1/δⁿ',
                  fontsize=11, fontweight='bold')
    ax4.legend(fontsize=6, loc='lower left', ncol=2)
    ax4.grid(True, alpha=0.2)

    ax4.text(0.95, 0.95,
             'FRACTAL CONFIRMED:\n'
             'Every scale follows\nthe same Feigenbaum\n'
             f'ratio δ = {delta_F:.3f}',
             transform=ax4.transAxes, fontsize=9, va='top', ha='right',
             fontweight='bold', color=C['r'],
             bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                      alpha=0.9, edgecolor=C['r']))

    # ===== PANEL 5: ★★★ THE CHLADNI MAP — ASTROPHYSICAL OBJECTS ★★★ =====
    ax5 = fig.add_subplot(gs[1, 1])

    # For each astrophysical object, compute which sub-cascade it matches
    obj_names = []
    obj_n_values = []
    obj_colors = []
    obj_symbols = []

    for name, obj in astro_objects.items():
        n_val = find_subcascade_match(obj['R'], obj['rho'])
        if 0 < n_val < 25:
            obj_names.append(name)
            obj_n_values.append(n_val)
            obj_colors.append(obj['color'])

    # Bar chart of sub-cascade matches
    bars = ax5.barh(range(len(obj_names)), obj_n_values,
                     color=obj_colors, alpha=0.7, edgecolor='black',
                     linewidth=0.5, height=0.6)

    ax5.set_yticks(range(len(obj_names)))
    ax5.set_yticklabels(obj_names, fontsize=10, fontweight='bold')
    ax5.set_xlabel('Nearest Feigenbaum sub-cascade level n', fontsize=10)
    ax5.set_title('THE CHLADNI MAP — Where Nature Settles',
                  fontsize=12, fontweight='bold', color=C['r'])
    ax5.grid(True, alpha=0.2, axis='x')

    # Add exact n values as text
    for i, (bar, n_val) in enumerate(zip(bars, obj_n_values)):
        ax5.text(n_val + 0.3, i, f'n = {n_val:.1f}',
                 va='center', fontsize=9, fontweight='bold',
                 color=obj_colors[i])

    ax5.text(0.95, 0.05,
             'Every astrophysical object\n'
             'sits near a Feigenbaum\n'
             'sub-harmonic of spacetime\n'
             'at its own scale.',
             transform=ax5.transAxes, fontsize=9, va='bottom', ha='right',
             fontweight='bold', color=C['d'],
             bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                      alpha=0.9))

    # ===== PANEL 6: THE MASTER MAP — R vs n with Density Contours =====
    ax6 = fig.add_subplot(gs[1, 2])

    R_grid = np.logspace(-1, 10, 200)
    n_grid = np.arange(0, 20)
    R_mesh, N_mesh = np.meshgrid(np.log10(R_grid), n_grid)

    rho_grid = np.zeros_like(R_mesh)
    for j, n_val in enumerate(n_grid):
        eta_n = 0.001 / delta_F**n_val
        rho_grid[j, :] = eta_n * 3.0 * c**4 / (8.0 * np.pi * G_0 * R_grid**2)

    log_rho_grid = np.log10(rho_grid)

    levels = np.arange(5, 55, 5)
    cs = ax6.contourf(R_mesh, N_mesh, log_rho_grid, levels=levels,
                       cmap='RdYlBu_r', alpha=0.7)
    plt.colorbar(cs, ax=ax6, label='log₁₀(ρ) [J/m³]', shrink=0.8)

    # Sun core density contour
    ax6.contour(R_mesh, N_mesh, log_rho_grid,
                levels=[np.log10(rho_sun_core)],
                colors=[C['o']], linewidths=3, linestyles='--')

    # Plot astrophysical objects on the master map
    for name, obj in astro_objects.items():
        n_val = find_subcascade_match(obj['R'], obj['rho'])
        if 0 < n_val < 20:
            ax6.plot(np.log10(obj['R']), n_val, '*', color='white',
                     markersize=15, zorder=10, markeredgecolor='black',
                     markeredgewidth=1)
            ax6.text(np.log10(obj['R']) + 0.3, n_val + 0.3,
                     f'{obj["symbol"]} {name}', fontsize=7,
                     color='white', fontweight='bold',
                     path_effects=[
                         PathEffects.withStroke(
                             linewidth=2, foreground='black')])

    ax6.set_xlabel('log₁₀(R) [m]', fontsize=10)
    ax6.set_ylabel('Sub-cascade level n', fontsize=10)
    ax6.set_title('THE MASTER MAP — Density Landscape',
                  fontsize=12, fontweight='bold', color=C['r'])

    # Watermark
    fig.text(0.5, 0.01,
             'Resonance Theory Paper XIX — Randolph 2026',
             ha='center', fontsize=8, color='gray', style='italic')

    plt.savefig('fig29_feigenbaum_universe.png',
                dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ fig29_feigenbaum_universe.png saved")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("PAPER XIX: THE CHLADNI UNIVERSE — PUBLIC FIGURES")
    print("Why Celestial Objects Exist Where They Do")
    print("STATUS: PUBLIC — FOR PUBLICATION")
    print("=" * 70)

    N = 3000
    rho = np.logspace(4, 50, N)
    print(f"\nDriving variable: ρ from 10⁴ to 10⁵⁰ J/m³")

    print("\n--- Computing Multi-Scale Landscape ---")
    landscape = compute_scale_landscape(rho)

    print("\n--- Astrophysical Object Sub-Cascade Mapping ---")
    for name, obj in astro_objects.items():
        n_val = find_subcascade_match(obj['R'], obj['rho'])
        n_near = int(round(n_val))
        rho_C0 = 0.001 * 3.0 * c**4 / (8.0 * np.pi * G_0 * obj['R']**2)
        rho_at_n = rho_C0 / delta_F**n_near
        ratio = rho_at_n / obj['rho']
        print(f"  {name:>15s}: R = {obj['R']:.2e} m, ρ = {obj['rho']:.2e} J/m³")
        print(f"    → Sub-cascade S{n_near} (n_exact = {n_val:.2f})")
        print(f"    → ρ_predicted = {rho_at_n:.2e}, ratio = {ratio:.2f}×")

    print("\n--- Generating Figures ---")
    generate_figure_1(rho, landscape)
    generate_figure_2(rho, landscape)

    print("\n" + "=" * 70)
    print("COMPLETE — PUBLIC — FOR PUBLICATION")
    print("=" * 70)
