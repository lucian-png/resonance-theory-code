"""
Script 95 -- Navier-Stokes Fractal Classification: Phase 2
          Harmonic Resonant Structure in Critical Reynolds Numbers

    Tests whether the critical Reynolds numbers for standard flow geometries
    fall at harmonic ratios predicted by the Feigenbaum cascade structure.

    Key hypothesis: if Navier-Stokes exhibits the Lucian Law's fractal
    architecture, then critical Re values should organize at Feigenbaum
    sub-harmonic ratios of each other, and successive ratios should
    converge toward δ ≈ 4.669.

Generates:
    fig96_harmonic_peaks.png    (critical Re vs predicted harmonic positions)
    fig97_ratio_convergence.png (ratio analysis for Feigenbaum convergence)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ==========================================================================
#  FUNDAMENTAL CONSTANTS
# ==========================================================================
DELTA_FEIG = 4.669201609102990
ALPHA_FEIG = 2.502907875095892
LN_DELTA   = np.log(DELTA_FEIG)

plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'figure.facecolor': 'white',
    'axes.facecolor': '#fafafa',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
})


# ==========================================================================
#  CRITICAL REYNOLDS NUMBERS — EXPERIMENTAL DATA
# ==========================================================================
# These are well-established values from fluid dynamics literature.
# Sources: Schlichting (1979), White (2006), Kundu & Cohen (2008)
#
# Organized by geometry and transition type.

# PRIMARY SEQUENCE: ordered critical Re values across flow types
# These represent successive instability thresholds in the
# Navier-Stokes landscape
CRITICAL_RE_SEQUENCE = np.array([
    # Re_crit    Label
    1.0,         # Stokes limit (inertia becomes relevant)
    5.0,         # Oseen correction significant
    20.0,        # Sphere: steady separation begins
    47.0,        # Cylinder: onset of vortex shedding
    190.0,       # Cylinder: 3D instability in wake
    270.0,       # Sphere: wake becomes unsteady
    500.0,       # Open flow instabilities emerge
    2300.0,      # Pipe flow: critical transition
    4000.0,      # Pipe flow: fully turbulent
    2e4,         # Sphere: subcritical turbulence
    1e5,         # Boundary layer instability
    3.7e5,       # Sphere: drag crisis (boundary layer transition)
    5e5,         # Flat plate: transition to turbulence
    3.5e6,       # Boundary layer: fully turbulent
    1e7,         # High-Re turbulence regime
    1e8,         # Very high-Re (wind tunnel limit)
])

CRITICAL_RE_LABELS = [
    'Stokes limit',
    'Oseen correction',
    'Sphere separation',
    'Cylinder vortex shedding',
    'Cylinder 3D instability',
    'Sphere wake unsteady',
    'Open flow instabilities',
    'Pipe flow critical',
    'Pipe fully turbulent',
    'Sphere subcritical turb.',
    'BL instability',
    'Sphere drag crisis',
    'Flat plate transition',
    'BL fully turbulent',
    'High-Re regime',
    'Very high-Re regime',
]

# PIPE FLOW sub-sequence (most studied, most precise)
PIPE_RE = np.array([
    2040.0,      # Avila et al. (2011) - transition onset
    2300.0,      # Classical critical Re
    2600.0,      # Intermittent turbulence
    3000.0,      # Sustained puffs
    4000.0,      # Fully turbulent
])
PIPE_LABELS = [
    'Transition onset (Avila)',
    'Classical Re_crit',
    'Intermittent turbulence',
    'Sustained puffs',
    'Fully turbulent',
]

# BOUNDARY LAYER sub-sequence
BL_RE = np.array([
    9e4,         # Tollmien-Schlichting waves
    1.5e5,       # Secondary instability
    3e5,         # Turbulent spots
    5e5,         # Transition complete
    3.5e6,       # Fully developed turbulent BL
])
BL_LABELS = [
    'T-S waves',
    'Secondary instability',
    'Turbulent spots',
    'Transition complete',
    'Fully developed turb. BL',
]


# ==========================================================================
#  FEIGENBAUM HARMONIC SPECTRUM
# ==========================================================================
def feigenbaum_spectrum(anchor_Re, n_harmonics=20):
    """
    Generate Feigenbaum sub-harmonic spectrum anchored at anchor_Re.
    Harmonics at: anchor_Re * δ^n  for n = ..., -2, -1, 0, 1, 2, ...
    """
    n_vals = np.arange(-n_harmonics, n_harmonics + 1)
    return anchor_Re * DELTA_FEIG**n_vals, n_vals


def nearest_harmonic_ratio(Re_val, anchor_Re):
    """
    Compute the ratio of Re_val to its nearest Feigenbaum harmonic.
    Returns (ratio, harmonic_index, nearest_harmonic_Re).
    """
    log_ratio = np.log(Re_val / anchor_Re) / LN_DELTA
    nearest_n = np.round(log_ratio)
    nearest_Re = anchor_Re * DELTA_FEIG**nearest_n
    ratio = Re_val / nearest_Re
    return ratio, int(nearest_n), nearest_Re


# ==========================================================================
#  FIGURE 96 — CRITICAL Re vs PREDICTED HARMONIC PEAKS
# ==========================================================================
def make_fig96():
    """Critical Re values plotted against Feigenbaum harmonic spectrum."""
    print("Generating Figure 96: Critical Re vs Harmonic Peaks...")

    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1], hspace=0.3)

    # --- Panel A: Full spectrum ---
    ax1 = fig.add_subplot(gs[0])

    # Use pipe flow critical Re as anchor
    anchor = 2300.0
    harmonics, h_indices = feigenbaum_spectrum(anchor, n_harmonics=12)

    # Plot harmonic lines
    for h_re, h_n in zip(harmonics, h_indices):
        if 0.1 <= h_re <= 1e12:
            log_h = np.log10(h_re)
            alpha_val = 0.6 if h_n == 0 else 0.25
            lw = 2 if h_n == 0 else 0.8
            ax1.axvline(log_h, color='#3498db', alpha=alpha_val,
                       linestyle='-', linewidth=lw)
            if -5 <= h_n <= 8:
                ax1.text(log_h, 1.02, f'n={h_n}', ha='center', va='bottom',
                        fontsize=7, color='#3498db', alpha=0.7,
                        transform=ax1.get_xaxis_transform())

    # Plot critical Re values
    log_crit = np.log10(CRITICAL_RE_SEQUENCE)
    y_positions = np.linspace(0.3, 0.8, len(CRITICAL_RE_SEQUENCE))

    for i, (re_val, label) in enumerate(zip(CRITICAL_RE_SEQUENCE, CRITICAL_RE_LABELS)):
        ratio, h_n, nearest = nearest_harmonic_ratio(re_val, anchor)
        color = '#e74c3c' if abs(ratio - 1.0) < 0.3 else '#e67e22'
        marker_size = 12 if abs(ratio - 1.0) < 0.15 else 8

        ax1.plot(np.log10(re_val), y_positions[i], 'o',
                color=color, markersize=marker_size, zorder=10,
                markeredgecolor='black', markeredgewidth=0.5)
        ax1.annotate(f'{label}\n(ratio={ratio:.3f})',
                    xy=(np.log10(re_val), y_positions[i]),
                    xytext=(np.log10(re_val) + 0.3, y_positions[i]),
                    fontsize=7, va='center',
                    arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))

    ax1.set_xlabel('log₁₀(Re)')
    ax1.set_ylabel('(visual separation)')
    ax1.set_title('Figure 96 — Critical Reynolds Numbers vs Feigenbaum Harmonic Spectrum\n'
                  '(Anchored at pipe flow Re_crit = 2,300)',
                  fontsize=14, fontweight='bold')
    ax1.set_xlim(-1, 10)
    ax1.set_yticks([])
    ax1.legend(['Feigenbaum harmonics (Re_crit × δⁿ)', 'Critical Re (near harmonic)',
                'Critical Re (off harmonic)'],
               loc='upper right', fontsize=9)

    # --- Panel B: Proximity to nearest harmonic ---
    ax2 = fig.add_subplot(gs[1])

    ratios = []
    for re_val in CRITICAL_RE_SEQUENCE:
        r, _, _ = nearest_harmonic_ratio(re_val, anchor)
        ratios.append(r)
    ratios = np.array(ratios)

    colors_bar = ['#2ecc71' if abs(r - 1.0) < 0.15 else
                  '#e67e22' if abs(r - 1.0) < 0.3 else
                  '#e74c3c' for r in ratios]

    bars = ax2.bar(range(len(ratios)), ratios, color=colors_bar,
                   edgecolor='black', linewidth=0.5)
    ax2.axhline(1.0, color='#3498db', linewidth=2, linestyle='--',
                label='Perfect harmonic alignment (ratio = 1.0)')
    ax2.axhspan(0.85, 1.15, alpha=0.1, color='#3498db',
                label='±15% harmonic zone')

    ax2.set_xlabel('Critical Re index')
    ax2.set_ylabel('Ratio to nearest\nFeigenbaum harmonic')
    ax2.set_title('Proximity of Critical Re Values to Feigenbaum Harmonics',
                  fontweight='bold')
    ax2.set_xticks(range(len(ratios)))
    ax2.set_xticklabels([l.split('\n')[0][:12] for l in CRITICAL_RE_LABELS],
                        rotation=45, ha='right', fontsize=7)
    ax2.legend(loc='upper right', fontsize=9)
    # No ylim cap — show true bar heights for all datapoints
    ax2.set_ylim(0, max(ratios) * 1.15)

    # Add value labels on each bar
    for i, (bar, r) in enumerate(zip(bars, ratios)):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(ratios)*0.01,
                 f'{r:.3f}', ha='center', va='bottom', fontsize=7, fontweight='bold')

    # Count hits
    n_close = np.sum(np.abs(ratios - 1.0) < 0.15)
    n_moderate = np.sum(np.abs(ratios - 1.0) < 0.3)
    ax2.text(0.02, 0.95,
             f'{n_close}/{len(ratios)} within 15% of harmonic\n'
             f'{n_moderate}/{len(ratios)} within 30% of harmonic',
             transform=ax2.transAxes, fontsize=10, va='top',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      edgecolor='gray', alpha=0.9))

    plt.tight_layout()
    outpath = os.path.join(SCRIPT_DIR, 'fig96_harmonic_peaks.png')
    plt.savefig(outpath, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {outpath}")
    return outpath


# ==========================================================================
#  FIGURE 97 — RATIO ANALYSIS FOR FEIGENBAUM CONVERGENCE
# ==========================================================================
def make_fig97():
    """Ratio analysis: do successive critical Re ratios converge to δ?"""
    print("Generating Figure 97: Ratio Analysis for Feigenbaum Convergence...")

    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

    # --- Panel A: Successive ratios of full sequence ---
    ax1 = fig.add_subplot(gs[0, 0])

    re_sorted = np.sort(CRITICAL_RE_SEQUENCE)
    intervals = np.diff(np.log10(re_sorted))
    ratios_full = intervals[:-1] / intervals[1:]

    ax1.plot(range(1, len(ratios_full) + 1), ratios_full, 'o-',
             color='#e74c3c', markersize=8, linewidth=2, markeredgecolor='black')
    ax1.axhline(DELTA_FEIG, color='#3498db', linewidth=2, linestyle='--',
                label=f'δ = {DELTA_FEIG:.4f}')
    ax1.axhspan(DELTA_FEIG * 0.8, DELTA_FEIG * 1.2, alpha=0.1, color='#3498db')

    ax1.set_xlabel('Ratio index')
    ax1.set_ylabel('Interval ratio rₙ/rₙ₊₁')
    ax1.set_title('A. Successive Interval Ratios\n(Full Critical Re Sequence)',
                  fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.set_ylim(0, 10)

    # --- Panel B: Pipe flow sub-sequence ---
    ax2 = fig.add_subplot(gs[0, 1])

    pipe_intervals = np.diff(PIPE_RE)
    pipe_ratios = pipe_intervals[:-1] / pipe_intervals[1:]

    ax2.plot(range(1, len(pipe_ratios) + 1), pipe_ratios, 's-',
             color='#2ecc71', markersize=10, linewidth=2, markeredgecolor='black')
    ax2.axhline(DELTA_FEIG, color='#3498db', linewidth=2, linestyle='--',
                label=f'δ = {DELTA_FEIG:.4f}')

    for i, (r, label) in enumerate(zip(pipe_ratios, PIPE_LABELS[1:-1])):
        ax2.annotate(f'{r:.3f}', xy=(i + 1, r),
                    xytext=(i + 1.2, r + 0.3), fontsize=9,
                    arrowprops=dict(arrowstyle='->', color='gray', lw=0.5))

    ax2.set_xlabel('Ratio index')
    ax2.set_ylabel('Interval ratio')
    ax2.set_title('B. Pipe Flow Sub-Sequence\n(Most studied transition)',
                  fontweight='bold')
    ax2.legend(fontsize=10)

    # --- Panel C: Boundary layer sub-sequence ---
    ax3 = fig.add_subplot(gs[1, 0])

    bl_intervals = np.diff(np.log10(BL_RE))
    bl_ratios = bl_intervals[:-1] / bl_intervals[1:]

    ax3.plot(range(1, len(bl_ratios) + 1), bl_ratios, 'D-',
             color='#9b59b6', markersize=10, linewidth=2, markeredgecolor='black')
    ax3.axhline(DELTA_FEIG, color='#3498db', linewidth=2, linestyle='--',
                label=f'δ = {DELTA_FEIG:.4f}')

    for i, r in enumerate(bl_ratios):
        ax3.annotate(f'{r:.3f}', xy=(i + 1, r),
                    xytext=(i + 1.2, r + 0.3), fontsize=9,
                    arrowprops=dict(arrowstyle='->', color='gray', lw=0.5))

    ax3.set_xlabel('Ratio index')
    ax3.set_ylabel('Interval ratio (log-space)')
    ax3.set_title('C. Boundary Layer Sub-Sequence\n(Log-space interval ratios)',
                  fontweight='bold')
    ax3.legend(fontsize=10)

    # --- Panel D: Summary — comparison to known Feigenbaum systems ---
    ax4 = fig.add_subplot(gs[1, 1])

    # Known convergence behavior in standard maps
    # Logistic map first 5 ratios: 4.7514, 4.6563, 4.6684, 4.6686, 4.6692
    logistic_ratios = np.array([4.7514, 4.6563, 4.6684, 4.6686, 4.6692])
    ns_log_ratios = np.diff(np.log10(re_sorted))
    ns_interval_ratios = ns_log_ratios[:-1] / ns_log_ratios[1:]

    ax4.plot(range(1, len(logistic_ratios) + 1), logistic_ratios, 'o-',
             color='#e74c3c', markersize=8, linewidth=2,
             label='Logistic map (exact)', markeredgecolor='black')
    # Plot subset of NS ratios that are in reasonable range
    ns_reasonable = ns_interval_ratios[np.abs(ns_interval_ratios) < 20]
    ax4.plot(range(1, len(ns_reasonable) + 1), np.abs(ns_reasonable), 's-',
             color='#3498db', markersize=8, linewidth=2,
             label='Navier-Stokes critical Re', markeredgecolor='black')
    ax4.axhline(DELTA_FEIG, color='gray', linewidth=2, linestyle='--',
                label=f'δ = {DELTA_FEIG:.4f}', alpha=0.7)

    ax4.set_xlabel('Ratio index')
    ax4.set_ylabel('|Interval ratio|')
    ax4.set_title('D. Comparison: Logistic Map vs\nNavier-Stokes Critical Re',
                  fontweight='bold')
    ax4.legend(fontsize=9, loc='upper right')
    ax4.set_ylim(0, 12)

    fig.suptitle('Figure 97 — Ratio Analysis: Testing Feigenbaum Convergence in Navier-Stokes',
                 fontsize=15, fontweight='bold', y=1.02)

    plt.tight_layout()
    outpath = os.path.join(SCRIPT_DIR, 'fig97_ratio_convergence.png')
    plt.savefig(outpath, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {outpath}")
    return outpath


# ==========================================================================
#  MAIN
# ==========================================================================
if __name__ == '__main__':
    print("=" * 72)
    print("  Script 95 — Navier-Stokes Phase 2: Harmonic Resonant Structure")
    print("=" * 72)

    fig96_path = make_fig96()
    fig97_path = make_fig97()

    print("\n" + "=" * 72)
    print("  Phase 2 complete. Two figures generated.")
    print("=" * 72)
