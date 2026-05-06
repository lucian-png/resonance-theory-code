"""
Script 94 -- Navier-Stokes Fractal Classification: Phase 1
          Reynolds Number Sweep & Geometric Morphology

    Drives the Reynolds number from 10^-2 to 10^15 and examines how
    the geometric morphology of the Navier-Stokes system changes across
    that 17-order-of-magnitude range.

    The Lucian Method applied to Navier-Stokes: hold the equations sacred,
    sweep the driving variable (Re) across extreme range, classify the
    geometric architecture of the response.

Generates:
    fig94_reynolds_sweep.png         (coupled variable behavior across 17 OOM)
    fig95_morphology_transition.png  (geometric transition map with thresholds)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ==========================================================================
#  FUNDAMENTAL CONSTANTS
# ==========================================================================
DELTA_FEIG = 4.669201609102990
ALPHA_FEIG = 2.502907875095892
LN_DELTA   = np.log(DELTA_FEIG)

# Style
COLORS = {
    'velocity':  '#e74c3c',   # red
    'pressure':  '#3498db',   # blue
    'vorticity': '#2ecc71',   # green
    'energy':    '#9b59b6',   # purple
    'laminar':   '#3498db',   # blue
    'transition':'#e67e22',   # orange
    'turbulent': '#e74c3c',   # red
    'extreme':   '#8e44ad',   # deep purple
    'fractal':   '#c0392b',   # dark red
}

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
#  KNOWN CRITICAL REYNOLDS NUMBERS (textbook values)
# ==========================================================================
# These are well-established experimental values from fluid dynamics literature
CRITICAL_RE = {
    'Stokes limit':           1.0,
    'Sphere separation':      20.0,
    'Cylinder vortex':        47.0,
    'Sphere wake instab.':    270.0,
    'Pipe flow (Re_crit)':    2300.0,
    'Flat plate transition':  5e5,
    'Pipe turbulence':        4000.0,
    'Boundary layer turb.':   3.5e6,
    'Drag crisis (sphere)':   3.7e5,
}


# ==========================================================================
#  NAVIER-STOKES RESPONSE MODEL
# ==========================================================================
# The Navier-Stokes equations in dimensionless form:
#   ∂u/∂t + (u·∇)u = -∇p + (1/Re)∇²u
#
# As Re increases:
#   - Viscous term (1/Re)∇²u diminishes
#   - Nonlinear advection (u·∇)u dominates
#   - System transitions from linear (Stokes) to fully nonlinear (turbulent)
#
# We model the characteristic magnitudes of:
#   |u|  - velocity field magnitude (normalized)
#   |∇p| - pressure gradient magnitude (normalized)
#   |ω|  - vorticity magnitude |∇×u| (normalized)
#   E_k  - turbulent kinetic energy density

def velocity_magnitude(Re):
    """
    Normalized velocity field magnitude as function of Re.
    In laminar regime: scales as Re (inertial buildup).
    At transition: fluctuations appear.
    In turbulent regime: mean + fluctuation structure.
    """
    Re = np.asarray(Re, dtype=float)
    # Base laminar scaling
    u_laminar = np.ones_like(Re)
    # Turbulent intensity grows as Re^(1/2) beyond transition
    # (from scaling analysis of turbulent boundary layers)
    u_turb_intensity = np.where(Re > 2300,
                                 0.1 * (Re / 2300)**0.5,
                                 0.0)
    # Total RMS velocity magnitude
    u_total = u_laminar + u_turb_intensity
    # Normalize
    return u_total / u_total.max()


def pressure_gradient(Re):
    """
    Normalized pressure gradient magnitude.
    Laminar: ∇p ~ Re (Hagen-Poiseuille: Δp ~ (μL/D²)·u ~ Re)
    Turbulent: ∇p ~ Re^1.75 (Blasius correlation for pipe flow)
    """
    Re = np.asarray(Re, dtype=float)
    # Laminar regime: linear in Re
    dp_laminar = Re / 2300.0
    # Turbulent regime: Blasius scaling f ~ Re^(-0.25), Δp ~ f·Re² ~ Re^1.75
    dp_turbulent = (Re / 2300.0)**1.75
    # Smooth transition using sigmoid
    sigma = 1.0 / (1.0 + np.exp(-2.0 * (np.log10(Re) - np.log10(2300))))
    dp = (1 - sigma) * dp_laminar + sigma * dp_turbulent
    # Normalize to [0, 1]
    return dp / dp.max()


def vorticity_magnitude(Re):
    """
    Normalized vorticity magnitude |ω| = |∇×u|.
    Laminar: ω ~ Re (simple shear)
    Transitional: vortex stretching activates
    Turbulent: ω ~ Re^(3/4) (Kolmogorov scaling of dissipation)
    The key: vorticity develops FRACTAL structure at high Re.
    """
    Re = np.asarray(Re, dtype=float)
    # Laminar vorticity
    omega_lam = Re / 2300.0
    # Turbulent vorticity with Kolmogorov scaling
    # ε ~ u³/L, ω² ~ ε/ν ~ Re^(3/2) → ω ~ Re^(3/4)
    omega_turb = (Re / 2300.0)**0.75
    # Transition
    sigma = 1.0 / (1.0 + np.exp(-2.0 * (np.log10(Re) - np.log10(2300))))
    omega = (1 - sigma) * omega_lam + sigma * omega_turb
    return omega / omega.max()


def turbulent_energy(Re):
    """
    Turbulent kinetic energy density.
    Zero in laminar regime.
    Grows as Re^(1/2) in turbulent regime (from TKE scaling).
    Develops cascade structure at high Re.
    """
    Re = np.asarray(Re, dtype=float)
    # Onset at transition
    tke = np.where(Re > 500,
                   (Re / 2300.0)**0.5 * (1.0 / (1.0 + np.exp(-3.0 * (np.log10(Re) - np.log10(2300))))),
                   1e-6 * Re / 500.0)
    return tke / tke.max()


def fractal_dimension(Re):
    """
    Effective fractal dimension of the velocity field.
    Laminar: D = 1.0 (smooth, Euclidean)
    Transitional: D increases as structure develops
    Turbulent: D approaches Kolmogorov's D ≈ 2.5-2.7 for isotropic turbulence
    Extreme: D saturates near theoretical maximum

    This is the Lucian Law prediction: the system transitions from
    Euclidean to fractal as the nonlinearity becomes dominant.
    """
    Re = np.asarray(Re, dtype=float)
    log_Re = np.log10(Re)
    # Smooth sigmoid transition from D=1 (Euclidean) to D≈2.65
    D = 1.0 + 1.65 / (1.0 + np.exp(-1.5 * (log_Re - 3.5)))
    # Fine structure: intermittency corrections at very high Re
    D += 0.05 * np.tanh(0.5 * (log_Re - 8.0))
    return D


def cascade_levels(Re):
    """
    Number of active cascade levels (decades of scale separation).
    Kolmogorov: η/L ~ Re^(-3/4), so number of decades ~ (3/4)·log10(Re)
    Below transition: 0 levels (no cascade)
    """
    Re = np.asarray(Re, dtype=float)
    n = np.where(Re > 2300,
                 0.75 * np.log10(Re / 2300.0),
                 0.0)
    return np.maximum(n, 0.0)


# ==========================================================================
#  FIGURE 94 — REYNOLDS NUMBER SWEEP
# ==========================================================================
def make_fig94():
    """Reynolds number sweep: coupled variable behavior across 17 OOM."""
    print("Generating Figure 94: Reynolds Number Sweep...")

    Re = np.logspace(-2, 15, 5000)
    log_Re = np.log10(Re)

    u_mag = velocity_magnitude(Re)
    dp_mag = pressure_gradient(Re)
    w_mag = vorticity_magnitude(Re)
    tke = turbulent_energy(Re)
    D_frac = fractal_dimension(Re)
    n_casc = cascade_levels(Re)

    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 1, height_ratios=[2, 1, 1], hspace=0.3)

    # --- Panel A: Coupled variable magnitudes ---
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(log_Re, u_mag, color=COLORS['velocity'], linewidth=2.5,
             label='Velocity |u| (normalized)', zorder=5)
    ax1.plot(log_Re, dp_mag, color=COLORS['pressure'], linewidth=2.5,
             label='Pressure gradient |∇p| (normalized)', zorder=5)
    ax1.plot(log_Re, w_mag, color=COLORS['vorticity'], linewidth=2.5,
             label='Vorticity |ω| (normalized)', zorder=5)
    ax1.plot(log_Re, tke, color=COLORS['energy'], linewidth=2.5,
             label='Turbulent KE (normalized)', zorder=5)

    # Mark critical Reynolds numbers
    for name, re_val in CRITICAL_RE.items():
        if re_val >= 1e-2 and re_val <= 1e15:
            ax1.axvline(np.log10(re_val), color='gray', alpha=0.4,
                       linestyle=':', linewidth=1)

    # Regime shading
    ax1.axvspan(-2, np.log10(47), alpha=0.08, color=COLORS['laminar'],
                label='Stokes/Creeping flow')
    ax1.axvspan(np.log10(47), np.log10(2300), alpha=0.08,
                color=COLORS['transition'], label='Laminar (structured)')
    ax1.axvspan(np.log10(2300), np.log10(1e6), alpha=0.08,
                color=COLORS['turbulent'], label='Transitional/Turbulent')
    ax1.axvspan(np.log10(1e6), 15, alpha=0.08, color=COLORS['extreme'],
                label='Extreme turbulence')

    ax1.set_xlabel('log₁₀(Re)')
    ax1.set_ylabel('Normalized magnitude')
    ax1.set_title('Figure 94 — Navier-Stokes Coupled Variable Behavior Across 17 Orders of Magnitude',
                  fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9, ncol=2)
    ax1.set_xlim(-2, 15)
    ax1.set_ylim(-0.05, 1.15)

    # Annotate key transitions
    ax1.annotate('Pipe flow\nRe = 2,300', xy=(np.log10(2300), 0.5),
                fontsize=8, ha='center', color='gray',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    ax1.annotate('Flat plate\nRe = 500,000', xy=(np.log10(5e5), 0.7),
                fontsize=8, ha='center', color='gray',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    # --- Panel B: Fractal dimension ---
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(log_Re, D_frac, color=COLORS['fractal'], linewidth=2.5)
    ax2.axhline(1.0, color='gray', linestyle='--', alpha=0.5, label='D = 1.0 (Euclidean)')
    ax2.axhline(5.0/3.0, color=COLORS['vorticity'], linestyle='--', alpha=0.5,
                label='D = 5/3 (Kolmogorov)')
    ax2.axhline(2.65, color=COLORS['energy'], linestyle='--', alpha=0.5,
                label='D ≈ 2.65 (isotropic turb.)')

    ax2.set_xlabel('log₁₀(Re)')
    ax2.set_ylabel('Fractal dimension D')
    ax2.set_title('Fractal Dimension of Velocity Field', fontweight='bold')
    ax2.legend(loc='center right', fontsize=9)
    ax2.set_xlim(-2, 15)
    ax2.set_ylim(0.8, 3.0)

    # Shade transition
    ax2.axvspan(np.log10(2300), np.log10(1e6), alpha=0.08,
                color=COLORS['transition'])

    # --- Panel C: Active cascade levels ---
    ax3 = fig.add_subplot(gs[2])
    ax3.plot(log_Re, n_casc, color=COLORS['turbulent'], linewidth=2.5)
    ax3.fill_between(log_Re, 0, n_casc, alpha=0.15, color=COLORS['turbulent'])

    ax3.set_xlabel('log₁₀(Re)')
    ax3.set_ylabel('Active cascade levels')
    ax3.set_title('Number of Active Energy Cascade Levels (Kolmogorov: η/L ~ Re⁻³ᐟ⁴)',
                  fontweight='bold')
    ax3.set_xlim(-2, 15)
    ax3.set_ylim(-0.2, 10)

    # Annotate
    ax3.annotate('No cascade\n(laminar)', xy=(1, 0.5), fontsize=9,
                ha='center', color='gray')
    ax3.annotate(f'~9 decades of\nscale separation\nat Re = 10¹⁵',
                xy=(14, n_casc[np.argmin(np.abs(log_Re - 14))]),
                fontsize=9, ha='right', color=COLORS['turbulent'],
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    plt.tight_layout()
    outpath = os.path.join(SCRIPT_DIR, 'fig94_reynolds_sweep.png')
    plt.savefig(outpath, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {outpath}")
    return outpath


# ==========================================================================
#  FIGURE 95 — GEOMETRIC MORPHOLOGY TRANSITION MAP
# ==========================================================================
def make_fig95():
    """Geometric morphology transition map with thresholds."""
    print("Generating Figure 95: Geometric Morphology Transition Map...")

    Re = np.logspace(-2, 15, 5000)
    log_Re = np.log10(Re)
    D_frac = fractal_dimension(Re)
    n_casc = cascade_levels(Re)

    fig, axes = plt.subplots(2, 1, figsize=(16, 10), height_ratios=[2, 1])

    # --- Panel A: Morphology map ---
    ax = axes[0]

    # Define regime boundaries
    regimes = [
        (-2, np.log10(1),    'Stokes\n(Re < 1)',       '#3498db', 'Euclidean\nD ≈ 1.00'),
        (np.log10(1), np.log10(47),   'Creeping\n(1 < Re < 47)',    '#2980b9', 'Euclidean\nD ≈ 1.00'),
        (np.log10(47), np.log10(2300), 'Laminar\n(47 < Re < 2300)', '#27ae60', 'Near-Euclidean\nD ≈ 1.00–1.05'),
        (np.log10(2300), np.log10(4000), 'Transitional\n(2300 < Re < 4000)', '#e67e22', 'Mixed\nD ≈ 1.05–1.5'),
        (np.log10(4000), np.log10(1e6),  'Turbulent\n(4000 < Re < 10⁶)',   '#e74c3c', 'Fractal\nD ≈ 1.5–2.4'),
        (np.log10(1e6), np.log10(1e9),   'Developed turb.\n(10⁶ < Re < 10⁹)', '#c0392b', 'Fractal\nD ≈ 2.4–2.6'),
        (np.log10(1e9), 15,              'Extreme turb.\n(Re > 10⁹)',        '#8e44ad', 'Deep fractal\nD ≈ 2.6–2.7'),
    ]

    for x0, x1, label, color, geom_label in regimes:
        ax.axvspan(x0, x1, alpha=0.2, color=color)
        xmid = (x0 + x1) / 2
        ax.text(xmid, 0.85, label, ha='center', va='top',
                transform=ax.get_xaxis_transform(), fontsize=9, fontweight='bold',
                color=color,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9))
        ax.text(xmid, 0.15, geom_label, ha='center', va='bottom',
                transform=ax.get_xaxis_transform(), fontsize=8, fontstyle='italic',
                color=color)

    # Plot fractal dimension curve
    ax.plot(log_Re, D_frac, color='black', linewidth=3, zorder=10,
            label='Fractal dimension D(Re)')

    # Mark critical Re values
    critical_markers = [
        (np.log10(47), 'Cylinder\nvortex\nshedding'),
        (np.log10(2300), 'Pipe flow\ncritical Re'),
        (np.log10(3.7e5), 'Drag crisis\n(sphere)'),
        (np.log10(5e5), 'Flat plate\ntransition'),
        (np.log10(3.5e6), 'Boundary\nlayer turb.'),
    ]

    for x_pos, label in critical_markers:
        D_at_x = np.interp(x_pos, log_Re, D_frac)
        ax.plot(x_pos, D_at_x, 'ko', markersize=8, zorder=11)
        ax.annotate(label, xy=(x_pos, D_at_x),
                   xytext=(x_pos, D_at_x + 0.25),
                   fontsize=7, ha='center', va='bottom',
                   arrowprops=dict(arrowstyle='->', color='black', lw=0.8),
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                            alpha=0.9, edgecolor='gray'))

    # Key reference lines
    ax.axhline(1.0, color='gray', linestyle='--', alpha=0.4)
    ax.axhline(5.0/3.0, color='green', linestyle=':', alpha=0.4)

    ax.set_xlabel('log₁₀(Re)')
    ax.set_ylabel('Fractal Dimension D')
    ax.set_title('Figure 95 — Geometric Morphology Transition Map: Navier-Stokes',
                fontsize=14, fontweight='bold')
    ax.set_xlim(-2, 15)
    ax.set_ylim(0.5, 3.2)
    ax.legend(loc='lower right', fontsize=10)

    # --- Panel B: Transition sharpness (dD/d(log Re)) ---
    ax2 = axes[1]
    dD = np.gradient(D_frac, log_Re)
    ax2.plot(log_Re, dD, color=COLORS['turbulent'], linewidth=2)
    ax2.fill_between(log_Re, 0, dD, alpha=0.15, color=COLORS['turbulent'])
    ax2.set_xlabel('log₁₀(Re)')
    ax2.set_ylabel('dD/d(log Re)')
    ax2.set_title('Transition Sharpness — Rate of Geometric Change', fontweight='bold')
    ax2.set_xlim(-2, 15)

    # Mark peak transition rate
    peak_idx = np.argmax(dD)
    ax2.annotate(f'Peak transition rate\nat Re ≈ {10**log_Re[peak_idx]:.0f}',
                xy=(log_Re[peak_idx], dD[peak_idx]),
                xytext=(log_Re[peak_idx] + 2, dD[peak_idx]),
                fontsize=9, ha='left',
                arrowprops=dict(arrowstyle='->', color='black'),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))

    plt.tight_layout()
    outpath = os.path.join(SCRIPT_DIR, 'fig95_morphology_transition.png')
    plt.savefig(outpath, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {outpath}")
    return outpath


# ==========================================================================
#  MAIN
# ==========================================================================
if __name__ == '__main__':
    print("=" * 72)
    print("  Script 94 — Navier-Stokes Phase 1: Reynolds Sweep & Morphology")
    print("=" * 72)

    fig94_path = make_fig94()
    fig95_path = make_fig95()

    print("\n" + "=" * 72)
    print("  Phase 1 complete. Two figures generated.")
    print("=" * 72)
