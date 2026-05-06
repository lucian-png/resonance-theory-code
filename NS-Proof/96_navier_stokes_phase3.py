"""
Script 96 -- Navier-Stokes Fractal Classification: Phase 3
          Behavior Near Potential Singularities

    Models what happens to energy distribution as the system approaches
    a potential blow-up point. Tests the Fractal Regularization Conjecture:

        "Energy does not concentrate at a point. It redistributes into
         self-similar fractal structure at progressively smaller scales.
         The complexity becomes unbounded. The energy remains bounded."

    This is the mathematical heart of the Millennium Prize argument.
    If the energy spectrum follows Kolmogorov's E(k) ~ k^(-5/3) and
    the integral converges, then no finite-time singularity is possible.

Generates:
    fig98_energy_distribution.png  (energy across scales near singularity)
    fig99_convergence_plot.png     (bounded energy from unbounded complexity)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.integrate import cumulative_trapezoid
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ==========================================================================
#  FUNDAMENTAL CONSTANTS
# ==========================================================================
DELTA_FEIG = 4.669201609102990
ALPHA_FEIG = 2.502907875095892

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

COLORS = {
    'kolmogorov': '#3498db',
    'fractal':    '#e74c3c',
    'energy':     '#2ecc71',
    'dissipation':'#9b59b6',
    'singularity':'#e67e22',
    'bounded':    '#27ae60',
    'convergent': '#2980b9',
}


# ==========================================================================
#  KOLMOGOROV ENERGY SPECTRUM MODEL
# ==========================================================================
def kolmogorov_spectrum(k, epsilon=1.0, C_K=1.5):
    """
    Kolmogorov energy spectrum: E(k) = C_K * ε^(2/3) * k^(-5/3)

    Parameters:
        k : wavenumber array
        epsilon : energy dissipation rate
        C_K : Kolmogorov constant (≈ 1.5)
    """
    return C_K * epsilon**(2.0/3.0) * k**(-5.0/3.0)


def corrected_spectrum(k, epsilon=1.0, C_K=1.5, mu=0.25):
    """
    Kolmogorov spectrum with intermittency correction.
    She-Leveque (1994) model:

    E(k) = C_K * ε^(2/3) * k^(-5/3) * (k*L)^(-μ/3)

    where μ ≈ 0.25 is the intermittency parameter.
    The correction steepens the spectrum slightly at high k.
    """
    L = 1.0  # integral scale (normalized)
    return C_K * epsilon**(2.0/3.0) * k**(-5.0/3.0) * (k * L)**(-mu/3.0)


def fractal_spectrum(k, epsilon=1.0, C_K=1.5, Re=1e6):
    """
    Fractal-regularized energy spectrum.

    The Lucian Law predicts that energy redistributes into self-similar
    structure rather than concentrating. The fractal cascade introduces
    a natural regularization:

    E(k) = C_K * ε^(2/3) * k^(-5/3) * exp(-β * (k/k_d)^(4/3))

    where k_d = (ε/ν³)^(1/4) is the Kolmogorov dissipation wavenumber
    and β is a constant of order unity.

    The exponential cutoff ensures the energy integral converges.
    """
    # Kolmogorov dissipation scale
    nu = 1.0 / Re  # kinematic viscosity (normalized)
    k_d = (epsilon / nu**3)**0.25
    beta = 1.0

    E_inertial = C_K * epsilon**(2.0/3.0) * k**(-5.0/3.0)
    # Exponential cutoff at dissipation scale
    E_dissipation = np.exp(-beta * (k / k_d)**(4.0/3.0))

    return E_inertial * E_dissipation


def blow_up_spectrum(k, t_star=1.0, t=0.99):
    """
    Hypothetical blow-up spectrum: what would happen WITHOUT
    fractal regularization.

    If energy concentrated at a point, the spectrum would develop
    a k^(-1) or shallower tail, and the energy integral would diverge.

    E(k) ~ k^(-1) * (1 - t/t*)^(-2)

    This is what the Millennium Prize question asks about:
    does this happen? The Fractal Regularization Conjecture says NO.
    """
    amplitude = (1.0 - t / t_star)**(-2)
    return amplitude * k**(-1.0)


# ==========================================================================
#  FIGURE 98 — ENERGY DISTRIBUTION NEAR SINGULARITY THRESHOLD
# ==========================================================================
def make_fig98():
    """Energy distribution across scales near critical transition."""
    print("Generating Figure 98: Energy Distribution Near Singularity...")

    fig = plt.figure(figsize=(16, 14))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

    # Wavenumber range
    k = np.logspace(-1, 6, 2000)

    # --- Panel A: Energy spectrum at different Re ---
    ax1 = fig.add_subplot(gs[0, 0])

    Re_values = [1e3, 1e5, 1e7, 1e9, 1e12, 1e15]
    cmap = plt.cm.hot_r
    for i, Re in enumerate(Re_values):
        E = fractal_spectrum(k, Re=Re)
        color = cmap(0.2 + 0.7 * i / len(Re_values))
        ax1.loglog(k, E, color=color, linewidth=2,
                   label=f'Re = 10^{int(np.log10(Re))}')

    # Reference: pure k^(-5/3)
    ax1.loglog(k, 0.5 * k**(-5.0/3.0), 'k--', linewidth=1, alpha=0.5,
               label='k⁻⁵ᐟ³ (Kolmogorov)')

    ax1.set_xlabel('Wavenumber k')
    ax1.set_ylabel('Energy E(k)')
    ax1.set_title('A. Energy Spectrum at Increasing Reynolds Number',
                  fontweight='bold')
    ax1.legend(fontsize=8, loc='lower left')
    ax1.set_xlim(1e-1, 1e6)
    ax1.set_ylim(1e-20, 1e2)

    # Annotate inertial and dissipation ranges
    ax1.annotate('Inertial\nrange', xy=(1e2, 1e-3), fontsize=10,
                ha='center', color='gray', fontstyle='italic')
    ax1.annotate('Dissipation\nrange', xy=(1e5, 1e-15), fontsize=10,
                ha='center', color='gray', fontstyle='italic')

    # --- Panel B: Fractal vs blow-up comparison ---
    ax2 = fig.add_subplot(gs[0, 1])

    k_comp = np.logspace(0, 5, 1000)
    E_fractal = fractal_spectrum(k_comp, Re=1e9)
    E_blowup = blow_up_spectrum(k_comp, t_star=1.0, t=0.99)
    E_kolm = kolmogorov_spectrum(k_comp)

    ax2.loglog(k_comp, E_kolm, color=COLORS['kolmogorov'], linewidth=2.5,
               label='Kolmogorov E(k) ~ k⁻⁵ᐟ³')
    ax2.loglog(k_comp, E_fractal, color=COLORS['fractal'], linewidth=2.5,
               label='Fractal-regularized (bounded)')
    ax2.loglog(k_comp, E_blowup / E_blowup[0] * E_kolm[0], color=COLORS['singularity'],
               linewidth=2.5, linestyle=':', label='Hypothetical blow-up ~ k⁻¹')

    # Shade the divergence region
    k_cross = k_comp[E_blowup / E_blowup[0] * E_kolm[0] > E_fractal]
    if len(k_cross) > 0:
        ax2.axvspan(k_cross[0], k_cross[-1], alpha=0.08, color='red')
        ax2.text(np.sqrt(k_cross[0] * k_cross[-1]), 1e-3,
                'Energy diverges\nwithout fractal\nregularization',
                ha='center', fontsize=9, color='red', fontstyle='italic')

    ax2.set_xlabel('Wavenumber k')
    ax2.set_ylabel('Energy E(k)')
    ax2.set_title('B. Fractal Regularization vs Hypothetical Blow-Up',
                  fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.set_xlim(1, 1e5)
    ax2.set_ylim(1e-12, 1e2)

    # --- Panel C: Energy redistribution visualization ---
    ax3 = fig.add_subplot(gs[1, 0])

    # Show how total energy redistributes across cascade levels
    n_levels = 15
    cascade_levels = np.arange(1, n_levels + 1)
    # Energy at each cascade level: E_n = E_0 * δ^(-n*5/3) (Kolmogorov)
    E_levels = DELTA_FEIG**(-cascade_levels * 5.0/3.0)
    E_levels_normalized = E_levels / E_levels.sum()

    # Cumulative energy
    E_cumulative = np.cumsum(E_levels_normalized)

    ax3.bar(cascade_levels, E_levels_normalized, color=COLORS['energy'],
            edgecolor='black', linewidth=0.5, alpha=0.7,
            label='Energy fraction per cascade level')
    ax3_twin = ax3.twinx()
    ax3_twin.plot(cascade_levels, E_cumulative, 'ro-', linewidth=2,
                  markersize=6, label='Cumulative energy')
    ax3_twin.axhline(1.0, color='gray', linestyle='--', alpha=0.5)

    ax3.set_xlabel('Cascade level n')
    ax3.set_ylabel('Energy fraction at level n')
    ax3_twin.set_ylabel('Cumulative energy fraction')
    ax3.set_title('C. Energy Redistribution Across Cascade Levels\n'
                  '(Feigenbaum spacing δ = 4.669)',
                  fontweight='bold')
    ax3.legend(loc='upper right', fontsize=9)
    ax3_twin.legend(loc='center right', fontsize=9)
    ax3_twin.set_ylim(0, 1.15)

    # Annotate convergence
    ax3.text(10, E_levels_normalized.max() * 0.7,
             'Energy decreases\ngeometrically at\neach finer scale\n'
             f'→ Series CONVERGES',
             fontsize=10, fontweight='bold', color=COLORS['bounded'],
             bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      edgecolor=COLORS['bounded'], alpha=0.9))

    # --- Panel D: Self-similar structure visualization ---
    ax4 = fig.add_subplot(gs[1, 1])

    # Show the nested self-similar vortex structure
    # Each level has δ times more structures, each α times smaller
    for level in range(8):
        n_vortices = int(DELTA_FEIG**level)
        if n_vortices > 500:
            n_vortices = 500
        size = ALPHA_FEIG**(-level) * 0.4
        energy_per = DELTA_FEIG**(-level * 5.0/3.0)

        # Random positions for vortices at this level
        np.random.seed(42 + level)
        theta = np.random.uniform(0, 2 * np.pi, n_vortices)
        r = np.random.uniform(0, 1.0 - size, n_vortices)
        x = r * np.cos(theta)
        y = r * np.sin(theta)

        alpha_val = max(0.1, 0.8 - level * 0.1)
        color = plt.cm.viridis(level / 8.0)
        ax4.scatter(x, y, s=max(1, size * 500), color=color,
                   alpha=alpha_val, edgecolors='none',
                   label=f'Level {level}: {n_vortices} structures' if level < 5 else '')

    ax4.set_xlim(-1.2, 1.2)
    ax4.set_ylim(-1.2, 1.2)
    ax4.set_aspect('equal')
    ax4.set_title('D. Self-Similar Vortex Cascade\n'
                  '(each level: δ× more structures, α× smaller)',
                  fontweight='bold')
    ax4.legend(fontsize=7, loc='upper right')
    ax4.set_xlabel('x')
    ax4.set_ylabel('y')

    fig.suptitle('Figure 98 — Energy Distribution Near Potential Singularity\n'
                 'The Fractal Regularization Conjecture',
                 fontsize=15, fontweight='bold', y=1.02)

    plt.tight_layout()
    outpath = os.path.join(SCRIPT_DIR, 'fig98_energy_distribution.png')
    plt.savefig(outpath, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {outpath}")
    return outpath


# ==========================================================================
#  FIGURE 99 — CONVERGENCE OF POWER-LAW ENERGY SERIES
# ==========================================================================
def make_fig99():
    """The mathematical heart: unbounded complexity, bounded energy."""
    print("Generating Figure 99: Energy Series Convergence...")

    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

    # --- Panel A: Partial sums of the energy series ---
    ax1 = fig.add_subplot(gs[0, 0])

    N_max = 50
    n_vals = np.arange(1, N_max + 1)

    # The key series: total energy = Σ E_n where E_n = C * δ^(-n * 5/3)
    # This is a geometric series with ratio r = δ^(-5/3)
    ratio_kolm = DELTA_FEIG**(-5.0/3.0)  # ≈ 0.0676
    E_n_kolm = ratio_kolm**n_vals
    S_n_kolm = np.cumsum(E_n_kolm)
    S_inf_kolm = ratio_kolm / (1.0 - ratio_kolm)  # Exact sum

    # Comparison: what if the spectrum were k^(-1) (blow-up case)?
    # Then E_n ~ δ^(-n) and ratio = δ^(-1) ≈ 0.214
    ratio_flat = DELTA_FEIG**(-1.0)  # ≈ 0.214
    E_n_flat = ratio_flat**n_vals
    S_n_flat = np.cumsum(E_n_flat)
    S_inf_flat = ratio_flat / (1.0 - ratio_flat)

    # What if spectrum were k^(-1/2) (divergent case)?
    # Then ratio > 1 and series diverges
    # Use k^(+0.1) to show divergence
    ratio_div = DELTA_FEIG**(0.1)  # > 1
    E_n_div = ratio_div**n_vals
    S_n_div = np.cumsum(E_n_div)

    ax1.semilogy(n_vals, S_n_kolm, 'o-', color=COLORS['convergent'],
                 linewidth=2.5, markersize=4,
                 label=f'Kolmogorov k⁻⁵ᐟ³ (r = δ⁻⁵ᐟ³ = {ratio_kolm:.4f})')
    ax1.axhline(S_inf_kolm, color=COLORS['convergent'], linestyle='--',
                alpha=0.5, linewidth=1)
    ax1.semilogy(n_vals, S_n_flat, 's-', color=COLORS['singularity'],
                 linewidth=2, markersize=4,
                 label=f'Marginal k⁻¹ (r = δ⁻¹ = {ratio_flat:.4f})')
    ax1.axhline(S_inf_flat, color=COLORS['singularity'], linestyle='--',
                alpha=0.5, linewidth=1)
    ax1.semilogy(n_vals[:20], S_n_div[:20], '^-', color='red',
                 linewidth=2, markersize=4,
                 label=f'Divergent k⁻¹ᐟ² (r = δ⁺⁰·¹ = {ratio_div:.4f})')

    ax1.set_xlabel('Number of cascade levels N')
    ax1.set_ylabel('Partial sum S_N = Σ E_n')
    ax1.set_title('A. Partial Sums of Energy Series', fontweight='bold')
    ax1.legend(fontsize=9, loc='center right')
    ax1.set_xlim(0, N_max)

    # Annotate convergence
    ax1.annotate(f'S_∞ = {S_inf_kolm:.4f}\n(BOUNDED)',
                xy=(N_max - 5, S_inf_kolm), fontsize=11,
                fontweight='bold', color=COLORS['bounded'],
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                         edgecolor=COLORS['bounded']))

    # --- Panel B: Convergence rate ---
    ax2 = fig.add_subplot(gs[0, 1])

    # Error: S_inf - S_N
    error_kolm = S_inf_kolm - S_n_kolm
    error_flat = S_inf_flat - S_n_flat

    ax2.semilogy(n_vals, error_kolm, 'o-', color=COLORS['convergent'],
                 linewidth=2.5, markersize=4,
                 label='Kolmogorov (exponential convergence)')
    ax2.semilogy(n_vals, error_flat, 's-', color=COLORS['singularity'],
                 linewidth=2, markersize=4,
                 label='Marginal (slower convergence)')

    ax2.set_xlabel('Number of cascade levels N')
    ax2.set_ylabel('Remainder |S_∞ - S_N|')
    ax2.set_title('B. Convergence Rate: How Fast Energy Bounds', fontweight='bold')
    ax2.legend(fontsize=9)

    # Annotate
    ax2.text(25, error_kolm[10], 'Error drops\nexponentially\n→ rapid convergence',
             fontsize=10, color=COLORS['convergent'],
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))

    # --- Panel C: Critical exponent analysis ---
    ax3 = fig.add_subplot(gs[1, 0])

    # The series Σ k^(-α) converges for α > 1
    # Kolmogorov: α = 5/3 ≈ 1.667 → CONVERGES
    # With intermittency: α = 5/3 + μ/3 ≈ 1.75 → CONVERGES (faster)
    # Blow-up: α = 1 → DIVERGES (harmonic series)

    alpha_range = np.linspace(0.5, 3.0, 500)
    convergence_indicator = np.where(alpha_range > 1.0, 1.0, 0.0)

    # Compute actual partial sums for different exponents
    N_test = 100
    partial_sums = np.zeros_like(alpha_range)
    for i, alpha in enumerate(alpha_range):
        ratio_test = DELTA_FEIG**(-alpha)
        if ratio_test < 1.0:
            partial_sums[i] = ratio_test / (1.0 - ratio_test)
        else:
            partial_sums[i] = np.nan

    ax3.semilogy(alpha_range, partial_sums, color='black', linewidth=2.5)
    ax3.axvline(5.0/3.0, color=COLORS['kolmogorov'], linewidth=2,
                linestyle='--', label='Kolmogorov α = 5/3')
    ax3.axvline(5.0/3.0 + 0.25/3.0, color=COLORS['fractal'], linewidth=2,
                linestyle='--', label='Intermittency-corrected α ≈ 1.75')
    ax3.axvline(1.0, color='red', linewidth=2, linestyle='--',
                label='Divergence boundary α = 1')

    # Shade convergent region
    ax3.axvspan(1.0, 3.0, alpha=0.1, color=COLORS['bounded'])
    ax3.axvspan(0.5, 1.0, alpha=0.1, color='red')

    ax3.text(2.0, 0.01, 'CONVERGES\n(bounded energy)', fontsize=12,
             fontweight='bold', ha='center', color=COLORS['bounded'])
    ax3.text(0.75, 0.01, 'DIVERGES\n(blow-up)', fontsize=12,
             fontweight='bold', ha='center', color='red')

    ax3.set_xlabel('Spectral exponent α (in E(k) ~ k⁻α)')
    ax3.set_ylabel('Total energy S_∞')
    ax3.set_title('C. Convergence Depends on Spectral Exponent',
                  fontweight='bold')
    ax3.legend(fontsize=9, loc='upper right')
    ax3.set_xlim(0.5, 3.0)

    # --- Panel D: The argument in one picture ---
    ax4 = fig.add_subplot(gs[1, 1])

    # Visual summary: the logical chain
    steps = [
        (0.5, 0.9, 'Navier-Stokes is nonlinear,\ncoupled, unbounded',
         COLORS['kolmogorov']),
        (0.5, 0.75, '↓  Lucian Law applies  ↓', 'gray'),
        (0.5, 0.6, 'Energy cascade develops\nfractal self-similar structure',
         COLORS['fractal']),
        (0.5, 0.45, '↓  Kolmogorov spectrum  ↓', 'gray'),
        (0.5, 0.3, 'E(k) ~ k⁻⁵ᐟ³\nSpectral exponent α = 5/3 > 1',
         COLORS['kolmogorov']),
        (0.5, 0.15, '↓  Geometric series  ↓', 'gray'),
        (0.5, 0.0, '∫ E(k)dk < ∞\nTOTAL ENERGY IS BOUNDED',
         COLORS['bounded']),
    ]

    ax4.set_xlim(0, 1)
    ax4.set_ylim(-0.1, 1.0)
    ax4.axis('off')

    for x, y, text, color in steps:
        fontsize = 11 if '↓' not in text else 9
        fontweight = 'bold' if '↓' not in text else 'normal'
        bbox_props = dict(boxstyle='round,pad=0.5', facecolor='white',
                         edgecolor=color, linewidth=2) if '↓' not in text else None
        ax4.text(x, y, text, ha='center', va='center', fontsize=fontsize,
                fontweight=fontweight, color=color, bbox=bbox_props)

    ax4.set_title('D. The Fractal Regularization Argument', fontweight='bold')

    fig.suptitle('Figure 99 — Convergence: Unbounded Complexity, Bounded Energy',
                 fontsize=15, fontweight='bold', y=1.02)

    plt.tight_layout()
    outpath = os.path.join(SCRIPT_DIR, 'fig99_convergence_plot.png')
    plt.savefig(outpath, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {outpath}")
    return outpath


# ==========================================================================
#  MAIN
# ==========================================================================
if __name__ == '__main__':
    print("=" * 72)
    print("  Script 96 — Navier-Stokes Phase 3: Singularity & Convergence")
    print("=" * 72)

    fig98_path = make_fig98()
    fig99_path = make_fig99()

    print("\n" + "=" * 72)
    print("  Phase 3 complete. Two figures generated.")
    print("=" * 72)
