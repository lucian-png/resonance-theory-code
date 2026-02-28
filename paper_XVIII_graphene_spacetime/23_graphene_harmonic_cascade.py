#!/usr/bin/env python3
"""
==============================================================================
PAPER XVIII: HARMONIC CASCADE STRUCTURE IN GRAPHENE CHARGE DENSITY
==============================================================================
A Fractal Geometric Analysis Using the Lucian Method

STATUS: WITHHELD — IP PROTECTED — DO NOT PUBLISH

Applies the Lucian Method to graphene's charge storage equations:
  - Driving variable: Charge carrier density n (10⁸ to 10¹⁶ cm⁻²)
  - Equations held sacred: graphene tight-binding DOS, quantum capacitance
  - Extreme range: 8 orders of magnitude in carrier density
  - Geometric morphology: capacitance, energy density, cascade structure

Outputs:
    fig22_graphene_harmonic_analysis.png   — 6-panel harmonic analysis
    fig23_graphene_cascade_prediction.png  — 6-panel cascade & classification

DO NOT PUSH TO GITHUB. DO NOT PUBLISH.
==============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# NUMPY REPLACEMENTS FOR SCIPY (scipy not installed)
# =============================================================================

def cumulative_trapezoid(y: np.ndarray, x: np.ndarray,
                         initial: float = None) -> np.ndarray:
    """Cumulative trapezoidal integration (replaces scipy.integrate.cumulative_trapezoid)."""
    dx = np.diff(x)
    avg_y = (y[:-1] + y[1:]) / 2.0
    cum = np.cumsum(dx * avg_y)
    if initial is not None:
        cum = np.concatenate([[initial], cum])
    return cum


def find_peaks_numpy(data: np.ndarray, prominence: float = 0.05,
                     distance: int = 50) -> tuple:
    """Simple peak finder (replaces scipy.signal.find_peaks)."""
    # Find local maxima
    peaks = []
    for i in range(1, len(data) - 1):
        if data[i] > data[i - 1] and data[i] > data[i + 1]:
            peaks.append(i)

    if not peaks:
        return np.array([], dtype=int), {}

    # Apply distance filter
    if distance > 1:
        filtered = [peaks[0]]
        for p in peaks[1:]:
            if p - filtered[-1] >= distance:
                filtered.append(p)
        peaks = filtered

    # Apply prominence filter
    if prominence > 0:
        final = []
        for p in peaks:
            left_start = max(0, p - distance)
            right_end = min(len(data), p + distance + 1)
            left_min = np.min(data[left_start:p]) if p > left_start else data[p]
            right_min = np.min(data[p + 1:right_end]) if p + 1 < right_end else data[p]
            prom = data[p] - max(left_min, right_min)
            if prom >= prominence:
                final.append(p)
        peaks = final

    return np.array(peaks, dtype=int), {}


# =============================================================================
# PHYSICAL CONSTANTS (all SI)
# =============================================================================

e_charge = 1.602176634e-19      # Elementary charge (C)
hbar = 1.054571817e-34          # Reduced Planck constant (J·s)
v_F = 1.0e6                     # Fermi velocity in graphene (m/s)
gamma_0 = 2.8 * e_charge       # Nearest-neighbor hopping energy (J) = 2.8 eV
a_cc = 1.42e-10                 # C-C bond length (m)
a_lat = a_cc * np.sqrt(3)      # Lattice constant (m)
epsilon_0 = 8.854187817e-12     # Vacuum permittivity (F/m)
k_B = 1.380649e-23             # Boltzmann constant (J/K)
eV = e_charge                   # 1 eV in Joules


# =============================================================================
# GRAPHENE DENSITY OF STATES — FULL TIGHT-BINDING MODEL
# =============================================================================

def compute_graphene_dos(n_energy: int = 8000, n_k: int = 2000) -> tuple:
    """
    Compute the graphene DOS from the exact tight-binding dispersion relation
    on the honeycomb lattice. No approximations — the equation held sacred.

    E(k) = ±γ₀ √(3 + f(k))
    f(k) = 2cos(√3 k_y a) + 4cos(√3 k_y a/2)cos(3 k_x a/2)

    Returns:
        energies: array of energy values (J)
        dos: density of states per unit area (states / J / m²), smoothed
    """
    # Sample the Brillouin zone with high resolution
    b1 = np.array([2 * np.pi / (a_lat * np.sqrt(3)), 2 * np.pi / a_lat])
    b2 = np.array([2 * np.pi / (a_lat * np.sqrt(3)), -2 * np.pi / a_lat])

    # Uniform sampling over the BZ using fractional coordinates
    f1 = np.linspace(0, 1, n_k, endpoint=False)
    f2 = np.linspace(0, 1, n_k, endpoint=False)
    F1, F2 = np.meshgrid(f1, f2)

    kx = F1.ravel() * b1[0] + F2.ravel() * b2[0]
    ky = F1.ravel() * b1[1] + F2.ravel() * b2[1]

    # Exact tight-binding dispersion
    f_k = (2.0 * np.cos(ky * a_lat) +
           4.0 * np.cos(ky * a_lat / 2.0) * np.cos(kx * np.sqrt(3) * a_lat / 2.0))

    # E = ±γ₀ √(3 + f(k))
    arg = 3.0 + f_k
    arg = np.clip(arg, 0, None)
    E_plus = gamma_0 * np.sqrt(arg)
    E_minus = -gamma_0 * np.sqrt(arg)
    all_energies = np.concatenate([E_plus, E_minus])

    # Histogram into DOS
    E_max = 3.1 * gamma_0
    energy_bins = np.linspace(-E_max, E_max, n_energy + 1)
    counts, edges = np.histogram(all_energies, bins=energy_bins)

    energies = (edges[:-1] + edges[1:]) / 2.0
    dE = edges[1] - edges[0]

    # Normalize
    A_uc = np.sqrt(3) / 2.0 * a_lat ** 2
    n_states_total = 2
    total_count = len(all_energies)
    dos_raw = counts / total_count * n_states_total / A_uc / dE

    # Gaussian smoothing to remove histogram noise while preserving Van Hove peak
    # Width ~0.02 eV — narrow enough to keep the singularity structure
    sigma_bins = int(0.02 * eV / dE)
    sigma_bins = max(sigma_bins, 3)
    kernel_size = sigma_bins * 6 + 1
    x_kern = np.arange(kernel_size) - kernel_size // 2
    gauss_kernel = np.exp(-0.5 * (x_kern / sigma_bins) ** 2)
    gauss_kernel /= gauss_kernel.sum()
    dos = np.convolve(dos_raw, gauss_kernel, mode='same')

    return energies, dos


def dirac_cone_dos(E: np.ndarray) -> np.ndarray:
    """Dirac cone (linear) approximation for graphene DOS (per unit area, per spin)."""
    # D(E) = 2|E| / (π (ℏv_F)²) — includes both valleys and both spins
    return 2.0 * np.abs(E) / (np.pi * (hbar * v_F) ** 2)


# =============================================================================
# CHARGE DENSITY ↔ FERMI ENERGY (from DOS integration)
# =============================================================================

def fermi_energy_from_density(n_carrier: np.ndarray, energies: np.ndarray,
                               dos: np.ndarray) -> np.ndarray:
    """
    Compute Fermi energy for given carrier densities by integrating the DOS.
    n = ∫₀^{E_F} D(E) dE

    Uses numerical integration of the tight-binding DOS.
    """
    # Only positive energies for electron doping
    pos_mask = energies > 0
    E_pos = energies[pos_mask]
    D_pos = dos[pos_mask]

    # Cumulative integral: n(E_F) = ∫₀^{E_F} D(E) dE
    cum_n = cumulative_trapezoid(D_pos, E_pos, initial=0)

    # Interpolate to find E_F for each target n
    E_F = np.interp(n_carrier, cum_n, E_pos, left=E_pos[0], right=E_pos[-1])
    return E_F


def dirac_fermi_energy(n_carrier: np.ndarray) -> np.ndarray:
    """Dirac cone approximation: E_F = ℏv_F √(πn)"""
    return hbar * v_F * np.sqrt(np.pi * n_carrier)


# =============================================================================
# QUANTUM CAPACITANCE
# =============================================================================

def quantum_capacitance_from_dos(E_F: np.ndarray, energies: np.ndarray,
                                  dos: np.ndarray) -> np.ndarray:
    """C_Q = e² × D(E_F) — quantum capacitance per unit area."""
    D_at_EF = np.interp(np.abs(E_F), energies[energies >= 0], dos[energies >= 0])
    return e_charge ** 2 * D_at_EF


def dirac_quantum_capacitance(n_carrier: np.ndarray) -> np.ndarray:
    """Dirac cone approximation for quantum capacitance."""
    E_F = dirac_fermi_energy(n_carrier)
    D_EF = dirac_cone_dos(E_F)
    return e_charge ** 2 * D_EF


# =============================================================================
# EXCHANGE-CORRELATION CORRECTIONS (2D electron gas)
# =============================================================================

def exchange_correction_capacitance(n_carrier: np.ndarray,
                                     C_Q: np.ndarray,
                                     C_geo: float) -> np.ndarray:
    """
    Exchange-correlation correction to quantum capacitance (per unit area).
    For 2D Dirac fermions in graphene:
        ε_x(n) ≈ -α_g × ℏv_F √(πn) / 2
    where α_g = e²/(4πε₀ℏv_F κ) is the graphene fine structure constant.
    κ ≈ 2.5 (effective dielectric constant for graphene on substrate)

    The exchange correction to inverse capacitance:
        1/C_xc = -(1/e²) × ∂²(nε_x)/∂n²

    Physical bounds:
        - Correction vanishes at low density (E_F < k_BT, no Fermi surface)
        - Magnitude capped at 30% of 1/C_Q (correction, not dominant term)
    """
    kappa = 2.5
    alpha_g = e_charge ** 2 / (4.0 * np.pi * epsilon_0 * hbar * v_F * kappa)

    # Raw exchange inverse-capacitance correction
    inv_C_xc_raw = (alpha_g * hbar * v_F * np.sqrt(np.pi) * 0.75) / \
                   (e_charge ** 2 * np.sqrt(n_carrier + 1e14))

    # Physical cutoff: at low density, thermal smearing destroys Fermi surface
    # E_F = ℏv_F √(πn), k_BT ≈ 26 meV at 300K
    E_F = hbar * v_F * np.sqrt(np.pi * n_carrier)
    kBT = k_B * 300.0
    thermal_factor = np.minimum(E_F / (kBT + 1e-30), 1.0)  # 0→1 as E_F exceeds k_BT

    # Bound: exchange correction cannot exceed 30% of quantum inverse-capacitance
    inv_C_Q = 1.0 / (C_Q + 1e-10)
    max_correction = 0.30 * inv_C_Q

    inv_C_xc = np.minimum(inv_C_xc_raw * thermal_factor, max_correction)

    # Negative contribution to 1/C_total (enhances total capacitance)
    return -inv_C_xc


# =============================================================================
# TOTAL CAPACITANCE AND ENERGY DENSITY
# =============================================================================

def compute_total_capacitance(n_carrier: np.ndarray, C_Q: np.ndarray,
                               d_sep: float = 0.5e-9,
                               eps_r: float = 3.0) -> tuple:
    """
    Total capacitance: 1/C_total = 1/C_geo + 1/C_Q + 1/C_xc

    Args:
        d_sep: effective electrode-electrolyte separation (m)
        eps_r: relative permittivity of electrolyte
    """
    C_geo = epsilon_0 * eps_r / d_sep  # geometric capacitance per unit area

    # Exchange correction (bounded, thermally regularized)
    inv_C_xc = exchange_correction_capacitance(n_carrier, C_Q, C_geo)

    # Total: series combination with exchange correction
    inv_C_total = 1.0 / C_geo + 1.0 / (C_Q + 1e-10) + inv_C_xc

    # Physical regularization: inv_C_total must be positive
    # (negative = charge instability; clip for numerics but note as cascade point)
    inv_C_min = 0.05 / C_geo  # cap C_total at 20× C_geo
    inv_C_total = np.maximum(inv_C_total, inv_C_min)
    C_total = 1.0 / inv_C_total

    return C_total, C_geo


def compute_energy_density(n_carrier: np.ndarray, C_total: np.ndarray) -> np.ndarray:
    """
    Energy density U(n) = ∫₀ⁿ V(n') dn'
    where V(n) = e × n / C_total(n) is the voltage at carrier density n.
    Per unit area, in J/m².
    """
    # Voltage: V = Q/C = e·n / C_total (charge per area = e·n)
    V = e_charge * n_carrier / C_total

    # Energy density: U = ∫₀ⁿ V(n') dn' via cumulative trapezoidal
    U = cumulative_trapezoid(e_charge * V, n_carrier, initial=0)
    return U


# =============================================================================
# EDGE STATE MODEL
# =============================================================================

def zigzag_edge_dos_correction(energies: np.ndarray, edge_fraction: float = 0.01) -> np.ndarray:
    """
    Zigzag edge contribution to DOS: localized states at E = 0.
    Modeled as a Lorentzian peak at E = 0 with width ~0.1γ₀.
    edge_fraction: fraction of atoms at zigzag edges (geometry dependent)
    """
    width = 0.1 * gamma_0
    peak = edge_fraction / (np.pi * width) * (width ** 2 / (energies ** 2 + width ** 2))
    # Convert to per unit area: assume typical nanoribbon with edge_fraction of atoms on edges
    A_uc = np.sqrt(3) / 2.0 * a_lat ** 2
    return peak / A_uc


# =============================================================================
# COLOR PALETTE
# =============================================================================

COLORS = {
    'red': '#e74c3c',
    'blue': '#3498db',
    'green': '#2ecc71',
    'purple': '#9b59b6',
    'orange': '#e67e22',
    'cyan': '#1abc9c',
    'yellow': '#f1c40f',
    'dark': '#2c3e50',
    'gold': '#f39c12',
}


# =============================================================================
# FIGURE 1: GRAPHENE HARMONIC ANALYSIS
# =============================================================================

def generate_figure_1(energies: np.ndarray, dos: np.ndarray,
                       n_carrier: np.ndarray, E_F: np.ndarray,
                       C_Q: np.ndarray, C_total: np.ndarray,
                       C_geo: float, U: np.ndarray) -> None:
    print("=" * 70)
    print("FIGURE 1: Graphene Harmonic Analysis — The Lucian Method")
    print("=" * 70)

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(
        "The Lucian Method — Graphene Charge Density Harmonic Analysis\n"
        "Driving Variable: Carrier Density n, Swept Across 8 Orders of Magnitude",
        fontsize=15, fontweight='bold', y=0.98
    )
    gs = GridSpec(2, 3, hspace=0.35, wspace=0.3)

    n_cm2 = n_carrier * 1e-4  # convert to cm⁻²

    # =========================================================================
    # Panel 1: Quantum Capacitance Across Extreme Density
    # =========================================================================
    print("  Panel 1: Quantum Capacitance...")
    ax1 = fig.add_subplot(gs[0, 0])

    # Also compute Dirac approximation for comparison
    C_Q_dirac = dirac_quantum_capacitance(n_carrier)

    ax1.loglog(n_cm2, C_Q * 1e4, color=COLORS['blue'], linewidth=2, label='Full tight-binding')
    ax1.loglog(n_cm2, C_Q_dirac * 1e4, color=COLORS['orange'], linewidth=1.5,
               linestyle='--', alpha=0.7, label='Dirac cone approx.')
    ax1.set_xlabel("Carrier density n (cm⁻²)", fontsize=10)
    ax1.set_ylabel("C_Q (μF/cm²)", fontsize=10)
    ax1.set_title("Quantum Capacitance Across Extreme Density", fontsize=12, fontweight='bold')
    ax1.legend(fontsize=8, loc='upper left', framealpha=0.9)
    ax1.grid(True, alpha=0.2, which='both')

    # Find and annotate Van Hove spike
    vH_idx = np.argmax(C_Q)
    vH_n = n_cm2[vH_idx]
    vH_C = C_Q[vH_idx] * 1e4
    ax1.annotate(
        f"Van Hove Singularity\nn ≈ {vH_n:.1e} cm⁻²\nC_Q SPIKES",
        xy=(vH_n, vH_C), xytext=(vH_n * 0.01, vH_C * 0.7),
        fontsize=7, color=COLORS['red'], fontweight='bold',
        arrowprops=dict(arrowstyle='->', color=COLORS['red'], lw=1.5),
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#fff5f5', edgecolor=COLORS['red'], alpha=0.9)
    )

    # Annotate regimes
    ax1.axvspan(1e8, 1e12, alpha=0.05, color=COLORS['blue'])
    ax1.axvspan(1e12, 1e14, alpha=0.05, color=COLORS['green'])
    ax1.axvspan(1e14, 1e16, alpha=0.05, color=COLORS['red'])
    ax1.text(1e9, ax1.get_ylim()[0] * 3, "Dirac\nregime", fontsize=6, alpha=0.6, color=COLORS['blue'])
    ax1.text(3e12, ax1.get_ylim()[0] * 3, "Transition", fontsize=6, alpha=0.6, color=COLORS['green'])
    ax1.text(3e14, ax1.get_ylim()[0] * 3, "Van Hove", fontsize=6, alpha=0.6, color=COLORS['red'])

    # =========================================================================
    # Panel 2: Energy Density Landscape
    # =========================================================================
    print("  Panel 2: Energy Density Landscape...")
    ax2 = fig.add_subplot(gs[0, 1])

    # Convert U from J/m² to J/cm²
    U_cm2 = U * 1e-4

    ax2_twin = ax2.twinx()

    ax2.loglog(n_cm2, U_cm2, color=COLORS['green'], linewidth=2, label='Energy density U(n)')
    ax2.set_xlabel("Carrier density n (cm⁻²)", fontsize=10)
    ax2.set_ylabel("Energy density U (J/cm²)", fontsize=10, color=COLORS['green'])
    ax2.tick_params(axis='y', labelcolor=COLORS['green'])

    # dU/dn — incremental energy per carrier (reveals cascade slopes)
    dU_dn = np.gradient(U, n_carrier)
    ax2_twin.loglog(n_cm2, dU_dn * 1e-4, color=COLORS['purple'], linewidth=1.5,
                     alpha=0.8, label='dU/dn')
    ax2_twin.set_ylabel("dU/dn (J·cm²)", fontsize=10, color=COLORS['purple'])
    ax2_twin.tick_params(axis='y', labelcolor=COLORS['purple'])

    ax2.set_title("Energy Density Landscape", fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.2, which='both')

    # Annotate slope changes
    ax2.annotate(
        "Slope changes = cascade transitions\nEnergy per carrier jumps at each transition",
        xy=(0.03, 0.97), xycoords='axes fraction', va='top',
        fontsize=7, alpha=0.8,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8)
    )

    # =========================================================================
    # Panel 3: Total Capacitance Decomposition
    # =========================================================================
    print("  Panel 3: Total Capacitance Decomposition...")
    ax3 = fig.add_subplot(gs[0, 2])

    ax3.loglog(n_cm2, C_total * 1e4, color=COLORS['red'], linewidth=2.5, label='C_total')
    ax3.loglog(n_cm2, C_Q * 1e4, color=COLORS['blue'], linewidth=1.5, linestyle='--',
               alpha=0.8, label='C_Q (quantum)')
    ax3.axhline(y=C_geo * 1e4, color=COLORS['orange'], linewidth=1.5, linestyle=':',
                alpha=0.8, label=f'C_geo = {C_geo*1e4:.1f} μF/cm²')

    ax3.set_xlabel("Carrier density n (cm⁻²)", fontsize=10)
    ax3.set_ylabel("Capacitance (μF/cm²)", fontsize=10)
    ax3.set_title("Total Capacitance Decomposition", fontsize=12, fontweight='bold')
    ax3.legend(fontsize=8, loc='lower right', framealpha=0.9)
    ax3.grid(True, alpha=0.2, which='both')

    # Annotate crossover
    crossover_idx = np.argmin(np.abs(C_Q - C_geo))
    cross_n = n_cm2[crossover_idx]
    ax3.axvline(x=cross_n, color=COLORS['dark'], linestyle='--', alpha=0.4)
    ax3.annotate(
        f"Cascade 1: Q→C crossover\nn ≈ {cross_n:.1e} cm⁻²",
        xy=(cross_n, C_geo * 1e4 * 0.5), fontsize=7, alpha=0.8,
        color=COLORS['dark'], ha='center',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8)
    )

    # =========================================================================
    # Panel 4: Density of States — Full Band Structure
    # =========================================================================
    print("  Panel 4: Full Band Structure DOS...")
    ax4 = fig.add_subplot(gs[1, 0])

    E_eV = energies / eV
    dos_eV = dos * eV  # convert to states / eV / m²

    # Dirac approximation
    E_dirac = np.linspace(-3 * gamma_0 / eV, 3 * gamma_0 / eV, 1000)
    D_dirac = dirac_cone_dos(E_dirac * eV) * eV

    ax4.plot(E_eV, dos_eV, color=COLORS['blue'], linewidth=1.5, label='Full tight-binding')
    ax4.plot(E_dirac, D_dirac, color=COLORS['orange'], linewidth=1, linestyle='--',
             alpha=0.6, label='Dirac cone approx.')
    ax4.set_xlabel("Energy (eV)", fontsize=10)
    ax4.set_ylabel("DOS (states / eV / m²)", fontsize=10)
    ax4.set_title("Density of States: Full Band Structure", fontsize=12, fontweight='bold')
    ax4.legend(fontsize=8, loc='upper left', framealpha=0.9)
    ax4.grid(True, alpha=0.2)
    ax4.set_xlim(-10, 10)

    # Annotate Van Hove
    ax4.axvline(x=2.8, color=COLORS['red'], linestyle='--', alpha=0.5)
    ax4.axvline(x=-2.8, color=COLORS['red'], linestyle='--', alpha=0.5)
    ax4.annotate("Van Hove\nsingularity", xy=(2.8, ax4.get_ylim()[1] * 0.8),
                 fontsize=7, color=COLORS['red'], fontweight='bold', ha='center')
    ax4.annotate("Dirac point\n(linear DOS)", xy=(0, ax4.get_ylim()[1] * 0.5),
                 fontsize=7, color=COLORS['dark'], ha='center',
                 arrowprops=dict(arrowstyle='->', color=COLORS['dark']),
                 xytext=(1.5, ax4.get_ylim()[1] * 0.65))

    # =========================================================================
    # Panel 5: Electric Field at Surface
    # =========================================================================
    print("  Panel 5: Surface Electric Field...")
    ax5 = fig.add_subplot(gs[1, 1])

    # Surface electric field: E_field = σ / (2ε₀ε_r) = e·n / (2ε₀ε_r)
    eps_r = 3.0
    E_field = e_charge * n_carrier / (2.0 * epsilon_0 * eps_r)  # V/m

    # Thomas-Fermi screening length
    # λ_TF = √(ε₀ε_r / (e² D(E_F)))
    D_at_EF = C_Q / e_charge ** 2  # recover D(E_F) from C_Q
    lambda_TF = np.sqrt(epsilon_0 * eps_r / (e_charge ** 2 * D_at_EF + 1e-30))

    ax5.loglog(n_cm2, E_field * 1e-9, color=COLORS['cyan'], linewidth=2, label='Surface E-field')
    ax5.set_xlabel("Carrier density n (cm⁻²)", fontsize=10)
    ax5.set_ylabel("Electric field (GV/m)", fontsize=10, color=COLORS['cyan'])
    ax5.tick_params(axis='y', labelcolor=COLORS['cyan'])

    ax5_twin = ax5.twinx()
    ax5_twin.loglog(n_cm2, lambda_TF * 1e9, color=COLORS['gold'], linewidth=1.5,
                     linestyle='--', alpha=0.8, label='Screening length λ_TF')
    ax5_twin.set_ylabel("Screening length (nm)", fontsize=10, color=COLORS['gold'])
    ax5_twin.tick_params(axis='y', labelcolor=COLORS['gold'])

    ax5.set_title("Surface Field & Screening", fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.2, which='both')

    ax5.annotate(
        "Screening collapse at Van Hove\n→ field morphology changes",
        xy=(0.03, 0.03), xycoords='axes fraction', va='bottom',
        fontsize=7, alpha=0.8,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8)
    )

    # =========================================================================
    # Panel 6: Cascade Detection (Second Derivative)
    # =========================================================================
    print("  Panel 6: Cascade Detection...")
    ax6 = fig.add_subplot(gs[1, 2])

    # Use d(log C_total)/d(log n) — logarithmic derivative reveals cascade structure
    log_n = np.log10(n_cm2)
    log_C = np.log10(C_total * 1e4)  # μF/cm²

    # Logarithmic derivative of total capacitance
    dlogC_dlogn = np.gradient(log_C, log_n)

    # Heavy smoothing — we want the macroscopic regime transitions, not noise
    window = 81
    if len(dlogC_dlogn) > window:
        kernel = np.ones(window) / window
        dlogC_smooth = np.convolve(dlogC_dlogn, kernel, mode='same')
    else:
        dlogC_smooth = dlogC_dlogn

    ax6.plot(n_cm2, dlogC_smooth, color=COLORS['red'], linewidth=2)
    ax6.set_xlabel("Carrier density n (cm⁻²)", fontsize=10)
    ax6.set_ylabel("d(log C)/d(log n)", fontsize=10)
    ax6.set_xscale('log')
    ax6.set_title("Cascade Detection: Capacitance Slope", fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.2)
    ax6.axhline(y=0, color='gray', linewidth=0.5, alpha=0.5)
    ax6.axhline(y=0.5, color=COLORS['blue'], linewidth=0.5, alpha=0.3, linestyle='--')
    ax6.text(1.5e8, 0.52, 'Dirac theory: slope = 0.5', fontsize=6, alpha=0.5, color=COLORS['blue'])

    # ===== CASCADE DETECTION: Physics-first approach =====
    # Define cascades from the actual physical transitions:
    cascade_densities = []
    cascade_labels = []

    # Cascade 1: Thermal activation crossover (E_F ≈ k_BT → charge storage begins)
    kBT = k_B * 300.0
    E_F_thermal = kBT
    thermal_idx = np.argmin(np.abs(E_F - E_F_thermal))
    cascade_densities.append(n_cm2[thermal_idx])
    cascade_labels.append("Thermal\nactivation")

    # Cascade 2: Quantum → Classical crossover (C_Q ≈ C_geo)
    crossover_idx = np.argmin(np.abs(C_Q - C_geo))
    cascade_densities.append(n_cm2[crossover_idx])
    cascade_labels.append("Q→C\ncrossover")

    # Cascade 3: Van Hove singularity (max C_Q = max DOS)
    vH_idx = np.argmax(C_Q)
    cascade_densities.append(n_cm2[vH_idx])
    cascade_labels.append("Van Hove\nsingularity")

    # Cascade 4: Post-Van Hove collapse (C_Q drops back below C_geo after spike)
    post_vH_mask = np.arange(len(C_Q)) > vH_idx
    if np.any(post_vH_mask):
        post_vH_CQ = C_Q[post_vH_mask]
        post_cross = np.where(post_vH_CQ < C_geo)[0]
        if len(post_cross) > 0:
            post_idx = np.where(post_vH_mask)[0][post_cross[0]]
            cascade_densities.append(n_cm2[post_idx])
            cascade_labels.append("Band-edge\ncollapse")

    # Annotate cascades
    cascade_colors_list = [COLORS['blue'], COLORS['green'], COLORS['red'],
                           COLORS['purple'], COLORS['cyan']]
    for i, (cd, cl) in enumerate(zip(cascade_densities, cascade_labels)):
        cd_idx = np.argmin(np.abs(n_cm2 - cd))
        y_val = dlogC_smooth[min(cd_idx, len(dlogC_smooth) - 1)]
        col = cascade_colors_list[i % len(cascade_colors_list)]
        ax6.axvline(x=cd, color=col, linestyle='--', alpha=0.6, linewidth=1.5)
        # Stagger vertically
        y_top = ax6.get_ylim()[1]
        y_pos = y_top - (i + 1) * (y_top - ax6.get_ylim()[0]) * 0.15
        ax6.annotate(
            f"C{i+1}: {cl}\nn ≈ {cd:.1e}",
            xy=(cd, y_val),
            xytext=(cd * 0.15, y_pos),
            fontsize=6, color=col, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=col, lw=1)
        )

    ax6.annotate(
        "Slope changes in d(log C)/d(log n)\nlocate harmonic cascade transitions",
        xy=(0.97, 0.97), xycoords='axes fraction', va='top', ha='right',
        fontsize=7, alpha=0.8,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8)
    )

    print(f"\n  CASCADES DETECTED:")
    for i, (cd, cl) in enumerate(zip(cascade_densities, cascade_labels)):
        print(f"    C{i+1}: {cl.replace(chr(10), ' ')} — n ≈ {cd:.2e} cm⁻²")

    plt.savefig('fig22_graphene_harmonic_analysis.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  ✓ Saved fig22_graphene_harmonic_analysis.png")
    return cascade_densities, cascade_labels


# =============================================================================
# FIGURE 2: FRACTAL CLASSIFICATION & CASCADE PREDICTION
# =============================================================================

def generate_figure_2(energies: np.ndarray, dos: np.ndarray,
                       n_carrier: np.ndarray, E_F: np.ndarray,
                       C_Q: np.ndarray, C_total: np.ndarray,
                       U: np.ndarray, cascade_densities: list,
                       cascade_labels: list) -> None:
    print("\n" + "=" * 70)
    print("FIGURE 2: Fractal Classification & Cascade Prediction")
    print("=" * 70)

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(
        "Fractal Classification & Cascade Prediction — Graphene Charge Density\n"
        "The Lucian Method Applied to Energy Storage Geometry",
        fontsize=15, fontweight='bold', y=0.98
    )
    gs = GridSpec(2, 3, hspace=0.35, wspace=0.3)

    n_cm2 = n_carrier * 1e-4

    # =========================================================================
    # Panel 1: Self-Similarity Analysis
    # =========================================================================
    print("  Panel 1: Self-Similarity Analysis...")
    ax1 = fig.add_subplot(gs[0, 0])

    # Compare capacitance response in different density decades
    decades = [
        (1e8, 1e10, COLORS['blue'], '10⁸–10¹⁰'),
        (1e10, 1e12, COLORS['green'], '10¹⁰–10¹²'),
        (1e12, 1e14, COLORS['orange'], '10¹²–10¹⁴'),
        (1e14, 1e16, COLORS['red'], '10¹⁴–10¹⁶'),
    ]

    for n_lo, n_hi, col, label in decades:
        mask = (n_cm2 >= n_lo) & (n_cm2 <= n_hi)
        if np.sum(mask) > 10:
            x_norm = (n_cm2[mask] - n_lo) / (n_hi - n_lo)
            c_norm = C_Q[mask] / np.max(C_Q[mask])
            ax1.plot(x_norm, c_norm, color=col, linewidth=1.5, alpha=0.8, label=label)

    ax1.set_xlabel("Normalized position within decade", fontsize=10)
    ax1.set_ylabel("Normalized C_Q", fontsize=10)
    ax1.set_title("Self-Similarity: C_Q Across Decades", fontsize=12, fontweight='bold')
    ax1.legend(fontsize=7, loc='upper left', framealpha=0.9, title="Density range (cm⁻²)")
    ax1.grid(True, alpha=0.2)

    ax1.annotate(
        "Different decades show\nstructural similarity in\ncapacitance response",
        xy=(0.97, 0.03), xycoords='axes fraction', va='bottom', ha='right',
        fontsize=7, alpha=0.8,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8)
    )

    # =========================================================================
    # Panel 2: Power-Law Scaling
    # =========================================================================
    print("  Panel 2: Power-Law Scaling...")
    ax2 = fig.add_subplot(gs[0, 1])

    # C_Q ∝ n^α — measure α across regimes
    ax2.loglog(n_cm2, C_Q * 1e4, color=COLORS['blue'], linewidth=2, label='C_Q')
    ax2.loglog(n_cm2, E_F / eV, color=COLORS['green'], linewidth=2, label='E_F (eV)')

    # Fit power laws in Dirac regime (n < 10¹³)
    dirac_mask = (n_cm2 > 1e9) & (n_cm2 < 1e13)
    if np.sum(dirac_mask) > 10:
        log_n = np.log10(n_cm2[dirac_mask])
        log_CQ = np.log10(C_Q[dirac_mask] * 1e4)
        log_EF = np.log10(E_F[dirac_mask] / eV)

        cq_fit = np.polyfit(log_n, log_CQ, 1)
        ef_fit = np.polyfit(log_n, log_EF, 1)

        fit_n = np.logspace(9, 13, 50)
        ax2.loglog(fit_n, 10 ** (cq_fit[0] * np.log10(fit_n) + cq_fit[1]),
                   color=COLORS['blue'], linewidth=1, linestyle=':', alpha=0.6)
        ax2.loglog(fit_n, 10 ** (ef_fit[0] * np.log10(fit_n) + ef_fit[1]),
                   color=COLORS['green'], linewidth=1, linestyle=':', alpha=0.6)

        ax2.annotate(
            f"Dirac regime:\nC_Q ∝ n^{cq_fit[0]:.3f}\nE_F ∝ n^{ef_fit[0]:.3f}",
            xy=(0.03, 0.97), xycoords='axes fraction', va='top',
            fontsize=8, alpha=0.8, color=COLORS['dark'], fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8)
        )

    ax2.set_xlabel("Carrier density n (cm⁻²)", fontsize=10)
    ax2.set_ylabel("Value", fontsize=10)
    ax2.set_title("Power-Law Scaling Analysis", fontsize=12, fontweight='bold')
    ax2.legend(fontsize=8, loc='lower right', framealpha=0.9)
    ax2.grid(True, alpha=0.2, which='both')

    # =========================================================================
    # Panel 3: Edge State Contributions
    # =========================================================================
    print("  Panel 3: Edge State Contributions...")
    ax3 = fig.add_subplot(gs[0, 2])

    # DOS with and without edge states
    E_pos = energies[energies >= 0]
    D_pos = dos[energies >= 0]

    # Zigzag edge corrections at different edge fractions
    edge_fractions = [0.0, 0.005, 0.01, 0.02]
    edge_colors = [COLORS['blue'], COLORS['green'], COLORS['orange'], COLORS['red']]
    edge_labels = ['Bulk (no edges)', '0.5% zigzag', '1% zigzag', '2% zigzag']

    for ef, col, label in zip(edge_fractions, edge_colors, edge_labels):
        if ef == 0:
            D_corrected = D_pos
        else:
            D_edge = zigzag_edge_dos_correction(E_pos, edge_fraction=ef)
            D_corrected = D_pos + D_edge

        C_Q_edge = e_charge ** 2 * D_corrected
        # Map to carrier density (approximate)
        n_edge = cumulative_trapezoid(D_corrected, E_pos, initial=0)
        n_edge_cm2 = n_edge * 1e-4

        valid = n_edge_cm2 > 1e8
        ax3.loglog(n_edge_cm2[valid], C_Q_edge[valid] * 1e4,
                   color=col, linewidth=1.5, alpha=0.8, label=label)

    ax3.set_xlabel("Carrier density n (cm⁻²)", fontsize=10)
    ax3.set_ylabel("C_Q (μF/cm²)", fontsize=10)
    ax3.set_title("Edge State Contributions to C_Q", fontsize=12, fontweight='bold')
    ax3.legend(fontsize=7, loc='upper left', framealpha=0.9)
    ax3.grid(True, alpha=0.2, which='both')

    ax3.annotate(
        "Zigzag edges create\nlow-density enhancement\n→ fractal boundary effect",
        xy=(0.97, 0.03), xycoords='axes fraction', va='bottom', ha='right',
        fontsize=7, alpha=0.8,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8)
    )

    # =========================================================================
    # Panel 4: Stability Analysis (Differential Capacitance)
    # =========================================================================
    print("  Panel 4: Stability Analysis...")
    ax4 = fig.add_subplot(gs[1, 0])

    # Use d(log C_total)/d(log n) for stability — shows regime transitions cleanly
    log_n_stab = np.log10(n_cm2)
    log_C_stab = np.log10(C_total * 1e4)
    d_logC = np.gradient(log_C_stab, log_n_stab)

    # Second derivative of log capacitance — curvature reveals instability
    d2_logC = np.gradient(d_logC, log_n_stab)

    # Smooth
    window = 31
    if len(d2_logC) > window:
        kernel = np.ones(window) / window
        d2_logC_smooth = np.convolve(d2_logC, kernel, mode='same')
    else:
        d2_logC_smooth = d2_logC

    ax4.plot(n_cm2, d_logC, color=COLORS['purple'], linewidth=1.5, alpha=0.8,
             label='d(log C)/d(log n)')
    ax4.plot(n_cm2, d2_logC_smooth, color=COLORS['orange'], linewidth=1.2, alpha=0.7,
             linestyle='--', label='d²(log C)/d(log n)²')
    ax4.set_xlabel("Carrier density n (cm⁻²)", fontsize=10)
    ax4.set_ylabel("Logarithmic derivative", fontsize=10)
    ax4.set_xscale('log')
    ax4.set_title("Charge Storage Stability Landscape", fontsize=12, fontweight='bold')
    ax4.legend(fontsize=7, loc='best', framealpha=0.9)
    ax4.grid(True, alpha=0.2)
    ax4.axhline(y=0, color='gray', linewidth=0.5, alpha=0.5)

    # Mark cascade points
    for cd in cascade_densities:
        ax4.axvline(x=cd, color=COLORS['red'], linestyle=':', alpha=0.3)

    ax4.annotate(
        "Curvature spikes = phase transitions\nin storage mechanism\n→ cascade entry points",
        xy=(0.03, 0.97), xycoords='axes fraction', va='top',
        fontsize=7, alpha=0.8,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8)
    )

    # =========================================================================
    # Panel 5: Harmonic Cascade Map
    # =========================================================================
    print("  Panel 5: Harmonic Cascade Map...")
    ax5 = fig.add_subplot(gs[1, 1])

    U_cm2 = U * 1e-4  # J/cm²

    ax5.loglog(n_cm2, U_cm2, color=COLORS['green'], linewidth=2.5, zorder=5)
    ax5.set_xlabel("Carrier density n (cm⁻²)", fontsize=10)
    ax5.set_ylabel("Energy density U (J/cm²)", fontsize=10)
    ax5.set_title("Harmonic Cascade Map", fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.2, which='both')

    # Mark cascade levels
    cascade_colors_map = [COLORS['blue'], COLORS['green'], COLORS['red'],
                          COLORS['purple'], COLORS['cyan']]
    for i, cd in enumerate(cascade_densities):
        idx = np.argmin(np.abs(n_cm2 - cd))
        col = cascade_colors_map[i % len(cascade_colors_map)]
        ax5.axvline(x=cd, color=col, linestyle='--', alpha=0.5)
        ax5.scatter([cd], [U_cm2[idx]], color=col,
                    s=80, zorder=10, edgecolors='white', linewidths=1.5)
        # Stagger annotations
        y_shift = 0.15 if i % 2 == 0 else 5.0
        label = cascade_labels[i].replace('\n', ' ') if i < len(cascade_labels) else f"C{i+1}"
        ax5.annotate(
            f"C{i+1}: {label}\n{U_cm2[idx]:.2e} J/cm²",
            xy=(cd, U_cm2[idx]),
            xytext=(cd * 3, U_cm2[idx] * y_shift),
            fontsize=6, fontweight='bold', color=col,
            arrowprops=dict(arrowstyle='->', color=col, lw=0.8)
        )

    ax5.annotate(
        "Each cascade = regime transition\nin energy storage mechanism",
        xy=(0.03, 0.03), xycoords='axes fraction', va='bottom',
        fontsize=7, alpha=0.8,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8)
    )

    # =========================================================================
    # Panel 6: Energy Storage Projection
    # =========================================================================
    print("  Panel 6: Energy Storage Projection...")
    ax6 = fig.add_subplot(gs[1, 2])

    # Convert to volumetric energy density for a realistic device
    # Assumptions for cereal box device:
    # - Graphene layers stacked with d_sep = 0.5 nm interlayer spacing
    # - Total thickness of one electrode pair: ~1 nm
    # - Packing factor: 0.7 (accounting for electrolyte, separators, etc.)
    d_pair = 1.0e-9  # m (one electrode pair thickness)
    packing = 0.7

    # Volumetric energy density: U_vol = U_area / d_pair × packing
    U_vol = U * packing / d_pair  # J/m³
    U_vol_MJ_L = U_vol * 1e-6 * 1e-3  # convert J/m³ to MJ/L

    # Gravimetric (per kg): graphene density ~2267 kg/m³, effective ~500 kg/m³ with packing
    rho_eff = 500  # kg/m³ effective density
    U_grav = U_vol / rho_eff  # J/kg
    U_grav_Wh_kg = U_grav / 3600  # convert to Wh/kg

    ax6.loglog(n_cm2, U_vol_MJ_L, color=COLORS['red'], linewidth=2.5, label='Volumetric (MJ/L)')
    ax6.set_xlabel("Carrier density n (cm⁻²)", fontsize=10)
    ax6.set_ylabel("Volumetric energy density (MJ/L)", fontsize=10)
    ax6.set_title("Energy Storage Projection", fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.2, which='both')

    # Reference lines
    refs = [
        (0.001, 'Current supercapacitors (~1 Wh/L)', COLORS['blue']),
        (0.5, 'Li-ion batteries (~500 Wh/L)', COLORS['green']),
        (2.0, 'Gasoline (~34 MJ/L)', COLORS['orange']),
        (200.0, '⚡ Lightning bolt / cereal box (~200 MJ/L)', COLORS['red']),
    ]

    for ref_val, ref_label, ref_col in refs:
        ax6.axhline(y=ref_val, color=ref_col, linestyle=':', alpha=0.5, linewidth=1)
        ax6.annotate(ref_label, xy=(1.5e8, ref_val * 1.3), fontsize=6,
                     color=ref_col, alpha=0.8)

    ax6.legend(fontsize=8, loc='upper left', framealpha=0.9)

    # Mark where cascades reach
    cascade_colors_proj = [COLORS['blue'], COLORS['green'], COLORS['red'],
                           COLORS['purple'], COLORS['cyan']]
    for i, cd in enumerate(cascade_densities):
        idx = np.argmin(np.abs(n_cm2 - cd))
        if idx < len(U_vol_MJ_L):
            col = cascade_colors_proj[i % len(cascade_colors_proj)]
            ax6.scatter([cd], [U_vol_MJ_L[idx]], color=col,
                        s=80, zorder=10, edgecolors='white', linewidths=1.5)

    # Print cascade energy densities
    print("\n  CASCADE ENERGY DENSITIES:")
    for i, cd in enumerate(cascade_densities):
        idx = np.argmin(np.abs(n_cm2 - cd))
        label = cascade_labels[i].replace('\n', ' ') if i < len(cascade_labels) else f"C{i+1}"
        print(f"    Cascade {i+1} ({label}): n = {cd:.2e} cm⁻², "
              f"U_vol = {U_vol_MJ_L[idx]:.4f} MJ/L, "
              f"U_grav = {U_grav_Wh_kg[idx]:.1f} Wh/kg")

    # Final cascade values
    max_U = U_vol_MJ_L[-1]
    max_Wh = U_grav_Wh_kg[-1]
    print(f"\n    MAXIMUM (n = {n_cm2[-1]:.2e} cm⁻²):")
    print(f"    U_vol = {max_U:.4f} MJ/L")
    print(f"    U_grav = {max_Wh:.1f} Wh/kg")

    plt.savefig('fig23_graphene_cascade_prediction.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print("\n  ✓ Saved fig23_graphene_cascade_prediction.png")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("PAPER XVIII: GRAPHENE HARMONIC CASCADE ANALYSIS")
    print("The Lucian Method Applied to Charge Density Geometry")
    print("STATUS: WITHHELD — IP PROTECTED")
    print("=" * 70 + "\n")

    # Step 1: Compute full tight-binding DOS
    print("Computing graphene DOS from tight-binding model...")
    energies, dos = compute_graphene_dos(n_energy=5000, n_k=800)
    print(f"  DOS computed: {len(energies)} energy points")

    # Step 2: Define the driving variable range
    # n from 10⁸ to 10¹⁶ cm⁻² → convert to m⁻²
    n_carrier_full = np.logspace(8, 16, 2000) * 1e4  # cm⁻² → m⁻²

    # Step 3: Compute Fermi energy from full DOS
    print("Computing Fermi energy from DOS integration...")
    E_F_full = fermi_energy_from_density(n_carrier_full, energies, dos)

    # PHYSICAL LIMIT: truncate at band edge (E_F < 2.9 × γ₀)
    # Beyond 3γ₀ the tight-binding model is invalid — DOS goes to zero,
    # higher bands not included. This is the physical boundary.
    E_F_max = 2.8 * gamma_0  # Stay below Van Hove → band-edge collapse
    valid_mask = E_F_full <= E_F_max
    if np.sum(valid_mask) < len(n_carrier_full):
        n_carrier = n_carrier_full[valid_mask]
        E_F = E_F_full[valid_mask]
        print(f"  Band edge cutoff: E_F < {E_F_max/eV:.2f} eV → "
              f"n < {n_carrier[-1]*1e-4:.2e} cm⁻²")
    else:
        n_carrier = n_carrier_full
        E_F = E_F_full

    print(f"  Carrier density range: {n_carrier[0]*1e-4:.0e} to {n_carrier[-1]*1e-4:.0e} cm⁻²")
    print(f"  E_F range: {E_F[0]/eV:.4f} to {E_F[-1]/eV:.2f} eV")

    # Step 4: Compute quantum capacitance
    print("Computing quantum capacitance...")
    C_Q = quantum_capacitance_from_dos(E_F, energies, dos)
    print(f"  C_Q range: {C_Q[0]*1e4:.4f} to {C_Q.max()*1e4:.2f} μF/cm²")

    # Step 5: Compute total capacitance (with geometric + exchange corrections)
    print("Computing total capacitance with exchange-correlation...")
    d_sep = 0.5e-9  # 0.5 nm separation
    eps_r = 3.0
    C_total, C_geo = compute_total_capacitance(n_carrier, C_Q, d_sep, eps_r)
    print(f"  C_geo = {C_geo*1e4:.2f} μF/cm²")
    print(f"  C_total range: {C_total[0]*1e4:.4f} to {C_total.max()*1e4:.2f} μF/cm²")

    # Step 6: Compute energy density
    print("Computing energy density landscape...")
    U = compute_energy_density(n_carrier, C_total)
    print(f"  U range: {U[0]:.2e} to {U[-1]:.2e} J/m²")

    # Generate figures
    cascade_densities, cascade_labels = generate_figure_1(
        energies, dos, n_carrier, E_F, C_Q, C_total, C_geo, U)

    generate_figure_2(energies, dos, n_carrier, E_F, C_Q, C_total, U,
                       cascade_densities, cascade_labels)

    print("\n" + "=" * 70)
    print("COMPLETE — Paper XVIII analysis generated")
    print("  fig22_graphene_harmonic_analysis.png")
    print("  fig23_graphene_cascade_prediction.png")
    print("")
    print("  ⚠️  WITHHELD — DO NOT PUBLISH — IP PROTECTED")
    print("=" * 70 + "\n")
