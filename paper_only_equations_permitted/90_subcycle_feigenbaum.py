"""
+============================================================================+
|  (c) 2026 Lucian Randolph. All rights reserved.                           |
|                                                                            |
|  Script 90 -- The Snap: Subcycle Feigenbaum Extraction                     |
+============================================================================+

Script 90 -- The Snap: Subcycle Feigenbaum Extraction

    The "whip crack" hypothesis: the Feigenbaum cascade is not truncated
    at merger -- it's COMPRESSED into the merger instant. Just as the
    tip of a whip exceeds the speed of sound by concentrating the entire
    chain's momentum into a tiny mass, the cascade concentrates infinite
    period-doubling into a finite-time singularity.

    This script extracts omega(t), omega_dot(t), omega_ddot(t) at FULL
    computational resolution from the SXS NR waveforms and searches for
    period-doubling structure in the subcycle intervals at merger.

    Key question: Do the subcycle interval ratios approach delta = 4.669?

Generates:
    fig90a_frequency_evolution.png   (omega, omega_dot, omega_ddot)
    fig90b_subcycle_intervals.png    (interval ratios from omega_ddot)
    fig90c_universality.png          (cross-waveform comparison)
    fig90d_whip_crack_zoom.png       (maximum resolution merger zoom)
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import h5py
from typing import Optional, Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(SCRIPT_DIR, "sxs_data")

# ==========================================================================
#  FUNDAMENTAL CONSTANTS
# ==========================================================================
DELTA_FEIG = 4.669201609102990
ALPHA_FEIG = 2.502907875095892
LAMBDA_R   = DELTA_FEIG / ALPHA_FEIG   # 1.86551077... The Randolph Constant
LN_DELTA   = np.log(DELTA_FEIG)

# Style constants
COLORS = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#e67e22',
          '#1abc9c', '#f39c12', '#c0392b', '#2980b9', '#27ae60']


# ==========================================================================
#  SXS WAVEFORM METADATA (from catalog.json, verified in Script 89)
# ==========================================================================
WAVEFORMS = [
    {"id": "0001", "q": 1.00,  "chi_f": 0.686462, "M_f": 0.951609,
     "desc": "q=1, non-spinning"},
    {"id": "0002", "q": 1.00,  "chi_f": 0.686448, "M_f": 0.951610,
     "desc": "q=1, low spin"},
    {"id": "0007", "q": 1.50,  "chi_f": 0.664091, "M_f": 0.955270,
     "desc": "q=1.5, non-spinning"},
    {"id": "0056", "q": 5.00,  "chi_f": 0.416614, "M_f": 0.982368,
     "desc": "q=5, non-spinning"},
    {"id": "0063", "q": 8.00,  "chi_f": 0.306744, "M_f": 0.989425,
     "desc": "q=8, non-spinning"},
    {"id": "0167", "q": 4.00,  "chi_f": 0.471553, "M_f": 0.977919,
     "desc": "q=4, non-spinning"},
    {"id": "0180", "q": 1.00,  "chi_f": 0.686430, "M_f": 0.951615,
     "desc": "q=1, non-spinning (Lev2)"},
    {"id": "0150", "q": 1.00,  "chi_f": 0.746431, "M_f": 0.945471,
     "desc": "q=1, aligned spin chi=0.2"},
    {"id": "0004", "q": 1.00,  "chi_f": 0.608209, "M_f": 0.957721,
     "desc": "q=1, chi1z=-0.5"},
    {"id": "1355", "q": 1.00,  "chi_f": 0.686054, "M_f": 0.951500,
     "desc": "q=1, precessing"},
]


# ==========================================================================
#  STEP 1: EXTRACT FULL-RESOLUTION FREQUENCY DATA
# ==========================================================================
def extract_full_resolution(filepath: str, wf_info: Dict,
                            window_M: float = 150.0) -> Optional[Dict]:
    """
    Extract omega(t), omega_dot(t), omega_ddot(t) at the FULL computational
    resolution of the NR simulation, focused on the merger window.

    Args:
        filepath: path to SXS HDF5 file
        wf_info: metadata dict
        window_M: half-width of merger window in units of M

    Returns:
        dict with full-resolution frequency data, or None on failure
    """
    try:
        with h5py.File(filepath, 'r') as f:
            for gname in ['Extrapolated_N2.dir', 'Extrapolated_N3.dir',
                          'Extrapolated_N4.dir']:
                if gname in f and 'Y_l2_m2.dat' in f[gname]:
                    data = f[gname]['Y_l2_m2.dat'][()]
                    t_full = data[:, 0]
                    h22_full = data[:, 1] + 1j * data[:, 2]
                    break
            else:
                print(f"    Cannot find (2,2) mode in {filepath}")
                return None
    except Exception as e:
        print(f"    Error reading {filepath}: {e}")
        return None

    # Amplitude and phase
    amp_full = np.abs(h22_full)
    phase_full = np.unwrap(np.angle(h22_full))

    # Find merger (peak amplitude)
    i_peak = np.argmax(amp_full)
    t_merger = t_full[i_peak]

    # Merger window
    mask = (t_full >= t_merger - window_M) & (t_full <= t_merger + window_M)
    t = t_full[mask]
    h22 = h22_full[mask]
    amp = amp_full[mask]
    phase = phase_full[mask]

    if len(t) < 100:
        print(f"    Too few points in merger window ({len(t)})")
        return None

    # Timestep diagnostics
    dt = np.diff(t)
    dt_min = dt.min()
    dt_max = dt.max()
    dt_median = np.median(dt)

    print(f"    Points in window: {len(t)}")
    print(f"    Timestep: min={dt_min:.6f} M, median={dt_median:.4f} M, max={dt_max:.4f} M")

    # =====================================================================
    #  FREQUENCY EXTRACTION AT FULL RESOLUTION
    # =====================================================================
    # omega = |d(phase)/dt| — use np.gradient for non-uniform spacing
    omega = np.abs(np.gradient(phase, t))

    # Smooth omega very lightly — Savitzky-Golay equivalent with small window
    # Only smooth if we have enough points
    if len(omega) > 21:
        # Rolling average with window = 11 points (much less than a cycle)
        kernel_size = min(11, len(omega) // 10)
        if kernel_size % 2 == 0:
            kernel_size += 1
        if kernel_size >= 3:
            kernel = np.ones(kernel_size) / kernel_size
            omega_smooth = np.convolve(omega, kernel, mode='same')
            # Fix edges
            half = kernel_size // 2
            omega_smooth[:half] = omega[:half]
            omega_smooth[-half:] = omega[-half:]
        else:
            omega_smooth = omega.copy()
    else:
        omega_smooth = omega.copy()

    # omega_dot = d(omega)/dt
    omega_dot = np.gradient(omega_smooth, t)

    # Smooth omega_dot lightly
    if len(omega_dot) > 31:
        k2 = min(21, len(omega_dot) // 8)
        if k2 % 2 == 0:
            k2 += 1
        if k2 >= 3:
            kernel2 = np.ones(k2) / k2
            omega_dot_smooth = np.convolve(omega_dot, kernel2, mode='same')
            half2 = k2 // 2
            omega_dot_smooth[:half2] = omega_dot[:half2]
            omega_dot_smooth[-half2:] = omega_dot[-half2:]
        else:
            omega_dot_smooth = omega_dot.copy()
    else:
        omega_dot_smooth = omega_dot.copy()

    # omega_ddot = d(omega_dot)/dt
    omega_ddot = np.gradient(omega_dot_smooth, t)

    # Smooth omega_ddot more aggressively (third derivative is noisy)
    if len(omega_ddot) > 51:
        k3 = min(41, len(omega_ddot) // 5)
        if k3 % 2 == 0:
            k3 += 1
        if k3 >= 3:
            kernel3 = np.ones(k3) / k3
            omega_ddot_smooth = np.convolve(omega_ddot, kernel3, mode='same')
            half3 = k3 // 2
            omega_ddot_smooth[:half3] = omega_ddot[:half3]
            omega_ddot_smooth[-half3:] = omega_ddot[-half3:]
        else:
            omega_ddot_smooth = omega_ddot.copy()
    else:
        omega_ddot_smooth = omega_ddot.copy()

    # =====================================================================
    #  SUBCYCLE INTERVAL EXTRACTION FROM omega_ddot ZERO CROSSINGS
    # =====================================================================
    # Zero crossings of omega_ddot define "subcycle" boundaries
    # where the frequency acceleration changes sign
    crossings_ddot = []
    for i in range(len(omega_ddot_smooth) - 1):
        if omega_ddot_smooth[i] * omega_ddot_smooth[i+1] < 0:
            # Linear interpolation for crossing time
            frac = -omega_ddot_smooth[i] / (omega_ddot_smooth[i+1] - omega_ddot_smooth[i])
            t_cross = t[i] + frac * (t[i+1] - t[i])
            crossings_ddot.append(t_cross)

    crossings_ddot = np.array(crossings_ddot)

    # Subcycle intervals
    if len(crossings_ddot) > 1:
        subcycle_intervals = np.diff(crossings_ddot)
    else:
        subcycle_intervals = np.array([])

    # Interval ratios: R_k = dt_k / dt_{k+1}
    if len(subcycle_intervals) > 1:
        interval_ratios = subcycle_intervals[:-1] / subcycle_intervals[1:]
    else:
        interval_ratios = np.array([])

    # =====================================================================
    #  HALF-PERIOD ANALYSIS (zero crossings of Re(h22))
    # =====================================================================
    h_real = np.real(h22)
    crossings_h = []
    for i in range(len(h_real) - 1):
        if h_real[i] * h_real[i+1] < 0:
            frac = -h_real[i] / (h_real[i+1] - h_real[i])
            t_cross = t[i] + frac * (t[i+1] - t[i])
            crossings_h.append(t_cross)
    crossings_h = np.array(crossings_h)

    # Half-periods
    if len(crossings_h) > 1:
        half_periods = np.diff(crossings_h)
    else:
        half_periods = np.array([])

    # Half-period ratios (consecutive)
    if len(half_periods) > 1:
        hp_ratios = half_periods[:-1] / half_periods[1:]
    else:
        hp_ratios = np.array([])

    # =====================================================================
    #  OMEGA_DOT ZERO CROSSINGS (where chirp rate changes sign)
    # =====================================================================
    crossings_odot = []
    for i in range(len(omega_dot_smooth) - 1):
        if omega_dot_smooth[i] * omega_dot_smooth[i+1] < 0:
            frac = -omega_dot_smooth[i] / (omega_dot_smooth[i+1] - omega_dot_smooth[i])
            t_cross = t[i] + frac * (t[i+1] - t[i])
            crossings_odot.append(t_cross)
    crossings_odot = np.array(crossings_odot)

    # Find the merger crossing index in h_real crossings
    merger_h_idx = -1
    if len(crossings_h) > 0:
        merger_h_idx = np.argmin(np.abs(crossings_h - t_merger))

    return {
        'id': wf_info['id'],
        'desc': wf_info['desc'],
        'q': wf_info['q'],
        'chi_f': wf_info['chi_f'],
        'M_f': wf_info['M_f'],
        't_merger': t_merger,
        # Full arrays
        't': t,
        'amp': amp,
        'h_real': h_real,
        'omega': omega_smooth,
        'omega_dot': omega_dot_smooth,
        'omega_ddot': omega_ddot_smooth,
        # Subcycle intervals from omega_ddot crossings
        'crossings_ddot': crossings_ddot,
        'subcycle_intervals': subcycle_intervals,
        'interval_ratios': interval_ratios,
        # Half-period analysis
        'crossings_h': crossings_h,
        'half_periods': half_periods,
        'hp_ratios': hp_ratios,
        'merger_h_idx': merger_h_idx,
        # omega_dot crossings
        'crossings_odot': crossings_odot,
        # Resolution diagnostics
        'dt_min': dt_min,
        'dt_median': dt_median,
        'dt_max': dt_max,
        'n_points': len(t),
    }


# ==========================================================================
#  STEP 2: NEAR-MERGER INTERVAL RATIO TRAJECTORIES
# ==========================================================================
def analyze_merger_trajectory(result: Dict) -> Dict:
    """
    Extract the interval ratio trajectory centered on the merger.
    Returns ratios indexed relative to merger crossing.
    """
    t_merger = result['t_merger']

    # --- Half-period ratio trajectory ---
    crossings_h = result['crossings_h']
    hp = result['half_periods']
    hp_r = result['hp_ratios']

    # Index closest to merger
    if len(crossings_h) > 0:
        i_m = np.argmin(np.abs(crossings_h - t_merger))
    else:
        i_m = -1

    # Build trajectory: ratio index relative to merger
    # hp_ratios[k] uses half_periods[k] and [k+1], which are between
    # crossings[k] to [k+1] and [k+1] to [k+2]
    hp_traj_idx = []
    hp_traj_val = []
    hp_traj_t = []
    if i_m >= 0 and len(hp_r) > 0:
        for k in range(len(hp_r)):
            rel_idx = k - (i_m - 1)  # center on merger
            if np.isfinite(hp_r[k]) and 0 < hp_r[k] < 50:
                hp_traj_idx.append(rel_idx)
                hp_traj_val.append(hp_r[k])
                hp_traj_t.append(crossings_h[k+1] - t_merger)

    # --- Subcycle interval ratio trajectory ---
    crossings_ddot = result['crossings_ddot']
    sc_int = result['subcycle_intervals']
    sc_r = result['interval_ratios']

    if len(crossings_ddot) > 0:
        i_m_sc = np.argmin(np.abs(crossings_ddot - t_merger))
    else:
        i_m_sc = -1

    sc_traj_idx = []
    sc_traj_val = []
    sc_traj_t = []
    if i_m_sc >= 0 and len(sc_r) > 0:
        for k in range(len(sc_r)):
            rel_idx = k - (i_m_sc - 1)
            if np.isfinite(sc_r[k]) and 0 < sc_r[k] < 50:
                sc_traj_idx.append(rel_idx)
                sc_traj_val.append(sc_r[k])
                sc_traj_t.append(crossings_ddot[k+1] - t_merger)

    result['hp_traj_idx'] = np.array(hp_traj_idx)
    result['hp_traj_val'] = np.array(hp_traj_val)
    result['hp_traj_t'] = np.array(hp_traj_t)
    result['sc_traj_idx'] = np.array(sc_traj_idx)
    result['sc_traj_val'] = np.array(sc_traj_val)
    result['sc_traj_t'] = np.array(sc_traj_t)

    return result


# ==========================================================================
#  FIGURE 90a: FREQUENCY EVOLUTION AT FULL RESOLUTION
# ==========================================================================
def plot_frequency_evolution(results: List[Dict]) -> None:
    """
    Three-panel figure showing omega, omega_dot, omega_ddot for a
    representative waveform (BBH:0001), with overlay of another (BBH:0007).
    """
    fig = plt.figure(figsize=(18, 14))
    gs = gridspec.GridSpec(3, 2, hspace=0.35, wspace=0.30)
    fig.suptitle('Script 90 -- The Snap: Full-Resolution Frequency Structure at Merger',
                 fontsize=15, fontweight='bold', color='#1A2A44', y=0.98)

    # Pick representative waveforms
    r0 = next((r for r in results if r['id'] == '0001'), results[0])
    r1 = next((r for r in results if r['id'] == '0007'), results[1] if len(results) > 1 else results[0])

    for col, (res, label) in enumerate([(r0, 'BBH:0001 (q=1)'),
                                         (r1, f"BBH:{r1['id']} (q={r1['q']:.1f})")]):
        t_rel = res['t'] - res['t_merger']

        # Panel 1: omega(t)
        ax1 = fig.add_subplot(gs[0, col])
        ax1.plot(t_rel, res['omega'], color=COLORS[col], lw=0.8, alpha=0.9)
        ax1.axvline(0, color='gray', ls='--', lw=1, alpha=0.5, label='Merger')

        # Mark omega_dot zero crossings
        for tc in res['crossings_odot']:
            ax1.axvline(tc - res['t_merger'], color='orange', ls=':', lw=0.5, alpha=0.4)

        ax1.set_ylabel(r'$\omega(t)$ [rad/M]', fontsize=10)
        ax1.set_title(f'A{col+1}. Instantaneous Frequency -- {label}',
                      fontsize=12, fontweight='bold', color='#1A2A44')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.2)
        ax1.set_xlim(-100, 80)

        # Panel 2: omega_dot(t)
        ax2 = fig.add_subplot(gs[1, col])
        ax2.plot(t_rel, res['omega_dot'], color=COLORS[2 + col], lw=0.8, alpha=0.9)
        ax2.axvline(0, color='gray', ls='--', lw=1, alpha=0.5)
        ax2.axhline(0, color='black', ls='-', lw=0.5, alpha=0.3)

        # Mark omega_dot crossings
        for tc in res['crossings_odot']:
            ax2.axvline(tc - res['t_merger'], color='orange', ls='-', lw=1, alpha=0.5)

        ax2.set_ylabel(r'$\dot{\omega}(t)$ [rad/M$^2$]', fontsize=10)
        ax2.set_title(f'B{col+1}. Chirp Rate (Frequency Derivative)',
                      fontsize=12, fontweight='bold', color='#1A2A44')
        ax2.grid(True, alpha=0.2)
        ax2.set_xlim(-100, 80)

        # Panel 3: omega_ddot(t) with subcycle boundaries
        ax3 = fig.add_subplot(gs[2, col])
        ax3.plot(t_rel, res['omega_ddot'], color=COLORS[4 + col], lw=0.8, alpha=0.9)
        ax3.axvline(0, color='gray', ls='--', lw=1, alpha=0.5)
        ax3.axhline(0, color='black', ls='-', lw=0.5, alpha=0.3)

        # Mark omega_ddot zero crossings (subcycle boundaries)
        for tc in res['crossings_ddot']:
            ax3.axvline(tc - res['t_merger'], color='red', ls='-', lw=0.7, alpha=0.4)

        n_sc = len(res['crossings_ddot'])
        ax3.set_xlabel(f'$t - t_{{merger}}$ [M]', fontsize=10)
        ax3.set_ylabel(r'$\ddot{\omega}(t)$ [rad/M$^3$]', fontsize=10)
        ax3.set_title(f'C{col+1}. Chirp Acceleration ({n_sc} subcycle boundaries)',
                      fontsize=12, fontweight='bold', color='#1A2A44')
        ax3.grid(True, alpha=0.2)
        ax3.set_xlim(-100, 80)

    out = os.path.join(SCRIPT_DIR, "fig90a_frequency_evolution.png")
    fig.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"\n  Saved: {out}  ({os.path.getsize(out):,} bytes)")


# ==========================================================================
#  FIGURE 90b: SUBCYCLE INTERVALS AND RATIOS
# ==========================================================================
def plot_subcycle_intervals(results: List[Dict]) -> None:
    """
    Six-panel figure: subcycle intervals, their ratios, and half-period
    ratios for representative waveforms.
    """
    fig = plt.figure(figsize=(18, 14))
    gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.30)
    fig.suptitle('Script 90 -- Subcycle Interval Analysis: Searching for Period-Doubling',
                 fontsize=15, fontweight='bold', color='#1A2A44', y=0.98)

    # Panel 1: Subcycle intervals for BBH:0001
    r0 = next((r for r in results if r['id'] == '0001'), results[0])
    ax1 = fig.add_subplot(gs[0, 0])
    if len(r0['subcycle_intervals']) > 0:
        t_sc = r0['crossings_ddot'][:-1] - r0['t_merger']
        ax1.semilogy(t_sc, r0['subcycle_intervals'], 'o-', color=COLORS[0],
                     markersize=3, lw=1, alpha=0.8)
    ax1.axvline(0, color='gray', ls='--', lw=1, alpha=0.5)
    ax1.set_xlabel('$t - t_{merger}$ [M]', fontsize=10)
    ax1.set_ylabel('Subcycle interval $\\Delta t_k$ [M]', fontsize=10)
    ax1.set_title(f'A. Subcycle Intervals -- BBH:0001',
                  fontsize=12, fontweight='bold', color='#1A2A44')
    ax1.grid(True, alpha=0.2)

    # Panel 2: Subcycle interval RATIOS for BBH:0001
    ax2 = fig.add_subplot(gs[0, 1])
    if len(r0['interval_ratios']) > 0:
        t_ir = r0['crossings_ddot'][1:-1] - r0['t_merger']
        mask_valid = (r0['interval_ratios'] > 0) & (r0['interval_ratios'] < 20)
        ax2.plot(t_ir[mask_valid], r0['interval_ratios'][mask_valid],
                 'o-', color=COLORS[0], markersize=3, lw=1, alpha=0.8)
    ax2.axhline(DELTA_FEIG, color='red', ls='--', lw=2, alpha=0.7,
                label=f'$\\delta$ = {DELTA_FEIG:.3f}')
    ax2.axhline(1.0, color='gray', ls=':', lw=1, alpha=0.5, label='R = 1')
    ax2.axvline(0, color='gray', ls='--', lw=1, alpha=0.5)
    ax2.set_xlabel('$t - t_{merger}$ [M]', fontsize=10)
    ax2.set_ylabel('Interval ratio $R_k = \\Delta t_k / \\Delta t_{k+1}$', fontsize=10)
    ax2.set_title(f'B. Subcycle Interval Ratios -- BBH:0001',
                  fontsize=12, fontweight='bold', color='#1A2A44')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.2)
    ax2.set_ylim(0, 12)

    # Panel 3: Half-period ratios for BBH:0001
    ax3 = fig.add_subplot(gs[0, 2])
    if len(r0['hp_traj_val']) > 0:
        ax3.plot(r0['hp_traj_t'], r0['hp_traj_val'],
                 'o-', color=COLORS[0], markersize=4, lw=1.5, alpha=0.8)
    ax3.axhline(DELTA_FEIG, color='red', ls='--', lw=2, alpha=0.7,
                label=f'$\\delta$ = {DELTA_FEIG:.3f}')
    ax3.axhline(1.0, color='gray', ls=':', lw=1, alpha=0.5)
    ax3.axvline(0, color='gray', ls='--', lw=1, alpha=0.5)
    ax3.set_xlabel('$t - t_{merger}$ [M]', fontsize=10)
    ax3.set_ylabel('Half-period ratio $T_k / T_{k+1}$', fontsize=10)
    ax3.set_title('C. Half-Period Ratios -- BBH:0001',
                  fontsize=12, fontweight='bold', color='#1A2A44')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.2)
    ax3.set_ylim(0, 12)

    # Panel 4: All waveforms subcycle ratios near merger (±30 M)
    ax4 = fig.add_subplot(gs[1, 0])
    for i, res in enumerate(results):
        if len(res['interval_ratios']) > 0:
            t_ir = res['crossings_ddot'][1:-1] - res['t_merger']
            mask_near = (np.abs(t_ir) < 30) & (res['interval_ratios'] > 0) & \
                        (res['interval_ratios'] < 15)
            if np.any(mask_near):
                ax4.plot(t_ir[mask_near], res['interval_ratios'][mask_near],
                         'o-', color=COLORS[i % len(COLORS)], markersize=3,
                         lw=1, alpha=0.7, label=f"BBH:{res['id']}")
    ax4.axhline(DELTA_FEIG, color='red', ls='--', lw=2, alpha=0.7)
    ax4.axvline(0, color='gray', ls='--', lw=1, alpha=0.5)
    ax4.set_xlabel('$t - t_{merger}$ [M]', fontsize=10)
    ax4.set_ylabel('Subcycle interval ratio', fontsize=10)
    ax4.set_title('D. Near-Merger Subcycle Ratios (all waveforms)',
                  fontsize=12, fontweight='bold', color='#1A2A44')
    ax4.legend(fontsize=6, ncol=2, loc='upper right')
    ax4.grid(True, alpha=0.2)
    ax4.set_ylim(0, 12)

    # Panel 5: All waveforms half-period ratios near merger
    ax5 = fig.add_subplot(gs[1, 1])
    for i, res in enumerate(results):
        if len(res['hp_traj_val']) > 0:
            mask_near = (np.abs(res['hp_traj_t']) < 30) & \
                        (res['hp_traj_val'] > 0) & (res['hp_traj_val'] < 15)
            if np.any(mask_near):
                ax5.plot(res['hp_traj_t'][mask_near], res['hp_traj_val'][mask_near],
                         'o-', color=COLORS[i % len(COLORS)], markersize=3,
                         lw=1, alpha=0.7, label=f"BBH:{res['id']}")
    ax5.axhline(DELTA_FEIG, color='red', ls='--', lw=2, alpha=0.7)
    ax5.axvline(0, color='gray', ls='--', lw=1, alpha=0.5)
    ax5.set_xlabel('$t - t_{merger}$ [M]', fontsize=10)
    ax5.set_ylabel('Half-period ratio', fontsize=10)
    ax5.set_title('E. Near-Merger Half-Period Ratios (all waveforms)',
                  fontsize=12, fontweight='bold', color='#1A2A44')
    ax5.legend(fontsize=6, ncol=2, loc='upper right')
    ax5.grid(True, alpha=0.2)
    ax5.set_ylim(0, 12)

    # Panel 6: Histogram of near-merger ratios
    ax6 = fig.add_subplot(gs[1, 2])
    all_near_sc = []
    all_near_hp = []
    for res in results:
        if len(res['interval_ratios']) > 0:
            t_ir = res['crossings_ddot'][1:-1] - res['t_merger']
            mask = (np.abs(t_ir) < 20) & (res['interval_ratios'] > 0.5) & \
                   (res['interval_ratios'] < 15)
            all_near_sc.extend(res['interval_ratios'][mask].tolist())
        if len(res['hp_traj_val']) > 0:
            mask = (np.abs(res['hp_traj_t']) < 20) & \
                   (res['hp_traj_val'] > 0.5) & (res['hp_traj_val'] < 15)
            all_near_hp.extend(res['hp_traj_val'][mask].tolist())

    if all_near_sc:
        ax6.hist(all_near_sc, bins=30, alpha=0.5, color=COLORS[0],
                 label=f'Subcycle (N={len(all_near_sc)})', density=True)
    if all_near_hp:
        ax6.hist(all_near_hp, bins=30, alpha=0.5, color=COLORS[1],
                 label=f'Half-period (N={len(all_near_hp)})', density=True)
    ax6.axvline(DELTA_FEIG, color='red', ls='--', lw=2, alpha=0.8,
                label=f'$\\delta$ = {DELTA_FEIG:.3f}')
    ax6.axvline(ALPHA_FEIG, color='purple', ls='--', lw=2, alpha=0.8,
                label=f'$\\alpha$ = {ALPHA_FEIG:.3f}')
    ax6.axvline(1.0, color='gray', ls=':', lw=1, alpha=0.5)
    ax6.set_xlabel('Ratio value', fontsize=10)
    ax6.set_ylabel('Density', fontsize=10)
    ax6.set_title('F. Distribution of Near-Merger Ratios',
                  fontsize=12, fontweight='bold', color='#1A2A44')
    ax6.legend(fontsize=7)
    ax6.grid(True, alpha=0.2)

    out = os.path.join(SCRIPT_DIR, "fig90b_subcycle_intervals.png")
    fig.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"\n  Saved: {out}  ({os.path.getsize(out):,} bytes)")


# ==========================================================================
#  FIGURE 90c: CROSS-WAVEFORM UNIVERSALITY
# ==========================================================================
def plot_universality(results: List[Dict]) -> None:
    """
    Six-panel figure checking whether the subcycle structure is universal
    across mass ratios and spins.
    """
    fig = plt.figure(figsize=(18, 14))
    gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.30)
    fig.suptitle('Script 90 -- Cross-Waveform Universality: Is the Snap Universal?',
                 fontsize=15, fontweight='bold', color='#1A2A44', y=0.98)

    # Panel 1: omega(t)/omega_max overlay (normalized frequency)
    ax1 = fig.add_subplot(gs[0, 0])
    for i, res in enumerate(results):
        t_rel = res['t'] - res['t_merger']
        omega_norm = res['omega'] / np.max(res['omega'])
        mask = (t_rel > -60) & (t_rel < 40)
        ax1.plot(t_rel[mask], omega_norm[mask], color=COLORS[i % len(COLORS)],
                 lw=0.8, alpha=0.7, label=f"BBH:{res['id']}")
    ax1.axvline(0, color='gray', ls='--', lw=1, alpha=0.5)
    ax1.set_xlabel('$t - t_{merger}$ [M]', fontsize=10)
    ax1.set_ylabel(r'$\omega / \omega_{max}$', fontsize=10)
    ax1.set_title('A. Normalized Frequency Overlay',
                  fontsize=12, fontweight='bold', color='#1A2A44')
    ax1.legend(fontsize=6, ncol=2)
    ax1.grid(True, alpha=0.2)

    # Panel 2: omega_dot / max(omega_dot) overlay
    ax2 = fig.add_subplot(gs[0, 1])
    for i, res in enumerate(results):
        t_rel = res['t'] - res['t_merger']
        odot_max = np.max(np.abs(res['omega_dot']))
        if odot_max > 0:
            odot_norm = res['omega_dot'] / odot_max
            mask = (t_rel > -60) & (t_rel < 40)
            ax2.plot(t_rel[mask], odot_norm[mask], color=COLORS[i % len(COLORS)],
                     lw=0.8, alpha=0.7)
    ax2.axvline(0, color='gray', ls='--', lw=1, alpha=0.5)
    ax2.axhline(0, color='black', ls='-', lw=0.5, alpha=0.3)
    ax2.set_xlabel('$t - t_{merger}$ [M]', fontsize=10)
    ax2.set_ylabel(r'$\dot{\omega} / |\dot{\omega}|_{max}$', fontsize=10)
    ax2.set_title('B. Normalized Chirp Rate Overlay',
                  fontsize=12, fontweight='bold', color='#1A2A44')
    ax2.grid(True, alpha=0.2)

    # Panel 3: Number of subcycle intervals vs mass ratio
    ax3 = fig.add_subplot(gs[0, 2])
    q_vals = [r['q'] for r in results]
    n_sc = [len(r['subcycle_intervals']) for r in results]
    n_hp = [len(r['half_periods']) for r in results]
    for i, res in enumerate(results):
        ax3.plot(res['q'], len(res['subcycle_intervals']), 'o',
                 color=COLORS[i % len(COLORS)], markersize=10, zorder=5)
        ax3.annotate(f"BBH:{res['id']}", (res['q'], len(res['subcycle_intervals'])),
                     textcoords="offset points", xytext=(5, 5), fontsize=7,
                     color=COLORS[i % len(COLORS)])
    ax3.set_xlabel('Mass ratio q', fontsize=10)
    ax3.set_ylabel('Number of subcycle intervals', fontsize=10)
    ax3.set_title('C. Subcycle Count vs Mass Ratio',
                  fontsize=12, fontweight='bold', color='#1A2A44')
    ax3.grid(True, alpha=0.2)

    # Panel 4: Near-merger ratio convergence rate
    ax4 = fig.add_subplot(gs[1, 0])
    for i, res in enumerate(results):
        if len(res['hp_traj_val']) > 3:
            # Take the last few pre-merger ratios
            mask_pre = (res['hp_traj_t'] < 0) & (res['hp_traj_t'] > -50) & \
                       (res['hp_traj_val'] > 0) & (res['hp_traj_val'] < 15)
            if np.sum(mask_pre) >= 2:
                t_pre = res['hp_traj_t'][mask_pre]
                v_pre = res['hp_traj_val'][mask_pre]
                ax4.plot(t_pre, v_pre, 'o-', color=COLORS[i % len(COLORS)],
                         markersize=4, lw=1, alpha=0.7,
                         label=f"BBH:{res['id']}")
    ax4.axhline(DELTA_FEIG, color='red', ls='--', lw=2, alpha=0.7,
                label=f'$\\delta$')
    ax4.axvline(0, color='gray', ls='--', lw=1, alpha=0.5)
    ax4.set_xlabel('$t - t_{merger}$ [M]', fontsize=10)
    ax4.set_ylabel('Half-period ratio', fontsize=10)
    ax4.set_title('D. Pre-Merger Ratio Approach to $\\delta$',
                  fontsize=12, fontweight='bold', color='#1A2A44')
    ax4.legend(fontsize=6, ncol=2)
    ax4.grid(True, alpha=0.2)
    ax4.set_ylim(0, 10)

    # Panel 5: Consecutive subcycle ratio differences (looking for convergence)
    ax5 = fig.add_subplot(gs[1, 1])
    for i, res in enumerate(results):
        ir = res['interval_ratios']
        if len(ir) > 2:
            valid = (ir > 0.1) & (ir < 20) & np.isfinite(ir)
            ir_v = ir[valid]
            if len(ir_v) > 2:
                # How do successive ratios change?
                r_diff = np.abs(np.diff(ir_v))
                if len(r_diff) > 1:
                    # Ratio of successive differences
                    rr = r_diff[:-1] / r_diff[1:]
                    rr_valid = rr[(rr > 0.1) & (rr < 30)]
                    if len(rr_valid) > 0:
                        ax5.plot(range(len(rr_valid)), rr_valid,
                                 'o-', color=COLORS[i % len(COLORS)],
                                 markersize=3, lw=1, alpha=0.6)
    ax5.axhline(DELTA_FEIG, color='red', ls='--', lw=2, alpha=0.7,
                label=f'$\\delta$ = {DELTA_FEIG:.3f}')
    ax5.set_xlabel('Successive ratio index', fontsize=10)
    ax5.set_ylabel('Ratio of successive differences', fontsize=10)
    ax5.set_title('E. Convergence Rate of Subcycle Ratios',
                  fontsize=12, fontweight='bold', color='#1A2A44')
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.2)
    ax5.set_ylim(0, 15)

    # Panel 6: Summary statistics
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')

    summary_text = "SUBCYCLE ANALYSIS SUMMARY\n" + "=" * 40 + "\n\n"
    for res in results:
        n_sc = len(res['subcycle_intervals'])
        n_hp = len(res['half_periods'])

        # Near-merger half-period ratios
        near_hp = []
        if len(res['hp_traj_val']) > 0:
            mask = (np.abs(res['hp_traj_t']) < 15) & \
                   (res['hp_traj_val'] > 0.5) & (res['hp_traj_val'] < 15)
            near_hp = res['hp_traj_val'][mask]

        mean_hp = np.mean(near_hp) if len(near_hp) > 0 else float('nan')
        max_hp = np.max(near_hp) if len(near_hp) > 0 else float('nan')

        summary_text += (f"BBH:{res['id']} (q={res['q']:.0f}): "
                        f"{n_sc} subcycles, {n_hp} half-periods, "
                        f"R_max={max_hp:.2f}\n")

    # Overall statistics
    all_near = []
    for res in results:
        if len(res['hp_traj_val']) > 0:
            mask = (np.abs(res['hp_traj_t']) < 10) & \
                   (res['hp_traj_val'] > 0.5) & (res['hp_traj_val'] < 15)
            all_near.extend(res['hp_traj_val'][mask].tolist())

    if all_near:
        summary_text += f"\nAll near-merger HP ratios (|t|<10 M):\n"
        summary_text += f"  N = {len(all_near)}\n"
        summary_text += f"  Mean = {np.mean(all_near):.3f}\n"
        summary_text += f"  Median = {np.median(all_near):.3f}\n"
        summary_text += f"  Max = {np.max(all_near):.3f}\n"
        summary_text += f"  delta = {DELTA_FEIG:.3f}\n"
        # How many are within 20% of delta?
        near_delta = sum(1 for v in all_near
                        if abs(v - DELTA_FEIG) / DELTA_FEIG < 0.2)
        summary_text += f"  Within 20% of delta: {near_delta}/{len(all_near)}\n"

    ax6.text(0.02, 0.98, summary_text, transform=ax6.transAxes,
             fontsize=7, va='top', ha='left', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    out = os.path.join(SCRIPT_DIR, "fig90c_universality.png")
    fig.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"\n  Saved: {out}  ({os.path.getsize(out):,} bytes)")


# ==========================================================================
#  FIGURE 90d: THE WHIP CRACK ZOOM
# ==========================================================================
def plot_whip_crack_zoom(results: List[Dict]) -> None:
    """
    Maximum-resolution view centered exactly on the merger instant.
    Shows the compressed cascade structure (if it exists).
    """
    fig = plt.figure(figsize=(18, 14))
    gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.30)
    fig.suptitle('Script 90 -- The Whip Crack: Maximum Resolution at the Merger Instant',
                 fontsize=15, fontweight='bold', color='#1A2A44', y=0.98)

    # Use BBH:0001 as primary
    r0 = next((r for r in results if r['id'] == '0001'), results[0])
    t_rel = r0['t'] - r0['t_merger']

    # Panel 1: Ultra-zoom waveform ±15 M
    ax1 = fig.add_subplot(gs[0, 0])
    mask_zoom = (t_rel > -15) & (t_rel < 15)
    ax1.plot(t_rel[mask_zoom], r0['h_real'][mask_zoom], 'k-', lw=1.5)
    ax1.fill_between(t_rel[mask_zoom], r0['amp'][mask_zoom],
                     -r0['amp'][mask_zoom], alpha=0.1, color='blue')
    # Mark half-period crossings
    for tc in r0['crossings_h']:
        tc_rel = tc - r0['t_merger']
        if abs(tc_rel) < 15:
            ax1.axvline(tc_rel, color='red', ls=':', lw=0.5, alpha=0.5)
    ax1.axvline(0, color='gray', ls='--', lw=1, alpha=0.5)
    ax1.set_xlabel('$t - t_{merger}$ [M]', fontsize=10)
    ax1.set_ylabel('Re($h_{22}$)', fontsize=10)
    ax1.set_title('A. Waveform Ultra-Zoom (BBH:0001)',
                  fontsize=12, fontweight='bold', color='#1A2A44')
    ax1.grid(True, alpha=0.2)

    # Panel 2: omega(t) ultra-zoom ±15 M
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(t_rel[mask_zoom], r0['omega'][mask_zoom], color=COLORS[0], lw=2)
    # Mark omega_dot zero crossings
    for tc in r0['crossings_odot']:
        tc_rel = tc - r0['t_merger']
        if abs(tc_rel) < 15:
            ax2.axvline(tc_rel, color='orange', ls='-', lw=1, alpha=0.6)
    ax2.axvline(0, color='gray', ls='--', lw=1, alpha=0.5)
    ax2.set_xlabel('$t - t_{merger}$ [M]', fontsize=10)
    ax2.set_ylabel(r'$\omega(t)$ [rad/M]', fontsize=10)
    ax2.set_title('B. Frequency Ultra-Zoom',
                  fontsize=12, fontweight='bold', color='#1A2A44')
    ax2.grid(True, alpha=0.2)

    # Panel 3: omega_ddot ultra-zoom ±15 M with subcycle markers
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(t_rel[mask_zoom], r0['omega_ddot'][mask_zoom], color=COLORS[4], lw=1.5)
    ax3.axhline(0, color='black', ls='-', lw=0.5, alpha=0.3)
    for tc in r0['crossings_ddot']:
        tc_rel = tc - r0['t_merger']
        if abs(tc_rel) < 15:
            ax3.axvline(tc_rel, color='red', ls='-', lw=1, alpha=0.5)
    ax3.axvline(0, color='gray', ls='--', lw=1, alpha=0.5)
    ax3.set_xlabel('$t - t_{merger}$ [M]', fontsize=10)
    ax3.set_ylabel(r'$\ddot{\omega}(t)$', fontsize=10)
    ax3.set_title('C. Chirp Acceleration Ultra-Zoom',
                  fontsize=12, fontweight='bold', color='#1A2A44')
    ax3.grid(True, alpha=0.2)

    # Panel 4: Half-period evolution through merger
    ax4 = fig.add_subplot(gs[1, 0])
    hp = r0['half_periods']
    if len(hp) > 0:
        # Plot half-periods centered on merger
        hp_t = r0['crossings_h'][:-1] - r0['t_merger']
        mask_hp = (np.abs(hp_t) < 60)
        ax4.semilogy(hp_t[mask_hp], hp[mask_hp], 'o-', color=COLORS[0],
                     markersize=5, lw=1.5)
    ax4.axvline(0, color='gray', ls='--', lw=1, alpha=0.5)
    ax4.set_xlabel('$t - t_{merger}$ [M]', fontsize=10)
    ax4.set_ylabel('Half-period [M]', fontsize=10)
    ax4.set_title('D. Half-Period Compression at Merger',
                  fontsize=12, fontweight='bold', color='#1A2A44')
    ax4.grid(True, alpha=0.2)

    # Panel 5: Rate of half-period compression
    ax5 = fig.add_subplot(gs[1, 1])
    if len(hp) > 1:
        # dT/dn = T_{n+1} - T_n
        dT = np.diff(hp)
        # Merger-centered index
        dT_t = r0['crossings_h'][1:-1] - r0['t_merger']
        mask_dt = (np.abs(dT_t) < 60) & (dT < 0)  # only compressing
        if np.any(mask_dt):
            ax5.semilogy(dT_t[mask_dt], np.abs(dT[mask_dt]), 'o-',
                         color=COLORS[2], markersize=4, lw=1)
    ax5.axvline(0, color='gray', ls='--', lw=1, alpha=0.5)
    ax5.set_xlabel('$t - t_{merger}$ [M]', fontsize=10)
    ax5.set_ylabel('|$\\Delta T_n$| = |$T_{n+1} - T_n$|', fontsize=10)
    ax5.set_title('E. Half-Period Compression Rate',
                  fontsize=12, fontweight='bold', color='#1A2A44')
    ax5.grid(True, alpha=0.2)

    # Panel 6: The money shot — ratio of compression rates
    ax6 = fig.add_subplot(gs[1, 2])
    if len(hp) > 2:
        dT = np.diff(hp)
        # Only negative (compressing) differences
        valid_dT = (dT < -1e-12)  # avoid division issues
        dT_neg = dT.copy()
        dT_neg[~valid_dT] = np.nan

        # Ratio of consecutive compression steps
        comp_ratio = np.full(len(dT_neg) - 1, np.nan)
        for k in range(len(dT_neg) - 1):
            if np.isfinite(dT_neg[k]) and np.isfinite(dT_neg[k+1]) and abs(dT_neg[k+1]) > 1e-12:
                comp_ratio[k] = dT_neg[k] / dT_neg[k+1]

        comp_t = r0['crossings_h'][1:-2] - r0['t_merger']
        mask_comp = (np.abs(comp_t) < 40) & np.isfinite(comp_ratio) & \
                    (comp_ratio > 0.1) & (comp_ratio < 15)

        if np.any(mask_comp):
            ax6.plot(comp_t[mask_comp], comp_ratio[mask_comp], 'o-',
                     color=COLORS[0], markersize=5, lw=1.5,
                     label='$\\Delta T_n / \\Delta T_{n+1}$')

            # Annotate values near delta
            for j in range(len(comp_t)):
                if mask_comp[j] and abs(comp_ratio[j] - DELTA_FEIG) / DELTA_FEIG < 0.3:
                    ax6.annotate(f'{comp_ratio[j]:.2f}',
                                (comp_t[j], comp_ratio[j]),
                                textcoords="offset points", xytext=(5, 5),
                                fontsize=7, color='red', fontweight='bold')

    ax6.axhline(DELTA_FEIG, color='red', ls='--', lw=2, alpha=0.7,
                label=f'$\\delta$ = {DELTA_FEIG:.3f}')
    ax6.axhline(ALPHA_FEIG, color='purple', ls='--', lw=1.5, alpha=0.5,
                label=f'$\\alpha$ = {ALPHA_FEIG:.3f}')
    ax6.axhline(1.0, color='gray', ls=':', lw=1, alpha=0.3)
    ax6.axvline(0, color='gray', ls='--', lw=1, alpha=0.5)
    ax6.set_xlabel('$t - t_{merger}$ [M]', fontsize=10)
    ax6.set_ylabel('Compression ratio', fontsize=10)
    ax6.set_title('F. THE SNAP: Compression Ratio at Merger',
                  fontsize=12, fontweight='bold', color='#1A2A44')
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.2)
    ax6.set_ylim(0, 10)

    out = os.path.join(SCRIPT_DIR, "fig90d_whip_crack_zoom.png")
    fig.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"\n  Saved: {out}  ({os.path.getsize(out):,} bytes)")


# ==========================================================================
#  MAIN
# ==========================================================================
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  Script 90 -- The Snap: Subcycle Feigenbaum Extraction")
    print("  The Whip Crack Hypothesis")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Step 1: Extract full-resolution data from all cached waveforms
    # ------------------------------------------------------------------
    print("\n--- STEP 1: Full-Resolution Frequency Extraction ---\n")

    results = []
    for wf in WAVEFORMS:
        filepath = os.path.join(DATA_DIR, f"SXS_BBH_{wf['id']}.h5")
        if not os.path.exists(filepath):
            print(f"  BBH:{wf['id']} -- file not found, skipping")
            continue

        print(f"  BBH:{wf['id']} -- {wf['desc']}")
        res = extract_full_resolution(filepath, wf, window_M=150.0)
        if res is not None:
            res = analyze_merger_trajectory(res)
            results.append(res)

    print(f"\n  Analyzed: {len(results)}/{len(WAVEFORMS)} waveforms")

    if len(results) == 0:
        print("  ERROR: No waveforms analyzed. Exiting.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step 2: Print subcycle analysis summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  SUBCYCLE ANALYSIS TABLE")
    print("=" * 70)
    print(f"\n  {'Waveform':<25s} {'N_pts':>6s} {'dt_min':>8s} "
          f"{'N_sc':>6s} {'N_hp':>6s} {'R_max':>8s} {'R_near_delta':>12s}")
    print("  " + "-" * 73)

    for res in results:
        # Count ratios near delta (within 30%)
        near_hp = []
        if len(res['hp_traj_val']) > 0:
            mask = (np.abs(res['hp_traj_t']) < 15) & \
                   (res['hp_traj_val'] > 0.5) & (res['hp_traj_val'] < 15)
            near_hp = res['hp_traj_val'][mask]

        n_near_delta = sum(1 for v in near_hp
                          if abs(v - DELTA_FEIG) / DELTA_FEIG < 0.3)
        max_hp = np.max(near_hp) if len(near_hp) > 0 else float('nan')

        print(f"  BBH:{res['id']} ({res['desc'][:18]:<18s}) "
              f"{res['n_points']:6d} {res['dt_min']:8.5f} "
              f"{len(res['subcycle_intervals']):6d} {len(res['half_periods']):6d} "
              f"{max_hp:8.3f} {n_near_delta:12d}")

    # ------------------------------------------------------------------
    # Step 3: Near-merger compression analysis
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  HALF-PERIOD COMPRESSION RATIOS AT MERGER")
    print("=" * 70)

    for res in results:
        hp = res['half_periods']
        if len(hp) < 3:
            continue

        crossings = res['crossings_h']
        t_m = res['t_merger']

        # Find the merger crossing
        i_m = np.argmin(np.abs(crossings - t_m))

        # Half-period differences
        dT = np.diff(hp)

        print(f"\n  BBH:{res['id']} (q={res['q']:.0f}):")

        # Print near-merger half-periods and their ratios
        for k in range(max(0, i_m - 5), min(len(hp), i_m + 5)):
            t_rel = crossings[k] - t_m
            T_k = hp[k]
            is_merger = "<<< MERGER" if k == i_m else ""

            if k < len(dT):
                dT_k = dT[k]
                if k > 0 and abs(dT[k]) > 1e-12:
                    ratio_comp = dT[k-1] / dT[k] if abs(dT[k-1]) > 1e-12 else float('nan')
                    if np.isfinite(ratio_comp) and 0 < ratio_comp < 20:
                        delta_pct = 100 * (ratio_comp - DELTA_FEIG) / DELTA_FEIG
                        print(f"    n={k:3d} t-t_m={t_rel:+8.2f} M  "
                              f"T={T_k:8.4f}  dT={dT_k:+9.5f}  "
                              f"R={ratio_comp:6.3f} "
                              f"(vs delta: {delta_pct:+.1f}%) {is_merger}")
                    else:
                        print(f"    n={k:3d} t-t_m={t_rel:+8.2f} M  "
                              f"T={T_k:8.4f}  dT={dT_k:+9.5f}  "
                              f"R=  ---   {is_merger}")
                else:
                    print(f"    n={k:3d} t-t_m={t_rel:+8.2f} M  "
                          f"T={T_k:8.4f}  dT={dT_k:+9.5f}  {is_merger}")
            else:
                print(f"    n={k:3d} t-t_m={t_rel:+8.2f} M  "
                      f"T={T_k:8.4f}  {is_merger}")

    # ------------------------------------------------------------------
    # Step 4: THE VERDICT
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  THE SNAP — VERDICT")
    print("=" * 70)

    # Collect all near-merger compression ratios from all waveforms
    all_comp_ratios = []
    for res in results:
        hp = res['half_periods']
        if len(hp) < 3:
            continue
        dT = np.diff(hp)
        crossings = res['crossings_h']
        t_m = res['t_merger']

        for k in range(1, len(dT)):
            t_rel = crossings[k] - t_m
            if abs(t_rel) < 20 and abs(dT[k]) > 1e-12:
                r = dT[k-1] / dT[k]
                if np.isfinite(r) and 0.1 < r < 20:
                    all_comp_ratios.append(r)

    if all_comp_ratios:
        arr = np.array(all_comp_ratios)
        print(f"\n  All near-merger (|t|<20 M) compression ratios:")
        print(f"    N = {len(arr)}")
        print(f"    Mean = {np.mean(arr):.4f}")
        print(f"    Median = {np.median(arr):.4f}")
        print(f"    Std = {np.std(arr):.4f}")
        print(f"    Max = {np.max(arr):.4f}")
        print(f"    Min = {np.min(arr):.4f}")

        n_near_delta = sum(1 for v in arr if abs(v - DELTA_FEIG) / DELTA_FEIG < 0.15)
        n_near_alpha = sum(1 for v in arr if abs(v - ALPHA_FEIG) / ALPHA_FEIG < 0.15)
        print(f"\n    Within 15% of delta ({DELTA_FEIG:.3f}): "
              f"{n_near_delta}/{len(arr)} ({100*n_near_delta/len(arr):.1f}%)")
        print(f"    Within 15% of alpha ({ALPHA_FEIG:.3f}): "
              f"{n_near_alpha}/{len(arr)} ({100*n_near_alpha/len(arr):.1f}%)")

        # Check for upward trajectory toward delta
        slopes_up = 0
        for res in results:
            if len(res['hp_traj_val']) > 2:
                mask = (res['hp_traj_t'] > -30) & (res['hp_traj_t'] < 5) & \
                       (res['hp_traj_val'] > 0) & (res['hp_traj_val'] < 15)
                if np.sum(mask) >= 2:
                    t_v = res['hp_traj_t'][mask]
                    r_v = res['hp_traj_val'][mask]
                    if len(t_v) >= 2:
                        slope = np.polyfit(t_v, r_v, 1)[0]
                        if slope > 0:
                            slopes_up += 1

        print(f"\n    Waveforms with upward-sloping ratios toward merger: "
              f"{slopes_up}/{len(results)}")

    else:
        print("\n  No valid compression ratios found.")

    # ------------------------------------------------------------------
    # Step 5: Generate figures
    # ------------------------------------------------------------------
    print("\n--- Generating Figures ---")
    plot_frequency_evolution(results)
    plot_subcycle_intervals(results)
    plot_universality(results)
    plot_whip_crack_zoom(results)

    print("\n" + "=" * 70)
    print("  Complete.")
    print("=" * 70)
