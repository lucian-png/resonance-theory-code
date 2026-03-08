"""
+============================================================================+
|  (c) 2026 Lucian Randolph. All rights reserved.                           |
|                                                                            |
|  Script 89 -- The Ringdown Resolution:                                     |
|  Exponential Decay or the Randolph Constant?                               |
+============================================================================+

Script 89 -- The one calculation that resolves the ringdown signal.

    For each SXS waveform, compute the QNM-predicted exponential decay
    ratio E_trivial = e^{pi/Q} and compare to the measured envelope ratio.

    If E_trivial matches E_measured:  exponential wins (honest negative).
    If E_trivial != E_measured AND the residual is universal:
        the Randolph Constant lambda_R = delta/alpha lives in the ringdown.

Also:
    Merger period ratio convergence trajectory analysis.
    Do the near-merger R_n values arc toward delta?

Generates:
    fig89a_ringdown_resolution.png   (3-panel QNM comparison)
    fig89b_merger_trajectory.png     (convergence trajectory plot)
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
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
SQRT2      = np.sqrt(2.0)

# The compound prediction: lambda_R * sqrt(2) = delta/alpha * sqrt(2)
LAMBDA_R_SQRT2 = LAMBDA_R * SQRT2      # 2.63818...

print(f"  Randolph Constant:    lambda_R = delta/alpha = {LAMBDA_R:.8f}")
print(f"  lambda_R * sqrt(2)  = {LAMBDA_R_SQRT2:.8f}")
print(f"  delta               = {DELTA_FEIG:.8f}")
print(f"  alpha               = {ALPHA_FEIG:.8f}")


# ==========================================================================
#  SXS WAVEFORM METADATA (from catalog.json, verified)
# ==========================================================================
# Correct mass ratios and remnant parameters from the SXS catalog
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
#  BERTI-CARDOSO-STARINETS QNM FITTING FORMULA
# ==========================================================================
# Reference: Berti, Cardoso, Will, PRD 73, 064030 (2006)
# Fitting coefficients for (l=2, m=2, n=0) fundamental mode
#
# M * omega_R = f1 + f2 * (1 - a_f)^f3
# Q = q1 + q2 * (1 - a_f)^q3        (quality factor)
# omega_I = omega_R / (2*Q)
#
# Accuracy: <1% across all spins 0 <= a_f < 0.99

BCS_F1, BCS_F2, BCS_F3 = 1.5251, -1.1568, 0.1292
BCS_Q1, BCS_Q2, BCS_Q3 = 0.7000,  1.4187, -0.4990


def qnm_220(chi_f: float) -> Tuple[float, float, float]:
    """
    Compute the fundamental (l=2, m=2, n=0) QNM of a Kerr black hole.

    Args:
        chi_f: Dimensionless spin of the final black hole (0 <= chi_f < 1)

    Returns:
        (M*omega_R, M*omega_I, Q) where:
            omega_R = oscillation frequency (real part)
            omega_I = damping rate (imaginary part, positive = decaying)
            Q = quality factor = omega_R / (2*omega_I)
    """
    x = 1.0 - chi_f
    M_omega_R = BCS_F1 + BCS_F2 * x**BCS_F3
    Q = BCS_Q1 + BCS_Q2 * x**BCS_Q3
    M_omega_I = M_omega_R / (2.0 * Q)
    return M_omega_R, M_omega_I, Q


def e_trivial(chi_f: float) -> float:
    """
    The trivial exponential decay prediction for the ringdown envelope ratio.

    For a damped sinusoid h ~ e^{-omega_I * t} * cos(omega_R * t),
    the ratio of consecutive amplitude changes (separated by one period
    P = 2*pi/omega_R) is:

        E_trivial = e^{2*pi * omega_I / omega_R} = e^{pi / Q}

    This is a property of the exponential, not Feigenbaum.
    """
    _, _, Q = qnm_220(chi_f)
    return np.exp(np.pi / Q)


# ==========================================================================
#  RINGDOWN ENVELOPE EXTRACTION
# ==========================================================================
def extract_ringdown_envelope(filepath: str, wf_info: Dict) -> Optional[Dict]:
    """
    Extract the ringdown envelope ratio from an SXS waveform.

    Focused analysis: only the ringdown portion (t > t_merger).
    Finds peaks of |h22|, computes amplitude differences, gets E_n ratios.
    """
    try:
        with h5py.File(filepath, 'r') as f:
            # Extract (2,2) mode
            for gname in ['Extrapolated_N2.dir', 'Extrapolated_N3.dir',
                          'Extrapolated_N4.dir']:
                if gname in f and 'Y_l2_m2.dat' in f[gname]:
                    data = f[gname]['Y_l2_m2.dat'][()]
                    t = data[:, 0]
                    h22 = data[:, 1] + 1j * data[:, 2]
                    break
            else:
                print(f"    Cannot find (2,2) mode in {filepath}")
                return None
    except Exception as e:
        print(f"    Error: {e}")
        return None

    amp = np.abs(h22)
    h_real = np.real(h22)

    # Find merger (peak amplitude)
    i_peak = np.argmax(amp)
    t_merger = t[i_peak]
    amp_peak = amp[i_peak]

    # --- RINGDOWN ANALYSIS ---
    # Focus on t_merger to t_merger + 300 M (pure ringdown)
    RINGDOWN_START = t_merger + 10.0   # skip the immediate merger peak
    RINGDOWN_END   = t_merger + 300.0

    mask_rd = (t >= RINGDOWN_START) & (t <= RINGDOWN_END)
    t_rd = t[mask_rd]
    h_rd = h_real[mask_rd]
    amp_rd = amp[mask_rd]

    if len(t_rd) < 50:
        print(f"    Too few ringdown points")
        return None

    # Find positive-going zero crossings in ringdown
    crossings = []
    for i in range(len(h_rd) - 1):
        if h_rd[i] <= 0 and h_rd[i+1] > 0:
            frac = -h_rd[i] / (h_rd[i+1] - h_rd[i])
            t_cross = t_rd[i] + frac * (t_rd[i+1] - t_rd[i])
            crossings.append(t_cross)

    crossings = np.array(crossings)
    n_cross = len(crossings)

    if n_cross < 4:
        print(f"    Only {n_cross} ringdown crossings — too few")
        return None

    # Peak amplitude in each cycle
    peak_amps = []
    for i in range(len(crossings) - 1):
        cycle_mask = (t >= crossings[i]) & (t < crossings[i+1])
        if np.any(cycle_mask):
            peak_amps.append(np.max(amp[cycle_mask]))
        else:
            peak_amps.append(np.nan)
    peak_amps = np.array(peak_amps)

    # Amplitude differences
    delta_amp = np.diff(peak_amps)  # A_{n+1} - A_n (negative during decay)

    # Envelope ratios
    valid = np.abs(delta_amp[1:]) > 1e-14
    E_n = np.full(len(delta_amp) - 1, np.nan)
    E_n[valid] = delta_amp[:-1][valid] / delta_amp[1:][valid]

    # Periods
    periods = np.diff(crossings)

    # Also get the amplitude ratios directly (A_n / A_{n+1})
    amp_ratios = peak_amps[:-1] / peak_amps[1:]

    # --- MERGER PERIOD RATIO ANALYSIS ---
    # Use wider window to capture inspiral → merger → ringdown
    WIDE_START = t_merger - 500.0
    WIDE_END   = t_merger + 200.0
    mask_wide = (t >= WIDE_START) & (t <= WIDE_END)
    t_wide = t[mask_wide]
    h_wide = h_real[mask_wide]

    wide_crossings = []
    for i in range(len(h_wide) - 1):
        if h_wide[i] <= 0 and h_wide[i+1] > 0:
            frac = -h_wide[i] / (h_wide[i+1] - h_wide[i])
            t_cross = t_wide[i] + frac * (t_wide[i+1] - t_wide[i])
            wide_crossings.append(t_cross)
    wide_crossings = np.array(wide_crossings)

    # Period ratios through merger
    wide_periods = np.diff(wide_crossings) if len(wide_crossings) > 1 else np.array([])
    wide_delta_T = np.diff(wide_periods) if len(wide_periods) > 1 else np.array([])
    wide_R_n = np.full(max(0, len(wide_delta_T) - 1), np.nan)
    if len(wide_delta_T) > 1:
        valid_w = np.abs(wide_delta_T[1:]) > 1e-10
        wide_R_n[valid_w] = wide_delta_T[:-1][valid_w] / wide_delta_T[1:][valid_w]

    # Find merger crossing index
    merger_idx = -1
    if len(wide_crossings) > 0:
        merger_idx = np.argmin(np.abs(wide_crossings - t_merger))

    # Extract the three near-merger ratios
    R_merger_m1 = np.nan  # R at merger-1
    R_merger_0  = np.nan  # R at merger
    R_merger_p1 = np.nan  # R at merger+1

    if merger_idx >= 0 and len(wide_R_n) > 0:
        # R_n index that corresponds to the merger crossing
        # merger_idx is in the crossings array; R_n[i] uses crossings[i]
        r_idx = merger_idx - 2  # -2 because R_n needs 3 consecutive periods

        if 0 <= r_idx - 1 < len(wide_R_n):
            R_merger_m1 = wide_R_n[r_idx - 1]
        if 0 <= r_idx < len(wide_R_n):
            R_merger_0 = wide_R_n[r_idx]
        if 0 <= r_idx + 1 < len(wide_R_n):
            R_merger_p1 = wide_R_n[r_idx + 1]

    return {
        'id': wf_info['id'],
        'desc': wf_info['desc'],
        'q': wf_info['q'],
        'chi_f': wf_info['chi_f'],
        'M_f': wf_info['M_f'],
        't_merger': t_merger,
        'amp_peak': amp_peak,
        # Ringdown
        'n_ringdown_cycles': n_cross,
        'E_n': E_n,
        'periods_rd': periods,
        'peak_amps_rd': peak_amps,
        'amp_ratios_rd': amp_ratios,
        'crossings_rd': crossings,
        # Merger period ratios
        'R_merger_m1': R_merger_m1,
        'R_merger_0': R_merger_0,
        'R_merger_p1': R_merger_p1,
        'wide_R_n': wide_R_n,
        'wide_crossings': wide_crossings,
        'merger_idx': merger_idx,
    }


# ==========================================================================
#  FIGURE 89a: THE RINGDOWN RESOLUTION
# ==========================================================================
def plot_resolution(results: List[Dict]) -> None:
    """Three-panel figure resolving the ringdown signal."""

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Script 89 — The Ringdown Resolution: '
                 'Exponential Decay or the Randolph Constant?',
                 fontsize=14, fontweight='bold', color='#1A2A44', y=1.02)

    # Collect data for plots
    chi_vals = []
    E_triv_vals = []
    E_meas_vals = []
    E_resid_vals = []
    labels = []

    for res in results:
        chi_f = res['chi_f']
        E_t = e_trivial(chi_f)

        # Use the MOST STABLE ringdown E values (skip first 2, last 2)
        E_arr = res['E_n']
        finite = np.isfinite(E_arr) & (E_arr > 1.0) & (E_arr < 20.0)
        if np.sum(finite) < 2:
            continue

        # Take the stable middle values
        good_E = E_arr[finite]
        if len(good_E) > 4:
            good_E = good_E[1:-1]  # skip first and last

        E_m = np.median(good_E)
        E_residual = E_m / E_t

        chi_vals.append(chi_f)
        E_triv_vals.append(E_t)
        E_meas_vals.append(E_m)
        E_resid_vals.append(E_residual)
        labels.append(f"BBH:{res['id']}\nq={res['q']:.0f}")

    chi_vals = np.array(chi_vals)
    E_triv_vals = np.array(E_triv_vals)
    E_meas_vals = np.array(E_meas_vals)
    E_resid_vals = np.array(E_resid_vals)

    # Color by mass ratio
    q_vals = np.array([r['q'] for r in results
                       if np.sum(np.isfinite(r['E_n']) &
                                 (r['E_n'] > 1.0) & (r['E_n'] < 20.0)) >= 2])
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(chi_vals)))

    # --- Panel A: E_trivial vs E_measured ---
    ax = axes[0]
    ax.plot([2.0, 5.0], [2.0, 5.0], 'k--', lw=1.5, alpha=0.5,
            label='Perfect match (E_trivial = E_measured)')
    for i in range(len(chi_vals)):
        ax.plot(E_triv_vals[i], E_meas_vals[i], 'o', markersize=10,
                color=colors[i], zorder=5)
        ax.annotate(labels[i], (E_triv_vals[i], E_meas_vals[i]),
                    textcoords="offset points", xytext=(8, -5),
                    fontsize=7, color=colors[i])

    # Mark lambda_R * sqrt(2) reference
    ax.axhline(y=LAMBDA_R_SQRT2, color='red', linestyle=':', alpha=0.7, lw=1.5,
               label=f'$\\lambda_R \\sqrt{{2}}$ = {LAMBDA_R_SQRT2:.4f}')

    ax.set_xlabel('$E_{trivial}$ = $e^{\\pi/Q}$ (QNM prediction)', fontsize=11)
    ax.set_ylabel('$E_{measured}$ (NR waveform)', fontsize=11)
    ax.set_title('A. Predicted vs Measured Envelope Ratio',
                 fontsize=12, fontweight='bold', color='#1A2A44')
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    lim_lo = min(min(E_triv_vals), min(E_meas_vals)) - 0.2
    lim_hi = max(max(E_triv_vals), max(E_meas_vals)) + 0.2
    ax.set_xlim(lim_lo, lim_hi)
    ax.set_ylim(lim_lo, lim_hi)

    # --- Panel B: E_residual vs chi_f ---
    ax = axes[1]
    for i in range(len(chi_vals)):
        ax.plot(chi_vals[i], E_resid_vals[i], 'o', markersize=10,
                color=colors[i], zorder=5)
        ax.annotate(labels[i], (chi_vals[i], E_resid_vals[i]),
                    textcoords="offset points", xytext=(8, -5),
                    fontsize=7, color=colors[i])

    # Reference lines
    ax.axhline(y=1.0, color='black', linestyle='-', alpha=0.5, lw=1.5,
               label='Perfect exponential (residual = 1)')
    ax.axhline(y=LAMBDA_R / ALPHA_FEIG, color='purple', linestyle=':', alpha=0.5,
               label=f'$\\lambda_R / \\alpha$ = {LAMBDA_R/ALPHA_FEIG:.4f}')

    ax.set_xlabel('Final spin $\\chi_f$', fontsize=11)
    ax.set_ylabel('$E_{residual}$ = $E_{measured}$ / $E_{trivial}$', fontsize=11)
    ax.set_title('B. Residual vs Final Spin',
                 fontsize=12, fontweight='bold', color='#1A2A44')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)

    # --- Panel C: Residual compared to Feigenbaum quantities ---
    ax = axes[2]
    feig_refs = {
        '1.0 (exponential)': 1.0,
        f'$\\lambda_R/\\alpha$ = {LAMBDA_R/ALPHA_FEIG:.4f}': LAMBDA_R / ALPHA_FEIG,
        f'$\\sqrt{{\\lambda_R}}$ = {np.sqrt(LAMBDA_R):.4f}': np.sqrt(LAMBDA_R),
        f'$1/\\ln(\\delta)$ = {1/LN_DELTA:.4f}': 1.0 / LN_DELTA,
    }

    # Box plot of residuals
    ax.boxplot([E_resid_vals], positions=[0], widths=0.4,
               patch_artist=True,
               boxprops=dict(facecolor='lightblue', alpha=0.7))

    for i in range(len(E_resid_vals)):
        ax.plot(0 + 0.15 * (np.random.random() - 0.5), E_resid_vals[i],
                'o', markersize=8, color=colors[i], zorder=5, alpha=0.8)

    y_offset = 0
    for name, val in feig_refs.items():
        ax.axhline(y=val, linestyle='--', alpha=0.6, lw=1.5)
        ax.text(0.55, val, name, fontsize=8, va='center')

    ax.set_xlim(-0.5, 2.0)
    ax.set_ylabel('$E_{residual}$', fontsize=11)
    ax.set_title('C. Residual vs Feigenbaum Quantities',
                 fontsize=12, fontweight='bold', color='#1A2A44')
    ax.set_xticks([0])
    ax.set_xticklabels(['All\nwaveforms'])
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    out = os.path.join(SCRIPT_DIR, "fig89a_ringdown_resolution.png")
    fig.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"\n  Saved: {out}  ({os.path.getsize(out):,} bytes)")


# ==========================================================================
#  FIGURE 89b: MERGER TRAJECTORY ANALYSIS
# ==========================================================================
def plot_merger_trajectory(results: List[Dict]) -> None:
    """Two-panel figure showing near-merger period ratio trajectories."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Script 89 — Merger Period Ratio Convergence Trajectory',
                 fontsize=14, fontweight='bold', color='#1A2A44', y=1.02)

    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

    # --- Panel A: Three-point trajectories ---
    ax = axes[0]

    trajectories = []
    for i, res in enumerate(results):
        R_m1 = res['R_merger_m1']
        R_0  = res['R_merger_0']
        R_p1 = res['R_merger_p1']

        pts = [R_m1, R_0, R_p1]
        valid = [not np.isnan(p) and np.isfinite(p) and abs(p) < 20 for p in pts]

        if sum(valid) >= 2:
            x = np.array([-1, 0, 1])
            y = np.array(pts)

            # Plot valid points
            v_mask = np.array(valid)
            ax.plot(x[v_mask], y[v_mask], 'o-', color=colors[i], markersize=8,
                    lw=2, alpha=0.8,
                    label=f"BBH:{res['id']} (q={res['q']:.0f})")
            trajectories.append((x[v_mask], y[v_mask], res))

    # Reference lines
    ax.axhline(y=DELTA_FEIG, color='red', linestyle='--', lw=2, alpha=0.7,
               label=f'$\\delta$ = {DELTA_FEIG:.3f}')
    ax.axhline(y=LAMBDA_R_SQRT2, color='purple', linestyle=':', lw=1.5, alpha=0.5,
               label=f'$\\lambda_R \\sqrt{{2}}$ = {LAMBDA_R_SQRT2:.3f}')

    ax.set_xlabel('Cycle offset from merger', fontsize=11)
    ax.set_ylabel('Period ratio $R_n$', fontsize=11)
    ax.set_xticks([-1, 0, 1])
    ax.set_xticklabels(['merger$-$1', 'merger', 'merger$+$1'])
    ax.set_title('A. Near-Merger Period Ratio Trajectories',
                 fontsize=12, fontweight='bold', color='#1A2A44')
    ax.legend(fontsize=7, loc='upper right', ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-2, 12)

    # --- Panel B: Extrapolation ---
    ax = axes[1]

    crossing_points = []
    for i, (xv, yv, res) in enumerate(trajectories):
        if len(xv) >= 2:
            # Linear fit
            coeffs = np.polyfit(xv, yv, 1)
            slope, intercept = coeffs

            # Extrapolate
            x_ext = np.linspace(-2, 5, 100)
            y_ext = slope * x_ext + intercept

            ax.plot(xv, yv, 'o', color=colors[i], markersize=8, zorder=5)
            ax.plot(x_ext, y_ext, '-', color=colors[i], alpha=0.4, lw=1)

            # Where does the extrapolation cross delta?
            if slope != 0:
                x_cross = (DELTA_FEIG - intercept) / slope
                crossing_points.append(x_cross)
                if -3 < x_cross < 8:
                    ax.plot(x_cross, DELTA_FEIG, 'x', color=colors[i],
                            markersize=10, mew=2, zorder=6)

    ax.axhline(y=DELTA_FEIG, color='red', linestyle='--', lw=2, alpha=0.7,
               label=f'$\\delta$ = {DELTA_FEIG:.3f}')

    # Report crossing points
    if crossing_points:
        valid_cp = [cp for cp in crossing_points if -5 < cp < 10]
        if valid_cp:
            mean_cp = np.mean(valid_cp)
            std_cp = np.std(valid_cp)
            ax.axvline(x=mean_cp, color='gray', linestyle=':', alpha=0.5)
            ax.text(mean_cp + 0.1, 2.0,
                    f'Mean crossing:\n{mean_cp:.1f} cycles\npast merger',
                    fontsize=9, color='gray')

    ax.set_xlabel('Cycle offset from merger', fontsize=11)
    ax.set_ylabel('Period ratio $R_n$ (extrapolated)', fontsize=11)
    ax.set_title('B. Linear Extrapolation Toward $\\delta$',
                 fontsize=12, fontweight='bold', color='#1A2A44')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-3, 8)
    ax.set_ylim(-2, 12)

    plt.tight_layout()
    out = os.path.join(SCRIPT_DIR, "fig89b_merger_trajectory.png")
    fig.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"\n  Saved: {out}  ({os.path.getsize(out):,} bytes)")


# ==========================================================================
#  MAIN
# ==========================================================================
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  Script 89 — The Ringdown Resolution")
    print("  Exponential Decay or the Randolph Constant?")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Step 1: QNM predictions for all waveforms
    # ------------------------------------------------------------------
    print("\n--- STEP 1: QNM Predictions (Berti-Cardoso-Starinets) ---\n")
    print(f"  {'Waveform':<30s} {'chi_f':>8s} {'M*wR':>8s} {'M*wI':>8s} "
          f"{'Q':>8s} {'E_trivial':>10s}")
    print("  " + "-" * 68)

    for wf in WAVEFORMS:
        M_wR, M_wI, Q = qnm_220(wf['chi_f'])
        E_t = e_trivial(wf['chi_f'])
        print(f"  BBH:{wf['id']} ({wf['desc']:<20s}) "
              f"{wf['chi_f']:8.4f} {M_wR:8.4f} {M_wI:8.4f} {Q:8.3f} {E_t:10.4f}")

    # Schwarzschild check
    M_wR_0, M_wI_0, Q_0 = qnm_220(0.0)
    print(f"\n  Schwarzschild check: M*wR={M_wR_0:.4f} "
          f"(exact: 0.3737), Q={Q_0:.3f} (exact: 2.10)")
    print(f"  E_trivial(a=0) = {e_trivial(0.0):.4f} (= e^{{pi/2.10}} = 4.47)")

    # ------------------------------------------------------------------
    # Step 2: Extract measured envelope ratios from cached waveforms
    # ------------------------------------------------------------------
    print("\n--- STEP 2: Extracting Ringdown Envelopes from NR Waveforms ---")

    results = []
    for wf in WAVEFORMS:
        filepath = os.path.join(DATA_DIR, f"SXS_BBH_{wf['id']}.h5")
        if not os.path.exists(filepath):
            print(f"\n  BBH:{wf['id']} — file not found, skipping")
            continue

        print(f"\n  BBH:{wf['id']} — {wf['desc']}")
        res = extract_ringdown_envelope(filepath, wf)
        if res is not None:
            results.append(res)

    print(f"\n  Analyzed: {len(results)}/{len(WAVEFORMS)} waveforms")

    # ------------------------------------------------------------------
    # Step 3: THE COMPARISON TABLE
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  THE CRITICAL TABLE — Exponential vs Measured")
    print("=" * 70)
    print(f"\n  {'Waveform':<25s} {'chi_f':>6s} {'Q':>6s} "
          f"{'E_trivial':>10s} {'E_measured':>10s} {'Diff%':>8s} "
          f"{'E_resid':>8s}")
    print("  " + "-" * 75)

    all_residuals = []
    for res in results:
        chi_f = res['chi_f']
        _, _, Q = qnm_220(chi_f)
        E_t = e_trivial(chi_f)

        # Measured: take median of stable ringdown values
        E_arr = res['E_n']
        finite = np.isfinite(E_arr) & (E_arr > 1.0) & (E_arr < 20.0)
        if np.sum(finite) < 2:
            print(f"  BBH:{res['id']:<20s} — too few valid E values")
            continue

        good_E = E_arr[finite]
        if len(good_E) > 4:
            good_E = good_E[1:-1]  # skip first and last for stability

        E_m = np.median(good_E)
        diff_pct = 100.0 * (E_m - E_t) / E_t
        E_resid = E_m / E_t

        all_residuals.append({
            'id': res['id'], 'desc': res['desc'], 'q': res['q'],
            'chi_f': chi_f, 'Q': Q,
            'E_trivial': E_t, 'E_measured': E_m,
            'diff_pct': diff_pct, 'E_residual': E_resid,
        })

        desc_short = res['desc'][:18]
        print(f"  BBH:{res['id']} ({desc_short:<18s}) "
              f"{chi_f:6.4f} {Q:6.3f} {E_t:10.4f} {E_m:10.4f} "
              f"{diff_pct:+7.2f}% {E_resid:8.4f}")

    # ------------------------------------------------------------------
    # Step 4: VERDICT
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  VERDICT")
    print("=" * 70)

    if all_residuals:
        diffs = [r['diff_pct'] for r in all_residuals]
        resids = [r['E_residual'] for r in all_residuals]
        mean_diff = np.mean(np.abs(diffs))
        max_diff = np.max(np.abs(diffs))
        mean_resid = np.mean(resids)
        std_resid = np.std(resids)

        print(f"\n  Mean |difference|:     {mean_diff:.2f}%")
        print(f"  Max |difference|:      {max_diff:.2f}%")
        print(f"  Mean residual:         {mean_resid:.4f}")
        print(f"  Std residual:          {std_resid:.4f}")

        print(f"\n  Reference values:")
        print(f"    lambda_R * sqrt(2) = {LAMBDA_R_SQRT2:.6f}")
        print(f"    delta              = {DELTA_FEIG:.6f}")
        print(f"    alpha              = {ALPHA_FEIG:.6f}")
        print(f"    lambda_R           = {LAMBDA_R:.6f}")

        if mean_diff < 1.0:
            print(f"\n  >>> OUTCOME A: Mean difference {mean_diff:.2f}% < 1%")
            print(f"  >>> The exponential decay e^{{pi/Q}} EXPLAINS the")
            print(f"  >>> ringdown envelope ratio.")
            print(f"  >>> Proximity to lambda_R * sqrt(2) = {LAMBDA_R_SQRT2:.4f}")
            print(f"  >>> is a COINCIDENCE with the QNM quality factor")
            print(f"  >>> at chi_f ~ 0.686.")
            print(f"  >>> HONEST NEGATIVE on the Feigenbaum connection.")
        elif std_resid < 0.05 * mean_resid:
            print(f"\n  >>> OUTCOME C: Difference {mean_diff:.2f}% > 1% AND")
            print(f"  >>> residual is UNIVERSAL (std/mean = "
                  f"{std_resid/mean_resid:.4f})")
            print(f"  >>> The Randolph Constant may be in the ringdown!")
        else:
            print(f"\n  >>> OUTCOME B: Difference {mean_diff:.2f}% > 1% BUT")
            print(f"  >>> residual is NOT universal (std/mean = "
                  f"{std_resid/mean_resid:.4f})")
            print(f"  >>> Something beyond exponential but not Feigenbaum.")

    # ------------------------------------------------------------------
    # Step 5: Merger period ratio analysis
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  MERGER PERIOD RATIO TRAJECTORIES")
    print("=" * 70)

    print(f"\n  {'Waveform':<25s} {'R(m-1)':>8s} {'R(m)':>8s} "
          f"{'R(m+1)':>8s} {'Slope':>8s} {'Cross delta at':>15s}")
    print("  " + "-" * 70)

    for res in results:
        R_m1 = res['R_merger_m1']
        R_0  = res['R_merger_0']
        R_p1 = res['R_merger_p1']

        pts = [R_m1, R_0, R_p1]
        valid_pts = [(x, y) for x, y in zip([-1, 0, 1], pts)
                     if not np.isnan(y) and np.isfinite(y) and abs(y) < 20]

        slope_str = "—"
        cross_str = "—"
        if len(valid_pts) >= 2:
            xv = np.array([p[0] for p in valid_pts])
            yv = np.array([p[1] for p in valid_pts])
            coeffs = np.polyfit(xv, yv, 1)
            slope = coeffs[0]
            intercept = coeffs[1]
            slope_str = f"{slope:+.2f}"

            if slope != 0:
                x_cross = (DELTA_FEIG - intercept) / slope
                cross_str = f"merger{x_cross:+.1f}"

        def fmt(v: float) -> str:
            if np.isnan(v) or not np.isfinite(v) or abs(v) > 20:
                return "   —   "
            return f"{v:+8.3f}"

        print(f"  BBH:{res['id']} ({res['desc'][:18]:<18s}) "
              f"{fmt(R_m1)} {fmt(R_0)} {fmt(R_p1)} "
              f"{slope_str:>8s} {cross_str:>15s}")

    # ------------------------------------------------------------------
    # Step 6: Figures
    # ------------------------------------------------------------------
    print("\n--- Generating Figures ---")

    if results:
        plot_resolution(results)
        plot_merger_trajectory(results)

    # ------------------------------------------------------------------
    # Final individual E_n printout for deep inspection
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  DETAILED RINGDOWN E_n VALUES")
    print("=" * 70)

    for res in results:
        E_arr = res['E_n']
        finite = np.isfinite(E_arr) & (np.abs(E_arr) < 50)
        print(f"\n  BBH:{res['id']} (chi_f={res['chi_f']:.4f}, "
              f"E_trivial={e_trivial(res['chi_f']):.4f}):")

        E_t = e_trivial(res['chi_f'])
        for i in range(len(E_arr)):
            if finite[i]:
                diff = 100.0 * (E_arr[i] - E_t) / E_t
                cross_t = res['crossings_rd'][i] - res['t_merger']
                print(f"    E_{i:2d} (t-t_m={cross_t:+8.1f} M): "
                      f"{E_arr[i]:8.4f}  "
                      f"(vs e^{{pi/Q}}={E_t:.4f}: {diff:+.2f}%)")

    print("\n" + "=" * 70)
    print("  Complete.")
    print("=" * 70)
