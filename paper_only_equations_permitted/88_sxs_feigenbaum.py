"""
+============================================================================+
|  (c) 2026 Lucian Randolph. All rights reserved.                           |
|                                                                            |
|  Script 88 -- Feigenbaum Extraction from SXS NR Waveforms                  |
+============================================================================+

Script 88 -- Avenue Three, Option A:

    Download public numerical relativity waveforms from the SXS
    Gravitational Waveform Catalog. Extract the (2,2) strain mode.
    Identify merger cycles. Compute period ratios, frequency ratios,
    and strain envelope ratios through the nonlinear merger transition.

    THE REAL TEST: not perturbative, not analytic, but the actual
    solution to Einstein's field equations on a computational grid.

Data source:
    SXS Gravitational Waveform Catalog (public)
    https://data.black-holes.org/waveforms/catalog.html

Generates:
    fig88a_nr_waveforms.png
    fig88b_merger_ratios.png
    fig88c_universality.png
    fig88d_money_shot.png  (if positive)
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from scipy.interpolate import UnivariateSpline
import h5py
import requests
from typing import Optional, Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(SCRIPT_DIR, "sxs_data")
os.makedirs(DATA_DIR, exist_ok=True)

# Feigenbaum constants
DELTA_FEIG = 4.669201609102990
ALPHA_FEIG = 2.502907875095892
LN_DELTA   = np.log(DELTA_FEIG)
DELTA_OVER_ALPHA = DELTA_FEIG / ALPHA_FEIG

# Reference quantities for comparison
FEIG_REFS = {
    r'$\delta$': DELTA_FEIG,
    r'$\ln(\delta)$': LN_DELTA,
    r'$\delta/\alpha$': DELTA_OVER_ALPHA,
    r'$\alpha$': ALPHA_FEIG,
}


# ============================================================================
#  SXS WAVEFORM CATALOG -- TARGET SIMULATIONS
# ============================================================================
# Ten carefully chosen waveforms spanning different physics
# Format: (SXS_ID, description, mass_ratio_q, spin_notes)
TARGETS = [
    ("0001", "q=1, non-spinning",     1.0,  "chi=0"),
    ("0002", "q=1, low spin",         1.0,  "chi~0.2"),
    ("0007", "q=1.5, non-spinning",   1.5,  "chi=0"),
    ("0056", "q=2, non-spinning",     2.0,  "chi=0"),
    ("0063", "q=3, non-spinning",     3.0,  "chi=0"),
    ("0167", "q=5, non-spinning",     5.0,  "chi=0"),
    ("0180", "q=8, non-spinning",     8.0,  "chi=0"),
    ("0150", "q=1, aligned high spin", 1.0, "chi~0.9"),
    ("0004", "q=1, mod spin aligned", 1.0,  "chi~0.44"),
    ("1355", "q=1, precessing",       1.0,  "precessing"),
]


# ============================================================================
#  DOWNLOAD
# ============================================================================
# Catalog-resolved Zenodo download URLs (from catalog.json)
# These are the correct bucket URLs for each waveform's rhOverM_CoM file.
ZENODO_URLS: Dict[str, str] = {
    "0001": "https://zenodo.org/api/records/3312723/files/SXS:BBH:0001/Lev5/rhOverM_Asymptotic_GeometricUnits_CoM.h5/content",
    "0002": "https://zenodo.org/api/records/3312772/files/SXS:BBH:0002/Lev4/rhOverM_Asymptotic_GeometricUnits_CoM.h5/content",
    "0004": "https://zenodo.org/api/records/3312662/files/SXS:BBH:0004/Lev4/rhOverM_Asymptotic_GeometricUnits_CoM.h5/content",
    "0007": "https://zenodo.org/api/records/3313587/files/SXS:BBH:0007/Lev4/rhOverM_Asymptotic_GeometricUnits_CoM.h5/content",
    "0056": "https://zenodo.org/api/records/3312192/files/SXS:BBH:0056/Lev3/rhOverM_Asymptotic_GeometricUnits_CoM.h5/content",
    "0063": "https://zenodo.org/api/records/3312567/files/SXS:BBH:0063/Lev3/rhOverM_Asymptotic_GeometricUnits_CoM.h5/content",
    "0150": "https://zenodo.org/api/records/3312384/files/SXS:BBH:0150/Lev1/rhOverM_Asymptotic_GeometricUnits_CoM.h5/content",
    "0167": "https://zenodo.org/api/records/3312651/files/SXS:BBH:0167/Lev3/rhOverM_Asymptotic_GeometricUnits_CoM.h5/content",
    "0180": "https://zenodo.org/api/records/3302035/files/SXS:BBH:0180/Lev2/rhOverM_Asymptotic_GeometricUnits_CoM.h5/content",
    "1355": "https://zenodo.org/api/records/3326529/files/SXS:BBH:1355/Lev1/rhOverM_Asymptotic_GeometricUnits_CoM.h5/content",
}


def download_sxs_waveform(sxs_id: str) -> Optional[str]:
    """
    Download an SXS waveform from the Zenodo-hosted catalog.
    Uses pre-resolved bucket URLs from catalog.json.
    Returns path to local HDF5 file, or None if download fails.
    """
    local_path = os.path.join(DATA_DIR, f"SXS_BBH_{sxs_id}.h5")
    if os.path.exists(local_path) and os.path.getsize(local_path) > 10000:
        print(f"    [cached] {local_path} ({os.path.getsize(local_path):,} bytes)")
        return local_path

    url = ZENODO_URLS.get(sxs_id)
    if url is None:
        print(f"    No URL for SXS:BBH:{sxs_id}")
        return None

    try:
        print(f"    Downloading from Zenodo...")
        resp = requests.get(url, timeout=600, stream=True, allow_redirects=True)
        if resp.status_code == 200:
            total = int(resp.headers.get('content-length', 0))
            total_mb = total / 1e6 if total > 0 else 0
            print(f"    Size: {total_mb:.0f} MB")
            downloaded = 0
            with open(local_path, 'wb') as f:
                for chunk in resp.iter_content(chunk_size=65536):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total > 0 and downloaded % (10 * 1024 * 1024) < 65536:
                        pct = 100 * downloaded / total
                        print(f"    ... {pct:.0f}%", flush=True)
            final_size = os.path.getsize(local_path)
            print(f"    Downloaded: {final_size:,} bytes")
            if final_size > 10000:
                return local_path
            else:
                os.remove(local_path)
        else:
            print(f"    HTTP {resp.status_code}")
    except Exception as e:
        print(f"    Error: {e}")
        if os.path.exists(local_path):
            os.remove(local_path)

    print(f"    FAILED: Could not download SXS:BBH:{sxs_id}")
    return None


# ============================================================================
#  WAVEFORM EXTRACTION
# ============================================================================
def extract_h22(filepath: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Extract the (2,2) strain mode from an SXS HDF5 file.

    Returns (time, h22_complex) where h22 = h+ - i*hx for the (2,2) mode.
    Time is in units of M (total mass).
    """
    try:
        with h5py.File(filepath, 'r') as f:
            # Print top-level structure for debugging
            keys = list(f.keys())

            # SXS format: look for 'Y_l2_m2.dat' or similar
            # The standard SXS HDF5 has groups like 'OutermostExtraction.dir'
            # or 'Extrapolated_N2.dir' containing mode data

            h22 = None
            t = None

            # Strategy 1: Look for Extrapolated data (preferred)
            for group_name in ['Extrapolated_N2.dir', 'Extrapolated_N3.dir',
                               'Extrapolated_N4.dir',
                               'OutermostExtraction.dir']:
                if group_name in f:
                    grp = f[group_name]
                    mode_key = 'Y_l2_m2.dat'
                    if mode_key in grp:
                        data = grp[mode_key][()]
                        t = data[:, 0]
                        h22 = data[:, 1] + 1j * data[:, 2]
                        print(f"    Mode source: {group_name}/{mode_key}")
                        break

            # Strategy 2: flat structure
            if h22 is None:
                for key in keys:
                    if 'l2_m2' in key.lower() or 'Y_l2_m2' in key:
                        data = f[key][()]
                        t = data[:, 0]
                        h22 = data[:, 1] + 1j * data[:, 2]
                        print(f"    Mode source: {key}")
                        break

            # Strategy 3: Recurse to find any (2,2) mode data
            if h22 is None:
                def find_22(group, path=""):
                    for k in group.keys():
                        full = f"{path}/{k}"
                        if isinstance(group[k], h5py.Dataset):
                            if '2_m2' in k or 'l2_m2' in k:
                                return full
                        elif isinstance(group[k], h5py.Group):
                            result = find_22(group[k], full)
                            if result:
                                return result
                    return None

                path_22 = find_22(f)
                if path_22:
                    data = f[path_22][()]
                    t = data[:, 0]
                    h22 = data[:, 1] + 1j * data[:, 2]
                    print(f"    Mode source: {path_22}")

            if h22 is None:
                print(f"    Could not find (2,2) mode. Keys: {keys[:10]}")
                # Try to explore deeper
                for k in keys[:5]:
                    if isinstance(f[k], h5py.Group):
                        subkeys = list(f[k].keys())[:10]
                        print(f"      {k}/ -> {subkeys}")
                return None

            print(f"    Time range: [{t[0]:.1f}, {t[-1]:.1f}] M")
            print(f"    Samples: {len(t):,}")
            return t, h22

    except Exception as e:
        print(f"    Error reading HDF5: {e}")
        return None


# ============================================================================
#  MERGER ANALYSIS
# ============================================================================
def analyze_merger(t: np.ndarray, h22: np.ndarray,
                   label: str) -> Optional[Dict]:
    """
    Full merger analysis:
    - Find peak amplitude (merger time)
    - Use FIXED wide window (±500 M) to capture many cycles through merger
    - Compute period ratios, frequency ratios, envelope ratios
    """
    # Amplitude and phase
    amp = np.abs(h22)
    phase = np.unwrap(np.angle(h22))

    # Instantaneous frequency (in units of 1/M)
    # Use |omega| — SXS phase convention can give negative dphi/dt
    dt = np.diff(t)
    omega_raw = np.diff(phase) / dt
    omega = np.abs(omega_raw)
    t_omega = 0.5 * (t[:-1] + t[1:])

    # Smooth frequency with rolling average (more robust than spline near merger)
    if len(t_omega) > 100:
        window = min(101, len(omega) // 20)
        if window % 2 == 0:
            window += 1
        if window >= 3:
            omega_smooth = np.convolve(omega, np.ones(window)/window,
                                        mode='same')
        else:
            omega_smooth = omega.copy()
    else:
        omega_smooth = omega.copy()

    # Find merger: peak amplitude
    i_peak = np.argmax(amp)
    t_merger = t[i_peak]
    amp_peak = amp[i_peak]
    print(f"\n  {label}")
    print(f"    Merger at t/M = {t_merger:.1f}, peak |rh/M| = {amp_peak:.4f}")

    # Merger frequency (absolute value)
    i_peak_omega = np.argmin(np.abs(t_omega - t_merger))
    omega_merger = omega_smooth[i_peak_omega]
    f_merger = omega_merger / (2.0 * np.pi)
    T_merger = 1.0 / f_merger if f_merger > 1e-6 else 20.0
    print(f"    |omega|_merger = {omega_merger:.3f} /M, "
          f"f_merger = {f_merger:.4f} /M, T_merger = {T_merger:.2f} M")

    # FIXED wide merger window: ±500 M covers ~15-25 late inspiral cycles
    # plus ringdown, regardless of frequency estimation
    WINDOW_HALF = 500.0  # in units of M
    t_start = t_merger - WINDOW_HALF
    t_end   = t_merger + WINDOW_HALF / 3  # less ringdown needed

    # Find ALL zero crossings of Re(h22) in the wide window
    h_real = np.real(h22)
    mask = (t >= t_start) & (t <= t_end)
    t_win = t[mask]
    h_win = h_real[mask]

    if len(t_win) < 20:
        print(f"    Too few points in window [{t_start:.0f}, {t_end:.0f}]")
        return None

    # Zero crossings (positive-going for full cycles)
    crossings = []
    for i in range(len(h_win) - 1):
        if h_win[i] <= 0 and h_win[i+1] > 0:
            frac = -h_win[i] / (h_win[i+1] - h_win[i])
            t_cross = t_win[i] + frac * (t_win[i+1] - t_win[i])
            crossings.append(t_cross)

    crossings = np.array(crossings)
    n_cycles = len(crossings)
    print(f"    Zero crossings in [{t_start:.0f}, {t_end:.0f}] M: {n_cycles}")

    if n_cycles < 4:
        # Also try negative-going crossings (opposite phase convention)
        crossings2 = []
        for i in range(len(h_win) - 1):
            if h_win[i] >= 0 and h_win[i+1] < 0:
                frac = h_win[i] / (h_win[i] - h_win[i+1])
                t_cross = t_win[i] + frac * (t_win[i+1] - t_win[i])
                crossings2.append(t_cross)
        if len(crossings2) > n_cycles:
            crossings = np.array(crossings2)
            n_cycles = len(crossings)
            print(f"    (Using negative-going crossings: {n_cycles})")

    if n_cycles < 4:
        # Try ALL zero crossings (both directions) to get half-periods
        all_crossings = []
        for i in range(len(h_win) - 1):
            if h_win[i] * h_win[i+1] < 0:
                frac = np.abs(h_win[i]) / np.abs(h_win[i+1] - h_win[i])
                t_cross = t_win[i] + frac * (t_win[i+1] - t_win[i])
                all_crossings.append(t_cross)
        if len(all_crossings) >= 8:
            # Take every other crossing for full-period analysis
            crossings = np.array(all_crossings[::2])
            n_cycles = len(crossings)
            print(f"    (Using alternating crossings: {n_cycles})")

    if n_cycles < 4:
        print(f"    Too few cycles ({n_cycles}) for ratio analysis")
        return None

    # Periods between successive same-type crossings
    periods = np.diff(crossings)
    print(f"    Cycles: {n_cycles}, "
          f"Period range: [{periods.min():.3f}, {periods.max():.3f}] M")

    # Period changes
    delta_T = np.diff(periods)

    # Period ratios: R_n = delta_T_n / delta_T_{n+1}
    valid = np.abs(delta_T[1:]) > 1e-10
    R_n = np.full(len(delta_T) - 1, np.nan)
    R_n[valid] = delta_T[:-1][valid] / delta_T[1:][valid]

    # Find which crossing is closest to merger
    merger_cross_idx = np.argmin(np.abs(crossings - t_merger))

    print(f"    Merger at crossing index: {merger_cross_idx}/{n_cycles}")
    print(f"    Period ratios through merger:")
    for i in range(len(R_n)):
        cross_t = crossings[i]
        rel = cross_t - t_merger
        near = " <-- NEAR MERGER" if abs(i - merger_cross_idx) <= 1 else ""
        if not np.isnan(R_n[i]) and np.isfinite(R_n[i]):
            print(f"      R_{i:2d} (t-t_m={rel:+8.2f} M): {R_n[i]:+10.4f}{near}")

    # Frequency at each crossing
    omega_at_cross = np.interp(crossings, t_omega, omega_smooth)
    delta_omega = np.diff(omega_at_cross)
    valid_f = np.abs(delta_omega[1:]) > 1e-10
    F_n = np.full(len(delta_omega) - 1, np.nan)
    F_n[valid_f] = delta_omega[:-1][valid_f] / delta_omega[1:][valid_f]

    print(f"\n    Frequency ratios through merger:")
    for i in range(len(F_n)):
        if not np.isnan(F_n[i]) and np.isfinite(F_n[i]):
            cross_t = crossings[i]
            rel = cross_t - t_merger
            print(f"      F_{i:2d} (t-t_m={rel:+8.2f} M): {F_n[i]:+10.4f}")

    # Strain envelope at each cycle: peak amplitude between crossings
    peak_amps = []
    for i in range(len(crossings) - 1):
        cycle_mask = (t >= crossings[i]) & (t < crossings[i+1])
        if np.any(cycle_mask):
            peak_amps.append(np.max(amp[cycle_mask]))
        else:
            peak_amps.append(np.nan)
    peak_amps = np.array(peak_amps)

    # Envelope ratios
    delta_amp = np.diff(peak_amps)
    valid_e = np.abs(delta_amp[1:]) > 1e-12
    E_n = np.full(len(delta_amp) - 1, np.nan)
    E_n[valid_e] = delta_amp[:-1][valid_e] / delta_amp[1:][valid_e]

    print(f"\n    Envelope ratios through merger:")
    for i in range(len(E_n)):
        if not np.isnan(E_n[i]) and np.isfinite(E_n[i]):
            cross_t = crossings[i]
            rel = cross_t - t_merger
            print(f"      E_{i:2d} (t-t_m={rel:+8.2f} M): {E_n[i]:+10.4f}")

    return {
        'label': label,
        't': t, 'h22': h22, 'amp': amp, 'phase': phase,
        't_omega': t_omega, 'omega': omega_smooth,
        't_merger': t_merger, 'amp_peak': amp_peak,
        'omega_merger': omega_merger,
        'crossings': crossings, 'periods': periods,
        'R_n': R_n, 'F_n': F_n, 'E_n': E_n,
        'peak_amps': peak_amps,
        'merger_cross_idx': merger_cross_idx,
        't_start': t_start, 't_end': t_end,
    }


# ============================================================================
#  FIGURES
# ============================================================================
def plot_waveforms(results: List[Dict]) -> None:
    """Figure 88a: Representative NR waveforms with merger windows."""
    n_show = min(4, len(results))
    fig, axes = plt.subplots(n_show, 1, figsize=(14, 3.5 * n_show))
    if n_show == 1:
        axes = [axes]

    fig.suptitle('SXS Numerical Relativity Waveforms — Merger Windows',
                 fontsize=14, fontweight='bold', color='#1A2A44', y=1.01)

    for ax, res in zip(axes, results[:n_show]):
        t = res['t']
        h_real = np.real(res['h22'])
        t_m = res['t_merger']

        # Show merger window region
        ax.axvspan(res['t_start'] - t_m, res['t_end'] - t_m,
                   alpha=0.15, color='red', label='Merger window')
        ax.plot((t - t_m), h_real, 'b-', lw=0.5, alpha=0.7)
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.5, lw=1)

        # Mark crossings
        for tc in res['crossings']:
            ax.axvline(x=tc - t_m, color='green', alpha=0.2, lw=0.5)

        ax.set_ylabel('Re(rh₂₂/M)', fontsize=10)
        ax.set_title(res['label'], fontsize=11, color='#1A2A44')
        ax.set_xlim(-200, 100)
        ax.grid(True, alpha=0.2)

    axes[-1].set_xlabel('Time relative to merger (M)', fontsize=11)
    plt.tight_layout()
    out = os.path.join(SCRIPT_DIR, "fig88a_nr_waveforms.png")
    fig.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"\nSaved: {out}  ({os.path.getsize(out):,} bytes)")


def plot_merger_ratios(results: List[Dict]) -> None:
    """Figure 88b: Period, frequency, and envelope ratios for lead waveform."""
    if not results:
        return

    res = results[0]  # Lead waveform (q=1 non-spinning)

    fig, axes = plt.subplots(3, 1, figsize=(12, 11))

    crossings = res['crossings']
    t_m = res['t_merger']
    cross_rel = crossings[:len(res['R_n'])] - t_m

    ref_colors = {'$\\delta$': 'red', '$\\ln(\\delta)$': 'green',
                  '$\\delta/\\alpha$': 'purple', '$\\alpha$': 'orange'}

    # Panel A: Period ratios
    ax = axes[0]
    R = res['R_n']
    finite = np.isfinite(R)
    ax.plot(cross_rel[finite], R[finite], 'ko-', markersize=6, lw=1.5,
            label='Period ratio $R_n$')
    for name, val in FEIG_REFS.items():
        ax.axhline(y=val, color=ref_colors.get(name, 'gray'),
                   linestyle='--', alpha=0.6, label=f'{name} = {val:.3f}')
        ax.axhline(y=-val, color=ref_colors.get(name, 'gray'),
                   linestyle=':', alpha=0.3)
    ax.axhline(y=1, color='gray', linestyle=':', alpha=0.3)
    ax.set_ylabel('$R_n = \\Delta T_n / \\Delta T_{n+1}$', fontsize=11)
    ax.set_title(f'Merger Cycle Ratios — {res["label"]}',
                 fontsize=13, fontweight='bold', color='#1A2A44')
    ax.legend(fontsize=8, loc='best', ncol=2)
    ax.grid(True, alpha=0.3)

    # Panel B: Frequency ratios
    ax = axes[1]
    F = res['F_n']
    cross_f = crossings[:len(F)] - t_m
    finite = np.isfinite(F)
    ax.plot(cross_f[finite], F[finite], 'bs-', markersize=6, lw=1.5,
            label='Frequency ratio $F_n$')
    for name, val in FEIG_REFS.items():
        ax.axhline(y=val, color=ref_colors.get(name, 'gray'),
                   linestyle='--', alpha=0.6, label=f'{name} = {val:.3f}')
    ax.axhline(y=1, color='gray', linestyle=':', alpha=0.3)
    ax.set_ylabel('$F_n = \\Delta\\omega_n / \\Delta\\omega_{n+1}$',
                  fontsize=11)
    ax.legend(fontsize=8, loc='best', ncol=2)
    ax.grid(True, alpha=0.3)

    # Panel C: Envelope ratios
    ax = axes[2]
    E = res['E_n']
    cross_e = crossings[:len(E)] - t_m
    finite = np.isfinite(E)
    ax.plot(cross_e[finite], E[finite], 'r^-', markersize=6, lw=1.5,
            label='Envelope ratio $E_n$')
    for name, val in FEIG_REFS.items():
        ax.axhline(y=val, color=ref_colors.get(name, 'gray'),
                   linestyle='--', alpha=0.6, label=f'{name} = {val:.3f}')
    ax.axhline(y=1, color='gray', linestyle=':', alpha=0.3)
    ax.set_ylabel('$E_n = \\Delta A_n / \\Delta A_{n+1}$', fontsize=11)
    ax.set_xlabel('Time relative to merger (M)', fontsize=11)
    ax.legend(fontsize=8, loc='best', ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(SCRIPT_DIR, "fig88b_merger_ratios.png")
    fig.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"\nSaved: {out}  ({os.path.getsize(out):,} bytes)")


def plot_universality(results: List[Dict]) -> None:
    """Figure 88c: Cross-waveform universality test."""
    if len(results) < 2:
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

    titles = ['Period Ratios $R_n$', 'Frequency Ratios $F_n$',
              'Envelope Ratios $E_n$']
    ratio_keys = ['R_n', 'F_n', 'E_n']
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

    for ax, title, key in zip(axes, titles, ratio_keys):
        for i, res in enumerate(results):
            ratios = res[key]
            finite = np.isfinite(ratios) & (np.abs(ratios) < 20)
            if np.any(finite):
                n = np.arange(len(ratios))[finite]
                ax.plot(n, ratios[finite], 'o-', markersize=4, lw=1,
                        color=colors[i], alpha=0.7,
                        label=res['label'][:20])

        for name, val in FEIG_REFS.items():
            ref_colors = {'$\\delta$': 'red', '$\\ln(\\delta)$': 'green',
                          '$\\delta/\\alpha$': 'purple', '$\\alpha$': 'orange'}
            ax.axhline(y=val, color=ref_colors.get(name, 'gray'),
                       linestyle='--', alpha=0.5, lw=1.5,
                       label=name if ax == axes[0] else None)

        ax.set_title(title, fontsize=12, fontweight='bold', color='#1A2A44')
        ax.set_xlabel('Cycle index', fontsize=10)
        ax.set_ylabel('Ratio', fontsize=10)
        ax.set_ylim(-10, 10)
        ax.grid(True, alpha=0.3)

    axes[0].legend(fontsize=7, loc='best', ncol=1)
    plt.suptitle('Cross-Waveform Universality Test',
                 fontsize=14, fontweight='bold', color='#1A2A44')
    plt.tight_layout()
    out = os.path.join(SCRIPT_DIR, "fig88c_universality.png")
    fig.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"\nSaved: {out}  ({os.path.getsize(out):,} bytes)")


def plot_money_shot(results: List[Dict]) -> None:
    """
    Figure 88d: If any ratio converges toward δ, show it.
    Otherwise, show the honest distribution of all merger ratios
    with Feigenbaum references.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Collect all finite ratios from all waveforms
    all_R = []
    all_F = []
    all_E = []

    for res in results:
        r = res['R_n'][np.isfinite(res['R_n'])]
        f = res['F_n'][np.isfinite(res['F_n'])]
        e = res['E_n'][np.isfinite(res['E_n'])]
        all_R.extend(r[(np.abs(r) < 20)])
        all_F.extend(f[(np.abs(f) < 20)])
        all_E.extend(e[(np.abs(e) < 20)])

    all_R = np.array(all_R)
    all_F = np.array(all_F)
    all_E = np.array(all_E)

    # Panel A: Distribution of all merger ratios
    ax = axes[0]
    bins = np.linspace(-10, 10, 81)
    if len(all_R) > 0:
        ax.hist(all_R, bins=bins, alpha=0.5, color='black',
                label=f'Period $R_n$ (n={len(all_R)})', density=True)
    if len(all_F) > 0:
        ax.hist(all_F, bins=bins, alpha=0.5, color='blue',
                label=f'Frequency $F_n$ (n={len(all_F)})', density=True)
    if len(all_E) > 0:
        ax.hist(all_E, bins=bins, alpha=0.5, color='red',
                label=f'Envelope $E_n$ (n={len(all_E)})', density=True)

    for name, val in FEIG_REFS.items():
        ref_colors = {'$\\delta$': 'red', '$\\ln(\\delta)$': 'green',
                      '$\\delta/\\alpha$': 'purple', '$\\alpha$': 'orange'}
        ax.axvline(x=val, color=ref_colors.get(name, 'gray'),
                   linestyle='--', lw=2, alpha=0.7, label=f'{name}={val:.3f}')

    ax.set_xlabel('Ratio value', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('A. Distribution of All Merger Ratios',
                 fontsize=12, fontweight='bold', color='#1A2A44')
    ax.legend(fontsize=8, loc='upper left')
    ax.set_xlim(-8, 8)
    ax.grid(True, alpha=0.3)

    # Panel B: Closest matches to Feigenbaum quantities
    ax = axes[1]
    all_combined = np.concatenate([all_R, all_F, all_E])
    if len(all_combined) > 0:
        for i, (name, val) in enumerate(FEIG_REFS.items()):
            # How many ratios fall within ±20% of this value?
            near = np.abs(all_combined - val) / val < 0.20
            count = np.sum(near)
            total = len(all_combined)
            pct = 100.0 * count / total if total > 0 else 0

            ax.barh(i, pct, height=0.6, alpha=0.7,
                    color=['red', 'green', 'purple', 'orange'][i])
            ax.text(pct + 0.5, i, f'{count}/{total} ({pct:.1f}%)',
                    va='center', fontsize=10)

        ax.set_yticks(range(len(FEIG_REFS)))
        ax.set_yticklabels([f'{n} = {v:.3f}'
                           for n, v in FEIG_REFS.items()], fontsize=10)

    ax.set_xlabel('% of ratios within ±20%', fontsize=11)
    ax.set_title('B. Proximity to Feigenbaum Constants',
                 fontsize=12, fontweight='bold', color='#1A2A44')
    ax.grid(True, alpha=0.3, axis='x')

    plt.suptitle('Merger Ratio Assessment — All Waveforms Combined',
                 fontsize=14, fontweight='bold', color='#1A2A44')
    plt.tight_layout()
    out = os.path.join(SCRIPT_DIR, "fig88d_money_shot.png")
    fig.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"\nSaved: {out}  ({os.path.getsize(out):,} bytes)")


# ============================================================================
#  SUMMARY
# ============================================================================
def print_summary(results: List[Dict]) -> None:
    """Print the honest scorecard."""
    print("\n" + "=" * 70)
    print("  SUMMARY -- Feigenbaum Extraction from SXS NR Waveforms")
    print("=" * 70)

    print(f"\n  Waveforms analyzed: {len(results)}")

    # Collect all ratios
    all_R = np.concatenate([r['R_n'][np.isfinite(r['R_n'])] for r in results])
    all_F = np.concatenate([r['F_n'][np.isfinite(r['F_n'])] for r in results])
    all_E = np.concatenate([r['E_n'][np.isfinite(r['E_n'])] for r in results])

    # Filter to physical range
    all_R = all_R[np.abs(all_R) < 20]
    all_F = all_F[np.abs(all_F) < 20]
    all_E = all_E[np.abs(all_E) < 20]

    print(f"\n  Total ratios extracted:")
    print(f"    Period R_n:    {len(all_R)}")
    print(f"    Frequency F_n: {len(all_F)}")
    print(f"    Envelope E_n:  {len(all_E)}")

    all_combined = np.concatenate([all_R, all_F, all_E])

    print(f"\n  Proximity to Feigenbaum constants:")
    for name, val in FEIG_REFS.items():
        near_10 = np.sum(np.abs(all_combined - val) / val < 0.10)
        near_20 = np.sum(np.abs(all_combined - val) / val < 0.20)
        total = len(all_combined)
        print(f"    {name:20s} = {val:.4f}:  "
              f"±10%: {near_10}/{total} ({100*near_10/total:.1f}%),  "
              f"±20%: {near_20}/{total} ({100*near_20/total:.1f}%)")

    # Per-waveform means in the near-merger region
    print(f"\n  Per-waveform near-merger statistics:")
    for res in results:
        R = res['R_n']
        finite = np.isfinite(R) & (np.abs(R) < 20)
        if np.any(finite):
            vals = R[finite]
            print(f"    {res['label']:30s}: "
                  f"mean(R)={np.mean(vals):+.3f}, "
                  f"median(R)={np.median(vals):+.3f}, "
                  f"n={len(vals)}")

    print(f"\n  Feigenbaum reference values:")
    print(f"    δ     = {DELTA_FEIG:.6f}")
    print(f"    ln(δ) = {LN_DELTA:.6f}")
    print(f"    δ/α   = {DELTA_OVER_ALPHA:.6f}")
    print(f"    α     = {ALPHA_FEIG:.6f}")

    print("\n" + "=" * 70)


# ============================================================================
#  MAIN
# ============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("  Script 88 — Feigenbaum Extraction from SXS NR Waveforms")
    print("=" * 70)

    # Step 1: Download waveforms
    print("\n--- STEP 1: Downloading SXS waveforms ---")
    downloaded = []
    for sxs_id, desc, q, spin in TARGETS:
        print(f"\n  SXS:BBH:{sxs_id} — {desc}")
        path = download_sxs_waveform(sxs_id)
        if path:
            downloaded.append((sxs_id, desc, q, spin, path))

    print(f"\n  Downloaded: {len(downloaded)}/{len(TARGETS)} waveforms")

    if len(downloaded) == 0:
        print("\n  ERROR: No waveforms downloaded. Check network / URLs.")
        print("  The SXS catalog may require authentication or different URLs.")
        print("  See: https://data.black-holes.org/waveforms/catalog.html")
        sys.exit(1)

    # Step 2: Extract and analyze
    print("\n--- STEP 2: Extracting (2,2) mode and analyzing mergers ---")
    results = []
    for sxs_id, desc, q, spin, path in downloaded:
        data = extract_h22(path)
        if data is None:
            continue

        t, h22 = data
        label = f"SXS:{sxs_id} ({desc})"
        result = analyze_merger(t, h22, label)
        if result is not None:
            results.append(result)

    print(f"\n  Successfully analyzed: {len(results)}/{len(downloaded)} waveforms")

    if len(results) == 0:
        print("\n  ERROR: No waveforms could be analyzed.")
        sys.exit(1)

    # Step 3: Figures
    print("\n--- STEP 3: Generating figures ---")
    plot_waveforms(results)
    plot_merger_ratios(results)
    if len(results) >= 2:
        plot_universality(results)
    plot_money_shot(results)

    # Step 4: Summary
    print_summary(results)

    print("\n" + "=" * 70)
    print("  Complete.")
    print("=" * 70)
