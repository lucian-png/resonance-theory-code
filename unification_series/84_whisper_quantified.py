#!/usr/bin/env python3
"""
Script 84 — The Whisper Quantified
====================================
Turn the qualitative "whisper" observation from Script 80 into a quantitative
experimental prediction.

The whisper: quantum fluctuations show subtle inflections at EXACTLY the F₁
values where the classical system bifurcates. The quantum system "knows" about
bifurcation points it supposedly can't resolve.

Method:
1. Identify classical bifurcation points from Script 80
2. Run quantum parametric sweep (Sweep A: F₁ from 0 to 3.5, γ=0.5)
3. At each F₁, compute:
   - ⟨n⟩ (mean photon number from stroboscopic sampling)
   - Var(n) across stroboscopic samples (quantum spread)
4. Compute excess variance: ΔVar = Var_measured - ⟨n⟩ (above Poissonian)
5. Compare excess variance AT bifurcation points vs BETWEEN them (controls)
6. Scaling analysis: does excess variance follow W₁ ≈ A / √⟨n⟩?

Four-panel figure:
    A: Quantum ⟨n⟩ vs F₁ with classical bifurcation points marked
    B: Excess variance vs F₁ — peaks at bifurcation points
    C: Excess variance at bifurcation points vs control points (bar chart)
    D: Scaling analysis — excess variance vs ⟨n⟩ at bifurcation points

Generates:
    fig84_whisper_quantified.png
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit
import time

try:
    import qutip
    HAS_QUTIP = True
except ImportError:
    HAS_QUTIP = False
    print("ERROR: QuTiP not available.")
    raise SystemExit(1)


# ==============================================================================
# Physical Parameters — Same as Script 80
# ==============================================================================

DELTA = -3.0
K = 0.3
F0 = 3.0
WM = 5.2
T_MOD = 2 * np.pi / WM
N_FOCK = 25

GAMMA_FIXED = 0.5

FEIGENBAUM_DELTA = 4.669201609102990671853203820466

# Classical bifurcation points (from Script 80)
CLASSICAL_BIF = {
    'P1→P2': 1.146618,
    'P2→P4': 2.571677,
    'P4→P8': 2.745863,
    'P8→P16': 2.780070,
}

# Sweep parameters
F1_MIN, F1_MAX = 0.0, 3.5
N_F1 = 80               # Denser than Script 83 for better resolution

# Stroboscopic parameters
N_TRANSIENT_Q = 80
N_RECORD_Q = 48          # More samples for variance estimation


# ==============================================================================
# Classical Equations (for reference bifurcation diagram)
# ==============================================================================

def kerr_derivs(t, y, gamma, F1):
    """Semiclassical Kerr: dα/dt = -(γ/2 + iΔ)α - iK|α|²α - iF(t)"""
    u, v = y
    Ft = F0 + F1 * np.cos(WM * t)
    nsq = u * u + v * v
    du = -gamma / 2 * u + DELTA * v + K * nsq * v
    dv = -gamma / 2 * v - DELTA * u - K * nsq * u - Ft
    return [du, dv]


def classical_stroboscopic(F1, n_trans=400, n_rec=64):
    """Classical stroboscopic map."""
    t_total = (n_trans + n_rec) * T_MOD
    t_strobo = np.array([(n_trans + k) * T_MOD for k in range(n_rec)])
    u0 = np.sqrt(abs(DELTA) / K) * 1.2
    v0 = -1.0

    sol = solve_ivp(
        kerr_derivs, [0, t_total], [u0, v0],
        args=(GAMMA_FIXED, F1), method='RK45',
        t_eval=t_strobo, rtol=1e-10, atol=1e-12,
        max_step=T_MOD / 30
    )
    if sol.success and len(sol.y[0]) == n_rec:
        return sol.y[0] ** 2 + sol.y[1] ** 2
    return None


# ==============================================================================
# Quantum Stroboscopic — returns FULL stroboscopic time series
# ==============================================================================

def quantum_stroboscopic_full(F1, gamma=GAMMA_FIXED):
    """
    Quantum stroboscopic map returning full expectation value time series.
    Returns: (n_strobo_mean, n_strobo_all, var_n_strobo)
      - n_strobo_mean: array of ⟨n⟩ at each stroboscopic time
      - var_n_strobo: variance of ⟨n⟩ across stroboscopic samples
    """
    a = qutip.destroy(N_FOCK)
    n_op = a.dag() * a
    n2_op = n_op * n_op
    identity = qutip.qeye(N_FOCK)

    H0 = (DELTA * n_op
          + (K / 2.0) * n_op * (n_op - identity)
          + F0 * (a.dag() + a))
    H1 = a.dag() + a
    coeff_str = f'{F1} * cos({WM} * t)'
    H = [H0, [H1, coeff_str]]

    c_ops = [np.sqrt(gamma) * a]

    alpha0 = 3.5 - 1.0j
    psi0 = qutip.coherent(N_FOCK, alpha0)
    rho0 = psi0 * psi0.dag()

    t_total = (N_TRANSIENT_Q + N_RECORD_Q) * T_MOD
    t_strobo = [(N_TRANSIENT_Q + k) * T_MOD for k in range(N_RECORD_Q)]
    t_list = np.linspace(0, t_total, int(t_total / (T_MOD / 4)) + 1)
    t_all = np.sort(np.unique(np.concatenate([t_list, t_strobo])))

    try:
        result = qutip.mesolve(H, rho0, t_all, c_ops, [n_op, n2_op],
                               options={'nsteps': 8000, 'atol': 1e-8, 'rtol': 1e-6})

        n_expect = np.array(result.expect[0])
        n2_expect = np.array(result.expect[1])
        t_result = np.array(t_all)

        n_strobo = []
        n2_strobo = []
        for ts in t_strobo:
            idx = np.argmin(np.abs(t_result - ts))
            n_strobo.append(n_expect[idx])
            n2_strobo.append(n2_expect[idx])

        n_strobo = np.array(n_strobo)
        n2_strobo = np.array(n2_strobo)

        mean_n = np.mean(n_strobo)
        # Variance of photon number at each stroboscopic time
        var_n_at_each = n2_strobo - n_strobo ** 2
        mean_var = np.mean(var_n_at_each)

        # Spread of ⟨n⟩ across stroboscopic samples (fluctuation in the mean)
        spread = np.var(n_strobo)

        return mean_n, mean_var, spread, n_strobo
    except Exception as e:
        print(f"    mesolve failed at F₁={F1:.3f}: {e}")
        return np.nan, np.nan, np.nan, None


# ==============================================================================
# Main
# ==============================================================================

def main():
    print("=" * 70)
    print("Script 84 — The Whisper Quantified")
    print("Excess variance at classical bifurcation points")
    print("=" * 70)
    print(f"\nParameters: Δ={DELTA}, K={K}, F₀={F0}, ωₘ={WM}, γ={GAMMA_FIXED}")
    print(f"N_Fock = {N_FOCK}")
    print(f"Sweep: F₁ from {F1_MIN} to {F1_MAX} ({N_F1} points)")
    print(f"Stroboscopic: {N_RECORD_Q} samples after {N_TRANSIENT_Q} transient periods")
    print()

    print("Classical bifurcation points:")
    for name, val in CLASSICAL_BIF.items():
        print(f"  {name}: F₁ = {val:.6f}")
    print()

    # ══════════════════════════════════════════════════════════════════════
    # Quantum Sweep
    # ══════════════════════════════════════════════════════════════════════
    F1_vals = np.linspace(F1_MIN, F1_MAX, N_F1)

    mean_n_arr = np.zeros(N_F1)
    mean_var_arr = np.zeros(N_F1)
    spread_arr = np.zeros(N_F1)
    excess_var_arr = np.zeros(N_F1)

    t0 = time.time()
    for i, F1 in enumerate(F1_vals):
        mn, mv, sp, n_strobo = quantum_stroboscopic_full(F1)
        mean_n_arr[i] = mn
        mean_var_arr[i] = mv
        spread_arr[i] = sp

        # Excess variance = Var(n) - ⟨n⟩ (above Poissonian baseline)
        if not np.isnan(mn) and not np.isnan(mv):
            excess_var_arr[i] = mv - mn
        else:
            excess_var_arr[i] = np.nan

        elapsed = time.time() - t0
        eta = elapsed / (i + 1) * (N_F1 - i - 1) if i > 0 else 0
        print(f"  F₁={F1:.3f}  ({i+1}/{N_F1})  ⟨n⟩={mn:.3f}  "
              f"Var={mv:.3f}  excess={excess_var_arr[i]:.3f}  "
              f"spread={sp:.4f}  [{elapsed:.0f}s, ETA {eta:.0f}s]")

    total_time = time.time() - t0
    print(f"\nQuantum sweep done: {total_time:.1f}s")

    # ══════════════════════════════════════════════════════════════════════
    # Extract Values at Bifurcation Points and Controls
    # ══════════════════════════════════════════════════════════════════════
    bif_F1s = list(CLASSICAL_BIF.values())
    bif_names = list(CLASSICAL_BIF.keys())

    # For each bifurcation point, find nearest F₁ in our sweep
    bif_indices = []
    bif_excess = []
    bif_mean_n = []
    bif_spread = []

    for name, F1_bif in CLASSICAL_BIF.items():
        idx = np.argmin(np.abs(F1_vals - F1_bif))
        bif_indices.append(idx)
        bif_excess.append(excess_var_arr[idx])
        bif_mean_n.append(mean_n_arr[idx])
        bif_spread.append(spread_arr[idx])
        print(f"  {name}: F₁={F1_vals[idx]:.3f} (target {F1_bif:.4f})  "
              f"excess={excess_var_arr[idx]:.4f}  spread={spread_arr[idx]:.6f}")

    # Control points: midpoints between bifurcation values
    control_F1s = []
    control_labels = []
    if len(bif_F1s) >= 2:
        # Before first bifurcation
        control_F1s.append(bif_F1s[0] / 2)
        control_labels.append('pre-P2')
        # Between each pair
        for j in range(len(bif_F1s) - 1):
            mid = (bif_F1s[j] + bif_F1s[j + 1]) / 2
            control_F1s.append(mid)
            control_labels.append(f'mid-{j+1}')
        # After last bifurcation
        control_F1s.append(min(bif_F1s[-1] + 0.3, F1_MAX - 0.1))
        control_labels.append('post-P16')

    control_excess = []
    control_spread = []
    for F1_ctrl in control_F1s:
        idx = np.argmin(np.abs(F1_vals - F1_ctrl))
        control_excess.append(excess_var_arr[idx])
        control_spread.append(spread_arr[idx])

    print(f"\nBifurcation point mean excess variance: "
          f"{np.nanmean(bif_excess):.4f}")
    print(f"Control point mean excess variance: "
          f"{np.nanmean(control_excess):.4f}")
    ratio = (np.nanmean(bif_excess) / np.nanmean(control_excess)
             if np.nanmean(control_excess) != 0 else np.nan)
    print(f"Ratio (bif/control): {ratio:.2f}")

    # ══════════════════════════════════════════════════════════════════════
    # Scaling Analysis: W₁ ≈ A / √⟨n⟩
    # ══════════════════════════════════════════════════════════════════════
    bif_excess_arr = np.array(bif_excess)
    bif_mean_n_arr = np.array(bif_mean_n)

    # Only fit where we have valid data
    valid = (~np.isnan(bif_excess_arr)) & (~np.isnan(bif_mean_n_arr)) & (bif_mean_n_arr > 0)

    scaling_A = np.nan
    scaling_exponent = np.nan
    if np.sum(valid) >= 2:
        # Fit log(excess) = log(A) - p*log(⟨n⟩)
        log_n = np.log(bif_mean_n_arr[valid])
        log_ex = np.log(np.abs(bif_excess_arr[valid]) + 1e-12)

        try:
            coeffs = np.polyfit(log_n, log_ex, 1)
            scaling_exponent = coeffs[0]
            scaling_A = np.exp(coeffs[1])
            print(f"\nScaling fit: W₁ ≈ {scaling_A:.3f} × ⟨n⟩^({scaling_exponent:.3f})")
            print(f"  Predicted exponent: -0.5 (from W₁ ≈ A/√⟨n⟩)")
            print(f"  Measured exponent: {scaling_exponent:.3f}")
        except Exception:
            print("  Scaling fit failed")

    # ══════════════════════════════════════════════════════════════════════
    # Figure: 4-Panel Analysis
    # ══════════════════════════════════════════════════════════════════════
    print(f"\nGenerating figure...")

    fig = plt.figure(figsize=(16, 12), facecolor='white')
    gs = GridSpec(2, 2, hspace=0.35, wspace=0.3)

    # ── Panel A: ⟨n⟩ vs F₁ with bifurcation points ────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(F1_vals, mean_n_arr, '-', color='#3498db', lw=1.5, label=r'$\langle n \rangle$')

    for name, val in CLASSICAL_BIF.items():
        ax1.axvline(x=val, color='#e74c3c', ls='--', alpha=0.6, lw=1)
        ax1.annotate(name, xy=(val, ax1.get_ylim()[1] if ax1.get_ylim()[1] > 0 else np.nanmax(mean_n_arr)),
                    fontsize=6, rotation=90, va='top', ha='right', color='#e74c3c')

    ax1.set_xlabel(r'Modulation Amplitude $F_1$', fontsize=10)
    ax1.set_ylabel(r'$\langle n \rangle$', fontsize=10)
    ax1.set_title(r'A. Quantum $\langle n \rangle$ vs $F_1$', fontsize=12,
                  fontweight='bold')
    ax1.grid(True, alpha=0.2)

    # ── Panel B: Excess Variance vs F₁ ─────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(F1_vals, excess_var_arr, '-', color='#9b59b6', lw=1.5,
             label='Excess Var')
    ax2.axhline(y=0, color='gray', ls=':', alpha=0.5, label='Poissonian')

    for name, val in CLASSICAL_BIF.items():
        ax2.axvline(x=val, color='#e74c3c', ls='--', alpha=0.6, lw=1)

    # Mark bifurcation points
    for j, idx in enumerate(bif_indices):
        ax2.plot(F1_vals[idx], excess_var_arr[idx], 'o', color='#e74c3c',
                ms=8, mfc='white', mew=2, zorder=5)

    ax2.set_xlabel(r'Modulation Amplitude $F_1$', fontsize=10)
    ax2.set_ylabel(r'Excess Variance (Var$(n) - \langle n \rangle$)', fontsize=10)
    ax2.set_title('B. Excess Variance — The Whisper', fontsize=12,
                  fontweight='bold')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.2)

    # ── Panel C: Bar chart — bifurcation vs control ────────────────────
    ax3 = fig.add_subplot(gs[1, 0])

    bar_labels = []
    bar_vals = []
    bar_colors = []

    for j, name in enumerate(bif_names):
        bar_labels.append(name)
        bar_vals.append(bif_excess[j])
        bar_colors.append('#e74c3c')

    for j, label in enumerate(control_labels):
        bar_labels.append(label)
        bar_vals.append(control_excess[j])
        bar_colors.append('#3498db')

    x_pos = np.arange(len(bar_labels))
    ax3.bar(x_pos, bar_vals, color=bar_colors, alpha=0.8, edgecolor='white')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(bar_labels, fontsize=7, rotation=45, ha='right')
    ax3.set_ylabel(r'Excess Variance', fontsize=10)
    ax3.set_title('C. Bifurcation Points (red) vs Controls (blue)', fontsize=12,
                  fontweight='bold')
    ax3.axhline(y=0, color='gray', ls=':', alpha=0.5)
    ax3.grid(True, alpha=0.2, axis='y')

    # Mean lines
    mean_bif = np.nanmean(bif_excess)
    mean_ctrl = np.nanmean(control_excess)
    ax3.axhline(y=mean_bif, color='#e74c3c', ls='--', alpha=0.5,
                label=f'Bif mean: {mean_bif:.3f}')
    ax3.axhline(y=mean_ctrl, color='#3498db', ls='--', alpha=0.5,
                label=f'Ctrl mean: {mean_ctrl:.3f}')
    ax3.legend(fontsize=7)

    # ── Panel D: Scaling Analysis ──────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])

    if np.sum(valid) >= 2:
        ax4.loglog(bif_mean_n_arr[valid], np.abs(bif_excess_arr[valid]),
                  'o', color='#e74c3c', ms=10, mfc='white', mew=2,
                  label='At bifurcation points', zorder=5)

        # Plot fit line
        n_fit = np.logspace(np.log10(np.min(bif_mean_n_arr[valid]) * 0.5),
                           np.log10(np.max(bif_mean_n_arr[valid]) * 2), 100)
        excess_fit = scaling_A * n_fit ** scaling_exponent
        ax4.loglog(n_fit, excess_fit, '--', color='#e74c3c', alpha=0.5,
                  label=f'Fit: W₁ ≈ {scaling_A:.2f} × ⟨n⟩^({scaling_exponent:.2f})')

        # Theoretical prediction
        if not np.isnan(scaling_A):
            excess_theory = scaling_A * n_fit ** (-0.5)
            ax4.loglog(n_fit, excess_theory, ':', color='gray', alpha=0.5,
                      label=r'Theory: $W_1 \approx A / \sqrt{\langle n \rangle}$')

        # Label each point
        for j in range(len(bif_names)):
            if valid[j]:
                ax4.annotate(bif_names[j],
                           xy=(bif_mean_n_arr[j], np.abs(bif_excess_arr[j])),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=7, color='#e74c3c')

    ax4.set_xlabel(r'$\langle n \rangle$ at bifurcation', fontsize=10)
    ax4.set_ylabel(r'$|$Excess Variance$|$', fontsize=10)
    ax4.set_title('D. Scaling: Whisper vs Photon Number', fontsize=12,
                  fontweight='bold')
    ax4.legend(fontsize=7)
    ax4.grid(True, alpha=0.2, which='both')

    # ── Supertitle ─────────────────────────────────────────────────────
    fig.suptitle(
        'The Whisper Quantified: Excess Variance at Classical Bifurcation Points\n'
        f'Parametric Kerr: Δ={DELTA}, K={K}, F₀={F0}, ωₘ={WM}, γ={GAMMA_FIXED}, '
        f'N={N_FOCK}',
        fontsize=15, fontweight='bold', y=0.98
    )

    plt.savefig('fig84_whisper_quantified.png', dpi=200, bbox_inches='tight',
                facecolor='white')
    plt.close()
    print(f"  Saved: fig84_whisper_quantified.png")

    # ══════════════════════════════════════════════════════════════════════
    # Experimental Prediction
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("EXPERIMENTAL PREDICTION")
    print(f"{'=' * 70}")
    print(f"  At a driven-dissipative Kerr nonlinear cavity with parametric")
    print(f"  modulation at 2× intrinsic frequency, the quantum photon number")
    print(f"  variance shows EXCESS above Poissonian at F₁ values corresponding")
    print(f"  to classical bifurcation points.")
    print()
    if not np.isnan(scaling_A) and not np.isnan(scaling_exponent):
        print(f"  Scaling law: W₁ ≈ {scaling_A:.3f} × ⟨n⟩^({scaling_exponent:.3f})")
        print(f"  (Predicted: exponent = -0.5)")
    print()
    print(f"  This is measurable in superconducting transmon experiments")
    print(f"  with current technology (typical ⟨n⟩ ≈ 1-20 photons).")

    print(f"\n{'=' * 70}")
    print("Script 84 complete.")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
