#!/usr/bin/env python3
"""
Script 82 — N-Convergence of the Quantum Phase Transition
==========================================================
Prove that quantum observables at the driven-dissipative Kerr phase transition
are STABLE as the Hilbert space dimension N → ∞.

Uses the SAME rotating-frame Hamiltonian as Script 79:
    H = Δ a†a + (K/2) a†a(a†a - 1) + F(a† + a)
    L = √γ · a

Parameters: Δ = -3.0, K = 0.3, F = 3.0

Sweep: γ from 0.1 to 3.0 (100 points) at SEVEN values of N_Fock:
    15, 20, 25, 30, 35, 40, 50

At each N, identify the quantum phase transition γ_q(N) as the γ value
where d⟨n⟩/dγ is most negative (steepest drop = transition point).

Five observables recorded at transition:
    1. γ_q(N)       — transition location
    2. Var(n)        — photon number variance at transition
    3. Tr(ρ²)        — purity at transition
    4. Fano factor   — Var(n)/⟨n⟩ at transition
    5. ⟨n⟩           — mean photon number just below transition

Kill criterion: ALL five observables change < 2% between N=40 and N=50.

Generates:
    fig82_n_convergence.png  — 6-panel convergence analysis
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import time

try:
    import qutip
    HAS_QUTIP = True
except ImportError:
    HAS_QUTIP = False
    print("ERROR: QuTiP not available. Install with: pip install qutip")
    raise SystemExit(1)


# ==============================================================================
# Physical Parameters — IDENTICAL to Script 79
# ==============================================================================

DELTA = -3.0        # Detuning (red-detuned for bistability)
K = 0.3             # Kerr nonlinearity
F = 3.0             # Coherent drive amplitude

# Sweep parameters
GAMMA_MIN = 0.1
GAMMA_MAX = 3.0
N_GAMMA = 100       # Coarser than Script 79 (speed — 7 full sweeps)

# Seven Hilbert space dimensions
N_FOCK_VALUES = [15, 20, 25, 30, 35, 40, 50]


# ==============================================================================
# Build Quantum Operators
# ==============================================================================

def build_system(N: int):
    """Build operators for the Kerr oscillator in the rotating frame."""
    a = qutip.destroy(N)
    n_op = a.dag() * a
    n2_op = n_op * n_op
    identity = qutip.qeye(N)

    # H = Δ a†a + (K/2) a†a(a†a - 1) + F(a† + a)
    H = (DELTA * n_op
         + (K / 2.0) * n_op * (n_op - identity)
         + F * (a.dag() + a))

    return a, n_op, n2_op, H


# ==============================================================================
# Steady State Computation
# ==============================================================================

def compute_steady_state(gamma: float, a, H, n_op, n2_op):
    """
    Compute steady-state density matrix and extract observables.

    Returns: mean_n, var_n, purity, fano
    """
    c_ops = [np.sqrt(gamma) * a]

    try:
        rho_ss = qutip.steadystate(H, c_ops)
    except Exception:
        try:
            rho_ss = qutip.steadystate(H, c_ops, method='direct')
        except Exception:
            return np.nan, np.nan, np.nan, np.nan

    mean_n = float(np.real(qutip.expect(n_op, rho_ss)))
    mean_n2 = float(np.real(qutip.expect(n2_op, rho_ss)))
    var_n = mean_n2 - mean_n ** 2
    purity = float(np.real((rho_ss * rho_ss).tr()))

    # Fano factor = Var(n) / ⟨n⟩
    fano = var_n / mean_n if mean_n > 1e-10 else np.nan

    return mean_n, var_n, purity, fano


# ==============================================================================
# Find Transition Point
# ==============================================================================

def find_transition(gammas: np.ndarray, mean_ns: np.ndarray):
    """
    Locate the quantum phase transition as the γ where d⟨n⟩/dγ is most negative.
    Returns: (idx_transition, gamma_q)
    """
    # Numerical gradient
    dndg = np.gradient(mean_ns, gammas)

    # Find steepest negative slope
    idx = np.nanargmin(dndg)

    return idx, gammas[idx]


# ==============================================================================
# Main Computation
# ==============================================================================

def main():
    print("=" * 70)
    print("Script 82 — N-Convergence of the Quantum Phase Transition")
    print("Hilbert space dimension convergence test")
    print("=" * 70)
    print(f"\nRotating-frame parameters:")
    print(f"  Δ = {DELTA}   (detuning)")
    print(f"  K = {K}    (Kerr nonlinearity)")
    print(f"  F = {F}    (drive amplitude)")
    print(f"  γ = {GAMMA_MIN} to {GAMMA_MAX}  ({N_GAMMA} points)")
    print(f"  N_Fock values: {N_FOCK_VALUES}")
    print()

    gammas = np.linspace(GAMMA_MIN, GAMMA_MAX, N_GAMMA)

    # Storage for transition observables at each N
    gamma_q_vals = []       # transition location
    var_at_trans = []       # Var(n) at transition
    purity_at_trans = []    # Tr(ρ²) at transition
    fano_at_trans = []      # Fano factor at transition
    mean_n_at_trans = []    # ⟨n⟩ just below transition

    # Also store full sweep curves for overlay plot
    all_mean_n = {}
    all_var_n = {}

    t_total = time.time()

    for N in N_FOCK_VALUES:
        t_start = time.time()
        print(f"\n{'─' * 50}")
        print(f"  N_Fock = {N}")
        print(f"{'─' * 50}")

        a, n_op, n2_op, H = build_system(N)

        mean_ns = np.zeros(N_GAMMA)
        var_ns = np.zeros(N_GAMMA)
        purity_s = np.zeros(N_GAMMA)
        fano_s = np.zeros(N_GAMMA)

        for i, gamma in enumerate(gammas):
            mn, vn, pu, fa = compute_steady_state(gamma, a, H, n_op, n2_op)
            mean_ns[i] = mn
            var_ns[i] = vn
            purity_s[i] = pu
            fano_s[i] = fa

            if (i + 1) % 20 == 0:
                print(f"    γ = {gamma:.2f}  |  ⟨n⟩ = {mn:.4f}  |  "
                      f"Var = {vn:.4f}  |  Purity = {pu:.4f}")

        # Store full curves
        all_mean_n[N] = mean_ns.copy()
        all_var_n[N] = var_ns.copy()

        # Find transition
        idx_trans, gq = find_transition(gammas, mean_ns)

        gamma_q_vals.append(gq)
        var_at_trans.append(var_ns[idx_trans])
        purity_at_trans.append(purity_s[idx_trans])
        fano_at_trans.append(fano_s[idx_trans])

        # ⟨n⟩ just below transition (2 points before)
        idx_below = max(0, idx_trans - 2)
        mean_n_at_trans.append(mean_ns[idx_below])

        elapsed = time.time() - t_start
        print(f"\n  γ_q = {gq:.4f}")
        print(f"  Var(n) at transition = {var_ns[idx_trans]:.4f}")
        print(f"  Purity at transition = {purity_s[idx_trans]:.4f}")
        print(f"  Fano at transition = {fano_s[idx_trans]:.4f}")
        print(f"  ⟨n⟩ below transition = {mean_ns[idx_below]:.4f}")
        print(f"  Time: {elapsed:.1f}s")

    total_time = time.time() - t_total
    print(f"\n{'=' * 70}")
    print(f"Total computation time: {total_time:.1f}s")

    # Convert to arrays
    N_vals = np.array(N_FOCK_VALUES, dtype=float)
    gamma_q_arr = np.array(gamma_q_vals)
    var_arr = np.array(var_at_trans)
    purity_arr = np.array(purity_at_trans)
    fano_arr = np.array(fano_at_trans)
    mean_n_arr = np.array(mean_n_at_trans)

    # ══════════════════════════════════════════════════════════════════════
    # Kill Criterion: < 2% change between N=40 and N=50
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("KILL CRITERION: All observables change < 2% from N=40 to N=50")
    print(f"{'=' * 70}")

    idx_40 = N_FOCK_VALUES.index(40)
    idx_50 = N_FOCK_VALUES.index(50)

    labels = ["γ_q", "Var(n)", "Purity", "Fano", "⟨n⟩"]
    vals_40 = [gamma_q_arr[idx_40], var_arr[idx_40], purity_arr[idx_40],
               fano_arr[idx_40], mean_n_arr[idx_40]]
    vals_50 = [gamma_q_arr[idx_50], var_arr[idx_50], purity_arr[idx_50],
               fano_arr[idx_50], mean_n_arr[idx_50]]

    all_pass = True
    for label, v40, v50 in zip(labels, vals_40, vals_50):
        if abs(v40) > 1e-10:
            pct = abs(v50 - v40) / abs(v40) * 100
        else:
            pct = 0.0 if abs(v50 - v40) < 1e-10 else 100.0
        status = "PASS" if pct < 2.0 else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"  {label:>8s}: N=40 → {v40:.6f}  |  N=50 → {v50:.6f}  |  "
              f"Δ = {pct:.3f}%  [{status}]")

    print(f"\n  Overall: {'ALL PASS — Convergence confirmed' if all_pass else 'SOME FAIL — Increase N_Fock range'}")

    # ══════════════════════════════════════════════════════════════════════
    # Figure: 6-Panel Convergence Analysis
    # ══════════════════════════════════════════════════════════════════════
    print(f"\nGenerating figure...")

    fig = plt.figure(figsize=(18, 14), facecolor='white')
    gs = GridSpec(2, 3, hspace=0.35, wspace=0.3)

    colors_N = ['#3498db', '#2ecc71', '#e67e22', '#9b59b6',
                '#e74c3c', '#1abc9c', '#34495e']
    marker_styles = ['o', 's', 'D', '^', 'v', 'P', 'X']

    # ── Panel A: γ_q vs N ─────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(N_vals, gamma_q_arr, 'o-', color='#e74c3c', lw=2, ms=8,
             mfc='white', mew=2, zorder=5)

    # Asymptotic line from last two points
    if len(gamma_q_arr) >= 2:
        ax1.axhline(y=gamma_q_arr[-1], color='#e74c3c', ls='--', alpha=0.4,
                     label=f'N=50 value: {gamma_q_arr[-1]:.4f}')

    ax1.set_xlabel('Hilbert Space Dimension N', fontsize=10)
    ax1.set_ylabel(r'$\gamma_q$ (transition location)', fontsize=10)
    ax1.set_title('A. Transition Location Convergence', fontsize=12,
                  fontweight='bold')
    ax1.legend(fontsize=7, loc='best')
    ax1.grid(True, alpha=0.2)

    # ── Panel B: Var(n) at transition vs N ─────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(N_vals, var_arr, 's-', color='#3498db', lw=2, ms=8,
             mfc='white', mew=2, zorder=5)
    ax2.axhline(y=var_arr[-1], color='#3498db', ls='--', alpha=0.4,
                 label=f'N=50 value: {var_arr[-1]:.4f}')
    ax2.set_xlabel('Hilbert Space Dimension N', fontsize=10)
    ax2.set_ylabel(r'Var$(n)$ at transition', fontsize=10)
    ax2.set_title('B. Variance Convergence', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=7, loc='best')
    ax2.grid(True, alpha=0.2)

    # ── Panel C: Purity at transition vs N ─────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(N_vals, purity_arr, 'D-', color='#2ecc71', lw=2, ms=8,
             mfc='white', mew=2, zorder=5)
    ax3.axhline(y=purity_arr[-1], color='#2ecc71', ls='--', alpha=0.4,
                 label=f'N=50 value: {purity_arr[-1]:.6f}')
    ax3.set_xlabel('Hilbert Space Dimension N', fontsize=10)
    ax3.set_ylabel(r'Tr$(\rho^2)$ at transition', fontsize=10)
    ax3.set_title('C. Purity Convergence', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=7, loc='best')
    ax3.grid(True, alpha=0.2)

    # ── Panel D: Fano factor at transition vs N ────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(N_vals, fano_arr, '^-', color='#9b59b6', lw=2, ms=8,
             mfc='white', mew=2, zorder=5)
    ax4.axhline(y=fano_arr[-1], color='#9b59b6', ls='--', alpha=0.4,
                 label=f'N=50 value: {fano_arr[-1]:.4f}')
    ax4.axhline(y=1.0, color='gray', ls=':', alpha=0.5, label='Poissonian (F=1)')
    ax4.set_xlabel('Hilbert Space Dimension N', fontsize=10)
    ax4.set_ylabel('Fano Factor at transition', fontsize=10)
    ax4.set_title('D. Fano Factor Convergence', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=7, loc='best')
    ax4.grid(True, alpha=0.2)

    # ── Panel E: ⟨n⟩ below transition vs N ─────────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(N_vals, mean_n_arr, 'v-', color='#e67e22', lw=2, ms=8,
             mfc='white', mew=2, zorder=5)
    ax5.axhline(y=mean_n_arr[-1], color='#e67e22', ls='--', alpha=0.4,
                 label=f'N=50 value: {mean_n_arr[-1]:.4f}')
    ax5.set_xlabel('Hilbert Space Dimension N', fontsize=10)
    ax5.set_ylabel(r'$\langle n \rangle$ below transition', fontsize=10)
    ax5.set_title(r'E. $\langle n \rangle$ Convergence', fontsize=12,
                  fontweight='bold')
    ax5.legend(fontsize=7, loc='best')
    ax5.grid(True, alpha=0.2)

    # ── Panel F: Convergence Metric (% change from N-1 to N) ──────────
    ax6 = fig.add_subplot(gs[1, 2])

    # Compute % change between successive N values
    obs_names = [r'$\gamma_q$', r'Var$(n)$', r'Tr$(\rho^2)$',
                 'Fano', r'$\langle n \rangle$']
    obs_arrays = [gamma_q_arr, var_arr, purity_arr, fano_arr, mean_n_arr]
    obs_colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#e67e22']

    for name, arr, col in zip(obs_names, obs_arrays, obs_colors):
        pct_changes = []
        for i in range(1, len(arr)):
            if abs(arr[i - 1]) > 1e-10:
                pct = abs(arr[i] - arr[i - 1]) / abs(arr[i - 1]) * 100
            else:
                pct = 0.0
            pct_changes.append(pct)
        ax6.semilogy(N_vals[1:], pct_changes, 'o-', color=col, lw=1.5,
                     ms=6, label=name)

    ax6.axhline(y=2.0, color='red', ls='--', lw=2, alpha=0.7,
                 label='2% threshold')
    ax6.set_xlabel('Hilbert Space Dimension N', fontsize=10)
    ax6.set_ylabel('% Change from Previous N', fontsize=10)
    ax6.set_title('F. Convergence Metric', fontsize=12, fontweight='bold')
    ax6.legend(fontsize=7, loc='best', ncol=2)
    ax6.grid(True, alpha=0.2)
    ax6.set_ylim(bottom=1e-3)

    # ── Supertitle ─────────────────────────────────────────────────────
    status_str = "ALL CONVERGED" if all_pass else "CONVERGENCE INCOMPLETE"
    fig.suptitle(
        f'N-Convergence of the Quantum Phase Transition\n'
        f'Kerr oscillator: Δ={DELTA}, K={K}, F={F}  |  '
        f'N ∈ {{{", ".join(str(n) for n in N_FOCK_VALUES)}}}  |  '
        f'{status_str}',
        fontsize=15, fontweight='bold', y=0.98
    )

    plt.savefig('fig82_n_convergence.png', dpi=200, bbox_inches='tight',
                facecolor='white')
    plt.close()
    print(f"  Saved: fig82_n_convergence.png")

    # ══════════════════════════════════════════════════════════════════════
    # Summary Table
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("Summary Table")
    print(f"{'=' * 70}")
    print(f"{'N':>5s}  {'γ_q':>10s}  {'Var(n)':>10s}  {'Purity':>10s}  "
          f"{'Fano':>10s}  {'⟨n⟩':>10s}")
    print(f"{'─' * 5}  {'─' * 10}  {'─' * 10}  {'─' * 10}  {'─' * 10}  {'─' * 10}")
    for i, N in enumerate(N_FOCK_VALUES):
        print(f"{N:>5d}  {gamma_q_arr[i]:>10.6f}  {var_arr[i]:>10.6f}  "
              f"{purity_arr[i]:>10.6f}  {fano_arr[i]:>10.6f}  {mean_n_arr[i]:>10.6f}")

    print(f"\n{'=' * 70}")
    print("Script 82 complete.")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
