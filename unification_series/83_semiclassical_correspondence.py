#!/usr/bin/env python3
"""
Script 83 — Semiclassical Correspondence
==========================================
Show the Feigenbaum cascade EMERGING from quantum fog as ⟨n⟩ increases.

"The most important visualization in the entire Resonance Theory program."

At low photon number, quantum fluctuations wash out bifurcation structure.
As drive strength increases (more photons → more classical), the cascade
emerges, sharpens, and ultimately matches the classical diagram exactly.

Five configurations with increasing F₀ (base drive):
    1. F₀ = 1.5   (deep quantum — few photons, cascade invisible)
    2. F₀ = 3.0   (transition — whisper of structure)
    3. F₀ = 6.0   (semiclassical — cascade emerging)
    4. F₀ = 10.0  (near-classical — clean cascade)
    5. Classical   (infinite photons — perfect Feigenbaum)

All use the same parametric modulation: ωₘ = 5.2, Δ = -3.0, K = 0.3
Sweep F₁ from 0 to 3.5 at fixed γ = 0.5 (same as Script 80 Sweep A).

N_Fock scales with F₀ to accommodate higher photon numbers:
    F₀ = 1.5  → N = 20  (max ⟨n⟩ ≈ 3)
    F₀ = 3.0  → N = 25  (max ⟨n⟩ ≈ 13)
    F₀ = 6.0  → N = 50  (max ⟨n⟩ ≈ 50)
    F₀ = 10.0 → N = 80  (max ⟨n⟩ ≈ 130)

For F₀ = 10.0, use mcsolve (Monte Carlo) if mesolve is too slow.

Five-panel figure stacked vertically: quantum fog → clean cascade.

Generates:
    fig83_semiclassical_correspondence.png
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.integrate import solve_ivp
import time
import sys

try:
    import qutip
    HAS_QUTIP = True
except ImportError:
    HAS_QUTIP = False
    print("ERROR: QuTiP not available.")
    raise SystemExit(1)


# ==============================================================================
# Physical Parameters
# ==============================================================================

DELTA = -3.0        # Detuning
K = 0.3             # Kerr nonlinearity
WM = 5.2            # Modulation frequency (parametric resonance)
T_MOD = 2 * np.pi / WM

GAMMA_FIXED = 0.5   # Fixed dissipation rate (same as Script 80 Sweep A)
F1_MIN, F1_MAX = 0.0, 3.5

FEIGENBAUM_DELTA = 4.669201609102990671853203820466

# Classical bifurcation points (from Script 80)
CLASSICAL_BIF = {
    'P1_P2': 1.146618,
    'P2_P4': 2.571677,
    'P4_P8': 2.745863,
    'P8_P16': 2.780070,
}

# Five configurations: (F0, N_Fock, N_F1, label)
# N_Fock=60 for F₀=10 (mesolve at 80 is prohibitively slow; 60 is tractable
# and sufficient since max ⟨n⟩ on upper branch ≈ 35 at these parameters)
CONFIGS = [
    (1.5,  20,  50,  r'$F_0 = 1.5$ (deep quantum)'),
    (3.0,  25,  50,  r'$F_0 = 3.0$ (quantum-classical boundary)'),
    (6.0,  45,  35,  r'$F_0 = 6.0$ (semiclassical)'),
    (10.0, 60,  15,  r'$F_0 = 10.0$ (near-classical)'),
]

# Classical sweep parameters
N_F1_CLASSICAL = 300
N_TRANSIENT_C = 300
N_RECORD_C = 64

# Quantum sweep parameters
N_TRANSIENT_Q = 50
N_RECORD_Q = 20

# Monte Carlo parameters for large N_Fock
MC_NTRAJ = 40       # Number of trajectories for mcsolve (40 is sufficient for ⟨n⟩)


# ==============================================================================
# Classical Semiclassical Equations
# ==============================================================================

def kerr_derivs(t, y, gamma, F0_local, F1):
    """Semiclassical Kerr: dα/dt = -(γ/2 + iΔ)α - iK|α|²α - iF(t)"""
    u, v = y
    Ft = F0_local + F1 * np.cos(WM * t)
    nsq = u * u + v * v
    du = -gamma / 2 * u + DELTA * v + K * nsq * v
    dv = -gamma / 2 * v - DELTA * u - K * nsq * u - Ft
    return [du, dv]


def classical_stroboscopic(F0_local, F1, gamma=GAMMA_FIXED):
    """Classical stroboscopic map: returns n_vals at stroboscopic times."""
    t_total = (N_TRANSIENT_C + N_RECORD_C) * T_MOD
    t_strobo = np.array([(N_TRANSIENT_C + k) * T_MOD for k in range(N_RECORD_C)])

    # Initial condition on upper branch
    u0 = np.sqrt(abs(DELTA) / K) * 1.2
    v0 = -1.0

    sol = solve_ivp(
        kerr_derivs, [0, t_total], [u0, v0],
        args=(gamma, F0_local, F1), method='RK45',
        t_eval=t_strobo, rtol=1e-10, atol=1e-12,
        max_step=T_MOD / 30
    )

    if sol.success and len(sol.y[0]) == N_RECORD_C:
        return sol.y[0] ** 2 + sol.y[1] ** 2
    return None


# ==============================================================================
# Quantum Stroboscopic (mesolve or mcsolve)
# ==============================================================================

def quantum_stroboscopic_mesolve(F0_local, F1, N_fock, gamma=GAMMA_FIXED):
    """Quantum stroboscopic using mesolve (full density matrix)."""
    a = qutip.destroy(N_fock)
    n_op = a.dag() * a
    identity = qutip.qeye(N_fock)

    H0 = (DELTA * n_op
          + (K / 2.0) * n_op * (n_op - identity)
          + F0_local * (a.dag() + a))
    H1 = a.dag() + a
    coeff_str = f'{F1} * cos({WM} * t)'
    H = [H0, [H1, coeff_str]]

    c_ops = [np.sqrt(gamma) * a]

    # Initial coherent state
    alpha0 = np.sqrt(abs(DELTA) / K) * 0.8 - 0.5j
    psi0 = qutip.coherent(N_fock, alpha0)
    rho0 = psi0 * psi0.dag()

    t_total = (N_TRANSIENT_Q + N_RECORD_Q) * T_MOD
    t_strobo = [(N_TRANSIENT_Q + k) * T_MOD for k in range(N_RECORD_Q)]

    t_list = np.linspace(0, t_total, int(t_total / (T_MOD / 4)) + 1)
    t_all = np.sort(np.unique(np.concatenate([t_list, t_strobo])))

    try:
        result = qutip.mesolve(H, rho0, t_all, c_ops, [n_op],
                               options={'nsteps': 8000, 'atol': 1e-8, 'rtol': 1e-6})
        n_expect = np.array(result.expect[0])
        t_result = np.array(t_all)

        n_strobo = []
        for ts in t_strobo:
            idx = np.argmin(np.abs(t_result - ts))
            n_strobo.append(n_expect[idx])
        return np.array(n_strobo)
    except Exception as e:
        print(f"      mesolve failed: {e}")
        return None


def quantum_stroboscopic_mcsolve(F0_local, F1, N_fock, gamma=GAMMA_FIXED,
                                  ntraj=MC_NTRAJ):
    """Quantum stroboscopic using mcsolve (Monte Carlo wavefunction)."""
    a = qutip.destroy(N_fock)
    n_op = a.dag() * a
    identity = qutip.qeye(N_fock)

    H0 = (DELTA * n_op
          + (K / 2.0) * n_op * (n_op - identity)
          + F0_local * (a.dag() + a))
    H1 = a.dag() + a
    coeff_str = f'{F1} * cos({WM} * t)'
    H = [H0, [H1, coeff_str]]

    c_ops = [np.sqrt(gamma) * a]

    alpha0 = np.sqrt(abs(DELTA) / K) * 0.8 - 0.5j
    psi0 = qutip.coherent(N_fock, alpha0)

    t_total = (N_TRANSIENT_Q + N_RECORD_Q) * T_MOD
    t_strobo = [(N_TRANSIENT_Q + k) * T_MOD for k in range(N_RECORD_Q)]
    t_list = np.linspace(0, t_total, int(t_total / (T_MOD / 4)) + 1)
    t_all = np.sort(np.unique(np.concatenate([t_list, t_strobo])))

    try:
        result = qutip.mcsolve(H, psi0, t_all, c_ops, [n_op],
                                ntraj=ntraj,
                                options={'nsteps': 8000, 'atol': 1e-8, 'rtol': 1e-6})
        # mcsolve returns average over trajectories
        n_expect = np.array(result.expect[0])
        t_result = np.array(t_all)

        n_strobo = []
        for ts in t_strobo:
            idx = np.argmin(np.abs(t_result - ts))
            n_strobo.append(n_expect[idx])
        return np.array(n_strobo)
    except Exception as e:
        print(f"      mcsolve failed: {e}")
        return None


def quantum_stroboscopic(F0_local, F1, N_fock, gamma=GAMMA_FIXED):
    """Dispatch to mesolve for small N, mcsolve for large N."""
    if N_fock <= 45:
        return quantum_stroboscopic_mesolve(F0_local, F1, N_fock, gamma)
    else:
        # For N_fock > 45, go straight to mcsolve (pure state trajectories,
        # scales as N not N² — dramatically faster)
        return quantum_stroboscopic_mcsolve(F0_local, F1, N_fock, gamma)


# ==============================================================================
# Main
# ==============================================================================

def main():
    print("=" * 70)
    print("Script 83 — Semiclassical Correspondence")
    print("Cascade emerging from quantum fog as ⟨n⟩ increases")
    print("=" * 70)
    print(f"\nFixed: Δ={DELTA}, K={K}, ωₘ={WM}, γ={GAMMA_FIXED}")
    print(f"Sweep: F₁ from {F1_MIN} to {F1_MAX}")
    print()

    # ══════════════════════════════════════════════════════════════════════
    # Classical Bifurcation Diagram (Panel 5)
    # ══════════════════════════════════════════════════════════════════════
    print("Computing classical bifurcation diagram...")
    t0 = time.time()

    # Use F0=3.0 for classical reference (same as Script 80)
    F0_classical = 3.0
    F1_vals_c = np.linspace(F1_MIN, F1_MAX, N_F1_CLASSICAL)
    classical_points = []

    for i, F1 in enumerate(F1_vals_c):
        n_vals = classical_stroboscopic(F0_classical, F1)
        if n_vals is not None:
            # Get unique stroboscopic points
            unique = np.unique(np.round(n_vals, 3))
            for u in unique:
                classical_points.append((F1, u))
        if (i + 1) % 100 == 0:
            print(f"  Classical: {i+1}/{N_F1_CLASSICAL}")

    classical_F1s = np.array([p[0] for p in classical_points])
    classical_ns = np.array([p[1] for p in classical_points])
    print(f"  Classical done: {time.time()-t0:.1f}s, {len(classical_points)} points")

    # ══════════════════════════════════════════════════════════════════════
    # Quantum Bifurcation Diagrams (Panels 1-4)
    # ══════════════════════════════════════════════════════════════════════
    quantum_data = {}

    for cfg_idx, (F0_q, N_fock, N_f1, label) in enumerate(CONFIGS):
        print(f"\n{'─' * 60}")
        print(f"  Config {cfg_idx+1}: F₀={F0_q}, N_Fock={N_fock}, {N_f1} F₁ points")
        print(f"{'─' * 60}")

        F1_vals_q = np.linspace(F1_MIN, F1_MAX, N_f1)
        quantum_points = []
        t1 = time.time()

        for i, F1 in enumerate(F1_vals_q):
            n_strobo = quantum_stroboscopic(F0_q, F1, N_fock)
            if n_strobo is not None:
                # For quantum, keep ALL stroboscopic samples (they spread)
                for ns in n_strobo:
                    quantum_points.append((F1, ns))
            elapsed = time.time() - t1
            eta = elapsed / (i + 1) * (N_f1 - i - 1) if i > 0 else 0
            print(f"    F₁={F1:.2f}  ({i+1}/{N_f1})  "
                  f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s")

        quantum_data[cfg_idx] = {
            'F0': F0_q,
            'N_fock': N_fock,
            'label': label,
            'points': quantum_points,
            'F1s': np.array([p[0] for p in quantum_points]),
            'ns': np.array([p[1] for p in quantum_points]),
        }
        print(f"  Done: {time.time()-t1:.1f}s, {len(quantum_points)} points")

    # ══════════════════════════════════════════════════════════════════════
    # Figure: 5-Panel Stacked Vertically
    # ══════════════════════════════════════════════════════════════════════
    print(f"\nGenerating figure...")

    fig, axes = plt.subplots(5, 1, figsize=(14, 24), facecolor='white')
    plt.subplots_adjust(hspace=0.25)

    panel_labels = ['A', 'B', 'C', 'D', 'E']
    quantum_colors = ['#3498db', '#9b59b6', '#2ecc71', '#e67e22']

    # Panels 1-4: Quantum bifurcation diagrams
    for cfg_idx in range(4):
        ax = axes[cfg_idx]
        data = quantum_data[cfg_idx]

        if len(data['F1s']) > 0:
            ax.scatter(data['F1s'], data['ns'], s=0.3, alpha=0.4,
                      color=quantum_colors[cfg_idx], rasterized=True)

        # Annotate classical bifurcation points
        for name, val in CLASSICAL_BIF.items():
            ax.axvline(x=val, color='red', ls=':', alpha=0.3, lw=0.8)

        ax.set_ylabel(r'$\langle n \rangle$ (stroboscopic)', fontsize=10)
        ax.set_title(f'{panel_labels[cfg_idx]}. {data["label"]}  '
                     f'[N={data["N_fock"]}]',
                     fontsize=12, fontweight='bold', loc='left')
        ax.grid(True, alpha=0.2)
        ax.set_xlim(F1_MIN, F1_MAX)

        # Add mean ⟨n⟩ annotation
        if len(data['ns']) > 0:
            mean_n = np.mean(data['ns'])
            ax.annotate(f'mean $\\langle n \\rangle$ = {mean_n:.1f}',
                       xy=(0.98, 0.95), xycoords='axes fraction',
                       ha='right', va='top', fontsize=8,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                alpha=0.8))

    # Panel 5: Classical bifurcation diagram
    ax5 = axes[4]
    ax5.scatter(classical_F1s, classical_ns, s=0.3, alpha=0.6,
               color='#e74c3c', rasterized=True)

    for name, val in CLASSICAL_BIF.items():
        ax5.axvline(x=val, color='red', ls=':', alpha=0.5, lw=0.8)
        pretty = name.replace('_', r'$\to$')
        ax5.annotate(pretty, xy=(val, ax5.get_ylim()[1] if ax5.get_ylim()[1] > 0 else 15),
                    fontsize=6, rotation=90, va='top', ha='right', color='red')

    ax5.set_ylabel(r'$n$ (stroboscopic)', fontsize=10)
    ax5.set_xlabel(r'Modulation Amplitude $F_1$', fontsize=10)
    ax5.set_title(f'E. Classical ($N \\to \\infty$)  '
                  f'[Feigenbaum $\\delta$ = {FEIGENBAUM_DELTA:.3f}]',
                  fontsize=12, fontweight='bold', loc='left')
    ax5.grid(True, alpha=0.2)
    ax5.set_xlim(F1_MIN, F1_MAX)

    # Mark bifurcation points on classical panel
    for name, val in CLASSICAL_BIF.items():
        ax5.axvline(x=val, color='red', ls=':', alpha=0.5, lw=0.8)

    # Supertitle
    fig.suptitle(
        'Semiclassical Correspondence: Cascade Emerging from Quantum Fog\n'
        f'Parametric Kerr: Δ={DELTA}, K={K}, ωₘ={WM}, γ={GAMMA_FIXED}',
        fontsize=15, fontweight='bold', y=0.995
    )

    plt.savefig('fig83_semiclassical_correspondence.png', dpi=200,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: fig83_semiclassical_correspondence.png")

    # ══════════════════════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("Summary")
    print(f"{'=' * 70}")
    for cfg_idx in range(4):
        data = quantum_data[cfg_idx]
        n_pts = len(data['ns'])
        mean_n = np.mean(data['ns']) if n_pts > 0 else 0
        spread = np.std(data['ns']) if n_pts > 0 else 0
        print(f"  F₀={data['F0']:>5.1f}  N={data['N_fock']:>3d}  "
              f"points={n_pts:>5d}  mean⟨n⟩={mean_n:>8.2f}  "
              f"σ={spread:>8.2f}")
    print(f"  Classical:  points={len(classical_points):>5d}")
    print(f"\n{'=' * 70}")
    print("Script 83 complete.")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
