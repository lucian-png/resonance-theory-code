#!/usr/bin/env python3
"""
Script 80 — The Quantum Kerr Cascade: Parametric Drive
=======================================================
Add time-periodic modulation to the Kerr oscillator to BREAK the
time-translation symmetry of the rotating frame, enabling period-doubling.

H(t) = Δ a†a + (K/2) a†a(a†a - 1) + [F₀ + F₁cos(ωₘt)](a† + a)
L = √γ · a

Key discovery from classical exploration:
  - ωₘ = 5.2 (≈ 2× intrinsic oscillation frequency ω_int ≈ 2.6)
  - This satisfies the PARAMETRIC RESONANCE condition for period-doubling
  - ωₘ = 2.0 (originally specified) gives NO period-doubling (off-resonance)

Classical cascade at ωₘ = 5.2, γ = 0.5:
  P1→P2:  F₁ = 1.1466
  P2→P4:  F₁ = 2.5717
  P4→P8:  F₁ = 2.7459
  P8→P16: F₁ = 2.7801
  δ₂ = 5.09 (9% error → converging to 4.669)

The question: does the period-doubling cascade SURVIVE in the full quantum
density matrix? Even ONE quantum period-doubling would confirm Feigenbaum
universality extends into quantum mechanics.

Sweep A: Fix γ=0.5, sweep F₁ from 0 to 3.5 — drives through the cascade
Sweep B: Fix F₁=2.0, sweep γ from 0.1 to 3.0 — tests decoherence killing cascade

Eight-panel figure:
  A: Classical bifurcation diagram (Sweep A) — stroboscopic ⟨n⟩ vs F₁
  B: Quantum bifurcation diagram (Sweep A) — stroboscopic ⟨n⟩ vs F₁
  C: Classical period detection — color-coded period vs F₁
  D: Quantum period detection — color-coded period vs F₁
  E: Classical bifurcation diagram (Sweep B) — ⟨n⟩ vs γ
  F: Quantum bifurcation diagram (Sweep B) — ⟨n⟩ vs γ
  G: δ extraction — classical vs quantum ratios
  H: Wigner snapshots at cascade points
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
    print("WARNING: QuTiP not available. Running classical only.")

# ==============================================================================
# Physical Parameters
# ==============================================================================

DELTA = -3.0        # Detuning (red-detuned for bistability)
K = 0.3             # Kerr nonlinearity
F0 = 3.0            # Static drive amplitude
WM = 5.2            # Modulation frequency (≈ 2× intrinsic freq for parametric resonance)
T_MOD = 2 * np.pi / WM  # Modulation period

N_FOCK = 25         # Fock space truncation (max ⟨n⟩ ≈ 13, so 25 is sufficient)

FEIGENBAUM_DELTA = 4.669201609102990671853203820466

# Sweep A: F₁ sweep at fixed γ
GAMMA_FIXED = 0.5
F1_MIN, F1_MAX = 0.0, 3.5
N_F1_CLASSICAL = 400       # Dense classical sweep
N_F1_QUANTUM = 60           # Sparse quantum sweep (mesolve is slow)

# Sweep B: γ sweep at fixed F₁
F1_FIXED = 2.0
GAMMA_MIN_B, GAMMA_MAX_B = 0.1, 3.0
N_GAMMA_CLASSICAL = 300
N_GAMMA_QUANTUM = 40

# Stroboscopic parameters
N_TRANSIENT = 300       # Modulation periods to discard (classical)
N_RECORD = 64           # Modulation periods to record (classical)
N_TRANSIENT_Q = 80      # Fewer for quantum (slow)
N_RECORD_Q = 32         # Fewer for quantum

# Classical bifurcation points (from refinement)
CLASSICAL_BIF = {
    'P1_P2': 1.146618,
    'P2_P4': 2.571677,
    'P4_P8': 2.745863,
    'P8_P16': 2.780070,
}


# ==============================================================================
# Classical Semiclassical Equations
# ==============================================================================

def kerr_derivs(t, y, gamma, F1):
    """
    Semiclassical Kerr oscillator: dα/dt = -(γ/2 + iΔ)α - iK|α|²α - iF(t)
    Decomposed into real/imaginary parts: α = u + iv
    """
    u, v = y
    Ft = F0 + F1 * np.cos(WM * t)
    nsq = u * u + v * v
    du = -gamma / 2 * u + DELTA * v + K * nsq * v
    dv = -gamma / 2 * v - DELTA * u - K * nsq * u - Ft
    return [du, dv]


def classical_stroboscopic(F1, gamma, u0=3.5, v0=-1.0,
                           n_trans=N_TRANSIENT, n_rec=N_RECORD):
    """
    Integrate classical Kerr equations stroboscopically.
    Returns array of ⟨n⟩ = u² + v² sampled at t = k·T_mod.
    """
    t_total = (n_trans + n_rec) * T_MOD
    t_strobo = np.array([(n_trans + k) * T_MOD for k in range(n_rec)])

    sol = solve_ivp(
        kerr_derivs, [0, t_total], [u0, v0],
        args=(gamma, F1), method='RK45',
        t_eval=t_strobo, rtol=1e-10, atol=1e-12,
        max_step=T_MOD / 30
    )

    if sol.success and len(sol.y[0]) == n_rec:
        n_vals = sol.y[0] ** 2 + sol.y[1] ** 2
        return n_vals
    return None


def detect_period(n_vals, tol=0.02):
    """Detect period from stroboscopic samples."""
    if n_vals is None or len(n_vals) < 8:
        return 0
    for p in [1, 2, 4, 8, 16, 32]:
        if len(n_vals) < 3 * p:
            break
        last = np.sort(n_vals[-p:])
        prev = np.sort(n_vals[-2 * p:-p])
        if np.max(np.abs(last - prev)) < tol:
            return p
    return 0  # chaos or quasiperiodic


# ==============================================================================
# Quantum Stroboscopic (QuTiP mesolve)
# ==============================================================================

def quantum_stroboscopic(F1, gamma, n_trans=N_TRANSIENT_Q, n_rec=N_RECORD_Q):
    """
    Quantum stroboscopic map using QuTiP mesolve with time-dependent H.
    Returns array of ⟨n⟩ sampled at t = k·T_mod after transient.
    """
    if not HAS_QUTIP:
        return None

    a = qutip.destroy(N_FOCK)
    n_op = a.dag() * a
    identity = qutip.qeye(N_FOCK)

    # Static part: H0 = Δ n + (K/2) n(n-1) + F₀(a† + a)
    H0 = (DELTA * n_op
          + (K / 2.0) * n_op * (n_op - identity)
          + F0 * (a.dag() + a))

    # Time-dependent part: H1 = F₁cos(ωₘt) · (a† + a)
    H1 = a.dag() + a
    coeff_str = f'{F1} * cos({WM} * t)'
    H = [H0, [H1, coeff_str]]

    c_ops = [np.sqrt(gamma) * a]

    # Initial state: coherent state near upper branch
    alpha0 = 3.5 - 1.0j
    psi0 = qutip.coherent(N_FOCK, alpha0)
    rho0 = psi0 * psi0.dag()

    # Stroboscopic times
    t_total = (n_trans + n_rec) * T_MOD
    t_strobo = [(n_trans + k) * T_MOD for k in range(n_rec)]

    # Also need intermediate times for solver stability
    t_list = np.linspace(0, t_total, int(t_total / (T_MOD / 4)) + 1)
    # Merge stroboscopic times
    t_all = np.sort(np.unique(np.concatenate([t_list, t_strobo])))

    try:
        result = qutip.mesolve(H, rho0, t_all, c_ops, [n_op],
                               options={'nsteps': 5000, 'atol': 1e-8, 'rtol': 1e-6})
        # Extract ⟨n⟩ at stroboscopic times
        n_expect = np.array(result.expect[0])
        t_result = np.array(t_all)

        n_strobo = []
        for ts in t_strobo:
            idx = np.argmin(np.abs(t_result - ts))
            n_strobo.append(n_expect[idx])
        return np.array(n_strobo)
    except Exception as e:
        print(f"    mesolve failed: {e}")
        return None


def quantum_stroboscopic_for_wigner(F1, gamma, n_trans=N_TRANSIENT_Q):
    """Get density matrix at a stroboscopic time for Wigner computation."""
    if not HAS_QUTIP:
        return None

    a = qutip.destroy(N_FOCK)
    n_op = a.dag() * a
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

    t_final = n_trans * T_MOD
    t_list = np.linspace(0, t_final, int(t_final / (T_MOD / 4)) + 1)

    try:
        result = qutip.mesolve(H, rho0, t_list, c_ops, [],
                               options={'nsteps': 5000, 'atol': 1e-8, 'rtol': 1e-6})
        return result.states[-1]
    except Exception as e:
        print(f"    mesolve (Wigner) failed: {e}")
        return None


# ==============================================================================
# Main
# ==============================================================================

def main():
    print("=" * 70)
    print("Script 80 — Quantum Kerr Cascade: Parametric Drive")
    print("Period-doubling cascade in driven-dissipative quantum system")
    print("=" * 70)
    print(f"\n  H(t) = Δ a†a + (K/2) a†a(a†a-1) + [F₀ + F₁cos(ωₘt)](a† + a)")
    print(f"  L = √γ · a")
    print(f"\n  Δ = {DELTA},  K = {K},  F₀ = {F0}")
    print(f"  ωₘ = {WM}  (T = {T_MOD:.4f})")
    print(f"  N_Fock = {N_FOCK}")
    print(f"\n  Classical bifurcation points (from refinement):")
    for label, val in CLASSICAL_BIF.items():
        print(f"    {label}: F₁ = {val:.6f}")
    d1 = (CLASSICAL_BIF['P2_P4'] - CLASSICAL_BIF['P1_P2']) / \
         (CLASSICAL_BIF['P4_P8'] - CLASSICAL_BIF['P2_P4'])
    d2 = (CLASSICAL_BIF['P4_P8'] - CLASSICAL_BIF['P2_P4']) / \
         (CLASSICAL_BIF['P8_P16'] - CLASSICAL_BIF['P4_P8'])
    print(f"  δ₁ = {d1:.2f}  (target 4.669)")
    print(f"  δ₂ = {d2:.2f}  (target 4.669)")

    # ==================================================================
    # SWEEP A: Classical F₁ sweep (γ = 0.5)
    # ==================================================================
    print(f"\n{'='*70}")
    print(f"SWEEP A — Classical: F₁ sweep, γ = {GAMMA_FIXED}")
    print(f"{'='*70}")

    F1_vals_cl = np.linspace(F1_MIN, F1_MAX, N_F1_CLASSICAL)
    bif_data_cl_A = []  # (F1, [n_strobo values])
    period_cl_A = np.zeros(N_F1_CLASSICAL, dtype=int)

    t0 = time.time()
    for i, F1 in enumerate(F1_vals_cl):
        n_vals = classical_stroboscopic(F1, GAMMA_FIXED)
        if n_vals is not None:
            bif_data_cl_A.append((F1, n_vals))
            period_cl_A[i] = detect_period(n_vals)
        else:
            bif_data_cl_A.append((F1, np.array([])))

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{N_F1_CLASSICAL}  F₁={F1:.3f}  "
                  f"period={period_cl_A[i]}  [{elapsed:.1f}s]")

    t_cl_A = time.time() - t0
    print(f"  Classical Sweep A complete: {t_cl_A:.1f}s")

    # Summarize periods found
    for p in [1, 2, 4, 8, 16]:
        count = np.sum(period_cl_A == p)
        if count > 0:
            idxs = np.where(period_cl_A == p)[0]
            f1_range = (F1_vals_cl[idxs[0]], F1_vals_cl[idxs[-1]])
            print(f"  Period-{p}: {count} points, F₁ ∈ [{f1_range[0]:.3f}, {f1_range[1]:.3f}]")
    chaos_count = np.sum(period_cl_A == 0)
    print(f"  Chaos/QP: {chaos_count} points")

    # ==================================================================
    # SWEEP A: Quantum F₁ sweep (γ = 0.5)
    # ==================================================================
    if HAS_QUTIP:
        print(f"\n{'='*70}")
        print(f"SWEEP A — Quantum: F₁ sweep, γ = {GAMMA_FIXED}")
        print(f"{'='*70}")

        # Focus quantum sweep on the cascade region
        F1_vals_q = np.linspace(0.0, 3.5, N_F1_QUANTUM)
        bif_data_q_A = []
        period_q_A = np.zeros(N_F1_QUANTUM, dtype=int)

        t0_q = time.time()
        for i, F1 in enumerate(F1_vals_q):
            print(f"  [{i+1}/{N_F1_QUANTUM}] F₁ = {F1:.3f} ...", end="", flush=True)
            n_vals = quantum_stroboscopic(F1, GAMMA_FIXED)
            if n_vals is not None:
                bif_data_q_A.append((F1, n_vals))
                period_q_A[i] = detect_period(n_vals, tol=0.05)
                mean_n = np.mean(n_vals[-8:])
                spread = np.max(n_vals[-8:]) - np.min(n_vals[-8:])
                print(f" ⟨n⟩={mean_n:.2f}, spread={spread:.3f}, "
                      f"P={period_q_A[i]}, [{time.time()-t0_q:.0f}s]")
            else:
                bif_data_q_A.append((F1, np.array([])))
                print(f" FAILED")

        t_q_A = time.time() - t0_q
        print(f"  Quantum Sweep A complete: {t_q_A:.1f}s")

        for p in [1, 2, 4, 8]:
            count = np.sum(period_q_A == p)
            if count > 0:
                idxs = np.where(period_q_A == p)[0]
                f1_range = (F1_vals_q[idxs[0]], F1_vals_q[idxs[-1]])
                print(f"  Period-{p}: {count} points, "
                      f"F₁ ∈ [{f1_range[0]:.3f}, {f1_range[1]:.3f}]")
    else:
        F1_vals_q = np.array([])
        bif_data_q_A = []
        period_q_A = np.array([])

    # ==================================================================
    # SWEEP B: Classical γ sweep (F₁ = 2.0)
    # ==================================================================
    print(f"\n{'='*70}")
    print(f"SWEEP B — Classical: γ sweep, F₁ = {F1_FIXED}")
    print(f"{'='*70}")

    gamma_vals_cl = np.linspace(GAMMA_MIN_B, GAMMA_MAX_B, N_GAMMA_CLASSICAL)
    bif_data_cl_B = []
    period_cl_B = np.zeros(N_GAMMA_CLASSICAL, dtype=int)

    t0 = time.time()
    for i, gamma in enumerate(gamma_vals_cl):
        n_vals = classical_stroboscopic(F1_FIXED, gamma)
        if n_vals is not None:
            bif_data_cl_B.append((gamma, n_vals))
            period_cl_B[i] = detect_period(n_vals)
        else:
            bif_data_cl_B.append((gamma, np.array([])))

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{N_GAMMA_CLASSICAL}  γ={gamma:.3f}  "
                  f"period={period_cl_B[i]}  [{elapsed:.1f}s]")

    t_cl_B = time.time() - t0
    print(f"  Classical Sweep B complete: {t_cl_B:.1f}s")

    for p in [1, 2, 4, 8, 16]:
        count = np.sum(period_cl_B == p)
        if count > 0:
            idxs = np.where(period_cl_B == p)[0]
            g_range = (gamma_vals_cl[idxs[0]], gamma_vals_cl[idxs[-1]])
            print(f"  Period-{p}: {count} points, "
                  f"γ ∈ [{g_range[0]:.3f}, {g_range[1]:.3f}]")

    # ==================================================================
    # SWEEP B: Quantum γ sweep (F₁ = 2.0)
    # ==================================================================
    if HAS_QUTIP:
        print(f"\n{'='*70}")
        print(f"SWEEP B — Quantum: γ sweep, F₁ = {F1_FIXED}")
        print(f"{'='*70}")

        gamma_vals_q = np.linspace(GAMMA_MIN_B, GAMMA_MAX_B, N_GAMMA_QUANTUM)
        bif_data_q_B = []
        period_q_B = np.zeros(N_GAMMA_QUANTUM, dtype=int)

        t0_q = time.time()
        for i, gamma in enumerate(gamma_vals_q):
            print(f"  [{i+1}/{N_GAMMA_QUANTUM}] γ = {gamma:.3f} ...",
                  end="", flush=True)
            n_vals = quantum_stroboscopic(F1_FIXED, gamma)
            if n_vals is not None:
                bif_data_q_B.append((gamma, n_vals))
                period_q_B[i] = detect_period(n_vals, tol=0.05)
                mean_n = np.mean(n_vals[-8:])
                spread = np.max(n_vals[-8:]) - np.min(n_vals[-8:])
                print(f" ⟨n⟩={mean_n:.2f}, spread={spread:.3f}, "
                      f"P={period_q_B[i]}, [{time.time()-t0_q:.0f}s]")
            else:
                bif_data_q_B.append((gamma, np.array([])))
                print(f" FAILED")

        t_q_B = time.time() - t0_q
        print(f"  Quantum Sweep B complete: {t_q_B:.1f}s")
    else:
        gamma_vals_q = np.array([])
        bif_data_q_B = []
        period_q_B = np.array([])

    # ==================================================================
    # δ EXTRACTION
    # ==================================================================
    print(f"\n{'='*70}")
    print("δ EXTRACTION")
    print(f"{'='*70}")

    # Classical δ (already known, restate)
    bif_pts_cl = [CLASSICAL_BIF['P1_P2'], CLASSICAL_BIF['P2_P4'],
                  CLASSICAL_BIF['P4_P8'], CLASSICAL_BIF['P8_P16']]
    deltas_cl = []
    for i in range(len(bif_pts_cl) - 2):
        d = (bif_pts_cl[i + 1] - bif_pts_cl[i]) / \
            (bif_pts_cl[i + 2] - bif_pts_cl[i + 1])
        deltas_cl.append(d)
        print(f"  Classical δ_{i+1} = {d:.4f}  "
              f"(error: {abs(d - FEIGENBAUM_DELTA)/FEIGENBAUM_DELTA*100:.1f}%)")

    # Quantum δ extraction — look for period transitions in quantum sweep
    if HAS_QUTIP and len(period_q_A) > 0:
        print(f"\n  Quantum period transitions (Sweep A):")
        bif_pts_q = []
        prev_p = period_q_A[0]
        for i in range(1, len(period_q_A)):
            if period_q_A[i] != prev_p and period_q_A[i] > 0 and prev_p > 0:
                if period_q_A[i] == 2 * prev_p:  # period doubling
                    f1_bif = (F1_vals_q[i - 1] + F1_vals_q[i]) / 2
                    print(f"    P{prev_p}→P{period_q_A[i]} at F₁ ≈ {f1_bif:.4f}")
                    bif_pts_q.append(f1_bif)
            prev_p = period_q_A[i]

        if len(bif_pts_q) >= 3:
            for i in range(len(bif_pts_q) - 2):
                dq = (bif_pts_q[i + 1] - bif_pts_q[i]) / \
                     (bif_pts_q[i + 2] - bif_pts_q[i + 1])
                print(f"    Quantum δ_{i+1} = {dq:.4f}  "
                      f"(error: {abs(dq - FEIGENBAUM_DELTA)/FEIGENBAUM_DELTA*100:.1f}%)")
        elif len(bif_pts_q) >= 1:
            print(f"    Found {len(bif_pts_q)} bifurcation(s) — "
                  f"need ≥3 for δ extraction")
        else:
            print(f"    No clean period-doubling detected in quantum sweep")
            print(f"    (Quantum fluctuations may smear cascade)")

    # ==================================================================
    # FIGURE GENERATION
    # ==================================================================
    print(f"\n{'='*70}")
    print("Generating 8-panel figure")
    print(f"{'='*70}")

    fig = plt.figure(figsize=(20, 22))
    fig.suptitle(
        "Script 80 — Quantum Kerr Cascade: Parametric Drive\n"
        r"$H(t) = \Delta\, a^\dagger a + \frac{K}{2}\,"
        r"a^\dagger a(a^\dagger a - 1) + "
        r"[F_0 + F_1\cos(\omega_m t)](a^\dagger + a)$,"
        r"$\quad L = \sqrt{\gamma}\, a$" "\n"
        f"$\\Delta = {DELTA}$,  $K = {K}$,  $F_0 = {F0}$,  "
        f"$\\omega_m = {WM}$,  $N_{{Fock}} = {N_FOCK}$",
        fontsize=13, fontweight="bold", y=0.995
    )

    gs = GridSpec(4, 2, hspace=0.35, wspace=0.3,
                  left=0.07, right=0.97, top=0.90, bottom=0.04)

    period_colors = {
        1: '#3498db',   # blue
        2: '#2ecc71',   # green
        4: '#e67e22',   # orange
        8: '#e74c3c',   # red
        16: '#9b59b6',  # purple
        32: '#1abc9c',  # teal
        0: '#7f8c8d',   # gray (chaos)
    }

    # ── Panel A: Classical bifurcation diagram (Sweep A) ──
    ax_a = fig.add_subplot(gs[0, 0])
    for F1, n_vals in bif_data_cl_A:
        if len(n_vals) > 0:
            last = n_vals[-min(32, len(n_vals)):]
            ax_a.plot([F1] * len(last), last, '.', color='#2c3e50',
                      markersize=0.4, alpha=0.6, rasterized=True)

    # Mark known bifurcation points
    for label, f1_val in CLASSICAL_BIF.items():
        ax_a.axvline(f1_val, color='#e74c3c', linewidth=0.8,
                     linestyle='--', alpha=0.5)

    ax_a.set_xlabel("$F_1$ (modulation depth)", fontsize=10)
    ax_a.set_ylabel("Stroboscopic $|\\alpha|^2$", fontsize=10)
    ax_a.set_title(f"A. Classical Bifurcation Diagram  "
                   f"($\\gamma = {GAMMA_FIXED}$)",
                   fontsize=11, fontweight='bold')
    ax_a.grid(True, alpha=0.2)
    ax_a.set_xlim(F1_MIN, F1_MAX)

    # ── Panel B: Quantum bifurcation diagram (Sweep A) ──
    ax_b = fig.add_subplot(gs[0, 1])
    if len(bif_data_q_A) > 0:
        for F1, n_vals in bif_data_q_A:
            if len(n_vals) > 0:
                last = n_vals[-min(16, len(n_vals)):]
                ax_b.plot([F1] * len(last), last, '.', color='#8e44ad',
                          markersize=1.5, alpha=0.7, rasterized=True)
        # Mark classical bifurcation points for comparison
        for label, f1_val in CLASSICAL_BIF.items():
            ax_b.axvline(f1_val, color='#e74c3c', linewidth=0.8,
                         linestyle='--', alpha=0.3)
    else:
        ax_b.text(0.5, 0.5, "QuTiP not available", transform=ax_b.transAxes,
                  ha='center', fontsize=12)

    ax_b.set_xlabel("$F_1$ (modulation depth)", fontsize=10)
    ax_b.set_ylabel("Stroboscopic $\\langle n \\rangle$", fontsize=10)
    ax_b.set_title(f"B. Quantum Bifurcation Diagram  "
                   f"($\\gamma = {GAMMA_FIXED}$)",
                   fontsize=11, fontweight='bold')
    ax_b.grid(True, alpha=0.2)
    ax_b.set_xlim(F1_MIN, F1_MAX)

    # ── Panel C: Classical period detection (Sweep A) ──
    ax_c = fig.add_subplot(gs[1, 0])
    for p_val in sorted(period_colors.keys()):
        mask = (period_cl_A == p_val)
        if np.any(mask):
            label = f"P-{p_val}" if p_val > 0 else "Chaos"
            ax_c.scatter(F1_vals_cl[mask], period_cl_A[mask],
                         c=period_colors[p_val], s=8, label=label,
                         edgecolors='none', alpha=0.8)
    ax_c.set_xlabel("$F_1$", fontsize=10)
    ax_c.set_ylabel("Detected Period", fontsize=10)
    ax_c.set_title("C. Classical Period Detection (Sweep A)",
                   fontsize=11, fontweight='bold')
    ax_c.legend(fontsize=8, loc='upper left', ncol=2)
    ax_c.grid(True, alpha=0.2)
    ax_c.set_xlim(F1_MIN, F1_MAX)
    ax_c.set_yticks([0, 1, 2, 4, 8, 16, 32])
    ax_c.set_yticklabels(['Ch', '1', '2', '4', '8', '16', '32'])

    # ── Panel D: Quantum period detection (Sweep A) ──
    ax_d = fig.add_subplot(gs[1, 1])
    if len(period_q_A) > 0:
        for p_val in sorted(period_colors.keys()):
            mask = (period_q_A == p_val)
            if np.any(mask):
                label = f"P-{p_val}" if p_val > 0 else "Chaos/QP"
                ax_d.scatter(F1_vals_q[mask], period_q_A[mask],
                             c=period_colors[p_val], s=20, label=label,
                             edgecolors='none', alpha=0.8)
        ax_d.legend(fontsize=8, loc='upper left', ncol=2)
    else:
        ax_d.text(0.5, 0.5, "QuTiP not available", transform=ax_d.transAxes,
                  ha='center', fontsize=12)
    ax_d.set_xlabel("$F_1$", fontsize=10)
    ax_d.set_ylabel("Detected Period", fontsize=10)
    ax_d.set_title("D. Quantum Period Detection (Sweep A)",
                   fontsize=11, fontweight='bold')
    ax_d.grid(True, alpha=0.2)
    ax_d.set_xlim(F1_MIN, F1_MAX)
    ax_d.set_yticks([0, 1, 2, 4, 8, 16])
    ax_d.set_yticklabels(['Ch', '1', '2', '4', '8', '16'])

    # ── Panel E: Classical bifurcation diagram (Sweep B) ──
    ax_e = fig.add_subplot(gs[2, 0])
    for gamma, n_vals in bif_data_cl_B:
        if len(n_vals) > 0:
            last = n_vals[-min(32, len(n_vals)):]
            ax_e.plot([gamma] * len(last), last, '.', color='#2c3e50',
                      markersize=0.4, alpha=0.6, rasterized=True)
    ax_e.set_xlabel("$\\gamma$ (dissipation rate)", fontsize=10)
    ax_e.set_ylabel("Stroboscopic $|\\alpha|^2$", fontsize=10)
    ax_e.set_title(f"E. Classical Bif. Diagram (Sweep B, $F_1 = {F1_FIXED}$)",
                   fontsize=11, fontweight='bold')
    ax_e.grid(True, alpha=0.2)
    ax_e.set_xlim(GAMMA_MIN_B, GAMMA_MAX_B)

    # ── Panel F: Quantum bifurcation diagram (Sweep B) ──
    ax_f = fig.add_subplot(gs[2, 1])
    if len(bif_data_q_B) > 0:
        for gamma, n_vals in bif_data_q_B:
            if len(n_vals) > 0:
                last = n_vals[-min(16, len(n_vals)):]
                ax_f.plot([gamma] * len(last), last, '.', color='#8e44ad',
                          markersize=1.5, alpha=0.7, rasterized=True)
    else:
        ax_f.text(0.5, 0.5, "QuTiP not available", transform=ax_f.transAxes,
                  ha='center', fontsize=12)
    ax_f.set_xlabel("$\\gamma$", fontsize=10)
    ax_f.set_ylabel("Stroboscopic $\\langle n \\rangle$", fontsize=10)
    ax_f.set_title(f"F. Quantum Bif. Diagram (Sweep B, $F_1 = {F1_FIXED}$)",
                   fontsize=11, fontweight='bold')
    ax_f.grid(True, alpha=0.2)
    ax_f.set_xlim(GAMMA_MIN_B, GAMMA_MAX_B)

    # ── Panel G: δ extraction comparison ──
    ax_g = fig.add_subplot(gs[3, 0])

    # Classical δ values
    delta_indices = list(range(1, len(deltas_cl) + 1))
    ax_g.plot(delta_indices, deltas_cl, 'o-', color='#2c3e50',
              markersize=10, linewidth=2, label='Classical', zorder=5)

    # Quantum δ if available
    if HAS_QUTIP and len(period_q_A) > 0:
        bif_pts_q = []
        prev_p = period_q_A[0]
        for i in range(1, len(period_q_A)):
            if (period_q_A[i] != prev_p and period_q_A[i] > 0
                    and prev_p > 0 and period_q_A[i] == 2 * prev_p):
                f1_bif = (F1_vals_q[i - 1] + F1_vals_q[i]) / 2
                bif_pts_q.append(f1_bif)
            prev_p = period_q_A[i]

        if len(bif_pts_q) >= 3:
            deltas_q = []
            for i in range(len(bif_pts_q) - 2):
                dq = (bif_pts_q[i + 1] - bif_pts_q[i]) / \
                     (bif_pts_q[i + 2] - bif_pts_q[i + 1])
                deltas_q.append(dq)
            dq_indices = list(range(1, len(deltas_q) + 1))
            ax_g.plot(dq_indices, deltas_q, 's-', color='#8e44ad',
                      markersize=10, linewidth=2, label='Quantum', zorder=5)

    ax_g.axhline(FEIGENBAUM_DELTA, color='#e74c3c', linewidth=2,
                 linestyle='--', label=f'$\\delta_F = {FEIGENBAUM_DELTA:.4f}$')
    ax_g.set_xlabel("Ratio index $k$", fontsize=10)
    ax_g.set_ylabel("$\\delta_k$", fontsize=10)
    ax_g.set_title("G. Feigenbaum δ Extraction", fontsize=11,
                   fontweight='bold')
    ax_g.legend(fontsize=9)
    ax_g.grid(True, alpha=0.2)
    if delta_indices:
        ax_g.set_xlim(0.5, max(delta_indices) + 0.5)
    ax_g.set_ylim(0, max(max(deltas_cl) * 1.3, FEIGENBAUM_DELTA * 1.5))

    # ── Panel H: Stroboscopic spread analysis ──
    ax_h = fig.add_subplot(gs[3, 1])

    # Plot the stroboscopic spread (max-min of last 16 samples) vs F₁
    # This shows where period-doubling bifurcates
    spreads_cl = []
    for F1, n_vals in bif_data_cl_A:
        if len(n_vals) >= 16:
            spread = np.max(n_vals[-16:]) - np.min(n_vals[-16:])
            spreads_cl.append((F1, spread))
    if spreads_cl:
        f1_sp, sp = zip(*spreads_cl)
        ax_h.semilogy(f1_sp, np.array(sp) + 1e-6, '-', color='#2c3e50',
                      linewidth=1, alpha=0.8, label='Classical')

    if HAS_QUTIP and len(bif_data_q_A) > 0:
        spreads_q = []
        for F1, n_vals in bif_data_q_A:
            if len(n_vals) >= 8:
                spread = np.max(n_vals[-8:]) - np.min(n_vals[-8:])
                spreads_q.append((F1, spread))
        if spreads_q:
            f1_sq, sq = zip(*spreads_q)
            ax_h.semilogy(f1_sq, np.array(sq) + 1e-6, 'o-', color='#8e44ad',
                          linewidth=1.5, markersize=4, alpha=0.8,
                          label='Quantum')

    for label, f1_val in CLASSICAL_BIF.items():
        ax_h.axvline(f1_val, color='#e74c3c', linewidth=0.8,
                     linestyle='--', alpha=0.4)

    ax_h.set_xlabel("$F_1$", fontsize=10)
    ax_h.set_ylabel("Stroboscopic spread (max - min)", fontsize=10)
    ax_h.set_title("H. Bifurcation Spread: Classical vs Quantum",
                   fontsize=11, fontweight='bold')
    ax_h.legend(fontsize=9)
    ax_h.grid(True, alpha=0.2)
    ax_h.set_xlim(F1_MIN, F1_MAX)

    outpath = ("/Users/lucianrandolph/Downloads/Projects/"
               "einstein_fractal_analysis/fig80_quantum_kerr_parametric.png")
    plt.savefig(outpath, dpi=200, bbox_inches='tight')
    print(f"\nSaved: {outpath}")
    plt.close()

    # ==================================================================
    # SUMMARY
    # ==================================================================
    print(f"\n{'='*70}")
    print("RESULTS SUMMARY — Script 80")
    print(f"{'='*70}")

    print(f"\nSystem: Parametrically driven Kerr oscillator")
    print(f"  H(t) = {DELTA} a†a + {K/2} a†a(a†a-1) "
          f"+ [{F0} + F₁cos({WM}t)](a† + a)")
    print(f"  L = √γ · a")
    print(f"  Parametric resonance: ωₘ = {WM} ≈ 2 × ω_int ≈ 2 × 2.6")

    print(f"\nClassical cascade (ωₘ = {WM}, γ = {GAMMA_FIXED}):")
    for label, val in CLASSICAL_BIF.items():
        print(f"  {label}: F₁ = {val:.6f}")
    for i, d in enumerate(deltas_cl):
        err = abs(d - FEIGENBAUM_DELTA) / FEIGENBAUM_DELTA * 100
        print(f"  δ_{i+1} = {d:.4f}  ({err:.1f}% error)")

    if HAS_QUTIP:
        print(f"\nQuantum results:")
        if len(period_q_A) > 0:
            for p in [1, 2, 4, 8]:
                count = np.sum(period_q_A == p)
                if count > 0:
                    idxs = np.where(period_q_A == p)[0]
                    print(f"  Period-{p}: {count}/{len(period_q_A)} points, "
                          f"F₁ ∈ [{F1_vals_q[idxs[0]]:.3f}, "
                          f"{F1_vals_q[idxs[-1]]:.3f}]")

            # Key comparison
            has_p2_q = np.any(period_q_A == 2)
            has_p4_q = np.any(period_q_A == 4)
            has_p8_q = np.any(period_q_A == 8)

            print(f"\n  *** KILL CRITERION CHECK ***")
            if has_p2_q:
                print(f"  ✓ Period-2 DETECTED in quantum density matrix")
                print(f"    → Period-doubling SURVIVES quantum decoherence!")
                if has_p4_q:
                    print(f"  ✓ Period-4 DETECTED — cascade continues!")
                    if has_p8_q:
                        print(f"  ✓ Period-8 DETECTED — FULL CASCADE in quantum!")
                    else:
                        print(f"  ✗ Period-8 not detected (smeared by fluctuations?)")
                else:
                    print(f"  ✗ Period-4 not detected (quantum fluctuations "
                          f"may arrest cascade)")
            else:
                print(f"  ✗ No period-doubling in quantum sweep")
                print(f"    Possible reasons:")
                print(f"    - Quantum fluctuations smear the bifurcation")
                print(f"    - N_Fock = {N_FOCK} too small")
                print(f"    - Need higher N_trans for convergence")
                print(f"    - Quantum expectation values are AVERAGES "
                      f"over Wigner function")

    total_time_str = ""
    print(f"\nScript 80 complete.")


if __name__ == "__main__":
    main()
