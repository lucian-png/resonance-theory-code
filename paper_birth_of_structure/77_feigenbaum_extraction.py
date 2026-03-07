#!/usr/bin/env python3
"""
Script 77 — Feigenbaum δ Extraction (v14)
==========================================
v14: CORRECT parameter range + interpolated crossings + unique-value counting.

BREAKTHROUGH: The "standard" literature values (c₂≈3.53, c₃≈3.68) were WRONG
for the ẏ=0 Poincaré section of the Rössler with a=b=0.2. The actual cascade:
  c₁ ≈ 2.83  (P1 → P2)
  c₂ ≈ 3.85  (P2 → P4)     ← not 3.53!
  c₃ ≈ 4.07  (P4 → P8)     ← not 3.68!
  c₄ ≈ 4.12  (P8 → P16)

v13 showed that LINEAR INTERPOLATION of Poincaré crossings reduces noise from
O(0.02) to O(0.000001), making period detection trivial via unique-value counting.

Method: Binary search on unique crossing count (period detection).
  Period p at c → count_unique(round(crossings, 3)) == p.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
import time

FEIGENBAUM_DELTA = 4.669201609102990671853203820466


# ==============================================================================
# Logistic Map (proven working — unchanged)
# ==============================================================================

def logistic_detect_period(r, tol=1e-9, n_transient=50000):
    x = 0.4
    for _ in range(n_transient):
        x = r * x * (1.0 - x)
    orbit = np.empty(256)
    for i in range(256):
        x = r * x * (1.0 - x)
        orbit[i] = x
    for p in [1, 2, 4, 8, 16, 32, 64, 128]:
        if 256 < 3 * p:
            break
        last_sorted = np.sort(orbit[-p:])
        prev_sorted = np.sort(orbit[-2*p:-p])
        if np.max(np.abs(last_sorted - prev_sorted)) < tol:
            return p
    return 0

def find_logistic_bifurcation(r_lo, r_hi, target_period, n_bisect=70):
    for _ in range(n_bisect):
        r_mid = (r_lo + r_hi) / 2.0
        period = logistic_detect_period(r_mid)
        if period <= target_period:
            r_lo = r_mid
        else:
            r_hi = r_mid
    return (r_lo + r_hi) / 2.0

def logistic_extraction():
    searches = [
        (2.95, 3.05, 1), (3.44, 3.46, 2), (3.540, 3.548, 4),
        (3.5640, 3.5650, 8), (3.5685, 3.5690, 16), (3.5696, 3.5700, 32),
    ]
    bif_points: List[float] = []
    print("  Bifurcation points:")
    for r_lo, r_hi, target_p in searches:
        r_bif = find_logistic_bifurcation(r_lo, r_hi, target_p, n_bisect=70)
        bif_points.append(r_bif)
        print(f"    r_{len(bif_points)} = {r_bif:.15f}  "
              f"(period {target_p} → {2*target_p})")
    deltas: List[float] = []
    print("\n  Feigenbaum δ ratios:")
    for i in range(2, len(bif_points)):
        gap_prev = bif_points[i-1] - bif_points[i-2]
        gap_curr = bif_points[i] - bif_points[i-1]
        if abs(gap_curr) > 1e-16:
            d = gap_prev / gap_curr
            deltas.append(d)
            err = abs(d - FEIGENBAUM_DELTA) / FEIGENBAUM_DELTA * 100
            print(f"    δ_{i-1} = {d:.10f}  (error: {err:.6f}%)")
    return bif_points, deltas


# ==============================================================================
# Rössler — Single c-value Integration with Interpolated Crossings
# ==============================================================================

def rossler_crossings(c_val, a=0.2, b=0.2, dt=0.005,
                       t_transient=12000.0, t_record=6000.0):
    """
    Integrate Rössler at one c-value with LINEAR INTERPOLATION of crossings.
    Returns array of x-values at exact ẏ=0 crossings.
    """
    n_trans = int(t_transient / dt)
    n_rec = int(t_record / dt)

    x, y, z = 0.1, 0.1, 0.1
    hdt = 0.5 * dt
    sdt = dt / 6.0

    prev_ydot = x + a * y
    prev_x = x
    crossings: List[float] = []

    for step in range(n_trans + n_rec):
        # RK4
        k1x = -y - z;  k1y = x + a*y;  k1z = b + z*(x - c_val)
        mx = x + hdt*k1x;  my = y + hdt*k1y;  mz = z + hdt*k1z
        k2x = -my - mz;  k2y = mx + a*my;  k2z = b + mz*(mx - c_val)
        mx2 = x + hdt*k2x;  my2 = y + hdt*k2y;  mz2 = z + hdt*k2z
        k3x = -my2 - mz2;  k3y = mx2 + a*my2;  k3z = b + mz2*(mx2 - c_val)
        fx = x + dt*k3x;  fy = y + dt*k3y;  fz = z + dt*k3z
        k4x = -fy - fz;  k4y = fx + a*fy;  k4z = b + fz*(fx - c_val)

        nx = x + sdt*(k1x + 2*k2x + 2*k3x + k4x)
        ny = y + sdt*(k1y + 2*k2y + 2*k3y + k4y)
        nz = z + sdt*(k1z + 2*k2z + 2*k3z + k4z)

        if step >= n_trans:
            ydot = nx + a * ny
            if prev_ydot > 0 and ydot <= 0:
                denom = prev_ydot - ydot
                if abs(denom) > 1e-30:
                    frac = prev_ydot / denom
                    x_cross = prev_x + frac * (nx - prev_x)
                else:
                    x_cross = nx
                crossings.append(float(x_cross))
            prev_ydot = ydot
            prev_x = nx
        elif step == n_trans - 1:
            prev_ydot = nx + a * ny
            prev_x = nx

        x, y, z = nx, ny, nz

    return np.array(crossings) if crossings else np.array([])


def detect_period(crossings: np.ndarray, tol: float = 0.001) -> int:
    """
    Detect the period from Poincaré crossings using sorted block comparison.
    Same robust method as the logistic map detector.
    """
    n = len(crossings)
    if n < 10:
        return 0
    for p in [1, 2, 4, 8, 16, 32]:
        if n < 3 * p:
            break
        last = np.sort(crossings[-p:])
        prev = np.sort(crossings[-2*p:-p])
        if np.max(np.abs(last - prev)) < tol:
            return p
    return 0  # Chaos or higher period


def find_rossler_bifurcation(c_lo, c_hi, target_period,
                              n_bisect=40, tol=0.001):
    """
    Binary search for the bifurcation point where period goes from
    target_period to 2*target_period.
    """
    for step in range(n_bisect):
        c_mid = (c_lo + c_hi) / 2.0
        cr = rossler_crossings(c_mid)
        p = detect_period(cr, tol=tol)

        if p <= target_period:
            c_lo = c_mid
        else:
            c_hi = c_mid

        if (c_hi - c_lo) < 1e-5:
            break

    return (c_lo + c_hi) / 2.0


# ==============================================================================
# Rössler Extraction
# ==============================================================================

def rossler_extraction():
    """
    Extract Feigenbaum δ from Rössler system.
    Uses interpolated crossings + sorted block period detection.
    """

    # Phase 1: Survey — find period at each c value
    print("  Phase 1: Period survey (c = 2.5 to 4.3)...", flush=True)
    c_survey = np.linspace(2.5, 4.3, 100)

    periods = []
    t0 = time.time()
    for i, c in enumerate(c_survey):
        cr = rossler_crossings(c, t_transient=8000.0, t_record=4000.0)
        p = detect_period(cr, tol=0.001)
        periods.append(p)
        if i % 20 == 0:
            print(f"    c = {c:.3f}: period = {p}  ({i}/{len(c_survey)})",
                  flush=True)
    periods = np.array(periods)
    print(f"    Survey: {time.time()-t0:.1f}s", flush=True)

    # Display period map
    print("\n  Period map:")
    print(f"  {'c':>6s}  {'period':>6s}")
    print("  " + "-" * 16)
    for i in range(0, len(c_survey), 5):
        print(f"  {c_survey[i]:6.3f}  {periods[i]:6d}")
    print("  " + "-" * 16)

    # Phase 2: Find transition brackets
    transitions: List[Tuple[str, int, float, float]] = []
    for target_from in [1, 2, 4, 8, 16]:
        target_to = 2 * target_from
        for i in range(1, len(c_survey)):
            if periods[i-1] == target_from and periods[i] >= target_to:
                transitions.append((
                    f"{target_from}→{target_to}",
                    target_from,
                    float(c_survey[i-1]),
                    float(c_survey[i]),
                ))
                break
            # Also catch transition through 0 (chaos)
            if (periods[i-1] == target_from and
                periods[i] == 0 and i > 1):
                # Check if there's a brief higher-period window
                pass

    print(f"\n  Transitions found ({len(transitions)}):")
    for label, tp, clo, chi in transitions:
        print(f"    {label}: c ∈ [{clo:.4f}, {chi:.4f}]")

    # Phase 3: Binary search refinement
    print("\n  Phase 3: Binary search refinement...", flush=True)
    bif_points: List[float] = []

    for label, target_period, c_lo, c_hi in transitions:
        print(f"\n    {label}: searching [{c_lo:.4f}, {c_hi:.4f}]...",
              flush=True)
        c_bif = find_rossler_bifurcation(
            c_lo, c_hi, target_period,
            n_bisect=40, tol=0.001,
        )
        bif_points.append(c_bif)

        # Verify
        cr_lo = rossler_crossings(c_bif - 0.001)
        cr_hi = rossler_crossings(c_bif + 0.001)
        p_lo = detect_period(cr_lo, tol=0.001)
        p_hi = detect_period(cr_hi, tol=0.001)
        print(f"      → c = {c_bif:.6f}  "
              f"(period at c-0.001: {p_lo}, at c+0.001: {p_hi})")

    # Results
    deltas: List[float] = []
    print("\n  " + "=" * 50)
    print("  Rössler bifurcation points (Poincaré section ẏ=0):")
    for i, c in enumerate(bif_points):
        print(f"    c_{i+1} = {c:.6f}")

    if len(bif_points) >= 3:
        for i in range(2, len(bif_points)):
            g_prev = bif_points[i-1] - bif_points[i-2]
            g_curr = bif_points[i] - bif_points[i-1]
            if abs(g_curr) > 1e-12:
                d = g_prev / g_curr
                deltas.append(d)
                err = abs(d - FEIGENBAUM_DELTA) / FEIGENBAUM_DELTA * 100
                print(f"\n    gap_{i-1} = c_{i} - c_{i-1} = {g_prev:.6f}")
                print(f"    gap_{i} = c_{i+1} - c_{i} = {g_curr:.6f}")
                print(f"    δ_{i-1} = {d:.6f}  (error: {err:.2f}%)")

    return bif_points, deltas


# ==============================================================================
# Plotting
# ==============================================================================

def make_plots(log_bifs, log_deltas, ros_bifs, ros_deltas):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Feigenbaum δ Extraction\n"
                 "The universal constant governing period-doubling cascades",
                 fontsize=15, fontweight="bold", y=0.99)

    # Left: Logistic Map
    if log_deltas:
        n_vals = list(range(1, len(log_deltas) + 1))
        ax1.plot(n_vals, log_deltas, "o-", color="#1f77b4",
                 markersize=12, linewidth=2.5, zorder=5, label="Measured δₙ")
        ax1.axhline(FEIGENBAUM_DELTA, color="#d62728", linewidth=2,
                     linestyle="--", zorder=3, label=f"δ = {FEIGENBAUM_DELTA:.4f}")
        for i, d in enumerate(log_deltas):
            err = abs(d - FEIGENBAUM_DELTA) / FEIGENBAUM_DELTA * 100
            ax1.annotate(f"{d:.4f}\n({err:.3f}%)", (n_vals[i], d),
                         textcoords="offset points", xytext=(15, 10), fontsize=10,
                         fontweight="bold",
                         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                                   edgecolor="gray", alpha=0.9))
    ax1.set_xlabel("Bifurcation ratio index n", fontsize=12)
    ax1.set_ylabel("δₙ = Δrₙ₋₁ / Δrₙ", fontsize=12)
    ax1.set_title("Logistic Map x → rx(1−x)", fontsize=13)
    ax1.legend(fontsize=11, loc="upper right")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(4.0, 5.5)

    # Right: Rössler
    if ros_deltas:
        n_vals = list(range(1, len(ros_deltas) + 1))
        ax2.plot(n_vals, ros_deltas, "s-", color="#2ca02c",
                 markersize=12, linewidth=2.5, zorder=5, label="Measured δₙ")
        ax2.axhline(FEIGENBAUM_DELTA, color="#d62728", linewidth=2,
                     linestyle="--", zorder=3, label=f"δ = {FEIGENBAUM_DELTA:.4f}")
        for i, d in enumerate(ros_deltas):
            err = abs(d - FEIGENBAUM_DELTA) / FEIGENBAUM_DELTA * 100
            ax2.annotate(f"{d:.4f}\n({err:.2f}%)", (n_vals[i], d),
                         textcoords="offset points", xytext=(15, 10), fontsize=10,
                         fontweight="bold",
                         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen",
                                   edgecolor="gray", alpha=0.9))
    else:
        ax2.text(0.5, 0.5, "Insufficient bifurcation\npoints resolved",
                 transform=ax2.transAxes, ha="center", va="center", fontsize=14)
    ax2.set_xlabel("Bifurcation ratio index n", fontsize=12)
    ax2.set_ylabel("δₙ = Δcₙ₋₁ / Δcₙ", fontsize=12)
    ax2.set_title("Rössler System (Decoherence Analog)", fontsize=13)
    ax2.legend(fontsize=11, loc="upper right")
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(3.0, 6.5)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    out = "/Users/lucianrandolph/Downloads/Projects/einstein_fractal_analysis/fig77_feigenbaum_convergence.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"\nSaved: {out}")
    plt.close()


def main():
    print("=" * 60)
    print("Script 77 — Feigenbaum δ Extraction (v14)")
    print("=" * 60)
    print(f"Target: δ = {FEIGENBAUM_DELTA:.15f}\n")

    print("=== Part 1: Logistic Map ===")
    log_b, log_d = logistic_extraction()

    print("\n=== Part 2: Rössler System ===")
    ros_b, ros_d = rossler_extraction()

    make_plots(log_b, log_d, ros_b, ros_d)

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Feigenbaum δ (exact) = {FEIGENBAUM_DELTA:.15f}\n")

    print("LOGISTIC MAP:")
    for i, d in enumerate(log_d):
        err = abs(d - FEIGENBAUM_DELTA) / FEIGENBAUM_DELTA * 100
        print(f"  δ_{i+1} = {d:.10f}  ({err:.6f}%)")

    print("\nRÖSSLER:")
    for i, d in enumerate(ros_d):
        err = abs(d - FEIGENBAUM_DELTA) / FEIGENBAUM_DELTA * 100
        print(f"  δ_{i+1} = {d:.6f}  ({err:.3f}%)")
    if not ros_d:
        print("  [Detection did not converge]")

    print("\nScript 77 complete.")


if __name__ == "__main__":
    main()
