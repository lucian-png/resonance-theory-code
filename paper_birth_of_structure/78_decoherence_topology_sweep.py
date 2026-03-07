#!/usr/bin/env python3
"""
Script 78 — Decoherence Topology Sweep
========================================
The universality test: do ALL bounded nonlinear coupled systems produce
the SAME Feigenbaum cascade, regardless of coupling topology?

Systems tested (all satisfy C₁, C₂, C₃):
  1. Logistic map:     x → rx(1-x)           — quadratic extremum (z=2)
  2. Cubic map:         x → 1 - r|x|³         — cubic extremum (z=3)
  3. Sine map:          x → (r/π)sin(πx)      — quadratic extremum (z=2)
  4. Rössler (ẏ=0):     3D ODE, Poincaré section (z=2)
  5. Duffing:           driven oscillator, stroboscopic section (z=2)

For each system: extract period-doubling bifurcation points → compute δ ratios.
All should converge to the SAME δ for the same coupling order:
  z=2 → δ = 4.669...    (logistic, sine, Rössler, Duffing)
  z=3 → δ = 5.968...    (cubic map)

This IS the universality proof: the cascade is set by topology, not dynamics.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import List, Tuple
import time

# Feigenbaum constants for different coupling orders
DELTA_Z2 = 4.669201609102990671853203820466  # quadratic extremum (z=2)
DELTA_Z3 = 5.9679687                          # cubic extremum (z=3)


# ==============================================================================
# System 1: Logistic Map x → rx(1-x) — z=2, quadratic extremum
# ==============================================================================

def logistic_period(r, tol=1e-9, n_trans=50000):
    x = 0.4
    for _ in range(n_trans):
        x = r * x * (1.0 - x)
    orbit = np.empty(256)
    for i in range(256):
        x = r * x * (1.0 - x)
        orbit[i] = x
    for p in [1, 2, 4, 8, 16, 32, 64, 128]:
        if 256 < 3 * p:
            break
        if np.max(np.abs(np.sort(orbit[-p:]) - np.sort(orbit[-2*p:-p]))) < tol:
            return p
    return 0

def logistic_bifurcation(r_lo, r_hi, target_p, n_bis=60):
    for _ in range(n_bis):
        r_mid = (r_lo + r_hi) / 2.0
        p = logistic_period(r_mid)
        if p <= target_p:
            r_lo = r_mid
        else:
            r_hi = r_mid
    return (r_lo + r_hi) / 2.0

def extract_logistic():
    """Extract δ from the logistic map."""
    searches = [
        (2.95, 3.05, 1), (3.44, 3.46, 2), (3.540, 3.548, 4),
        (3.5640, 3.5650, 8), (3.5685, 3.5690, 16), (3.5696, 3.5700, 32),
    ]
    bifs = [logistic_bifurcation(lo, hi, tp) for lo, hi, tp in searches]
    deltas = []
    for i in range(2, len(bifs)):
        g1 = bifs[i-1] - bifs[i-2]
        g2 = bifs[i] - bifs[i-1]
        if abs(g2) > 1e-16:
            deltas.append(g1 / g2)
    return bifs, deltas


# ==============================================================================
# System 2: Cubic Map x → 1 - r|x|³ — z=3, TRUE cubic extremum
# ==============================================================================

def cubic_period(r, tol=1e-8, n_trans=50000):
    """
    Period detection for the standard z=3 Feigenbaum map: x → 1 - r|x|³.
    Maps [-1, 1] → [-1, 1]. Critical point at x=0 with cubic tangency.
    This gives δ(z=3) = 5.9679687...
    """
    x = 0.1
    for _ in range(n_trans):
        x = 1.0 - r * abs(x) ** 3
        if abs(x) > 2 or not np.isfinite(x):
            return 0
    orbit = np.empty(256)
    for i in range(256):
        x = 1.0 - r * abs(x) ** 3
        if abs(x) > 2 or not np.isfinite(x):
            return 0
        orbit[i] = x
    for p in [1, 2, 4, 8, 16, 32, 64, 128]:
        if 256 < 3 * p:
            break
        if np.max(np.abs(np.sort(orbit[-p:]) - np.sort(orbit[-2*p:-p]))) < tol:
            return p
    return 0

def cubic_bifurcation(r_lo, r_hi, target_p, n_bis=60):
    for _ in range(n_bis):
        r_mid = (r_lo + r_hi) / 2.0
        p = cubic_period(r_mid)
        if p <= target_p:
            r_lo = r_mid
        else:
            r_hi = r_mid
    return (r_lo + r_hi) / 2.0

def extract_cubic():
    """
    Extract δ from x → 1 - r|x|³.
    TRUE z=3 universality class: δ(z=3) = 5.9679687...
    Period-1 → period-2 onset near r ≈ 1.5.
    """
    print("    Cubic z=3 map: coarse survey...", flush=True)
    r_test = np.linspace(1.0, 2.5, 500)
    periods = []
    for r in r_test:
        periods.append(cubic_period(r))
    periods = np.array(periods)

    # Find transition brackets
    searches: List[Tuple[float, float, int]] = []
    for target_from in [1, 2, 4, 8, 16, 32]:
        target_to = 2 * target_from
        for i in range(1, len(r_test)):
            if periods[i-1] == target_from and periods[i] >= target_to:
                searches.append((float(r_test[i-1]), float(r_test[i]), target_from))
                break

    print(f"    Found {len(searches)} transitions", flush=True)
    bifs = [cubic_bifurcation(lo, hi, tp) for lo, hi, tp in searches]
    for i, b in enumerate(bifs):
        print(f"      c_{i+1} = {b:.10f}")
    deltas = []
    for i in range(2, len(bifs)):
        g1 = bifs[i-1] - bifs[i-2]
        g2 = bifs[i] - bifs[i-1]
        if abs(g2) > 1e-16:
            deltas.append(g1 / g2)
    return bifs, deltas


# ==============================================================================
# System 3: Sine Map x → (r/π)sin(πx) — z=2, transcendental coupling
# ==============================================================================

def sine_period(r, tol=1e-9, n_trans=50000):
    """Period detection for sine map x → (r/π)sin(πx)."""
    x = 0.5
    for _ in range(n_trans):
        x = (r / np.pi) * np.sin(np.pi * x)
        if not np.isfinite(x):
            return 0
    orbit = np.empty(256)
    for i in range(256):
        x = (r / np.pi) * np.sin(np.pi * x)
        if not np.isfinite(x):
            return 0
        orbit[i] = x
    for p in [1, 2, 4, 8, 16, 32, 64, 128]:
        if 256 < 3 * p:
            break
        if np.max(np.abs(np.sort(orbit[-p:]) - np.sort(orbit[-2*p:-p]))) < tol:
            return p
    return 0

def sine_bifurcation(r_lo, r_hi, target_p, n_bis=60):
    for _ in range(n_bis):
        r_mid = (r_lo + r_hi) / 2.0
        p = sine_period(r_mid)
        if p <= target_p:
            r_lo = r_mid
        else:
            r_hi = r_mid
    return (r_lo + r_hi) / 2.0

def extract_sine():
    """
    Extract δ from the sine map x → (r/π)sin(πx).
    Has a quadratic maximum at x=0.5 → same universality class as logistic.
    δ should equal 4.669... despite completely different functional form.
    """
    print("    Sine map: coarse survey...", flush=True)
    r_test = np.linspace(2.5, 3.8, 500)
    periods = []
    for r in r_test:
        periods.append(sine_period(r))
    periods = np.array(periods)

    # Find transition brackets
    searches: List[Tuple[float, float, int]] = []
    for target_from in [1, 2, 4, 8, 16, 32]:
        target_to = 2 * target_from
        for i in range(1, len(r_test)):
            if periods[i-1] == target_from and periods[i] >= target_to:
                searches.append((float(r_test[i-1]), float(r_test[i]), target_from))
                break

    print(f"    Found {len(searches)} transitions", flush=True)
    bifs = [sine_bifurcation(lo, hi, tp) for lo, hi, tp in searches]
    for i, b in enumerate(bifs):
        print(f"      r_{i+1} = {b:.10f}")
    deltas = []
    for i in range(2, len(bifs)):
        g1 = bifs[i-1] - bifs[i-2]
        g2 = bifs[i] - bifs[i-1]
        if abs(g2) > 1e-16:
            deltas.append(g1 / g2)
    return bifs, deltas


# ==============================================================================
# System 4: Rössler (from Script 77 — reference values)
# ==============================================================================

def extract_rossler():
    """
    Rössler bifurcation points from Script 77 v14.
    These were computed with 12000 time unit transient, interpolated crossings,
    and binary search on sorted-block period detection.
    """
    # From Script 77 v14 binary search results:
    bifs = [2.831095, 3.836404, 4.123948, 4.186954]
    deltas = []
    for i in range(2, len(bifs)):
        g1 = bifs[i-1] - bifs[i-2]
        g2 = bifs[i] - bifs[i-1]
        if abs(g2) > 1e-12:
            deltas.append(g1 / g2)
    return bifs, deltas


# ==============================================================================
# System 5: Duffing Oscillator — stroboscopic section
# ==============================================================================

def duffing_period(F, delta=0.25, omega=1.0, tol=5e-4, n_trans=500,
                    n_record=200, steps_per=300):
    """
    Period detection for the Duffing oscillator ẍ + δẋ - x + x³ = F cos(ωt).
    Stroboscopic section at T = 2π/ω.
    """
    T = 2.0 * np.pi / omega
    dt = T / steps_per
    x, v = 0.5, 0.0
    # Transient
    for period_idx in range(n_trans):
        for step in range(steps_per):
            t = (period_idx * steps_per + step) * dt
            k1x = v
            k1v = -delta * v + x - x**3 + F * np.cos(omega * t)
            xm = x + 0.5*dt*k1x; vm = v + 0.5*dt*k1v; tm = t + 0.5*dt
            k2x = vm
            k2v = -delta*vm + xm - xm**3 + F*np.cos(omega*tm)
            xm2 = x + 0.5*dt*k2x; vm2 = v + 0.5*dt*k2v
            k3x = vm2
            k3v = -delta*vm2 + xm2 - xm2**3 + F*np.cos(omega*tm)
            xf = x + dt*k3x; vf = v + dt*k3v; tf = t + dt
            k4x = vf
            k4v = -delta*vf + xf - xf**3 + F*np.cos(omega*tf)
            x += (dt/6.0)*(k1x + 2*k2x + 2*k3x + k4x)
            v += (dt/6.0)*(k1v + 2*k2v + 2*k3v + k4v)
    # Record
    orbit = np.empty(n_record)
    for period_idx in range(n_record):
        for step in range(steps_per):
            t = ((n_trans + period_idx) * steps_per + step) * dt
            k1x = v
            k1v = -delta * v + x - x**3 + F * np.cos(omega * t)
            xm = x + 0.5*dt*k1x; vm = v + 0.5*dt*k1v; tm = t + 0.5*dt
            k2x = vm
            k2v = -delta*vm + xm - xm**3 + F*np.cos(omega*tm)
            xm2 = x + 0.5*dt*k2x; vm2 = v + 0.5*dt*k2v
            k3x = vm2
            k3v = -delta*vm2 + xm2 - xm2**3 + F*np.cos(omega*tm)
            xf = x + dt*k3x; vf = v + dt*k3v; tf = t + dt
            k4x = vf
            k4v = -delta*vf + xf - xf**3 + F*np.cos(omega*tf)
            x += (dt/6.0)*(k1x + 2*k2x + 2*k3x + k4x)
            v += (dt/6.0)*(k1v + 2*k2v + 2*k3v + k4v)
        orbit[period_idx] = x
    for p in [1, 2, 4, 8, 16, 32, 64]:
        if n_record < 3 * p:
            break
        if np.max(np.abs(np.sort(orbit[-p:]) - np.sort(orbit[-2*p:-p]))) < tol:
            return p
    return 0

def duffing_bifurcation(F_lo, F_hi, target_p, n_bis=50, damp=0.15):
    for _ in range(n_bis):
        F_mid = (F_lo + F_hi) / 2.0
        p = duffing_period(F_mid, delta=damp)
        if p <= target_p:
            F_lo = F_mid
        else:
            F_hi = F_mid
    return (F_lo + F_hi) / 2.0

def extract_duffing():
    """
    Extract δ from Duffing oscillator.
    ẍ + 0.15ẋ - x + x³ = F cos(t), stroboscopic at T = 2π.
    Lower damping (0.15) gives cleaner cascade than 0.25.
    """
    # Coarse survey — wider range, lower damping
    print("    Duffing: coarse survey (δ=0.15, F=0.10-0.80)...", flush=True)
    F_test = np.linspace(0.10, 0.80, 200)
    periods = []
    for F in F_test:
        periods.append(duffing_period(F, delta=0.15))
    periods = np.array(periods)

    # Find transitions
    searches: List[Tuple[float, float, int]] = []
    for target_from in [1, 2, 4, 8, 16]:
        target_to = 2 * target_from
        for i in range(1, len(F_test)):
            if periods[i-1] == target_from and periods[i] >= target_to:
                searches.append((float(F_test[i-1]), float(F_test[i]),
                                 target_from))
                break

    if not searches:
        print("    Duffing: no transitions found in survey range")
        # Print what periods were found for debugging
        unique_p = np.unique(periods)
        print(f"    Periods seen: {unique_p}")
        return [], []

    bifs = []
    for lo, hi, tp in searches:
        F_bif = duffing_bifurcation(lo, hi, tp, n_bis=50)
        bifs.append(F_bif)
        print(f"      F_{len(bifs)} = {F_bif:.10f}")

    deltas = []
    for i in range(2, len(bifs)):
        g1 = bifs[i-1] - bifs[i-2]
        g2 = bifs[i] - bifs[i-1]
        if abs(g2) > 1e-16:
            deltas.append(g1 / g2)
    return bifs, deltas


# ==============================================================================
# Main Analysis & Plotting
# ==============================================================================

def main():
    print("=" * 60)
    print("Script 78 — Decoherence Topology Sweep")
    print("=" * 60)
    print("Testing universality: same δ for same coupling order\n")

    results = {}

    # System 1: Logistic
    print("--- System 1: Logistic Map (z=2) ---")
    bifs, deltas = extract_logistic()
    results["Logistic\n(z=2)"] = (bifs, deltas, DELTA_Z2)
    print(f"    {len(bifs)} bifurcation points, {len(deltas)} δ ratios")
    for i, d in enumerate(deltas):
        err = abs(d - DELTA_Z2) / DELTA_Z2 * 100
        print(f"    δ_{i+1} = {d:.6f}  ({err:.3f}%)")

    # System 2: Cubic
    print("\n--- System 2: Cubic Map (z=3) ---")
    bifs, deltas = extract_cubic()
    results["Cubic\n(z=3)"] = (bifs, deltas, DELTA_Z3)
    print(f"    {len(bifs)} bifurcation points, {len(deltas)} δ ratios")
    for i, d in enumerate(deltas):
        err = abs(d - DELTA_Z3) / DELTA_Z3 * 100
        print(f"    δ_{i+1} = {d:.6f}  ({err:.3f}%)")

    # System 3: Sine map
    print("\n--- System 3: Sine Map (z=2) ---")
    bifs, deltas = extract_sine()
    results["Sine\n(z=2)"] = (bifs, deltas, DELTA_Z2)
    print(f"    {len(bifs)} bifurcation points, {len(deltas)} δ ratios")
    for i, d in enumerate(deltas):
        err = abs(d - DELTA_Z2) / DELTA_Z2 * 100
        print(f"    δ_{i+1} = {d:.6f}  ({err:.3f}%)")

    # System 4: Rössler
    print("\n--- System 4: Rössler System (z=2) ---")
    bifs, deltas = extract_rossler()
    results["Rössler\n(z=2)"] = (bifs, deltas, DELTA_Z2)
    print(f"    {len(bifs)} bifurcation points, {len(deltas)} δ ratios")
    for i, d in enumerate(deltas):
        err = abs(d - DELTA_Z2) / DELTA_Z2 * 100
        print(f"    δ_{i+1} = {d:.6f}  ({err:.3f}%)")

    # System 5: Duffing
    print("\n--- System 5: Duffing Oscillator (z=2) ---")
    bifs, deltas = extract_duffing()
    results["Duffing\n(z=2)"] = (bifs, deltas, DELTA_Z2)
    print(f"    {len(bifs)} bifurcation points, {len(deltas)} δ ratios")
    for i, d in enumerate(deltas):
        err = abs(d - DELTA_Z2) / DELTA_Z2 * 100
        print(f"    δ_{i+1} = {d:.6f}  ({err:.3f}%)")

    # =========================================================================
    # Plot: 6-panel topology sweep
    # =========================================================================
    fig = plt.figure(figsize=(18, 14))
    fig.patch.set_facecolor("white")

    gs = GridSpec(2, 3, hspace=0.35, wspace=0.3)

    colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6", "#e67e22"]
    systems = list(results.keys())

    # Panel 1-5: Individual system δ convergence
    for idx, (name, color) in enumerate(zip(systems, colors)):
        bifs_s, deltas_s, delta_target = results[name]
        row, col = idx // 3, idx % 3
        ax = fig.add_subplot(gs[row, col])
        ax.set_facecolor("white")

        if deltas_s:
            n_vals = list(range(1, len(deltas_s) + 1))
            ax.plot(n_vals, deltas_s, "o-", color=color, markersize=10,
                    linewidth=2.5, zorder=5, label="Measured δₙ")
            ax.axhline(delta_target, color="black", linewidth=1.5,
                       linestyle="--", zorder=3,
                       label=f"δ = {delta_target:.4f}")

            for i, d in enumerate(deltas_s):
                err = abs(d - delta_target) / delta_target * 100
                ax.annotate(f"{d:.3f}\n({err:.1f}%)",
                            (n_vals[i], d),
                            textcoords="offset points", xytext=(12, 8),
                            fontsize=8, fontweight="bold",
                            bbox=dict(boxstyle="round,pad=0.2",
                                      facecolor="lightyellow",
                                      edgecolor="gray", alpha=0.9))
        else:
            ax.text(0.5, 0.5, "No bifurcations\nresolved",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=12, color="gray")

        ax.set_xlabel("Ratio index n", fontsize=10)
        ax.set_ylabel("δₙ", fontsize=10)
        ax.set_title(name, fontsize=12, fontweight="bold", color=color)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.2)
        if delta_target < 5.5:
            ax.set_ylim(3.5, 5.8)
        else:
            ax.set_ylim(4.5, 7.5)

    # Panel 6: Combined universality plot
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.set_facecolor("white")

    # Plot best δ from each z=2 system
    z2_systems = []
    z2_best_deltas = []
    z2_colors = []
    for name, color in zip(systems, colors):
        bifs_s, deltas_s, delta_target = results[name]
        if deltas_s and abs(delta_target - DELTA_Z2) < 0.1:
            best_d = deltas_s[-1]  # Last (most converged) ratio
            z2_systems.append(name.replace("\n", " "))
            z2_best_deltas.append(best_d)
            z2_colors.append(color)

    # Also get cubic best
    cubic_bifs, cubic_deltas, _ = results.get("Cubic\n(z=3)", ([], [], 0))

    if z2_best_deltas:
        x_pos = range(len(z2_best_deltas))
        bars = ax6.bar(x_pos, z2_best_deltas, color=z2_colors, alpha=0.8,
                       edgecolor="black", linewidth=0.5, zorder=5)
        ax6.axhline(DELTA_Z2, color="black", linewidth=2, linestyle="--",
                     zorder=3, label=f"δ(z=2) = {DELTA_Z2:.4f}")
        ax6.set_xticks(x_pos)
        ax6.set_xticklabels(z2_systems, fontsize=8, rotation=15)

        for i, (d, c) in enumerate(zip(z2_best_deltas, z2_colors)):
            err = abs(d - DELTA_Z2) / DELTA_Z2 * 100
            ax6.text(i, d + 0.02, f"{d:.3f}\n±{err:.1f}%",
                     ha="center", fontsize=7, fontweight="bold")

    ax6.set_ylabel("Best δ estimate", fontsize=10)
    ax6.set_title("Universality Test\n(z=2 systems)", fontsize=12,
                   fontweight="bold")
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.2, axis="y")
    ax6.set_ylim(3.0, 5.5)

    fig.suptitle("Decoherence Topology Sweep\n"
                 "Feigenbaum universality across coupling topologies",
                 fontsize=15, fontweight="bold", y=0.99)

    out = "/Users/lucianrandolph/Downloads/Projects/einstein_fractal_analysis/fig78_topology_sweep.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"\nSaved: {out}")
    plt.close()

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("UNIVERSALITY SUMMARY")
    print("=" * 60)
    print(f"\nTarget δ(z=2) = {DELTA_Z2:.10f}")
    print(f"Target δ(z=3) = {DELTA_Z3:.7f}\n")

    for name in systems:
        bifs_s, deltas_s, delta_target = results[name]
        clean_name = name.replace("\n", " ")
        if deltas_s:
            best = deltas_s[-1]
            err = abs(best - delta_target) / delta_target * 100
            print(f"  {clean_name:20s}: best δ = {best:.6f}  "
                  f"(error: {err:.3f}%)")
        else:
            print(f"  {clean_name:20s}: [no bifurcations resolved]")

    # Check universality
    z2_deltas = []
    for name in systems:
        bifs_s, deltas_s, delta_target = results[name]
        if deltas_s and abs(delta_target - DELTA_Z2) < 0.1:
            z2_deltas.append(deltas_s[-1])

    if len(z2_deltas) >= 2:
        spread = max(z2_deltas) - min(z2_deltas)
        mean_d = np.mean(z2_deltas)
        rel_spread = spread / mean_d * 100
        print(f"\n  z=2 universality: {len(z2_deltas)} systems, "
              f"spread = {spread:.4f} ({rel_spread:.2f}%)")
        print(f"  Mean δ(z=2) = {mean_d:.6f}")

    print("\nScript 78 complete.")


if __name__ == "__main__":
    main()
