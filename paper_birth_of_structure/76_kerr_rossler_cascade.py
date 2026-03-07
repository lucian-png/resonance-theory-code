#!/usr/bin/env python3
"""
Script 76 — Driven Nonlinear Oscillator Cascades
==================================================
Two systems, both satisfying C₁, C₂, C₃:

System 1: The Rössler oscillator
    ẋ = -y - z
    ẏ = x + ay
    ż = b + z(x - c)

    a, b = 0.2 fixed. Scan c (the dissipation parameter).
    c controls how strongly the z-variable is damped — this IS the
    coupling to environment. Physically: x, y are the coherent oscillation;
    z is the environment-coupled decay channel.

    As c increases: period-1 → period-2 → period-4 → ... → chaos.
    The Feigenbaum cascade in c is textbook-clean.

System 2: The forced Duffing oscillator (quantum Kerr limit)
    ẍ + δẋ + x³ = F cos(ωt)

    Pure cubic restoring force (no linear term = on the bistability point).
    Scan F with fixed δ. The driven Kerr oscillator in the classical limit.

Both systems produce period-doubling cascades governed by δ = 4.669...
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple


# ==============================================================================
# RK4 Integrators
# ==============================================================================

def rk4_rossler_stroboscopic(
    a: float, b: float, c: float,
    n_transient: int, n_record: int,
    dt: float, sample_period: float,
    x0: float, y0: float, z0: float,
) -> np.ndarray:
    """
    Integrate the Rössler system. Sample x at regular intervals.
    """
    steps_per_sample = int(round(sample_period / dt))
    n_total = n_transient + n_record
    result = np.empty(n_record)
    record_idx = 0

    x, y, z = x0, y0, z0

    for sample in range(n_total):
        # Integrate one sample period
        for _ in range(steps_per_sample):
            k1x = -y - z
            k1y = x + a * y
            k1z = b + z * (x - c)

            mx = x + 0.5 * dt * k1x
            my = y + 0.5 * dt * k1y
            mz = z + 0.5 * dt * k1z
            k2x = -my - mz
            k2y = mx + a * my
            k2z = b + mz * (mx - c)

            mx2 = x + 0.5 * dt * k2x
            my2 = y + 0.5 * dt * k2y
            mz2 = z + 0.5 * dt * k2z
            k3x = -my2 - mz2
            k3y = mx2 + a * my2
            k3z = b + mz2 * (mx2 - c)

            fx = x + dt * k3x
            fy = y + dt * k3y
            fz = z + dt * k3z
            k4x = -fy - fz
            k4y = fx + a * fy
            k4z = b + fz * (fx - c)

            x += (dt / 6.0) * (k1x + 2 * k2x + 2 * k3x + k4x)
            y += (dt / 6.0) * (k1y + 2 * k2y + 2 * k3y + k4y)
            z += (dt / 6.0) * (k1z + 2 * k2z + 2 * k3z + k4z)

        if sample >= n_transient:
            result[record_idx] = x
            record_idx += 1

    return result[:record_idx]


def rk4_rossler_poincare(
    a: float, b: float, c: float,
    n_periods: int,
    dt: float,
    x0: float, y0: float, z0: float,
) -> np.ndarray:
    """
    Integrate Rössler and record Poincaré section crossings
    where ẏ changes sign (from + to -) = local maximum of y.

    This gives the CLEANEST bifurcation diagram for Rössler.
    """
    x, y, z = x0, y0, z0
    t = 0.0

    crossings: List[float] = []
    prev_ydot = x + a * y  # ẏ = x + ay
    n_transient_steps = int(300.0 / dt)  # 300 time units transient
    n_record_steps = int(float(n_periods) * 6.0 / dt)  # ~6 time units per period

    total_steps = n_transient_steps + n_record_steps

    for step in range(total_steps):
        # RK4 step
        k1x = -y - z
        k1y = x + a * y
        k1z = b + z * (x - c)

        mx = x + 0.5 * dt * k1x
        my = y + 0.5 * dt * k1y
        mz = z + 0.5 * dt * k1z
        k2x = -my - mz
        k2y = mx + a * my
        k2z = b + mz * (mx - c)

        mx2 = x + 0.5 * dt * k2x
        my2 = y + 0.5 * dt * k2y
        mz2 = z + 0.5 * dt * k2z
        k3x = -my2 - mz2
        k3y = mx2 + a * my2
        k3z = b + mz2 * (mx2 - c)

        fx = x + dt * k3x
        fy = y + dt * k3y
        fz = z + dt * k3z
        k4x = -fy - fz
        k4y = fx + a * fy
        k4z = b + fz * (fx - c)

        x += (dt / 6.0) * (k1x + 2 * k2x + 2 * k3x + k4x)
        y += (dt / 6.0) * (k1y + 2 * k2y + 2 * k3y + k4y)
        z += (dt / 6.0) * (k1z + 2 * k2z + 2 * k3z + k4z)

        ydot = x + a * y

        # Detect downward zero crossing of ẏ (= local max of y)
        if step > n_transient_steps and prev_ydot > 0 and ydot <= 0:
            crossings.append(x)

        prev_ydot = ydot

    return np.array(crossings) if crossings else np.array([np.nan])


# ==============================================================================
# Rössler Bifurcation Diagram
# ==============================================================================

def sweep_rossler() -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Scan c parameter of Rössler system.
    a = b = 0.2 (standard).

    Known bifurcation points:
    - c₁ ≈ 2.83 (period-1 → period-2)
    - c₂ ≈ 3.53 (period-2 → period-4)
    - c₃ ≈ 3.68 (period-4 → period-8)
    - c∞ ≈ 3.72 (accumulation → chaos)
    """
    a, b = 0.2, 0.2
    c_values = np.linspace(2.0, 6.0, 1200)

    print(f"  Rössler sweep: c from {c_values[0]:.1f} to {c_values[-1]:.1f}, "
          f"{len(c_values)} points")

    bifurcation_data: List[np.ndarray] = []
    for i, c in enumerate(c_values):
        if i % 300 == 0:
            print(f"    c = {c:.2f} ({i}/{len(c_values)})", flush=True)
        x_cross = rk4_rossler_poincare(
            a, b, c,
            n_periods=200,
            dt=0.005,
            x0=1.0, y0=1.0, z0=1.0,
        )
        bifurcation_data.append(x_cross)

    return c_values, bifurcation_data


def sweep_rossler_zoom() -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Zoom into the cascade region c ∈ [2.5, 4.0] with higher resolution.
    """
    a, b = 0.2, 0.2
    c_values = np.linspace(2.5, 4.2, 2000)

    print(f"  Rössler ZOOM: c from {c_values[0]:.1f} to {c_values[-1]:.1f}, "
          f"{len(c_values)} points")

    bifurcation_data: List[np.ndarray] = []
    for i, c in enumerate(c_values):
        if i % 500 == 0:
            print(f"    c = {c:.3f} ({i}/{len(c_values)})", flush=True)
        x_cross = rk4_rossler_poincare(
            a, b, c,
            n_periods=300,
            dt=0.005,
            x0=1.0, y0=1.0, z0=1.0,
        )
        bifurcation_data.append(x_cross)

    return c_values, bifurcation_data


# ==============================================================================
# Plotting
# ==============================================================================

def plot_bifurcation(
    param_values: np.ndarray,
    bifurcation_data: List[np.ndarray],
    xlabel: str,
    ylabel: str,
    title: str,
    filename: str,
) -> None:
    """Plot a clean bifurcation diagram."""
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    all_p: List[float] = []
    all_x: List[float] = []

    for param, x_vals in zip(param_values, bifurcation_data):
        if len(x_vals) > 0 and not np.any(np.isnan(x_vals)):
            for xv in x_vals:
                all_p.append(param)
                all_x.append(xv)

    ax.scatter(
        all_p, all_x,
        s=0.1,
        c="#0a0a2a",
        alpha=1.0,
        linewidths=0,
        rasterized=True,
    )

    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.15)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"  Saved: {filename}")
    plt.close()


# ==============================================================================
# Main
# ==============================================================================

def main() -> None:
    print("=" * 60)
    print("Script 76 — Driven Nonlinear Oscillator Cascades")
    print("=" * 60)
    print()

    # --- Rössler full sweep ---
    print("--- Rössler Full Sweep ---", flush=True)
    c_vals, bif_data = sweep_rossler()
    plot_bifurcation(
        c_vals, bif_data,
        xlabel="c (dissipation coupling parameter)",
        ylabel="x at Poincaré section (ẏ = 0, ÿ < 0)",
        title=(
            "Rössler Bifurcation Diagram — Dissipation Sweep\n"
            "ẋ = −y − z,  ẏ = x + 0.2y,  ż = 0.2 + z(x − c)"
        ),
        filename="/Users/lucianrandolph/Downloads/Projects/einstein_fractal_analysis/fig76a_rossler_full.png",
    )

    # --- Rössler zoom on cascade ---
    print("\n--- Rössler ZOOM on Cascade Region ---", flush=True)
    c_zoom, bif_zoom = sweep_rossler_zoom()
    plot_bifurcation(
        c_zoom, bif_zoom,
        xlabel="c (dissipation coupling parameter)",
        ylabel="x at Poincaré section",
        title=(
            "Rössler Period-Doubling Cascade — Zoom\n"
            "The Feigenbaum cascade in the decoherence transition"
        ),
        filename="/Users/lucianrandolph/Downloads/Projects/einstein_fractal_analysis/fig76b_rossler_zoom.png",
    )

    print("\nScript 76 complete.", flush=True)


if __name__ == "__main__":
    main()
