#!/usr/bin/env python3
"""
Script 75 — Duffing Oscillator Bifurcation Diagram
====================================================
The driven dissipative Duffing oscillator:

    ẍ + δẋ - x + x³ = F cos(ωt)

This is the classical limit of the driven quantum Kerr oscillator.
It IS the decoherence problem with a nonlinear medium:
- x is the field quadrature (classical limit of ⟨a + a†⟩)
- δ is the damping = system-environment coupling = γ
- F is the coherent drive
- The -x + x³ is the double-well potential (Kerr nonlinearity)

C₁: Bounded (double-well potential confines the trajectory)
C₂: Nonlinear (x³ term)
C₃: Coupled (δ damping = environment coupling)

All three Lucian Law prerequisites active simultaneously.

Sweep 1: Scan F (drive amplitude) with fixed δ — the classic cascade.
Sweep 2: Scan δ (damping/decoherence rate) with fixed F — the γ story.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
import sys


# ==============================================================================
# Fast Fixed-Step RK4 Integrator
# ==============================================================================

def rk4_duffing_stroboscopic(
    delta: float,
    F: float,
    omega: float,
    n_transient: int,
    n_record: int,
    steps_per_period: int,
    x0: float,
    v0: float,
) -> np.ndarray:
    """
    Integrate the Duffing oscillator using fixed-step RK4.
    Return stroboscopic samples (one per drive period) after transient.

    Much faster than adaptive solve_ivp for this use case.
    """
    T = 2.0 * np.pi / omega
    dt = T / steps_per_period
    n_total_periods = n_transient + n_record
    n_steps_per_period = steps_per_period

    x, v = x0, v0
    result = np.empty(n_record)
    record_idx = 0

    for period in range(n_total_periods):
        # Integrate one full drive period
        for step in range(n_steps_per_period):
            t = (period * n_steps_per_period + step) * dt

            # RK4 step
            k1x = v
            k1v = -delta * v + x - x**3 + F * np.cos(omega * t)

            xm = x + 0.5 * dt * k1x
            vm = v + 0.5 * dt * k1v
            tm = t + 0.5 * dt
            k2x = vm
            k2v = -delta * vm + xm - xm**3 + F * np.cos(omega * tm)

            xm2 = x + 0.5 * dt * k2x
            vm2 = v + 0.5 * dt * k2v
            k3x = vm2
            k3v = -delta * vm2 + xm2 - xm2**3 + F * np.cos(omega * tm)

            xf = x + dt * k3x
            vf = v + dt * k3v
            tf = t + dt
            k4x = vf
            k4v = -delta * vf + xf - xf**3 + F * np.cos(omega * tf)

            x += (dt / 6.0) * (k1x + 2 * k2x + 2 * k3x + k4x)
            v += (dt / 6.0) * (k1v + 2 * k2v + 2 * k3v + k4v)

        # Stroboscopic sample at end of period
        if period >= n_transient:
            result[record_idx] = x
            record_idx += 1

    return result[:record_idx]


# ==============================================================================
# Sweep 1: Scan F (drive amplitude) — The Classic Cascade
# ==============================================================================

def sweep_drive_amplitude() -> Tuple[np.ndarray, List[np.ndarray]]:
    """Scan drive amplitude F across the period-doubling regime."""
    delta = 0.25
    omega = 1.0

    F_values = np.linspace(0.20, 0.50, 800)

    print(f"  Sweep 1: F from {F_values[0]:.2f} to {F_values[-1]:.2f}, {len(F_values)} points")

    bifurcation_data: List[np.ndarray] = []
    # Run from BOTH wells to capture coexisting attractors
    ics = [(0.5, 0.0), (-0.5, 0.0), (1.0, 0.0), (-1.0, 0.0)]
    for i, F in enumerate(F_values):
        if i % 200 == 0:
            print(f"    F = {F:.3f} ({i}/{len(F_values)})", flush=True)
        all_pts: List[float] = []
        for x0, v0 in ics:
            x_poin = rk4_duffing_stroboscopic(
                delta, F, omega,
                n_transient=300, n_record=100,
                steps_per_period=200, x0=x0, v0=v0,
            )
            all_pts.extend(x_poin.tolist())
        bifurcation_data.append(np.array(all_pts))

    return F_values, bifurcation_data


# ==============================================================================
# Sweep 2: Scan δ (damping) — The Decoherence Sweep
# ==============================================================================

def sweep_damping() -> Tuple[np.ndarray, List[np.ndarray]]:
    """Scan damping δ (= environment coupling γ) from weak to strong."""
    F = 0.40
    omega = 1.0

    delta_values = np.linspace(0.05, 0.60, 800)

    print(f"  Sweep 2: δ from {delta_values[0]:.2f} to {delta_values[-1]:.2f}, {len(delta_values)} points")

    bifurcation_data: List[np.ndarray] = []
    ics = [(0.5, 0.0), (-0.5, 0.0), (1.0, 0.0), (-1.0, 0.0)]
    for i, delta_val in enumerate(delta_values):
        if i % 200 == 0:
            print(f"    δ = {delta_val:.3f} ({i}/{len(delta_values)})", flush=True)
        all_pts: List[float] = []
        for x0, v0 in ics:
            x_poin = rk4_duffing_stroboscopic(
                delta_val, F, omega,
                n_transient=300, n_record=100,
                steps_per_period=200, x0=x0, v0=v0,
            )
            all_pts.extend(x_poin.tolist())
        bifurcation_data.append(np.array(all_pts))

    return delta_values, bifurcation_data


# ==============================================================================
# Plotting
# ==============================================================================

def plot_bifurcation_diagram(
    param_values: np.ndarray,
    bifurcation_data: List[np.ndarray],
    xlabel: str,
    title: str,
    filename: str,
) -> None:
    """Plot the bifurcation diagram."""
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Collect all points for scatter plot
    all_x_param: List[float] = []
    all_x_vals: List[float] = []

    for param, x_vals in zip(param_values, bifurcation_data):
        if len(x_vals) > 0 and not np.any(np.isnan(x_vals)):
            for xv in x_vals:
                all_x_param.append(param)
                all_x_vals.append(xv)

    ax.scatter(
        all_x_param,
        all_x_vals,
        s=0.2,
        c="#1a1a2e",
        alpha=1.0,
        linewidths=0,
        rasterized=True,
    )

    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel("x (Poincaré section)", fontsize=13)
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
    print("Script 75 — Duffing Oscillator Bifurcation Diagram")
    print("=" * 60)
    print(flush=True)

    # --- Sweep 1: Drive amplitude F ---
    print("--- Sweep 1: Drive Amplitude F ---", flush=True)
    F_values, bif_F = sweep_drive_amplitude()
    plot_bifurcation_diagram(
        F_values, bif_F,
        xlabel="F (drive amplitude)",
        title=(
            "Duffing Bifurcation Diagram — Drive Amplitude Sweep\n"
            "ẍ + 0.25ẋ − x + x³ = F cos(t)  |  Poincaré section at T = 2π"
        ),
        filename="/Users/lucianrandolph/Downloads/Projects/einstein_fractal_analysis/fig75a_duffing_bif_F.png",
    )

    # --- Sweep 2: Damping δ (= decoherence rate γ) ---
    print("\n--- Sweep 2: Damping δ (Decoherence Rate) ---", flush=True)
    delta_values, bif_delta = sweep_damping()
    plot_bifurcation_diagram(
        delta_values, bif_delta,
        xlabel="δ (damping = environment coupling γ)",
        title=(
            "Duffing Bifurcation Diagram — Damping Sweep\n"
            "ẍ + δẋ − x + x³ = 0.40 cos(t)  |  Poincaré section at T = 2π"
        ),
        filename="/Users/lucianrandolph/Downloads/Projects/einstein_fractal_analysis/fig75b_duffing_bif_delta.png",
    )

    print("\nScript 75 complete.", flush=True)


if __name__ == "__main__":
    main()
