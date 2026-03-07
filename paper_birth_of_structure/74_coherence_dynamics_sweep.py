#!/usr/bin/env python3
"""
Script 74 — Coherence Dynamics Sweep
=====================================
Time evolution of the off-diagonal density matrix element ρ₁₂(t).

This IS the coherence. It's what decays during decoherence.
- Weak coupling: ρ₁₂ oscillates cleanly (Rabi oscillations, coherent dynamics).
- Strong coupling: ρ₁₂ decays monotonically (fully decohered).
- Between: the transition region.

We plot ρ₁₂(t) at multiple γ values across the transition region.
Look for oscillation structure changing as γ increases.

Also: the DRIVEN qubit version, where a continuous Rabi drive competes
with dissipation. This is where nonlinear dynamics should emerge.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from typing import Tuple


# ==============================================================================
# Lindblad Superoperator (reused from Script 73)
# ==============================================================================

def build_lindblad_superoperator(
    omega: float,
    gamma_relax: float,
    gamma_dephase: float,
    drive_amplitude: float = 0.0,
) -> np.ndarray:
    """
    Build 4×4 Lindblad superoperator for a single qubit.

    Parameters
    ----------
    omega : float
        Qubit transition frequency.
    gamma_relax : float
        Relaxation rate (T₁).
    gamma_dephase : float
        Pure dephasing rate (T₂*).
    drive_amplitude : float
        Rabi drive amplitude Ω (continuous drive in x-direction).

    Returns
    -------
    L : np.ndarray, shape (4, 4), complex
    """
    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
    sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
    sigma_m = np.array([[0, 0], [1, 0]], dtype=complex)
    I2 = np.eye(2, dtype=complex)

    # Hamiltonian: H = (ω/2)σ_z + (Ω/2)σ_x
    H = (omega / 2.0) * sigma_z + (drive_amplitude / 2.0) * sigma_x

    # Commutator part
    L_H = -1j * (np.kron(H, I2) - np.kron(I2, H.T))

    # Dissipator
    L_D = np.zeros((4, 4), dtype=complex)

    # Relaxation: L = σ₋
    if gamma_relax > 0:
        Lop = sigma_m
        Ldag_L = Lop.conj().T @ Lop
        L_D += gamma_relax * (
            np.kron(Lop, Lop.conj())
            - 0.5 * np.kron(Ldag_L, I2)
            - 0.5 * np.kron(I2, Ldag_L.T)
        )

    # Pure dephasing: L = σ_z
    if gamma_dephase > 0:
        Lop = sigma_z
        Ldag_L = Lop.conj().T @ Lop
        L_D += gamma_dephase * (
            np.kron(Lop, Lop.conj())
            - 0.5 * np.kron(Ldag_L, I2)
            - 0.5 * np.kron(I2, Ldag_L.T)
        )

    return L_H + L_D


def evolve_density_matrix(
    L: np.ndarray,
    rho0_vec: np.ndarray,
    times: np.ndarray,
) -> np.ndarray:
    """
    Evolve vectorized density matrix under Lindbladian.

    ρ(t) = exp(Lt) ρ(0)

    Returns
    -------
    rho_t : np.ndarray, shape (len(times), 4)
        Vectorized density matrix at each time.
    """
    result = np.zeros((len(times), 4), dtype=complex)
    for i, t in enumerate(times):
        prop = expm(L * t)
        result[i] = prop @ rho0_vec
    return result


# ==============================================================================
# Part 1: Static Lindblad — Coherence Decay
# ==============================================================================

def plot_static_coherence() -> None:
    """Coherence ρ₁₂(t) for the static (undriven) qubit."""
    omega = 1.0

    # Initial state: |+⟩ = (|0⟩ + |1⟩)/√2
    # ρ₀ = |+⟩⟨+| = [[0.5, 0.5], [0.5, 0.5]]
    rho0 = np.array([0.5, 0.5, 0.5, 0.5], dtype=complex)

    # γ values spanning the transition
    gamma_values = [1e-3, 1e-2, 0.05, 0.1, 0.5, 1.0, 5.0, 20.0]
    times = np.linspace(0, 60, 3000)

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(
        "Coherence Dynamics ρ₁₂(t) — Static Qubit\n"
        "ω = 1.0, both relaxation and dephasing, initial state |+⟩",
        fontsize=14,
        fontweight="bold",
    )

    for idx, gamma in enumerate(gamma_values):
        ax = axes[idx // 4, idx % 4]
        L = build_lindblad_superoperator(omega, gamma, gamma)
        rho_t = evolve_density_matrix(L, rho0, times)

        # ρ₁₂ is the second element (index 1) in our vectorization
        rho12 = rho_t[:, 1]

        ax.plot(times, rho12.real, linewidth=0.8, color="#1f77b4", label="Re(ρ₁₂)")
        ax.plot(times, rho12.imag, linewidth=0.8, color="#ff7f0e", label="Im(ρ₁₂)")
        ax.plot(times, np.abs(rho12), linewidth=0.8, color="#2ca02c",
                linestyle="--", label="|ρ₁₂|")
        ax.set_title(f"γ = {gamma}", fontsize=11)
        ax.set_xlabel("t", fontsize=10)
        ax.set_ylabel("ρ₁₂(t)", fontsize=10)
        ax.set_ylim(-0.55, 0.55)
        ax.axhline(0, color="gray", linewidth=0.4)
        ax.grid(True, alpha=0.2)
        if idx == 0:
            ax.legend(fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    outpath = "/Users/lucianrandolph/Downloads/Projects/einstein_fractal_analysis/fig74a_static_coherence.png"
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    print(f"Saved: {outpath}")
    plt.close()


# ==============================================================================
# Part 2: Driven Qubit — Coherence Under Competition
# ==============================================================================

def plot_driven_coherence() -> None:
    """
    Coherence ρ₁₂(t) for the DRIVEN qubit.

    Rabi drive Ω competes with dissipation γ.
    This is the driven-dissipative system where nonlinear dynamics should emerge.
    """
    omega = 1.0
    drive = 2.0  # Rabi frequency — fixed

    # Initial state: ground state |0⟩
    # ρ₀ = |0⟩⟨0| = [[1, 0], [0, 0]]
    rho0 = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)

    # γ values: from drive-dominated to dissipation-dominated
    gamma_values = [1e-3, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
    times = np.linspace(0, 80, 4000)

    fig, axes = plt.subplots(3, 4, figsize=(20, 14))
    fig.suptitle(
        "Coherence Dynamics ρ₁₂(t) — Driven Qubit (Ω = 2.0)\n"
        "Competition between Rabi drive and dissipation",
        fontsize=14,
        fontweight="bold",
    )

    for idx, gamma in enumerate(gamma_values):
        ax = axes[idx // 4, idx % 4]
        L = build_lindblad_superoperator(omega, gamma, gamma, drive_amplitude=drive)
        rho_t = evolve_density_matrix(L, rho0, times)

        rho12 = rho_t[:, 1]

        ax.plot(times, rho12.real, linewidth=0.6, color="#1f77b4", label="Re(ρ₁₂)")
        ax.plot(times, rho12.imag, linewidth=0.6, color="#ff7f0e", label="Im(ρ₁₂)")
        ax.plot(times, np.abs(rho12), linewidth=0.6, color="#2ca02c",
                linestyle="--", label="|ρ₁₂|")
        ax.set_title(f"γ = {gamma}", fontsize=11)
        ax.set_xlabel("t", fontsize=10)
        ax.set_ylabel("ρ₁₂(t)", fontsize=10)
        ax.axhline(0, color="gray", linewidth=0.4)
        ax.grid(True, alpha=0.2)
        if idx == 0:
            ax.legend(fontsize=7)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    outpath = "/Users/lucianrandolph/Downloads/Projects/einstein_fractal_analysis/fig74b_driven_coherence.png"
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    print(f"Saved: {outpath}")
    plt.close()


# ==============================================================================
# Part 3: Driven Qubit — ρ₁₁(t) Population Dynamics
# ==============================================================================

def plot_driven_population() -> None:
    """
    Population ρ₁₁(t) for the driven qubit.

    The excited state population shows Rabi oscillations at weak coupling
    and monotonic equilibration at strong coupling. The transition between
    these regimes is where the structure lives.
    """
    omega = 1.0
    drive = 2.0

    rho0 = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)

    gamma_values = [1e-3, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
    times = np.linspace(0, 80, 4000)

    fig, axes = plt.subplots(3, 4, figsize=(20, 14))
    fig.suptitle(
        "Population Dynamics ρ₁₁(t) — Driven Qubit (Ω = 2.0)\n"
        "Rabi oscillations → damped oscillations → monotonic equilibration",
        fontsize=14,
        fontweight="bold",
    )

    for idx, gamma in enumerate(gamma_values):
        ax = axes[idx // 4, idx % 4]
        L = build_lindblad_superoperator(omega, gamma, gamma, drive_amplitude=drive)
        rho_t = evolve_density_matrix(L, rho0, times)

        # ρ₁₁ is the fourth element (index 3)
        rho11 = rho_t[:, 3].real

        ax.plot(times, rho11, linewidth=0.8, color="#d62728")
        ax.set_title(f"γ = {gamma}", fontsize=11)
        ax.set_xlabel("t", fontsize=10)
        ax.set_ylabel("ρ₁₁(t)", fontsize=10)
        ax.set_ylim(-0.05, 0.55)
        ax.axhline(0, color="gray", linewidth=0.4)
        ax.grid(True, alpha=0.2)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    outpath = "/Users/lucianrandolph/Downloads/Projects/einstein_fractal_analysis/fig74c_driven_population.png"
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    print(f"Saved: {outpath}")
    plt.close()


# ==============================================================================
# Main
# ==============================================================================

def main() -> None:
    print("=" * 60)
    print("Script 74 — Coherence Dynamics Sweep")
    print("=" * 60)

    print("\n--- Part 1: Static Qubit ---")
    plot_static_coherence()

    print("\n--- Part 2: Driven Qubit — Coherence ---")
    plot_driven_coherence()

    print("\n--- Part 3: Driven Qubit — Population ---")
    plot_driven_population()

    print("\nScript 74 complete.")


if __name__ == "__main__":
    main()
