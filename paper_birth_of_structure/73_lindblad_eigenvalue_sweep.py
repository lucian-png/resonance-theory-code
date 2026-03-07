#!/usr/bin/env python3
"""
Script 73 — Lindblad Eigenvalue Sweep
======================================
The Lucian Method applied to quantum decoherence.

System: Single qubit coupled to environment via Lindblad master equation.
Driving variable: system-environment coupling strength γ (10⁻⁶ to 10²).
Observable: Eigenvalues of the Lindbladian superoperator.

The Lindblad equation for a qubit with dephasing rate γ_φ and relaxation rate γ_1:
    dρ/dt = -i[H, ρ] + Σ_k γ_k (L_k ρ L_k† - ½{L_k† L_k, ρ})

We vectorize the density matrix (ρ → vec(ρ)) and build the 4×4 superoperator L
whose eigenvalues govern the decay rates and oscillation frequencies.

Question: Does the eigenvalue spectrum show bifurcation structure as γ is driven?
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

# ==============================================================================
# Lindblad Superoperator Construction
# ==============================================================================

def build_lindblad_superoperator(
    omega: float,
    gamma_relax: float,
    gamma_dephase: float,
) -> np.ndarray:
    """
    Build the 4×4 Lindblad superoperator for a single qubit.

    Parameters
    ----------
    omega : float
        Qubit transition frequency (energy splitting / ℏ).
    gamma_relax : float
        Relaxation rate (T₁ process — energy decay).
    gamma_dephase : float
        Pure dephasing rate (T₂* process — phase scrambling).

    Returns
    -------
    L : np.ndarray, shape (4, 4)
        The Lindbladian superoperator in the vectorized basis
        |ρ⟩ = (ρ₀₀, ρ₀₁, ρ₁₀, ρ₁₁)ᵀ
    """
    # Pauli matrices
    sigma_m = np.array([[0, 0], [1, 0]], dtype=complex)  # |0⟩⟨1| lowering
    sigma_p = np.array([[0, 1], [0, 0]], dtype=complex)  # |1⟩⟨0| raising
    sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)

    # Hamiltonian: H = (ω/2) σ_z
    H = (omega / 2.0) * sigma_z

    # Identity
    I2 = np.eye(2, dtype=complex)

    # Commutator part: -i(H⊗I - I⊗Hᵀ)
    L_H = -1j * (np.kron(H, I2) - np.kron(I2, H.T))

    # Dissipator for a single Lindblad operator L_k with rate γ_k:
    # γ_k * (L_k⊗L_k* - ½(L_k†L_k⊗I + I⊗L_k^T L_k*))
    L_D = np.zeros((4, 4), dtype=complex)

    # Relaxation channel: L₁ = σ₋ (decay from |1⟩ to |0⟩)
    if gamma_relax > 0:
        Lop = sigma_m
        Ldag_L = Lop.conj().T @ Lop
        L_D += gamma_relax * (
            np.kron(Lop, Lop.conj())
            - 0.5 * np.kron(Ldag_L, I2)
            - 0.5 * np.kron(I2, Ldag_L.T)
        )

    # Pure dephasing channel: L₂ = σ_z
    if gamma_dephase > 0:
        Lop = sigma_z
        Ldag_L = Lop.conj().T @ Lop
        L_D += gamma_dephase * (
            np.kron(Lop, Lop.conj())
            - 0.5 * np.kron(Ldag_L, I2)
            - 0.5 * np.kron(I2, Ldag_L.T)
        )

    return L_H + L_D


def compute_eigenvalues(
    omega: float,
    gamma_values: np.ndarray,
    coupling_mode: str = "both",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sweep γ and compute Lindbladian eigenvalues at each point.

    Parameters
    ----------
    omega : float
        Qubit frequency.
    gamma_values : np.ndarray
        Array of coupling strengths to sweep.
    coupling_mode : str
        "relax" — relaxation only (T₁)
        "dephase" — pure dephasing only (T₂*)
        "both" — equal relaxation and dephasing

    Returns
    -------
    real_parts : np.ndarray, shape (N, 4)
        Real parts of eigenvalues at each γ.
    imag_parts : np.ndarray, shape (N, 4)
        Imaginary parts of eigenvalues at each γ.
    """
    N = len(gamma_values)
    real_parts = np.zeros((N, 4))
    imag_parts = np.zeros((N, 4))

    for i, gamma in enumerate(gamma_values):
        if coupling_mode == "relax":
            g_r, g_d = gamma, 0.0
        elif coupling_mode == "dephase":
            g_r, g_d = 0.0, gamma
        else:  # both
            g_r, g_d = gamma, gamma

        L = build_lindblad_superoperator(omega, g_r, g_d)
        evals = np.linalg.eigvals(L)

        # Sort by real part (most negative last)
        idx = np.argsort(-evals.real)
        evals = evals[idx]

        real_parts[i] = evals.real
        imag_parts[i] = evals.imag

    return real_parts, imag_parts


# ==============================================================================
# Main Sweep
# ==============================================================================

def main() -> None:
    # Qubit frequency — normalized to 1
    omega = 1.0

    # Coupling strength sweep: 10⁻⁶ to 10²
    gamma_values = np.logspace(-6, 2, 2000)

    print("=" * 60)
    print("Script 73 — Lindblad Eigenvalue Sweep")
    print("=" * 60)
    print(f"Qubit frequency ω = {omega}")
    print(f"γ range: [{gamma_values[0]:.1e}, {gamma_values[-1]:.1e}]")
    print(f"Points: {len(gamma_values)}")
    print()

    # ---- Three coupling topologies ----
    modes = ["relax", "dephase", "both"]
    mode_labels = [
        "Relaxation Only (T₁)",
        "Pure Dephasing (T₂*)",
        "Both Channels",
    ]

    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    fig.suptitle(
        "Lindblad Eigenvalue Spectrum vs. Coupling Strength γ\n"
        "Single Qubit — Three Coupling Topologies",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    for row, (mode, label) in enumerate(zip(modes, mode_labels)):
        print(f"Computing: {label}...")
        real_parts, imag_parts = compute_eigenvalues(omega, gamma_values, mode)

        # --- Left column: Real parts (decay rates) ---
        ax_r = axes[row, 0]
        for j in range(4):
            ax_r.semilogx(gamma_values, real_parts[:, j], linewidth=0.8)
        ax_r.set_xlabel("γ (coupling strength)", fontsize=11)
        ax_r.set_ylabel("Re(λ) — decay rate", fontsize=11)
        ax_r.set_title(f"{label} — Real Parts", fontsize=12)
        ax_r.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        ax_r.grid(True, alpha=0.3)

        # --- Right column: Imaginary parts (oscillation frequencies) ---
        ax_i = axes[row, 1]
        for j in range(4):
            ax_i.semilogx(gamma_values, imag_parts[:, j], linewidth=0.8)
        ax_i.set_xlabel("γ (coupling strength)", fontsize=11)
        ax_i.set_ylabel("Im(λ) — oscillation frequency", fontsize=11)
        ax_i.set_title(f"{label} — Imaginary Parts", fontsize=12)
        ax_i.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        ax_i.grid(True, alpha=0.3)

        # Report eigenvalue structure at key points
        for g_test in [1e-4, 1e-1, 1.0, 10.0]:
            idx = np.argmin(np.abs(gamma_values - g_test))
            evals_here = real_parts[idx] + 1j * imag_parts[idx]
            print(f"  γ = {g_test:.0e}: λ = {np.array2string(evals_here, precision=4)}")
        print()

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    outpath = "/Users/lucianrandolph/Downloads/Projects/einstein_fractal_analysis/fig73_lindblad_eigenvalue_sweep.png"
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    print(f"Saved: {outpath}")
    plt.close()

    # ---- Detailed view: eigenvalue crossings / near-bifurcation structure ----
    print("\nComputing detailed 'both channels' sweep for crossing analysis...")
    real_parts, imag_parts = compute_eigenvalues(omega, gamma_values, "both")

    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig2.suptitle(
        "Lindblad Eigenvalue Detail — Both Channels\n"
        "Looking for Crossings, Mergers, and Bifurcation Structure",
        fontsize=13,
        fontweight="bold",
    )

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    labels_eig = ["λ₁ (steady state)", "λ₂", "λ₃", "λ₄"]

    for j in range(4):
        ax1.semilogx(gamma_values, real_parts[:, j], color=colors[j],
                      linewidth=1.2, label=labels_eig[j])
        ax2.semilogx(gamma_values, imag_parts[:, j], color=colors[j],
                      linewidth=1.2, label=labels_eig[j])

    ax1.set_xlabel("γ", fontsize=12)
    ax1.set_ylabel("Re(λ)", fontsize=12)
    ax1.set_title("Decay Rates", fontsize=12)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(0, color="gray", linewidth=0.5, linestyle="--")

    ax2.set_xlabel("γ", fontsize=12)
    ax2.set_ylabel("Im(λ)", fontsize=12)
    ax2.set_title("Oscillation Frequencies", fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(0, color="gray", linewidth=0.5, linestyle="--")

    plt.tight_layout()
    outpath2 = "/Users/lucianrandolph/Downloads/Projects/einstein_fractal_analysis/fig73b_eigenvalue_detail.png"
    plt.savefig(outpath2, dpi=200, bbox_inches="tight")
    print(f"Saved: {outpath2}")
    plt.close()

    print("\nScript 73 complete.")


if __name__ == "__main__":
    main()
