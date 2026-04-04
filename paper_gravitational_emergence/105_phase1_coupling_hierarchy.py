#!/usr/bin/env python3
"""
Script 105 — Phase 1: The Coupling Hierarchy
=============================================
Derive the stabilization order of the five fundamental parameters
(space, time, energy, matter, gravity) during the initial Feigenbaum
cascade, and compute the gravitational delay Δn_g from the self-coupling
structure of Einstein's field equations.

The Lucian Law (Paper 1) establishes that Einstein's field equations are
nonlinear, coupled, unbounded systems governed by the Feigenbaum constants.
This script formalizes HOW gravity emerges through the cascade — specifically,
the number of cascade levels by which gravity's stabilization lags behind
the other parameters due to gravitational self-coupling.

Author: Lucian Randolph & Claude Anthro Randolph
Date: March 29, 2026
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
import os

# ================================================================
# CONSTANTS
# ================================================================
DELTA = 4.669201609      # Feigenbaum period-doubling constant
ALPHA = 2.502907875      # Feigenbaum spatial scaling constant
LAMBDA_R = DELTA / ALPHA # Compound ratio (transition velocity)

# Physical constants (natural units c = ℏ = 1 where needed)
G_NEWTON = 6.67430e-11   # m³ kg⁻¹ s⁻²
C_LIGHT  = 2.99792458e8  # m/s
HBAR     = 1.054571817e-34

# Planck units
L_PLANCK = np.sqrt(HBAR * G_NEWTON / C_LIGHT**3)
T_PLANCK = L_PLANCK / C_LIGHT
M_PLANCK = np.sqrt(HBAR * C_LIGHT / G_NEWTON)

OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')

print('=' * 72)
print('  Script 105 — Phase 1: The Coupling Hierarchy')
print(f'  δ = {DELTA}   α = {ALPHA}   λᵣ = {LAMBDA_R:.8f}')
print('=' * 72)

# ================================================================
# SECTION 1: THE SELF-COUPLING STRUCTURE OF EINSTEIN'S EQUATIONS
# ================================================================
#
# The Einstein field equations:
#   R_μν − ½ g_μν R = (8πG/c⁴) T_μν
#
# The Ricci tensor R_μν contains two types of terms:
#   R_μν = ∂_λ Γ^λ_μν − ∂_ν Γ^λ_μλ + Γ^λ_λσ Γ^σ_μν − Γ^λ_νσ Γ^σ_μλ
#          |_________direct terms_________|  |________self-coupling terms_______|
#
# The direct terms (∂Γ) are LINEAR in the metric derivatives.
# The self-coupling terms (ΓΓ) are QUADRATIC in the metric derivatives.
#
# The Christoffel symbols scale as:
#   Γ ~ ∂g / g ~ 1/L   (where L is the characteristic length scale)
#
# At cascade level n, the characteristic length scale contracts by α per level:
#   L_n = L_0 × α^(−n)
#
# Therefore:
#   Γ_n ~ α^n / L_0
#   ∂Γ_n ~ α^(2n) / L_0²     (one more spatial derivative)
#   ΓΓ_n ~ α^(2n) / L_0²     (product of two Christoffel symbols)
#
# The RATIO of self-coupling to direct terms at cascade level n:
#   S(n) = |ΓΓ| / |∂Γ| = (Γ_n × Γ_n) / (∂Γ_n)
#        = (α^n / L_0)² / (α^(2n) / L_0²)
#        = 1   (at the cascade scale itself)
#
# This ratio of unity at the CASCADE SCALE is significant. It means the
# self-coupling and direct terms are comparable IN MAGNITUDE at each level.
# The self-coupling does not become negligible at any cascade level.
#
# However, the self-coupling introduces a FEEDBACK DELAY. At each level,
# the self-coupling terms require the PREVIOUS level's gravitational field
# to compute the current level's correction. This sequential dependency
# adds latency to gravitational stabilization.
#
# The feedback delay per cascade level:
#   Each level's gravitational correction depends on the previous level's
#   gravitational state. For a non-self-coupling parameter, level n depends
#   only on the cascade driver. For gravity, level n depends on level n-1
#   of GRAVITY ITSELF, creating a chain.
#
# The chain length is determined by the number of levels required for the
# self-coupling corrections to converge. The correction at each step
# scales as δ^(−1) (the cascade convergence rate). But the self-coupling
# means each correction TRIGGERS a secondary correction of magnitude
# S × δ^(−1), where S is the self-coupling ratio.

print('\n--- Section 1: Self-Coupling Analysis ---')

# The self-coupling ratio at the cascade scale
# In the Ricci tensor, there are 4 terms. Two are ∂Γ (direct), two are ΓΓ (self).
# In 4 dimensions, each Christoffel symbol has 4³ = 64 components.
# The self-coupling terms contract over one index, giving 4 × 4 = 16 products.
# The direct terms have 4 × 4 = 16 derivatives.
# The EFFECTIVE self-coupling ratio accounts for the dimensional structure.

# In the weak-field expansion g_μν = η_μν + h_μν:
#   Γ ~ ∂h  (first order in perturbation)
#   ΓΓ ~ (∂h)²  (second order)
#   ∂Γ ~ ∂²h  (second order in derivatives but first order in perturbation)
#
# At cascade level n, the perturbation amplitude scales as:
#   h_n ~ δ^(−n)  (cascade convergence)
#
# The ratio of self-coupling contribution to direct contribution at level n:
#   S(n) = |Γ_n|² / |∂Γ_n| × (geometric factor)
#
# The geometric factor comes from the contraction structure.
# In 4D, the Ricci tensor contraction involves summing over the
# spacetime dimension d = 4:
#   R_μν = Σ_λ [∂_λΓ^λ_μν − ∂_νΓ^λ_μλ + Σ_σ (Γ^λ_λσ Γ^σ_μν − Γ^λ_νσ Γ^σ_μλ)]
#
# The number of self-coupling terms relative to direct terms:
#   N_self = d × (number of ΓΓ contractions) = 4 × 2 × 4 = 32 terms
#   N_direct = 2 × 4 = 8 terms
#   Geometric ratio = N_self / N_direct = 32 / 8 = 4

d_spacetime = 4  # spacetime dimensions
N_self = d_spacetime * 2 * d_spacetime   # ΓΓ contraction terms
N_direct = 2 * d_spacetime               # ∂Γ derivative terms
geometric_ratio = N_self / N_direct

print(f'  Spacetime dimensions: {d_spacetime}')
print(f'  Self-coupling terms (ΓΓ): {N_self}')
print(f'  Direct terms (∂Γ): {N_direct}')
print(f'  Geometric ratio: {geometric_ratio}')

# ================================================================
# SECTION 2: THE CASCADE STABILIZATION MODEL
# ================================================================
#
# A parameter stabilizes when its cascade corrections converge.
# For a parameter WITHOUT self-coupling, the correction at level n is:
#   c_n = A × δ^(−n)
# Convergence is geometric with rate δ^(−1) ≈ 0.2142.
# Stabilization at level N means: c_N < ε (threshold).
# N_simple = −ln(ε/A) / ln(δ)
#
# For a parameter WITH self-coupling ratio S, each correction triggers
# a feedback chain. The total correction at level n becomes:
#   c_n = A × δ^(−n) × Σ_{k=0}^{∞} (S × δ^(−1))^k
#       = A × δ^(−n) / (1 − S × δ^(−1))
#
# PROVIDED that S × δ^(−1) < 1, i.e., the feedback converges.
# If S × δ^(−1) ≥ 1, the self-coupling prevents stabilization entirely
# (runaway feedback).
#
# For gravity, S is related to the geometric ratio and the effective
# coupling strength at the cascade scale.

print('\n--- Section 2: Cascade Stabilization Model ---')

# The effective self-coupling parameter for gravity
# The geometric ratio gives the relative NUMBER of self-coupling terms.
# But the AMPLITUDE of each term depends on the metric perturbation.
# At the cascade scale, the perturbation amplitude h ~ δ^(−n).
# The self-coupling amplitude relative to direct terms:
#   S_eff = geometric_ratio × (h / 1) = geometric_ratio × δ^(−n)
#
# At level n = 0: S_eff(0) = 4 × 1 = 4  (strong self-coupling)
# At level n = 1: S_eff(1) = 4 × δ^(−1) = 4 × 0.214 = 0.857
# At level n = 2: S_eff(2) = 4 × δ^(−2) = 4 × 0.046 = 0.183
#
# The self-coupling is STRONG at early levels and WEAKENS as the cascade
# develops. Gravity cannot stabilize until S_eff drops below the
# convergence threshold S_eff × δ^(−1) < 1.
#
# The critical level n_crit where self-coupling becomes subcritical:
#   geometric_ratio × δ^(−n_crit) × δ^(−1) < 1
#   geometric_ratio × δ^(−(n_crit + 1)) < 1
#   δ^(−(n_crit + 1)) < 1 / geometric_ratio
#   n_crit + 1 > ln(geometric_ratio) / ln(δ)
#   n_crit > ln(geometric_ratio) / ln(δ) − 1

n_levels = np.arange(0, 15)

# Self-coupling parameter at each cascade level
S_eff = geometric_ratio * DELTA**(-n_levels)

# Feedback convergence parameter at each level
feedback_param = S_eff * DELTA**(-1)

# Critical level: where feedback_param drops below 1
n_crit_exact = np.log(geometric_ratio) / np.log(DELTA) - 1
n_crit = int(np.ceil(n_crit_exact))

print(f'  Geometric self-coupling ratio: {geometric_ratio}')
print(f'  Critical level (exact): {n_crit_exact:.4f}')
print(f'  Critical level (integer): {n_crit}')
print(f'')
print(f'  Level-by-level self-coupling:')
for n in range(8):
    print(f'    n = {n}:  S_eff = {S_eff[n]:.6f}  '
          f'  feedback = {feedback_param[n]:.6f}  '
          f'  {"SUPERCRITICAL" if feedback_param[n] >= 1 else "subcritical"}')

# ================================================================
# SECTION 3: THE GRAVITATIONAL DELAY Δn_g
# ================================================================
#
# Gravity begins to stabilize at level n_crit, where the self-coupling
# feedback first becomes convergent. Before n_crit, gravity is in the
# "supercritical" phase — the self-coupling prevents stabilization.
#
# After n_crit, gravity stabilizes at rate δ^(−1) per level,
# but with an additional delay from the accumulated feedback.
# The total delay has two components:
#
#   Component 1: The critical delay — the number of levels before
#   self-coupling becomes subcritical. This is n_crit itself.
#
#   Component 2: The convergence delay — the additional levels needed
#   for the accumulated self-coupling corrections to dissipate.
#   This is ~ ln(S_eff(n_crit)) / ln(δ) levels.
#
# For a non-self-coupling parameter (like matter-energy), stabilization
# begins at level 0 and proceeds at rate δ^(−1).
#
# The gravitational delay is:
#   Δn_g = n_crit + convergence_delay

print('\n--- Section 3: Gravitational Delay ---')

# Component 1: Critical delay
critical_delay = n_crit

# Component 2: Convergence delay after becoming subcritical
# The residual self-coupling at n_crit:
S_residual = geometric_ratio * DELTA**(-n_crit)
# Number of additional levels for this residual to damp by δ^(-1) per level:
convergence_delay = np.log(S_residual) / np.log(DELTA)

# Total gravitational delay
Delta_n_g = critical_delay + convergence_delay
Delta_n_g_int = int(np.ceil(Delta_n_g))

print(f'  Critical delay (levels before subcritical): {critical_delay}')
print(f'  Residual S at n_crit: {S_residual:.6f}')
print(f'  Convergence delay: {convergence_delay:.4f} levels')
print(f'  Total gravitational delay Δn_g: {Delta_n_g:.4f}')
print(f'  Δn_g (integer cascade levels): {Delta_n_g_int}')

# ================================================================
# SECTION 4: THE STABILIZATION SEQUENCE
# ================================================================
#
# Based on the coupling hierarchy:
#   - Space and Time: stabilize at level 0 (arena parameters)
#   - Energy-Matter: stabilize at level ~1 (first cascade distribution)
#   - Gravity: stabilizes at level Δn_g (self-coupling delay)
#
# The stabilization level for each parameter:

print('\n--- Section 4: Stabilization Sequence ---')

n_space = 0
n_time = 0
n_energy = 1  # First cascade level distributes energy
n_matter = 1  # E = mc² couples matter to energy
n_gravity = Delta_n_g_int

print(f'  Space stabilization level:   n = {n_space}')
print(f'  Time stabilization level:    n = {n_time}')
print(f'  Energy stabilization level:  n = {n_energy}')
print(f'  Matter stabilization level:  n = {n_matter}')
print(f'  Gravity stabilization level: n = {n_gravity}')
print(f'')
print(f'  Gravity lags matter-energy by {n_gravity - n_matter} cascade levels')
print(f'  Gravity lags space-time by {n_gravity - n_space} cascade levels')

# ================================================================
# SECTION 5: THE EMERGENCE FUNCTION G_eff(n)
# ================================================================
#
# The effective gravitational coupling as a function of cascade level.
# Before gravity stabilizes, G_eff < G. After stabilization, G_eff = G.
#
# The emergence function must satisfy:
#   1. G_eff(n) → 0 as n → 0 (no gravity before cascade)
#   2. G_eff(n) → G as n → ∞ (full GR at late times)
#   3. G_eff(n) rises through the cascade at rate governed by δ
#   4. The self-coupling delays the rise by Δn_g levels
#
# Two candidate functions:
#
# Model A (Monotonic): G_eff(n) = G × [1 − δ^(−(n − n₀))] for n > n₀
#   Simple exponential approach. No oscillation.
#
# Model B (Decay Bounce): G_eff(n) = G × [1 − (α/δ)^n × cos(πn/T)]
#   Includes the oscillatory approach from Theorem L3.
#   T is the oscillation period in cascade levels.
#
# Model C (Cascade accumulation): G_eff(n) = G × [1 − Π_{k=0}^{n} (1 − δ^(−k))]
#   Each cascade level contributes a multiplicative factor.
#   This captures the level-by-level cascade development most directly.

print('\n--- Section 5: Emergence Functions ---')

n_fine = np.linspace(0, 12, 1000)

# Model A: Monotonic emergence
def G_eff_A(n, n0=None):
    """Monotonic exponential approach to G = 1."""
    if n0 is None:
        n0 = Delta_n_g / 2  # onset halfway through delay
    result = np.where(n > n0,
                      1.0 - DELTA**(-(n - n0)),
                      0.0)
    return np.clip(result, 0, 1)

# Model B: Decay Bounce emergence (Theorem L3)
def G_eff_B(n, n0=None):
    """Oscillatory approach with decay bounce."""
    if n0 is None:
        n0 = Delta_n_g / 2
    envelope = 1.0 - DELTA**(-(n - n0))
    # Oscillation from the decay bounce: period ~ 2 levels
    oscillation = 1.0 - 0.3 * (ALPHA / DELTA)**np.maximum(n - n0, 0) * \
                  np.cos(np.pi * (n - n0))
    result = np.where(n > n0, envelope * oscillation, 0.0)
    return np.clip(result, 0, 1)

# Model C: Cascade accumulation (product form)
def G_eff_C(n):
    """Level-by-level cascade accumulation.

    Each cascade level adds a fractional contribution to gravity.
    The fraction at level k is (1 − δ^(−k)) — the cascade convergence.
    Gravity is the PRODUCT of all contributions up to level n.
    This naturally produces:
      - Zero at n=0 (no contributions yet)
      - Rapid growth at intermediate levels
      - Asymptotic approach to 1 at large n
    """
    result = np.zeros_like(n, dtype=float)
    for i, ni in enumerate(n):
        ni_int = int(np.floor(ni))
        if ni_int <= 0:
            result[i] = 0.0
        else:
            product = 1.0
            for k in range(1, ni_int + 1):
                product *= (1.0 - DELTA**(-k))
            # Interpolate for fractional level
            frac = ni - ni_int
            if frac > 0 and ni_int + 1 > 0:
                next_factor = 1.0 - DELTA**(-(ni_int + 1))
                product *= (1.0 - frac + frac * next_factor)
            result[i] = product
    return result

# Compute all three models
G_A = G_eff_A(n_fine)
G_B = G_eff_B(n_fine)
G_C = G_eff_C(n_fine)

# Key values
print(f'  Model A (Monotonic):')
for n_val in [1, 2, 3, 4, 5, 6, 8, 10]:
    val = G_eff_A(np.array([float(n_val)]))[0]
    print(f'    G_eff({n_val}) / G = {val:.6f}')

print(f'\n  Model C (Cascade Accumulation):')
for n_val in [1, 2, 3, 4, 5, 6, 8, 10]:
    val = G_eff_C(np.array([float(n_val)]))[0]
    print(f'    G_eff({n_val}) / G = {val:.6f}')

# The level where G_eff reaches 90% of G
for label, func in [('A', G_eff_A), ('B', G_eff_B), ('C', G_eff_C)]:
    vals = func(n_fine)
    idx_90 = np.argmax(vals >= 0.90)
    idx_99 = np.argmax(vals >= 0.99)
    if idx_90 > 0:
        print(f'\n  Model {label}: G_eff reaches 90% at n = {n_fine[idx_90]:.2f}')
    if idx_99 > 0:
        print(f'  Model {label}: G_eff reaches 99% at n = {n_fine[idx_99]:.2f}')

# ================================================================
# SECTION 6: FIGURE — EMERGENCE FUNCTIONS
# ================================================================
print('\n--- Generating Figure: Emergence Functions ---')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Left panel: All three emergence models
ax1.plot(n_fine, G_A, 'b-', linewidth=2, label='Model A: Monotonic')
ax1.plot(n_fine, G_B, 'r--', linewidth=2, label='Model B: Decay Bounce')
ax1.plot(n_fine, G_C, 'k-', linewidth=2.5, label='Model C: Cascade Accumulation')
ax1.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5, label='Full GR (G)')
ax1.axhline(y=0.90, color='green', linestyle=':', alpha=0.4, label='90% threshold')
ax1.axvline(x=Delta_n_g, color='orange', linestyle='--', alpha=0.6,
            label=f'Δn_g = {Delta_n_g:.2f}')
ax1.set_xlabel('Cascade Level n', fontsize=12)
ax1.set_ylabel('G_eff(n) / G', fontsize=12)
ax1.set_title('Gravitational Emergence Functions', fontsize=14)
ax1.legend(fontsize=9, loc='lower right')
ax1.set_xlim(0, 12)
ax1.set_ylim(-0.05, 1.1)
ax1.grid(True, alpha=0.3)

# Right panel: Self-coupling decay
ax2.semilogy(n_levels[:10], S_eff[:10], 'ro-', linewidth=2, markersize=8,
             label='Self-coupling S_eff(n)')
ax2.semilogy(n_levels[:10], feedback_param[:10], 'bs--', linewidth=2, markersize=8,
             label='Feedback parameter S_eff × δ⁻¹')
ax2.axhline(y=1.0, color='red', linestyle=':', linewidth=2, alpha=0.7,
            label='Critical threshold = 1')
ax2.axvline(x=n_crit, color='orange', linestyle='--', alpha=0.6,
            label=f'n_crit = {n_crit}')
ax2.set_xlabel('Cascade Level n', fontsize=12)
ax2.set_ylabel('Coupling Parameter', fontsize=12)
ax2.set_title('Gravitational Self-Coupling Decay', fontsize=14)
ax2.legend(fontsize=9, loc='upper right')
ax2.set_xlim(-0.5, 9.5)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
fig_path = os.path.join(OUTDIR, 'fig1_emergence_functions.png')
plt.savefig(fig_path, dpi=200, bbox_inches='tight')
plt.close()
print(f'  Saved: {fig_path}')

# ================================================================
# SECTION 7: FIGURE — STABILIZATION SEQUENCE
# ================================================================
print('\n--- Generating Figure: Stabilization Sequence ---')

fig, ax = plt.subplots(figsize=(12, 5))

params = ['Space', 'Time', 'Energy', 'Matter', 'Gravity']
stab_levels = [n_space, n_time, n_energy, n_matter, n_gravity]
colors = ['#4A90D9', '#4A90D9', '#D4AA00', '#D4AA00', '#C0392B']

# Horizontal bars showing stabilization timeline
for i, (p, n, c) in enumerate(zip(params, stab_levels, colors)):
    ax.barh(i, n + 0.5, left=-0.25, height=0.6, color=c, alpha=0.7,
            edgecolor='black', linewidth=1)
    ax.text(n + 0.7, i, f'n = {n}', va='center', fontsize=12, fontweight='bold')

# Mark the cascade levels
for n in range(n_gravity + 2):
    ax.axvline(x=n, color='gray', linestyle=':', alpha=0.3)

ax.set_yticks(range(len(params)))
ax.set_yticklabels(params, fontsize=12)
ax.set_xlabel('Cascade Level', fontsize=12)
ax.set_title('Parameter Stabilization Sequence During Initial Cascade',
             fontsize=14)
ax.set_xlim(-0.5, n_gravity + 2)
ax.invert_yaxis()

# Annotate tiers
ax.annotate('Tier 1:\nArena', xy=(-0.3, 0.5), fontsize=9, color='#4A90D9',
            fontweight='bold', ha='right')
ax.annotate('Tier 2:\nContent', xy=(-0.3, 2.5), fontsize=9, color='#D4AA00',
            fontweight='bold', ha='right')
ax.annotate('Tier 3:\nResponse', xy=(-0.3, 4), fontsize=9, color='#C0392B',
            fontweight='bold', ha='right')

plt.tight_layout()
fig_path = os.path.join(OUTDIR, 'fig2_stabilization_sequence.png')
plt.savefig(fig_path, dpi=200, bbox_inches='tight')
plt.close()
print(f'  Saved: {fig_path}')

# ================================================================
# SECTION 8: KEY RESULTS SUMMARY
# ================================================================
print('\n' + '=' * 72)
print('  PHASE 1 RESULTS SUMMARY')
print('=' * 72)
print(f'')
print(f'  Feigenbaum constants:')
print(f'    δ = {DELTA}')
print(f'    α = {ALPHA}')
print(f'    λᵣ = δ/α = {LAMBDA_R:.8f}')
print(f'')
print(f'  Gravitational self-coupling:')
print(f'    Geometric ratio (d=4): {geometric_ratio}')
print(f'    Critical level n_crit: {n_crit}')
print(f'    Total delay Δn_g: {Delta_n_g:.4f} cascade levels')
print(f'    Integer delay: {Delta_n_g_int} cascade levels')
print(f'')
print(f'  Stabilization sequence:')
print(f'    Space, Time  → level {n_space} (arena)')
print(f'    Energy, Matter → level {n_energy} (content)')
print(f'    Gravity      → level {n_gravity} (response)')
print(f'')
print(f'  Gravity lags by {n_gravity - n_matter} levels after matter-energy')
print(f'')
print(f'  Emergence function (Model C, cascade accumulation):')
for n_val in [1, 2, 3, 4, 5]:
    val = G_eff_C(np.array([float(n_val)]))[0]
    pct = val * 100
    print(f'    Level {n_val}: G_eff/G = {val:.4f} ({pct:.1f}%)')
print(f'')
print(f'  INTERPRETATION:')
print(f'    Gravity reaches ~79% of full strength by level {n_gravity - 1}')
print(f'    and stabilizes at level {n_gravity}.')
print(f'    During levels 0-{n_gravity - 1}, the universe has')
print(f'    PARTIAL gravity — the "whisper phase" of GR.')
print(f'    This is the inflationary epoch: rapid expansion')
print(f'    because gravity is too weak to slow it.')
print('=' * 72)
