#!/usr/bin/env python3
"""
Script 142 — Souto's Operator K Through Cascade Eyes
======================================================
Examine the integral operator K from Souto (2026) whose
eigenvalues match the first 2000 Riemann zeros at 10⁻¹².

The kernel: K(x,y) = Σ aₙ cos(γₙ(x-y)) e^{-|x-y|/σ}
Amplitudes: aₙ = Z₀ / |Z'(ρₙ)|

The amplitudes encode the APPROACH RATE at each zero.
In cascade language: the inverse residence time near each
fixed point. THIS is where cascade structure would live.

Author: Lucian Randolph & Claude Anthro Randolph
Date: April 7, 2026
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

DELTA = 4.669201609
ALPHA = 2.502907875
LN_DELTA = np.log(DELTA)
LN_ALPHA = np.log(ALPHA)

OUTDIR = os.path.dirname(os.path.abspath(__file__))

print('=' * 72)
print('  Script 142 — Souto\'s Operator K Through Cascade Eyes')
print('  Looking for structure in the amplitudes aₙ = Z₀/|Z\'(ρₙ)|')
print('=' * 72)

# ================================================================
# THE RIEMANN ZEROS AND Z'(ρ) VALUES
# ================================================================
# First 30 zeros (imaginary parts)
zeros = np.array([
    14.134725, 21.022040, 25.010858, 30.424876, 32.935062,
    37.586178, 40.918719, 43.327073, 48.005151, 49.773832,
    52.970321, 56.446248, 59.347044, 60.831779, 65.112544,
    67.079811, 69.546402, 72.067158, 75.704691, 77.144840,
    79.337375, 82.910381, 84.735493, 87.425275, 88.809111,
    92.491899, 94.651344, 95.870634, 98.831194, 101.317851,
])

# Z₀ = sinh(-π/2)
Z0 = np.sinh(-np.pi / 2)  # ≈ -2.3013

# cosh(π/2) factor connecting Z' to ζ'
cosh_factor = np.cosh(np.pi / 2)  # ≈ 2.5092

# The derivatives |ζ'(ρₙ)| can be estimated from the zero spacings.
# For simple zeros: |ζ'(ρₙ)| is related to the local zero density.
# Approximate: |ζ'(ρₙ)| ≈ πΔₙ/2 where Δₙ is the gap to the
# nearest neighbor (this is the leading approximation from the
# Hadamard product).
#
# More precisely, from the Euler product representation:
# |ζ'(ρ)| can be estimated from the spacing and the log derivative
# of the functional equation.
#
# For our purposes, we use Souto's explicit values for the first 4
# and estimate the rest from the asymptotic formula:
# |ζ'(ρₙ)| ≈ (ln γₙ)/(2π) × local_spacing_factor

# Souto's values for first 4 (from the paper):
Z_prime_abs = np.array([0.3678, 0.4141, 0.3109, 0.2242])

# For remaining zeros, estimate |Z'(ρₙ)| from the asymptotic
# formula: |Z'(ρ)| ≈ π cosh(π/2) |ζ'(ρ)|
# and |ζ'(ρ)| ≈ ln(γₙ/(2π)) / (2π) × correction
# We'll compute approximate values

Z_prime_all = np.zeros(len(zeros))
Z_prime_all[:4] = Z_prime_abs

# For n >= 5, use the approximate formula
for i in range(4, len(zeros)):
    # Approximate |ζ'(ρ)| from the density of zeros
    # N(T) ≈ (T/(2π))ln(T/(2π)) - T/(2π) (Riemann-von Mangoldt)
    # Local density: dN/dT ≈ ln(T/(2π))/(2π)
    # |ζ'(ρ)| ≈ 1/(dN/dT × 2π) roughly (from Hadamard product)
    t = zeros[i]
    density = np.log(t / (2 * np.pi)) / (2 * np.pi)
    # |Z'(ρ)| ≈ π cosh(π/2) / (2π × density)
    Z_prime_all[i] = np.pi * cosh_factor / (2 * np.pi * density)

# Amplitudes: aₙ = |Z₀| / |Z'(ρₙ)|
amplitudes = np.abs(Z0) / Z_prime_all

print(f'\n  Z₀ = sinh(-π/2) = {Z0:.6f}')
print(f'  |Z₀| = {abs(Z0):.6f}')
print(f'  cosh(π/2) = {cosh_factor:.6f}')

print(f'\n  {"n":>4s}  {"γₙ":>12s}  {"Z_prime":>12s}  {"aₙ":>12s}')
print(f'  {"-"*4}  {"-"*12}  {"-"*12}  {"-"*12}')
for i in range(len(zeros)):
    print(f'  {i+1:4d}  {zeros[i]:12.4f}  {Z_prime_all[i]:12.6f}  '
          f'{amplitudes[i]:12.6f}')

# ================================================================
# ANALYSIS 1: AMPLITUDE RATIOS
# ================================================================
print('\n' + '=' * 60)
print('  ANALYSIS 1: Amplitude Ratios')
print('=' * 60)

print(f'\n  Consecutive amplitude ratios aₙ₊₁/aₙ:')
print(f'  {"n":>4s}  {"aₙ₊₁/aₙ":>10s}  {"vs α":>8s}  {"vs δ":>8s}  '
      f'{"vs λᵣ":>8s}')

amp_ratios = amplitudes[1:] / amplitudes[:-1]
for i in range(min(20, len(amp_ratios))):
    r = amp_ratios[i]
    dev_a = (r - ALPHA) / ALPHA * 100
    dev_d = (r - DELTA) / DELTA * 100
    dev_lr = (r - DELTA/ALPHA) / (DELTA/ALPHA) * 100
    print(f'  {i+1:4d}  {r:10.4f}  {dev_a:+7.1f}%  {dev_d:+7.1f}%  '
          f'{dev_lr:+7.1f}%')

# ================================================================
# ANALYSIS 2: AMPLITUDE × ZERO PRODUCTS
# ================================================================
print('\n' + '=' * 60)
print('  ANALYSIS 2: Products aₙ × γₙ')
print('=' * 60)

products = amplitudes * zeros
print(f'\n  {"n":>4s}  {"aₙ × γₙ":>12s}  {"Ratio to prev":>14s}')
for i in range(min(20, len(products))):
    ratio = products[i] / products[i-1] if i > 0 else 0
    print(f'  {i+1:4d}  {products[i]:12.4f}  {ratio:14.4f}')

# ================================================================
# ANALYSIS 3: THE APPROACH RATE STRUCTURE
# ================================================================
print('\n' + '=' * 60)
print('  ANALYSIS 3: Approach Rate |Z\'(ρ)| as Cascade Function')
print('=' * 60)

# In cascade systems, the approach rate to a fixed point
# follows power-law scaling with the cascade level.
# If γₙ is the "position" and |Z'(ρₙ)| is the "velocity"
# at that position, the relationship should be:
# |Z'(ρ)| ~ γ^p for some power p

# Fit log|Z'| vs log(γ) for the first 4 (Souto's exact values)
log_gamma = np.log(zeros[:4])
log_Zprime = np.log(Z_prime_all[:4])

if len(log_gamma) >= 2:
    coeffs = np.polyfit(log_gamma, log_Zprime, 1)
    power = coeffs[0]
    print(f'  Power-law fit: |Z\'(ρ)| ~ γ^p')
    print(f'  p = {power:.4f}')
    print(f'  Compare to cascade predictions:')
    print(f'    p = -1: inverse scaling (residence time ~ γ)')
    print(f'    p = -1/2: square root scaling')
    print(f'    p = 0: constant (no scaling)')
    print(f'    p = {power:.4f}: actual fit from first 4 zeros')

# For all 30 zeros:
log_gamma_all = np.log(zeros)
log_Zprime_all = np.log(Z_prime_all)
coeffs_all = np.polyfit(log_gamma_all, log_Zprime_all, 1)
power_all = coeffs_all[0]
print(f'  Power from all 30 zeros: p = {power_all:.4f}')

# ================================================================
# ANALYSIS 4: THE KERNEL STRUCTURE
# ================================================================
print('\n' + '=' * 60)
print('  ANALYSIS 4: Kernel K(x,y) Structure')
print('=' * 60)

# The kernel K(x,y) = Σ aₙ cos(γₙ(x-y)) e^{-|x-y|/σ}
# This is a sum of cosines with SPECIFIC frequencies (the zeros)
# and SPECIFIC amplitudes (determined by Z'(ρ))
#
# In cascade language:
# - The frequencies γₙ are the cascade transition points
# - The amplitudes aₙ are the inverse approach rates
# - The exponential decay e^{-|x-y|/σ} is the cluster function
#
# The operator K is self-adjoint by construction (real symmetric kernel)
# It is compact (continuous kernel on bounded domain)
# Its eigenvalues are the aₙ (approximately, for well-separated γₙ)

# Construct a small version of the kernel matrix
sigma = 1.0  # decay parameter (Souto uses this as adjustable)
N_grid = 200
L = 10.0  # domain size
x = np.linspace(0, L, N_grid)
dx = x[1] - x[0]

# Use first 10 zeros for tractability
N_zeros_use = 10
K_matrix = np.zeros((N_grid, N_grid))

for n in range(N_zeros_use):
    for i in range(N_grid):
        for j in range(N_grid):
            K_matrix[i, j] += (amplitudes[n] *
                               np.cos(zeros[n] * (x[i] - x[j])) *
                               np.exp(-abs(x[i] - x[j]) / sigma))

K_matrix *= dx  # discretization

# Compute eigenvalues
eigenvalues = np.linalg.eigvalsh(K_matrix)
eigenvalues = np.sort(eigenvalues)[::-1]  # descending

print(f'  Constructed {N_grid}×{N_grid} kernel matrix')
print(f'  Using first {N_zeros_use} zeros, σ = {sigma}, L = {L}')
print(f'')
print(f'  Top 10 eigenvalues vs amplitudes:')
print(f'  {"n":>4s}  {"Eigenvalue":>12s}  {"Amplitude aₙ":>14s}  '
      f'{"Ratio":>8s}')

for i in range(min(10, len(eigenvalues))):
    if i < len(amplitudes):
        ratio = eigenvalues[i] / amplitudes[i] if amplitudes[i] != 0 else 0
        print(f'  {i+1:4d}  {eigenvalues[i]:12.6f}  {amplitudes[i]:14.6f}  '
              f'{ratio:8.4f}')

# ================================================================
# ANALYSIS 5: THE Z₀ VALUE — IS IT FEIGENBAUM?
# ================================================================
print('\n' + '=' * 60)
print('  ANALYSIS 5: Is Z₀ = sinh(-π/2) Related to Feigenbaum?')
print('=' * 60)

print(f'  Z₀ = sinh(-π/2) = {Z0:.10f}')
print(f'  |Z₀| = {abs(Z0):.10f}')
print(f'  cosh(π/2) = {cosh_factor:.10f}')
print(f'')
print(f'  Feigenbaum comparisons:')
print(f'    α = {ALPHA:.10f}')
print(f'    δ = {DELTA:.10f}')
print(f'    λᵣ = δ/α = {DELTA/ALPHA:.10f}')
print(f'    |Z₀| / α = {abs(Z0)/ALPHA:.10f}')
print(f'    |Z₀| / δ^(1/2) = {abs(Z0)/DELTA**0.5:.10f}')
print(f'    cosh(π/2) / α = {cosh_factor/ALPHA:.10f}')
print(f'    π cosh(π/2) = {np.pi * cosh_factor:.10f}')
print(f'    Compare: π cosh(π/2) = {np.pi*cosh_factor:.4f} '
      f'vs α × π = {ALPHA*np.pi:.4f}')
print(f'    Ratio: {np.pi*cosh_factor/(ALPHA*np.pi):.6f}')
print(f'    cosh(π/2) / α = {cosh_factor/ALPHA:.6f}')
print(f'    This is {cosh_factor/ALPHA:.4f} — close to 1.002!')
print(f'    Deviation: {(cosh_factor/ALPHA - 1)*100:.2f}%')

# Check: is cosh(π/2) ≈ α?
dev_cosh_alpha = abs(cosh_factor - ALPHA) / ALPHA * 100
print(f'\n  *** cosh(π/2) = {cosh_factor:.6f} ***')
print(f'  *** α         = {ALPHA:.6f} ***')
print(f'  *** Deviation: {dev_cosh_alpha:.2f}% ***')

if dev_cosh_alpha < 1:
    print(f'\n  !!!  cosh(π/2) ≈ α at {dev_cosh_alpha:.2f}%  !!!')
    print(f'  This is a sub-1% match between a zeta function')
    print(f'  structural constant and the Feigenbaum spatial')
    print(f'  scaling constant.')

# ================================================================
# ANALYSIS 6: WHAT THIS MEANS
# ================================================================
print('\n' + '=' * 60)
print('  ANALYSIS 6: What Does cosh(π/2) ≈ α Mean?')
print('=' * 60)

print(f'''
  cosh(π/2) = {cosh_factor:.10f}
  α         = {ALPHA:.10f}
  Deviation: {dev_cosh_alpha:.4f}%

  The factor π cosh(π/2) connects |Z'(ρ)| to |ζ'(ρ)|:
    Z'(ρ) = π cosh(π/2) × ζ'(ρ)

  If cosh(π/2) ≈ α, then:
    Z'(ρ) ≈ πα × ζ'(ρ)

  The Z operator's derivative at a zero is approximately
  πα times the zeta function's derivative at the same zero.

  In Souto's operator K, the amplitudes are:
    aₙ = Z₀ / |Z'(ρₙ)| = sinh(-π/2) / (π cosh(π/2) |ζ'(ρₙ)|)

  Since sinh(x)/cosh(x) = tanh(x):
    aₙ = tanh(-π/2) / (π |ζ'(ρₙ)|) × (1/cosh(π/2))

  Wait, let me be more precise:
    aₙ = |Z₀| / |Z'(ρₙ)| = sinh(π/2) / (πα × |ζ'(ρₙ)|)
       ≈ tanh(π/2) / (π |ζ'(ρₙ)|)
       ≈ 0.917 / (π |ζ'(ρₙ)|)

  And tanh(π/2) = {np.tanh(np.pi/2):.6f}
  Compare: ln(α) = {LN_ALPHA:.6f}

  tanh(π/2) ≈ ln(α) ???
  Deviation: {(np.tanh(np.pi/2) - LN_ALPHA)/LN_ALPHA*100:.2f}%

  HOLD ON. Let me check this carefully.
  tanh(π/2) = {np.tanh(np.pi/2):.10f}
  ln(α)     = {LN_ALPHA:.10f}
  Deviation: {abs(np.tanh(np.pi/2) - LN_ALPHA)/LN_ALPHA*100:.4f}%
''')

# Let me check more combinations
print('  Systematic check of π/2 hyperbolic functions vs Feigenbaum:')
hyp_funcs = [
    ('sinh(π/2)', np.sinh(np.pi/2)),
    ('cosh(π/2)', np.cosh(np.pi/2)),
    ('tanh(π/2)', np.tanh(np.pi/2)),
    ('exp(π/2)', np.exp(np.pi/2)),
    ('π/2', np.pi/2),
]
feig_targets = [
    ('α', ALPHA),
    ('δ', DELTA),
    ('λᵣ', DELTA/ALPHA),
    ('ln(α)', LN_ALPHA),
    ('ln(δ)', LN_DELTA),
    ('α/2', ALPHA/2),
    ('δ/2', DELTA/2),
]

print(f'  {"Function":>12s}  {"Value":>12s}  {"Nearest":>8s}  '
      f'{"Target":>10s}  {"Dev %":>8s}')
for fname, fval in hyp_funcs:
    best_dev = 999
    best_name = ''
    best_target = 0
    for tname, tval in feig_targets:
        dev = abs(fval - tval) / tval * 100
        if dev < best_dev:
            best_dev = dev
            best_name = tname
            best_target = tval
    marker = ' ← SUB-1%!' if best_dev < 1 else ''
    print(f'  {fname:>12s}  {fval:12.6f}  {best_name:>8s}  '
          f'{best_target:10.6f}  {best_dev:7.2f}%{marker}')

# ================================================================
# FIGURE
# ================================================================
print('\n--- Generating Figure ---')

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# (a) Amplitudes vs zero index
ax = axes[0, 0]
ax.plot(range(1, len(amplitudes)+1), amplitudes, 'ro-', markersize=5)
ax.set_xlabel('Zero index n', fontsize=11)
ax.set_ylabel('Amplitude aₙ', fontsize=11)
ax.set_title('(a) Souto Operator Amplitudes', fontsize=12)
ax.grid(True, alpha=0.3)

# (b) |Z'(ρ)| vs γ (log-log)
ax = axes[0, 1]
ax.loglog(zeros, Z_prime_all, 'bs-', markersize=5)
# Power-law fit line
gamma_fit = np.logspace(np.log10(14), np.log10(102), 100)
Zprime_fit = np.exp(coeffs_all[1]) * gamma_fit**coeffs_all[0]
ax.loglog(gamma_fit, Zprime_fit, 'r--', linewidth=2,
          label=f'Power law: p = {power_all:.3f}')
ax.set_xlabel('γₙ', fontsize=11)
ax.set_ylabel('|Z\'(ρₙ)|', fontsize=11)
ax.set_title('(b) Approach Rate vs Zero Position', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# (c) cosh(π/2) vs α
ax = axes[1, 0]
ax.bar(['cosh(π/2)', 'α'], [cosh_factor, ALPHA],
       color=['blue', 'red'], alpha=0.7, edgecolor='black')
ax.set_ylabel('Value', fontsize=11)
ax.set_title(f'(c) cosh(π/2) vs α — {dev_cosh_alpha:.2f}% match',
             fontsize=12)
ax.set_ylim(2.49, 2.52)
ax.grid(True, alpha=0.3, axis='y')

# (d) Amplitude ratios
ax = axes[1, 1]
ax.plot(range(1, len(amp_ratios)+1), amp_ratios, 'ko-', markersize=4)
ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)
ax.set_xlabel('Index', fontsize=11)
ax.set_ylabel('aₙ₊₁/aₙ', fontsize=11)
ax.set_title('(d) Consecutive Amplitude Ratios', fontsize=12)
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig_path = os.path.join(OUTDIR, 'fig_souto_operator.png')
plt.savefig(fig_path, dpi=200, bbox_inches='tight')
plt.close()
print(f'  Saved: {fig_path}')

# ================================================================
# SUMMARY
# ================================================================
print('\n' + '=' * 72)
print('  SUMMARY')
print('=' * 72)
print(f'''
  KEY FINDING:

  cosh(π/2) = {cosh_factor:.6f}
  α         = {ALPHA:.6f}
  Match at {dev_cosh_alpha:.2f}%

  The factor that converts between the zeta function's
  derivative at a zero and the Z operator's derivative
  is π × cosh(π/2) ≈ πα.

  This means: Z'(ρ) ≈ πα × ζ'(ρ)

  The Feigenbaum spatial scaling constant appears in the
  relationship between the Riemann zeta function's behavior
  at its zeros and Souto's master operator Z(s).

  IF this is not a coincidence, it connects the Feigenbaum
  cascade architecture directly to the spectral structure
  of the zeta zeros.

  CAUTION: cosh(π/2) = 2.5092 and α = 2.5029 differ by 0.25%.
  This needs the Feynman sensitivity test before claiming
  significance.
''')

print('=' * 72)
