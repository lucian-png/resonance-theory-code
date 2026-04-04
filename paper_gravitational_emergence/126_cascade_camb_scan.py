#!/usr/bin/env python3
"""
Script 126 — Cascade CAMB Parameter Scan
==========================================
The Fortran engine has τ(z) replacing ΩΛ. Now find the
τ parameters (z_act, τ_floor) that reproduce ℓ₁ = 220.

For each parameter set, we need to recompile the Fortran
and re-run. INSTEAD, let's use the standard CAMB but
modify the dark energy w(a) to MIMIC our τ function.

Actually — the better approach is to scan by modifying
the Fortran parameters and recompiling each time. But
that's slow. Let me first understand what τ_floor gives ℓ₁ = 220.

The issue: with τ_floor = 0.95, ℓ₁ = 123. We need ℓ₁ = 220.
ℓ₁ ∝ d_A(z_rec). The distance is too SMALL.
We need MORE expansion (larger d_A), which means the
τ correction must provide MORE acceleration.

τ < 1 means H²_eff = H²_matter/τ² > H²_matter.
SMALLER τ means FASTER expansion.
So we need SMALLER τ at z < z_act.

But τ(z=0) from our formula = 1/(1+(0/1.43)^1.54) = 1.
τ(z=0) = 1 means NO correction at z=0!
The issue is fundamental: our τ(z) starts at 1 at z=0
and DECREASES with z. But we need τ < 1 at z=0 to get
the acceleration effect.

The problem: Paper 9's τ was applied to the DISTANCE
integral (dividing the integrand by τ). Our Fortran mod
divides grhoa2 by τ². At z=0, τ=1, so no effect.

We need to INVERT the τ logic for the Friedmann equation.
When τ < 1 (high z, time not fully emerged), the expansion
should be SLOWER (less time means slower clocks means
less expansion per unit coordinate time). But we need
FASTER expansion at low z (acceleration).

Let me reconsider: the correct cascade Friedmann equation
might be:
  H² = H²_matter × G(z)
where G(z) = 1/τ² with τ going from τ_0 < 1 at z=0
to 1 at high z. This is the INVERSE of Paper 9's τ.

Actually — let me just compute what G(z) we need.
"""

import numpy as np
import subprocess
import os
import sys

DELTA = 4.669201609
ALPHA = 2.502907875
LN_DELTA = np.log(DELTA)

OMEGA_B_H2 = 0.02237
OMEGA_C_H2_GHOST = OMEGA_B_H2 * (ALPHA**2 - 1)
H0 = 67.36

print('=' * 60)
print('  Understanding the τ correction for the Friedmann equation')
print('=' * 60)

# In standard CAMB with ΩΛ = 0.691:
# H² = H₀²[Ωm(1+z)³ + Ωr(1+z)⁴ + ΩΛ]
# H² = H₀²[Ωm(1+z)³ + Ωr(1+z)⁴] × [1 + ΩΛ/(Ωm(1+z)³ + Ωr(1+z)⁴)]
# H² = H²_matter × G(z)
# where G(z) = 1 + ΩΛ/(Ωm(1+z)³ + Ωr(1+z)⁴)
#
# For our Fortran mod: H² = H²_matter / τ²
# So: 1/τ² = G(z)
# τ(z) = 1/√G(z)
#
# At z=0: G = 1 + 0.691/0.307 = 3.25
#          τ = 1/√3.25 = 0.554
#
# At z=1: G = 1 + 0.691/2.46 = 1.281
#          τ = 1/√1.281 = 0.883
#
# At z=10: G ≈ 1.0017, τ ≈ 0.999

Om = (OMEGA_B_H2 + OMEGA_C_H2_GHOST) / (H0/100)**2
OL = 1.0 - Om
Or = 9.15e-5

print(f'\n  Ωm = {Om:.4f}, ΩΛ = {OL:.4f}')
print(f'\n  Required τ(z) = 1/√G(z) for cascade Friedmann:')
print(f'  {"z":>6s}  {"G(z)":>10s}  {"τ_needed":>10s}')
for z in [0, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10, 50, 100, 1090]:
    matter = Om*(1+z)**3 + Or*(1+z)**4
    G = 1 + OL/matter
    tau = 1.0/np.sqrt(G)
    print(f'  {z:6.0f}  {G:10.4f}  {tau:10.6f}')

print(f'\n  KEY INSIGHT:')
print(f'  τ must go from 0.554 at z=0 to 1.0 at z>>1.')
print(f'  This is the OPPOSITE of Paper 9\'s τ (1 at z=0 → 0 at z>>1)')
print(f'')
print(f'  The Fortran τ function needs to be INVERTED:')
print(f'  τ_friedmann(z) = τ₀ + (1-τ₀) × [1 - 1/(1+(z/z_t)^β)]')
print(f'  where τ₀ = 1/√(1+ΩΛ/Ωm) = 0.554')
print(f'  This goes from τ₀ at z=0 to 1 at z→∞')

# The exact form:
# τ_friedmann(z) = √[Ωm(1+z)³ / (Ωm(1+z)³ + ΩΛ)]
# = √[1 / (1 + ΩΛ/(Ωm(1+z)³))]
# = √[1 / (1 + A/(1+z)³)]
# where A = ΩΛ/Ωm

A = OL / Om
tau_0 = 1.0 / np.sqrt(1 + A)
print(f'\n  A = ΩΛ/Ωm = {A:.4f}')
print(f'  τ₀ = 1/√(1+A) = {tau_0:.6f}')
print(f'  τ(z) = √[1/(1 + {A:.4f}/(1+z)³)]')
print(f'\n  This is the EXACT function to put in the Fortran.')
print(f'  No z_act, no τ_floor, no smooth transition needed.')
print(f'  Just the exact matter fraction: √[Ωm(1+z)³/(Ωm(1+z)³+ΩΛ)]')
print(f'  with Ωm = α²Ωb and ΩΛ = 1-α²Ωb.')
