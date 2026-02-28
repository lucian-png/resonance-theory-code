#!/usr/bin/env python3
"""
Test 3: COUPLING TOPOLOGY — Lorenz vs Rössler
The Lucian Law Falsification Protocol

Purpose: Test whether systems with SAME coupling topology (3 variables,
pairwise coupling) but DIFFERENT equations produce same classification.

System A: Lorenz (σ, ρ, β) — driving variable ρ
System B: Rössler (a, b, c) — driving variable c
Both: 3 coupled nonlinear ODEs with full pairwise coupling
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy import stats

def lorenz_attractor_metrics(rho_values, sigma=10.0, beta=8/3,
                              t_max=100, t_transient=30):
    """
    For each ρ, integrate Lorenz system and extract geometric metrics
    from the attractor.
    """
    max_x = np.zeros(len(rho_values))
    max_z = np.zeros(len(rho_values))
    mean_amplitude = np.zeros(len(rho_values))
    variance = np.zeros(len(rho_values))

    for i, rho in enumerate(rho_values):
        def lorenz(t, y):
            x, y_var, z = y
            return [
                sigma * (y_var - x),
                x * (rho - z) - y_var,
                x * y_var - beta * z
            ]

        try:
            sol = solve_ivp(lorenz, [0, t_max], [1.0, 1.0, 1.0],
                           t_eval=np.linspace(t_transient, t_max, 500),
                           method='Radau', rtol=1e-6, atol=1e-8)
            
            if sol.success and len(sol.y[0]) > 100:
                max_x[i] = np.max(np.abs(sol.y[0]))
                max_z[i] = np.max(sol.y[2])
                mean_amplitude[i] = np.mean(np.sqrt(sol.y[0]**2 + sol.y[1]**2 + sol.y[2]**2))
                variance[i] = np.var(sol.y[0])
            else:
                max_x[i] = max_z[i] = mean_amplitude[i] = variance[i] = np.nan
        except:
            max_x[i] = max_z[i] = mean_amplitude[i] = variance[i] = np.nan
    
    return max_x, max_z, mean_amplitude, variance

def rossler_attractor_metrics(c_values, a=0.2, b=0.2,
                               t_max=150, t_transient=50):
    """
    For each c, integrate Rössler system and extract geometric metrics.
    """
    max_x = np.zeros(len(c_values))
    max_z = np.zeros(len(c_values))
    mean_amplitude = np.zeros(len(c_values))
    variance = np.zeros(len(c_values))

    for i, c in enumerate(c_values):
        def rossler(t, y):
            x, y_var, z = y
            return [
                -(y_var + z),
                x + a * y_var,
                b + z * (x - c)
            ]

        try:
            sol = solve_ivp(rossler, [0, t_max], [1.0, 1.0, 0.0],
                           t_eval=np.linspace(t_transient, t_max, 500),
                           method='Radau', rtol=1e-6, atol=1e-8)
            
            if sol.success and len(sol.y[0]) > 100:
                max_x[i] = np.max(np.abs(sol.y[0]))
                max_z[i] = np.max(np.abs(sol.y[2]))
                mean_amplitude[i] = np.mean(np.sqrt(sol.y[0]**2 + sol.y[1]**2 + sol.y[2]**2))
                variance[i] = np.var(sol.y[0])
            else:
                max_x[i] = max_z[i] = mean_amplitude[i] = variance[i] = np.nan
        except:
            max_x[i] = max_z[i] = mean_amplitude[i] = variance[i] = np.nan
    
    return max_x, max_z, mean_amplitude, variance

def fractal_metrics(driving_var, response):
    """Compute fractal classification metrics."""
    valid = np.isfinite(response) & (response > 0)
    if np.sum(valid) < 50:
        return {'ks': np.nan, 'r2': np.nan, 'dim': np.nan, 'transitions': 0}
    
    rv = response[valid]
    dv = driving_var[valid]
    
    # Self-similarity
    n_seg = 4
    seg_size = len(rv) // n_seg
    ks_list = []
    for j in range(n_seg - 1):
        s1 = rv[j*seg_size:(j+1)*seg_size]
        s2 = rv[(j+1)*seg_size:(j+2)*seg_size]
        if np.std(s1) > 0 and np.std(s2) > 0:
            ks, _ = stats.ks_2samp(
                (s1 - np.mean(s1))/np.std(s1),
                (s2 - np.mean(s2))/np.std(s2))
            ks_list.append(ks)
    ks_mean = np.mean(ks_list) if ks_list else 1.0
    
    # Power law
    mask = (dv > 0) & (rv > 0)
    if np.sum(mask) > 10:
        _, _, r, _, _ = stats.linregress(np.log10(dv[mask]), np.log10(rv[mask]))
        r2 = r**2
    else:
        r2 = 0.0
    
    # Box dimension
    rv_norm = (rv - np.min(rv)) / (np.max(rv) - np.min(rv) + 1e-15)
    x_norm = np.linspace(0, 1, len(rv_norm))
    sizes = [8, 16, 32, 64, 128]
    log_s, log_c = [], []
    for nb in sizes:
        bs = 1.0 / nb
        occ = set()
        for xi, yi in zip(x_norm, rv_norm):
            occ.add((min(int(xi/bs), nb-1), min(int(yi/bs), nb-1)))
        log_s.append(np.log(nb))
        log_c.append(np.log(len(occ)))
    dim, _, _, _, _ = stats.linregress(log_s, log_c) if len(log_s) > 2 else (1.0, 0, 0, 0, 0)
    
    # Transitions
    d_rv = np.diff(rv)
    transitions = 0
    if np.std(d_rv) > 0:
        z = np.abs(d_rv - np.mean(d_rv)) / np.std(d_rv)
        transitions = int(np.sum(z > 5))
    
    return {'ks': ks_mean, 'r2': r2, 'dim': dim, 'transitions': transitions}

# ============================================================
# MAIN ANALYSIS
# ============================================================

print("=" * 70)
print("TEST 3: COUPLING TOPOLOGY — Lorenz vs Rössler")
print("The Lucian Law Falsification Protocol")
print("=" * 70)

# Lorenz: ρ from 0.1 to 1000 (4 orders of magnitude)
rho_values = np.logspace(-1, 3, 100)
print(f"\nLorenz: ρ from {rho_values[0]:.1f} to {rho_values[-1]:.0f} ({len(rho_values)} points)")
print("Computing Lorenz attractor metrics...", flush=True)
l_max_x, l_max_z, l_mean_amp, l_var = lorenz_attractor_metrics(rho_values)

# Rössler: c from 0.5 to 50 — the dynamically interesting range
# (c > 100 creates extreme stiffness that isn't physically meaningful)
c_values = np.logspace(-0.3, 1.7, 100)
print(f"\nRössler: c from {c_values[0]:.1f} to {c_values[-1]:.0f} ({len(c_values)} points)")
print("Computing Rössler attractor metrics...", flush=True)
r_max_x, r_max_z, r_mean_amp, r_var = rossler_attractor_metrics(c_values)

# Compute fractal metrics for each
print("\n--- Lorenz Classification ---")
l_metrics = fractal_metrics(rho_values, l_mean_amp)
print(f"  Self-similarity KS: {l_metrics['ks']:.4f}")
print(f"  Power-law R²: {l_metrics['r2']:.4f}")
print(f"  Box dimension: {l_metrics['dim']:.4f}")
print(f"  Transitions: {l_metrics['transitions']}")

print("\n--- Rössler Classification ---")
r_metrics = fractal_metrics(c_values, r_mean_amp)
print(f"  Self-similarity KS: {r_metrics['ks']:.4f}")
print(f"  Power-law R²: {r_metrics['r2']:.4f}")
print(f"  Box dimension: {r_metrics['dim']:.4f}")
print(f"  Transitions: {r_metrics['transitions']}")

# Compare
print("\n--- COMPARISON ---")
print(f"  KS difference: {abs(l_metrics['ks'] - r_metrics['ks']):.4f}")
print(f"  R² difference: {abs(l_metrics['r2'] - r_metrics['r2']):.4f}")
print(f"  Dim difference: {abs(l_metrics['dim'] - r_metrics['dim']):.4f}")

# ============================================================
# FIGURE
# ============================================================

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('TEST 3: COUPLING TOPOLOGY — Same Structure, Different Equations\n'
             'Lorenz (ρ-driven) vs Rössler (c-driven)',
             fontsize=14, fontweight='bold')

# Panel 1: Lorenz response
ax = axes[0, 0]
valid = np.isfinite(l_mean_amp)
ax.loglog(rho_values[valid], l_mean_amp[valid], 'b-', linewidth=1, alpha=0.7)
ax.set_xlabel('ρ (driving variable)')
ax.set_ylabel('Mean amplitude')
ax.set_title('Lorenz: Mean Attractor Amplitude vs ρ')
ax.grid(True, alpha=0.3)

# Panel 2: Rössler response
ax = axes[0, 1]
valid = np.isfinite(r_mean_amp)
ax.loglog(c_values[valid], r_mean_amp[valid], 'r-', linewidth=1, alpha=0.7)
ax.set_xlabel('c (driving variable)')
ax.set_ylabel('Mean amplitude')
ax.set_title('Rössler: Mean Attractor Amplitude vs c')
ax.grid(True, alpha=0.3)

# Panel 3: Variance comparison (log-log)
ax = axes[0, 2]
valid_l = np.isfinite(l_var) & (l_var > 0)
valid_r = np.isfinite(r_var) & (r_var > 0)
ax.loglog(rho_values[valid_l], l_var[valid_l], 'b-', linewidth=1.5, label='Lorenz', alpha=0.7)
ax.loglog(c_values[valid_r], r_var[valid_r], 'r-', linewidth=1.5, label='Rössler', alpha=0.7)
ax.set_xlabel('Driving variable')
ax.set_ylabel('Variance of x(t)')
ax.set_title('Variance Response Comparison')
ax.legend()
ax.grid(True, alpha=0.3)

# Panel 4: Lorenz max values (coupled response)
ax = axes[1, 0]
valid = np.isfinite(l_max_x) & np.isfinite(l_max_z)
ax.loglog(rho_values[valid], l_max_x[valid], 'b-', linewidth=1.5, label='max|x|', alpha=0.7)
ax.loglog(rho_values[valid], l_max_z[valid], 'c-', linewidth=1.5, label='max(z)', alpha=0.7)
ax.set_xlabel('ρ')
ax.set_ylabel('Maximum values')
ax.set_title('Lorenz: Coupled Variable Maxima')
ax.legend()
ax.grid(True, alpha=0.3)

# Panel 5: Rössler max values (coupled response)
ax = axes[1, 1]
valid = np.isfinite(r_max_x) & np.isfinite(r_max_z)
ax.loglog(c_values[valid], r_max_x[valid], 'r-', linewidth=1.5, label='max|x|', alpha=0.7)
ax.loglog(c_values[valid], r_max_z[valid], 'm-', linewidth=1.5, label='max|z|', alpha=0.7)
ax.set_xlabel('c')
ax.set_ylabel('Maximum values')
ax.set_title('Rössler: Coupled Variable Maxima')
ax.legend()
ax.grid(True, alpha=0.3)

# Panel 6: Classification comparison
ax = axes[1, 2]
ax.axis('off')

comparison = f"""
COUPLING TOPOLOGY COMPARISON

Both systems: 3 variables, pairwise coupling
Different: Specific equations, parameters, physics

                    Lorenz      Rössler
Self-sim KS:        {l_metrics['ks']:.4f}      {r_metrics['ks']:.4f}
Power-law R²:       {l_metrics['r2']:.4f}      {r_metrics['r2']:.4f}
Box dimension:      {l_metrics['dim']:.4f}      {r_metrics['dim']:.4f}
Transitions:        {l_metrics['transitions']}           {r_metrics['transitions']}

IF SAME CLASSIFICATION:
  → Coupling topology determines classification
  → Equation content is irrelevant
  → Component 3 of Lucian Law CONFIRMED

IF DIFFERENT CLASSIFICATION:
  → Equation content matters
  → Component 3 needs revision
  → Law is weaker than claimed
"""
ax.text(0.05, 0.95, comparison, transform=ax.transAxes, fontsize=9,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig('38_coupling_topology.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nFigure saved: 38_coupling_topology.png")
