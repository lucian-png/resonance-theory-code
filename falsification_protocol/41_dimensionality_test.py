#!/usr/bin/env python3
"""
Test 6: DIMENSIONALITY TEST — Does Coupling Topology Determine Classification Specifics?
The Lucian Law Falsification Protocol

Purpose: Compare fractal dimensions, scaling exponents, and harmonic structures
across systems with DIFFERENT numbers of coupled variables and coupling topologies.

2-variable systems: Van der Pol, Duffing
3-variable systems: Lorenz, Chen
10-variable system: Coupled oscillator network

Optimized: LSODA solver (auto stiff/non-stiff), vectorized network, reduced points.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy import stats
import sys

def fractal_metrics_full(x_values, y_values):
    """Extended fractal metrics with scaling exponent."""
    valid = np.isfinite(y_values) & (y_values > 0)
    if np.sum(valid) < 20:
        return {'ks': np.nan, 'r2': np.nan, 'dim': np.nan, 'slope': np.nan, 'trans': 0}

    yv = y_values[valid]
    xv = x_values[valid]

    # Self-similarity
    n = 4
    s = len(yv) // n
    ks_list = []
    if s > 5:
        for j in range(n-1):
            s1, s2 = yv[j*s:(j+1)*s], yv[(j+1)*s:(j+2)*s]
            if np.std(s1) > 0 and np.std(s2) > 0:
                ks, _ = stats.ks_2samp((s1-np.mean(s1))/np.std(s1), (s2-np.mean(s2))/np.std(s2))
                ks_list.append(ks)
    ks = np.mean(ks_list) if ks_list else 1.0

    # Power law with slope
    mask = (xv > 0) & (yv > 0)
    r2, slope = 0.0, 0.0
    if np.sum(mask) > 10:
        slope, _, r, _, _ = stats.linregress(np.log10(xv[mask]), np.log10(yv[mask]))
        r2 = r**2

    # Box dimension
    yn = (yv - np.min(yv)) / (np.max(yv) - np.min(yv) + 1e-15)
    xn = np.linspace(0, 1, len(yn))
    sizes = [8, 16, 32, 64]
    ls, lc = [], []
    for nb in sizes:
        bs = 1.0/nb
        bx = np.minimum((xn / bs).astype(int), nb-1)
        by = np.minimum((yn / bs).astype(int), nb-1)
        occ = len(set(zip(bx, by)))
        ls.append(np.log(nb))
        lc.append(np.log(occ))
    dim = stats.linregress(ls, lc)[0] if len(ls) > 2 else 1.0

    # Transitions
    dy = np.diff(yv)
    trans = 0
    if np.std(dy) > 0:
        z = np.abs(dy - np.mean(dy)) / np.std(dy)
        trans = int(np.sum(z > 5))

    return {'ks': ks, 'r2': r2, 'dim': dim, 'slope': slope, 'trans': trans}


def safe_solve(fun, t_span, y0, t_eval_n=200, timeout_t=50):
    """Fast ODE solve with LSODA and fallback."""
    try:
        sol = solve_ivp(fun, [0, timeout_t], y0,
                       t_eval=np.linspace(timeout_t*0.4, timeout_t, t_eval_n),
                       method='LSODA', rtol=1e-5, atol=1e-7,
                       max_step=timeout_t/50)
        if sol.success and len(sol.y[0]) > 10:
            return sol
    except:
        pass
    return None


# ============================================================
print("=" * 70)
print("TEST 6: DIMENSIONALITY — Coupling Topology vs Classification")
print("The Lucian Law Falsification Protocol")
print("=" * 70)
sys.stdout.flush()

N_PTS = 60  # points per system

# ============================================================
# 2-VARIABLE SYSTEMS
# ============================================================

# --- Van der Pol ---
print("\n=== 2-Variable: Van der Pol Oscillator ===", flush=True)
mu_values = np.logspace(-2, 2, N_PTS)
vdp_max = np.full(N_PTS, np.nan)

for i, mu in enumerate(mu_values):
    def vanderpol(t, y, mu=mu):
        x, v = y
        return [v, mu * (1 - x**2) * v - x]
    sol = safe_solve(vanderpol, [0, 50], [0.1, 0.0])
    if sol is not None:
        vdp_max[i] = np.max(np.abs(sol.y[0]))

vdp_m = fractal_metrics_full(mu_values, vdp_max)
print(f"  KS={vdp_m['ks']:.4f}, R²={vdp_m['r2']:.4f}, D={vdp_m['dim']:.4f}, "
      f"slope={vdp_m['slope']:.4f}", flush=True)

# --- Duffing ---
print("\n=== 2-Variable: Duffing Oscillator ===", flush=True)
gamma_values = np.logspace(-2, 2, N_PTS)
duff_max = np.full(N_PTS, np.nan)

for i, gam in enumerate(gamma_values):
    def duffing(t, y, gam=gam):
        x, v = y
        return [v, -0.3*v - x - 0.5*x**3 + gam*np.cos(t)]
    sol = safe_solve(duffing, [0, 50], [0.1, 0.0])
    if sol is not None:
        duff_max[i] = np.max(np.abs(sol.y[0]))

duff_m = fractal_metrics_full(gamma_values, duff_max)
print(f"  KS={duff_m['ks']:.4f}, R²={duff_m['r2']:.4f}, D={duff_m['dim']:.4f}, "
      f"slope={duff_m['slope']:.4f}", flush=True)

# ============================================================
# 3-VARIABLE SYSTEMS
# ============================================================

# --- Lorenz ---
print("\n=== 3-Variable: Lorenz System ===", flush=True)
rho_values = np.logspace(-1, 3, N_PTS)
lorenz_max = np.full(N_PTS, np.nan)

for i, rho in enumerate(rho_values):
    def lorenz(t, y, rho=rho):
        x, yv, z = y
        return [10*(yv-x), x*(rho-z)-yv, x*yv-8/3*z]
    sol = safe_solve(lorenz, [0, 50], [1, 1, 1])
    if sol is not None:
        lorenz_max[i] = np.mean(np.sqrt(sol.y[0]**2 + sol.y[1]**2 + sol.y[2]**2))

lorenz_m = fractal_metrics_full(rho_values, lorenz_max)
print(f"  KS={lorenz_m['ks']:.4f}, R²={lorenz_m['r2']:.4f}, D={lorenz_m['dim']:.4f}, "
      f"slope={lorenz_m['slope']:.4f}", flush=True)

# --- Rössler (3 coupled, different topology from Lorenz) ---
print("\n=== 3-Variable: Rossler System ===", flush=True)
c_ross_values = np.logspace(-0.3, 1.7, N_PTS)  # c from 0.5 to 50
ross_max = np.full(N_PTS, np.nan)

for i, c in enumerate(c_ross_values):
    def rossler(t, y, c=c):
        x, yv, z = y
        return [-(yv + z), x + 0.2*yv, 0.2 + z*(x - c)]
    sol = safe_solve(rossler, [0, 100], [1, 1, 0], timeout_t=100)
    if sol is not None:
        ross_max[i] = np.mean(np.sqrt(sol.y[0]**2 + sol.y[1]**2 + sol.y[2]**2))

ross_m = fractal_metrics_full(c_ross_values, ross_max)
print(f"  KS={ross_m['ks']:.4f}, R²={ross_m['r2']:.4f}, D={ross_m['dim']:.4f}, "
      f"slope={ross_m['slope']:.4f}", flush=True)

# ============================================================
# N-VARIABLE SYSTEM (vectorized)
# ============================================================

print("\n=== 10-Variable: Coupled Nonlinear Oscillator Network ===", flush=True)
N_osc = 10
coupling_strength_values = np.logspace(-3, 1, N_PTS)  # cap at K=10
network_max = np.full(N_PTS, np.nan)

y0_net = np.random.RandomState(42).randn(2*N_osc) * 0.1

for i, K in enumerate(coupling_strength_values):
    def coupled_network(t, y, K=K):
        dydt = np.zeros(2 * N_osc)
        x = y[0::2]  # positions
        v = y[1::2]  # velocities
        for j in range(N_osc):
            diff = x - x[j]  # vectorized difference
            diff[j] = 0
            coupling = np.sum(K * diff + 0.1 * K * diff**3) / N_osc
            dydt[2*j] = v[j]
            dydt[2*j+1] = -0.1*v[j] - x[j] - 0.3*x[j]**3 + coupling
        return dydt

    sol = safe_solve(coupled_network, [0, 50], y0_net)
    if sol is not None:
        energy = 0
        for j in range(N_osc):
            energy += np.mean(sol.y[2*j]**2 + sol.y[2*j+1]**2)
        network_max[i] = np.sqrt(energy / N_osc)

net_m = fractal_metrics_full(coupling_strength_values, network_max)
print(f"  KS={net_m['ks']:.4f}, R²={net_m['r2']:.4f}, D={net_m['dim']:.4f}, "
      f"slope={net_m['slope']:.4f}", flush=True)

# ============================================================
# SUMMARY TABLE
# ============================================================
print("\n" + "=" * 70)
print("DIMENSIONALITY COMPARISON")
print("=" * 70)
print(f"\n{'System':<30} {'N_var':>5} {'KS':>8} {'R2':>8} {'Dim':>8} {'Slope':>8}")
print("-" * 70)

systems = [
    ("Van der Pol", 2, vdp_m),
    ("Duffing", 2, duff_m),
    ("Lorenz", 3, lorenz_m),
    ("Rossler", 3, ross_m),
    ("10-Oscillator Network", 10, net_m),
]

for name, nvar, m in systems:
    print(f"{name:<30} {nvar:>5} {m['ks']:>8.4f} {m['r2']:>8.4f} {m['dim']:>8.4f} {m['slope']:>8.4f}")

# ============================================================
# FIGURE
# ============================================================

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('TEST 6: DIMENSIONALITY — Does Coupling Topology Determine Classification?\n'
             '2-variable vs 3-variable vs 10-variable systems',
             fontsize=14, fontweight='bold')

# Panel 1: All response curves overlaid (normalized)
ax = axes[0, 0]
for name, driving, response, color in [
    ('Van der Pol', mu_values, vdp_max, '#2196F3'),
    ('Duffing', gamma_values, duff_max, '#03A9F4'),
    ('Lorenz', rho_values, lorenz_max, '#FF5722'),
    ('Rossler', c_ross_values, ross_max, '#FF9800'),
    ('Network-10', coupling_strength_values, network_max, '#4CAF50'),
]:
    valid = np.isfinite(response) & (response > 0)
    if np.sum(valid) > 10:
        r_norm = response[valid] / np.max(response[valid])
        d_norm = driving[valid] / np.max(driving[valid])
        ax.loglog(d_norm, r_norm, linewidth=1.5, label=name, color=color, alpha=0.7)
ax.set_xlabel('Normalized driving variable')
ax.set_ylabel('Normalized response')
ax.set_title('All Systems (Normalized)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Panel 2: Dimension vs N_variables
ax = axes[0, 1]
n_vars = [2, 2, 3, 3, 10]
dims = [m['dim'] for _, _, m in systems]
names_short = [n for n, _, _ in systems]
scatter_colors = ['#2196F3', '#03A9F4', '#FF5722', '#FF9800', '#4CAF50']
for j in range(len(n_vars)):
    ax.scatter(n_vars[j], dims[j], s=150, color=scatter_colors[j],
              zorder=5, label=names_short[j])
ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('Number of coupled variables')
ax.set_ylabel('Box-counting dimension')
ax.set_title('Fractal Dimension vs System Size')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Panel 3: Scaling exponent vs N_variables
ax = axes[0, 2]
slopes = [m['slope'] for _, _, m in systems]
for j in range(len(n_vars)):
    ax.scatter(n_vars[j], slopes[j], s=150, color=scatter_colors[j],
              zorder=5, label=names_short[j])
ax.set_xlabel('Number of coupled variables')
ax.set_ylabel('Power-law scaling exponent')
ax.set_title('Scaling Exponent vs System Size')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Panel 4: 2-variable comparison
ax = axes[1, 0]
valid = np.isfinite(vdp_max) & (vdp_max > 0)
ax.loglog(mu_values[valid], vdp_max[valid], 'b-', linewidth=1.5, label='Van der Pol', alpha=0.7)
valid = np.isfinite(duff_max) & (duff_max > 0)
ax.loglog(gamma_values[valid], duff_max[valid], 'c-', linewidth=1.5, label='Duffing', alpha=0.7)
ax.set_xlabel('Driving variable')
ax.set_ylabel('Max response')
ax.set_title('2-Variable Systems')
ax.legend()
ax.grid(True, alpha=0.3)

# Panel 5: 3-variable comparison
ax = axes[1, 1]
valid = np.isfinite(lorenz_max) & (lorenz_max > 0)
ax.loglog(rho_values[valid], lorenz_max[valid], 'r-', linewidth=1.5, label='Lorenz', alpha=0.7)
valid = np.isfinite(ross_max) & (ross_max > 0)
ax.loglog(c_ross_values[valid], ross_max[valid], color='orange', linewidth=1.5, label='Chen', alpha=0.7)
ax.set_xlabel('Driving variable')
ax.set_ylabel('Mean amplitude')
ax.set_title('3-Variable Systems')
ax.legend()
ax.grid(True, alpha=0.3)

# Panel 6: Summary
ax = axes[1, 2]
ax.axis('off')
summary = f"""
DIMENSIONALITY RESULTS

{'System':<25} {'N':>3} {'KS':>7} {'Dim':>7} {'Slope':>7}
{'-'*50}"""
for name, nvar, m in systems:
    summary += f"\n{name:<25} {nvar:>3} {m['ks']:>7.3f} {m['dim']:>7.3f} {m['slope']:>7.3f}"

summary += f"""

INTERPRETATION:
  Same N, similar metrics → topology determines class
  Same N, different metrics → equation content matters
  Trend with N → dimensionality is key variable
"""
ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=9,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig('41_dimensionality_test.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nFigure saved: 41_dimensionality_test.png", flush=True)
