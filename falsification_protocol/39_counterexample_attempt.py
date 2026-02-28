#!/usr/bin/env python3
"""
Test 4: CONSTRUCTED COUNTEREXAMPLE — Can We Build a Non-Fractal Nonlinear System?
The Lucian Law Falsification Protocol

Purpose: Deliberately attempt to construct nonlinear coupled systems that
DO NOT exhibit fractal geometric classification. If we fail to construct
a counterexample, the law may be mathematical (a theorem) not empirical.

Three candidate systems designed to "contain" nonlinearity:
  A) Piecewise-linear with nonlinear switching
  B) Saturating nonlinearity (tanh coupling)
  C) Bounded polynomial coupling
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy import stats

def fractal_metrics(x_values, y_values, label=""):
    """Standard fractal metric computation."""
    valid = np.isfinite(y_values) & (y_values > 0)
    if np.sum(valid) < 50:
        print(f"  {label}: Insufficient valid data ({np.sum(valid)} points)")
        return {'ks': np.nan, 'r2': np.nan, 'dim': np.nan, 'trans': 0, 'monotonic': True}
    
    yv = y_values[valid]
    xv = x_values[valid]
    
    # Self-similarity
    n = 4
    s = len(yv) // n
    ks_list = []
    for j in range(n-1):
        s1, s2 = yv[j*s:(j+1)*s], yv[(j+1)*s:(j+2)*s]
        if np.std(s1) > 0 and np.std(s2) > 0:
            ks, _ = stats.ks_2samp((s1-np.mean(s1))/np.std(s1), (s2-np.mean(s2))/np.std(s2))
            ks_list.append(ks)
    ks = np.mean(ks_list) if ks_list else 1.0
    
    # Power law
    mask = (xv > 0) & (yv > 0)
    r2 = 0.0
    if np.sum(mask) > 10:
        _, _, r, _, _ = stats.linregress(np.log10(xv[mask]), np.log10(yv[mask]))
        r2 = r**2
    
    # Box dimension
    yn = (yv - np.min(yv)) / (np.max(yv) - np.min(yv) + 1e-15)
    xn = np.linspace(0, 1, len(yn))
    sizes = [8, 16, 32, 64, 128]
    ls, lc = [], []
    for nb in sizes:
        bs = 1.0/nb
        occ = set()
        for xi, yi in zip(xn, yn):
            occ.add((min(int(xi/bs), nb-1), min(int(yi/bs), nb-1)))
        ls.append(np.log(nb))
        lc.append(np.log(len(occ)))
    dim = stats.linregress(ls, lc)[0] if len(ls) > 2 else 1.0
    
    # Transitions
    dy = np.diff(yv)
    trans = 0
    if np.std(dy) > 0:
        z = np.abs(dy - np.mean(dy)) / np.std(dy)
        trans = int(np.sum(z > 5))
    
    # Monotonic
    sign_changes = np.sum(np.diff(np.sign(np.diff(yv))) != 0)
    mono = sign_changes < len(yv) * 0.01  # less than 1% direction changes
    
    return {'ks': ks, 'r2': r2, 'dim': dim, 'trans': trans, 'monotonic': mono}

# ============================================================
# CANDIDATE A: Piecewise-Linear with Nonlinear Switching
# Nonlinearity exists only at switching boundaries
# ============================================================

def piecewise_system(driving_values):
    """
    Two coupled variables with piecewise-linear dynamics.
    Nonlinearity only at switching threshold.
    
    x_out = k1 * x_in  if x_in < threshold
    x_out = k2 * x_in  if x_in >= threshold
    y_out = coupling * x_out + k3 * driving
    """
    threshold = 1.0
    k1, k2, k3, coupling = 0.5, 2.0, 1.0, 0.3
    
    x_out = np.where(driving_values < threshold,
                     k1 * driving_values,
                     k2 * driving_values - (k2 - k1) * threshold)
    y_out = coupling * x_out + k3 * np.sqrt(np.abs(driving_values))
    
    return x_out, y_out

# ============================================================
# CANDIDATE B: Saturating Nonlinearity (tanh coupling)
# Nonlinearity is bounded — cannot grow without limit
# ============================================================

def saturating_system(driving_values):
    """
    Three coupled variables with tanh (saturating) nonlinearity.
    All outputs are bounded regardless of input magnitude.
    
    x = A * tanh(α * driving)
    y = B * tanh(β * (driving + coupling_xy * x))
    z = C * tanh(γ * (driving + coupling_xz * x + coupling_yz * y))
    """
    A, B, C = 10.0, 8.0, 12.0
    alpha, beta, gamma = 0.01, 0.005, 0.002
    c_xy, c_xz, c_yz = 0.3, 0.2, 0.4
    
    x = A * np.tanh(alpha * driving_values)
    y = B * np.tanh(beta * (driving_values + c_xy * x))
    z = C * np.tanh(gamma * (driving_values + c_xz * x + c_yz * y))
    
    return x, y, z

# ============================================================
# CANDIDATE C: Bounded Polynomial Coupling
# Polynomial nonlinearity but with damping that prevents unbounded growth
# ============================================================

def bounded_polynomial_system(driving_values):
    """
    Coupled system with polynomial nonlinearity but exponential damping.
    f(x) = x^3 * exp(-x^2/σ^2)
    This peaks and then decays — nonlinear but self-limiting.
    """
    sigma = 10.0
    
    x = driving_values**2 * np.exp(-driving_values**2 / (2 * sigma**2))
    coupling = 0.5
    y = (driving_values + coupling * x)**3 * np.exp(-(driving_values + coupling * x)**2 / (2 * sigma**2 * 4))
    z = np.sqrt(np.abs(x * y + 0.01)) * np.exp(-np.abs(x * y) / (sigma**4))
    
    return x, y, z

# ============================================================
# MAIN ANALYSIS
# ============================================================

print("=" * 70)
print("TEST 4: CONSTRUCTED COUNTEREXAMPLE")
print("Can we build a nonlinear coupled system that is NOT fractal?")
print("=" * 70)

# Driving variable: 12 orders of magnitude
driving = np.logspace(-6, 6, 2000)

# --- Candidate A ---
print("\n=== CANDIDATE A: Piecewise-Linear Switching ===")
xa, ya = piecewise_system(driving)
ma_x = fractal_metrics(driving, xa, "x_out")
ma_y = fractal_metrics(driving, ya, "y_out")
print(f"  x_out: KS={ma_x['ks']:.4f}, R²={ma_x['r2']:.4f}, D={ma_x['dim']:.4f}, "
      f"trans={ma_x['trans']}, mono={ma_x['monotonic']}")
print(f"  y_out: KS={ma_y['ks']:.4f}, R²={ma_y['r2']:.4f}, D={ma_y['dim']:.4f}, "
      f"trans={ma_y['trans']}, mono={ma_y['monotonic']}")

# --- Candidate B ---
print("\n=== CANDIDATE B: Saturating (tanh) Coupling ===")
xb, yb, zb = saturating_system(driving)
mb_x = fractal_metrics(driving, np.abs(xb) + 1e-15, "x")
mb_y = fractal_metrics(driving, np.abs(yb) + 1e-15, "y")
mb_z = fractal_metrics(driving, np.abs(zb) + 1e-15, "z")
print(f"  x: KS={mb_x['ks']:.4f}, R²={mb_x['r2']:.4f}, D={mb_x['dim']:.4f}, "
      f"trans={mb_x['trans']}")
print(f"  y: KS={mb_y['ks']:.4f}, R²={mb_y['r2']:.4f}, D={mb_y['dim']:.4f}, "
      f"trans={mb_y['trans']}")
print(f"  z: KS={mb_z['ks']:.4f}, R²={mb_z['r2']:.4f}, D={mb_z['dim']:.4f}, "
      f"trans={mb_z['trans']}")

# --- Candidate C ---
print("\n=== CANDIDATE C: Bounded Polynomial ===")
xc, yc, zc = bounded_polynomial_system(driving)
mc_x = fractal_metrics(driving, np.abs(xc) + 1e-15, "x")
mc_y = fractal_metrics(driving, np.abs(yc) + 1e-15, "y")
mc_z = fractal_metrics(driving, np.abs(zc) + 1e-15, "z")
print(f"  x: KS={mc_x['ks']:.4f}, R²={mc_x['r2']:.4f}, D={mc_x['dim']:.4f}, "
      f"trans={mc_x['trans']}")
print(f"  y: KS={mc_y['ks']:.4f}, R²={mc_y['r2']:.4f}, D={mc_y['dim']:.4f}, "
      f"trans={mc_y['trans']}")
print(f"  z: KS={mc_z['ks']:.4f}, R²={mc_z['r2']:.4f}, D={mc_z['dim']:.4f}, "
      f"trans={mc_z['trans']}")

# ============================================================
# FIGURE
# ============================================================

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('TEST 4: CONSTRUCTED COUNTEREXAMPLE — Three Attempts to Break the Law\n'
             'Can nonlinear coupled systems avoid fractal classification?',
             fontsize=14, fontweight='bold')

# Panel 1: Candidate A
ax = axes[0, 0]
ax.loglog(driving, xa, 'b-', linewidth=1.5, label='x_out', alpha=0.8)
ax.loglog(driving, ya, 'r-', linewidth=1.5, label='y_out', alpha=0.8)
ax.set_xlabel('Driving variable')
ax.set_ylabel('Output')
ax.set_title('A: Piecewise-Linear Switching')
ax.legend()
ax.grid(True, alpha=0.3)

# Panel 2: Candidate B
ax = axes[0, 1]
ax.semilogx(driving, xb, 'b-', linewidth=1.5, label='x', alpha=0.8)
ax.semilogx(driving, yb, 'r-', linewidth=1.5, label='y', alpha=0.8)
ax.semilogx(driving, zb, 'g-', linewidth=1.5, label='z', alpha=0.8)
ax.set_xlabel('Driving variable')
ax.set_ylabel('Output')
ax.set_title('B: Saturating (tanh) Coupling')
ax.legend()
ax.grid(True, alpha=0.3)

# Panel 3: Candidate C
ax = axes[0, 2]
valid = (np.abs(xc) > 1e-20) | (np.abs(yc) > 1e-20) | (np.abs(zc) > 1e-20)
ax.semilogx(driving[valid], xc[valid], 'b-', linewidth=1.5, label='x', alpha=0.8)
ax.semilogx(driving[valid], yc[valid], 'r-', linewidth=1.5, label='y', alpha=0.8)
ax.semilogx(driving[valid], zc[valid], 'g-', linewidth=1.5, label='z', alpha=0.8)
ax.set_xlabel('Driving variable')
ax.set_ylabel('Output')
ax.set_title('C: Bounded Polynomial')
ax.legend()
ax.grid(True, alpha=0.3)

# Panel 4: Dimension comparison
ax = axes[1, 0]
candidates = ['A-x', 'A-y', 'B-x', 'B-y', 'B-z', 'C-x', 'C-y', 'C-z']
dims = [ma_x['dim'], ma_y['dim'], mb_x['dim'], mb_y['dim'], mb_z['dim'],
        mc_x['dim'], mc_y['dim'], mc_z['dim']]
colors = ['#2196F3', '#2196F3', '#FF5722', '#FF5722', '#FF5722', 
          '#4CAF50', '#4CAF50', '#4CAF50']
ax.bar(range(len(dims)), dims, color=colors, alpha=0.7)
ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=2, label='D=1 (Euclidean)')
ax.set_xticks(range(len(candidates)))
ax.set_xticklabels(candidates, rotation=45)
ax.set_ylabel('Box-counting dimension')
ax.set_title('Fractal Dimension Across All Candidates')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Panel 5: Self-similarity comparison
ax = axes[1, 1]
ks_vals = [ma_x['ks'], ma_y['ks'], mb_x['ks'], mb_y['ks'], mb_z['ks'],
           mc_x['ks'], mc_y['ks'], mc_z['ks']]
ax.bar(range(len(ks_vals)), ks_vals, color=colors, alpha=0.7)
ax.axhline(y=0.3, color='red', linestyle='--', linewidth=2, label='Self-similarity threshold')
ax.set_xticks(range(len(candidates)))
ax.set_xticklabels(candidates, rotation=45)
ax.set_ylabel('KS statistic (lower = more self-similar)')
ax.set_title('Self-Similarity Across All Candidates')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Panel 6: Verdict
ax = axes[1, 2]
ax.axis('off')

verdict_text = """
COUNTEREXAMPLE RESULTS

Three strategies to avoid fractal classification:

A) Piecewise-linear: Nonlinearity only at boundaries
   → Does switching alone generate fractal structure?

B) Saturating (tanh): Bounded outputs
   → Does bounded nonlinearity still produce fractals?

C) Bounded polynomial: Self-limiting growth
   → Does exponential damping suppress fractal geometry?

IF ALL THREE SHOW FRACTAL SIGNATURES:
  → Cannot construct counterexample
  → Law may be mathematical theorem
  → Nonlinearity + coupling → fractal is NECESSARY

IF ANY SHOW EUCLIDEAN:
  → Counterexample found
  → Law has boundaries
  → Identify what property prevents fractal emergence

KEY QUESTION: Does the "extreme range" precondition
fail for bounded systems? If outputs saturate, is the
extreme range condition NOT satisfied?
"""
ax.text(0.05, 0.95, verdict_text, transform=ax.transAxes, fontsize=9,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig('39_counterexample_attempt.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nFigure saved: 39_counterexample_attempt.png")
