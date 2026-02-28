#!/usr/bin/env python3
"""
Test 5: BLIND PREDICTION — Three Unexamined Systems
The Lucian Law Falsification Protocol

Purpose: Use the Lucian Law to PREDICT classification BEFORE analysis.
Prediction recorded: All three systems will exhibit fractal geometric
classification when driving variable is extended across extreme range.

System A: Lotka-Volterra predator-prey (2 coupled, nonlinear)
System B: FitzHugh-Nagumo neuron model (2 coupled, nonlinear)
System C: Brusselator chemical oscillator (2 coupled, nonlinear)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy import stats

def fractal_metrics(x_values, y_values):
    """Standard fractal metric computation."""
    valid = np.isfinite(y_values) & (y_values > 0)
    if np.sum(valid) < 50:
        return {'ks': np.nan, 'r2': np.nan, 'dim': np.nan, 'trans': 0}
    
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
    
    return {'ks': ks, 'r2': r2, 'dim': dim, 'trans': trans}

# ============================================================
# PREDICTION (RECORDED BEFORE ANALYSIS)
# ============================================================
print("=" * 70)
print("TEST 5: BLIND PREDICTION")
print("The Lucian Law Falsification Protocol")
print("=" * 70)
print("""
PREDICTIONS (recorded before running any analysis):

Based on the Lucian Law — all nonlinear coupled systems with well-defined
extreme-range behavior exhibit fractal geometric classification:

  System A (Lotka-Volterra): FRACTAL GEOMETRIC
  System B (FitzHugh-Nagumo): FRACTAL GEOMETRIC  
  System C (Brusselator): FRACTAL GEOMETRIC

Now running analysis to test predictions...
""")

# ============================================================
# SYSTEM A: Lotka-Volterra Predator-Prey
# dx/dt = αx - βxy
# dy/dt = δxy - γy
# Driving variable: α (prey growth rate) across extreme range
# ============================================================

print("=== SYSTEM A: Lotka-Volterra ===")
alpha_values = np.logspace(-4, 4, 500)  # 8 orders of magnitude
beta, delta_lv, gamma = 0.1, 0.075, 0.5

# For each alpha, find steady-state amplitudes
# Equilibrium: x* = γ/δ, y* = α/β (when it exists)
# We compute the equilibrium values and oscillation characteristics

lv_x_eq = np.full(len(alpha_values), gamma / delta_lv)  # x* = γ/δ (constant)
lv_y_eq = alpha_values / beta  # y* = α/β (scales with α)

# For dynamics, integrate and get max amplitude
lv_max_x = np.zeros(len(alpha_values))
lv_max_y = np.zeros(len(alpha_values))

for i, alpha in enumerate(alpha_values):
    def lotka_volterra(t, state):
        x, y = state
        if x < 0: x = 0
        if y < 0: y = 0
        dxdt = alpha * x - beta * x * y
        dydt = delta_lv * x * y - gamma * y
        return [dxdt, dydt]
    
    x0 = gamma / delta_lv * 1.5  # Start near but not at equilibrium
    y0 = alpha / beta * 0.5
    
    if x0 > 0 and y0 > 0 and np.isfinite(x0) and np.isfinite(y0):
        try:
            sol = solve_ivp(lotka_volterra, [0, 200], [x0, y0],
                           t_eval=np.linspace(100, 200, 1000),
                           method='RK45', max_step=0.5, rtol=1e-8, atol=1e-10)
            if sol.success and len(sol.y[0]) > 10:
                lv_max_x[i] = np.max(sol.y[0])
                lv_max_y[i] = np.max(sol.y[1])
            else:
                lv_max_x[i] = lv_max_y[i] = np.nan
        except:
            lv_max_x[i] = lv_max_y[i] = np.nan
    else:
        lv_max_x[i] = lv_max_y[i] = np.nan

lv_metrics = fractal_metrics(alpha_values, lv_max_y)
print(f"  Driving: α from {alpha_values[0]:.0e} to {alpha_values[-1]:.0e}")
print(f"  KS={lv_metrics['ks']:.4f}, R²={lv_metrics['r2']:.4f}, "
      f"D={lv_metrics['dim']:.4f}, trans={lv_metrics['trans']}")

# ============================================================
# SYSTEM B: FitzHugh-Nagumo Neuron Model
# dv/dt = v - v³/3 - w + I_ext
# dw/dt = ε(v + a - bw)
# Driving variable: I_ext (external current) across extreme range
# ============================================================

print("\n=== SYSTEM B: FitzHugh-Nagumo ===")
I_values = np.logspace(-4, 4, 500)
epsilon, a_fhn, b_fhn = 0.08, 0.7, 0.8

fhn_max_v = np.zeros(len(I_values))
fhn_max_w = np.zeros(len(I_values))

for i, I_ext in enumerate(I_values):
    def fitzhugh_nagumo(t, state):
        v, w = state
        dvdt = v - v**3/3 - w + I_ext
        dwdt = epsilon * (v + a_fhn - b_fhn * w)
        return [dvdt, dwdt]
    
    try:
        sol = solve_ivp(fitzhugh_nagumo, [0, 500], [0.0, 0.0],
                       t_eval=np.linspace(200, 500, 1000),
                       method='RK45', max_step=1.0, rtol=1e-8, atol=1e-10)
        if sol.success and len(sol.y[0]) > 10:
            fhn_max_v[i] = np.max(np.abs(sol.y[0]))
            fhn_max_w[i] = np.max(np.abs(sol.y[1]))
        else:
            fhn_max_v[i] = fhn_max_w[i] = np.nan
    except:
        fhn_max_v[i] = fhn_max_w[i] = np.nan

fhn_metrics = fractal_metrics(I_values, fhn_max_v)
print(f"  Driving: I_ext from {I_values[0]:.0e} to {I_values[-1]:.0e}")
print(f"  KS={fhn_metrics['ks']:.4f}, R²={fhn_metrics['r2']:.4f}, "
      f"D={fhn_metrics['dim']:.4f}, trans={fhn_metrics['trans']}")

# ============================================================
# SYSTEM C: Brusselator Chemical Oscillator
# dx/dt = A - (B+1)x + x²y
# dy/dt = Bx - x²y
# Driving variable: B (control parameter) across extreme range
# ============================================================

print("\n=== SYSTEM C: Brusselator ===")
B_values = np.logspace(-2, 4, 500)
A_brus = 1.0

brus_max_x = np.zeros(len(B_values))
brus_max_y = np.zeros(len(B_values))

for i, B in enumerate(B_values):
    def brusselator(t, state):
        x, y = state
        if x < 0: x = 1e-10
        if y < 0: y = 1e-10
        dxdt = A_brus - (B + 1) * x + x**2 * y
        dydt = B * x - x**2 * y
        return [dxdt, dydt]
    
    try:
        sol = solve_ivp(brusselator, [0, 500], [A_brus, B/A_brus],
                       t_eval=np.linspace(200, 500, 1000),
                       method='RK45', max_step=0.5, rtol=1e-8, atol=1e-10)
        if sol.success and len(sol.y[0]) > 10:
            brus_max_x[i] = np.max(np.abs(sol.y[0]))
            brus_max_y[i] = np.max(np.abs(sol.y[1]))
        else:
            brus_max_x[i] = brus_max_y[i] = np.nan
    except:
        brus_max_x[i] = brus_max_y[i] = np.nan

brus_metrics = fractal_metrics(B_values, brus_max_x)
print(f"  Driving: B from {B_values[0]:.0e} to {B_values[-1]:.0e}")
print(f"  KS={brus_metrics['ks']:.4f}, R²={brus_metrics['r2']:.4f}, "
      f"D={brus_metrics['dim']:.4f}, trans={brus_metrics['trans']}")

# ============================================================
# VERDICT
# ============================================================
print("\n" + "=" * 70)
print("PREDICTION RESULTS")
print("=" * 70)

systems = [
    ("Lotka-Volterra", lv_metrics),
    ("FitzHugh-Nagumo", fhn_metrics),
    ("Brusselator", brus_metrics)
]

for name, m in systems:
    # Simple classification heuristic
    fractal_score = 0
    if m['ks'] < 0.5: fractal_score += 1  # some self-similarity
    if m['r2'] > 0.7: fractal_score += 1  # power-law scaling
    if m['dim'] > 1.05: fractal_score += 1  # non-integer dimension
    if m['trans'] > 0: fractal_score += 1  # phase transitions present
    
    classification = "FRACTAL" if fractal_score >= 2 else "EUCLIDEAN" if fractal_score == 0 else "AMBIGUOUS"
    correct = "✓ PREDICTION CORRECT" if classification == "FRACTAL" else "✗ PREDICTION FAILED" if classification == "EUCLIDEAN" else "? NEEDS DEEPER ANALYSIS"
    
    print(f"\n  {name}: {classification} (score {fractal_score}/4) {correct}")

# ============================================================
# FIGURE
# ============================================================

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('TEST 5: BLIND PREDICTION — Three Unexamined Systems\n'
             'Prediction: ALL exhibit fractal geometric classification',
             fontsize=14, fontweight='bold')

# Panel 1: Lotka-Volterra
ax = axes[0, 0]
valid = np.isfinite(lv_max_x) & (lv_max_x > 0)
ax.loglog(alpha_values[valid], lv_max_x[valid], 'b-', linewidth=1.5, label='max(prey)', alpha=0.7)
valid = np.isfinite(lv_max_y) & (lv_max_y > 0)
ax.loglog(alpha_values[valid], lv_max_y[valid], 'r-', linewidth=1.5, label='max(predator)', alpha=0.7)
ax.set_xlabel('α (prey growth rate)')
ax.set_ylabel('Maximum population')
ax.set_title('A: Lotka-Volterra Predator-Prey')
ax.legend()
ax.grid(True, alpha=0.3)

# Panel 2: FitzHugh-Nagumo
ax = axes[0, 1]
valid = np.isfinite(fhn_max_v) & (fhn_max_v > 0)
ax.loglog(I_values[valid], fhn_max_v[valid], 'b-', linewidth=1.5, label='max|v|', alpha=0.7)
valid = np.isfinite(fhn_max_w) & (fhn_max_w > 0)
ax.loglog(I_values[valid], fhn_max_w[valid], 'r-', linewidth=1.5, label='max|w|', alpha=0.7)
ax.set_xlabel('I_ext (external current)')
ax.set_ylabel('Maximum values')
ax.set_title('B: FitzHugh-Nagumo Neuron')
ax.legend()
ax.grid(True, alpha=0.3)

# Panel 3: Brusselator
ax = axes[0, 2]
valid = np.isfinite(brus_max_x) & (brus_max_x > 0)
ax.loglog(B_values[valid], brus_max_x[valid], 'b-', linewidth=1.5, label='max|x|', alpha=0.7)
valid = np.isfinite(brus_max_y) & (brus_max_y > 0)
ax.loglog(B_values[valid], brus_max_y[valid], 'r-', linewidth=1.5, label='max|y|', alpha=0.7)
ax.set_xlabel('B (control parameter)')
ax.set_ylabel('Maximum concentrations')
ax.set_title('C: Brusselator Chemical Oscillator')
ax.legend()
ax.grid(True, alpha=0.3)

# Panel 4: Classification scores
ax = axes[1, 0]
names = ['Lotka-\nVolterra', 'FitzHugh-\nNagumo', 'Brusselator']
scores = []
for _, m in systems:
    s = 0
    if m['ks'] < 0.5: s += 1
    if m['r2'] > 0.7: s += 1
    if m['dim'] > 1.05: s += 1
    if m['trans'] > 0: s += 1
    scores.append(s)
colors = ['#4CAF50' if s >= 2 else '#FF5722' if s == 0 else '#FFC107' for s in scores]
ax.bar(names, scores, color=colors, alpha=0.7)
ax.axhline(y=2, color='green', linestyle='--', alpha=0.5, label='Fractal threshold')
ax.set_ylabel('Fractal criteria satisfied (of 4)')
ax.set_title('Classification Scores')
ax.legend()
ax.set_ylim(0, 4.5)
ax.grid(True, alpha=0.3, axis='y')

# Panel 5: Dimension comparison
ax = axes[1, 1]
dims = [lv_metrics['dim'], fhn_metrics['dim'], brus_metrics['dim']]
ax.bar(names, dims, color=['#2196F3', '#FF5722', '#4CAF50'], alpha=0.7)
ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=2, label='D=1 (Euclidean)')
ax.set_ylabel('Box-counting dimension')
ax.set_title('Fractal Dimension')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Panel 6: Verdict
ax = axes[1, 2]
ax.axis('off')
verdict = """
BLIND PREDICTION RESULTS

Prediction: All three systems → FRACTAL GEOMETRIC

Lotka-Volterra:  KS={lv_ks:.3f}  R²={lv_r2:.3f}  D={lv_d:.3f}
FitzHugh-Nagumo: KS={fhn_ks:.3f}  R²={fhn_r2:.3f}  D={fhn_d:.3f}
Brusselator:     KS={br_ks:.3f}  R²={br_r2:.3f}  D={br_d:.3f}

IF ALL THREE CONFIRM FRACTAL:
  → Law has predictive power
  → Classification follows from structure alone
  → 3/3 blind predictions correct

IF ANY FAIL:
  → Identify what property differs
  → Refine boundary conditions of law
  → Prediction failure = most informative result
""".format(
    lv_ks=lv_metrics['ks'], lv_r2=lv_metrics['r2'], lv_d=lv_metrics['dim'],
    fhn_ks=fhn_metrics['ks'], fhn_r2=fhn_metrics['r2'], fhn_d=fhn_metrics['dim'],
    br_ks=brus_metrics['ks'], br_r2=brus_metrics['r2'], br_d=brus_metrics['dim']
)
ax.text(0.05, 0.95, verdict, transform=ax.transAxes, fontsize=9,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig('40_blind_prediction.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nFigure saved: 40_blind_prediction.png")
