#!/usr/bin/env python3
"""
Test 2: NONLINEARITY THRESHOLD — Duffing Oscillator
The Lucian Law Falsification Protocol

Purpose: Determine whether fractal geometry appears at ANY nonzero nonlinearity
or only above a threshold. Uses Duffing oscillator with tunable beta.

System: x'' + delta*x' + alpha*x + beta*x^3 = gamma*cos(omega*t)
Driving variable: omega (forcing frequency) swept across 4 orders of magnitude
Nonlinearity parameter: beta swept from 0 to 1 in 11 steps

Optimized: LSODA solver, adaptive integration, reduced resolution.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy import stats
import sys


def duffing_response(omega_values, beta, delta=0.3, alpha=1.0, gamma=0.5):
    """
    Compute steady-state amplitude response of Duffing oscillator.
    Uses LSODA with adaptive integration time.
    """
    amplitudes = np.full(len(omega_values), np.nan)

    for i, omega in enumerate(omega_values):
        period = 2 * np.pi / max(omega, 0.01)
        t_max = min(max(50, 10 * period), 500)  # cap at 500
        t_trans = min(max(15, 3 * period), 200)

        def duffing(t, y, omega=omega, beta=beta):
            x, v = y
            return [v, -delta * v - alpha * x - beta * x**3 + gamma * np.cos(omega * t)]

        try:
            sol = solve_ivp(duffing, [0, t_max], [0.1, 0],
                           t_eval=np.linspace(t_trans, t_max, 200),
                           method='LSODA', rtol=1e-5, atol=1e-7,
                           max_step=min(1.0, period/10))
            if sol.success and len(sol.y[0]) > 10:
                amplitudes[i] = np.max(np.abs(sol.y[0])) - np.min(np.abs(sol.y[0]))
        except:
            pass

    return amplitudes


def compute_fractal_metrics(omega_values, amplitudes):
    """Compute fractal classification metrics for a frequency response curve."""
    metrics = {}

    valid = np.isfinite(amplitudes) & (amplitudes > 0)
    if np.sum(valid) < 20:
        return {'self_similarity_ks': np.nan, 'power_law_r2': np.nan,
                'box_dim': np.nan, 'n_transitions': 0}

    amp_valid = amplitudes[valid]
    omega_valid = omega_values[valid]

    # 1. Self-similarity (KS test across segments)
    n_seg = 4
    seg_size = len(amp_valid) // n_seg
    ks_stats = []
    if seg_size > 5:
        for j in range(n_seg - 1):
            s1 = amp_valid[j*seg_size:(j+1)*seg_size]
            s2 = amp_valid[(j+1)*seg_size:(j+2)*seg_size]
            if np.std(s1) > 0 and np.std(s2) > 0:
                s1n = (s1 - np.mean(s1)) / np.std(s1)
                s2n = (s2 - np.mean(s2)) / np.std(s2)
                ks, _ = stats.ks_2samp(s1n, s2n)
                ks_stats.append(ks)
    metrics['self_similarity_ks'] = np.mean(ks_stats) if ks_stats else 1.0

    # 2. Power-law R^2
    mask = (omega_valid > 0) & (amp_valid > 0)
    if np.sum(mask) > 10:
        slope, intercept, r, p, se = stats.linregress(
            np.log10(omega_valid[mask]), np.log10(amp_valid[mask]))
        metrics['power_law_r2'] = r**2
    else:
        metrics['power_law_r2'] = 0.0

    # 3. Box-counting dimension
    amp_norm = (amp_valid - np.min(amp_valid)) / (np.max(amp_valid) - np.min(amp_valid) + 1e-15)
    x_norm = np.linspace(0, 1, len(amp_norm))
    n_boxes_list = [8, 16, 32, 64]
    log_inv_size, log_count = [], []
    for nb in n_boxes_list:
        bs = 1.0 / nb
        bx = np.minimum((x_norm / bs).astype(int), nb-1)
        by = np.minimum((amp_norm / bs).astype(int), nb-1)
        occ = len(set(zip(bx, by)))
        log_inv_size.append(np.log(nb))
        log_count.append(np.log(occ))
    if len(log_inv_size) > 2:
        slope, _, r, _, _ = stats.linregress(log_inv_size, log_count)
        metrics['box_dim'] = slope
    else:
        metrics['box_dim'] = 1.0

    # 4. Phase transitions
    d_amp = np.diff(amp_valid)
    if np.std(d_amp) > 0:
        z = np.abs(d_amp - np.mean(d_amp)) / np.std(d_amp)
        metrics['n_transitions'] = int(np.sum(z > 5))
    else:
        metrics['n_transitions'] = 0

    return metrics


# ============================================================
# MAIN ANALYSIS
# ============================================================

print("=" * 70)
print("TEST 2: NONLINEARITY THRESHOLD — Duffing Oscillator")
print("The Lucian Law Falsification Protocol")
print("=" * 70)
sys.stdout.flush()

# Nonlinearity parameter sweep
beta_values = np.linspace(0, 1.0, 11)

# Driving variable: forcing frequency — 4 orders centered on resonance
omega_values = np.logspace(-1, 3, 80)

print(f"\nDriving variable: omega from {omega_values[0]:.0e} to {omega_values[-1]:.0e}")
print(f"Nonlinearity beta: {len(beta_values)} values from 0 to 1")
print(f"Points per response curve: {len(omega_values)}")
sys.stdout.flush()

# Store results
all_metrics = []
all_amplitudes = []

for idx, beta in enumerate(beta_values):
    print(f"\n  beta = {beta:.3f} ({idx+1}/{len(beta_values)})...", end='', flush=True)

    amp = duffing_response(omega_values, beta)
    metrics = compute_fractal_metrics(omega_values, amp)
    all_metrics.append(metrics)
    all_amplitudes.append(amp)

    print(f" KS={metrics['self_similarity_ks']:.3f}, "
          f"R2={metrics['power_law_r2']:.3f}, "
          f"D={metrics['box_dim']:.3f}, "
          f"transitions={metrics['n_transitions']}", flush=True)

# ============================================================
# FIGURE
# ============================================================

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('TEST 2: NONLINEARITY THRESHOLD — Duffing Oscillator\n'
             'How does fractal classification emerge as nonlinearity increases?',
             fontsize=14, fontweight='bold')

# Panel 1: Response curves at selected beta values
ax = axes[0, 0]
selected_betas = [0, 2, 5, 8, 10]
colors_sel = plt.cm.viridis(np.linspace(0.1, 0.9, len(selected_betas)))
for i, idx in enumerate(selected_betas):
    valid = np.isfinite(all_amplitudes[idx])
    ax.loglog(omega_values[valid], all_amplitudes[idx][valid],
             color=colors_sel[i], linewidth=1.5, label=f'beta={beta_values[idx]:.2f}')
ax.set_xlabel('omega (driving variable)')
ax.set_ylabel('Response amplitude')
ax.set_title('Frequency Response at Selected beta')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Panel 2: Self-similarity KS vs beta
ax = axes[0, 1]
ks_vals = [m['self_similarity_ks'] for m in all_metrics]
ax.plot(beta_values, ks_vals, 'o-', color='#2196F3', linewidth=2)
ax.axhline(y=0.3, color='red', linestyle='--', alpha=0.5, label='Threshold (KS=0.3)')
ax.set_xlabel('beta (nonlinearity)')
ax.set_ylabel('KS statistic (lower = more self-similar)')
ax.set_title('Self-Similarity vs Nonlinearity')
ax.legend()
ax.grid(True, alpha=0.3)

# Panel 3: Power-law R^2 vs beta
ax = axes[0, 2]
r2_vals = [m['power_law_r2'] for m in all_metrics]
ax.plot(beta_values, r2_vals, 'o-', color='#FF5722', linewidth=2)
ax.set_xlabel('beta (nonlinearity)')
ax.set_ylabel('R^2 (power-law fit)')
ax.set_title('Power-Law Scaling vs Nonlinearity')
ax.grid(True, alpha=0.3)

# Panel 4: Box-counting dimension vs beta
ax = axes[1, 0]
dim_vals = [m['box_dim'] for m in all_metrics]
ax.plot(beta_values, dim_vals, 'o-', color='#4CAF50', linewidth=2)
ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='D=1 (Euclidean)')
ax.set_xlabel('beta (nonlinearity)')
ax.set_ylabel('Box-counting dimension')
ax.set_title('Fractal Dimension vs Nonlinearity')
ax.legend()
ax.grid(True, alpha=0.3)

# Panel 5: Number of transitions vs beta
ax = axes[1, 1]
trans_vals = [m['n_transitions'] for m in all_metrics]
ax.plot(beta_values, trans_vals, 'o-', color='#9C27B0', linewidth=2)
ax.set_xlabel('beta (nonlinearity)')
ax.set_ylabel('Number of phase transitions')
ax.set_title('Phase Transitions vs Nonlinearity')
ax.grid(True, alpha=0.3)

# Panel 6: Composite score and interpretation
ax = axes[1, 2]
ax.axis('off')
summary_text = f"""
NONLINEARITY THRESHOLD RESULTS

System: Duffing Oscillator x'' + dx' + ax + bx^3 = g*cos(wt)
Driving variable: omega across 4 orders of magnitude
Nonlinearity: beta swept from 0 to 1

{'beta':>6} {'KS':>8} {'R2':>8} {'Dim':>8} {'Trans':>6}
{'-'*40}"""
for idx, beta in enumerate(beta_values):
    m = all_metrics[idx]
    summary_text += f"\n{beta:>6.2f} {m['self_similarity_ks']:>8.3f} {m['power_law_r2']:>8.3f} {m['box_dim']:>8.3f} {m['n_transitions']:>6d}"

ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=8,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig('37_nonlinearity_threshold.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n\nFigure saved: 37_nonlinearity_threshold.png", flush=True)
