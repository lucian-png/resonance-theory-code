#!/usr/bin/env python3
"""
Script 61: THE FULL SPARC SWEEP
================================
175 galaxies. Three values of p. One question.

For each galaxy, compute χ² with:
  1. p = 0.5000  (standard RAR — the empirical benchmark)
  2. p = 0.4499  (log_δ(2) — the derived value)
  3. p = free    (best fit per galaxy — the distribution)

If the histogram of best-fit p peaks near 0.45: the architecture wins.
If it peaks near 0.50: McGaugh was right and NGC 3198 was coincidence.

The shop doesn't care what we want. The shop tells us what's true.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.optimize import minimize_scalar
from scipy.stats import median_abs_deviation
import os
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("SCRIPT 61: THE FULL SPARC SWEEP")
print("175 Galaxies · Three Exponents · One Question")
print("=" * 70, flush=True)

# ============================================================
# CONSTANTS
# ============================================================
DELTA = 4.669201609102990
ALPHA_F = 2.502907875095892
LN_DELTA = np.log(DELTA)
LOG_D_2 = np.log(2) / LN_DELTA    # log_δ(2) = 0.44986...

A0_MOND = 1.2e-10                 # m/s²
KPC_TO_M = 3.0857e19
KM_TO_M = 1000.0

Y_DISK = 0.5
Y_BUL = 0.7

BASE = '/Users/lucianrandolph/Downloads/Projects/einstein_fractal_analysis'
SPARC_DIR = '/tmp/sparc_data'

print(f"\n  p values under test:")
print(f"    p₁ = 0.5000  (standard RAR, McGaugh+2016)")
print(f"    p₂ = {LOG_D_2:.4f}  (log_δ(2) = ln(2)/ln(δ), derived)")
print(f"    p₃ = free    (best-fit per galaxy)")

# ============================================================
# LOAD ALL SPARC DATA
# ============================================================
print("\n" + "=" * 70)
print("LOADING SPARC DATA")
print("=" * 70, flush=True)

# --- Galaxy properties (table1) ---
# Parse by splitting on whitespace (more robust than byte positions)
# Fields: Name T D e_D f_D Inc e_Inc L36 e_L36 Reff SBeff Rdisk SBdisk
#         MHI RHI Vflat e_Vflat Q Ref
galaxy_props = {}  # type: dict
with open(os.path.join(SPARC_DIR, 'table1.dat'), 'r') as f:
    lines_t1 = f.readlines()

# Find last '---' separator — data starts after it
last_sep = 0
for i, line in enumerate(lines_t1):
    if line.rstrip().startswith('---'):
        last_sep = i

for line in lines_t1[last_sep + 1:]:
    parts = line.split()
    if len(parts) < 18:
        continue
    try:
        name = parts[0]
        hubble_type = int(parts[1])
        dist = float(parts[2])
        inc = float(parts[5])
        vflat = float(parts[15])
        qual = int(parts[17])
        galaxy_props[name] = {
            'type': hubble_type,
            'dist': dist,
            'inc': inc,
            'Vflat': vflat,
            'quality': qual,
        }
    except (ValueError, IndexError):
        continue

print(f"  Galaxy properties: {len(galaxy_props)}")

# --- Rotation curves (table2) ---
galaxy_data = {}  # type: dict
with open(os.path.join(SPARC_DIR, 'table2.dat'), 'r') as f:
    for line in f:
        if line.strip() == '' or not line[0].isalpha():
            continue
        if line.startswith('Title') or line.startswith('Authors') or \
           line.startswith('Table') or line.startswith('Note') or \
           line.startswith('Byte') or line.startswith('---'):
            continue
        try:
            name = line[0:11].strip()
            rad = float(line[19:25].strip())
            vobs = float(line[26:32].strip())
            e_vobs = float(line[33:38].strip())
            vgas = float(line[39:45].strip())
            vdisk = float(line[46:52].strip())
            vbul = float(line[53:59].strip())

            if name not in galaxy_data:
                galaxy_data[name] = []
            galaxy_data[name].append({
                'R': rad,
                'Vobs': vobs,
                'e_Vobs': max(e_vobs, 1.0),
                'Vgas': vgas,
                'Vdisk': vdisk,
                'Vbul': vbul,
            })
        except (ValueError, IndexError):
            continue

n_galaxies = len(galaxy_data)
n_points = sum(len(v) for v in galaxy_data.values())
print(f"  Rotation curves: {n_galaxies} galaxies, {n_points} data points")

# ============================================================
# CORE FUNCTIONS
# ============================================================


def tau_gen_rar(x_arr: np.ndarray, p: float) -> np.ndarray:
    """Generalized RAR: τ = √(1 - exp(-x^p))"""
    xp = np.power(np.maximum(x_arr, 1e-30), p)
    return np.sqrt(np.maximum(1e-20, 1.0 - np.exp(-xp)))


def compute_galaxy(rows: list[dict[str, float]],
                   p_val: float) -> tuple[float, int]:
    """Compute reduced χ² for a galaxy with given p.
    Returns (chi2_red, n_points)."""
    R_kpc = np.array([r['R'] for r in rows])
    Vobs = np.array([r['Vobs'] for r in rows])
    e_Vobs = np.array([r['e_Vobs'] for r in rows])
    Vgas = np.array([r['Vgas'] for r in rows])
    Vdisk = np.array([r['Vdisk'] for r in rows])
    Vbul = np.array([r['Vbul'] for r in rows])

    # Baryonic velocity
    Vbar_sq = (np.sign(Vgas) * Vgas**2 +
               Y_DISK * Vdisk**2 +
               Y_BUL * Vbul**2)
    Vbar_sq = np.maximum(Vbar_sq, 1.0)
    Vbar = np.sqrt(Vbar_sq)

    # Acceleration
    R_m = R_kpc * KPC_TO_M
    valid = R_m > 0
    g_bar = np.zeros_like(R_kpc)
    g_bar[valid] = Vbar_sq[valid] * KM_TO_M**2 / R_m[valid]

    # Dimensionless acceleration
    x = g_bar / A0_MOND

    # τ and predicted velocity
    tau = tau_gen_rar(x, p_val)
    Vpred = np.abs(Vbar) / np.maximum(tau, 1e-10)

    # χ²
    resid = (Vobs - Vpred) / e_Vobs
    n = len(resid)
    dof = max(n - 1, 1)
    chi2_red = float(np.sum(resid**2) / dof)

    return chi2_red, n


def find_best_p(rows: list[dict[str, float]]) -> tuple[float, float]:
    """Find best-fit p for a galaxy. Returns (p_best, chi2_best)."""
    def objective(p: float) -> float:
        chi2, _ = compute_galaxy(rows, p)
        return chi2

    result = minimize_scalar(objective, bounds=(0.1, 2.0), method='bounded')
    return float(result.x), float(result.fun)


# ============================================================
# THE SWEEP
# ============================================================
print("\n" + "=" * 70)
print("THE SWEEP: 175 GALAXIES × 3 EXPONENTS")
print("=" * 70, flush=True)

P_STANDARD = 0.5
P_DERIVED = LOG_D_2  # ln(2)/ln(δ)

results = []  # type: list[dict]
n_processed = 0
n_skipped = 0

for name, rows in galaxy_data.items():
    if len(rows) < 5:
        n_skipped += 1
        continue

    # Quality filter — use Q=1 and Q=2
    qual = galaxy_props.get(name, {}).get('quality', 3)
    if qual > 2:
        n_skipped += 1
        continue

    # Compute χ² for standard RAR
    chi2_std, n_pts = compute_galaxy(rows, P_STANDARD)

    # Compute χ² for derived p
    chi2_der, _ = compute_galaxy(rows, P_DERIVED)

    # Find best-fit p
    p_best, chi2_best = find_best_p(rows)

    results.append({
        'name': name,
        'n_pts': n_pts,
        'quality': qual,
        'chi2_std': chi2_std,
        'chi2_der': chi2_der,
        'chi2_free': chi2_best,
        'p_best': p_best,
    })

    n_processed += 1
    if n_processed % 25 == 0:
        print(f"  Processed {n_processed} galaxies...", flush=True)

print(f"\n  Processed: {n_processed} galaxies")
print(f"  Skipped: {n_skipped} (low quality or too few points)")

# ============================================================
# AGGREGATE RESULTS
# ============================================================
print("\n" + "=" * 70)
print("AGGREGATE RESULTS")
print("=" * 70, flush=True)

chi2_std_all = np.array([r['chi2_std'] for r in results])
chi2_der_all = np.array([r['chi2_der'] for r in results])
chi2_free_all = np.array([r['chi2_free'] for r in results])
p_best_all = np.array([r['p_best'] for r in results])
n_pts_all = np.array([r['n_pts'] for r in results])

# Weighted averages (weight by number of points)
total_pts = np.sum(n_pts_all)
w = n_pts_all / total_pts

chi2_std_wavg = np.sum(w * chi2_std_all)
chi2_der_wavg = np.sum(w * chi2_der_all)
chi2_free_wavg = np.sum(w * chi2_free_all)

print(f"\n  {'Metric':<35s}  {'p=0.5':>10s}  {'p=log_δ(2)':>10s}  {'p=free':>10s}")
print(f"  {'---':<35s}  {'---':>10s}  {'---':>10s}  {'---':>10s}")
print(f"  {'Median χ²_red':<35s}  {np.median(chi2_std_all):10.2f}  "
      f"{np.median(chi2_der_all):10.2f}  {np.median(chi2_free_all):10.2f}")
print(f"  {'Mean χ²_red':<35s}  {np.mean(chi2_std_all):10.2f}  "
      f"{np.mean(chi2_der_all):10.2f}  {np.mean(chi2_free_all):10.2f}")
print(f"  {'Weighted mean χ²_red':<35s}  {chi2_std_wavg:10.2f}  "
      f"{chi2_der_wavg:10.2f}  {chi2_free_wavg:10.2f}")
print(f"  {'Fraction χ² < 5':<35s}  "
      f"{np.mean(chi2_std_all < 5)*100:9.1f}%  "
      f"{np.mean(chi2_der_all < 5)*100:9.1f}%  "
      f"{np.mean(chi2_free_all < 5)*100:9.1f}%")
print(f"  {'Fraction χ² < 2':<35s}  "
      f"{np.mean(chi2_std_all < 2)*100:9.1f}%  "
      f"{np.mean(chi2_der_all < 2)*100:9.1f}%  "
      f"{np.mean(chi2_free_all < 2)*100:9.1f}%")

# Which p wins per galaxy?
n_std_wins = np.sum(chi2_std_all < chi2_der_all)
n_der_wins = np.sum(chi2_der_all < chi2_std_all)
n_ties = np.sum(chi2_std_all == chi2_der_all)

print(f"\n  HEAD TO HEAD (p=0.5 vs p=log_δ(2)):")
print(f"    p=0.5 wins:      {n_std_wins} galaxies ({n_std_wins/len(results)*100:.1f}%)")
print(f"    p=log_δ(2) wins: {n_der_wins} galaxies ({n_der_wins/len(results)*100:.1f}%)")

# Chi2 improvement ratio
ratio = chi2_der_all / chi2_std_all
print(f"\n  χ²(derived) / χ²(standard) ratio:")
print(f"    Median: {np.median(ratio):.4f}")
print(f"    Mean:   {np.mean(ratio):.4f}")
print(f"    (< 1 means derived is better, > 1 means standard is better)")

# ============================================================
# BEST-FIT p DISTRIBUTION
# ============================================================
print("\n" + "=" * 70)
print("BEST-FIT p DISTRIBUTION")
print("=" * 70, flush=True)

# Filter out extreme p values (optimizer hitting bounds)
p_valid = p_best_all[(p_best_all > 0.15) & (p_best_all < 1.5)]
n_valid = len(p_valid)

print(f"\n  Total galaxies: {len(results)}")
print(f"  Valid p fits (0.15 < p < 1.5): {n_valid}")
print(f"\n  Distribution of best-fit p:")
print(f"    Mean:     {np.mean(p_valid):.4f}")
print(f"    Median:   {np.median(p_valid):.4f}")
print(f"    Std dev:  {np.std(p_valid):.4f}")
print(f"    MAD:      {median_abs_deviation(p_valid):.4f}")
print(f"    25th pct: {np.percentile(p_valid, 25):.4f}")
print(f"    75th pct: {np.percentile(p_valid, 75):.4f}")

print(f"\n  Key benchmarks vs median ({np.median(p_valid):.4f}):")
benchmarks = [
    ("p = 0.5 (standard RAR)", 0.5),
    ("p = log_δ(2) = 0.4499", LOG_D_2),
    ("p = 1/√δ = 0.4628", 1.0/np.sqrt(DELTA)),
    ("p = 1/α = 0.3996", 1.0/ALPHA_F),
]
for label, val in benchmarks:
    dev = (val - np.median(p_valid)) / np.median(p_valid) * 100
    print(f"    {label:<30s}  deviation: {dev:+.1f}%")

# How many galaxies have p_best closer to log_δ(2) than to 0.5?
closer_to_derived = np.sum(np.abs(p_valid - LOG_D_2) < np.abs(p_valid - 0.5))
closer_to_standard = np.sum(np.abs(p_valid - 0.5) < np.abs(p_valid - LOG_D_2))
print(f"\n  Galaxies with p_best closer to:")
print(f"    log_δ(2) = {LOG_D_2:.4f}: {closer_to_derived} "
      f"({closer_to_derived/n_valid*100:.1f}%)")
print(f"    0.5:            {closer_to_standard} "
      f"({closer_to_standard/n_valid*100:.1f}%)")

# ============================================================
# TOP 20 BEST-MEASURED GALAXIES
# ============================================================
print("\n" + "=" * 70)
print("TOP 20 BEST-MEASURED GALAXIES (most data points)")
print("=" * 70, flush=True)

sorted_results = sorted(results, key=lambda r: -r['n_pts'])
print(f"\n  {'Galaxy':<12s}  {'N':>4s}  {'χ²(0.5)':>8s}  {'χ²(logδ2)':>9s}  "
      f"{'χ²(free)':>8s}  {'p_best':>7s}  {'Winner':>8s}")
print(f"  {'---':<12s}  {'---':>4s}  {'---':>8s}  {'---':>9s}  "
      f"{'---':>8s}  {'---':>7s}  {'---':>8s}")

for r in sorted_results[:20]:
    winner = "log_δ(2)" if r['chi2_der'] < r['chi2_std'] else "p=0.5"
    print(f"  {r['name']:<12s}  {r['n_pts']:4d}  {r['chi2_std']:8.2f}  "
          f"{r['chi2_der']:9.2f}  {r['chi2_free']:8.2f}  "
          f"{r['p_best']:7.3f}  {winner:>8s}")

# ============================================================
# TOTAL χ² (GLOBAL GOODNESS OF FIT)
# ============================================================
print("\n" + "=" * 70)
print("GLOBAL GOODNESS OF FIT")
print("=" * 70, flush=True)

# Compute total χ² across ALL data points (not per-galaxy average)
total_chi2_std = 0.0
total_chi2_der = 0.0
total_chi2_logd2 = 0.0
total_dof = 0

for r in results:
    n = r['n_pts']
    total_chi2_std += r['chi2_std'] * max(n - 1, 1)
    total_chi2_der += r['chi2_der'] * max(n - 1, 1)
    total_dof += max(n - 1, 1)

global_chi2_std = total_chi2_std / total_dof
global_chi2_der = total_chi2_der / total_dof

print(f"\n  Total data points across all galaxies: {total_dof + len(results)}")
print(f"  Total degrees of freedom: {total_dof}")
print(f"\n  Global reduced χ²:")
print(f"    p = 0.5 (standard):    {global_chi2_std:.4f}")
print(f"    p = log_δ(2) (derived): {global_chi2_der:.4f}")
print(f"    Ratio (derived/standard): {global_chi2_der/global_chi2_std:.4f}")

if global_chi2_der < global_chi2_std:
    improvement = (1 - global_chi2_der / global_chi2_std) * 100
    print(f"    → log_δ(2) is BETTER by {improvement:.2f}%")
else:
    degradation = (global_chi2_der / global_chi2_std - 1) * 100
    print(f"    → p=0.5 is BETTER by {degradation:.2f}%")

# ============================================================
# GENERATE FIGURES
# ============================================================
print("\n" + "=" * 70)
print("GENERATING FIGURES")
print("=" * 70, flush=True)

fig = plt.figure(figsize=(20, 16))
gs = GridSpec(3, 3, figure=fig, hspace=0.38, wspace=0.35)
fig.suptitle('Script 61: THE FULL SPARC SWEEP\n'
             '175 Galaxies — Is the RAR Exponent Derived from Feigenbaum Space?',
             fontsize=16, fontweight='bold', y=0.98)

# --- Panel A: Histogram of best-fit p ---
ax = fig.add_subplot(gs[0, 0])
bins_p = np.linspace(0.15, 1.2, 40)
ax.hist(p_valid, bins=bins_p, color='#3498db', alpha=0.7,
        edgecolor='black', linewidth=0.5, density=True)
ax.axvline(x=0.5, color='green', linewidth=2.5, linestyle='--',
           label=f'p=0.5 (RAR)', alpha=0.8)
ax.axvline(x=LOG_D_2, color='red', linewidth=2.5, linestyle='-',
           label=f'p=log_δ(2)={LOG_D_2:.4f}', alpha=0.8)
ax.axvline(x=np.median(p_valid), color='orange', linewidth=2,
           linestyle=':',
           label=f'Median={np.median(p_valid):.3f}')
ax.set_xlabel('Best-fit p', fontsize=12)
ax.set_ylabel('Probability Density', fontsize=12)
ax.set_title('Distribution of Best-Fit Exponent\nAcross 175 SPARC Galaxies',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=9)
ax.set_xlim(0.15, 1.2)
ax.grid(True, alpha=0.3)

# --- Panel B: χ² comparison (derived vs standard) ---
ax = fig.add_subplot(gs[0, 1])
max_chi2 = 50
c_std = np.clip(chi2_std_all, 0, max_chi2)
c_der = np.clip(chi2_der_all, 0, max_chi2)
ax.scatter(c_std, c_der, c=p_best_all, cmap='RdYlBu_r',
           s=20, alpha=0.7, vmin=0.2, vmax=0.8, edgecolors='gray',
           linewidths=0.3)
ax.plot([0, max_chi2], [0, max_chi2], 'k--', linewidth=1, alpha=0.5,
        label='Equal fit')
ax.set_xlabel('χ² (p=0.5, standard RAR)', fontsize=12)
ax.set_ylabel(f'χ² (p=log_δ(2)={LOG_D_2:.3f})', fontsize=12)
ax.set_title('Per-Galaxy χ² Comparison\nBelow diagonal = derived wins',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=9)
ax.set_xlim(0, max_chi2)
ax.set_ylim(0, max_chi2)
ax.grid(True, alpha=0.3)
cbar = plt.colorbar(ax.collections[0], ax=ax, label='Best-fit p')

# --- Panel C: Cumulative distribution ---
ax = fig.add_subplot(gs[0, 2])
p_sorted = np.sort(p_valid)
cdf = np.arange(1, len(p_sorted)+1) / len(p_sorted)
ax.plot(p_sorted, cdf, 'b-', linewidth=2)
ax.axvline(x=0.5, color='green', linewidth=2, linestyle='--',
           label='p=0.5', alpha=0.7)
ax.axvline(x=LOG_D_2, color='red', linewidth=2, linestyle='-',
           label=f'log_δ(2)', alpha=0.7)
ax.axvline(x=np.median(p_valid), color='orange', linewidth=1.5,
           linestyle=':', alpha=0.7, label=f'Median')
ax.axhline(y=0.5, color='gray', linewidth=0.5, linestyle=':', alpha=0.3)
ax.set_xlabel('Best-fit p', fontsize=12)
ax.set_ylabel('Cumulative Fraction', fontsize=12)
ax.set_title('CDF of Best-Fit Exponents',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=9)
ax.set_xlim(0.15, 1.2)
ax.grid(True, alpha=0.3)

# --- Panel D: χ² improvement per galaxy ---
ax = fig.add_subplot(gs[1, 0])
improvement = chi2_std_all - chi2_der_all  # positive = derived better
imp_sorted = np.sort(improvement)
colors = ['red' if v > 0 else 'blue' for v in imp_sorted]
ax.bar(range(len(imp_sorted)), imp_sorted, color=colors, alpha=0.6, width=1.0)
ax.axhline(y=0, color='black', linewidth=1)
ax.set_xlabel('Galaxy (sorted by improvement)', fontsize=12)
ax.set_ylabel('χ²(p=0.5) - χ²(log_δ(2))', fontsize=12)
ax.set_title(f'Per-Galaxy Improvement\n'
             f'Red = log_δ(2) wins ({n_der_wins}), '
             f'Blue = p=0.5 wins ({n_std_wins})',
             fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)

# --- Panel E: p_best vs galaxy properties ---
ax = fig.add_subplot(gs[1, 1])
n_pts_arr = np.array([r['n_pts'] for r in results])
valid_mask = (p_best_all > 0.15) & (p_best_all < 1.5)
ax.scatter(n_pts_arr[valid_mask], p_best_all[valid_mask],
           c=chi2_free_all[valid_mask], cmap='viridis',
           s=25, alpha=0.7, vmin=0, vmax=10,
           edgecolors='gray', linewidths=0.3)
ax.axhline(y=0.5, color='green', linewidth=2, linestyle='--',
           label='p=0.5', alpha=0.7)
ax.axhline(y=LOG_D_2, color='red', linewidth=2, linestyle='-',
           label=f'log_δ(2)', alpha=0.7)
ax.set_xlabel('Number of Data Points', fontsize=12)
ax.set_ylabel('Best-fit p', fontsize=12)
ax.set_title('Best-Fit p vs Data Quality\nMore points = more constrained',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=9)
ax.set_xlim(0, max(n_pts_arr) + 5)
ax.set_ylim(0.1, 1.5)
ax.grid(True, alpha=0.3)
plt.colorbar(ax.collections[0], ax=ax, label='χ²(free)')

# --- Panel F: χ² distributions ---
ax = fig.add_subplot(gs[1, 2])
bins_chi2 = np.linspace(0, 30, 40)
ax.hist(chi2_std_all, bins=bins_chi2, alpha=0.5, color='green',
        label=f'p=0.5 (med={np.median(chi2_std_all):.1f})',
        edgecolor='green', linewidth=0.5, density=True)
ax.hist(chi2_der_all, bins=bins_chi2, alpha=0.5, color='red',
        label=f'log_δ(2) (med={np.median(chi2_der_all):.1f})',
        edgecolor='red', linewidth=0.5, density=True)
ax.axvline(x=1.0, color='black', linewidth=1.5, linestyle=':',
           alpha=0.5, label='χ²=1 (ideal)')
ax.set_xlabel('Reduced χ²', fontsize=12)
ax.set_ylabel('Probability Density', fontsize=12)
ax.set_title('χ² Distribution Comparison',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=9)
ax.set_xlim(0, 30)
ax.grid(True, alpha=0.3)

# --- Panel G: Well-constrained galaxies (N > 20) ---
ax = fig.add_subplot(gs[2, 0:2])
well_constrained = [r for r in results
                    if r['n_pts'] > 20 and 0.15 < r['p_best'] < 1.5]
wc_p = np.array([r['p_best'] for r in well_constrained])
wc_names = [r['name'] for r in well_constrained]
wc_chi2_std = np.array([r['chi2_std'] for r in well_constrained])
wc_chi2_der = np.array([r['chi2_der'] for r in well_constrained])

# Sort by p_best
sort_idx = np.argsort(wc_p)
wc_p = wc_p[sort_idx]
wc_names = [wc_names[i] for i in sort_idx]
wc_chi2_std = wc_chi2_std[sort_idx]
wc_chi2_der = wc_chi2_der[sort_idx]

ax.barh(range(len(wc_p)), wc_p, color=['red' if abs(p - LOG_D_2) < abs(p - 0.5)
        else 'green' for p in wc_p], alpha=0.6, edgecolor='gray', linewidth=0.3)
ax.axvline(x=0.5, color='green', linewidth=2, linestyle='--', alpha=0.7,
           label='p=0.5')
ax.axvline(x=LOG_D_2, color='red', linewidth=2, linestyle='-', alpha=0.7,
           label=f'log_δ(2)={LOG_D_2:.4f}')
ax.axvline(x=np.median(wc_p), color='orange', linewidth=1.5, linestyle=':',
           alpha=0.7, label=f'Median={np.median(wc_p):.3f}')
ax.set_yticks(range(len(wc_p)))
ax.set_yticklabels(wc_names, fontsize=6)
ax.set_xlabel('Best-fit p', fontsize=12)
ax.set_title(f'Well-Constrained Galaxies (N > 20 points, n={len(wc_p)})\n'
             f'Red = closer to log_δ(2), Green = closer to p=0.5',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=9, loc='upper right')
ax.set_xlim(0.1, 1.2)
ax.grid(True, alpha=0.3)

# --- Panel H: Summary ---
ax = fig.add_subplot(gs[2, 2])
ax.axis('off')

verdict = ""
if abs(np.median(p_valid) - LOG_D_2) < abs(np.median(p_valid) - 0.5):
    verdict = f"MEDIAN CLOSER TO log_δ(2)\nΔ = {abs(np.median(p_valid) - LOG_D_2):.4f}"
else:
    verdict = f"MEDIAN CLOSER TO p=0.5\nΔ = {abs(np.median(p_valid) - 0.5):.4f}"

summary = (
    f"175-GALAXY SWEEP\n"
    f"════════════════════════\n\n"
    f"Galaxies analyzed: {len(results)}\n"
    f"Valid p fits: {n_valid}\n\n"
    f"MEDIAN p_best: {np.median(p_valid):.4f}\n"
    f"  vs 0.5000 (RAR)\n"
    f"  vs {LOG_D_2:.4f} (log_δ(2))\n\n"
    f"HEAD TO HEAD:\n"
    f"  p=0.5 wins: {n_std_wins}\n"
    f"  log_δ(2) wins: {n_der_wins}\n\n"
    f"GLOBAL χ²:\n"
    f"  p=0.5:    {global_chi2_std:.3f}\n"
    f"  log_δ(2): {global_chi2_der:.3f}\n"
    f"  Ratio:    {global_chi2_der/global_chi2_std:.4f}\n\n"
    f"VERDICT:\n"
    f"{verdict}"
)
ax.text(0.05, 0.95, summary, transform=ax.transAxes,
        fontsize=10, verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

fig.savefig(os.path.join(BASE, 'fig61_sparc_sweep.png'),
            dpi=150, bbox_inches='tight')
print(f"\n  Figure saved: fig61_sparc_sweep.png")

# ============================================================
# FINAL REPORT
# ============================================================
print("\n" + "=" * 70)
print("FINAL REPORT")
print("=" * 70, flush=True)

wc_median = np.median(wc_p) if len(wc_p) > 0 else 0

print(f"""
  THE QUESTION:
    Does the histogram of best-fit p peak near 0.45 or 0.50?

  THE ANSWER:
    All galaxies:
      Median p_best = {np.median(p_valid):.4f}
      Mean p_best   = {np.mean(p_valid):.4f}

    Well-constrained (N > 20):
      Median p_best = {wc_median:.4f}
      n = {len(wc_p)} galaxies

    Key values:
      log_δ(2) = {LOG_D_2:.4f}
      Standard = 0.5000
      Median   = {np.median(p_valid):.4f}

    Median is {abs(np.median(p_valid) - LOG_D_2):.4f} from log_δ(2)
    Median is {abs(np.median(p_valid) - 0.5):.4f} from 0.5

  HEAD TO HEAD:
    p=0.5 wins:      {n_std_wins} / {len(results)} galaxies
    log_δ(2) wins:   {n_der_wins} / {len(results)} galaxies

  GLOBAL FIT:
    χ²(p=0.5):      {global_chi2_std:.4f}
    χ²(log_δ(2)):   {global_chi2_der:.4f}
    Improvement:     {abs(1 - global_chi2_der/global_chi2_std)*100:.2f}%
    Winner:          {'log_δ(2)' if global_chi2_der < global_chi2_std else 'p=0.5'}
""")

print("=" * 70)
print("Script 61 complete.")
print("=" * 70)
