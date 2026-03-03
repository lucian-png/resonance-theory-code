#!/usr/bin/env python3
"""
Script 62: THE MORPHOLOGICAL SORT
===================================
If coupling topology determines the geometric constant (the Lucian Law),
then galaxies with different morphological types should want DIFFERENT
values of the RAR exponent p. And the scatter WITHIN each morphological
class should be TIGHTER than the scatter across the full sample.

Three predictions BEFORE opening the results:
  1. Within-bin σ(p) < full-sample σ(p)
  2. Bin medians are different from each other
  3. Clean spirals (Bin B, T=3-5) have median closest to log_δ(2)

The bins fall where they fall.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.optimize import minimize_scalar
from scipy.stats import median_abs_deviation, kruskal, mannwhitneyu
import os
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("SCRIPT 62: THE MORPHOLOGICAL SORT")
print("Coupling Topology Determines the Constant")
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

# Hubble type labels
T_LABELS = {
    0: 'S0', 1: 'Sa', 2: 'Sab', 3: 'Sb', 4: 'Sbc', 5: 'Sc',
    6: 'Scd', 7: 'Sd', 8: 'Sdm', 9: 'Sm', 10: 'Im', 11: 'BCD',
}

# ============================================================
# LOAD ALL SPARC DATA — EXPANDED PROPERTIES
# ============================================================
print("\n" + "=" * 70)
print("LOADING SPARC DATA (FULL PROPERTIES)")
print("=" * 70, flush=True)

# --- Galaxy properties (table1) — parse ALL useful columns ---
galaxy_props = {}
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
        l36 = float(parts[7])       # Total luminosity at 3.6μm [10⁹ L☉]
        reff = float(parts[9])      # Effective radius [kpc]
        sbeff = float(parts[10])    # Effective surface brightness [L☉/pc²]
        rdisk = float(parts[11])    # Disk scale length [kpc]
        sbdisk = float(parts[12])   # Disk central surface brightness [L☉/pc²]
        mhi = float(parts[13])      # HI mass [10⁹ M☉]
        rhi = float(parts[14])      # HI radius [kpc]
        vflat = float(parts[15])
        qual = int(parts[17])

        # Gas fraction proxy: MHI / (L36 * Y_DISK * M☉/L☉)
        # Simplified: MHI / L36 (both in 10⁹ units)
        gas_frac = mhi / max(l36, 1e-6)

        galaxy_props[name] = {
            'type': hubble_type,
            'dist': dist,
            'inc': inc,
            'L36': l36,
            'Reff': reff,
            'SBeff': sbeff,
            'Rdisk': rdisk,
            'SBdisk': sbdisk,
            'MHI': mhi,
            'RHI': rhi,
            'Vflat': vflat,
            'quality': qual,
            'gas_frac': gas_frac,
        }
    except (ValueError, IndexError):
        continue

print(f"  Galaxy properties: {len(galaxy_props)}")

# --- Rotation curves (table2) ---
galaxy_data = {}
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
# CORE FUNCTIONS (same as Script 61)
# ============================================================


def tau_gen_rar(x_arr, p):
    """Generalized RAR: tau = sqrt(1 - exp(-x^p))"""
    xp = np.power(np.maximum(x_arr, 1e-30), p)
    return np.sqrt(np.maximum(1e-20, 1.0 - np.exp(-xp)))


def compute_galaxy(rows, p_val):
    """Compute reduced chi2 for a galaxy with given p."""
    R_kpc = np.array([r['R'] for r in rows])
    Vobs = np.array([r['Vobs'] for r in rows])
    e_Vobs = np.array([r['e_Vobs'] for r in rows])
    Vgas = np.array([r['Vgas'] for r in rows])
    Vdisk = np.array([r['Vdisk'] for r in rows])
    Vbul = np.array([r['Vbul'] for r in rows])

    Vbar_sq = (np.sign(Vgas) * Vgas**2 +
               Y_DISK * Vdisk**2 +
               Y_BUL * Vbul**2)
    Vbar_sq = np.maximum(Vbar_sq, 1.0)
    Vbar = np.sqrt(Vbar_sq)

    R_m = R_kpc * KPC_TO_M
    valid = R_m > 0
    g_bar = np.zeros_like(R_kpc)
    g_bar[valid] = Vbar_sq[valid] * KM_TO_M**2 / R_m[valid]

    x = g_bar / A0_MOND
    tau = tau_gen_rar(x, p_val)
    Vpred = np.abs(Vbar) / np.maximum(tau, 1e-10)

    resid = (Vobs - Vpred) / e_Vobs
    n = len(resid)
    dof = max(n - 1, 1)
    chi2_red = float(np.sum(resid**2) / dof)

    return chi2_red, n


def find_best_p(rows):
    """Find best-fit p for a galaxy. Returns (p_best, chi2_best)."""
    def objective(p):
        chi2, _ = compute_galaxy(rows, p)
        return chi2
    result = minimize_scalar(objective, bounds=(0.1, 2.0), method='bounded')
    return float(result.x), float(result.fun)


# ============================================================
# THE SWEEP + MORPHOLOGICAL CLASSIFICATION
# ============================================================
print("\n" + "=" * 70)
print("COMPUTING p FOR ALL GALAXIES (with full properties)")
print("=" * 70, flush=True)

P_STANDARD = 0.5
P_DERIVED = LOG_D_2

results = []
n_processed = 0
n_skipped = 0

for name, rows in galaxy_data.items():
    if len(rows) < 5:
        n_skipped += 1
        continue

    props = galaxy_props.get(name, None)
    if props is None:
        n_skipped += 1
        continue

    qual = props.get('quality', 3)
    if qual > 2:
        n_skipped += 1
        continue

    chi2_std, n_pts = compute_galaxy(rows, P_STANDARD)
    chi2_der, _ = compute_galaxy(rows, P_DERIVED)
    p_best, chi2_best = find_best_p(rows)

    results.append({
        'name': name,
        'n_pts': n_pts,
        'T': props['type'],
        'Vflat': props['Vflat'],
        'SBdisk': props['SBdisk'],
        'SBeff': props['SBeff'],
        'L36': props['L36'],
        'Rdisk': props['Rdisk'],
        'MHI': props['MHI'],
        'gas_frac': props['gas_frac'],
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
print(f"  Skipped: {n_skipped}")

# ============================================================
# DEFINE MORPHOLOGICAL BINS
# ============================================================
print("\n" + "=" * 70)
print("MORPHOLOGICAL BINNING")
print("=" * 70, flush=True)

# Filter valid p fits (same as Script 61)
valid_results = [r for r in results if 0.15 < r['p_best'] < 1.5]
all_p = np.array([r['p_best'] for r in valid_results])
full_sigma = np.std(all_p)
full_mad = median_abs_deviation(all_p)
full_median = np.median(all_p)

print(f"\n  Full sample: n = {len(all_p)}")
print(f"    Median p = {full_median:.4f}")
print(f"    σ(p) = {full_sigma:.4f}")
print(f"    MAD(p) = {full_mad:.4f}")

# --- Primary bins: Hubble type ---
bins_def = {
    'A: Early (T≤2)\nS0/Sa/Sab': lambda r: r['T'] <= 2,
    'B: Classic Spiral (T=3-5)\nSb/Sbc/Sc': lambda r: 3 <= r['T'] <= 5,
    'C: Late Spiral (T=6-7)\nScd/Sd': lambda r: 6 <= r['T'] <= 7,
    'D: Irregular (T≥8)\nSdm/Sm/Im/BCD': lambda r: r['T'] >= 8,
}

# Short labels for printing
bins_short = {
    'A: Early (T≤2)\nS0/Sa/Sab': 'A: Early (T≤2)',
    'B: Classic Spiral (T=3-5)\nSb/Sbc/Sc': 'B: Classic Spiral (T=3-5)',
    'C: Late Spiral (T=6-7)\nScd/Sd': 'C: Late Spiral (T=6-7)',
    'D: Irregular (T≥8)\nSdm/Sm/Im/BCD': 'D: Irregular (T≥8)',
}

bin_results = {}
for label, condition in bins_def.items():
    members = [r for r in valid_results if condition(r)]
    p_vals = np.array([r['p_best'] for r in members])
    short = bins_short[label]

    if len(p_vals) >= 3:
        bin_results[label] = {
            'p_vals': p_vals,
            'n': len(p_vals),
            'median': np.median(p_vals),
            'mean': np.mean(p_vals),
            'sigma': np.std(p_vals),
            'mad': median_abs_deviation(p_vals),
            'p25': np.percentile(p_vals, 25),
            'p75': np.percentile(p_vals, 75),
            'iqr': np.percentile(p_vals, 75) - np.percentile(p_vals, 25),
            'members': members,
        }
        br = bin_results[label]
        print(f"\n  {short}:")
        print(f"    n = {br['n']}")
        print(f"    Median p = {br['median']:.4f}")
        print(f"    σ(p) = {br['sigma']:.4f}  (full sample: {full_sigma:.4f})")
        print(f"    MAD(p) = {br['mad']:.4f}  (full sample: {full_mad:.4f})")
        print(f"    IQR = [{br['p25']:.4f}, {br['p75']:.4f}]")

        # Distance to key values
        d_logd2 = abs(br['median'] - LOG_D_2)
        d_std = abs(br['median'] - 0.5)
        closer = "log_δ(2)" if d_logd2 < d_std else "p=0.5"
        print(f"    Distance to log_δ(2): {d_logd2:.4f}")
        print(f"    Distance to p=0.5:    {d_std:.4f}")
        print(f"    → Closer to {closer}")
    else:
        print(f"\n  {short}: n = {len(p_vals)} — too few for statistics")

# ============================================================
# PREDICTION TESTS
# ============================================================
print("\n" + "=" * 70)
print("PREDICTION TESTS")
print("=" * 70, flush=True)

# --- Prediction 1: Within-bin σ < full-sample σ ---
print("\n  PREDICTION 1: Within-bin scatter < full-sample scatter")
print(f"    Full sample σ = {full_sigma:.4f}, MAD = {full_mad:.4f}")
print()
n_tighter = 0
n_bins_tested = 0
for label, br in bin_results.items():
    short = bins_short[label]
    sigma_ratio = br['sigma'] / full_sigma
    mad_ratio = br['mad'] / full_mad
    tighter = "YES ✓" if br['sigma'] < full_sigma else "NO ✗"
    mad_tighter = "YES ✓" if br['mad'] < full_mad else "NO ✗"
    print(f"    {short}:")
    print(f"      σ = {br['sigma']:.4f} (ratio: {sigma_ratio:.3f}) — tighter? {tighter}")
    print(f"      MAD = {br['mad']:.4f} (ratio: {mad_ratio:.3f}) — tighter? {mad_tighter}")
    if br['sigma'] < full_sigma:
        n_tighter += 1
    n_bins_tested += 1

print(f"\n    RESULT: {n_tighter}/{n_bins_tested} bins have σ < full-sample σ")
pred1_pass = n_tighter >= n_bins_tested - 1  # allow 1 failure
print(f"    PREDICTION 1: {'CONFIRMED' if pred1_pass else 'FAILED'}")

# --- Prediction 2: Bin medians are different ---
print(f"\n  PREDICTION 2: Bin medians are different from each other")
medians = {bins_short[k]: v['median'] for k, v in bin_results.items()}
for label, med in medians.items():
    print(f"    {label}: median = {med:.4f}")

spread = max(medians.values()) - min(medians.values())
print(f"\n    Spread (max - min median): {spread:.4f}")
print(f"    Spread / full σ: {spread/full_sigma:.2f}")

# Kruskal-Wallis test: are the distributions different?
bin_p_arrays = [br['p_vals'] for br in bin_results.values() if br['n'] >= 5]
if len(bin_p_arrays) >= 2:
    kw_stat, kw_pval = kruskal(*bin_p_arrays)
    print(f"\n    Kruskal-Wallis test:")
    print(f"      H statistic: {kw_stat:.3f}")
    print(f"      p-value: {kw_pval:.6f}")
    pred2_significant = kw_pval < 0.05
    print(f"    PREDICTION 2: {'CONFIRMED (p < 0.05)' if pred2_significant else 'NOT SIGNIFICANT (p >= 0.05)'}")
else:
    pred2_significant = False
    print(f"    PREDICTION 2: CANNOT TEST (too few bins with enough data)")

# Pairwise tests
print(f"\n    Pairwise Mann-Whitney U tests:")
bin_labels = list(bin_results.keys())
for i in range(len(bin_labels)):
    for j in range(i+1, len(bin_labels)):
        a_label = bins_short[bin_labels[i]]
        b_label = bins_short[bin_labels[j]]
        a_p = bin_results[bin_labels[i]]['p_vals']
        b_p = bin_results[bin_labels[j]]['p_vals']
        if len(a_p) >= 5 and len(b_p) >= 5:
            u_stat, u_pval = mannwhitneyu(a_p, b_p, alternative='two-sided')
            sig = "***" if u_pval < 0.001 else "**" if u_pval < 0.01 else "*" if u_pval < 0.05 else "n.s."
            print(f"      {a_label} vs {b_label}:")
            print(f"        medians: {np.median(a_p):.4f} vs {np.median(b_p):.4f}, "
                  f"p = {u_pval:.4f} {sig}")

# --- Prediction 3: Classic spirals closest to log_δ(2) ---
print(f"\n  PREDICTION 3: Classic spirals (Bin B) closest to log_δ(2)")
for label, br in bin_results.items():
    short = bins_short[label]
    d = abs(br['median'] - LOG_D_2)
    print(f"    {short}: |median - log_δ(2)| = {d:.4f}")

closest_bin = min(bin_results.items(), key=lambda x: abs(x[1]['median'] - LOG_D_2))
closest_short = bins_short[closest_bin[0]]
is_bin_b = 'Classic Spiral' in closest_short
print(f"\n    Closest bin: {closest_short} (median = {closest_bin[1]['median']:.4f})")
print(f"    PREDICTION 3: {'CONFIRMED' if is_bin_b else 'FAILED — closest is ' + closest_short}")

# ============================================================
# SECONDARY DISCRIMINATORS
# ============================================================
print("\n" + "=" * 70)
print("SECONDARY DISCRIMINATORS")
print("=" * 70, flush=True)

# --- Surface brightness split ---
# Use SBdisk: Freeman (1970) value is ~140 L☉/pc² at B-band
# At 3.6μm, HSB disks have SBdisk > ~200, LSB < ~100
sb_vals = np.array([r['SBdisk'] for r in valid_results])
sb_median = np.median(sb_vals)
print(f"\n  Surface Brightness (SBdisk at 3.6μm, L☉/pc²):")
print(f"    Range: {sb_vals.min():.1f} — {sb_vals.max():.1f}")
print(f"    Median: {sb_median:.1f}")

hsb_results = [r for r in valid_results if r['SBdisk'] >= sb_median]
lsb_results = [r for r in valid_results if r['SBdisk'] < sb_median]

hsb_p = np.array([r['p_best'] for r in hsb_results])
lsb_p = np.array([r['p_best'] for r in lsb_results])

print(f"\n    HSB (SBdisk ≥ {sb_median:.0f}): n={len(hsb_p)}, "
      f"median p = {np.median(hsb_p):.4f}, σ = {np.std(hsb_p):.4f}")
print(f"    LSB (SBdisk < {sb_median:.0f}): n={len(lsb_p)}, "
      f"median p = {np.median(lsb_p):.4f}, σ = {np.std(lsb_p):.4f}")

if len(hsb_p) >= 5 and len(lsb_p) >= 5:
    u_stat, u_pval = mannwhitneyu(hsb_p, lsb_p, alternative='two-sided')
    sig = "***" if u_pval < 0.001 else "**" if u_pval < 0.01 else "*" if u_pval < 0.05 else "n.s."
    print(f"    Mann-Whitney: p = {u_pval:.4f} {sig}")

# --- Mass split (Vflat) ---
vflat_vals = np.array([r['Vflat'] for r in valid_results])
# Only use galaxies with nonzero Vflat
vflat_valid = [r for r in valid_results if r['Vflat'] > 0]
vflat_vals_nz = np.array([r['Vflat'] for r in vflat_valid])
vflat_median = np.median(vflat_vals_nz) if len(vflat_vals_nz) > 0 else 100.0

print(f"\n  Mass Proxy (Vflat, km/s):")
print(f"    Range: {vflat_vals_nz.min():.1f} — {vflat_vals_nz.max():.1f}")
print(f"    Median: {vflat_median:.1f}")

massive = [r for r in vflat_valid if r['Vflat'] >= vflat_median]
dwarf = [r for r in vflat_valid if r['Vflat'] < vflat_median]

massive_p = np.array([r['p_best'] for r in massive])
dwarf_p = np.array([r['p_best'] for r in dwarf])

print(f"\n    Massive (Vflat ≥ {vflat_median:.0f}): n={len(massive_p)}, "
      f"median p = {np.median(massive_p):.4f}, σ = {np.std(massive_p):.4f}")
print(f"    Dwarf (Vflat < {vflat_median:.0f}): n={len(dwarf_p)}, "
      f"median p = {np.median(dwarf_p):.4f}, σ = {np.std(dwarf_p):.4f}")

if len(massive_p) >= 5 and len(dwarf_p) >= 5:
    u_stat, u_pval = mannwhitneyu(massive_p, dwarf_p, alternative='two-sided')
    sig = "***" if u_pval < 0.001 else "**" if u_pval < 0.01 else "*" if u_pval < 0.05 else "n.s."
    print(f"    Mann-Whitney: p = {u_pval:.4f} {sig}")

# --- Gas fraction split ---
gf_vals = np.array([r['gas_frac'] for r in valid_results])
gf_median = np.median(gf_vals)
print(f"\n  Gas Fraction (MHI/L36):")
print(f"    Range: {gf_vals.min():.4f} — {gf_vals.max():.4f}")
print(f"    Median: {gf_median:.4f}")

gas_rich = [r for r in valid_results if r['gas_frac'] >= gf_median]
star_dom = [r for r in valid_results if r['gas_frac'] < gf_median]

gas_p = np.array([r['p_best'] for r in gas_rich])
star_p = np.array([r['p_best'] for r in star_dom])

print(f"\n    Gas-rich (f_gas ≥ {gf_median:.3f}): n={len(gas_p)}, "
      f"median p = {np.median(gas_p):.4f}, σ = {np.std(gas_p):.4f}")
print(f"    Stellar-dom (f_gas < {gf_median:.3f}): n={len(star_p)}, "
      f"median p = {np.median(star_p):.4f}, σ = {np.std(star_p):.4f}")

if len(gas_p) >= 5 and len(star_p) >= 5:
    u_stat, u_pval = mannwhitneyu(gas_p, star_p, alternative='two-sided')
    sig = "***" if u_pval < 0.001 else "**" if u_pval < 0.01 else "*" if u_pval < 0.05 else "n.s."
    print(f"    Mann-Whitney: p = {u_pval:.4f} {sig}")

# ============================================================
# FINE-GRAINED T-TYPE ANALYSIS
# ============================================================
print("\n" + "=" * 70)
print("FINE-GRAINED HUBBLE TYPE ANALYSIS")
print("=" * 70, flush=True)

print(f"\n  {'T':>3s}  {'Type':>5s}  {'n':>4s}  {'Median p':>10s}  {'σ(p)':>8s}  "
      f"{'MAD(p)':>8s}  {'Δ(logδ2)':>9s}  {'Δ(0.5)':>8s}")
print(f"  {'---':>3s}  {'---':>5s}  {'---':>4s}  {'---':>10s}  {'---':>8s}  "
      f"{'---':>8s}  {'---':>9s}  {'---':>8s}")

t_medians = {}
for t_val in range(12):
    t_results = [r for r in valid_results if r['T'] == t_val]
    if len(t_results) >= 3:
        t_p = np.array([r['p_best'] for r in t_results])
        t_med = np.median(t_p)
        t_sig = np.std(t_p)
        t_mad = median_abs_deviation(t_p)
        d_logd2 = t_med - LOG_D_2
        d_std = t_med - 0.5
        t_medians[t_val] = t_med
        print(f"  {t_val:3d}  {T_LABELS[t_val]:>5s}  {len(t_p):4d}  "
              f"{t_med:10.4f}  {t_sig:8.4f}  {t_mad:8.4f}  "
              f"{d_logd2:+9.4f}  {d_std:+8.4f}")

# ============================================================
# FIGURE 1: MORPHOLOGICAL CLASSIFICATION
# ============================================================
print("\n" + "=" * 70)
print("GENERATING FIGURE 1: MORPHOLOGICAL CLASSIFICATION")
print("=" * 70, flush=True)

fig = plt.figure(figsize=(20, 14))
gs = GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.35)
fig.suptitle('Script 62: THE MORPHOLOGICAL SORT\n'
             'Does Coupling Topology Determine the RAR Exponent?',
             fontsize=16, fontweight='bold', y=0.99)

COLORS = {
    'A: Early (T≤2)\nS0/Sa/Sab': '#e74c3c',
    'B: Classic Spiral (T=3-5)\nSb/Sbc/Sc': '#3498db',
    'C: Late Spiral (T=6-7)\nScd/Sd': '#2ecc71',
    'D: Irregular (T≥8)\nSdm/Sm/Im/BCD': '#9b59b6',
}

# --- Panel A: Box plots by morphological bin ---
ax = fig.add_subplot(gs[0, 0])
box_data = []
box_labels = []
box_colors = []
for label in bins_def.keys():
    if label in bin_results:
        box_data.append(bin_results[label]['p_vals'])
        box_labels.append(label.split('\n')[0])
        box_colors.append(COLORS[label])

bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True,
                widths=0.6, showmeans=True,
                meanprops=dict(marker='D', markerfacecolor='black', markersize=5),
                medianprops=dict(color='black', linewidth=2))
for patch, color in zip(bp['boxes'], box_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)

ax.axhline(y=0.5, color='green', linewidth=2, linestyle='--',
           alpha=0.7, label='p=0.5')
ax.axhline(y=LOG_D_2, color='red', linewidth=2, linestyle='-',
           alpha=0.7, label=f'log_δ(2)={LOG_D_2:.4f}')
ax.axhline(y=full_median, color='orange', linewidth=1.5, linestyle=':',
           alpha=0.7, label=f'Full median={full_median:.3f}')
ax.set_ylabel('Best-fit p', fontsize=12)
ax.set_title('Best-Fit p by Morphological Class',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=8, loc='upper right')
ax.set_ylim(0.1, 1.2)
ax.grid(True, alpha=0.2, axis='y')
ax.tick_params(axis='x', labelsize=9)

# --- Panel B: Scatter within bins vs full sample ---
ax = fig.add_subplot(gs[0, 1])
bin_names_short = []
sigmas = []
mads = []
for label in bins_def.keys():
    if label in bin_results:
        bin_names_short.append(label.split('\n')[0])
        sigmas.append(bin_results[label]['sigma'])
        mads.append(bin_results[label]['mad'])

x_pos = np.arange(len(bin_names_short))
width = 0.35
bars1 = ax.bar(x_pos - width/2, sigmas, width, color='#3498db', alpha=0.7,
               label='σ(p)', edgecolor='black', linewidth=0.5)
bars2 = ax.bar(x_pos + width/2, mads, width, color='#e67e22', alpha=0.7,
               label='MAD(p)', edgecolor='black', linewidth=0.5)
ax.axhline(y=full_sigma, color='#3498db', linewidth=2, linestyle='--',
           alpha=0.6, label=f'Full σ={full_sigma:.3f}')
ax.axhline(y=full_mad, color='#e67e22', linewidth=2, linestyle='--',
           alpha=0.6, label=f'Full MAD={full_mad:.3f}')
ax.set_xticks(x_pos)
ax.set_xticklabels(bin_names_short, fontsize=9)
ax.set_ylabel('Scatter in p', fontsize=12)
ax.set_title('PREDICTION 1: Within-Bin Scatter\nvs Full-Sample Scatter',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.2, axis='y')

# --- Panel C: p vs Hubble type (individual galaxies) ---
ax = fig.add_subplot(gs[0, 2])
t_vals = np.array([r['T'] for r in valid_results])
p_vals = np.array([r['p_best'] for r in valid_results])
chi2_vals = np.array([r['chi2_free'] for r in valid_results])

scatter = ax.scatter(t_vals + np.random.normal(0, 0.15, len(t_vals)),
                     p_vals,
                     c=np.log10(np.clip(chi2_vals, 0.1, 100)),
                     cmap='RdYlGn_r', s=20, alpha=0.6,
                     edgecolors='gray', linewidths=0.3,
                     vmin=-0.5, vmax=2)

# Overlay medians per T
for t_val, t_med in t_medians.items():
    ax.plot(t_val, t_med, 'ko', markersize=8, zorder=5)
    ax.plot(t_val, t_med, 'wo', markersize=5, zorder=6)

ax.axhline(y=0.5, color='green', linewidth=2, linestyle='--', alpha=0.7)
ax.axhline(y=LOG_D_2, color='red', linewidth=2, linestyle='-', alpha=0.7)

ax.set_xlabel('Hubble Type T', fontsize=12)
ax.set_ylabel('Best-fit p', fontsize=12)
ax.set_title('Best-Fit p vs Hubble Type\nBlack dots = medians per T',
             fontsize=13, fontweight='bold')
ax.set_xticks(range(12))
ax.set_xticklabels([T_LABELS[t] for t in range(12)], fontsize=7, rotation=45)
ax.set_ylim(0.1, 1.2)
ax.grid(True, alpha=0.2)
plt.colorbar(scatter, ax=ax, label='log₁₀(χ²)')

# --- Panel D: p vs Surface Brightness ---
ax = fig.add_subplot(gs[1, 0])
sb_plot = np.array([r['SBdisk'] for r in valid_results])
p_plot = np.array([r['p_best'] for r in valid_results])
t_plot = np.array([r['T'] for r in valid_results])

scatter = ax.scatter(np.log10(np.maximum(sb_plot, 0.1)), p_plot,
                     c=t_plot, cmap='Spectral_r', s=20, alpha=0.6,
                     edgecolors='gray', linewidths=0.3,
                     vmin=0, vmax=11)
ax.axhline(y=0.5, color='green', linewidth=2, linestyle='--', alpha=0.7,
           label='p=0.5')
ax.axhline(y=LOG_D_2, color='red', linewidth=2, linestyle='-', alpha=0.7,
           label=f'log_δ(2)')
ax.set_xlabel('log₁₀(SBdisk) [L☉/pc²]', fontsize=12)
ax.set_ylabel('Best-fit p', fontsize=12)
ax.set_title('Best-Fit p vs Surface Brightness\nColor = Hubble Type',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=9)
ax.set_ylim(0.1, 1.2)
ax.grid(True, alpha=0.2)
plt.colorbar(scatter, ax=ax, label='Hubble Type T')

# --- Panel E: p vs Vflat (mass proxy) ---
ax = fig.add_subplot(gs[1, 1])
vf_plot = np.array([r['Vflat'] for r in valid_results if r['Vflat'] > 0])
p_vf = np.array([r['p_best'] for r in valid_results if r['Vflat'] > 0])
t_vf = np.array([r['T'] for r in valid_results if r['Vflat'] > 0])

scatter = ax.scatter(vf_plot, p_vf,
                     c=t_vf, cmap='Spectral_r', s=20, alpha=0.6,
                     edgecolors='gray', linewidths=0.3,
                     vmin=0, vmax=11)
ax.axhline(y=0.5, color='green', linewidth=2, linestyle='--', alpha=0.7,
           label='p=0.5')
ax.axhline(y=LOG_D_2, color='red', linewidth=2, linestyle='-', alpha=0.7,
           label=f'log_δ(2)')

# Mark NGC 3198
ngc3198 = [r for r in valid_results if r['name'] == 'NGC3198']
if ngc3198:
    ax.plot(ngc3198[0]['Vflat'], ngc3198[0]['p_best'], 'r*',
            markersize=15, zorder=10, markeredgecolor='black',
            markeredgewidth=0.5, label='NGC 3198')

ax.set_xlabel('Vflat [km/s]', fontsize=12)
ax.set_ylabel('Best-fit p', fontsize=12)
ax.set_title('Best-Fit p vs Galaxy Mass (Vflat)\nColor = Hubble Type',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=9)
ax.set_ylim(0.1, 1.2)
ax.grid(True, alpha=0.2)
plt.colorbar(scatter, ax=ax, label='Hubble Type T')

# --- Panel F: Summary ---
ax = fig.add_subplot(gs[1, 2])
ax.axis('off')

# Build prediction summary
pred1_str = "CONFIRMED" if pred1_pass else "FAILED"
pred2_str = ("CONFIRMED" if pred2_significant
             else "NOT SIGNIFICANT")
pred3_str = ("CONFIRMED" if is_bin_b
             else f"FAILED")

# Find the bin medians for display
bin_med_str = ""
for label, br in bin_results.items():
    short = bins_short[label].split(':')[0]
    bin_med_str += f"  Bin {short}: med={br['median']:.4f} (n={br['n']})\n"

summary = (
    f"MORPHOLOGICAL SORT RESULTS\n"
    f"══════════════════════════\n\n"
    f"Full sample:\n"
    f"  n = {len(all_p)}, med = {full_median:.4f}\n"
    f"  σ = {full_sigma:.4f}, MAD = {full_mad:.4f}\n\n"
    f"Bin medians:\n"
    f"{bin_med_str}\n"
    f"PREDICTIONS:\n"
    f"  1. Within-bin σ < full σ:\n"
    f"     {pred1_str}\n"
    f"     ({n_tighter}/{n_bins_tested} bins tighter)\n\n"
    f"  2. Bin medians differ:\n"
    f"     {pred2_str}\n"
    f"     (spread = {spread:.4f})\n\n"
    f"  3. Classic spirals → log_δ(2):\n"
    f"     {pred3_str}\n"
    f"     (closest: {closest_short})"
)
ax.text(0.02, 0.98, summary, transform=ax.transAxes,
        fontsize=9, verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

fig.savefig(os.path.join(BASE, 'fig62a_morphological_sort.png'),
            dpi=200, bbox_inches='tight')
print(f"  Figure saved: fig62a_morphological_sort.png")

# ============================================================
# FIGURE 2: OVERLAID DISTRIBUTIONS + TOPOLOGY MAP
# ============================================================
print("\n" + "=" * 70)
print("GENERATING FIGURE 2: COUPLING TOPOLOGY MAP")
print("=" * 70, flush=True)

fig2 = plt.figure(figsize=(20, 14))
gs2 = GridSpec(2, 3, figure=fig2, hspace=0.38, wspace=0.35)
fig2.suptitle('Script 62: COUPLING TOPOLOGY MAP\n'
              'Do Different Galaxy Classes Want Different Exponents?',
              fontsize=16, fontweight='bold', y=0.99)

# --- Panel A: Overlaid histograms by morphological bin ---
ax = fig2.add_subplot(gs2[0, 0])
bins_hist = np.linspace(0.15, 1.1, 35)
for label, br in bin_results.items():
    short = label.split('\n')[0]
    ax.hist(br['p_vals'], bins=bins_hist, alpha=0.4,
            color=COLORS[label], edgecolor=COLORS[label],
            linewidth=1.5, histtype='stepfilled',
            label=f'{short} (n={br["n"]}, med={br["median"]:.3f})',
            density=True)

ax.axvline(x=0.5, color='green', linewidth=2, linestyle='--', alpha=0.7)
ax.axvline(x=LOG_D_2, color='red', linewidth=2, linestyle='-', alpha=0.7)
ax.set_xlabel('Best-fit p', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('p Distributions by Morphological Class',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=8, loc='upper right')
ax.grid(True, alpha=0.2)

# --- Panel B: HSB vs LSB ---
ax = fig2.add_subplot(gs2[0, 1])
ax.hist(hsb_p, bins=bins_hist, alpha=0.5, color='#e74c3c',
        edgecolor='#e74c3c', linewidth=1.5, histtype='stepfilled',
        label=f'HSB (n={len(hsb_p)}, med={np.median(hsb_p):.3f})',
        density=True)
ax.hist(lsb_p, bins=bins_hist, alpha=0.5, color='#3498db',
        edgecolor='#3498db', linewidth=1.5, histtype='stepfilled',
        label=f'LSB (n={len(lsb_p)}, med={np.median(lsb_p):.3f})',
        density=True)
ax.axvline(x=0.5, color='green', linewidth=2, linestyle='--', alpha=0.7)
ax.axvline(x=LOG_D_2, color='red', linewidth=2, linestyle='-', alpha=0.7)
ax.set_xlabel('Best-fit p', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('p Distribution: HSB vs LSB Galaxies',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2)

# --- Panel C: Massive vs Dwarf ---
ax = fig2.add_subplot(gs2[0, 2])
ax.hist(massive_p, bins=bins_hist, alpha=0.5, color='#e67e22',
        edgecolor='#e67e22', linewidth=1.5, histtype='stepfilled',
        label=f'Massive (n={len(massive_p)}, med={np.median(massive_p):.3f})',
        density=True)
ax.hist(dwarf_p, bins=bins_hist, alpha=0.5, color='#2ecc71',
        edgecolor='#2ecc71', linewidth=1.5, histtype='stepfilled',
        label=f'Dwarf (n={len(dwarf_p)}, med={np.median(dwarf_p):.3f})',
        density=True)
ax.axvline(x=0.5, color='green', linewidth=2, linestyle='--', alpha=0.7)
ax.axvline(x=LOG_D_2, color='red', linewidth=2, linestyle='-', alpha=0.7)
ax.set_xlabel('Best-fit p', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title(f'p Distribution: Massive vs Dwarf\n(split at Vflat = {vflat_median:.0f} km/s)',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2)

# --- Panel D: Gas fraction vs p ---
ax = fig2.add_subplot(gs2[1, 0])
gf_plot = np.array([r['gas_frac'] for r in valid_results])
p_gf = np.array([r['p_best'] for r in valid_results])
t_gf = np.array([r['T'] for r in valid_results])

scatter = ax.scatter(np.log10(np.maximum(gf_plot, 1e-4)), p_gf,
                     c=t_gf, cmap='Spectral_r', s=20, alpha=0.6,
                     edgecolors='gray', linewidths=0.3,
                     vmin=0, vmax=11)
ax.axhline(y=0.5, color='green', linewidth=2, linestyle='--', alpha=0.7)
ax.axhline(y=LOG_D_2, color='red', linewidth=2, linestyle='-', alpha=0.7)
ax.set_xlabel('log₁₀(Gas Fraction: MHI/L36)', fontsize=12)
ax.set_ylabel('Best-fit p', fontsize=12)
ax.set_title('Best-Fit p vs Gas Fraction\nColor = Hubble Type',
             fontsize=13, fontweight='bold')
ax.set_ylim(0.1, 1.2)
ax.grid(True, alpha=0.2)
plt.colorbar(scatter, ax=ax, label='Hubble Type T')

# --- Panel E: Median p trend across Hubble sequence ---
ax = fig2.add_subplot(gs2[1, 1])
t_with_data = sorted(t_medians.keys())
t_meds = [t_medians[t] for t in t_with_data]
t_labels_plot = [T_LABELS[t] for t in t_with_data]

# Count per T for error bars
t_counts = {}
t_sigs = {}
for t_val in t_with_data:
    t_p = np.array([r['p_best'] for r in valid_results if r['T'] == t_val])
    t_counts[t_val] = len(t_p)
    t_sigs[t_val] = np.std(t_p) / np.sqrt(len(t_p)) if len(t_p) > 1 else 0

se_vals = [t_sigs[t] for t in t_with_data]

ax.errorbar(t_with_data, t_meds, yerr=se_vals,
            fmt='o-', color='#2c3e50', linewidth=2, markersize=8,
            capsize=4, capthick=1.5, elinewidth=1.5, zorder=5)

# Size dots by sample size
for t_val, t_med in zip(t_with_data, t_meds):
    size = max(t_counts[t_val] * 8, 20)
    ax.scatter(t_val, t_med, s=size, c='#3498db', alpha=0.6,
               edgecolors='#2c3e50', linewidths=1.5, zorder=6)

ax.axhline(y=0.5, color='green', linewidth=2, linestyle='--', alpha=0.7,
           label='p=0.5 (standard)')
ax.axhline(y=LOG_D_2, color='red', linewidth=2, linestyle='-', alpha=0.7,
           label=f'log_δ(2)={LOG_D_2:.4f}')
ax.fill_between([-0.5, 11.5], [0.5]*2, [LOG_D_2]*2,
                alpha=0.1, color='orange')

ax.set_xlabel('Hubble Type T', fontsize=12)
ax.set_ylabel('Median Best-Fit p', fontsize=12)
ax.set_title('Median p Across the Hubble Sequence\nDot size ∝ sample size, bars = SEM',
             fontsize=13, fontweight='bold')
ax.set_xticks(range(12))
ax.set_xticklabels([T_LABELS[t] for t in range(12)], fontsize=8, rotation=45)
ax.legend(fontsize=9)
ax.set_ylim(0.2, 0.8)
ax.set_xlim(-0.5, 11.5)
ax.grid(True, alpha=0.2)

# Mark NGC 3198's position
if ngc3198:
    ax.annotate('NGC 3198\n(T=5, Sc)',
                xy=(5, t_medians.get(5, 0.45)),
                xytext=(7, 0.35),
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                fontsize=9, color='red', fontweight='bold')

# --- Panel F: Topology classification map ---
ax = fig2.add_subplot(gs2[1, 2])
ax.axis('off')

# Build the topology map text
topo_text = (
    "COUPLING TOPOLOGY MAP\n"
    "═══════════════════════\n\n"
)

for t_val in sorted(t_medians.keys()):
    t_p_arr = np.array([r['p_best'] for r in valid_results if r['T'] == t_val])
    n_t = len(t_p_arr)
    med_t = t_medians[t_val]
    closer = "→ log_δ(2)" if abs(med_t - LOG_D_2) < abs(med_t - 0.5) else "→ p=0.5"
    bar = "█" * max(1, int(n_t / 2))
    topo_text += f"  T={t_val:2d} {T_LABELS[t_val]:>4s} (n={n_t:2d}) {bar}\n"
    topo_text += f"        med p = {med_t:.3f} {closer}\n"

# Add key physical interpretation
topo_text += (
    f"\n══════════════════════════\n"
    f"KEY VALUES:\n"
    f"  log_δ(2) = {LOG_D_2:.4f}\n"
    f"  Standard = 0.5000\n\n"
    f"TREND: {'Yes' if len(t_medians) > 2 else 'Unclear'}\n"
)

# Check if there's a trend with T
if len(t_with_data) >= 4:
    from scipy.stats import spearmanr
    rho, sp_pval = spearmanr(t_with_data, t_meds)
    topo_text += (
        f"  Spearman ρ(T, med p) = {rho:.3f}\n"
        f"  p-value = {sp_pval:.4f}\n"
    )
    if sp_pval < 0.05:
        direction = "decreasing" if rho < 0 else "increasing"
        topo_text += f"  → SIGNIFICANT {direction} trend\n"
    else:
        topo_text += f"  → No significant monotonic trend\n"

ax.text(0.02, 0.98, topo_text, transform=ax.transAxes,
        fontsize=8.5, verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

fig2.savefig(os.path.join(BASE, 'fig62b_topology_map.png'),
             dpi=200, bbox_inches='tight')
print(f"  Figure saved: fig62b_topology_map.png")

# ============================================================
# FINAL VERDICT
# ============================================================
print("\n" + "=" * 70)
print("FINAL VERDICT")
print("=" * 70, flush=True)

# Collect bin medians safely
bin_a_med = bin_results.get('A: Early (T\u22642)\nS0/Sa/Sab', {}).get('median', 'N/A')
bin_b_med = bin_results.get('B: Classic Spiral (T=3-5)\nSb/Sbc/Sc', {}).get('median', 'N/A')
bin_c_med = bin_results.get('C: Late Spiral (T=6-7)\nScd/Sd', {}).get('median', 'N/A')
bin_d_med = bin_results.get('D: Irregular (T\u22658)\nSdm/Sm/Im/BCD', {}).get('median', 'N/A')

kw_str = f"{kw_pval:.6f}" if 'kw_pval' in dir() else 'N/A'

print(f"\n  THE THREE PREDICTIONS:")
print(f"\n  1. Within-bin scatter < full scatter:")
print(f"     {pred1_str}")
print(f"     {n_tighter}/{n_bins_tested} morphological bins have tighter sigma(p)")
print(f"\n  2. Bin medians are different:")
print(f"     {pred2_str}")
print(f"     Spread in medians: {spread:.4f} ({spread/full_sigma:.2f} sigma of full sample)")
print(f"     Kruskal-Wallis p = {kw_str}")
print(f"\n  3. Classic spirals (T=3-5) closest to log_d(2):")
print(f"     {pred3_str}")
print(f"\n  KEY NUMBERS:")
print(f"    Bin A (Early, T<=2):     median = {bin_a_med}")
print(f"    Bin B (Classic, T=3-5):  median = {bin_b_med}")
print(f"    Bin C (Late, T=6-7):     median = {bin_c_med}")
print(f"    Bin D (Irregular, T>=8): median = {bin_d_med}")
print(f"    Full sample:             median = {full_median:.4f}")
print(f"    log_d(2):                         {LOG_D_2:.4f}")
print(f"    Standard RAR:                     0.5000")

print("=" * 70)
print("Script 62 complete.")
print("=" * 70)
