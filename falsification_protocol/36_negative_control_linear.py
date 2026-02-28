#!/usr/bin/env python3
"""
Test 1: NEGATIVE CONTROL — Linear Coupled System
The Lucian Law Falsification Protocol

Purpose: Verify the Lucian Method correctly identifies a LINEAR coupled system
as NON-fractal (Euclidean). If this test fails, the method is broken.

System: 3 coupled harmonic oscillators
Driving variable: Spring constant k1 swept across 12 orders of magnitude
Expected: Smooth curves, integer dimensions, no self-similarity
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# ============================================================
# SYSTEM: Three Coupled Harmonic Oscillators
# 
# Eigenfrequencies of the coupled system as functions of k1:
#   m1*ω² = k1 + k12
#   m2*ω² = k2 + k12 + k23
#   m3*ω² = k3 + k23
#
# For the coupled system, we solve the eigenvalue problem:
#   det(K - ω²M) = 0
# where K is the stiffness matrix, M is the mass matrix
# ============================================================

def coupled_linear_system(k1_values, k2=1.0, k3=1.0, k12=0.5, k23=0.5, 
                          m1=1.0, m2=1.0, m3=1.0):
    """
    Compute eigenfrequencies and mode shapes for 3 coupled harmonic oscillators
    as a function of driving variable k1.
    
    Returns: eigenfrequencies (3 x N), mode_shapes (3 x 3 x N)
    """
    N = len(k1_values)
    eigenfreqs = np.zeros((3, N))
    mode_ratios = np.zeros((3, N))  # ratio of mode amplitudes
    
    for i, k1 in enumerate(k1_values):
        # Stiffness matrix
        K = np.array([
            [k1 + k12,    -k12,        0],
            [-k12,         k2 + k12 + k23, -k23],
            [0,           -k23,         k3 + k23]
        ])
        
        # Mass matrix
        M = np.diag([m1, m2, m3])
        
        # Solve generalized eigenvalue problem K*v = ω²*M*v
        # Equivalent to M^{-1}K*v = ω²*v for diagonal M
        M_inv = np.diag([1/m1, 1/m2, 1/m3])
        A = M_inv @ K
        
        eigvals, eigvecs = np.linalg.eigh(A)
        eigenfreqs[:, i] = np.sqrt(np.abs(eigvals))
        
        # Mode shape ratios (amplitude of oscillator 2 / oscillator 1 for each mode)
        for j in range(3):
            if eigvecs[0, j] != 0:
                mode_ratios[j, i] = eigvecs[1, j] / eigvecs[0, j]
            else:
                mode_ratios[j, i] = np.inf
    
    return eigenfreqs, mode_ratios

def test_self_similarity(data, log_values, n_segments=4):
    """
    Test for self-similarity by comparing statistical distributions
    across different segments of the log-scale range.
    Returns KS statistic — high values = different distributions = NOT self-similar
    """
    segment_size = len(log_values) // n_segments
    ks_stats = []
    
    for i in range(n_segments - 1):
        seg1 = data[i*segment_size:(i+1)*segment_size]
        seg2 = data[(i+1)*segment_size:(i+2)*segment_size]
        
        # Normalize each segment
        if np.std(seg1) > 0 and np.std(seg2) > 0:
            seg1_norm = (seg1 - np.mean(seg1)) / np.std(seg1)
            seg2_norm = (seg2 - np.mean(seg2)) / np.std(seg2)
            ks, _ = stats.ks_2samp(seg1_norm, seg2_norm)
            ks_stats.append(ks)
    
    return np.mean(ks_stats) if ks_stats else 1.0

def test_power_law(x, y):
    """
    Test whether y vs x follows power law (linear in log-log).
    Returns R² of log-log fit. High R² = power law. Low R² = not power law.
    """
    mask = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    if np.sum(mask) < 10:
        return 0.0, 0.0
    
    log_x = np.log10(x[mask])
    log_y = np.log10(y[mask])
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_x, log_y)
    return r_value**2, slope

def box_counting_dimension(data, n_boxes_range=None):
    """
    Estimate fractal dimension via box counting on 1D data.
    For smooth (Euclidean) data, dimension should be ~1.0.
    """
    data_norm = (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-15)
    
    if n_boxes_range is None:
        n_boxes_range = [4, 8, 16, 32, 64, 128, 256]
    
    box_sizes = []
    box_counts = []
    
    for n_boxes in n_boxes_range:
        box_size = 1.0 / n_boxes
        # Count occupied boxes in 2D (index, value) space
        x_indices = np.arange(len(data_norm))
        x_norm = x_indices / len(data_norm)
        
        occupied = set()
        for xi, yi in zip(x_norm, data_norm):
            bx = int(xi / box_size) if xi < 1.0 else n_boxes - 1
            by = int(yi / box_size) if yi < 1.0 else n_boxes - 1
            occupied.add((bx, by))
        
        box_sizes.append(box_size)
        box_counts.append(len(occupied))
    
    # Fit log-log
    log_inv_size = np.log(1.0 / np.array(box_sizes))
    log_count = np.log(np.array(box_counts))
    
    slope, intercept, r_value, _, _ = stats.linregress(log_inv_size, log_count)
    return slope, r_value**2

# ============================================================
# MAIN ANALYSIS
# ============================================================

print("=" * 70)
print("TEST 1: NEGATIVE CONTROL — Linear Coupled Oscillators")
print("The Lucian Law Falsification Protocol")
print("=" * 70)

# Driving variable: k1 across 12 orders of magnitude
k1_values = np.logspace(-6, 6, 2000)

# Compute system response
eigenfreqs, mode_ratios = coupled_linear_system(k1_values)

print(f"\nDriving variable: k1 from {k1_values[0]:.0e} to {k1_values[-1]:.0e}")
print(f"Range: 12 orders of magnitude")
print(f"Points: {len(k1_values)}")

# ============================================================
# CRITERION 1: Self-Similarity
# ============================================================
print("\n--- CRITERION 1: Self-Similarity ---")
for mode in range(3):
    ks = test_self_similarity(eigenfreqs[mode], np.log10(k1_values))
    verdict = "NOT self-similar" if ks > 0.3 else "SELF-SIMILAR (ALERT!)"
    print(f"  Mode {mode+1} eigenfrequency: KS = {ks:.4f} → {verdict}")

# ============================================================
# CRITERION 2: Power-Law Scaling
# ============================================================
print("\n--- CRITERION 2: Power-Law Scaling ---")
for mode in range(3):
    r2, slope = test_power_law(k1_values, eigenfreqs[mode])
    # Linear systems should show ω ∝ √k → slope ≈ 0.5, R² high
    # But this is a SIMPLE power law, not fractal power law
    print(f"  Mode {mode+1}: R² = {r2:.4f}, slope = {slope:.4f}")
    if abs(slope - 0.5) < 0.1:
        print(f"    → Simple √k scaling (expected for linear oscillator)")

# ============================================================
# CRITERION 3: Fractal Dimension
# ============================================================
print("\n--- CRITERION 3: Fractal Dimension ---")
for mode in range(3):
    D, r2 = box_counting_dimension(eigenfreqs[mode])
    verdict = "EUCLIDEAN (expected)" if D < 1.1 else f"NON-INTEGER (ALERT! D={D:.3f})"
    print(f"  Mode {mode+1}: D = {D:.4f} (R² = {r2:.4f}) → {verdict}")

# ============================================================
# CRITERION 4: Phase Transitions
# ============================================================
print("\n--- CRITERION 4: Phase Transitions ---")
for mode in range(3):
    # Check for discontinuities in derivative
    dw_dk = np.diff(eigenfreqs[mode]) / np.diff(k1_values)
    d2w_dk2 = np.diff(dw_dk) / np.diff(k1_values[:-1])
    
    # Normalize
    if np.std(d2w_dk2) > 0:
        z_scores = np.abs(d2w_dk2 - np.mean(d2w_dk2)) / np.std(d2w_dk2)
        n_transitions = np.sum(z_scores > 5)
    else:
        n_transitions = 0
    
    verdict = "No transitions (expected)" if n_transitions == 0 else f"{n_transitions} TRANSITIONS (ALERT!)"
    print(f"  Mode {mode+1}: {verdict}")

# ============================================================
# CRITERION 5: Cascade Dynamics
# ============================================================
print("\n--- CRITERION 5: Cascade Dynamics ---")
# Check for period doubling or bifurcation structure
# Linear systems should show smooth, monotonic eigenfrequency evolution
for mode in range(3):
    dw = np.diff(eigenfreqs[mode])
    sign_changes = np.sum(np.diff(np.sign(dw)) != 0)
    monotonic = sign_changes == 0
    verdict = "Monotonic (expected)" if monotonic else f"{sign_changes} sign changes (ALERT!)"
    print(f"  Mode {mode+1}: {verdict}")

# ============================================================
# OVERALL VERDICT
# ============================================================
print("\n" + "=" * 70)
print("VERDICT")
print("=" * 70)
print("""
If all five criteria return EXPECTED results (Euclidean, no self-similarity,
no phase transitions, monotonic, integer dimension), the Lucian Method 
correctly discriminates linear from nonlinear systems.

The method is calibrated as a valid instrument.

If ANY criterion returns an ALERT, investigate whether the method is 
introducing artifacts or whether the linear system has unexpected structure.
""")

# ============================================================
# FIGURE
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('TEST 1: NEGATIVE CONTROL — Linear Coupled Oscillators\nThe Lucian Law Falsification Protocol', 
             fontsize=14, fontweight='bold')

# Panel 1: Eigenfrequencies vs driving variable
ax = axes[0, 0]
colors = ['#2196F3', '#FF5722', '#4CAF50']
for mode in range(3):
    ax.loglog(k1_values, eigenfreqs[mode], color=colors[mode], linewidth=2, label=f'Mode {mode+1}')
ax.set_xlabel('k₁ (driving variable)')
ax.set_ylabel('Eigenfrequency ω')
ax.set_title('Coupled Output: Eigenfrequencies')
ax.legend()
ax.grid(True, alpha=0.3)

# Panel 2: Self-similarity test — overlay normalized segments
ax = axes[0, 1]
log_k = np.log10(k1_values)
n_seg = 4
seg_size = len(log_k) // n_seg
for i in range(n_seg):
    seg = eigenfreqs[0][i*seg_size:(i+1)*seg_size]
    seg_norm = (seg - np.min(seg)) / (np.max(seg) - np.min(seg) + 1e-15)
    x_norm = np.linspace(0, 1, len(seg_norm))
    ax.plot(x_norm, seg_norm, alpha=0.7, linewidth=2, label=f'Segment {i+1}')
ax.set_xlabel('Normalized position')
ax.set_ylabel('Normalized eigenfrequency')
ax.set_title('Self-Similarity Test (Mode 1)\nSegments should NOT overlap if Euclidean')
ax.legend()
ax.grid(True, alpha=0.3)

# Panel 3: Log-log for power law test
ax = axes[0, 2]
for mode in range(3):
    ax.plot(np.log10(k1_values), np.log10(eigenfreqs[mode]), color=colors[mode], 
            linewidth=2, label=f'Mode {mode+1}')
ax.set_xlabel('log₁₀(k₁)')
ax.set_ylabel('log₁₀(ω)')
ax.set_title('Power-Law Test\nLinear in log-log = simple power law')
ax.legend()
ax.grid(True, alpha=0.3)

# Panel 4: Mode ratios
ax = axes[1, 0]
for mode in range(3):
    valid = np.isfinite(mode_ratios[mode]) & (np.abs(mode_ratios[mode]) < 100)
    ax.semilogx(k1_values[valid], mode_ratios[mode][valid], color=colors[mode], 
                linewidth=2, label=f'Mode {mode+1}')
ax.set_xlabel('k₁ (driving variable)')
ax.set_ylabel('Mode ratio (x₂/x₁)')
ax.set_title('Coupling Response: Mode Shape Ratios')
ax.legend()
ax.grid(True, alpha=0.3)

# Panel 5: Derivative (looking for discontinuities)
ax = axes[1, 1]
dw_dk = np.diff(eigenfreqs[0]) / np.diff(k1_values)
ax.semilogx(k1_values[:-1], dw_dk, color='#2196F3', linewidth=1)
ax.set_xlabel('k₁')
ax.set_ylabel('dω₁/dk₁')
ax.set_title('Derivative Test\nSmooth = Euclidean, Discontinuous = Phase Transition')
ax.grid(True, alpha=0.3)

# Panel 6: Summary verdict
ax = axes[1, 2]
ax.axis('off')
summary_text = """
NEGATIVE CONTROL RESULTS

System: 3 Coupled Harmonic Oscillators (LINEAR)
Driving variable: k₁ across 12 orders of magnitude

Expected: EUCLIDEAN geometry
  ✓ No self-similarity across scales
  ✓ Simple power-law (ω ∝ √k), not fractal
  ✓ Integer dimension (D ≈ 1.0)
  ✓ No phase transitions
  ✓ Monotonic, no cascade dynamics

If ALL criteria confirm Euclidean:
  → Method correctly discriminates
  → Instrument is calibrated for law testing

If ANY criterion shows fractal signatures:
  → Method may be introducing artifacts
  → STOP and investigate before proceeding
"""
ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig('36_negative_control_linear.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nFigure saved: 36_negative_control_linear.png")
