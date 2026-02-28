#!/usr/bin/env python3
"""
Create a set of figures numbered to match the Feigenbaum derivation paper.
Copies existing panels with paper-matched filenames and generates
the missing Figure 2C (meta-system result text box) as a standalone.

Leaves all original 42_panel_* files untouched.
"""

import shutil
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

BASE = '/Users/lucianrandolph/Downloads/Projects/einstein_fractal_analysis'
OUT = os.path.join(BASE, 'paper_figures')
os.makedirs(OUT, exist_ok=True)

# ============================================================
# Mapping: paper figure name → existing panel file
# ============================================================
copies = {
    'Figure_1A_Bifurcation_Diagrams.png':
        '42_panel_1A_bifurcation_diagrams.png',
    'Figure_1B_Universal_Shape.png':
        '42_panel_1B_universal_shape.png',
    'Figure_1C_Ratio_Convergence.png':
        '42_panel_1C_ratio_convergence.png',
    'Figure_2A_Self_Similarity.png':
        '42_panel_2A_lucian_method_meta.png',
    'Figure_2B_Parallel_Slopes.png':
        '42_panel_2B_parallel_slopes.png',
    # 2C generated below
    'Figure_3A_Three_Constraint_Intersection.png':
        '42_panel_3A_constraint_intersection.png',
    'Figure_3B_Topology_Determines_Constant.png':
        '42_panel_3B_topology_determines_constant.png',
    'Figure_3C_Universal_Function.png':
        '42_panel_3C_universal_function.png',
    'Figure_4_Law_to_Constant_to_Stars.png':
        '42_panel_4_law_to_stars.png',
    'Figure_Composite_12_Panel.png':
        '42_feigenbaum_derivation_composite.png',
}

print("Creating paper figure set...\n")

for fig_name, panel_name in copies.items():
    src = os.path.join(BASE, panel_name)
    dst = os.path.join(OUT, fig_name)
    if os.path.exists(src):
        shutil.copy2(src, dst)
        size_kb = os.path.getsize(dst) / 1024
        print(f"  {fig_name:50s}  ({size_kb:.0f} KB)")
    else:
        print(f"  WARNING: {panel_name} not found!")

# ============================================================
# Generate Figure 2C: Meta-System Result (standalone text box)
# ============================================================
print("\nGenerating Figure 2C (Meta-System Result)...")

DELTA_TRUE = 4.669201609102990
exp_slope = -np.log(DELTA_TRUE)  # -1.5410

fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.axis('off')

meta_text = (
    "THE META-SYSTEM RESULT\n"
    "══════════════════════════════\n\n"
    "The interval sequence {dₙ}\n"
    "from period-doubling IS itself\n"
    "a nonlinear coupled system.\n\n"
    "The Lucian Method applied to it:\n\n"
    "  • Self-similarity:  YES  (constant δₙ)\n"
    "  • Power-law:        YES  (geometric decay)\n"
    "  • Sub-Euclidean:    YES\n\n"
    "The slope of ln(dₙ) vs n\n"
    "DIRECTLY ENCODES δ:\n\n"
    f"  slope = −ln(δ) = {exp_slope:.4f}\n"
    f"  δ = e^(−slope) = {DELTA_TRUE:.4f}\n\n"
    "The Lucian Method doesn't just\n"
    "DETECT geometric organization.\n"
    "It MEASURES the Feigenbaum constant\n"
    "as the geometric signature.\n\n"
    "══════════════════════════════\n"
    "Four maps.  One slope.  One constant.\n"
    f"δ = {DELTA_TRUE}"
)

ax.text(0.5, 0.5, meta_text,
        transform=ax.transAxes,
        fontsize=16,
        va='center', ha='center',
        fontfamily='monospace',
        bbox=dict(boxstyle='round,pad=1.0',
                  facecolor='lightyellow',
                  edgecolor='#333333',
                  linewidth=2,
                  alpha=0.95))

fig.patch.set_facecolor('white')
dst_2c = os.path.join(OUT, 'Figure_2C_Meta_System_Result.png')
fig.savefig(dst_2c, dpi=200, bbox_inches='tight', facecolor='white')
plt.close(fig)
size_kb = os.path.getsize(dst_2c) / 1024
print(f"  Figure_2C_Meta_System_Result.png                    ({size_kb:.0f} KB)")

# ============================================================
# Summary
# ============================================================
all_figs = sorted(os.listdir(OUT))
print(f"\n{'='*60}")
print(f"Paper figures saved to: {OUT}/")
print(f"Total figures: {len(all_figs)}")
print(f"{'='*60}")
for f in all_figs:
    print(f"  {f}")
print(f"\nOriginal 42_panel_* files are untouched.")
