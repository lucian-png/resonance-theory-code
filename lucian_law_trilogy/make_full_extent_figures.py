#!/usr/bin/env python3
"""
Generate figures for Paper 3: The Full Extent of the Lucian Law.
Four conceptual/schematic figures matching the paper's figure numbering.

Figure 1: The Self-Application Hierarchy
Figure 2: The Basin Transition Curve (inflation as Phase II)
Figure 3: Cosmic Web as Dual Attractor Architecture (schematic)
Figure 4: Cross-Scale Self-Similarity
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import os

OUT = '/Users/lucianrandolph/Downloads/Projects/einstein_fractal_analysis/paper_figures'
os.makedirs(OUT, exist_ok=True)

# Color palette — deep blues, subtle golds, clean
DEEP_BLUE = '#1a365d'
MED_BLUE = '#2c5282'
LIGHT_BLUE = '#bee3f8'
GOLD = '#d4a843'
DARK_GOLD = '#b8860b'
SOFT_RED = '#e53e3e'
SOFT_GREEN = '#38a169'
WARM_GRAY = '#4a5568'
OFF_WHITE = '#f7fafc'
CREAM = '#fefcf3'

# ============================================================
# FIGURE 1: THE SELF-APPLICATION HIERARCHY
# ============================================================
print("Generating Figure 1: Self-Application Hierarchy...")

fig1, ax1 = plt.subplots(1, 1, figsize=(10, 12))
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 14)
ax1.axis('off')
fig1.patch.set_facecolor('white')

# Title
ax1.text(5, 13.5, 'THE SELF-APPLICATION HIERARCHY',
         ha='center', va='center', fontsize=18, fontweight='bold',
         fontfamily='serif', color=DEEP_BLUE)
ax1.text(5, 13.0, 'Each layer produces a qualifying system for the next',
         ha='center', va='center', fontsize=11, fontstyle='italic',
         fontfamily='serif', color=WARM_GRAY)

# Layer boxes — from bottom to top
layers = [
    {
        'y': 1.5, 'label': 'LAYER 1', 'color': '#e2e8f0', 'edge': MED_BLUE,
        'title': 'Individual Systems',
        'content': 'Einstein  ·  Yang-Mills  ·  Navier-Stokes  ·  Boltzmann\n'
                   '19 systems tested  ·  0 refutations',
        'icon': 'Geometric organization in each system'
    },
    {
        'y': 4.5, 'label': 'LAYER 2', 'color': '#dbeafe', 'edge': MED_BLUE,
        'title': 'The Space of All Systems',
        'content': 'Systems cluster in basins  ·  Transition zones depleted\n'
                   'Confirmed by falsification protocol',
        'icon': 'Dual attractor architecture across systems'
    },
    {
        'y': 7.5, 'label': 'LAYER 3', 'color': '#bfdbfe', 'edge': MED_BLUE,
        'title': 'The Space of All Such Spaces',
        'content': 'Quantum collection  ·  Cosmological collection  ·  Biological collection\n'
                   'Each collection is a Layer 2 space',
        'icon': 'Same architecture: dual attractors, self-similarity'
    },
    {
        'y': 10.5, 'label': 'LAYER N', 'color': '#93c5fd', 'edge': DEEP_BLUE,
        'title': 'The Tower Has No Top',
        'content': 'Each layer qualifies for the next\n'
                   'Same architecture at every level',
        'icon': 'Self-similar: the hierarchy IS the law\'s signature'
    },
]

for layer in layers:
    y = layer['y']
    # Main box
    box = FancyBboxPatch((1.0, y - 1.0), 8.0, 2.2,
                         boxstyle="round,pad=0.15",
                         facecolor=layer['color'], edgecolor=layer['edge'],
                         linewidth=2.0, alpha=0.9)
    ax1.add_patch(box)

    # Layer label
    ax1.text(1.5, y + 0.85, layer['label'],
             ha='left', va='center', fontsize=10, fontweight='bold',
             fontfamily='monospace', color=layer['edge'])

    # Title
    ax1.text(5, y + 0.85, layer['title'],
             ha='center', va='center', fontsize=13, fontweight='bold',
             fontfamily='serif', color=DEEP_BLUE)

    # Content
    ax1.text(5, y + 0.15, layer['content'],
             ha='center', va='center', fontsize=9.5,
             fontfamily='serif', color=WARM_GRAY)

    # Icon text
    ax1.text(5, y - 0.55, layer['icon'],
             ha='center', va='center', fontsize=9,
             fontfamily='serif', fontstyle='italic', color=DARK_GOLD)

# Arrows between layers
for i in range(3):
    y_start = layers[i]['y'] + 1.2
    y_end = layers[i + 1]['y'] - 1.0
    ax1.annotate('', xy=(5, y_end), xytext=(5, y_start),
                 arrowprops=dict(arrowstyle='->', color=GOLD,
                                 lw=2.5, connectionstyle='arc3,rad=0'))
    # "Qualifies" label
    mid_y = (y_start + y_end) / 2
    ax1.text(6.2, mid_y, 'qualifies →',
             ha='left', va='center', fontsize=9,
             fontfamily='serif', fontstyle='italic', color=DARK_GOLD)

# Dots between Layer 3 and Layer N
for dot_y in [9.7, 9.9, 10.1]:
    ax1.plot(5, dot_y - 0.5, 'o', color=GOLD, markersize=4)

# Bottom annotation
ax1.text(5, 0.3,
         'The law at Layer 47 is identical to the law at Layer 1.\n'
         'Not regress. Self-similarity. A fractal, not a tower of turtles.',
         ha='center', va='center', fontsize=10, fontfamily='serif',
         fontstyle='italic', color=DEEP_BLUE,
         bbox=dict(boxstyle='round,pad=0.4', facecolor=CREAM,
                   edgecolor=GOLD, alpha=0.9))

fig1.savefig(os.path.join(OUT, 'Figure_1_Self_Application_Hierarchy.png'),
             dpi=200, bbox_inches='tight', facecolor='white')
plt.close(fig1)
print("  Done.")

# ============================================================
# FIGURE 2: THE BASIN TRANSITION CURVE
# ============================================================
print("Generating Figure 2: Basin Transition Curve...")

fig2, ax2 = plt.subplots(1, 1, figsize=(12, 8))

# Generate the transition curve
t = np.linspace(-3, 7, 1000)

# Sigmoid-based transition with overshoot/oscillation at end
def basin_curve(t):
    """
    Basin transition: slow departure → exponential acceleration →
    deceleration → settling with damped oscillation.
    """
    # Core sigmoid transition
    sigmoid = 1 / (1 + np.exp(-1.8 * (t - 1.5)))
    # Add damped oscillation at the end (settling)
    oscillation = 0.04 * np.exp(-0.8 * np.maximum(t - 3.5, 0)) * \
                  np.sin(4 * np.maximum(t - 3.5, 0))
    return sigmoid + oscillation

y = basin_curve(t)

# Background zones
ax2.axvspan(-3, 0, alpha=0.08, color='blue', label='Old Basin')
ax2.axvspan(0, 1, alpha=0.06, color='orange')
ax2.axvspan(1, 3.5, alpha=0.08, color='red')
ax2.axvspan(3.5, 5, alpha=0.06, color='orange')
ax2.axvspan(5, 7, alpha=0.08, color='green', label='New Basin')

# Main curve
ax2.plot(t, y, color=DEEP_BLUE, linewidth=3.5, zorder=5)

# Basin levels
ax2.axhline(y=0, color=WARM_GRAY, linewidth=1, linestyle=':', alpha=0.5)
ax2.axhline(y=1, color=WARM_GRAY, linewidth=1, linestyle=':', alpha=0.5)
ax2.text(-2.5, -0.06, 'Old basin level', fontsize=9, color=WARM_GRAY,
         fontfamily='serif', fontstyle='italic')
ax2.text(5.5, 1.04, 'New basin level', fontsize=9, color=WARM_GRAY,
         fontfamily='serif', fontstyle='italic')

# Phase labels — Roman numerals
phases = [
    (-1.5, 'I', 'PRE-INFLATION\nSlow departure\nfrom old state', 0.08),
    (0.5, 'II', 'INFLATION\nExponential\nacceleration', 0.45),
    (2.5, 'III', 'REHEATING\nDeceleration\ninto new basin', 0.85),
    (5.5, 'IV', 'EXPANSION\nSettling\n13.8 Gyr', 0.97),
]

for x_pos, numeral, label, y_pos in phases:
    ax2.text(x_pos, 1.18, f'Phase {numeral}',
             ha='center', va='bottom', fontsize=12, fontweight='bold',
             fontfamily='serif', color=DEEP_BLUE)
    ax2.text(x_pos, 1.12, label,
             ha='center', va='bottom', fontsize=9,
             fontfamily='serif', color=WARM_GRAY)

# Phase boundary markers
for x_bound in [0, 1, 3.5, 5]:
    ax2.axvline(x=x_bound, color=GOLD, linewidth=1.5, linestyle='--', alpha=0.5)

# Mark the Big Bang moment
ax2.annotate('Big Bang\nthreshold',
             xy=(0, basin_curve(np.array([0]))[0]),
             xytext=(-2, 0.5),
             fontsize=11, fontweight='bold', fontfamily='serif', color=SOFT_RED,
             ha='center',
             arrowprops=dict(arrowstyle='->', color=SOFT_RED, lw=2))

# Mark the steepest point (peak inflation)
steepest_t = 1.5
ax2.plot(steepest_t, basin_curve(np.array([steepest_t]))[0], 'o',
         color=SOFT_RED, markersize=10, zorder=6)
ax2.annotate('Peak exponential\nacceleration',
             xy=(steepest_t, basin_curve(np.array([steepest_t]))[0]),
             xytext=(3.5, 0.3),
             fontsize=10, fontfamily='serif', color=SOFT_RED,
             ha='center',
             arrowprops=dict(arrowstyle='->', color=SOFT_RED, lw=1.5))

# Title
ax2.set_title('THE BASIN TRANSITION CURVE\n'
              'Inflation as the Geometric Shape of the Dual Attractor Transition',
              fontsize=16, fontweight='bold', fontfamily='serif',
              color=DEEP_BLUE, pad=70)

ax2.set_xlabel('Transition coordinate', fontsize=12, fontfamily='serif')
ax2.set_ylabel('Basin state', fontsize=12, fontfamily='serif')
ax2.set_ylim(-0.15, 1.35)
ax2.set_xlim(-3, 7)
ax2.tick_params(labelsize=9)

# Bottom annotation
ax2.text(2, -0.12,
         'No inflaton field required. The exponential phase is the system\n'
         'falling through the steepest gradient between dual attractor basins.',
         ha='center', va='top', fontsize=10, fontfamily='serif',
         fontstyle='italic', color=DEEP_BLUE,
         bbox=dict(boxstyle='round,pad=0.4', facecolor=CREAM,
                   edgecolor=GOLD, alpha=0.9))

fig2.savefig(os.path.join(OUT, 'Figure_2_Basin_Transition_Curve.png'),
             dpi=200, bbox_inches='tight', facecolor='white')
plt.close(fig2)
print("  Done.")

# ============================================================
# FIGURE 3: COSMIC WEB AS DUAL ATTRACTOR ARCHITECTURE
# ============================================================
print("Generating Figure 3: Cosmic Web Dual Attractor...")

fig3, axes3 = plt.subplots(1, 3, figsize=(16, 6))

# Panel A: Schematic cosmic web (filaments + voids)
ax3a = axes3[0]
np.random.seed(42)

# Generate filament structure schematically
n_nodes = 30
node_x = np.random.uniform(0.1, 0.9, n_nodes)
node_y = np.random.uniform(0.1, 0.9, n_nodes)

# Draw filaments as connections between nearby nodes
from scipy.spatial import Delaunay
points = np.column_stack([node_x, node_y])
tri = Delaunay(points)

# Draw edges
for simplex in tri.simplices:
    for i in range(3):
        p1 = simplex[i]
        p2 = simplex[(i + 1) % 3]
        dist = np.sqrt((node_x[p1] - node_x[p2])**2 + (node_y[p1] - node_y[p2])**2)
        if dist < 0.35:
            ax3a.plot([node_x[p1], node_x[p2]], [node_y[p1], node_y[p2]],
                      color=MED_BLUE, linewidth=1.5 * (1 - dist), alpha=0.4)

# Galaxies along filaments (clustered near nodes)
for nx_i, ny_i in zip(node_x, node_y):
    n_gal = np.random.randint(5, 20)
    gx = nx_i + np.random.normal(0, 0.025, n_gal)
    gy = ny_i + np.random.normal(0, 0.025, n_gal)
    sizes = np.random.uniform(1, 8, n_gal)
    ax3a.scatter(gx, gy, s=sizes, c=MED_BLUE, alpha=0.6, edgecolors='none')

# Mark voids
void_centers = [(0.3, 0.6), (0.7, 0.3), (0.5, 0.85)]
for vx, vy in void_centers:
    circle = plt.Circle((vx, vy), 0.12, fill=True,
                         facecolor='white', edgecolor=GOLD,
                         linewidth=1.5, linestyle='--', alpha=0.4)
    ax3a.add_patch(circle)
    ax3a.text(vx, vy, 'VOID', ha='center', va='center',
              fontsize=7, color=DARK_GOLD, fontfamily='serif', fontstyle='italic')

ax3a.set_xlim(0, 1)
ax3a.set_ylim(0, 1)
ax3a.set_aspect('equal')
ax3a.set_title('Cosmic Web Structure\n(Schematic)', fontsize=12,
               fontweight='bold', fontfamily='serif', color=DEEP_BLUE)
ax3a.text(0.5, -0.05, 'Filaments = populated basins\nVoids = depleted transition zones',
          ha='center', va='top', fontsize=9, fontfamily='serif',
          fontstyle='italic', color=WARM_GRAY, transform=ax3a.transAxes)
ax3a.axis('off')

# Panel B: Density distribution — dual peaks
ax3b = axes3[1]
x_dens = np.linspace(-1, 6, 500)
# Filament population (high density)
filament_pop = 1.2 * np.exp(-0.5 * ((x_dens - 3.5) / 0.8)**2)
# Void population (low density)
void_pop = 0.8 * np.exp(-0.5 * ((x_dens - 0.5) / 0.6)**2)
# Depleted zone
total = filament_pop + void_pop

ax3b.fill_between(x_dens, void_pop, alpha=0.3, color=SOFT_GREEN, label='Void galaxies')
ax3b.fill_between(x_dens, filament_pop, alpha=0.3, color=MED_BLUE, label='Filament galaxies')
ax3b.plot(x_dens, total, color=DEEP_BLUE, linewidth=2, label='Combined')

# Mark depleted zone
ax3b.axvspan(1.5, 2.5, alpha=0.1, color=SOFT_RED)
ax3b.text(2.0, 0.5, 'Depleted\nzone', ha='center', va='center',
          fontsize=9, color=SOFT_RED, fontweight='bold', fontfamily='serif')

ax3b.set_xlabel('log(Galaxy Density)', fontsize=10, fontfamily='serif')
ax3b.set_ylabel('Population', fontsize=10, fontfamily='serif')
ax3b.set_title('Predicted Distribution\n(Dual Attractor Basins)', fontsize=12,
               fontweight='bold', fontfamily='serif', color=DEEP_BLUE)
ax3b.legend(fontsize=8, loc='upper right')
ax3b.tick_params(labelsize=8)

# Panel C: The analogy — stellar (confirmed) vs cosmic (predicted)
ax3c = axes3[2]
ax3c.axis('off')

comparison_text = (
    "CONFIRMED\n"
    "═══════════════════\n\n"
    "Gaia DR3 Stellar Data\n"
    "50,000 stars\n\n"
    "Active basin:   23,133\n"
    "Passive basin:  26,867\n"
    "Depleted gap:   YES\n"
    "KS p-value:     < 10⁻³⁰⁰\n\n"
    "─ ─ ─ ─ ─ ─ ─ ─ ─ ─\n\n"
    "PREDICTED\n"
    "═══════════════════\n\n"
    "SDSS Cosmic Web\n"
    "~1 million galaxies\n\n"
    "Filament basin:  ?\n"
    "Void basin:      ?\n"
    "Depleted gap:    YES (predicted)\n"
    "Same statistics: TESTABLE"
)

ax3c.text(0.5, 0.5, comparison_text,
          ha='center', va='center', fontsize=11,
          fontfamily='monospace',
          transform=ax3c.transAxes,
          bbox=dict(boxstyle='round,pad=0.8', facecolor=CREAM,
                    edgecolor=GOLD, linewidth=2, alpha=0.95))
ax3c.set_title('Stellar → Cosmic\n(Same Framework)', fontsize=12,
               fontweight='bold', fontfamily='serif', color=DEEP_BLUE)

fig3.suptitle('THE COSMIC WEB AS DUAL ATTRACTOR ARCHITECTURE',
              fontsize=16, fontweight='bold', fontfamily='serif',
              color=DEEP_BLUE, y=1.02)
fig3.tight_layout()
fig3.savefig(os.path.join(OUT, 'Figure_3_Cosmic_Web_Dual_Attractor.png'),
             dpi=200, bbox_inches='tight', facecolor='white')
plt.close(fig3)
print("  Done.")

# ============================================================
# FIGURE 4: CROSS-SCALE SELF-SIMILARITY
# ============================================================
print("Generating Figure 4: Cross-Scale Self-Similarity...")

fig4, axes4 = plt.subplots(1, 3, figsize=(16, 7))

scales = [
    {
        'ax': axes4[0],
        'title': 'QUANTUM SCALE',
        'subtitle': 'Bound / Unbound States',
        'basin_a': 'Bound states',
        'basin_b': 'Unbound states',
        'pop_a': 0.62,
        'pop_b': 0.38,
        'color_a': '#4299e1',
        'color_b': '#fc8181',
        'status': 'Predicted',
        'stat_text': 'Same structural\ntype predicted',
    },
    {
        'ax': axes4[1],
        'title': 'STELLAR SCALE',
        'subtitle': 'Active / Passive Stars',
        'basin_a': 'Active (23,133)',
        'basin_b': 'Passive (26,867)',
        'pop_a': 0.463,
        'pop_b': 0.537,
        'color_a': '#4299e1',
        'color_b': '#fc8181',
        'status': 'Confirmed',
        'stat_text': 'p < 10⁻³⁰⁰\n50,000 stars',
    },
    {
        'ax': axes4[2],
        'title': 'COSMOLOGICAL SCALE',
        'subtitle': 'Filaments / Voids',
        'basin_a': 'Filament galaxies',
        'basin_b': 'Void galaxies',
        'pop_a': 0.55,
        'pop_b': 0.45,
        'color_a': '#4299e1',
        'color_b': '#fc8181',
        'status': 'Predicted',
        'stat_text': 'Testable with\nSDSS data',
    },
]

for s in scales:
    ax = s['ax']

    # Dual basin visualization — two Gaussians with gap
    x = np.linspace(-3, 7, 500)
    g1 = s['pop_a'] * 1.5 * np.exp(-0.5 * ((x - 1.0) / 0.8)**2)
    g2 = s['pop_b'] * 1.5 * np.exp(-0.5 * ((x - 4.5) / 0.9)**2)

    ax.fill_between(x, g1, alpha=0.4, color=s['color_a'], label=s['basin_a'])
    ax.fill_between(x, g2, alpha=0.4, color=s['color_b'], label=s['basin_b'])
    ax.plot(x, g1 + g2, color=DEEP_BLUE, linewidth=2)

    # Depleted zone
    ax.axvspan(2.2, 3.3, alpha=0.1, color=SOFT_RED)
    ax.annotate('gap', xy=(2.75, 0.08), fontsize=9, ha='center',
                color=SOFT_RED, fontweight='bold', fontfamily='serif')

    # Title
    ax.set_title(f'{s["title"]}\n{s["subtitle"]}',
                 fontsize=13, fontweight='bold', fontfamily='serif',
                 color=DEEP_BLUE)

    # Status badge
    badge_color = SOFT_GREEN if s['status'] == 'Confirmed' else GOLD
    ax.text(0.95, 0.95, s['status'],
            transform=ax.transAxes, ha='right', va='top',
            fontsize=10, fontweight='bold', fontfamily='serif',
            color='white',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=badge_color,
                      edgecolor='none', alpha=0.9))

    # Stats
    ax.text(0.5, -0.12, s['stat_text'],
            transform=ax.transAxes, ha='center', va='top',
            fontsize=10, fontfamily='serif', fontstyle='italic',
            color=WARM_GRAY)

    ax.legend(fontsize=8, loc='upper left')
    ax.set_xlabel('Density (normalized)', fontsize=9, fontfamily='serif')
    ax.set_ylabel('Population', fontsize=9, fontfamily='serif')
    ax.tick_params(labelsize=7)

# Arrows between panels
for i in range(2):
    fig4.text(0.355 + i * 0.315, 0.5, '⟷',
              fontsize=28, ha='center', va='center',
              color=GOLD, fontweight='bold',
              transform=fig4.transFigure)

fig4.suptitle('CROSS-SCALE SELF-SIMILARITY\n'
              'Same Dual Attractor Architecture at Every Scale',
              fontsize=16, fontweight='bold', fontfamily='serif',
              color=DEEP_BLUE, y=1.05)

# Bottom text
fig4.text(0.5, -0.06,
          'Different specific parameters (Layer 2) · Same organizational type (Layer 1)\n'
          'The hierarchy predicts identical structure across all qualifying scales.',
          ha='center', va='top', fontsize=11, fontfamily='serif',
          fontstyle='italic', color=DEEP_BLUE,
          transform=fig4.transFigure)

fig4.tight_layout()
fig4.savefig(os.path.join(OUT, 'Figure_4_Cross_Scale_Self_Similarity.png'),
             dpi=200, bbox_inches='tight', facecolor='white')
plt.close(fig4)
print("  Done.")

# ============================================================
# SUMMARY
# ============================================================
print(f"\n{'='*60}")
print(f"Paper 3 figures saved to: {OUT}/")
all_p3 = [f for f in sorted(os.listdir(OUT)) if f.startswith('Figure_') and 'Hierarchy' in f
          or 'Basin_Transition' in f or 'Cosmic_Web' in f or 'Cross_Scale' in f]
# Just list all Figure_ files
all_figs = sorted([f for f in os.listdir(OUT) if f.startswith('Figure_')])
print(f"\nAll figures in paper_figures/:")
for f in all_figs:
    size_kb = os.path.getsize(os.path.join(OUT, f)) / 1024
    paper = ''
    if any(x in f for x in ['1A_', '1B_', '1C_', '2A_', '2B_', '2C_', '3A_', '3B_', '3C_',
                              '4_Law', 'Composite']):
        paper = '(Paper 2)'
    elif any(x in f for x in ['Hierarchy', 'Basin', 'Cosmic', 'Cross_Scale']):
        paper = '(Paper 3)'
    print(f"  {f:55s} {size_kb:6.0f} KB  {paper}")
print(f"{'='*60}")
