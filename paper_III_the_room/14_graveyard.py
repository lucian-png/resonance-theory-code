"""
Paper Three - Resonance Theory III: The Room Is Larger Than We Thought
Script 14: The Graveyard of Unnecessary Physics (Figure 10)

The most efficient figure in the history of science:
One image that replaces twelve research programs.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.edgecolor': '#333333',
    'axes.labelcolor': '#222222',
    'text.color': '#222222',
    'xtick.color': '#444444',
    'ytick.color': '#444444',
    'grid.color': '#cccccc',
    'grid.alpha': 0.5,
    'font.family': 'serif',
    'font.size': 10,
})

COLORS = {
    'quantum': '#9b59b6',
    'nuclear': '#e74c3c',
    'cosmo': '#f39c12',
    'gold': '#d4af37',
    'blue': '#3498db',
    'green': '#2ecc71',
    'cyan': '#1abc9c',
    'white': '#222222',
    'pink': '#e91e8c',
    'orange': '#ff6b35',
    'dark_red': '#8b0000',
    'stone': '#6b6b7b',
}


def figure_10() -> None:
    """
    The Graveyard of Unnecessary Physics.
    Twelve categories of proposed but undetected physics
    rendered unnecessary by Resonance Theory.
    """
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(
        "Figure 10: The Graveyard of Unnecessary Physics",
        fontsize=18, fontweight='bold', color=COLORS['gold'], y=0.97
    )

    # Main axis for the graveyard
    ax = fig.add_axes([0.03, 0.06, 0.94, 0.86])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # Subtitle
    ax.text(0.5, 0.96,
            'Twelve Categories of Proposed Physics Rendered Unnecessary',
            fontsize=13, color=COLORS['stone'], ha='center', style='italic')
    ax.text(0.5, 0.93,
            'Not disproven. Unnecessary. Solutions to problems that do not exist.',
            fontsize=11, color=COLORS['cosmo'], ha='center')

    # ================================================================
    # The twelve gravestones arranged in three rows of four
    # ================================================================
    # From Paper Two (top row)
    paper_two_items = [
        ('Dark Matter\nParticles',
         'Decades of null\ndetection results',
         'Harmonic resonance\nat galactic scales',
         'Paper Two, Sec. 7.2'),
        ('Dark Energy\nas Substance',
         'Unknown substance\n68% of universe',
         'Harmonic phase\ntransition',
         'Paper Two, Sec. 7.3'),
        ('Supersymmetry\n(SUSY)',
         'LHC: no particles\nfound',
         'Hierarchy is the\nfractal structure',
         'Paper Two, Sec. 4.1'),
        ('Extra\nDimensions',
         'No experimental\nevidence',
         'Forces unified in\n4 dimensions',
         'Paper Two, Sec. 4.4'),
    ]

    # Also from Paper Two (middle row left)
    paper_two_items_2 = [
        ('String Theory\nas Unification',
         'No testable\npredictions',
         'Unification in\nexisting equations',
         'Paper Two, Sec. 4.4'),
        ('New Physics\nBeyond SM',
         'Standard Model\n"incomplete"',
         'SM is complete\nwhen classified',
         'Paper Two, Sec. 6'),
    ]

    # From Paper Three (middle row right + bottom row)
    paper_three_items = [
        ('Many-Worlds\nInterpretation',
         'Infinite parallel\nuniverses',
         'Harmonic phase\ntransition',
         'Paper Three, Sec. 2'),
        ('Firewalls &\nFuzzballs',
         'Black hole\ninfo solutions',
         'Info in harmonic\nspectrum',
         'Paper Three, Sec. 5'),
        ('Black Hole\nRemnants',
         'Stable leftover\nobjects',
         'Page curve from\nfractal geometry',
         'Paper Three, Sec. 5'),
        ('Axions\n(Peccei-Quinn)',
         '40+ years of\nnull searches',
         'Theta = 0 is\nground state',
         'Paper Three, Sec. 7'),
        ('Heavy See-Saw\nPartners',
         'Never detected\nat any collider',
         'Neutrinos at\nharmonic nodes',
         'Paper Three, Sec. 8'),
        ('New CP Violation\nSources',
         'Beyond-SM CP\nrequired',
         'Harmonic bias\nat baryogenesis',
         'Paper Three, Sec. 6'),
    ]

    all_items = paper_two_items + paper_two_items_2 + paper_three_items

    # Layout: 3 rows, 4 columns
    rows = 3
    cols = 4
    x_start = 0.04
    x_end = 0.96
    y_start = 0.85
    y_end = 0.10

    x_positions = np.linspace(x_start, x_end, cols + 1)
    y_positions = np.linspace(y_start, y_end, rows + 1)

    stone_color = '#e8e8f0'
    border_color_p2 = '#4466aa'
    border_color_p3 = '#aa4444'

    for idx, (name, old_status, resolution, source) in enumerate(all_items):
        row = idx // cols
        col = idx % cols

        cx = (x_positions[col] + x_positions[col + 1]) / 2
        cy = (y_positions[row] + y_positions[row + 1]) / 2

        w = 0.20
        h = 0.22

        # Determine paper source for coloring
        is_paper_three = 'Paper Three' in source
        border = border_color_p3 if is_paper_three else border_color_p2
        paper_label_color = COLORS['nuclear'] if is_paper_three else COLORS['blue']

        # Gravestone shape (rounded rectangle)
        stone = FancyBboxPatch(
            (cx - w / 2, cy - h / 2), w, h,
            boxstyle="round,pad=0.015",
            facecolor=stone_color,
            edgecolor=border,
            linewidth=2,
            alpha=0.9,
            transform=ax.transData,
        )
        ax.add_patch(stone)

        # Cross-out line
        ax.plot([cx - w / 2 + 0.01, cx + w / 2 - 0.01],
                [cy + h / 2 - 0.01, cy - h / 2 + 0.01],
                color=COLORS['nuclear'], alpha=0.2, linewidth=3)

        # Name (top, bold)
        ax.text(cx, cy + h / 2 - 0.035, name, fontsize=9,
                color=COLORS['white'], ha='center', va='top',
                fontweight='bold')

        # Old status (middle, dim)
        ax.text(cx, cy - 0.01, old_status, fontsize=7,
                color=COLORS['stone'], ha='center', va='center',
                style='italic')

        # Resolution (bottom)
        ax.text(cx, cy - h / 2 + 0.05, resolution, fontsize=7,
                color=COLORS['gold'], ha='center', va='bottom')

        # Source reference
        ax.text(cx, cy - h / 2 + 0.015, source, fontsize=5.5,
                color=paper_label_color, ha='center', va='bottom',
                alpha=0.7)

    # ================================================================
    # Row labels
    # ================================================================
    ax.text(0.01, 0.87, 'FROM\nPAPER\nTWO', fontsize=8, color=COLORS['blue'],
            va='top', fontweight='bold', rotation=0)
    ax.text(0.01, 0.57, 'FROM\nPAPERS\nTWO &\nTHREE', fontsize=8,
            color=COLORS['quantum'], va='top', fontweight='bold')
    ax.text(0.01, 0.28, 'FROM\nPAPER\nTHREE', fontsize=8,
            color=COLORS['nuclear'], va='top', fontweight='bold')

    # ================================================================
    # Bottom summary
    # ================================================================
    # Summary box
    summary_y = 0.03
    ax.text(0.5, summary_y,
            'TWELVE research programs.  BILLIONS of dollars.  DECADES of searching.\n'
            'Every null result is CONFIRMED as the correct result.\n'
            'The experiments were right. The classification was wrong.',
            fontsize=12, color=COLORS['gold'], ha='center', va='center',
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.6', facecolor='#f0f0f5',
                      edgecolor=COLORS['gold'], linewidth=2, alpha=0.95))

    # Stats on the left and right
    ax.text(0.08, summary_y, '12', fontsize=30, color=COLORS['nuclear'],
            ha='center', va='center', fontweight='bold', alpha=0.3)
    ax.text(0.92, summary_y, '0', fontsize=30, color=COLORS['green'],
            ha='center', va='center', fontweight='bold', alpha=0.3)

    fig.text(0.5, 0.01,
             'Not disproven. Unnecessary. '
             'Solutions to problems that do not exist when the equations are properly classified.',
             ha='center', fontsize=11, color=COLORS['gold'], style='italic')

    path = os.path.join(BASE_DIR, 'p3_fig10_graveyard.png')
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("PAPER THREE --- THE GRAVEYARD (Figure 10)")
    print("=" * 60)

    print("\nFigure 10: The Graveyard of Unnecessary Physics...")
    figure_10()

    print("\n" + "=" * 60)
    print("FIGURE 10 COMPLETE")
    print("The most efficient figure in the history of science.")
    print("=" * 60)
