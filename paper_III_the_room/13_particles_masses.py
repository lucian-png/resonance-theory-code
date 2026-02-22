"""
Paper Three - Resonance Theory III: The Room Is Larger Than We Thought
Script 13: Particle Physics Figures (7-9)

Figure 7: Matter-antimatter harmonic bias
Figure 8: Strong CP — theta at harmonic ground state
Figure 9: Neutrino masses as harmonic nodes on the landscape
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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
}


# ============================================================
# FIGURE 7: Matter-Antimatter Harmonic Bias
# ============================================================
def figure_7() -> None:
    """
    Shows matter-antimatter asymmetry as a harmonic bias
    in the fractal geometric landscape.
    """
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(
        "Figure 7: Matter-Antimatter Asymmetry --- The Harmonic Bias",
        fontsize=16, fontweight='bold', color=COLORS['gold'], y=0.97
    )

    gs = gridspec.GridSpec(2, 3, hspace=0.45, wspace=0.35,
                           left=0.08, right=0.95, top=0.91, bottom=0.06)

    # === Panel A: Symmetric potential (what you'd expect) ===
    ax_a = fig.add_subplot(gs[0, 0])
    phi = np.linspace(-3, 3, 500)
    V_sym = phi ** 4 - 4 * phi ** 2
    ax_a.plot(phi, V_sym, color=COLORS['cosmo'], linewidth=2.5)
    ax_a.fill_between(phi, V_sym, -10, alpha=0.1, color=COLORS['cosmo'])

    # Mark the two minima
    ax_a.plot(-np.sqrt(2), -4, 'o', color=COLORS['quantum'],
              markersize=12, zorder=5)
    ax_a.plot(np.sqrt(2), -4, 'o', color=COLORS['nuclear'],
              markersize=12, zorder=5)
    ax_a.text(-np.sqrt(2), -5.5, 'Matter', fontsize=9,
              color=COLORS['quantum'], ha='center', fontweight='bold')
    ax_a.text(np.sqrt(2), -5.5, 'Anti-\nmatter', fontsize=9,
              color=COLORS['nuclear'], ha='center', fontweight='bold')
    ax_a.text(0, 1, 'SYMMETRIC\n(Non-fractal)', fontsize=10,
              color=COLORS['cosmo'], ha='center', fontweight='bold')

    ax_a.set_xlabel('Field value', fontsize=9)
    ax_a.set_ylabel('V(field)', fontsize=9)
    ax_a.set_title('Integer-Dimensional:\nPerfectly Symmetric', fontsize=10,
                   color=COLORS['cosmo'])
    ax_a.set_ylim(-7, 5)

    # === Panel B: Asymmetric potential (fractal geometric) ===
    ax_b = fig.add_subplot(gs[0, 1])
    # Slight asymmetry from fractal geometry
    epsilon = 0.15  # Small bias
    V_asym = phi ** 4 - 4 * phi ** 2 - epsilon * phi ** 3 + 0.3 * epsilon * phi
    ax_b.plot(phi, V_asym, color=COLORS['gold'], linewidth=2.5)
    ax_b.fill_between(phi, V_asym, -10, alpha=0.1, color=COLORS['gold'])

    # Find minima
    min_left = phi[np.argmin(V_asym[phi < 0])]
    min_right_idx = np.argmin(V_asym[phi > 0]) + len(phi[phi <= 0])
    min_right = phi[min_right_idx]
    V_left = V_asym[np.argmin(V_asym[phi < 0])]
    V_right = V_asym[min_right_idx]

    ax_b.plot(min_left, V_left, 'o', color=COLORS['quantum'],
              markersize=14, zorder=5)
    ax_b.plot(min_right, V_right, 'o', color=COLORS['nuclear'],
              markersize=10, zorder=5)
    ax_b.text(min_left, V_left - 1.5, 'Matter\n(deeper)', fontsize=9,
              color=COLORS['quantum'], ha='center', fontweight='bold')
    ax_b.text(min_right, V_right - 1.5, 'Anti-\nmatter', fontsize=9,
              color=COLORS['nuclear'], ha='center')

    # Show the energy difference
    ax_b.annotate('', xy=(2.3, V_left), xytext=(2.3, V_right),
                  arrowprops=dict(arrowstyle='<->', color=COLORS['gold'], lw=2))
    ax_b.text(2.7, (V_left + V_right) / 2, 'Harmonic\nbias', fontsize=8,
              color=COLORS['gold'], ha='center')

    ax_b.text(0, 2, 'ASYMMETRIC\n(Fractal geometric)', fontsize=10,
              color=COLORS['gold'], ha='center', fontweight='bold')

    ax_b.set_xlabel('Field value', fontsize=9)
    ax_b.set_ylabel('V(field)', fontsize=9)
    ax_b.set_title('Fractal Geometric:\nNatural Asymmetry', fontsize=10,
                   color=COLORS['gold'])
    ax_b.set_ylim(-7, 5)

    # === Panel C: The ratio ===
    ax_c = fig.add_subplot(gs[0, 2])
    # Big Bang particle counts
    labels = ['Matter\nparticles', 'Antimatter\nparticles', 'Excess\nmatter']
    counts = [1e9 + 1, 1e9, 1]
    colors_bars = [COLORS['quantum'], COLORS['nuclear'], COLORS['gold']]

    # Show as bar chart (log scale)
    y_pos = [0.7, 0.4, 0.1]
    for y, label, count, col in zip(y_pos, labels, counts, colors_bars):
        bar_width = np.log10(count) / 10
        ax_c.barh(y, bar_width, height=0.15, color=col, alpha=0.7)
        ax_c.text(bar_width + 0.02, y, f'{count:.0e}', fontsize=9,
                  color=col, va='center', fontweight='bold')
        ax_c.text(-0.02, y, label, fontsize=9, color=col,
                  va='center', ha='right')

    ax_c.text(0.5, 0.95, 'For every 10^9 pairs:', fontsize=10,
              color=COLORS['white'], ha='center', transform=ax_c.transAxes)
    ax_c.text(0.5, 0.02, '1 extra matter particle\n= ALL visible matter\n'
              'in the universe', fontsize=9, color=COLORS['gold'],
              ha='center', transform=ax_c.transAxes,
              bbox=dict(boxstyle='round,pad=0.3', facecolor='#f0f0f5',
                        edgecolor=COLORS['gold'], alpha=0.8))

    ax_c.set_xlim(-0.35, 1.1)
    ax_c.set_ylim(-0.1, 1)
    ax_c.set_title('The Ratio:\n1 in 10^9', fontsize=10,
                   color=COLORS['gold'])
    ax_c.axis('off')

    # === Panel D: Harmonic landscape showing bias ===
    ax_d = fig.add_subplot(gs[1, 0])
    # Harmonic mode amplitudes for matter vs antimatter
    modes = np.arange(1, 21)
    # Matter modes — slightly higher amplitude
    A_matter = (1.0 / modes ** 0.8) * (1 + 0.05 * np.sin(modes * 0.7))
    # Antimatter modes — slightly lower
    A_anti = (1.0 / modes ** 0.8) * (1 - 0.05 * np.sin(modes * 0.7))

    ax_d.bar(modes - 0.2, A_matter, width=0.35, color=COLORS['quantum'],
             alpha=0.7, label='Matter modes')
    ax_d.bar(modes + 0.2, A_anti, width=0.35, color=COLORS['nuclear'],
             alpha=0.7, label='Antimatter modes')

    # Highlight the difference
    for m in modes:
        if A_matter[m - 1] > A_anti[m - 1]:
            ax_d.plot(m, A_matter[m - 1] + 0.02, 'v', color=COLORS['gold'],
                      markersize=4)

    ax_d.set_xlabel('Harmonic Mode n', fontsize=9)
    ax_d.set_ylabel('Mode Amplitude', fontsize=9)
    ax_d.set_title('Harmonic Bias:\nMatter Modes Slightly Favored', fontsize=10,
                   color=COLORS['quantum'])
    ax_d.legend(fontsize=8, loc='upper right',
                facecolor='#f0f0f5', edgecolor='#999999', labelcolor='#222222')

    # === Panel E: CP violation connection ===
    ax_e = fig.add_subplot(gs[1, 1])

    cp_data = [
        ('K meson\nCP violation', '1964', 'Cronin & Fitch\n(Nobel 1980)', 0.82),
        ('B meson\nCP violation', '2001', 'BaBar & Belle\n(Nobel 2008)', 0.58),
        ('Sakharov\nconditions', '1967', 'B, C, CP violation\n+ non-equilibrium', 0.34),
        ('Fractal geometric\nharmonic bias', '2026', 'Natural asymmetry\nat baryogenesis scale', 0.10),
    ]
    for label, year, desc, y in cp_data:
        color = COLORS['gold'] if year == '2026' else COLORS['blue']
        ax_e.text(0.02, y, label, fontsize=9, color=color, va='center')
        ax_e.text(0.42, y, year, fontsize=9, color=color, va='center',
                  fontweight='bold')
        ax_e.text(0.55, y, desc, fontsize=8, color=color, va='center')

    # Connecting line
    ax_e.plot([0.45, 0.45], [0.08, 0.88], color=COLORS['white'],
              alpha=0.2, linewidth=2)
    for _, _, _, y in cp_data:
        ax_e.plot(0.45, y, 'o', color=COLORS['blue'] if y > 0.2 else COLORS['gold'],
                  markersize=6)

    ax_e.set_xlim(0, 1)
    ax_e.set_ylim(-0.05, 0.95)
    ax_e.axis('off')
    ax_e.set_title('CP Violation History:\nFrom Discovery to Resolution', fontsize=10,
                   color=COLORS['blue'])

    # === Panel F: Summary ===
    ax_f = fig.add_subplot(gs[1, 2])

    ax_f.text(0.5, 0.88, 'Why is there something\nrather than nothing?',
              fontsize=12, color=COLORS['white'], ha='center',
              fontweight='bold', va='center')

    ax_f.text(0.5, 0.65, 'Because a perfectly symmetric\nuniverse would be\nnon-fractal.',
              fontsize=10, color=COLORS['cosmo'], ha='center', va='center')

    ax_f.text(0.5, 0.42, 'Fractal structures are\ninherently asymmetric\nat specific scales.',
              fontsize=10, color=COLORS['gold'], ha='center', va='center',
              fontweight='bold')

    ax_f.text(0.5, 0.18, 'The asymmetry IS\nthe fractal geometry\nexpressing itself.',
              fontsize=11, color=COLORS['gold'], ha='center', va='center',
              bbox=dict(boxstyle='round,pad=0.5', facecolor='#f0f0f5',
                        edgecolor=COLORS['gold'], alpha=0.8))

    ax_f.set_xlim(0, 1)
    ax_f.set_ylim(0, 1)
    ax_f.axis('off')

    fig.text(0.5, 0.01,
             'A perfectly symmetric universe would be non-fractal. '
             'The asymmetry IS the fractal geometry expressing itself.',
             ha='center', fontsize=11, color=COLORS['gold'], style='italic')

    path = os.path.join(BASE_DIR, 'p3_fig07_matter_antimatter.png')
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================
# FIGURE 8: Strong CP — Theta at the Harmonic Ground State
# ============================================================
def figure_8() -> None:
    """
    Shows theta = 0 is the harmonic ground state of SU(3),
    not fine-tuning.
    """
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        r"Figure 8: The Strong CP Problem --- $\theta$ at the Harmonic Ground State",
        fontsize=16, fontweight='bold', color=COLORS['gold'], y=0.97
    )

    gs = gridspec.GridSpec(2, 3, hspace=0.45, wspace=0.35,
                           left=0.08, right=0.95, top=0.90, bottom=0.07)

    # === Panel A: Standard QCD theta potential ===
    ax_a = fig.add_subplot(gs[0, 0])
    theta = np.linspace(-np.pi, np.pi, 500)
    # V(theta) = -cos(theta) in standard QCD (any theta equally "natural")
    V_standard = 1 - np.cos(theta)
    ax_a.plot(theta / np.pi, V_standard, color=COLORS['cosmo'], linewidth=2.5)
    ax_a.fill_between(theta / np.pi, 0, V_standard, alpha=0.15, color=COLORS['cosmo'])

    # Show that theta = 0 is a minimum but theta could be anywhere
    ax_a.axvline(0, color=COLORS['white'], linestyle=':', alpha=0.3)
    ax_a.plot(0, 0, 'o', color=COLORS['cosmo'], markersize=10, zorder=5)

    # Arrow showing "any value equally natural"
    ax_a.annotate('', xy=(0.5, 0.3), xytext=(-0.5, 0.3),
                  arrowprops=dict(arrowstyle='<->', color=COLORS['nuclear'], lw=2))
    ax_a.text(0, 0.45, 'Any value\n"equally natural"', fontsize=8,
              color=COLORS['nuclear'], ha='center')

    ax_a.set_xlabel(r'$\theta / \pi$', fontsize=10)
    ax_a.set_ylabel(r'$V(\theta)$', fontsize=10)
    ax_a.set_title(r'Standard QCD: $\theta$ Could Be Anything', fontsize=10,
                   color=COLORS['cosmo'])

    # === Panel B: Experimental constraint ===
    ax_b = fig.add_subplot(gs[0, 1])
    # Neutron electric dipole moment constraint
    theta_range = np.logspace(-15, 0, 200)
    d_n = theta_range * 3.6e-16  # in e*cm (rough scaling)

    ax_b.loglog(theta_range, d_n, color=COLORS['blue'], linewidth=2.5)
    ax_b.axhline(1e-26, color=COLORS['nuclear'], linestyle='--', linewidth=2,
                 label='Experimental limit')
    ax_b.axvline(1e-10, color=COLORS['gold'], linestyle='--', linewidth=2,
                 label=r'$\theta < 10^{-10}$')

    # Shade the excluded region
    ax_b.fill_between(theta_range, d_n, 1e-10,
                      where=(d_n > 1e-26), alpha=0.15, color=COLORS['nuclear'])

    ax_b.set_xlabel(r'$\theta$', fontsize=10)
    ax_b.set_ylabel('Neutron EDM (e cm)', fontsize=10)
    ax_b.set_title(r'Measurement: $\theta < 10^{-10}$' + '\nExtraordinary Fine-Tuning?',
                   fontsize=10, color=COLORS['blue'])
    ax_b.legend(fontsize=8, loc='lower right',
                facecolor='#f0f0f5', edgecolor='#999999', labelcolor='#222222')
    ax_b.set_xlim(1e-15, 1)
    ax_b.set_ylim(1e-31, 1e-14)

    # === Panel C: Fractal geometric harmonic ground state ===
    ax_c = fig.add_subplot(gs[0, 2])
    # Harmonic potential with sharp minimum at theta = 0
    V_fractal = theta ** 2 * (1 + 0.3 * theta ** 2)
    V_fractal /= V_fractal.max()

    ax_c.plot(theta / np.pi, V_fractal, color=COLORS['gold'], linewidth=2.5)
    ax_c.fill_between(theta / np.pi, 0, V_fractal, alpha=0.15, color=COLORS['gold'])

    # Ground state ball
    ax_c.plot(0, 0, 'o', color=COLORS['gold'], markersize=14, zorder=5)
    ax_c.annotate(r'$\theta = 0$' + '\nGround state', xy=(0, 0),
                  xytext=(0.4, 0.3), fontsize=10, color=COLORS['gold'],
                  fontweight='bold',
                  arrowprops=dict(arrowstyle='->', color=COLORS['gold'], lw=2))

    # Restoring force arrows
    ax_c.annotate('', xy=(0.05 / np.pi, 0.02), xytext=(0.3 / np.pi, 0.15),
                  arrowprops=dict(arrowstyle='->', color=COLORS['green'], lw=2))
    ax_c.annotate('', xy=(-0.05 / np.pi, 0.02), xytext=(-0.3 / np.pi, 0.15),
                  arrowprops=dict(arrowstyle='->', color=COLORS['green'], lw=2))
    ax_c.text(-0.5, 0.5, 'Restoring\nforce', fontsize=8,
              color=COLORS['green'], ha='center')

    ax_c.set_xlabel(r'$\theta / \pi$', fontsize=10)
    ax_c.set_ylabel(r'$V_{fractal}(\theta)$', fontsize=10)
    ax_c.set_title(r'Fractal Geometric: $\theta = 0$' + '\nIs the Harmonic Ground State',
                   fontsize=10, color=COLORS['gold'])

    # === Panel D: Guitar string analogy ===
    ax_d = fig.add_subplot(gs[1, 0])
    x_string = np.linspace(0, 1, 200)

    # Excited state
    y_excited = 0.15 * np.sin(np.pi * x_string) + 0.08 * np.sin(3 * np.pi * x_string)
    ax_d.plot(x_string, y_excited + 0.3, color=COLORS['blue'], linewidth=2,
              label='Excited (any mode)')

    # Ground state (at rest)
    ax_d.plot(x_string, np.zeros_like(x_string), color=COLORS['gold'],
              linewidth=3, label='Ground state (rest)')

    # Fixed endpoints
    ax_d.plot([0, 1], [0, 0], 's', color=COLORS['white'], markersize=8)
    ax_d.plot([0, 1], [0.3, 0.3], 's', color=COLORS['white'], markersize=8)

    # Arrow showing relaxation
    ax_d.annotate('', xy=(0.5, 0.05), xytext=(0.5, 0.25),
                  arrowprops=dict(arrowstyle='->', color=COLORS['green'], lw=2))
    ax_d.text(0.6, 0.15, 'Naturally\nrelaxes', fontsize=9,
              color=COLORS['green'])

    ax_d.set_xlabel('Position', fontsize=9)
    ax_d.set_ylabel('Displacement', fontsize=9)
    ax_d.set_title(r'Guitar String: $\theta = 0$' + '\nis Like String at Rest',
                   fontsize=10, color=COLORS['gold'])
    ax_d.legend(fontsize=8, loc='upper right',
                facecolor='#f0f0f5', edgecolor='#999999', labelcolor='#222222')
    ax_d.set_ylim(-0.15, 0.5)

    # === Panel E: Axion search history ===
    ax_e = fig.add_subplot(gs[1, 1])

    searches = [
        ('ADMX', '2003-now', 'No detection', COLORS['nuclear']),
        ('CAST', '2003-2015', 'No detection', COLORS['nuclear']),
        ('ABRACADABRA', '2019-now', 'No detection', COLORS['nuclear']),
        ('HAYSTAC', '2017-now', 'No detection', COLORS['nuclear']),
        ('ORGAN', '2019-now', 'No detection', COLORS['nuclear']),
    ]

    ax_e.text(0.5, 0.95, 'AXION SEARCHES', fontsize=12, color=COLORS['nuclear'],
              ha='center', fontweight='bold')

    for i, (name, years, result, color) in enumerate(searches):
        y = 0.82 - i * 0.13
        ax_e.text(0.05, y, name, fontsize=10, color=color,
                  va='center', fontweight='bold')
        ax_e.text(0.35, y, years, fontsize=9, color=color, va='center')
        ax_e.text(0.65, y, result, fontsize=9, color=color, va='center')

    ax_e.text(0.5, 0.12, 'No axion found because\nthere is no axion to find.\n'
              r'$\theta = 0$ is the ground state.',
              fontsize=10, color=COLORS['gold'], ha='center',
              bbox=dict(boxstyle='round,pad=0.4', facecolor='#f0f0f5',
                        edgecolor=COLORS['gold'], alpha=0.8))

    ax_e.set_xlim(0, 1)
    ax_e.set_ylim(0, 1)
    ax_e.axis('off')
    ax_e.set_title('40+ Years of Null Results', fontsize=10,
                   color=COLORS['nuclear'])

    # === Panel F: Resolution comparison ===
    ax_f = fig.add_subplot(gs[1, 2])

    comparisons = [
        ('Problem:', r'Why is $\theta \approx 0$?', COLORS['white'], 0.88),
        ('', '', COLORS['white'], 0.78),
        ('Standard:', 'New particle (axion)\ndrives it to zero', COLORS['cosmo'], 0.68),
        ('Result:', '40 years, no axion', COLORS['nuclear'], 0.52),
        ('', '', COLORS['white'], 0.42),
        ('Resonance:', r'$\theta = 0$ is the' + '\nharmonic ground state\nof SU(3)', COLORS['gold'], 0.32),
        ('Result:', 'No particle needed.\nNot fine-tuned.\nNatural.', COLORS['gold'], 0.12),
    ]
    for label, text, color, y in comparisons:
        if label:
            ax_f.text(0.05, y, label, fontsize=9, color=color,
                      fontweight='bold', va='top')
            ax_f.text(0.35, y, text, fontsize=9, color=color, va='top')

    ax_f.set_xlim(0, 1)
    ax_f.set_ylim(0, 1)
    ax_f.axis('off')
    ax_f.set_title('Resolution', fontsize=10, color=COLORS['gold'])

    fig.text(0.5, 0.01,
             r'$\theta = 0$ is not fine-tuned. It is the harmonic ground state. '
             'No axion needed. Another particle that was never there.',
             ha='center', fontsize=11, color=COLORS['gold'], style='italic')

    path = os.path.join(BASE_DIR, 'p3_fig08_strong_cp.png')
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================
# FIGURE 9: Neutrino Masses as Harmonic Nodes
# ============================================================
def figure_9() -> None:
    """
    Shows neutrino masses as harmonic nodes on the fractal
    geometric landscape between weak and gravitational scales.
    """
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        "Figure 9: Neutrino Masses --- Harmonic Nodes on the Landscape",
        fontsize=16, fontweight='bold', color=COLORS['gold'], y=0.97
    )

    gs = gridspec.GridSpec(2, 3, hspace=0.45, wspace=0.35,
                           left=0.08, right=0.95, top=0.90, bottom=0.07)

    # === Panel A: Particle mass spectrum (log scale) ===
    ax_a = fig.add_subplot(gs[0, 0:2])
    particles = {
        r'$\nu_1$': (-1.7, COLORS['gold']),
        r'$\nu_2$': (-1.1, COLORS['gold']),
        r'$\nu_3$': (-0.7, COLORS['gold']),
        'e': (np.log10(0.511), COLORS['blue']),
        r'$\mu$': (np.log10(105.7), COLORS['blue']),
        r'$\tau$': (np.log10(1777), COLORS['blue']),
        'u': (np.log10(2.2), COLORS['green']),
        'd': (np.log10(4.7), COLORS['green']),
        'c': (np.log10(1270), COLORS['green']),
        's': (np.log10(95), COLORS['green']),
        't': (np.log10(173000), COLORS['green']),
        'b': (np.log10(4180), COLORS['green']),
        'W': (np.log10(80400), COLORS['nuclear']),
        'Z': (np.log10(91200), COLORS['nuclear']),
        'H': (np.log10(125000), COLORS['nuclear']),
    }

    y_offset = 0
    for name, (log_mass, color) in particles.items():
        marker_size = 10 if 'nu' in name else 7
        ax_a.plot(log_mass, y_offset % 3, 'o', color=color,
                  markersize=marker_size, alpha=0.8, zorder=5)
        ax_a.text(log_mass, (y_offset % 3) + 0.2, name, fontsize=8,
                  color=color, ha='center', va='bottom')
        y_offset += 1

    # Highlight the neutrino gap
    ax_a.axvspan(-2, -0.5, alpha=0.1, color=COLORS['gold'])
    ax_a.text(-1.3, 2.7, 'NEUTRINOS\n(mysteriously light)',
              fontsize=10, color=COLORS['gold'], ha='center',
              fontweight='bold')

    # Gap annotation
    ax_a.annotate('', xy=(-0.5, -0.3), xytext=(np.log10(0.511), -0.3),
                  arrowprops=dict(arrowstyle='<->', color=COLORS['cosmo'], lw=2))
    ax_a.text(-0.2, -0.6, '10^6 gap', fontsize=9, color=COLORS['cosmo'],
              ha='center', fontweight='bold')

    ax_a.set_xlabel('log10(mass / MeV)', fontsize=10)
    ax_a.set_title('Standard Model Mass Spectrum: Neutrinos Are 10^6 Lighter Than Expected',
                   fontsize=10, color=COLORS['gold'])
    ax_a.set_xlim(-2.5, 6)
    ax_a.set_ylim(-1, 3.5)
    ax_a.set_yticks([])

    # === Panel B: See-saw mechanism ===
    ax_b = fig.add_subplot(gs[0, 2])
    # See-saw diagram: heavy partner ↔ light neutrino
    ax_b.text(0.5, 0.88, 'SEE-SAW MECHANISM', fontsize=11,
              color=COLORS['cosmo'], ha='center', fontweight='bold')
    ax_b.text(0.5, 0.75, r'm_light ~ v^2 / M_heavy', fontsize=10,
              color=COLORS['cosmo'], ha='center')

    # See-saw visual
    # Fulcrum
    ax_b.plot([0.45, 0.5, 0.55], [0.45, 0.5, 0.45], '-',
              color=COLORS['white'], linewidth=2)
    # Beam (tilted)
    ax_b.plot([0.15, 0.85], [0.6, 0.42], '-', color=COLORS['blue'],
              linewidth=3)
    # Heavy ball
    ax_b.plot(0.82, 0.42, 'o', color=COLORS['nuclear'], markersize=20)
    ax_b.text(0.82, 0.30, r'M_heavy ~ 10^14 GeV', fontsize=7,
              color=COLORS['nuclear'], ha='center')
    ax_b.text(0.82, 0.42, 'H', fontsize=10, color='white', ha='center',
              va='center', fontweight='bold')
    # Light ball
    ax_b.plot(0.18, 0.6, 'o', color=COLORS['gold'], markersize=8)
    ax_b.text(0.18, 0.68, r'm_light ~ 0.05 eV', fontsize=7,
              color=COLORS['gold'], ha='center')

    ax_b.text(0.5, 0.18, 'Requires a HEAVY\npartner particle\n(never detected)',
              fontsize=9, color=COLORS['nuclear'], ha='center',
              bbox=dict(boxstyle='round,pad=0.3', facecolor='#f0f0f5',
                        edgecolor=COLORS['nuclear'], alpha=0.8))

    ax_b.set_xlim(0, 1)
    ax_b.set_ylim(0.05, 0.95)
    ax_b.axis('off')
    ax_b.set_title('Standard Explanation:\nSee-Saw (Undetected)', fontsize=10,
                   color=COLORS['cosmo'])

    # === Panel C: Position on fractal landscape ===
    ax_c = fig.add_subplot(gs[1, 0:2])
    # Harmonic landscape
    log_E = np.linspace(-3, 20, 1000)
    # Harmonic response with resonant peaks
    response = 0.1 * np.ones_like(log_E)
    peaks = {
        'Neutrinos': (-1.3, 0.8, 1.5, COLORS['gold']),
        'Electron': (np.log10(0.511), 0.4, 0.5, COLORS['blue']),
        'QCD scale': (np.log10(200), 0.6, 1.0, COLORS['nuclear']),
        'EW scale': (np.log10(100000), 0.5, 0.8, COLORS['cyan']),
        'Planck': (19, 0.7, 1.2, COLORS['quantum']),
    }

    for name, (center, amp, width, color) in peaks.items():
        peak = amp * np.exp(-((log_E - center) / width) ** 2)
        response += peak

    ax_c.fill_between(log_E, 0, response, alpha=0.2, color=COLORS['blue'])
    ax_c.plot(log_E, response, color=COLORS['blue'], linewidth=2)

    # Mark the neutrino position
    for name, (center, amp, width, color) in peaks.items():
        peak_val = amp + 0.1
        ax_c.plot(center, peak_val, 'v', color=color, markersize=10, zorder=5)
        ax_c.text(center, peak_val + 0.1, name, fontsize=8, color=color,
                  ha='center', fontweight='bold')

    # Highlight that neutrinos sit at a NODE (low amplitude region)
    # The key insight: neutrinos are at a harmonic NODE
    ax_c.annotate('Neutrinos sit near\na harmonic NODE\n(naturally small amplitude)',
                  xy=(-1.3, 0.9), xytext=(4, 0.85),
                  fontsize=9, color=COLORS['gold'],
                  arrowprops=dict(arrowstyle='->', color=COLORS['gold'], lw=2),
                  bbox=dict(boxstyle='round,pad=0.3', facecolor='#f0f0f5',
                            edgecolor=COLORS['gold'], alpha=0.8))

    ax_c.set_xlabel('log10(Energy / MeV)', fontsize=10)
    ax_c.set_ylabel('Harmonic Amplitude', fontsize=10)
    ax_c.set_title('Fractal Geometric Harmonic Landscape: Neutrinos at a Node',
                   fontsize=10, color=COLORS['gold'])

    # === Panel D: Resolution ===
    ax_d = fig.add_subplot(gs[1, 2])

    items = [
        ('Problem:', 'Why are neutrinos\nso light?', COLORS['white'], 0.90),
        ('Standard:', 'Heavy partner\n(see-saw)\nnever found', COLORS['cosmo'], 0.72),
        ('Resonance:', 'Neutrino masses\nare harmonic\nnodes on the\nlandscape', COLORS['gold'], 0.48),
        ('Position:', 'Between weak\nand gravitational\nscales --- naturally\nsmall amplitude', COLORS['gold'], 0.22),
    ]
    for label, text, color, y in items:
        ax_d.text(0.03, y, label, fontsize=9, color=color,
                  fontweight='bold', va='top')
        ax_d.text(0.33, y, text, fontsize=9, color=color, va='top')

    ax_d.text(0.5, 0.03, 'No heavy partner needed.\nSmallness = position on landscape.',
              fontsize=9, color=COLORS['gold'], ha='center',
              bbox=dict(boxstyle='round,pad=0.3', facecolor='#f0f0f5',
                        edgecolor=COLORS['gold'], alpha=0.8))

    ax_d.set_xlim(0, 1)
    ax_d.set_ylim(0, 1)
    ax_d.axis('off')
    ax_d.set_title('Resolution', fontsize=10, color=COLORS['gold'])

    fig.text(0.5, 0.01,
             'Neutrino masses are not mysteriously small. They sit at a harmonic node '
             'on the fractal geometric landscape. No heavy partner needed.',
             ha='center', fontsize=11, color=COLORS['gold'], style='italic')

    path = os.path.join(BASE_DIR, 'p3_fig09_neutrino_masses.png')
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("PAPER THREE --- PARTICLES & MASSES (Figures 7-9)")
    print("=" * 60)

    print("\nFigure 7: Matter-Antimatter Asymmetry...")
    figure_7()

    print("\nFigure 8: Strong CP Problem...")
    figure_8()

    print("\nFigure 9: Neutrino Masses...")
    figure_9()

    print("\n" + "=" * 60)
    print("ALL THREE FIGURES COMPLETE")
    print("=" * 60)
