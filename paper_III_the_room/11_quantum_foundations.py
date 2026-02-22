"""
Paper Three - Resonance Theory III: The Room Is Larger Than We Thought
Script 11: Quantum Foundations Figures (1-3)

Figure 1: Wave function "collapse" as harmonic phase transition
Figure 2: Fractal topology vs Euclidean distance — entanglement
Figure 3: Bell inequality in Euclidean vs fractal topology
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle, Arc
from matplotlib.collections import LineCollection
import matplotlib.gridspec as gridspec
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# Global style
# ============================================================
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
# FIGURE 1: Wave Function Collapse as Harmonic Phase Transition
# ============================================================
def figure_1() -> None:
    """
    Shows that 'collapse' is a harmonic phase transition —
    the SAME mathematics at quantum, nuclear, and cosmological scales.
    """
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(
        'Figure 1: Wave Function "Collapse" as Harmonic Phase Transition',
        fontsize=16, fontweight='bold', color=COLORS['gold'], y=0.97
    )

    gs = gridspec.GridSpec(3, 3, hspace=0.45, wspace=0.35,
                           left=0.08, right=0.95, top=0.91, bottom=0.06)

    # === Row 1: Quantum Scale — Superposition to single state ===
    # Panel A: Superposition (multiple harmonic modes)
    ax1a = fig.add_subplot(gs[0, 0])
    x = np.linspace(0, 4 * np.pi, 500)
    modes = [
        (1.0, 1, '#9b59b6'),
        (0.7, 2, '#3498db'),
        (0.5, 3, '#2ecc71'),
        (0.3, 5, '#e74c3c'),
    ]
    superposition = np.zeros_like(x)
    for amp, freq, color in modes:
        y = amp * np.sin(freq * x)
        ax1a.plot(x, y, color=color, alpha=0.4, linewidth=1)
        superposition += y
    ax1a.plot(x, superposition, color=COLORS['white'], linewidth=2, alpha=0.9)
    ax1a.set_title('Quantum: Superposition\n(Multiple Harmonic Modes)', fontsize=9,
                   color=COLORS['quantum'])
    ax1a.set_xlabel('Position', fontsize=8)
    ax1a.set_ylabel('ψ(x)', fontsize=8)
    ax1a.set_ylim(-3, 3)

    # Panel B: Phase transition (arrow showing transition)
    ax1b = fig.add_subplot(gs[0, 1])
    t = np.linspace(0, 1, 200)
    # Start with superposition, end with single mode
    for i, frac in enumerate(np.linspace(0, 1, 8)):
        y_start = sum(amp * np.sin(freq * x) for amp, freq, _ in modes)
        y_end = 2.0 * np.sin(x)
        y = (1 - frac) * y_start + frac * y_end
        alpha = 0.15 + 0.1 * frac
        color_val = int(155 + 100 * frac)
        ax1b.plot(x, y, color=f'#{color_val:02x}89{255 - color_val:02x}',
                  alpha=alpha, linewidth=1)
    ax1b.plot(x, 2.0 * np.sin(x), color=COLORS['gold'], linewidth=2.5)
    ax1b.annotate('PHASE\nTRANSITION', xy=(6, 0), fontsize=12,
                  color=COLORS['gold'], fontweight='bold',
                  ha='center', va='center',
                  bbox=dict(boxstyle='round,pad=0.3', facecolor='#f0f0f5',
                            edgecolor=COLORS['gold'], alpha=0.8))
    ax1b.set_title('Measurement = Scale Interaction\n→ Harmonic Phase Transition', fontsize=9,
                   color=COLORS['gold'])
    ax1b.set_ylim(-3, 3)

    # Panel C: Single mode (collapsed state)
    ax1c = fig.add_subplot(gs[0, 2])
    y_final = 2.0 * np.sin(x)
    ax1c.fill_between(x, 0, y_final, alpha=0.3, color=COLORS['quantum'])
    ax1c.plot(x, y_final, color=COLORS['quantum'], linewidth=2.5)
    ax1c.set_title('"Collapsed" State\n(Single Harmonic Mode)', fontsize=9,
                   color=COLORS['quantum'])
    ax1c.set_xlabel('Position', fontsize=8)
    ax1c.set_ylabel('ψ(x)', fontsize=8)
    ax1c.set_ylim(-3, 3)

    # === Row 2: Nuclear Scale — QCD Confinement Transition ===
    ax2a = fig.add_subplot(gs[1, 0])
    T = np.linspace(100, 250, 300)
    Tc = 155.0
    # Order parameter (chiral condensate)
    order = np.where(T < Tc, 1.0 - 0.3 * (T / Tc) ** 2,
                     0.7 * np.exp(-3 * (T - Tc) / Tc))
    ax2a.plot(T, order, color=COLORS['nuclear'], linewidth=2.5)
    ax2a.axvline(Tc, color=COLORS['gold'], linestyle='--', alpha=0.7)
    ax2a.annotate('T_c ≈ 155 MeV', xy=(Tc, 0.5), fontsize=9,
                  color=COLORS['gold'], ha='left',
                  xytext=(Tc + 5, 0.6),
                  arrowprops=dict(arrowstyle='->', color=COLORS['gold']))
    ax2a.set_title('Nuclear: Confinement\n(Order Parameter)', fontsize=9,
                   color=COLORS['nuclear'])
    ax2a.set_xlabel('Temperature (MeV)', fontsize=8)
    ax2a.set_ylabel('⟨q̄q⟩ / ⟨q̄q⟩₀', fontsize=8)

    ax2b = fig.add_subplot(gs[1, 1])
    # Susceptibility peak at Tc
    chi = 1.0 / ((T - Tc) ** 2 + 5 ** 2)
    chi /= chi.max()
    ax2b.fill_between(T, 0, chi, alpha=0.3, color=COLORS['nuclear'])
    ax2b.plot(T, chi, color=COLORS['nuclear'], linewidth=2.5)
    ax2b.axvline(Tc, color=COLORS['gold'], linestyle='--', alpha=0.7)
    ax2b.annotate('PHASE\nTRANSITION', xy=(Tc, 0.8), fontsize=12,
                  color=COLORS['gold'], fontweight='bold',
                  ha='center', va='center',
                  bbox=dict(boxstyle='round,pad=0.3', facecolor='#f0f0f5',
                            edgecolor=COLORS['gold'], alpha=0.8))
    ax2b.set_title('QCD Phase Transition\n→ Same Mathematics', fontsize=9,
                   color=COLORS['gold'])
    ax2b.set_xlabel('Temperature (MeV)', fontsize=8)
    ax2b.set_ylabel('Susceptibility χ', fontsize=8)

    ax2c = fig.add_subplot(gs[1, 2])
    # Two phases: confined (hadrons) vs deconfined (QGP)
    theta = np.linspace(0, 2 * np.pi, 100)
    # Confined — tight bound state
    r_conf = 0.3 + 0.05 * np.sin(6 * theta)
    ax2c.plot(0.7 * r_conf * np.cos(theta) - 0.5,
              0.7 * r_conf * np.sin(theta), color=COLORS['nuclear'],
              linewidth=2)
    ax2c.text(-0.5, -0.55, 'Confined\n(Hadrons)', fontsize=8,
              color=COLORS['nuclear'], ha='center')
    # Deconfined — free quarks
    for angle in [0, 2 * np.pi / 3, 4 * np.pi / 3]:
        cx = 0.5 + 0.2 * np.cos(angle)
        cy = 0.2 * np.sin(angle)
        circle = plt.Circle((cx, cy), 0.08, fill=True,
                            facecolor=COLORS['nuclear'], alpha=0.6)
        ax2c.add_patch(circle)
    ax2c.text(0.5, -0.55, 'Deconfined\n(QGP)', fontsize=8,
              color=COLORS['nuclear'], ha='center')
    ax2c.set_xlim(-1, 1)
    ax2c.set_ylim(-0.7, 0.7)
    ax2c.set_aspect('equal')
    ax2c.set_title('Confinement → Deconfinement\n(Single Mode Selection)', fontsize=9,
                   color=COLORS['nuclear'])
    ax2c.axis('off')

    # === Row 3: Cosmological Scale — Dark Energy Phase Transition ===
    ax3a = fig.add_subplot(gs[2, 0])
    z = np.linspace(0, 3, 300)
    a = 1.0 / (1.0 + z)
    # Deceleration parameter
    Om = 0.3
    OL = 0.7
    q_LCDM = 0.5 * Om * (1 + z) ** 3 / (Om * (1 + z) ** 3 + OL) - OL / (Om * (1 + z) ** 3 + OL)
    ax3a.plot(z, q_LCDM, color=COLORS['cosmo'], linewidth=2.5)
    ax3a.axhline(0, color=COLORS['white'], linestyle=':', alpha=0.3)
    z_trans = z[np.argmin(np.abs(q_LCDM))]
    ax3a.axvline(z_trans, color=COLORS['gold'], linestyle='--', alpha=0.7)
    ax3a.annotate(f'z_t ≈ {z_trans:.1f}', xy=(z_trans, 0), fontsize=9,
                  color=COLORS['gold'], ha='left',
                  xytext=(z_trans + 0.2, 0.15),
                  arrowprops=dict(arrowstyle='->', color=COLORS['gold']))
    ax3a.set_title('Cosmological: Deceleration → Acceleration\n(Order Parameter q(z))',
                   fontsize=9, color=COLORS['cosmo'])
    ax3a.set_xlabel('Redshift z', fontsize=8)
    ax3a.set_ylabel('Deceleration q(z)', fontsize=8)
    ax3a.set_ylim(-1, 0.6)
    ax3a.text(2.5, 0.4, 'Decelerating', fontsize=8, color=COLORS['cosmo'], ha='center')
    ax3a.text(0.2, -0.7, 'Accelerating', fontsize=8, color=COLORS['cosmo'], ha='center')

    ax3b = fig.add_subplot(gs[2, 1])
    # Response function (like susceptibility) at cosmological transition
    response = 1.0 / ((z - z_trans) ** 2 + 0.1 ** 2)
    response /= response.max()
    ax3b.fill_between(z, 0, response, alpha=0.3, color=COLORS['cosmo'])
    ax3b.plot(z, response, color=COLORS['cosmo'], linewidth=2.5)
    ax3b.axvline(z_trans, color=COLORS['gold'], linestyle='--', alpha=0.7)
    ax3b.annotate('PHASE\nTRANSITION', xy=(z_trans, 0.8), fontsize=12,
                  color=COLORS['gold'], fontweight='bold',
                  ha='center', va='center',
                  bbox=dict(boxstyle='round,pad=0.3', facecolor='#f0f0f5',
                            edgecolor=COLORS['gold'], alpha=0.8))
    ax3b.set_title('Cosmic Phase Transition\n→ Same Mathematics', fontsize=9,
                   color=COLORS['gold'])
    ax3b.set_xlabel('Redshift z', fontsize=8)
    ax3b.set_ylabel('Transition Response', fontsize=8)

    ax3c = fig.add_subplot(gs[2, 2])
    # Summary: three scales, same math
    scales = [
        ('QUANTUM\n10⁻¹⁵ m', COLORS['quantum'], 0.8),
        ('NUCLEAR\n10⁻¹⁵ m', COLORS['nuclear'], 0.5),
        ('COSMOLOGICAL\n10²⁶ m', COLORS['cosmo'], 0.2),
    ]
    for label, color, y_pos in scales:
        ax3c.annotate(label, xy=(0.5, y_pos), fontsize=11,
                      color=color, fontweight='bold', ha='center', va='center')

    ax3c.annotate('', xy=(0.5, 0.65), xytext=(0.5, 0.73),
                  arrowprops=dict(arrowstyle='<->', color=COLORS['gold'], lw=2))
    ax3c.annotate('', xy=(0.5, 0.35), xytext=(0.5, 0.43),
                  arrowprops=dict(arrowstyle='<->', color=COLORS['gold'], lw=2))

    ax3c.text(0.5, 0.95, 'THREE SCALES', fontsize=13, color=COLORS['gold'],
              fontweight='bold', ha='center', va='center')
    ax3c.text(0.5, 0.05, 'ONE MATHEMATICS', fontsize=13, color=COLORS['gold'],
              fontweight='bold', ha='center', va='center')
    ax3c.text(0.82, 0.69, 'Same\nmath', fontsize=8, color=COLORS['gold'],
              ha='center', style='italic')
    ax3c.text(0.82, 0.39, 'Same\nmath', fontsize=8, color=COLORS['gold'],
              ha='center', style='italic')
    ax3c.set_xlim(0, 1)
    ax3c.set_ylim(0, 1)
    ax3c.axis('off')
    ax3c.set_title('Harmonic Phase Transition\nAt Every Scale', fontsize=9,
                   color=COLORS['gold'])

    # Bottom annotation
    fig.text(0.5, 0.01,
             '"Collapse" is not collapse. It is a harmonic phase transition — '
             'the same mathematics at every scale.',
             ha='center', fontsize=11, color=COLORS['gold'], style='italic')

    path = os.path.join(BASE_DIR, 'p3_fig01_measurement_problem.png')
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================
# FIGURE 2: Fractal Topology vs Euclidean Distance — Entanglement
# ============================================================
def figure_2() -> None:
    """
    Shows entangled particles as adjacent in fractal topology
    but distant in Euclidean projection.
    """
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        'Figure 2: Quantum Entanglement — Adjacent in Fractal Topology',
        fontsize=16, fontweight='bold', color=COLORS['gold'], y=0.97
    )

    gs = gridspec.GridSpec(2, 3, hspace=0.4, wspace=0.35,
                           left=0.06, right=0.96, top=0.90, bottom=0.08)

    # === Panel A: Koch-like fractal curve showing topology ===
    ax_a = fig.add_subplot(gs[0, 0])

    # Generate a simplified fractal path
    def koch_segment(p1: np.ndarray, p2: np.ndarray, depth: int) -> list:
        if depth == 0:
            return [p1, p2]
        d = p2 - p1
        a = p1 + d / 3
        b = p1 + 2 * d / 3
        # Rotate d/3 by 60 degrees for the peak
        angle = np.pi / 3
        peak = a + np.array([
            d[0] / 3 * np.cos(angle) - d[1] / 3 * np.sin(angle),
            d[0] / 3 * np.sin(angle) + d[1] / 3 * np.cos(angle)
        ])
        pts1 = koch_segment(p1, a, depth - 1)
        pts2 = koch_segment(a, peak, depth - 1)
        pts3 = koch_segment(peak, b, depth - 1)
        pts4 = koch_segment(b, p2, depth - 1)
        return pts1 + pts2[1:] + pts3[1:] + pts4[1:]

    koch_pts = koch_segment(np.array([0.0, 0.0]), np.array([1.0, 0.0]), 3)
    koch_arr = np.array(koch_pts)
    ax_a.plot(koch_arr[:, 0], koch_arr[:, 1], color=COLORS['blue'], linewidth=1.5, alpha=0.8)

    # Mark two points that are close in embedding but far along curve
    idx_a = len(koch_arr) // 6
    idx_b = len(koch_arr) * 5 // 6
    pt_a = koch_arr[idx_a]
    pt_b = koch_arr[idx_b]

    ax_a.plot(*pt_a, 'o', color=COLORS['quantum'], markersize=12, zorder=5)
    ax_a.plot(*pt_b, 'o', color=COLORS['nuclear'], markersize=12, zorder=5)
    ax_a.annotate('A', xy=pt_a, fontsize=10, color='white', fontweight='bold',
                  ha='center', va='center', zorder=6)
    ax_a.annotate('B', xy=pt_b, fontsize=10, color='white', fontweight='bold',
                  ha='center', va='center', zorder=6)

    # Draw Euclidean distance
    mid = (pt_a + pt_b) / 2
    euclid_dist = np.linalg.norm(pt_b - pt_a)
    ax_a.annotate('', xy=pt_b, xytext=pt_a,
                  arrowprops=dict(arrowstyle='<->', color=COLORS['cosmo'],
                                 lw=2, linestyle='--'))
    ax_a.text(mid[0], mid[1] - 0.08, f'Euclidean: {euclid_dist:.2f}',
              fontsize=8, color=COLORS['cosmo'], ha='center')

    ax_a.set_title('Fractal Curve:\nFar Along Path, Close in Space', fontsize=10,
                   color=COLORS['blue'])
    ax_a.set_aspect('equal')
    ax_a.axis('off')

    # === Panel B: Spacetime fractal topology ===
    ax_b = fig.add_subplot(gs[0, 1])

    # Draw a warped "fractal spacetime" with two entangled particles
    theta = np.linspace(0, 2 * np.pi, 500)
    # Nested spiraling structure suggesting fractal topology
    for i, r_base in enumerate([0.9, 0.6, 0.3]):
        n_loops = 3 + i * 2
        r = r_base + 0.08 * np.sin(n_loops * theta)
        x_spiral = r * np.cos(theta)
        y_spiral = r * np.sin(theta)
        ax_b.plot(x_spiral, y_spiral, color=COLORS['blue'],
                  alpha=0.2 + 0.1 * i, linewidth=1)

    # Two particles — "far apart" in Euclidean but connected through fractal topology
    p1 = np.array([-0.8, 0.3])
    p2 = np.array([0.8, -0.3])
    ax_b.plot(*p1, 'o', color=COLORS['quantum'], markersize=15, zorder=5)
    ax_b.plot(*p2, 'o', color=COLORS['nuclear'], markersize=15, zorder=5)
    ax_b.text(p1[0], p1[1] + 0.15, 'Particle A', fontsize=9,
              color=COLORS['quantum'], ha='center', fontweight='bold')
    ax_b.text(p2[0], p2[1] - 0.15, 'Particle B', fontsize=9,
              color=COLORS['nuclear'], ha='center', fontweight='bold')

    # Euclidean distance line (dashed)
    ax_b.plot([p1[0], p2[0]], [p1[1], p2[1]], '--',
              color=COLORS['cosmo'], linewidth=2, alpha=0.5)
    ax_b.text(0, 0.15, 'Euclidean:\nFAR APART', fontsize=9,
              color=COLORS['cosmo'], ha='center', fontweight='bold')

    # Fractal topology connection (through the center - short path)
    t_conn = np.linspace(0, 1, 50)
    x_conn = p1[0] + (p2[0] - p1[0]) * t_conn + 0.3 * np.sin(np.pi * t_conn)
    y_conn = p1[1] + (p2[1] - p1[1]) * t_conn + 0.3 * np.sin(2 * np.pi * t_conn)
    ax_b.plot(x_conn, y_conn, color=COLORS['gold'], linewidth=3, alpha=0.8)
    ax_b.text(0.35, -0.4, 'Fractal:\nADJACENT', fontsize=9,
              color=COLORS['gold'], ha='center', fontweight='bold')

    ax_b.set_xlim(-1.2, 1.2)
    ax_b.set_ylim(-1.2, 1.2)
    ax_b.set_aspect('equal')
    ax_b.set_title('Fractal Spacetime Topology:\nEntangled = Adjacent', fontsize=10,
                   color=COLORS['gold'])
    ax_b.axis('off')

    # === Panel C: Comparison diagram ===
    ax_c = fig.add_subplot(gs[0, 2])

    # Euclidean view
    ax_c.text(0.5, 0.92, 'EUCLIDEAN VIEW', fontsize=11, color=COLORS['cosmo'],
              ha='center', fontweight='bold')
    ax_c.plot([0.15, 0.85], [0.78, 0.78], '--', color=COLORS['cosmo'],
              linewidth=2, alpha=0.7)
    ax_c.plot(0.15, 0.78, 'o', color=COLORS['quantum'], markersize=14, zorder=5)
    ax_c.plot(0.85, 0.78, 'o', color=COLORS['nuclear'], markersize=14, zorder=5)
    ax_c.text(0.5, 0.72, '← FAR APART →', fontsize=9, color=COLORS['cosmo'],
              ha='center')
    ax_c.text(0.5, 0.65, '"Spooky action at a distance"', fontsize=8,
              color=COLORS['cosmo'], ha='center', style='italic')

    # Fractal view
    ax_c.text(0.5, 0.52, 'FRACTAL VIEW', fontsize=11, color=COLORS['gold'],
              ha='center', fontweight='bold')
    ax_c.plot(0.45, 0.38, 'o', color=COLORS['quantum'], markersize=14, zorder=5)
    ax_c.plot(0.55, 0.38, 'o', color=COLORS['nuclear'], markersize=14, zorder=5)
    ax_c.plot([0.45, 0.55], [0.38, 0.38], '-', color=COLORS['gold'],
              linewidth=3)
    ax_c.text(0.5, 0.31, 'ADJACENT', fontsize=9, color=COLORS['gold'],
              ha='center', fontweight='bold')
    ax_c.text(0.5, 0.24, '"Obvious action at no distance"', fontsize=8,
              color=COLORS['gold'], ha='center', style='italic')

    # Arrow connecting the two views
    ax_c.annotate('', xy=(0.5, 0.55), xytext=(0.5, 0.62),
                  arrowprops=dict(arrowstyle='->', color=COLORS['white'],
                                  lw=2))
    ax_c.text(0.78, 0.58, 'Same\nparticles', fontsize=8, color=COLORS['white'],
              ha='center', style='italic')

    # Bottom conclusion
    ax_c.text(0.5, 0.08, 'Distance is a\nEuclidean illusion', fontsize=11,
              color=COLORS['gold'], ha='center', fontweight='bold',
              bbox=dict(boxstyle='round,pad=0.4', facecolor='#f0f0f5',
                        edgecolor=COLORS['gold'], alpha=0.8))

    ax_c.set_xlim(0, 1)
    ax_c.set_ylim(0, 1)
    ax_c.axis('off')

    # === Row 2: Correlation demonstration ===
    # Panel D: EPR correlation curve
    ax_d = fig.add_subplot(gs[1, 0])
    theta_angles = np.linspace(0, 2 * np.pi, 500)
    # Quantum correlation: -cos(theta)
    E_QM = -np.cos(theta_angles)
    # Classical limit
    E_classical = np.where(theta_angles < np.pi,
                            1 - 2 * theta_angles / np.pi,
                            -1 + 2 * (theta_angles - np.pi) / np.pi)
    ax_d.plot(np.degrees(theta_angles), E_QM, color=COLORS['quantum'],
              linewidth=2.5, label='Quantum (measured)')
    ax_d.plot(np.degrees(theta_angles), E_classical, '--',
              color=COLORS['cosmo'], linewidth=2, label='Local hidden variable')
    ax_d.fill_between(np.degrees(theta_angles), E_classical, E_QM,
                      where=(E_QM < E_classical), alpha=0.2,
                      color=COLORS['quantum'])
    ax_d.set_xlabel('Measurement Angle θ (degrees)', fontsize=9)
    ax_d.set_ylabel('Correlation E(θ)', fontsize=9)
    ax_d.set_title('EPR Correlations:\nQuantum Exceeds Classical', fontsize=10,
                   color=COLORS['quantum'])
    ax_d.legend(fontsize=8, loc='upper right',
                facecolor='#f0f0f5', edgecolor='#999999', labelcolor='#222222')
    ax_d.axhline(0, color=COLORS['white'], alpha=0.2, linewidth=0.5)

    # Panel E: Fractal dimension vs scale — showing topology change
    ax_e = fig.add_subplot(gs[1, 1])
    log_scale = np.linspace(-35, 27, 500)
    # Effective fractal dimension — varies with scale
    D_eff = 3.0 + 0.5 * np.tanh((log_scale + 5) / 10) - 0.3 * np.exp(-((log_scale - 0) / 5) ** 2)
    ax_e.plot(log_scale, D_eff, color=COLORS['blue'], linewidth=2.5)
    ax_e.axhline(3.0, color=COLORS['white'], linestyle=':', alpha=0.3, linewidth=1)
    ax_e.fill_between(log_scale, 3.0, D_eff, where=(D_eff < 3.0),
                      alpha=0.3, color=COLORS['quantum'])
    ax_e.fill_between(log_scale, 3.0, D_eff, where=(D_eff > 3.0),
                      alpha=0.3, color=COLORS['cosmo'])
    ax_e.set_xlabel('log₁₀(Scale / meters)', fontsize=9)
    ax_e.set_ylabel('Effective Topological Dimension', fontsize=9)
    ax_e.set_title('Fractal Topology ≠ Euclidean\nDimension Varies with Scale', fontsize=10,
                   color=COLORS['blue'])
    ax_e.text(-30, 2.75, 'D < 3:\nShortcuts\nexist', fontsize=8,
              color=COLORS['quantum'], ha='center')
    ax_e.text(20, 3.35, 'D > 3:\nExpanded\ntopology', fontsize=8,
              color=COLORS['cosmo'], ha='center')

    # Panel F: Summary
    ax_f = fig.add_subplot(gs[1, 2])
    summary_text = [
        ("Einstein:", '"Spooky action\nat a distance"', COLORS['cosmo'], 0.82),
        ("Resonance\nTheory:", '"Obvious action\nat no distance"', COLORS['gold'], 0.52),
    ]
    for label, quote, color, y in summary_text:
        ax_f.text(0.15, y, label, fontsize=11, color=color,
                  fontweight='bold', va='center')
        ax_f.text(0.55, y, quote, fontsize=11, color=color,
                  va='center', style='italic')

    ax_f.text(0.5, 0.22, 'Entangled particles are not\ncommunicating across distance.\n'
              'They are ONE harmonic mode\nat TWO points that are\nADJACENT in fractal topology.',
              fontsize=10, color=COLORS['white'], ha='center', va='center',
              bbox=dict(boxstyle='round,pad=0.5', facecolor='#f0f0f5',
                        edgecolor=COLORS['gold'], alpha=0.8))
    ax_f.set_xlim(0, 1)
    ax_f.set_ylim(0, 1)
    ax_f.axis('off')

    fig.text(0.5, 0.01,
             'Spooky action at a distance becomes obvious action at no distance '
             'when the topology is fractal geometric.',
             ha='center', fontsize=11, color=COLORS['gold'], style='italic')

    path = os.path.join(BASE_DIR, 'p3_fig02_entanglement.png')
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================
# FIGURE 3: Bell Inequality in Euclidean vs Fractal Topology
# ============================================================
def figure_3() -> None:
    """
    Shows that Bell's theorem correlation is LOCAL in fractal space.
    """
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        'Figure 3: Bell\'s Theorem — Local in Fractal Topology',
        fontsize=16, fontweight='bold', color=COLORS['gold'], y=0.97
    )

    gs = gridspec.GridSpec(2, 3, hspace=0.45, wspace=0.35,
                           left=0.08, right=0.95, top=0.90, bottom=0.07)

    # === Panel A: CHSH inequality ===
    ax_a = fig.add_subplot(gs[0, 0])
    # S parameter for different theories
    theories = {
        'Local Hidden\nVariable': (2.0, COLORS['cosmo'], '--'),
        'Quantum\nMechanics': (2 * np.sqrt(2), COLORS['quantum'], '-'),
        'Experimental\nResult': (2.7, COLORS['green'], ':'),
    }
    y_pos = 0.75
    for name, (val, color, ls) in theories.items():
        ax_a.barh(y_pos, val, height=0.12, color=color, alpha=0.6)
        ax_a.text(val + 0.05, y_pos, f'S = {val:.2f}', fontsize=9,
                  color=color, va='center', fontweight='bold')
        ax_a.text(-0.1, y_pos, name, fontsize=8, color=color,
                  va='center', ha='right')
        y_pos -= 0.25

    ax_a.axvline(2.0, color=COLORS['cosmo'], linestyle='--', alpha=0.5)
    ax_a.text(2.0, 0.95, 'Bell Limit\nS ≤ 2', fontsize=8,
              color=COLORS['cosmo'], ha='center')
    ax_a.set_xlim(-0.1, 3.2)
    ax_a.set_ylim(0, 1)
    ax_a.set_xlabel('CHSH Parameter S', fontsize=9)
    ax_a.set_title('Bell/CHSH Inequality:\nQuantum Violates Classical Bound', fontsize=10,
                   color=COLORS['quantum'])
    ax_a.set_yticks([])

    # === Panel B: Euclidean locality ===
    ax_b = fig.add_subplot(gs[0, 1])
    # Light cone diagram
    # Time axis vertical, space horizontal
    t = np.linspace(0, 1, 100)
    ax_b.fill_between(t, t, -t + 0, alpha=0.1, color=COLORS['cosmo'])
    ax_b.plot(t, t, color=COLORS['cosmo'], linewidth=1.5, alpha=0.7)
    ax_b.plot(t, -t, color=COLORS['cosmo'], linewidth=1.5, alpha=0.7)
    ax_b.plot(-t, t, color=COLORS['cosmo'], linewidth=1.5, alpha=0.7)
    ax_b.plot(-t, -t, color=COLORS['cosmo'], linewidth=1.5, alpha=0.7)

    # Two spacelike-separated events
    ax_b.plot(-0.7, 0.2, 'o', color=COLORS['quantum'], markersize=14, zorder=5)
    ax_b.plot(0.7, 0.2, 'o', color=COLORS['nuclear'], markersize=14, zorder=5)
    ax_b.text(-0.7, 0.35, 'A', fontsize=12, color=COLORS['quantum'],
              ha='center', fontweight='bold')
    ax_b.text(0.7, 0.35, 'B', fontsize=12, color=COLORS['nuclear'],
              ha='center', fontweight='bold')
    ax_b.plot([-0.7, 0.7], [0.2, 0.2], '--', color=COLORS['white'],
              alpha=0.5, linewidth=1)
    ax_b.text(0, 0.08, 'Spacelike separated\n(No causal contact)', fontsize=8,
              color=COLORS['white'], ha='center')

    ax_b.set_xlabel('Space', fontsize=9)
    ax_b.set_ylabel('Time', fontsize=9)
    ax_b.set_title('Euclidean Topology:\nA and B are NON-LOCAL', fontsize=10,
                   color=COLORS['cosmo'])
    ax_b.set_xlim(-1, 1)
    ax_b.set_ylim(-0.5, 1)
    ax_b.text(0, -0.35, 'Bell says: no local\nhidden variables', fontsize=8,
              color=COLORS['cosmo'], ha='center',
              bbox=dict(boxstyle='round,pad=0.3', facecolor='#f0f0f5',
                        edgecolor=COLORS['cosmo'], alpha=0.8))

    # === Panel C: Fractal locality ===
    ax_c = fig.add_subplot(gs[0, 2])
    # In fractal topology, A and B are adjacent
    theta = np.linspace(0, 2 * np.pi, 200)

    # Draw fractal-like structure around both particles
    for r in [0.2, 0.35, 0.5, 0.65, 0.8]:
        n = int(3 + r * 10)
        r_mod = r + 0.03 * np.sin(n * theta)
        ax_c.plot(r_mod * np.cos(theta), r_mod * np.sin(theta),
                  color=COLORS['blue'], alpha=0.15, linewidth=0.8)

    # Particles ADJACENT in fractal topology
    ax_c.plot(-0.08, 0, 'o', color=COLORS['quantum'], markersize=16, zorder=5)
    ax_c.plot(0.08, 0, 'o', color=COLORS['nuclear'], markersize=16, zorder=5)
    ax_c.text(-0.08, 0, 'A', fontsize=10, color='white', ha='center',
              va='center', fontweight='bold', zorder=6)
    ax_c.text(0.08, 0, 'B', fontsize=10, color='white', ha='center',
              va='center', fontweight='bold', zorder=6)

    # Gold connection
    ax_c.plot([-0.08, 0.08], [0, 0], '-', color=COLORS['gold'],
              linewidth=4, zorder=4)
    ax_c.text(0, -0.15, 'ADJACENT', fontsize=10, color=COLORS['gold'],
              ha='center', fontweight='bold')

    ax_c.set_title('Fractal Topology:\nA and B are LOCAL', fontsize=10,
                   color=COLORS['gold'])
    ax_c.set_xlim(-1, 1)
    ax_c.set_ylim(-1, 1)
    ax_c.set_aspect('equal')
    ax_c.text(0, -0.55, 'Bell\'s locality condition\nis SATISFIED', fontsize=8,
              color=COLORS['gold'], ha='center',
              bbox=dict(boxstyle='round,pad=0.3', facecolor='#f0f0f5',
                        edgecolor=COLORS['gold'], alpha=0.8))
    ax_c.axis('off')

    # === Row 2 ===
    # Panel D: S parameter vs angle showing quantum > classical
    ax_d = fig.add_subplot(gs[1, 0])
    phi = np.linspace(0, np.pi, 200)
    # For optimal CHSH angles
    S_qm = 2 * np.sqrt(2) * np.abs(np.cos(phi))
    S_local = 2 * np.abs(np.cos(phi))

    ax_d.plot(np.degrees(phi), S_qm, color=COLORS['quantum'], linewidth=2.5,
              label='Quantum (fractal local)')
    ax_d.plot(np.degrees(phi), S_local, '--', color=COLORS['cosmo'],
              linewidth=2, label='Euclidean local')
    ax_d.axhline(2.0, color=COLORS['cosmo'], linestyle=':', alpha=0.5)
    ax_d.fill_between(np.degrees(phi), S_local, S_qm,
                      where=(S_qm > S_local), alpha=0.15, color=COLORS['quantum'])
    ax_d.text(45, 2.5, 'Quantum\nexceeds\nclassical', fontsize=8,
              color=COLORS['quantum'], ha='center')
    ax_d.set_xlabel('Measurement Angle (degrees)', fontsize=9)
    ax_d.set_ylabel('CHSH Parameter S', fontsize=9)
    ax_d.set_title('CHSH: Quantum Exceeds\nEuclidean-Local Bound', fontsize=10,
                   color=COLORS['quantum'])
    ax_d.legend(fontsize=8, loc='upper right',
                facecolor='#f0f0f5', edgecolor='#999999', labelcolor='#222222')

    # Panel E: Topology diagram explaining the resolution
    ax_e = fig.add_subplot(gs[1, 1])

    # Old picture vs new picture
    rows = [
        ('Bell\'s Theorem:', '', COLORS['white'], 0.92, True),
        ('No LOCAL hidden', '', COLORS['cosmo'], 0.82, False),
        ('variables (in', '', COLORS['cosmo'], 0.74, False),
        ('Euclidean topology)', '', COLORS['cosmo'], 0.66, False),
        ('', '', COLORS['white'], 0.56, False),
        ('Resonance Theory:', '', COLORS['gold'], 0.48, True),
        ('The hidden variable', '', COLORS['gold'], 0.38, False),
        ('IS the fractal', '', COLORS['gold'], 0.30, False),
        ('geometric structure', '', COLORS['gold'], 0.22, False),
        ('— not hidden, just', '', COLORS['gold'], 0.14, False),
        ('unrecognized', '', COLORS['gold'], 0.06, False),
    ]
    for text, _, color, y, bold in rows:
        ax_e.text(0.5, y, text, fontsize=10, color=color, ha='center',
                  fontweight='bold' if bold else 'normal')

    ax_e.set_xlim(0, 1)
    ax_e.set_ylim(0, 1)
    ax_e.axis('off')
    ax_e.set_title('The Resolution', fontsize=10, color=COLORS['gold'])

    # Panel F: Summary table
    ax_f = fig.add_subplot(gs[1, 2])

    table_data = [
        ('Concept', 'Euclidean', 'Fractal', 0.90, True),
        ('Topology', 'Integer dim', 'Fractal dim', 0.78, False),
        ('Locality', 'Light cone', 'Fractal adjacency', 0.66, False),
        ('Bell limit', 'S ≤ 2', 'S ≤ 2√2', 0.54, False),
        ('Hidden var', 'Ruled out', 'IS the structure', 0.42, False),
        ('Entangle-\nment', '"Spooky"', 'Natural', 0.28, False),
    ]
    for concept, euclid, fractal, y, is_header in table_data:
        weight = 'bold' if is_header else 'normal'
        ax_f.text(0.05, y, concept, fontsize=9, color=COLORS['white'],
                  va='center', fontweight=weight)
        ax_f.text(0.40, y, euclid, fontsize=9, color=COLORS['cosmo'],
                  va='center', fontweight=weight)
        ax_f.text(0.72, y, fractal, fontsize=9, color=COLORS['gold'],
                  va='center', fontweight=weight)
    # Header underline
    ax_f.plot([0.02, 0.98], [0.85, 0.85], color=COLORS['white'],
              alpha=0.3, linewidth=1)

    ax_f.text(0.5, 0.08, 'Bell\'s theorem is not violated.\n'
              'The locality condition is satisfied\nin the CORRECT topology.',
              fontsize=9, color=COLORS['gold'], ha='center',
              bbox=dict(boxstyle='round,pad=0.4', facecolor='#f0f0f5',
                        edgecolor=COLORS['gold'], alpha=0.8))
    ax_f.set_xlim(0, 1)
    ax_f.set_ylim(0, 1)
    ax_f.axis('off')
    ax_f.set_title('Euclidean vs Fractal', fontsize=10, color=COLORS['gold'])

    fig.text(0.5, 0.01,
             'Bell\'s theorem proves no Euclidean-local hidden variables. '
             'The hidden variable is the fractal geometric structure itself.',
             ha='center', fontsize=11, color=COLORS['gold'], style='italic')

    path = os.path.join(BASE_DIR, 'p3_fig03_bell_inequality.png')
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("PAPER THREE — QUANTUM FOUNDATIONS (Figures 1-3)")
    print("=" * 60)

    print("\nFigure 1: Measurement Problem...")
    figure_1()

    print("\nFigure 2: Quantum Entanglement...")
    figure_2()

    print("\nFigure 3: Bell Inequality...")
    figure_3()

    print("\n" + "=" * 60)
    print("ALL THREE FIGURES COMPLETE")
    print("=" * 60)
