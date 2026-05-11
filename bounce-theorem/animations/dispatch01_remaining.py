"""
Dispatch 001 — Remaining Animations
Scene A: Operation Count — 234 trillion vs 2,793
Scene B: Carmichael Reveal — three prime spheres in a trench coat
"""
from manim import *
import numpy as np

BG_COLOR    = "#09090E"
GOLD        = "#C9A84C"
GOLD_DIM    = "#8A7535"
WHITE_WARM  = "#EAE8E2"
BLUE_COOL   = "#4A7090"
RED_SOFT    = "#C8614A"
TEXT_DIM    = "#5A584F"

PRIME_FILL      = "#F0E6C8"
PRIME_STROKE    = "#C9A84C"
COMPOSITE_FILL  = "#4A7090"
COMPOSITE_STROKE= "#2A5070"

def is_prime(n):
    if n < 2: return False
    if n == 2: return True
    if n % 2 == 0: return False
    for i in range(3, int(n**0.5)+1, 2):
        if n % i == 0: return False
    return True

def make_prime_sphere(radius=0.42):
    s = Sphere(radius=radius, resolution=(24, 24))
    s.set_fill(PRIME_FILL, opacity=0.97)
    s.set_stroke(PRIME_STROKE, width=0.8, opacity=0.9)
    return s

def make_composite_cluster(n, base_r=0.33):
    """Lumpy cluster of prime-sphere components."""
    rng = np.random.default_rng(n)
    # factor count with multiplicity
    k, tmp, d = 0, n, 2
    while d*d <= tmp:
        while tmp % d == 0: k += 1; tmp //= d
        d += 1
    if tmp > 1: k += 1
    k = min(max(k, 2), 6)

    grp = VGroup()
    for i in range(k):
        s = Sphere(radius=base_r, resolution=(18, 18))
        s.set_fill(COMPOSITE_FILL, opacity=0.93)
        s.set_stroke(COMPOSITE_STROKE, width=0.5, opacity=0.7)
        if i == 0:
            offset = ORIGIN
        else:
            theta  = rng.uniform(0, TAU)
            phi_a  = rng.uniform(0.3, 0.9)
            r_off  = rng.uniform(0.38, 0.54)
            offset = np.array([r_off*np.sin(phi_a)*np.cos(theta),
                               r_off*np.sin(phi_a)*np.sin(theta),
                               r_off*np.cos(phi_a)*0.5])
        s.shift(offset)
        grp.add(s)
    return grp


# ════════════════════════════════════════════════════════════════
#  SCENE A — Operation Count
# ════════════════════════════════════════════════════════════════
class SceneA_OperationCount(Scene):
    def construct(self):
        self.camera.background_color = BG_COLOR

        title = Text("The Speed Comparison",
                     font="Cormorant Garamond", font_size=40, color=WHITE_WARM)
        title.to_edge(UP, buff=0.5)
        self.play(FadeIn(title), run_time=0.6)

        # Context line
        context = Text("1,400-bit integers  ·  cryptographic scale",
                       font="JetBrains Mono", font_size=18, color=TEXT_DIM)
        context.next_to(title, DOWN, buff=0.25)
        self.play(FadeIn(context), run_time=0.5)
        self.wait(0.3)

        # ── AKS bar ───────────────────────────────────────────────
        aks_label = Text("AKS  (gold standard, 2002)",
                         font="JetBrains Mono", font_size=20, color=RED_SOFT)
        aks_label.move_to(LEFT*3.5 + UP*1.2)
        self.play(FadeIn(aks_label), run_time=0.4)

        aks_bar = Rectangle(width=0.01, height=0.7,
                            fill_color=RED_SOFT, fill_opacity=0.85,
                            stroke_width=0)
        aks_bar.align_to(LEFT*3.5, LEFT).shift(UP*0.4)

        aks_num = Text("234,000,000,000,000",
                       font="JetBrains Mono", font_size=22,
                       color=RED_SOFT, weight=BOLD)
        aks_num.next_to(aks_bar, RIGHT, buff=0.3)

        self.add(aks_bar)
        self.play(
            aks_bar.animate.stretch_to_fit_width(9.5).align_to(LEFT*3.5, LEFT),
            FadeIn(aks_num),
            run_time=2.0,
            rate_func=rush_into,
        )

        ops_label_aks = Text("operations", font="JetBrains Mono",
                             font_size=15, color=RED_SOFT)
        ops_label_aks.next_to(aks_num, RIGHT, buff=0.2)
        self.play(FadeIn(ops_label_aks), run_time=0.3)
        self.wait(0.5)

        # ── Cascade bar ───────────────────────────────────────────
        cascade_label = Text("Cascade  (Bounce Theorem)",
                             font="JetBrains Mono", font_size=20, color=GOLD)
        cascade_label.move_to(LEFT*3.5 + DOWN*0.5)
        self.play(FadeIn(cascade_label), run_time=0.4)

        cascade_bar = Rectangle(width=0.01, height=0.7,
                                fill_color=GOLD, fill_opacity=0.85,
                                stroke_width=0)
        cascade_bar.align_to(LEFT*3.5, LEFT).shift(DOWN*1.1)

        self.add(cascade_bar)

        # This bar is barely visible — that's the point
        self.play(
            cascade_bar.animate.stretch_to_fit_width(0.035).align_to(LEFT*3.5, LEFT),
            run_time=0.8,
        )

        cascade_num = Text("2,793",
                           font="JetBrains Mono", font_size=28,
                           color=GOLD, weight=BOLD)
        cascade_num.next_to(cascade_bar, RIGHT, buff=0.3)
        ops_label_cas = Text("operations", font="JetBrains Mono",
                             font_size=15, color=GOLD)
        ops_label_cas.next_to(cascade_num, RIGHT, buff=0.2)
        self.play(FadeIn(cascade_num), FadeIn(ops_label_cas), run_time=0.5)
        self.wait(0.8)

        # ── The ratio ─────────────────────────────────────────────
        ratio_box = Rectangle(width=5.5, height=0.9,
                              fill_color=GOLD, fill_opacity=0.08,
                              stroke_color=GOLD, stroke_width=1.2)
        ratio_text = Text("84,000,000,000×  faster",
                          font="JetBrains Mono", font_size=26,
                          color=GOLD, weight=BOLD)
        ratio_group = VGroup(ratio_box, ratio_text)
        ratio_group.arrange(ORIGIN)
        ratio_group.to_edge(DOWN, buff=0.7)

        zero_errors = Text("zero classification errors",
                           font="JetBrains Mono", font_size=18, color=GOLD_DIM)
        zero_errors.next_to(ratio_group, UP, buff=0.3)

        self.play(FadeIn(ratio_box), Write(ratio_text), run_time=1.0)
        self.play(FadeIn(zero_errors), run_time=0.5)
        self.wait(2.5)


# ════════════════════════════════════════════════════════════════
#  SCENE B — Carmichael Reveal
# ════════════════════════════════════════════════════════════════
class SceneB_CarmichaelReveal(ThreeDScene):
    def construct(self):
        self.camera.background_color = BG_COLOR
        self.set_camera_orientation(phi=72*DEGREES, theta=-50*DEGREES, zoom=0.9)

        # Floor
        from manim.mobject.three_d.three_dimensions import Surface
        floor = Surface(
            lambda u, v: np.array([u, v, 0]),
            u_range=[-6, 6], v_range=[-3, 3],
            fill_color=GOLD, fill_opacity=0.10,
            stroke_color=GOLD, stroke_width=0.4,
        )
        glow_line = Line3D([-6,0,0.01], [6,0,0.01],
                           color=GOLD, stroke_width=2.5)
        self.add(floor, glow_line)

        # ── Title overlay ─────────────────────────────────────────
        title = Text("The Carmichael Problem",
                     font="Cormorant Garamond", font_size=34, color=WHITE_WARM)
        title.to_edge(UP, buff=0.4)
        self.add_fixed_in_frame_mobjects(title)
        self.play(FadeIn(title), run_time=0.5)

        sub = Text("561 = 3 × 11 × 17  ·  passes every Fermat test  ·  not prime",
                   font="JetBrains Mono", font_size=16, color=TEXT_DIM)
        sub.next_to(title, DOWN, buff=0.2)
        self.add_fixed_in_frame_mobjects(sub)
        self.play(FadeIn(sub), run_time=0.5)

        # ── Show 561 as a cluster of 3 prime spheres ───────────────
        # Three component primes: 3, 11, 17
        sphere_3  = make_prime_sphere(0.38)
        sphere_11 = make_prime_sphere(0.38)
        sphere_17 = make_prime_sphere(0.38)

        # Labels for each component
        lbl_3  = Text("3",  font="JetBrains Mono", font_size=20, color=GOLD)
        lbl_11 = Text("11", font="JetBrains Mono", font_size=20, color=GOLD)
        lbl_17 = Text("17", font="JetBrains Mono", font_size=20, color=GOLD)

        # Start high, separated — reveal the three primes
        sphere_3.move_to([-1.2, 0, 5])
        sphere_11.move_to([0, 0, 5.8])
        sphere_17.move_to([1.2, 0, 5])

        self.add(sphere_3, sphere_11, sphere_17)
        self.wait(0.4)

        # Brief flash: "These are primes"
        note = Text("three primes  ·  individually they touch the floor",
                    font="JetBrains Mono", font_size=15, color=GOLD_DIM)
        note.to_edge(DOWN, buff=0.5)
        self.add_fixed_in_frame_mobjects(note)
        self.play(FadeIn(note), run_time=0.5)
        self.wait(0.8)

        # ── Bond them into the Carmichael cluster ──────────────────
        bond_note = Text("bonded together as 561  ·  arithmetic disguise activated",
                         font="JetBrains Mono", font_size=15, color=COMPOSITE_FILL)
        bond_note.move_to(note.get_center())

        # Move spheres together into a tight cluster
        self.play(
            sphere_3.animate.move_to([-0.45, 0, 5.2]),
            sphere_11.animate.move_to([0, 0.3, 5.6]),
            sphere_17.animate.move_to([0.45, 0, 5.2]),
            Transform(note, bond_note),
            run_time=1.0,
        )

        # Color them composite-blue — the disguise
        self.play(
            sphere_3.animate.set_fill(COMPOSITE_FILL, opacity=0.93)
                             .set_stroke(COMPOSITE_STROKE, width=0.5),
            sphere_11.animate.set_fill(COMPOSITE_FILL, opacity=0.93)
                              .set_stroke(COMPOSITE_STROKE, width=0.5),
            sphere_17.animate.set_fill(COMPOSITE_FILL, opacity=0.93)
                              .set_stroke(COMPOSITE_STROKE, width=0.5),
            run_time=0.7,
        )
        self.wait(0.6)

        # ── The cluster descends toward the floor ──────────────────
        approach_note = Text("approaching the cascade floor...",
                             font="JetBrains Mono", font_size=15, color=GOLD_DIM)
        approach_note.move_to(note.get_center())
        self.play(Transform(note, approach_note), run_time=0.4)

        cluster_center = np.array([0, 0, 5.35])  # centroid
        drop_target   = np.array([0, 0, 0.5])    # near floor

        self.play(
            sphere_3.animate.shift( drop_target - cluster_center + np.array([-0.45,0,0])),
            sphere_11.animate.shift(drop_target - cluster_center + np.array([0,0.3,0.25])),
            sphere_17.animate.shift(drop_target - cluster_center + np.array([0.45,0,0])),
            run_time=1.8,
            rate_func=rush_into,
        )

        # ── THE BOUNCE ─────────────────────────────────────────────
        bounce_note = Text("residual ≠ 0  ·  the floor sees what arithmetic cannot",
                           font="JetBrains Mono", font_size=15, color=GOLD)
        bounce_note.move_to(note.get_center())
        self.play(Transform(note, bounce_note), run_time=0.3)

        # Bounce back up
        self.play(
            sphere_3.animate.shift([0, 0, 1.4]),
            sphere_11.animate.shift([0, 0, 1.4]),
            sphere_17.animate.shift([0, 0, 1.4]),
            run_time=0.55,
            rate_func=rush_from,
        )
        self.wait(0.4)

        # Settle at hover height
        self.play(
            sphere_3.animate.shift([0, 0, -0.3]),
            sphere_11.animate.shift([0, 0, -0.3]),
            sphere_17.animate.shift([0, 0, -0.3]),
            run_time=0.4,
        )
        self.wait(0.8)

        # ── Final statement ────────────────────────────────────────
        final_note = Text("not prime  ·  never was  ·  the geometry always knew",
                          font="JetBrains Mono", font_size=16, color=GOLD)
        final_note.move_to(note.get_center())
        self.play(Transform(note, final_note), run_time=0.5)

        # Camera slowly rotates to reveal the separation
        self.move_camera(phi=80*DEGREES, theta=0*DEGREES,
                         zoom=0.8, run_time=3.5, rate_func=smooth)
        self.wait(2.5)
