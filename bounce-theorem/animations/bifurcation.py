"""
Bifurcation Diagram Animation — Feigenbaum at Los Alamos
Scene: Order → Period-doubling → Chaos → δ = 4.669... → Universality
"""
from manim import *
import numpy as np

BG_COLOR     = "#09090E"
GOLD         = "#C9A84C"
GOLD_DIM     = "#8A7535"
WHITE_WARM   = "#EAE8E2"
BLUE_COOL    = "#4A7090"
TEXT_DIM     = "#5A584F"
RED_SOFT     = "#C8614A"

def logistic_attractor(r, n_iter=1000, n_last=200, x0=0.5):
    """Return settled x values for logistic map at parameter r."""
    x = x0
    for _ in range(n_iter):
        x = r * x * (1 - x)
    pts = []
    for _ in range(n_last):
        x = r * x * (1 - x)
        pts.append(x)
    return pts

def sine_attractor(r, n_iter=1000, n_last=200, x0=0.5):
    """Return settled x values for sine map at parameter r."""
    x = x0
    for _ in range(n_iter):
        x = r * np.sin(np.pi * x)
    pts = []
    for _ in range(n_last):
        x = r * np.sin(np.pi * x)
        pts.append(x)
    return pts


class BifurcationDiagram(Scene):
    def construct(self):
        self.camera.background_color = BG_COLOR

        # ── Axes ──────────────────────────────────────────────────
        axes = Axes(
            x_range=[2.5, 4.05, 0.5],
            y_range=[0, 1.05, 0.25],
            x_length=11,
            y_length=6,
            axis_config={"color": GOLD_DIM, "stroke_width": 1.2,
                         "include_ticks": True, "tick_size": 0.05},
            tips=False,
        )
        axes.shift(DOWN * 0.3)

        x_label = Text("r  (growth parameter)", font="JetBrains Mono",
                       font_size=18, color=GOLD_DIM)
        x_label.next_to(axes, DOWN, buff=0.35)
        y_label = Text("x", font="JetBrains Mono", font_size=18, color=GOLD_DIM)
        y_label.next_to(axes, LEFT, buff=0.25)

        # Title
        title = Text("The Bifurcation Diagram",
                     font="Cormorant Garamond", font_size=38,
                     color=WHITE_WARM)
        title.to_edge(UP, buff=0.35)

        subtitle = Text("logistic map  ·  f(x) = rx(1−x)",
                        font="JetBrains Mono", font_size=20, color=GOLD_DIM)
        subtitle.next_to(title, DOWN, buff=0.18)

        self.play(FadeIn(title), FadeIn(subtitle), run_time=0.8)
        self.play(Create(axes), FadeIn(x_label), FadeIn(y_label), run_time=1.0)

        # ── Build diagram dot by dot ───────────────────────────────
        r_values = np.linspace(2.5, 4.0, 1800)
        all_dots = VGroup()

        for r in r_values:
            pts = logistic_attractor(r, n_iter=800, n_last=150)
            for xv in set([round(p, 4) for p in pts]):
                dot = Dot(
                    axes.c2p(r, xv),
                    radius=0.006,
                    color=GOLD,
                )
                dot.set_opacity(0.65)
                all_dots.add(dot)

        # Animate diagram growing left to right in chunks
        chunk = len(all_dots) // 60
        self.play(
            LaggedStart(
                *[FadeIn(all_dots[i:i+chunk]) for i in range(0, len(all_dots), chunk)],
                lag_ratio=0.04,
            ),
            run_time=5.0,
        )
        self.wait(0.8)

        # ── Mark bifurcation points ────────────────────────────────
        # Approximate r values where period doublings occur
        bif_r = [3.0, 3.449, 3.5441, 3.5644, 3.5688]
        bif_labels = ["period 1→2", "2→4", "4→8", "8→16", "..."]
        bif_colors = [GOLD, GOLD, GOLD_DIM, GOLD_DIM, TEXT_DIM]

        bif_lines = VGroup()
        bif_texts = VGroup()
        for r_val, lbl, col in zip(bif_r[:3], bif_labels[:3], bif_colors[:3]):
            vl = DashedLine(
                axes.c2p(r_val, 0),
                axes.c2p(r_val, 1.02),
                color=col, stroke_width=0.9, dash_length=0.08,
            )
            lt = Text(lbl, font="JetBrains Mono", font_size=13, color=col)
            lt.next_to(axes.c2p(r_val, 1.04), UP, buff=0.05)
            bif_lines.add(vl)
            bif_texts.add(lt)

        self.play(
            LaggedStart(*[Create(l) for l in bif_lines], lag_ratio=0.4),
            LaggedStart(*[FadeIn(t) for t in bif_texts], lag_ratio=0.4),
            run_time=2.0,
        )
        self.wait(0.6)

        # ── Ratio counter → δ ─────────────────────────────────────
        ratio_title = Text("ratio of successive bifurcation intervals:",
                           font="JetBrains Mono", font_size=18, color=WHITE_WARM)
        ratio_title.to_edge(UP, buff=3.5)

        # Intervals
        d1 = bif_r[1] - bif_r[0]   # 0.449
        d2 = bif_r[2] - bif_r[1]   # 0.0951
        d3 = bif_r[3] - bif_r[2]   # 0.0203
        ratios = [d1/d2, d2/d3]

        ratio_vals = [
            f"Δ₁/Δ₂ = {d1:.4f}/{d2:.4f} = {d1/d2:.4f}",
            f"Δ₂/Δ₃ = {d2:.4f}/{d3:.4f} = {d2/d3:.4f}",
        ]

        self.play(FadeIn(ratio_title), run_time=0.5)

        ratio_displays = VGroup()
        for i, rv in enumerate(ratio_vals):
            rd = Text(rv, font="JetBrains Mono", font_size=17, color=GOLD)
            rd.next_to(ratio_title, DOWN, buff=0.25 + i*0.38)
            ratio_displays.add(rd)
            self.play(FadeIn(rd), run_time=0.6)
            self.wait(0.4)

        # The convergence reveal
        delta_box = Rectangle(width=6.2, height=0.7,
                              fill_color=GOLD, fill_opacity=0.08,
                              stroke_color=GOLD, stroke_width=1.2)
        delta_text = Text("δ  =  4.66920160910299...",
                          font="JetBrains Mono", font_size=24,
                          color=GOLD, weight=BOLD)
        delta_group = VGroup(delta_box, delta_text)
        delta_group.arrange(ORIGIN)
        delta_group.next_to(ratio_displays, DOWN, buff=0.45)

        self.play(
            FadeIn(delta_box),
            Write(delta_text),
            run_time=1.2,
        )
        self.wait(1.5)

        # ── Transition: same constant, sine map ────────────────────
        transition = Text("Now watch a completely different equation.",
                          font="JetBrains Mono", font_size=20, color=WHITE_WARM)
        transition.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(transition), run_time=0.6)
        self.wait(0.8)

        # Fade logistic diagram, bring in sine map
        self.play(
            FadeOut(all_dots),
            FadeOut(bif_lines), FadeOut(bif_texts),
            FadeOut(ratio_displays), FadeOut(ratio_title),
            FadeOut(transition),
            run_time=0.8,
        )

        new_subtitle = Text("sine map  ·  f(x) = r·sin(πx)",
                            font="JetBrains Mono", font_size=20, color=BLUE_COOL)
        new_subtitle.next_to(title, DOWN, buff=0.18)
        self.play(Transform(subtitle, new_subtitle), run_time=0.5)

        # Build sine map diagram
        sine_dots = VGroup()
        r_sine = np.linspace(0.6, 1.0, 1600)
        for r in r_sine:
            pts = sine_attractor(r, n_iter=800, n_last=150)
            for xv in set([round(p, 4) for p in pts]):
                if 0 <= xv <= 1:
                    dot = Dot(
                        axes.c2p(2.5 + (r - 0.6) / (1.0 - 0.6) * 1.5, xv),
                        radius=0.006,
                        color=BLUE_COOL,
                    )
                    dot.set_opacity(0.65)
                    sine_dots.add(dot)

        self.play(
            LaggedStart(
                *[FadeIn(sine_dots[i:i+50]) for i in range(0, min(len(sine_dots), 3000), 50)],
                lag_ratio=0.03,
            ),
            run_time=3.5,
        )
        self.wait(0.5)

        # Show same δ appearing
        same_delta = Text("same ratio  ·  same constant  ·  different equation",
                          font="JetBrains Mono", font_size=19, color=BLUE_COOL)
        same_delta.to_edge(DOWN, buff=0.8)

        delta_box2 = delta_box.copy().set_color(BLUE_COOL)
        delta_text2 = Text("δ  =  4.66920160910299...",
                           font="JetBrains Mono", font_size=24,
                           color=BLUE_COOL, weight=BOLD)
        delta_group2 = VGroup(delta_box2, delta_text2)
        delta_group2.arrange(ORIGIN)
        delta_group2.next_to(same_delta, UP, buff=0.3)

        self.play(FadeIn(same_delta), run_time=0.5)
        self.play(FadeIn(delta_box2), Write(delta_text2), run_time=1.0)
        self.wait(1.0)

        # ── Final: UNIVERSAL ──────────────────────────────────────
        self.play(
            FadeOut(sine_dots), FadeOut(same_delta),
            FadeOut(subtitle), FadeOut(delta_group2),
            run_time=0.7,
        )

        # Both diagrams side by side — logistic (gold) + sine (blue)
        final_label = Text("UNIVERSAL",
                           font="Cormorant Garamond", font_size=62,
                           color=GOLD, weight=BOLD)
        final_label.move_to(ORIGIN + UP * 0.5)

        final_sub = Text(
            "every system in the universality class\nconverges to the same constant",
            font="JetBrains Mono", font_size=18,
            color=GOLD_DIM, line_spacing=1.4,
        )
        final_sub.next_to(final_label, DOWN, buff=0.4)

        delta_final = Text("δ  =  4.66920160910299...",
                           font="JetBrains Mono", font_size=22, color=GOLD)
        delta_final.next_to(final_sub, DOWN, buff=0.45)

        self.play(
            FadeOut(axes), FadeOut(x_label), FadeOut(y_label),
            FadeOut(delta_group), FadeOut(title),
            run_time=0.6,
        )
        self.play(
            Write(final_label),
            run_time=1.2,
        )
        self.play(FadeIn(final_sub), FadeIn(delta_final), run_time=0.8)
        self.wait(3.0)
