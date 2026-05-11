"""
Cascade Floor Animation — The Bounce Theorem
Universal Cascade Dynamics / Lucian Randolph

Scene 1: Wide isometric — integers descend. Primes land perfectly.
         Composites bounce. Carmichael numbers revealed.
Scene 2: Side close-up — one prime, residual to zero. One composite, bouncing.
Scene 3: THE ROTATION — camera sweeps end-on. Infinite separation visible.
         Two layers. One floor. The gap that never closes.
"""

from manim import *
import numpy as np

# ── Brand palette ──────────────────────────────────────────────
BG_COLOR        = "#09090E"   # deep dark
FLOOR_COLOR     = "#C9A84C"   # gold
FLOOR_DIM       = "#8A7535"   # dim gold
PRIME_FILL      = "#F0E6C8"   # warm white
PRIME_STROKE    = "#C9A84C"   # gold
COMPOSITE_FILL  = "#4A7090"   # cool blue-grey
COMPOSITE_STROKE= "#2A5070"   # darker blue
LABEL_COLOR     = "#C9A84C"   # gold
TEXT_DIM        = "#5A584F"   # dim

# ── Math helpers ────────────────────────────────────────────────
def is_prime(n):
    if n < 2: return False
    if n == 2: return True
    if n % 2 == 0: return False
    for i in range(3, int(n**0.5)+1, 2):
        if n % i == 0: return False
    return True

def factor_count(n):
    """Prime factors with multiplicity."""
    count = 0
    d = 2
    tmp = n
    while d * d <= tmp:
        while tmp % d == 0:
            count += 1
            tmp //= d
        d += 1
    if tmp > 1:
        count += 1
    return max(count, 2)


# ── Sphere builders ─────────────────────────────────────────────
def make_prime_sphere(radius=0.42):
    """Perfect sphere — gold-white glow."""
    s = Sphere(radius=radius, resolution=(24, 24))
    s.set_fill(PRIME_FILL, opacity=0.97)
    s.set_stroke(PRIME_STROKE, width=0.8, opacity=0.9)
    return s

def make_composite_cluster(n, base_r=0.33):
    """
    Lumpy cluster: one sphere per prime factor (with multiplicity), capped at 6.
    Deterministic irregular offsets so it always looks the same.
    """
    rng = np.random.default_rng(n)
    k   = min(factor_count(n), 6)
    grp = VGroup()

    for i in range(k):
        s = Sphere(radius=base_r, resolution=(18, 18))
        s.set_fill(COMPOSITE_FILL, opacity=0.93)
        s.set_stroke(COMPOSITE_STROKE, width=0.5, opacity=0.7)

        if i == 0:
            offset = ORIGIN
        else:
            theta   = rng.uniform(0, TAU)
            phi_ang = rng.uniform(0.2, 0.8)
            r_off   = rng.uniform(0.28, 0.46)
            offset  = np.array([
                r_off * np.sin(phi_ang) * np.cos(theta),
                r_off * np.sin(phi_ang) * np.sin(theta),
                r_off * np.cos(phi_ang) * 0.6,
            ])
        s.shift(offset)
        grp.add(s)
    return grp


# ── Floor builder ────────────────────────────────────────────────
def make_floor(width=22, depth=9):
    plane = Surface(
        lambda u, v: np.array([u, v, 0]),
        u_range=[-width/2, width/2],
        v_range=[-depth/2, depth/2],
        resolution=(1, 1),
        fill_color=FLOOR_COLOR,
        fill_opacity=0.10,
        stroke_color=FLOOR_COLOR,
        stroke_width=0.4,
    )
    # Glowing centre line along x-axis
    glow = Line3D(
        start=np.array([-width/2, 0, 0.01]),
        end=np.array([width/2, 0, 0.01]),
        color=FLOOR_COLOR,
        stroke_width=2.5,
    )
    return VGroup(plane, glow)


# ════════════════════════════════════════════════════════════════
#  SCENE 1 — Wide isometric view
# ════════════════════════════════════════════════════════════════
class Scene01_WideIsometric(ThreeDScene):
    def construct(self):
        self.camera.background_color = BG_COLOR
        self.set_camera_orientation(phi=68*DEGREES, theta=-55*DEGREES, zoom=0.6)

        # Floor
        floor = make_floor()
        self.add(floor)

        # Numbers: 8 primes + 6 composites + 561 (Carmichael)
        numbers  = [2, 4, 3, 9,  5, 6,  7, 10, 11, 15, 13, 12, 17, 8,  561]
        x_start  = -9.8
        x_step   =  1.4
        drop_z   =  6.5
        PRIME_R  =  0.42
        COMP_HVR =  0.90  # composite hover height above floor

        objects  = []
        x_pos    = []

        for i, n in enumerate(numbers):
            x = x_start + i * x_step
            x_pos.append(x)
            if is_prime(n):
                obj = make_prime_sphere(PRIME_R)
                obj.move_to([x, 0, drop_z])
            else:
                obj = make_composite_cluster(n)
                obj.move_to([x, 0, drop_z])
            objects.append(obj)
            self.add(obj)

        self.wait(0.4)

        # ── Drop in three staggered waves ──
        waves = [
            [0, 1, 2, 3, 4],
            [5, 6, 7, 8, 9],
            [10, 11, 12, 13, 14],
        ]

        for wave_idx in waves:
            drop_anims = []
            for idx in wave_idx:
                n   = numbers[idx]
                obj = objects[idx]
                x   = x_pos[idx]

                if is_prime(n):
                    target_z = PRIME_R          # sits ON the floor
                else:
                    target_z = drop_z * 0.15    # approaches floor…

                drop_anims.append(
                    obj.animate.move_to([x, 0, target_z])
                )

            self.play(*drop_anims, run_time=1.0, rate_func=rush_into)

            # Composites in this wave bounce back up
            bounce_anims = []
            for idx in wave_idx:
                n   = numbers[idx]
                obj = objects[idx]
                x   = x_pos[idx]
                if not is_prime(n):
                    fc   = factor_count(n)
                    hvr  = COMP_HVR + fc * 0.06
                    bounce_anims.append(
                        obj.animate.move_to([x, 0, hvr])
                    )
            if bounce_anims:
                self.play(*bounce_anims, run_time=0.45, rate_func=rush_from)

            self.wait(0.25)

        self.wait(2.0)


# ════════════════════════════════════════════════════════════════
#  SCENE 2 — Side close-up: residual proof
# ════════════════════════════════════════════════════════════════
class Scene02_SideCloseup(ThreeDScene):
    def construct(self):
        self.camera.background_color = BG_COLOR
        self.set_camera_orientation(phi=90*DEGREES, theta=0*DEGREES, zoom=1.1)

        floor = make_floor(width=10, depth=6)
        self.add(floor)

        # ── Prime 7 drops ──
        prime = make_prime_sphere(0.44)
        prime.move_to([0, 0, 5])
        self.add(prime)

        # Label above sphere
        label_p = Text("7", font="JetBrains Mono", font_size=32,
                       color=LABEL_COLOR, weight=BOLD)
        label_p.fix_in_frame()
        label_p.move_to(UP * 3.1)
        self.add(label_p)

        # Residual counter (2D overlay)
        residual_label = Text("residual = 1.000", font="JetBrains Mono",
                              font_size=26, color=FLOOR_DIM)
        residual_label.fix_in_frame()
        residual_label.move_to(DOWN * 3.0)
        self.add(residual_label)

        # Drop prime toward floor, counting residual to zero
        def update_residual(mob, alpha):
            val = 1.0 - alpha   # 1 → 0 as prime approaches floor
            mob.become(
                Text(f"residual = {val:.3f}", font="JetBrains Mono",
                     font_size=26,
                     color=interpolate_color(FLOOR_DIM, FLOOR_COLOR, alpha))
                .fix_in_frame()
                .move_to(DOWN * 3.0)
            )

        self.play(
            prime.animate.move_to([0, 0, 0.44]),
            UpdateFromAlphaFunc(residual_label, update_residual),
            run_time=2.0,
            rate_func=smooth,
        )

        # Flash "= 0 exactly"
        exact = Text("residual = 0  ← exact", font="JetBrains Mono",
                     font_size=26, color=FLOOR_COLOR, weight=BOLD)
        exact.fix_in_frame()
        exact.move_to(DOWN * 3.0)
        self.play(Transform(residual_label, exact), run_time=0.4)
        self.wait(1.2)

        # Fade prime, bring in composite 9
        comp = make_composite_cluster(9, base_r=0.34)
        comp.move_to([0, 0, 5])
        self.add(comp)

        label_c = Text("9", font="JetBrains Mono", font_size=32,
                       color=COMPOSITE_FILL, weight=BOLD)
        label_c.fix_in_frame()
        label_c.move_to(UP * 3.1)

        self.play(
            FadeOut(prime), FadeOut(label_p), FadeOut(residual_label),
            FadeIn(label_c),
            run_time=0.6,
        )

        res_c = Text("residual = 1.000", font="JetBrains Mono",
                     font_size=26, color=COMPOSITE_FILL)
        res_c.fix_in_frame()
        res_c.move_to(DOWN * 3.0)
        self.add(res_c)

        # Composite descends to just above floor, stops at 0.12
        def update_residual_comp(mob, alpha):
            val = 1.0 - alpha * 0.88  # 1 → 0.12
            mob.become(
                Text(f"residual = {val:.3f}", font="JetBrains Mono",
                     font_size=26, color=COMPOSITE_FILL)
                .fix_in_frame()
                .move_to(DOWN * 3.0)
            )

        self.play(
            comp.animate.move_to([0, 0, 0.5]),
            UpdateFromAlphaFunc(res_c, update_residual_comp),
            run_time=1.4,
            rate_func=rush_into,
        )
        # Bounce back
        bounce_label = Text("residual = 0.121  ← cannot reach zero",
                            font="JetBrains Mono", font_size=22,
                            color=COMPOSITE_FILL)
        bounce_label.fix_in_frame()
        bounce_label.move_to(DOWN * 3.0)

        self.play(
            comp.animate.move_to([0, 0, 1.1]),
            Transform(res_c, bounce_label),
            run_time=0.5,
            rate_func=rush_from,
        )
        self.wait(2.0)


# ════════════════════════════════════════════════════════════════
#  SCENE 3 — THE ROTATION: infinite separation
# ════════════════════════════════════════════════════════════════
class Scene03_TheRotation(ThreeDScene):
    def construct(self):
        self.camera.background_color = BG_COLOR
        self.set_camera_orientation(phi=68*DEGREES, theta=-55*DEGREES, zoom=0.55)

        floor = make_floor(width=26, depth=8)
        self.add(floor)

        # ── Populate floor with many primes and composites ──
        primes_list    = [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,
                          53,59,61,67,71,73,79,83,89,97]
        composites_list= [4,6,8,9,10,12,14,15,16,18,20,21,22,24,25,
                          26,27,28,561,1105,1729]

        PRIME_R  = 0.38
        COMP_HVR = 0.92

        # Place primes ON the floor along x-axis, spread in depth (y)
        for i, p in enumerate(primes_list):
            x = -10 + (i % 13) * 1.65
            y = -1.2 + (i // 13) * 2.4
            obj = make_prime_sphere(PRIME_R)
            obj.move_to([x, y, PRIME_R])
            self.add(obj)

        # Place composites ABOVE the floor
        for i, c in enumerate(composites_list):
            x   = -8 + (i % 11) * 1.65
            y   = -1.5 + (i // 11) * 3.0
            fc  = factor_count(c)
            hvr = COMP_HVR + fc * 0.08
            obj = make_composite_cluster(c, base_r=0.30)
            obj.move_to([x, y, hvr])
            self.add(obj)

        self.wait(1.5)

        # ── THE ROTATION ──────────────────────────────────────────
        # Sweep from wide isometric to end-on (looking down the plane)
        # phi stays near 80° (nearly horizontal), theta sweeps to 90°
        self.move_camera(
            phi=80*DEGREES,
            theta=90*DEGREES,   # now looking along the x-axis
            zoom=0.45,
            run_time=5.0,
            rate_func=smooth,
        )

        # Hold on the separation view
        self.wait(4.5)

        # Optional: gentle slow zoom in toward the horizon
        self.move_camera(
            zoom=0.65,
            run_time=3.0,
            rate_func=smooth,
        )
        self.wait(3.0)


# ════════════════════════════════════════════════════════════════
#  FULL CUT — all three scenes in sequence
# ════════════════════════════════════════════════════════════════
class CascadeFloorFull(ThreeDScene):
    """Render all three scenes back to back."""
    def construct(self):
        # ── Scene 1 ──
        s1 = Scene01_WideIsometric(renderer=self.renderer)
        s1.construct()
        self.clear()

        # ── Scene 2 ──
        s2 = Scene02_SideCloseup(renderer=self.renderer)
        s2.construct()
        self.clear()

        # ── Scene 3 ──
        s3 = Scene03_TheRotation(renderer=self.renderer)
        s3.construct()
