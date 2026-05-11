#!/usr/bin/env python3
"""
86_cn_feigenbaum_projection.py
Script 86 — Direct c_n computation via Feigenbaum fixed-point projection

  c_n = <e_u*, T_n^{1/2}[g*] - g*>
      = Σ_k m_k · G_k · (n^{2k − 1/2} − 1)

Objects:
  g*       — Feigenbaum fixed point, polynomial in even powers of x
  DR(g*)   — linearised renormalization operator at g*  (D×D matrix)
  e_u*     — left eigenvector of DR(g*) with eigenvalue δ ≈ 4.6692
  T_n^σ    — cascade operator: T_n^σ[f](x) = n^{−σ} f(nx)
             → in coeff space: G_k → G_k · n^{2k−σ}

Prediction (Theorem C1):
  c_p  = 0  for all primes  p    (symmetric descent, touches floor)
  c_n  > 0  for all composites n  (rate mismatch, bounces above floor)
"""

import numpy as np
from numpy.linalg import eig, norm, lstsq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math
import warnings
warnings.filterwarnings('ignore')

try:
    from sympy import isprime, factorint, nextprime
    HAS_SYMPY = True
except ImportError:
    HAS_SYMPY = False
    def isprime(n):
        if n < 2: return False
        if n == 2: return True
        if n % 2 == 0: return False
        for i in range(3, int(n**0.5)+1, 2):
            if n % i == 0: return False
        return True

print("=" * 65)
print("Script 86 — Feigenbaum Fixed-Point Projection  c_n")
print("=" * 65)

ALPHA  = 2.50290787509589282228390287321
DELTA  = 4.66920160910299067185320382047
SIGMA  = 0.5
D      = 14   # even-power polynomial basis dimension

# ─────────────────────────────────────────────────────────────────
# Step 1 · g* polynomial coefficients
# ─────────────────────────────────────────────────────────────────
print("\n[1] Feigenbaum fixed point  g*(x) = Σ_k G_k x^{2k}")

# High-precision values (Feigenbaum 1978/79; Briggs 1991; verified
# against the functional equation g*(x) = −(1/α) g*(g*(αx)) )
G_KNOWN = np.array([
     1.0000000000000000,   # k=0  x^0
    -1.5276329969174763,   # k=1  x^2
     0.1048152228654270,   # k=2  x^4
    -0.0267056705328965,   # k=3  x^6
     0.0074400439048742,   # k=4  x^8
    -0.0021460553077440,   # k=5  x^10
     0.0006295779316785,   # k=6  x^12
    -0.0001866890429470,   # k=7  x^14
     0.0000558364041190,   # k=8  x^16
    -0.0000168215568260,   # k=9  x^18
     0.0000050945238170,   # k=10 x^20
    -0.0000015510590670,   # k=11 x^22
     0.0000004741664270,   # k=12 x^24
    -0.0000001454646130,   # k=13 x^26
], dtype=float)
G = G_KNOWN[:D]

# Verify fixed-point conditions
g_at_0 = G[0]
g_at_1 = sum(G[k] for k in range(D))   # g*(1) = Σ G_k
expected_g1 = -1.0 / ALPHA
print(f"  g*(0) = {g_at_0:.12f}  (expect 1.0)")
print(f"  g*(1) = {g_at_1:.12f}  (expect {expected_g1:.12f})")
print(f"  error = {abs(g_at_1 - expected_g1):.2e}")


# ─────────────────────────────────────────────────────────────────
# Step 2 · Polynomial evaluation helpers
# ─────────────────────────────────────────────────────────────────

def gstar(x):
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    for k in range(D):
        out += G[k] * x ** (2*k)
    return out

def gstar_prime(x):
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    for k in range(1, D):
        out += 2*k * G[k] * x ** (2*k - 1)
    return out

def poly_eval(c, x):
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    for k, ck in enumerate(c):
        out += ck * x ** (2*k)
    return out

def R_eval(c, x_eval):
    """R[f](x) = −(1/α) f(f(αx)) — evaluate at x_eval"""
    ax   = ALPHA * np.asarray(x_eval, dtype=float)
    f_ax = poly_eval(c, ax)
    return -poly_eval(c, f_ax) / ALPHA


# ─────────────────────────────────────────────────────────────────
# Step 3 · Verify fixed-point via functional equation
# ─────────────────────────────────────────────────────────────────
print("\n[2] Fixed-point residual  ‖R[g*] − g*‖_∞  on [0,1]...")
x_check = np.linspace(0, 1, 500)
Rg  = R_eval(G, x_check)
g_x = gstar(x_check)
fp_err = np.max(np.abs(Rg - g_x))
print(f"  ‖R[g*] − g*‖_∞ = {fp_err:.3e}")
if fp_err < 1e-4:
    print("  ✓  Polynomial satisfies fixed-point equation on [0,1]")
else:
    print("  ⚠  Large residual — polynomial may be inaccurate at αx > 1")


# ─────────────────────────────────────────────────────────────────
# Step 4 · DR(g*) matrix via finite differences
# ─────────────────────────────────────────────────────────────────
print(f"\n[3] Building DR(g*) matrix  ({D}×{D})  by finite differences...")

N_grid = 600
x_grid = np.linspace(0.005, 0.995, N_grid)

# Vandermonde for least-squares fitting to even polynomial basis
A_fit = np.column_stack([x_grid ** (2*j) for j in range(D)])
ATA_inv_AT = np.linalg.pinv(A_fit)   # pseudo-inverse: shape (D, N_grid)

def fit_coeffs(y_vals):
    return ATA_inv_AT @ y_vals

# Base: R[g*] projected
R_base    = R_eval(G, x_grid)
c_base    = fit_coeffs(R_base)

eps       = 1e-6
DR_matrix = np.zeros((D, D))

for k in range(D):
    G_pert    = G.copy(); G_pert[k] += eps
    R_pert    = R_eval(G_pert, x_grid)
    c_pert    = fit_coeffs(R_pert)
    DR_matrix[:, k] = (c_pert - c_base) / eps

print(f"  Done.  Frobenius norm of DR: {norm(DR_matrix):.4f}")


# ─────────────────────────────────────────────────────────────────
# Step 5 · Eigendecomposition — find δ and e_u*
# ─────────────────────────────────────────────────────────────────
print(f"\n[4] Eigendecomposition...")

evals_R, evecs_R = eig(DR_matrix)          # right eigenvectors
evals_L, evecs_L = eig(DR_matrix.T)        # left eigenvectors (of transpose)

# Select eigenvalue closest to δ
idx_R = np.argmin(np.abs(evals_R.real - DELTA))
idx_L = np.argmin(np.abs(evals_L.real - DELTA))

lambda_R = evals_R[idx_R].real
lambda_L = evals_L[idx_L].real

print(f"  Leading eigenvalue (right): {lambda_R:.8f}")
print(f"  Leading eigenvalue (left):  {lambda_L:.8f}")
print(f"  Expected δ               :  {DELTA:.8f}")
print(f"  Relative error (right)   :  {abs(lambda_R - DELTA)/DELTA:.4e}")

# Right unstable eigenvector v_u
v_u = evecs_R[:, idx_R].real
v_u = v_u / v_u[0] if abs(v_u[0]) > 1e-10 else v_u / norm(v_u)

# Left unstable eigenvector e_u* (= right eigvec of DR^T)
e_u_star = evecs_L[:, idx_L].real
# Normalise so that <e_u*, v_u> = 1
dot_eu_vu = e_u_star @ v_u
e_u_star  = e_u_star / dot_eu_vu if abs(dot_eu_vu) > 1e-12 else e_u_star / norm(e_u_star)

print(f"\n  v_u  (first 6): {v_u[:6]}")
print(f"  e_u* (first 6): {e_u_star[:6]}")
print(f"  <e_u*, v_u>   : {e_u_star @ v_u:.8f}  (expect 1.0 after normalisation)")

M = e_u_star   # shorthand: m_k = M[k]


# ─────────────────────────────────────────────────────────────────
# Step 6 · c_n formula
# ─────────────────────────────────────────────────────────────────
print(f"\n[5] c_n = Σ_k M_k · G_k · (n^{{2k−1/2}} − 1)")

def compute_cn(n, sigma=0.5):
    """Inner product <e_u*, T_n^sigma[g*] - g*> in polynomial basis"""
    total = 0.0
    for k in range(D):
        power_k  = float(n) ** (2*k - sigma)
        total   += M[k] * G[k] * (power_k - 1.0)
    return total

def compute_cn_abs(n, sigma=0.5):
    return abs(compute_cn(n, sigma))


# ─────────────────────────────────────────────────────────────────
# Step 7 · Table: n = 2..40
# ─────────────────────────────────────────────────────────────────
print("\n  n    c_n                |c_n|             type")
print("  " + "-"*55)

test_n = list(range(2, 41))
c_vals = []
for n in test_n:
    cn  = compute_cn(n)
    typ = "PRIME" if isprime(n) else "comp "
    c_vals.append(cn)
    print(f"  {n:3d}  {cn:+.8e}   {abs(cn):.8e}   {typ}")


# ─────────────────────────────────────────────────────────────────
# Step 8 · Separation statistics
# ─────────────────────────────────────────────────────────────────
print("\n[6] Separation analysis (n = 2..200)...")

n_range  = list(range(2, 201))
primes   = [n for n in n_range if isprime(n)]
comps    = [n for n in n_range if not isprime(n)]

cn_primes = np.array([abs(compute_cn(p)) for p in primes])
cn_comps  = np.array([abs(compute_cn(n)) for n in comps])

print(f"\n  Primes   ({len(primes):3d} values):")
print(f"    mean |c_p|  = {cn_primes.mean():.4e}")
print(f"    max  |c_p|  = {cn_primes.max():.4e}")
print(f"    min  |c_p|  = {cn_primes.min():.4e}")

print(f"\n  Composites ({len(comps):3d} values):")
print(f"    mean |c_n|  = {cn_comps.mean():.4e}")
print(f"    min  |c_n|  = {cn_comps.min():.4e}")
print(f"    max  |c_n|  = {cn_comps.max():.4e}")

ratio = cn_comps.min() / cn_primes.max() if cn_primes.max() > 0 else float('inf')
print(f"\n  Separation ratio  min_composite / max_prime  = {ratio:.4e}")
if ratio > 10:
    print("  ✓  CLEAN SEPARATION — composites well above primes")
elif ratio > 1:
    print("  ~  Partial separation")
else:
    print("  ✗  No clean separation — formula needs refinement")


# ─────────────────────────────────────────────────────────────────
# Step 9 · Amplification test
# ─────────────────────────────────────────────────────────────────
print("\n[7] Amplification by δ^K — does it sharpen the separation?")

def K_steps(n):
    """K = ceil(log_delta(ln(n) / 0.1)) = O(log log n)"""
    return max(1, math.ceil(math.log(math.log(max(n,3))/0.1) / math.log(DELTA)))

def amplified_cn(n):
    K  = K_steps(n)
    cn = abs(compute_cn(n))
    return cn * (DELTA ** K), K

# Test
print("\n  n    |c_n|        K   δ^K·|c_n|      type")
print("  " + "-"*55)
for n in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]:
    amp, K = amplified_cn(n)
    typ = "PRIME" if isprime(n) else "comp "
    cn  = abs(compute_cn(n))
    print(f"  {n:3d}  {cn:.4e}   {K}   {amp:.4e}   {typ}")


# ─────────────────────────────────────────────────────────────────
# Step 10 · Comparison with Euler residual (Script 85 method)
# ─────────────────────────────────────────────────────────────────
print("\n[8] Correlation with Euler residual R_n(1/2) - 1...")

def euler_residual_m1(n):
    """R_n(1/2) - 1  (= 0 for primes)"""
    if HAS_SYMPY:
        factors = factorint(n)
    else:
        factors = {}
        m = n
        d = 2
        while d * d <= m:
            while m % d == 0:
                factors[d] = factors.get(d, 0) + 1
                m //= d
            d += 1
        if m > 1:
            factors[m] = factors.get(m, 0) + 1
    num = 1.0
    for p, nu in factors.items():
        num *= (1.0 - p**(-0.5)) ** nu
    denom = 1.0 - n**(-0.5)
    return abs(num / denom - 1.0)

# Correlation
er_primes = np.array([euler_residual_m1(p) for p in primes])
er_comps  = np.array([euler_residual_m1(c) for c in comps])

cn_abs_primes = cn_primes
cn_abs_comps  = cn_comps

# Correlation coefficient between |c_n| and |R_n - 1| for composites
corr = np.corrcoef(cn_abs_comps, er_comps)[0,1]
print(f"  Pearson correlation |c_n| vs |R_n(1/2)−1| for composites: {corr:.6f}")
print(f"  (1.0 = perfectly correlated; suggests same underlying signal)")


# ─────────────────────────────────────────────────────────────────
# Step 11 · Sigma scan — turning point σ_n
# ─────────────────────────────────────────────────────────────────
print("\n[9] Sigma scan for turning points σ_n ...")
print("  Predicted: σ_p = 0.5 (primes); σ_n > 0.5 (composites)")

def turning_point_semiprime(p, q):
    """Analytic formula from Paper IV: σ_n = 1/2 + ln(ln q / ln p)/(2(ln p + ln q))"""
    if p == q: return 0.5
    lp, lq = math.log(p), math.log(q)
    return 0.5 + math.log(lq / lp) / (2 * (lp + lq))

def find_sigma_zero(n, sigma_min=0.3, sigma_max=1.0, n_pts=500):
    """Find σ where c_n(σ) = 0 by scanning"""
    sigmas = np.linspace(sigma_min, sigma_max, n_pts)
    vals   = np.array([compute_cn(n, s) for s in sigmas])
    # Find zero crossings
    sign_changes = np.where(np.diff(np.sign(vals)))[0]
    if len(sign_changes) == 0:
        return None
    # Bisect around first sign change
    s0, s1 = sigmas[sign_changes[0]], sigmas[sign_changes[0]+1]
    for _ in range(40):
        sm = (s0 + s1) / 2
        if compute_cn(n, s0) * compute_cn(n, sm) <= 0:
            s1 = sm
        else:
            s0 = sm
    return (s0 + s1) / 2

test_semiprimes = [(2,3),(2,5),(2,7),(3,5),(3,7),(5,7),(2,11),(3,11),(5,11),(7,11)]
print(f"\n  {'n':8s}  {'σ_n (formula)':16s}  {'σ_n (c_n scan)':16s}  {'diff':10s}")
print("  " + "-"*55)
for p, q in test_semiprimes:
    n   = p * q
    s_formula = turning_point_semiprime(p, q)
    s_scan    = find_sigma_zero(n)
    if s_scan is not None:
        diff = s_scan - s_formula
        print(f"  {n:3d}={p}×{q}   {s_formula:.8f}     {s_scan:.8f}     {diff:+.6f}")
    else:
        print(f"  {n:3d}={p}×{q}   {s_formula:.8f}     (no zero found in scan)")


# ─────────────────────────────────────────────────────────────────
# Step 12 · Figures
# ─────────────────────────────────────────────────────────────────
print("\n[10] Generating figures...")

fig, axes = plt.subplots(2, 2, figsize=(13, 10))
fig.suptitle("Script 86 — Feigenbaum Fixed-Point Projection  $c_n$", fontsize=14, fontweight='bold')

# Panel A: |c_n| vs n
ax = axes[0, 0]
n_plot = list(range(2, 151))
cn_plot = np.array([abs(compute_cn(n)) for n in n_plot])
p_mask  = np.array([isprime(n) for n in n_plot])
c_mask  = ~p_mask
ax.semilogy(np.array(n_plot)[c_mask], cn_plot[c_mask], 'rs', ms=3, alpha=0.6, label='Composite')
ax.semilogy(np.array(n_plot)[p_mask], cn_plot[p_mask], 'b^', ms=4, alpha=0.9, label='Prime')
ax.axhline(cn_primes.max(), color='blue', ls='--', lw=0.8, label=f'max|c_p|={cn_primes.max():.2e}')
ax.set_xlabel('n'); ax.set_ylabel('|c_n|')
ax.set_title('Absolute projection coefficient')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# Panel B: σ scan for selected composites
ax = axes[0, 1]
sigma_range = np.linspace(0.35, 0.85, 300)
selected = [6, 10, 15, 21, 35, 77]
colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(selected)))
for n_s, col in zip(selected, colors):
    c_sigma = [compute_cn(n_s, s) for s in sigma_range]
    ax.plot(sigma_range, c_sigma, color=col, lw=1.5, label=f'n={n_s}')
ax.axvline(0.5, color='black', ls='--', lw=1.5, label='σ=½ (floor)')
ax.axhline(0, color='gray', ls=':', lw=0.8)
ax.set_xlabel('σ'); ax.set_ylabel('c_n(σ)')
ax.set_title('Cascade trajectory  c_n(σ) vs σ')
ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

# Panel C: Correlation with Euler residual
ax = axes[1, 0]
x_corr = np.log1p(cn_abs_comps)
y_corr = np.log1p(er_comps)
ax.scatter(x_corr, y_corr, s=4, alpha=0.5, color='purple')
ax.set_xlabel('log(1 + |c_n|)'); ax.set_ylabel('log(1 + |R_n(½)−1|)')
ax.set_title(f'Correlation of c_n with Euler residual\nr={corr:.4f} (composites)')
ax.grid(True, alpha=0.3)

# Panel D: DR(g*) eigenvalue spectrum
ax = axes[1, 1]
evals_plot = evals_R.real
evals_imag = evals_R.imag
ax.scatter(evals_plot, evals_imag, s=20, color='darkgreen', zorder=3)
ax.axvline(DELTA, color='red', ls='--', lw=1.5, label=f'δ={DELTA:.4f}')
ax.axvline(-1, color='orange', ls=':', lw=1, label='|λ|=1')
ax.axvline(1, color='orange', ls=':', lw=1)
ax.set_xlabel('Re(λ)'); ax.set_ylabel('Im(λ)')
ax.set_title('DR(g*) eigenvalue spectrum')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

plt.tight_layout()
OUTDIR = "/Users/lucianrandolph/Downloads/Projects/einstein_fractal_analysis/The_Last_Law"
fig_path = f"{OUTDIR}/script86_cn_projection.png"
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"  Saved: {fig_path}")
plt.close()


# ─────────────────────────────────────────────────────────────────
# Step 13 · Classification accuracy
# ─────────────────────────────────────────────────────────────────
print("\n[11] Classification test (c_n(1/2) < threshold → PRIME)...")

# Find optimal threshold
all_cn   = np.concatenate([cn_primes, cn_comps])
all_lab  = np.array([1]*len(cn_primes) + [0]*len(cn_comps))  # 1=prime

thresholds = np.linspace(all_cn.min(), all_cn.max(), 10000)
best_acc   = 0; best_thresh = 0
for t in thresholds:
    pred   = (all_cn < t).astype(int)
    acc    = (pred == all_lab).mean()
    if acc > best_acc:
        best_acc = acc; best_thresh = t

print(f"  Optimal threshold:  {best_thresh:.4e}")
print(f"  Classification accuracy: {best_acc*100:.2f}%")

pred_best  = (all_cn < best_thresh).astype(int)
TP = ((pred_best == 1) & (all_lab == 1)).sum()
TN = ((pred_best == 0) & (all_lab == 0)).sum()
FP = ((pred_best == 1) & (all_lab == 0)).sum()
FN = ((pred_best == 0) & (all_lab == 1)).sum()
print(f"  TP={TP} TN={TN} FP={FP} FN={FN}")


# ─────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("SCRIPT 86 SUMMARY")
print("="*65)
print(f"  Fixed-point residual   ‖R[g*]−g*‖_∞ = {fp_err:.2e}")
print(f"  Eigenvalue found       λ = {lambda_R:.6f}  (δ = {DELTA:.6f})")
print(f"  Eigenvalue error       {abs(lambda_R-DELTA)/DELTA:.2e}")
print()
print(f"  Primes  (n≤200):  mean|c_p| = {cn_primes.mean():.4e}  max = {cn_primes.max():.4e}")
print(f"  Compos. (n≤200):  mean|c_n| = {cn_comps.mean():.4e}  min = {cn_comps.min():.4e}")
print(f"  Separation ratio  = {ratio:.4e}")
print(f"  Classification accuracy = {best_acc*100:.2f}%")
print(f"  Euler-residual corr.    = {corr:.6f}")
print()
if ratio > 100:
    verdict = "STRONG separation — Theorem C1 well-supported geometrically"
elif ratio > 10:
    verdict = "GOOD separation — consistent with Theorem C1"
elif ratio > 1:
    verdict = "PARTIAL separation — formula captures signal, refinement needed"
else:
    verdict = "NO separation at this polynomial degree — theory needs work"
print(f"  Verdict: {verdict}")
print("="*65)
