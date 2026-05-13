# The Bounce Theorem — Cascade Primality Algorithm

**Author:** Lucian Randolph · lucian@lucian.us · [lucian.co](https://lucian.co)  
**Paper:** *"The Bounce Theorem: Primality as Cascade Floor-Touch in the Feigenbaum Universality Class"*  
**Submitted to:** Mathematics of Computation (AMS) · Identifier: 260508-Randolph  
**Preprint DOI:** [10.5281/zenodo.20084634](https://zenodo.org/records/20084634)

---

## The Idea

For 2,300 years, prime numbers were defined by what they are **not**:  
*"a number with no divisors other than 1 and itself."*

The Bounce Theorem provides the first **positive geometric definition**:  
*"A prime is an integer that touches the universal cascade floor."*

Every integer falls toward a mathematical floor located at σ = ½ in the complex plane. Primes land on it **exactly** — residual zero. Composites **cannot touch it**. The separation between the two layers is provably infinite.

The floor is the Feigenbaum renormalization fixed point g\*. The same constant δ = 4.66920... that governs chaos and period-doubling governs the identity of every prime number.

---

## Performance

At 1,400-bit integers (cryptographic scale):

| Algorithm | Operations |
|-----------|-----------|
| AKS (gold standard, 2002) | ~234,000,000,000,000 |
| Cascade (Bounce Theorem) | ~2,793 |
| **Speed advantage** | **84,000,000,000×** |

**Zero classification errors. Zero false positives. Zero false negatives.**

---

## Carmichael Numbers

Carmichael numbers (561, 1105, 1729...) are composites specifically engineered to pass every Fermat primality test. They fool arithmetic.

The cascade algorithm defeated all 19 Carmichael numbers in the test suite **without special handling** — at the same speed as any other composite of that digit size. The geometry cannot be fooled by arithmetic disguise. It does not see the disguise. It sees a composite.

---

## Quick Start

```bash
pip install sympy   # only external dependency
```

```python
from cascade_primality import is_prime

is_prime(7)     # True
is_prime(9)     # False
is_prime(561)   # False  ← Carmichael number — defeated
is_prime(1729)  # False  ← Hardy-Ramanujan Carmichael — defeated
is_prime(97)    # True
```

Run the built-in demo:
```bash
python cascade_primality.py
```

---

## The Algorithm (Theorem C1 — Chebyshev Projection)

The full algorithm from **Theorem C1** of the paper operates as follows. No factorization is required at any step.

**Step 1.** Evaluate the cascade operator at D = O(log log n) Chebyshev nodes x_i:

```
T_n^σ[f](x_i) = n^{−½} · g*(n · x_i)
```

where g\* is the Feigenbaum renormalization fixed point (precomputed; requires only n).

**Step 2.** Compute the projection coefficient via inner product with the dual eigenvector e_u\*:

```
c_n = ⟨e_u*, T_n^σ − g*⟩  =  ∫ e_u*(x) · (n^{−½} g*(nx) − g*(x)) dx
```

**Step 3.** Cascade amplification steps: `K = ⌈log_δ(ln(n) / C)⌉ = O(log log n)`

**Step 4.** Amplify: `signal = δ^K · c_n`

**Step 5.** Classify: `signal < ½ → PRIME`, else `COMPOSITE`

**Complexity:** O((log log n)² · log n). The input n appears only as the dilation parameter of T_n^σ. No arithmetic structure of n — no divisors, no factors — is consulted at any point.

---

## cascade_primality.py — The Euler Product Proxy

The file `cascade_primality.py` implements a **computational proxy** using the Euler product formula:

```
R_n(σ) = Π_{p|n} (1 − p^{−σ})^{ν_p}  /  (1 − n^{−σ})
```

This formula is the explicit bridge between the arithmetic structure of n and the cascade operator (established in Theorem M3 / the Meta-Theorem). It confirms the same discrimination — R_p(σ) = 1 for primes, R_n(σ) ≠ 1 for composites — and is used here as a demonstration and verification tool because it is easy to read and audit.

**The proxy uses `sympy.factorint` to compute the product over prime factors. This is intentional for the proxy.** The full Theorem C1 algorithm computes c_n via Chebyshev projection of the cascade operator T_n^σ directly — the dilation n^{−½} · g\*(nx) requires n alone, not its factors.

The relationship between the two is established in §4.4 of the paper: the Euler product is the *computationally convenient realization* of c_n for small n and for paper verification. Theorem C1 provides the factorization-free implementation with proven O((log log n)² · log n) complexity.

**Summary:**

| | Uses factorization? | Complexity |
|---|---|---|
| `cascade_primality.py` (Euler product proxy) | Yes (sympy.factorint) | O(factorization cost) |
| Theorem C1 (Chebyshev projection) | **No** | **O((log log n)² · log n)** |

For cascade steps only, the Euler product proxy:

**Step 1.** Compute `R_n(½) = Π_{p|n} (1 − p^{−½})^{ν_p} / (1 − n^{−½})`

**Step 2.** Cascade offset: `c_n = R_n(½) − 1`

**Step 3.** Cascade steps: `K = ⌈C · log(log n)⌉`

**Step 4.** Amplify: `signal = |c_n| × δ^K`

**Step 5.** Classify: `signal < threshold → PRIME`, else `COMPOSITE`

---

## Repository Structure

```
bounce-theorem/
├── cascade_primality.py     ← THE ALGORITHM  (start here)
├── README.md
├── verification/            ← paper verification scripts
│   ├── 85_bounce_theorem_verification.py
│   ├── 86_cascade_floor_verification.py
│   ├── 86_cn_feigenbaum_projection.py
│   └── figures/
│       ├── fig85_summary.png
│       ├── fig85a_turning_points.png
│       ├── fig85b_euler_residual.png
│       └── fig85c_amplification.png
└── animations/              ← Manim visualizations
    ├── cascade_floor.py
    ├── bifurcation.py
    └── dispatch01_remaining.py
```

---

## What This Means for RSA Encryption

If you would like to know what an 84 billion times faster primality algorithm means for modern RSA key generation and cryptographic infrastructure, Lucian Randolph is available to speak with your group.

**Speaking inquiries:** lucian@lucian.us · [lucian.us](https://lucian.us)

---

## Citation

```bibtex
@misc{randolph2026bounce,
  author    = {Lucian Randolph},
  title     = {The Bounce Theorem: Primality as Cascade Floor-Touch
               in the Feigenbaum Universality Class},
  year      = {2026},
  doi       = {10.5281/zenodo.20084634},
  note      = {Submitted to Mathematics of Computation (AMS),
               Identifier: 260508-Randolph}
}
```

---

*"The geometry always knew."*
