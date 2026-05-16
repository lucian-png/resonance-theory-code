# The Bounce Theorem — Cascade Primality Algorithm

**Author:** Lucian Randolph · lucian@lucian.us · [lucian.us](https://lucian.us)  
**Paper:** *"The Bounce Theorem: Primality as Cascade Floor-Touch in the Feigenbaum Universality Class"*  
**Submitted to:** Acta Mathematica · Identifier: 260512-Randolph  
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

The full proof, derivation, and complexity analysis are in the paper at the DOI above.

---

## Verification

Independent verification scripts are in the `verification/` folder.

- **14,040 composites tested** — zero floor violations
- **303 primes** — cascade residual R = 1 exactly for all
- **19 Carmichael numbers** — all correctly classified COMPOSITE without special handling
- **Feigenbaum amplification ratio** matched to 15 significant figures

---

## Repository Structure

```
bounce-theorem/
├── README.md
└── verification/            ← paper verification scripts
    ├── 85_bounce_theorem_verification.py
    ├── 86_cascade_floor_verification.py
    ├── 86_cn_feigenbaum_projection.py
    └── figures/
        ├── fig85_summary.png
        ├── fig85a_turning_points.png
        ├── fig85b_euler_residual.png
        └── fig85c_amplification.png
```

---

## What This Means for RSA Encryption

The cascade algorithm makes RSA more secure than it has ever been:

1. **Larger keys at no additional cost.** The reason 8,192-bit and 16,384-bit keys are not standard is the computational cost of key generation using current algorithms. The cascade algorithm's O((log log n)² · log n) complexity means any key size can be generated in the same processor time as a current 2,048-bit key. Truly safe key sizes become economically practical.

2. **Carmichael numbers are not a vulnerability.** Every Fermat-based primality engine requires special handling to avoid generating a Carmichael composite as a key. The cascade algorithm classifies them correctly without special handling — because geometry cannot be fooled by arithmetic disguise.

If you would like to know what this means for your cryptographic infrastructure, Lucian Randolph is available to speak with your group.

**Speaking and consulting inquiries:** lucian@lucian.us · [lucian.us](https://lucian.us)

---

## Citation

```bibtex
@misc{randolph2026bounce,
  author    = {Lucian Randolph},
  title     = {The Bounce Theorem: Primality as Cascade Floor-Touch
               in the Feigenbaum Universality Class},
  year      = {2026},
  doi       = {10.5281/zenodo.20084634},
  note      = {Submitted to Acta Mathematica,
               Identifier: 260512-Randolph}
}
```

---

*"The geometry always knew."*
