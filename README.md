# A Way of Seeing Primes

**Observations on Doubled Intervals and a Structural Path to Cramér**

[![License: CC0](https://img.shields.io/badge/License-CC0_1.0-lightgrey.svg)](https://creativecommons.org/publicdomain/zero/1.0/)

---

## Overview

This repository contains the paper, code, and data for "A Way of Seeing Primes" — an investigation into prime distribution through the lens of **doubled prime intervals**.

The key contributions are:

1. **The Index Bound Theorem** (proven): Interval *i* requires only √(2i) primes for compositeness verification
2. **The Balance Property** (empirical): Primes split ~50-50 around the fulcrum despite asymmetric geometry
3. **The Safe Gap Conjecture** (conjectured): A threshold formula predicting when intervals contain primes
4. **A Structural Path to Cramér**: If the Safe Gap Conjecture holds, it implies Cramér's bound on prime gaps

## The Core Idea

We view primes as *generators*: numbers that must exist because nothing smaller can create them.

We organize the integers using **doubled prime intervals**:

```
I_i = [2p_i, 2p_{i+1})
```

These intervals partition all integers ≥ 4. The **fulcrum** of each interval:

```
F_i = p_i + p_{i+1} + 1
```

acts as an arithmetic center of mass around which primes balance with surprising precision.

## Key Results

### Proven
- **Index Bound Theorem**: Every composite in interval *I_i* has a prime divisor among the first √(2i) primes
- **Recursive Bootstrap**: All primes derive from {2, 3} via the Index Bound
- **Residue Framework**: The "Danger Zone Criterion" determines compositeness via modular arithmetic

### Empirically Verified (68 Million Intervals)
- **Balance Property**: 131M primes split 49.997% / 50.003% around the fulcrum (deviation: 0.003%)
- **Deviation Formula**: Deviation from 50% ≈ −1/(2g) where g is the gap size
- **Safe Gap Threshold**: T(n) tracks the emptiness frontier with positive margin at 3 of 4 tested scales

### Conjectured
- **Safe Gap Conjecture**: 
  ```
  If g_i ≥ (1/π) · [ln(2)·ln(2i)·ln(p_i) + 2·ln(i)],
  then interval I_i contains at least one prime.
  ```
- **Structural Path to Cramér**: If Safe Gap + 2× Rule hold, then maximum prime gaps ≤ ln²(p)

## Repository Structure

```
├── README.md                 # This file
├── LICENSE                   # CC0 Public Domain
├── paper/
│   └── A_Way_of_Seeing_Primes.pdf   # The full paper
├── scripts/
│   ├── prime_harvest_v3_1.py      # Core interval analysis
│   ├── safe_gap_analysis.py  # Safe Gap threshold verification
│   ├── fulcrum_analysis.py   # Fulcrum condition analysis
│   └── visualize_frontier.py # Generate threshold vs frontier charts
├── data/
│   └── harvest_68M_v3_1.json      # Precomputed results (68M intervals)
└── examples/
    └── quickstart.py         # Simple example to get started
```

## Quick Start

### Requirements
- Python 3.x (standard library only — no external dependencies)

### Run a Small Analysis

```bash
# Analyze 100,000 intervals (takes ~30 seconds)
python scripts/prime_harvest_v3_1.py --intervals 100000 --output results_100k.json --verbose
```

### Verify the Safe Gap Threshold

```bash
# Show formula predictions at various scales
python scripts/safe_gap_analysis.py --predictions

# Verify against precomputed 68M data
python scripts/safe_gap_analysis.py --data harvest_68M_v3_1.json
```

### Reproduce Key Results

```bash
# Full 68M analysis (takes ~2-3 hours, requires ~8GB RAM)
python scripts/prime_harvest_v3_1.py --intervals 68000000 --output harvest_68M_v3_1.json --verbose
```

## The Traversal Argument (Path to Cramér)

The most significant implication of the framework:

| Step | Description | Maximum Size |
|------|-------------|--------------|
| 1 | Begin just after a prime | 0 |
| 2 | Traverse consecutive empty intervals | ≈ ⅔ ln²(2p) |
| 3 | Streak terminates (by Safe Gap) | — |
| 4 | Prime at far end of next interval | ≈ ⅓ ln²(2p) |
| **Total** | **Maximum prime gap** | **≈ ln²(p)** |

This recovers **Cramér's bound** structurally, not probabilistically.

## Citation

If you use this work, please cite:

```bibtex
@article{ilieva2026primes,
  title={A Way of Seeing Primes: Observations on Doubled Intervals and a Structural Path to Cramér},
  author={Ilieva, Veneta},
  year={2026},
  note={Available at: https://github.com/[username]/way-of-seeing-primes}
}
```

## Acknowledgments

This work was developed in collaboration with Claude (Anthropic) and ChatGPT 5.2. Core intuitions, design and observations are human; code, articulation, challenge, and refinement emerged through dialogue.

Special thanks to mathematics teachers Nikola Ginkov, Maria Denkova, Georgi Geninsky, Maxim Yordanov, Daria Marinova, and Snezhina Nedyalkova.

## License

This work is released to the **public domain** under [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/).

---

*We looked, we saw something, we thought about it. Here is our best interpretation. We could be wrong. We hope it is useful.*

---

## Quick Start

### Prerequisites

- Python 3.8+
- No external dependencies (uses only standard library)

### Verify the Index Bound Theorem (Lemma 3)

```bash
# Full verification (computational + explicit bounds)
python scripts/lemma3_verification.py

# Output shows:
#   Part A: Direct verification for i ≤ 10,000 ✓
#   Part B: Explicit bounds proof for i > 10,000 ✓
```

### Verify the Balance Property

```bash
# Quick test (100K intervals)
python scripts/balance_verification.py --intervals 100000

# Full verification (68M intervals, ~45 min)
python scripts/balance_verification.py --intervals 68000000 --parallel
```

### Test the Safe Gap Conjecture

```bash
# Analyze safe gap threshold
python scripts/safe_gap_corrected.py --data data/results_68M.json

# Streak analysis
python scripts/verify_safe_gap_v3.py --intervals 1000000
```

---

## The Framework

### 1. Doubled-Prime Intervals

For consecutive primes p_i and p_{i+1}, define:

```
I_i = [2p_i, 2p_{i+1})
```

These intervals partition ℤ≥4. Each has length 2g_i where g_i = p_{i+1} − p_i.

### 2. The Fulcrum

```
F_i = p_i + p_{i+1} + 1
```

The fulcrum is the **odd projection of the midpoint** into the doubled interval. It sits at relative position (g+1)/(2g) — above center for small gaps, approaching 50% for large gaps.

### 3. The Balance Property

Despite the fulcrum sitting *above* the midpoint, primes split almost exactly 50-50 around it:

```
68 million intervals:
  Left of fulcrum:  49.997%
  Right of fulcrum: 50.003%
  Deviation:        0.003%
```

### 4. Generators, Not Survivors

The conventional view: primes are *survivors* of sieving.

Our view: primes are *generators*. From {2, 3} alone, all primes can be generated:
- {2, 3} cannot generate 5 → 5 is prime
- {2, 3, 5} cannot generate 7 → 7 is prime
- And so on...

The Index Bound Theorem proves this process is self-sustaining: at step i, we need ≤ √(2i) primes, but we've already discovered ~i primes.

---

## Key Empirical Results (68M intervals)

### Balance Property
```
Total intervals:     68,000,000
Total primes:        131,426,164
Left of fulcrum:     65,709,343 (49.997%)
Right of fulcrum:    65,716,821 (50.003%)
```

### Fulcrum Statistics
```
Prime fulcrums:      ~22.8% (vs ~10% for random odd numbers)
Square fulcrums:     ~1.5% (of these, 73% have prime base k)
```

### Safe Gap Verification
```
Gap    Empty Intervals    Status
< 88   Many               Expected
88-96  Exactly 1 each     Transition zone
≥ 98   Zero               SAFE ✓
```

---

## The Signature of 2

The factor of 2 echoes through every formula:
- Doubled intervals: [2p_i, 2p_{i+1})
- Index Bound: j ≤ √(**2**i)
- Fulcrum: F = p_i + p_{i+1} + 1 = **2**m + 1
- Safe Gap coefficient: ln(**2**)/π

The mathematics remembers the structure it emerged from.

---

## Mathematical Details

### Index Bound Proof Structure

**Lemma 1 (Elementary):** Every composite n has a prime factor p ≤ √n.

**Lemma 2 (Interval Bound):** For n ∈ I_i, we have n < 2p_{i+1}.

**Lemma 3 (Index-Value Bound):** If p_j ≤ √(2p_{i+1}), then j ≤ √(2i).

*Proof of Lemma 3:*
- Part A: Computational verification for i ≤ 10,000
- Part B: For i > 10,000, use Dusart's bounds:
  - Lower: p_k > k(ln k + ln ln k − 1)
  - Upper: p_k < k(ln k + ln ln k)
  
  Show p_{j₀}² > 31.2i > 23i > 2p_{i+1} for j₀ = ⌊√(2i)⌋ + 1

### Safe Gap Formula Derivation

The corrected formula has two terms:

```
g_safe = (ln2/π) · ln(2i) · ln(p_i) + (2/π) · ln(i)
       = (1/π) · [ln(2) · ln(2i) · ln(p_i) + 2 · ln(i)]
```

- First term: Primary threshold from interval structure
- Second term: Sample-size correction (lottery analogy)

---

## Related Work

This paper has a companion:

- **"The Parent-Child Framework"** — Investigates how primes propagate across generations, discovering Fulcrum Doubling (F_child/F_parent → 2) and Generational Compensation.

---

## Citation

```
Ilieva, V. * "A Way of Seeing Primes: Observations on Doubled Intervals 
and the Index Bound Theorem."* January 2026.
```

---

## License

This work is released to the **public domain**. Use it freely.

---

## Acknowledgments

This work was developed in collaboration with Claude (Anthropic). The collaboration was genuine: core observations and intuitions are human; articulation, challenge, and refinement emerged through dialogue. All mathematical claims were verified computationally.

---

*"The ancient Greeks distinguished two kinds of time: Chronos (the uniform) and Kairos (the rhythmic). We found ourselves thinking of this distinction when looking at primes."*
