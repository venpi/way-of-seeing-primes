#!/usr/bin/env python3
"""
Prime Interval Data Harvester - VERSION 3.1 (FINAL)
====================================================

Fixes from v3:
- Renamed to largest_prime_in_list for clarity, added sieve_bound
- Removed prime_set entirely - use bisect for fulcrum primality (memory safe for 68M)
- Added consistency check in verification

Convention (THE LAW):
- primes[0] = 2 corresponds to p_1 = 2
- p_i = primes[i-1]
- Interval I_i = [2p_i, 2p_{i+1})
- Fulcrum F_i = p_i + p_{i+1} + 1

Usage:
    python prime_harvest_v3_1.py --intervals 1000000 --output harvest_1M.json --verbose
    python prime_harvest_v3_1.py --intervals 68000000 --output harvest_68M.json --verbose
"""

import argparse
import json
import math
import sys
import time
from bisect import bisect_left
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

# ============================================================================
# PRIME CORE (THE KEYSTONE)
# ============================================================================

def sieve(n: int) -> Tuple[List[int], int]:
    """
    Return list of primes <= n and the bound used.
    Returns: (primes, sieve_bound)
    """
    if n < 2:
        return [], n
    is_prime = bytearray(b"\x01") * (n + 1)
    is_prime[0:2] = b"\x00\x00"
    for p in range(2, int(n**0.5) + 1):
        if is_prime[p]:
            is_prime[p*p::p] = b"\x00" * (((n - p*p) // p) + 1)
    return [i for i in range(n + 1) if is_prime[i]], n


def pnt_upper_bound_for_nth_prime(n: int) -> int:
    """A safe upper bound for the nth prime for n >= 1."""
    if n < 6:
        small = [0, 2, 3, 5, 7, 11, 13]
        return small[n] if n < len(small) else 100
    # Dusart: p_n < n * (ln(n) + ln(ln(n))) for n >= 6
    x = n * (math.log(n) + math.log(math.log(n)))
    return int(x) + 100  # Safety margin


def primes_for_intervals(n_intervals: int, safety_factor: float = 1.05) -> Tuple[List[int], Dict]:
    """
    Return primes list guaranteed to cover:
    - Indices up to n_intervals + 1 (i.e., p_{n_intervals+1})
    - Values up to max(2 * p_{n_intervals+1} - 1, F_{n_intervals})
    
    Also returns coverage metadata for verification.
    """
    needed_index = n_intervals + 1  # We need p_{n_intervals+1} = primes[n_intervals]
    bound = pnt_upper_bound_for_nth_prime(needed_index + 1)
    bound = int(bound * safety_factor)
    primes, sieve_bound = sieve(bound)
    
    # Ensure we have enough primes by index
    while len(primes) < needed_index + 1:
        bound = int(bound * 1.25)
        primes, sieve_bound = sieve(bound)
    
    # Compute coverage requirements:
    # - Largest interval end: 2 * p_{n_intervals+1} = 2 * primes[n_intervals]
    # - Largest candidate in interval: 2 * primes[n_intervals] - 1
    # - Largest fulcrum: F_{n_intervals} = p_{n_intervals} + p_{n_intervals+1} + 1
    #                  = primes[n_intervals-1] + primes[n_intervals] + 1
    
    max_interval_candidate = 2 * primes[n_intervals] - 1
    max_fulcrum = primes[n_intervals - 1] + primes[n_intervals] + 1
    max_needed = max(max_interval_candidate, max_fulcrum)
    
    # Ensure sieve covers max_needed
    if primes[-1] < max_needed:
        bound2 = int(max_needed * 1.05) + 100
        primes, sieve_bound = sieve(bound2)
    
    # Final assertions
    assert len(primes) > n_intervals, \
        f"Not enough primes: {len(primes)} <= {n_intervals}"
    assert primes[-1] >= max_needed, \
        f"Prime list doesn't cover max needed: {primes[-1]} < {max_needed}"
    
    coverage_metadata = {
        "sieve_bound": sieve_bound,
        "largest_prime_in_list": primes[-1],  # Largest prime in final list (≤ sieve_bound)
        "primes_generated": len(primes),
        "max_interval_candidate": max_interval_candidate,
        "max_fulcrum": max_fulcrum,
        "max_needed": max_needed,
        "coverage_ok": primes[-1] >= max_needed
    }
    
    return primes, coverage_metadata


def interval_from_i(primes: List[int], i: int) -> Tuple[int, int, int, int, int]:
    """
    Paper-consistent mapping:
    primes[0] = 2 => p_1 = 2. So p_i = primes[i-1].
    Interval I_i = [2p_i, 2p_{i+1})
    Fulcrum F_i = p_i + p_{i+1} + 1
    
    Returns: (p_i, p_{i+1}, start, end, fulcrum)
    """
    p_i = primes[i - 1]      # p_i
    p_next = primes[i]       # p_{i+1}
    start = 2 * p_i
    end = 2 * p_next
    fulcrum = p_i + p_next + 1
    return p_i, p_next, start, end, fulcrum


def primes_in_interval_bisect(primes: List[int], start: int, end: int) -> Tuple[int, int, int]:
    """
    Find primes in [start, end) using bisect.
    Returns: (lo, hi, count) where primes[lo:hi] are the primes in the interval.
    """
    lo = bisect_left(primes, start)
    hi = bisect_left(primes, end)
    return lo, hi, hi - lo


def count_left_right_and_fulcrum(primes: List[int], lo: int, hi: int, fulcrum: int) -> Tuple[int, int, bool]:
    """
    Count primes left of fulcrum and at/right of fulcrum.
    Also check if fulcrum is prime (using bisect, no set needed).
    
    Left: primes < fulcrum
    Right: primes >= fulcrum (fulcrum included if prime)
    
    Returns: (left_count, right_count, fulcrum_is_prime)
    """
    mid = bisect_left(primes, fulcrum, lo, hi)
    left = mid - lo
    right = hi - mid
    
    # Fulcrum is prime iff primes[mid] == fulcrum (and mid is valid)
    fulcrum_is_prime = (mid < hi and primes[mid] == fulcrum)
    
    return left, right, fulcrum_is_prime


def is_perfect_square(n: int) -> Tuple[bool, Optional[int]]:
    """Check if n is a perfect square, return (is_square, root)."""
    if n < 0:
        return False, None
    root = int(math.isqrt(n))
    if root * root == n:
        return True, root
    return False, None


# ============================================================================
# MAIN HARVEST FUNCTION
# ============================================================================

def harvest_prime_data(n_intervals: int, verbose: bool = False) -> Tuple[Dict, Dict]:
    """
    Harvest comprehensive prime interval data using bisect for exact counting.
    
    This harvester collects RAW DATA only. Model comparisons (e.g., the -1/(2g)
    deviation formula) should be done in separate analysis scripts.
    
    Returns: (results_dict, coverage_metadata)
    """
    start_time = time.time()
    
    # ========================================================================
    # PHASE 1: Generate primes with coverage guarantee
    # ========================================================================
    if verbose:
        print("=" * 70)
        print("PRIME INTERVAL DATA HARVESTER (V3.1 - FINAL)")
        print("=" * 70)
        print(f"Target: {n_intervals:,} intervals")
        print(f"\n[1/3] GENERATING PRIMES")
    
    sieve_start = time.time()
    primes, coverage_meta = primes_for_intervals(n_intervals)
    sieve_time = time.time() - sieve_start
    
    if verbose:
        print(f"  Primes generated: {coverage_meta['primes_generated']:,}")
        print(f"  Sieve bound: {coverage_meta['sieve_bound']:,}")
        print(f"  Largest prime: {coverage_meta['largest_prime_in_list']:,}")
        print(f"  Max interval candidate: {coverage_meta['max_interval_candidate']:,}")
        print(f"  Max fulcrum: {coverage_meta['max_fulcrum']:,}")
        print(f"  Coverage OK: {coverage_meta['coverage_ok']}")
        print(f"  Sieve time: {sieve_time:.2f}s")
        print(f"  (No prime_set built - using bisect for memory efficiency)")
    
    # NO prime_set - we use bisect for fulcrum primality check
    
    # ========================================================================
    # PHASE 2: Analyze intervals using bisect
    # ========================================================================
    if verbose:
        print(f"\n[2/3] ANALYZING INTERVALS")
    
    phase2_start = time.time()
    
    # Aggregates
    total_primes_found = 0
    total_left = 0
    total_right = 0
    empty_count = 0
    prime_fulcrum_count = 0
    square_fulcrum_count = 0
    
    by_gap = defaultdict(lambda: {
        "count": 0,
        "primes_found": 0,
        "primes_left": 0,
        "primes_right": 0,
        "empty": 0,
        "prime_fulcrum": 0
    })
    
    # Sample storage (for verification)
    samples = []
    
    # Progress step (avoid divide by zero for small N)
    progress_step = max(1, n_intervals // 20)
    
    # Streak tracking for fulcrum analysis
    streaks = []
    current_streak_start = None
    
    for i in range(1, n_intervals + 1):
        # Get interval using paper convention
        p_i, p_next, start, end, fulcrum = interval_from_i(primes, i)
        gap = p_next - p_i
        
        # Find primes in interval using bisect
        lo, hi, count = primes_in_interval_bisect(primes, start, end)
        
        # Count left/right of fulcrum AND check fulcrum primality (all via bisect)
        left, right, fulcrum_is_prime = count_left_right_and_fulcrum(primes, lo, hi, fulcrum)
        
        # Check if fulcrum is perfect square
        is_square, square_root = is_perfect_square(fulcrum)
        
        # Update aggregates
        total_primes_found += count
        total_left += left
        total_right += right
        
        is_empty = (count == 0)
        if is_empty:
            empty_count += 1
        if fulcrum_is_prime:
            prime_fulcrum_count += 1
        if is_square:
            square_fulcrum_count += 1
        
        # By gap
        gap_data = by_gap[gap]
        gap_data["count"] += 1
        gap_data["primes_found"] += count
        gap_data["primes_left"] += left
        gap_data["primes_right"] += right
        if is_empty:
            gap_data["empty"] += 1
        if fulcrum_is_prime:
            gap_data["prime_fulcrum"] += 1
        
        # Streak tracking
        if is_empty:
            if current_streak_start is None:
                current_streak_start = i
        else:
            if current_streak_start is not None:
                # End of streak - record it with boundary info
                streak_end = i - 1
                streak_len = streak_end - current_streak_start + 1
                
                # I_a = interval before streak (current_streak_start - 1)
                # I_b = interval after streak (i) = current interval
                idx_a = current_streak_start - 1
                idx_b = i
                
                streak_info = {
                    "start_i": current_streak_start,
                    "end_i": streak_end,
                    "length": streak_len,
                    "a": idx_a,  # Index of left bounding interval
                    "b": idx_b,  # Index of right bounding interval
                    "has_left_boundary": idx_a >= 1,
                    "has_right_boundary": True,
                }
                
                # Add left boundary info if we have valid I_a
                if idx_a >= 1:
                    p_a, p_a_next, start_a, end_a, F_a = interval_from_i(primes, idx_a)
                    lo_a, hi_a, _ = primes_in_interval_bisect(primes, start_a, end_a)
                    if lo_a < hi_a:  # I_a is non-empty (should always be true)
                        q = primes[hi_a - 1]  # Last prime in I_a
                        streak_info["p_a"] = p_a
                        streak_info["p_a_plus_1"] = p_a_next
                        streak_info["q"] = q
                        streak_info["F_a"] = F_a
                        streak_info["gap_a"] = p_a_next - p_a
                
                # Add right boundary info (I_b is current interval, known non-empty)
                # p_i and p_next are already computed for current interval
                streak_info["p_b"] = p_i
                streak_info["p_b_plus_1"] = p_next
                streak_info["r"] = primes[lo]  # First prime in I_b
                streak_info["F_b"] = fulcrum
                streak_info["gap_b"] = gap
                
                # Compute derived values for fulcrum analysis (if both boundaries exist)
                if "q" in streak_info and "r" in streak_info:
                    gap_qr = streak_info["r"] - streak_info["q"]
                    delta_F = streak_info["F_b"] - streak_info["F_a"]
                    streak_info["gap_qr"] = gap_qr      # r - q (actual prime gap)
                    streak_info["delta_F"] = delta_F    # F_b - F_a (fulcrum distance)
                    streak_info["exceeds"] = gap_qr > delta_F  # Does gap exceed fulcrum bound?
                
                # Sanity check (belt-and-suspenders)
                assert current_streak_start <= streak_end, "Streak start > end"
                assert idx_a == current_streak_start - 1, "idx_a mismatch"
                assert idx_b == i, "idx_b mismatch"
                
                streaks.append(streak_info)
                current_streak_start = None
        
        # Store samples (first 100 + every 10000th)
        if i <= 100 or i % 10000 == 0:
            samples.append({
                "i": i,
                "p_i": p_i,
                "p_next": p_next,
                "gap": gap,
                "start": start,
                "end": end,
                "fulcrum": fulcrum,
                "primes_in_interval": count,
                "left": left,
                "right": right,
                "is_empty": is_empty,
                "fulcrum_is_prime": fulcrum_is_prime
            })
        
        # Progress
        if verbose and i % progress_step == 0:
            elapsed = time.time() - phase2_start
            pct = i / n_intervals * 100
            eta = elapsed / (i / n_intervals) - elapsed if i > 0 else 0
            print(f"  Progress: {pct:.0f}% ({i:,}/{n_intervals:,}) - ETA: {eta:.0f}s")
    
    phase2_time = time.time() - phase2_start
    
    # Handle streak at end of data (terminal streak - no right boundary)
    if current_streak_start is not None:
        streak_end = n_intervals
        idx_a = current_streak_start - 1
        
        streak_info = {
            "start_i": current_streak_start,
            "end_i": streak_end,
            "length": streak_end - current_streak_start + 1,
            "a": idx_a,
            "b": None,  # No right boundary
            "has_left_boundary": idx_a >= 1,
            "has_right_boundary": False,
        }
        
        # Add left boundary info if available
        if idx_a >= 1:
            p_a, p_a_next, start_a, end_a, F_a = interval_from_i(primes, idx_a)
            lo_a, hi_a, _ = primes_in_interval_bisect(primes, start_a, end_a)
            if lo_a < hi_a:
                streak_info["p_a"] = p_a
                streak_info["p_a_plus_1"] = p_a_next
                streak_info["q"] = primes[hi_a - 1]
                streak_info["F_a"] = F_a
                streak_info["gap_a"] = p_a_next - p_a
        
        streaks.append(streak_info)
    
    # ========================================================================
    # PHASE 3: Build results
    # ========================================================================
    if verbose:
        print(f"\n[3/3] BUILDING RESULTS")
    
    total_time = time.time() - start_time
    
    results = {
        "metadata": {
            "version": "3.1",
            "description": "Raw interval data. Model comparisons in separate scripts.",
            "indexing_convention": "paper: primes[0]=2 => p_i=primes[i-1]",
            "left_right_convention": "left: primes < fulcrum; right: primes >= fulcrum",
            "n_intervals": n_intervals,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "primes_generated": coverage_meta["primes_generated"],
            "sieve_bound": coverage_meta["sieve_bound"],
            "largest_prime_in_list": coverage_meta["largest_prime_in_list"],
            "p_n": primes[n_intervals - 1],      # p_n (the n-th prime, for threshold formulas)
            "p_n_plus_1": primes[n_intervals],   # p_{n+1} (defines last interval end)
            "max_interval_candidate": coverage_meta["max_interval_candidate"],
            "max_fulcrum": coverage_meta["max_fulcrum"],
            "max_needed": coverage_meta["max_needed"],
            "coverage_ok": coverage_meta["coverage_ok"],
            "sieve_time_seconds": sieve_time,
            "analysis_time_seconds": phase2_time,
            "total_time_seconds": total_time
        },
        "overall": {
            "total_primes": total_primes_found,
            "left_of_fulcrum": total_left,
            "right_of_fulcrum": total_right,
            "left_fraction": total_left / total_primes_found if total_primes_found > 0 else 0,
            "right_fraction": total_right / total_primes_found if total_primes_found > 0 else 0
        },
        "intervals": {
            "total": n_intervals,
            "empty_count": empty_count,
            "empty_rate": empty_count / n_intervals,
            "prime_fulcrum_count": prime_fulcrum_count,
            "prime_fulcrum_rate": prime_fulcrum_count / n_intervals,
            "square_fulcrum_count": square_fulcrum_count
        },
        "by_gap": {str(k): v for k, v in sorted(by_gap.items())},
        "streaks": streaks,  # For fulcrum analysis - includes boundary info
        "samples": samples
    }
    
    if verbose:
        print(f"\n{'=' * 70}")
        print("HARVEST COMPLETE")
        print(f"{'=' * 70}")
        print(f"Total time: {total_time:.1f}s")
        print(f"Intervals analyzed: {n_intervals:,}")
        print(f"Primes found in intervals: {total_primes_found:,}")
        print(f"\nBALANCE PROPERTY (RAW):")
        print(f"  Left of fulcrum:  {total_left:,} ({results['overall']['left_fraction']:.6f})")
        print(f"  Right of fulcrum: {total_right:,} ({results['overall']['right_fraction']:.6f})")
        print(f"\nINTERVALS:")
        print(f"  Empty: {empty_count:,} ({results['intervals']['empty_rate']*100:.4f}%)")
        print(f"  Prime fulcrums: {prime_fulcrum_count:,} ({results['intervals']['prime_fulcrum_rate']*100:.2f}%)")
        print(f"  Empty streaks: {len(streaks):,}")
    
    return results, coverage_meta


# ============================================================================
# VERIFICATION FUNCTIONS
# ============================================================================

def verify_samples(results: Dict, primes: List[int], coverage_meta: Dict) -> bool:
    """Verify samples against direct computation."""
    print("\nVERIFYING SAMPLES...")
    
    # First check coverage metadata consistency
    stored_meta = results["metadata"]
    if stored_meta["max_needed"] != coverage_meta["max_needed"]:
        print(f"❌ max_needed mismatch: stored={stored_meta['max_needed']}, computed={coverage_meta['max_needed']}")
        return False
    if not stored_meta["coverage_ok"]:
        print(f"❌ coverage_ok is False in stored results!")
        return False
    print(f"  Coverage metadata consistent ✓")
    
    errors = []
    for sample in results["samples"][:20]:
        i = sample["i"]
        p_i, p_next, start, end, fulcrum = interval_from_i(primes, i)
        
        # Check values match
        if sample["p_i"] != p_i:
            errors.append(f"p_i mismatch at i={i}: {sample['p_i']} vs {p_i}")
        if sample["p_next"] != p_next:
            errors.append(f"p_next mismatch at i={i}")
        if sample["start"] != start:
            errors.append(f"start mismatch at i={i}")
        if sample["end"] != end:
            errors.append(f"end mismatch at i={i}")
        if sample["fulcrum"] != fulcrum:
            errors.append(f"fulcrum mismatch at i={i}")
        
        # Direct count
        lo, hi, count = primes_in_interval_bisect(primes, start, end)
        left, right, fulcrum_is_prime = count_left_right_and_fulcrum(primes, lo, hi, fulcrum)
        
        if sample["primes_in_interval"] != count:
            errors.append(f"count mismatch at i={i}: {sample['primes_in_interval']} vs {count}")
        if sample["left"] != left:
            errors.append(f"left mismatch at i={i}: {sample['left']} vs {left}")
        if sample["right"] != right:
            errors.append(f"right mismatch at i={i}: {sample['right']} vs {right}")
        if sample["fulcrum_is_prime"] != fulcrum_is_prime:
            errors.append(f"fulcrum_is_prime mismatch at i={i}")
        
        print(f"  i={i}: p_i={p_i}, p_{{i+1}}={p_next}, I=[{start},{end}), "
              f"F={fulcrum}, primes={count} (L:{left}, R:{right}) ✓")
    
    if errors:
        print(f"\n❌ ERRORS FOUND:")
        for e in errors:
            print(f"  {e}")
        return False
    
    print("✓ All samples verified!")
    return True


def verify_by_gap_consistency(results: Dict) -> bool:
    """Verify that by_gap totals match overall totals."""
    print("\nVERIFYING BY-GAP CONSISTENCY...")
    
    total_intervals = sum(d["count"] for d in results["by_gap"].values())
    total_primes = sum(d["primes_found"] for d in results["by_gap"].values())
    total_left = sum(d["primes_left"] for d in results["by_gap"].values())
    total_right = sum(d["primes_right"] for d in results["by_gap"].values())
    total_empty = sum(d["empty"] for d in results["by_gap"].values())
    
    errors = []
    
    if total_intervals != results["intervals"]["total"]:
        errors.append(f"Interval count: by_gap={total_intervals}, overall={results['intervals']['total']}")
    if total_primes != results["overall"]["total_primes"]:
        errors.append(f"Prime count: by_gap={total_primes}, overall={results['overall']['total_primes']}")
    if total_left != results["overall"]["left_of_fulcrum"]:
        errors.append(f"Left count: by_gap={total_left}, overall={results['overall']['left_of_fulcrum']}")
    if total_right != results["overall"]["right_of_fulcrum"]:
        errors.append(f"Right count: by_gap={total_right}, overall={results['overall']['right_of_fulcrum']}")
    if total_empty != results["intervals"]["empty_count"]:
        errors.append(f"Empty count: by_gap={total_empty}, overall={results['intervals']['empty_count']}")
    
    if errors:
        print(f"❌ CONSISTENCY ERRORS:")
        for e in errors:
            print(f"  {e}")
        return False
    
    print(f"  Intervals: {total_intervals:,} ✓")
    print(f"  Primes: {total_primes:,} ✓")
    print(f"  Left: {total_left:,} ✓")
    print(f"  Right: {total_right:,} ✓")
    print(f"  Empty: {total_empty:,} ✓")
    print("✓ All totals consistent!")
    return True


def verify_streaks(results: Dict, primes: List[int]) -> bool:
    """
    Verify streak invariants:
    1. length == end_i - start_i + 1
    2. Streaks are disjoint and ordered
    3. Boundary intervals are non-empty (when has_left/right_boundary is True)
    """
    print("\nVERIFYING STREAK INVARIANTS...")
    
    streaks = results.get("streaks", [])
    if not streaks:
        print("  No streaks to verify")
        return True
    
    errors = []
    prev_end = 0
    
    for idx, streak in enumerate(streaks):
        start_i = streak["start_i"]
        end_i = streak["end_i"]
        length = streak["length"]
        
        # Check 1: length consistency
        expected_length = end_i - start_i + 1
        if length != expected_length:
            errors.append(f"Streak {idx}: length={length} but expected {expected_length}")
        
        # Check 2: disjoint and ordered
        if start_i <= prev_end:
            errors.append(f"Streak {idx}: start_i={start_i} overlaps with previous end={prev_end}")
        prev_end = end_i
        
        # Check 3: left boundary non-empty
        if streak.get("has_left_boundary", False):
            a = streak.get("a", start_i - 1)
            if a >= 1:
                p_a, p_a_next, start_a, end_a, F_a = interval_from_i(primes, a)
                lo_a, hi_a, _ = primes_in_interval_bisect(primes, start_a, end_a)
                if lo_a == hi_a:
                    errors.append(f"Streak {idx}: left boundary I_{a} is empty but has_left_boundary=True")
        
        # Check 4: right boundary non-empty
        if streak.get("has_right_boundary", False):
            b = streak.get("b", end_i + 1)
            if b is not None:
                p_b, p_b_next, start_b, end_b, F_b = interval_from_i(primes, b)
                lo_b, hi_b, _ = primes_in_interval_bisect(primes, start_b, end_b)
                if lo_b == hi_b:
                    errors.append(f"Streak {idx}: right boundary I_{b} is empty but has_right_boundary=True")
        
        # Check 5: derived values consistency (if present)
        if "gap_qr" in streak and "q" in streak and "r" in streak:
            expected_gap = streak["r"] - streak["q"]
            if streak["gap_qr"] != expected_gap:
                errors.append(f"Streak {idx}: gap_qr={streak['gap_qr']} but r-q={expected_gap}")
        
        if "delta_F" in streak and "F_a" in streak and "F_b" in streak:
            expected_delta = streak["F_b"] - streak["F_a"]
            if streak["delta_F"] != expected_delta:
                errors.append(f"Streak {idx}: delta_F={streak['delta_F']} but F_b-F_a={expected_delta}")
    
    if errors:
        print(f"❌ STREAK ERRORS:")
        for e in errors[:10]:  # Limit output
            print(f"  {e}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors")
        return False
    
    print(f"  Streaks checked: {len(streaks):,}")
    print(f"  All lengths correct ✓")
    print(f"  All disjoint and ordered ✓")
    print(f"  All boundary conditions valid ✓")
    print("✓ All streak invariants satisfied!")
    return True


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Harvest prime interval data (v3.1 - final, memory-efficient)",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--intervals", "-n", type=int, default=100_000,
                        help="Number of intervals to analyze")
    parser.add_argument("--output", "-o", type=str, default="harvest_v3_1.json",
                        help="Output JSON file")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print progress")
    parser.add_argument("--verify", action="store_true",
                        help="Run verification after harvest")
    
    args = parser.parse_args()
    
    # Run harvest
    results, coverage_meta = harvest_prime_data(n_intervals=args.intervals, verbose=args.verbose)
    
    # Optional verification
    if args.verify:
        primes, verify_coverage_meta = primes_for_intervals(args.intervals)
        ok1 = verify_samples(results, primes, verify_coverage_meta)
        ok2 = verify_by_gap_consistency(results)
        ok3 = verify_streaks(results, primes)
        if not (ok1 and ok2 and ok3):
            print("\n⚠️  VERIFICATION FAILED")
            return 1
    
    # Save
    if args.verbose:
        print(f"\nSaving to {args.output}...")
    
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    
    if args.verbose:
        print(f"Done!")
        
        # Summary by gap
        print(f"\n{'=' * 70}")
        print("RAW DATA BY GAP (first 15 gaps)")
        print(f"{'=' * 70}")
        print(f"{'Gap':<6} {'Count':<10} {'Primes':<10} {'Left':<10} {'Right':<10} {'Empty':<8} {'Empty%':<8}")
        print("-" * 62)
        
        for gap in sorted([int(g) for g in results["by_gap"].keys()])[:15]:
            d = results["by_gap"][str(gap)]
            print(f"{gap:<6} {d['count']:<10,} {d['primes_found']:<10,} "
                  f"{d['primes_left']:<10,} {d['primes_right']:<10,} "
                  f"{d['empty']:<8,} {d['empty']/d['count']*100:>6.2f}%")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
