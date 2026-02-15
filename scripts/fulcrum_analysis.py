#!/usr/bin/env python3
"""
================================================================================
FULCRUM CONDITION FAILURES ANALYSIS (v3.1 Compatible)
================================================================================

Analyzes cases where the Fulcrum Bound conditions don't hold.

The Fulcrum Bound Theorem states:
    IF q ≥ F_a (last prime in I_a is at/right of fulcrum)
    AND r ≤ F_b (first prime in I_b is at/left of fulcrum)
    THEN gap = r - q ≤ F_b - F_a

This script investigates:
1. What percentage satisfy both conditions?
2. What happens in the ~32% where at least one fails?
3. Does Cramér still hold for actual gaps in failure cases?

FAST PATH: If harvest JSON includes streaks with boundary info (q, r, F_a, F_b),
           analysis runs instantly with NO prime generation needed.

SLOW PATH: If boundary info missing or --regenerate specified, generates primes
           and scans all intervals. Warning: requires significant RAM at scale.

Usage:
    python fulcrum_analysis.py harvest_68M_v3_1.json
    python fulcrum_analysis.py harvest_68M_v3_1.json --max-streaks 1000

Author: Collaborative work (Human + Claude)
Date: January 2026
================================================================================
"""

import json
import math
import argparse
import sys
from bisect import bisect_left
from collections import defaultdict
from typing import List, Dict, Tuple, Optional


# =============================================================================
# PRIME GENERATION (with coverage guarantee)
# =============================================================================

def sieve(n: int) -> List[int]:
    """Return list of primes <= n."""
    if n < 2:
        return []
    is_prime = bytearray(b"\x01") * (n + 1)
    is_prime[0:2] = b"\x00\x00"
    for p in range(2, int(n**0.5) + 1):
        if is_prime[p]:
            is_prime[p*p::p] = b"\x00" * (((n - p*p) // p) + 1)
    return [i for i in range(n + 1) if is_prime[i]]


def primes_for_intervals(n_intervals: int) -> Tuple[List[int], Dict]:
    """
    Generate primes with COVERAGE GUARANTEE for both:
    - Index coverage: need primes[0..n_intervals] (i.e., p_1 through p_{n+1})
    - Value coverage: need all primes up to max(2*p_{n+1}, p_n + p_{n+1} + 1)
    
    Returns: (primes, coverage_metadata)
    """
    # Step 1: Get enough primes by index
    needed_index = n_intervals + 1
    bound = int(needed_index * (math.log(needed_index) + math.log(math.log(needed_index))) * 1.2) + 100
    primes = sieve(bound)
    
    while len(primes) < needed_index + 1:
        bound = int(bound * 1.5)
        primes = sieve(bound)
    
    # Step 2: Compute value coverage requirement
    p_n = primes[n_intervals - 1]
    p_n_plus_1 = primes[n_intervals]
    max_interval_end = 2 * p_n_plus_1
    max_fulcrum = p_n + p_n_plus_1 + 1
    max_needed = max(max_interval_end, max_fulcrum)
    
    # Step 3: Re-sieve if value coverage insufficient
    if primes[-1] < max_needed:
        bound2 = int(max_needed * 1.05) + 100
        primes = sieve(bound2)
    
    # Final assertion
    assert primes[-1] >= max_needed, \
        f"Coverage failure: {primes[-1]} < {max_needed}"
    
    coverage_meta = {
        'n_intervals': n_intervals,
        'p_n': p_n,
        'p_n_plus_1': p_n_plus_1,
        'max_needed': max_needed,
        'largest_prime': primes[-1],
        'primes_generated': len(primes),
        'coverage_ok': primes[-1] >= max_needed
    }
    
    return primes, coverage_meta


def interval_from_i(primes: List[int], i: int) -> Tuple[int, int, int, int, int]:
    """
    Paper-consistent mapping: p_i = primes[i-1]
    Returns: (p_i, p_{i+1}, start, end, fulcrum)
    """
    p_i = primes[i - 1]
    p_next = primes[i]
    start = 2 * p_i
    end = 2 * p_next
    fulcrum = p_i + p_next + 1
    return p_i, p_next, start, end, fulcrum


def primes_in_interval_indices(primes: List[int], start: int, end: int) -> Tuple[int, int]:
    """
    Get indices of primes in [start, end) using bisect.
    Returns: (lo, hi) where primes[lo:hi] are the primes in the interval.
    
    NOTE: Does NOT allocate a slice - just returns indices for efficiency.
    """
    lo = bisect_left(primes, start)
    hi = bisect_left(primes, end)
    return lo, hi


# =============================================================================
# STREAK DETECTION
# =============================================================================

def analyze_streaks_from_json(json_streaks: List[dict]) -> Dict:
    """
    Analyze pre-computed streaks from JSON that include boundary info.
    
    Each streak should have: start_i, end_i, length, a, b, has_left_boundary, has_right_boundary
    And when both boundaries exist: q, r, F_a, F_b, gap_qr, delta_F, exceeds
    
    This is the FAST path - no prime generation needed!
    """
    both_hold = []
    only_q_fails = []
    only_r_fails = []
    both_fail = []
    skipped = 0
    
    for streak in json_streaks:
        # Skip streaks without both boundaries
        has_left = streak.get('has_left_boundary', 'q' in streak)
        has_right = streak.get('has_right_boundary', 'r' in streak)
        
        if not (has_left and has_right):
            skipped += 1
            continue
        
        # Skip if missing required boundary data
        if 'q' not in streak or 'r' not in streak:
            skipped += 1
            continue
        
        q = streak['q']
        r = streak['r']
        F_a = streak['F_a']
        F_b = streak['F_b']
        
        # Sanity checks on boundary values
        if F_a >= F_b or q >= r:
            print(f"  WARNING: Invalid streak data at {streak['start_i']}-{streak['end_i']}")
            skipped += 1
            continue
        
        # Use pre-computed values if available, otherwise compute
        prime_gap = streak.get('gap_qr', r - q)
        fulcrum_distance = streak.get('delta_F', F_b - F_a)
        
        # Cramér comparison: gap / (ln(m))² where m = (q+r)/2
        m = (q + r) // 2
        ln_m_sq = math.log(m) ** 2 if m > 1 else 1
        
        case_data = {
            'streak': (streak['start_i'], streak['end_i']),
            'streak_length': streak['length'],
            'idx_a': streak.get('a', streak['start_i'] - 1),
            'idx_b': streak.get('b', streak['end_i'] + 1),
            'q': q,
            'r': r,
            'F_a': F_a,
            'F_b': F_b,
            'q_dist': q - F_a,  # q - F_a (positive means q ≥ F_a)
            'r_dist': r - F_b,  # r - F_b (negative means r ≤ F_b)
            'prime_gap': prime_gap,  # r - q
            'fulcrum_distance': fulcrum_distance,  # F_b - F_a
            'exceeds': streak.get('exceeds', prime_gap > fulcrum_distance),
            'gap_pct_cramer': 100 * prime_gap / ln_m_sq,  # (r-q) / (ln m)² as %
            'fulcrum_pct_cramer': 100 * fulcrum_distance / ln_m_sq,
            'gap_a': streak.get('gap_a', 0),
            'gap_b': streak.get('gap_b', 0),
            'p_a': streak.get('p_a'),
            'p_b': streak.get('p_b'),
        }
        
        q_holds = q >= F_a
        r_holds = r <= F_b
        
        if q_holds and r_holds:
            both_hold.append(case_data)
        elif not q_holds and r_holds:
            only_q_fails.append(case_data)
        elif q_holds and not r_holds:
            only_r_fails.append(case_data)
        else:
            both_fail.append(case_data)
    
    if skipped > 0:
        print(f"  (Skipped {skipped} streaks without complete boundary info)")
    
    return {
        'both_hold': both_hold,
        'only_q_fails': only_q_fails,
        'only_r_fails': only_r_fails,
        'both_fail': both_fail,
    }


def compute_streaks_from_primes(primes: List[int], n_intervals: int, 
                                 progress_every: int = 100000) -> List[dict]:
    """
    Compute empty interval streaks directly from primes.
    
    Uses index-only operations for efficiency (no list slicing).
    
    Returns list of streak info dictionaries.
    """
    streaks = []
    current_streak_start = None
    
    for i in range(1, n_intervals + 1):
        p_i, p_next, start, end, fulcrum = interval_from_i(primes, i)
        lo, hi = primes_in_interval_indices(primes, start, end)
        is_empty = (lo == hi)
        
        if is_empty:
            if current_streak_start is None:
                current_streak_start = i
        else:
            if current_streak_start is not None:
                # End of streak - record it
                end_i = i - 1  # Last empty interval
                length = end_i - current_streak_start + 1  # Inclusive count
                streaks.append({
                    'start_i': current_streak_start,
                    'end_i': end_i,
                    'length': length
                })
                current_streak_start = None
        
        if i % progress_every == 0:
            print(f"  Scanned {i:,} intervals, found {len(streaks):,} streaks...")
    
    # Handle streak at end
    if current_streak_start is not None:
        streaks.append({
            'start_i': current_streak_start,
            'end_i': n_intervals,
            'length': n_intervals - current_streak_start + 1
        })
    
    return streaks


# =============================================================================
# FULCRUM CONDITION ANALYSIS
# =============================================================================

def analyze_streak_bounds(primes: List[int], streaks: List[dict]) -> Dict:
    """
    For each streak, analyze the bounding intervals I_a and I_b.
    
    I_a = interval just before the streak (should be non-empty)
    I_b = interval just after the streak (should be non-empty)
    
    Check:
    - q = last prime in I_a, does q ≥ F_a?
    - r = first prime in I_b, does r ≤ F_b?
    
    Uses index-only operations for efficiency.
    """
    
    both_hold = []
    only_q_fails = []
    only_r_fails = []
    both_fail = []
    
    for streak in streaks:
        start_i = streak['start_i']
        end_i = streak['end_i']
        
        # I_a is interval (start_i - 1), I_b is interval (end_i + 1)
        idx_a = start_i - 1
        idx_b = end_i + 1
        
        if idx_a < 1:
            continue  # No bounding interval before
        
        # Get I_a info
        p_a, p_a_next, start_a, end_a, F_a = interval_from_i(primes, idx_a)
        lo_a, hi_a = primes_in_interval_indices(primes, start_a, end_a)
        
        if lo_a == hi_a:
            continue  # I_a is empty (shouldn't happen for proper streak bounds)
        
        # Get I_b info
        try:
            p_b, p_b_next, start_b, end_b, F_b = interval_from_i(primes, idx_b)
            lo_b, hi_b = primes_in_interval_indices(primes, start_b, end_b)
        except IndexError:
            continue  # Beyond range
        
        if lo_b == hi_b:
            continue  # I_b is empty
        
        # Key values - get directly from indices, no slicing
        q = primes[hi_a - 1]  # Last prime in I_a
        r = primes[lo_b]      # First prime in I_b
        
        prime_gap = r - q
        fulcrum_distance = F_b - F_a
        
        # Cramér at this scale - use midpoint of q and r for value-based estimate
        x_mid = (q + r) // 2
        ln_sq = math.log(x_mid) ** 2 if x_mid > 1 else 1
        
        case_data = {
            'streak': (start_i, end_i),
            'streak_length': end_i - start_i + 1,
            'idx_a': idx_a,
            'idx_b': idx_b,
            'q': q,
            'r': r,
            'F_a': F_a,
            'F_b': F_b,
            'q_dist': q - F_a,  # positive = q is right of F_a
            'r_dist': r - F_b,  # negative = r is left of F_b
            'prime_gap': prime_gap,
            'fulcrum_distance': fulcrum_distance,
            'gap_pct_cramer': 100 * prime_gap / ln_sq if ln_sq > 0 else 0,
            'fulcrum_pct_cramer': 100 * fulcrum_distance / ln_sq if ln_sq > 0 else 0,
            'gap_a': p_a_next - p_a,
            'gap_b': p_b_next - p_b,
            'num_primes_a': hi_a - lo_a,  # Count from indices, no len()
            'num_primes_b': hi_b - lo_b,
        }
        
        q_holds = q >= F_a
        r_holds = r <= F_b
        
        if q_holds and r_holds:
            both_hold.append(case_data)
        elif not q_holds and r_holds:
            only_q_fails.append(case_data)
        elif q_holds and not r_holds:
            only_r_fails.append(case_data)
        else:
            both_fail.append(case_data)
    
    return {
        'both_hold': both_hold,
        'only_q_fails': only_q_fails,
        'only_r_fails': only_r_fails,
        'both_fail': both_fail,
    }


# =============================================================================
# REPORTING
# =============================================================================

def print_report(results: Dict, n_intervals: int) -> None:
    """Print analysis report."""
    
    both_hold = results['both_hold']
    only_q_fails = results['only_q_fails']
    only_r_fails = results['only_r_fails']
    both_fail = results['both_fail']
    
    total = len(both_hold) + len(only_q_fails) + len(only_r_fails) + len(both_fail)
    
    if total == 0:
        print("No streak-bounding cases to analyze.")
        return
    
    print("=" * 75)
    print("FULCRUM CONDITION ANALYSIS")
    print("=" * 75)
    print()
    
    print("-" * 75)
    print("THE FULCRUM BOUND THEOREM")
    print("-" * 75)
    print("For a streak of empty intervals from I_{a+1} to I_{b-1}:")
    print()
    print("  IF q ≥ F_a  (last prime in I_a is at/right of fulcrum)")
    print("  AND r ≤ F_b (first prime in I_b is at/left of fulcrum)")
    print("  THEN gap(q,r) ≤ F_b - F_a")
    print()
    
    print("-" * 75)
    print("CATEGORIZATION")
    print("-" * 75)
    print(f"  Both conditions hold:     {len(both_hold):>8,}  ({100*len(both_hold)/total:>5.1f}%)")
    print(f"  Only q fails (q < F_a):   {len(only_q_fails):>8,}  ({100*len(only_q_fails)/total:>5.1f}%)")
    print(f"  Only r fails (r > F_b):   {len(only_r_fails):>8,}  ({100*len(only_r_fails)/total:>5.1f}%)")
    print(f"  Both fail:                {len(both_fail):>8,}  ({100*len(both_fail)/total:>5.1f}%)")
    print(f"  {'─'*40}")
    print(f"  Total streaks analyzed:   {total:>8,}")
    print()
    
    print("-" * 75)
    print("CRAMÉR CHECK BY CATEGORY")
    print("-" * 75)
    print("  Cramér % = 100 × (r-q) / (ln m)²  where m = (q+r)/2")
    print()
    print(f"  {'Category':<20}  {'Count':>8}  {'Max Gap':>10}  {'Max Cramér%':>12}  {'Avg Cramér%':>12}")
    print(f"  {'-'*20}  {'-'*8}  {'-'*10}  {'-'*12}  {'-'*12}")
    
    categories = [
        ("Both hold", both_hold),
        ("Only q fails", only_q_fails),
        ("Only r fails", only_r_fails),
        ("Both fail", both_fail),
    ]
    
    for name, cases in categories:
        if cases:
            max_gap = max(c['prime_gap'] for c in cases)
            max_pct = max(c['gap_pct_cramer'] for c in cases)
            avg_pct = sum(c['gap_pct_cramer'] for c in cases) / len(cases)
            print(f"  {name:<20}  {len(cases):>8,}  {max_gap:>10}  {max_pct:>11.1f}%  {avg_pct:>11.1f}%")
        else:
            print(f"  {name:<20}  {0:>8}  {'N/A':>10}  {'N/A':>12}  {'N/A':>12}")
    print()
    
    # Check if gap ever exceeds fulcrum distance in failure cases
    fail_cases = only_q_fails + only_r_fails + both_fail
    gap_exceeds = [c for c in fail_cases if c['prime_gap'] > c['fulcrum_distance']]
    
    print("-" * 75)
    print("KEY FINDINGS")
    print("-" * 75)
    print(f"  Cases where r - q > F_b - F_a: {len(gap_exceeds):,}")
    print(f"  (i.e., actual prime gap exceeds fulcrum distance)")
    print()
    
    if gap_exceeds:
        print("  When fulcrum conditions fail, gap CAN exceed F_b - F_a.")
        print()
        print("  Worst cases (gap > fulcrum distance):")
        print(f"  {'Streak':>12}  {'Gap':>8}  {'F_b-F_a':>10}  {'Excess':>8}  {'%Cramér':>10}")
        print(f"  {'-'*12}  {'-'*8}  {'-'*10}  {'-'*8}  {'-'*10}")
        
        gap_exceeds.sort(key=lambda x: x['prime_gap'] - x['fulcrum_distance'], reverse=True)
        for c in gap_exceeds[:10]:
            s, e = c['streak']
            excess = c['prime_gap'] - c['fulcrum_distance']
            print(f"  {s:>5}-{e:<5}  {c['prime_gap']:>8}  {c['fulcrum_distance']:>10}  {excess:>8}  {c['gap_pct_cramer']:>9.1f}%")
    else:
        print("  Even in failure cases, gap ≤ F_b - F_a always holds!")
    print()
    
    # Characteristics comparison
    print("-" * 75)
    print("CHARACTERISTICS: HOLD vs FAIL")
    print("-" * 75)
    
    if both_hold and fail_cases:
        avg_gap_a_hold = sum(c['gap_a'] for c in both_hold) / len(both_hold)
        avg_gap_a_fail = sum(c['gap_a'] for c in fail_cases) / len(fail_cases)
        
        avg_streak_hold = sum(c['streak_length'] for c in both_hold) / len(both_hold)
        avg_streak_fail = sum(c['streak_length'] for c in fail_cases) / len(fail_cases)
        
        print(f"  {'Metric':<25}  {'Both Hold':>12}  {'Any Failure':>12}")
        print(f"  {'-'*25}  {'-'*12}  {'-'*12}")
        print(f"  {'Avg gap of I_a':<25}  {avg_gap_a_hold:>12.1f}  {avg_gap_a_fail:>12.1f}")
        print(f"  {'Avg streak length':<25}  {avg_streak_hold:>12.2f}  {avg_streak_fail:>12.2f}")
    print()
    
    # Summary
    all_cases = both_hold + fail_cases
    max_pct_all = max(c['gap_pct_cramer'] for c in all_cases) if all_cases else 0
    
    # Check if any case exceeds 100% of Cramér
    cramer_exceeded = [c for c in all_cases if c['gap_pct_cramer'] > 100]
    
    print("=" * 75)
    print("SUMMARY")
    print("=" * 75)
    print(f"  • Fulcrum conditions hold in {100*len(both_hold)/total:.1f}% of streak-bounding cases")
    print(f"  • Maximum (r-q)/(ln m)² observed: {max_pct_all:.1f}%")
    if cramer_exceeded:
        print(f"  • WARNING: {len(cramer_exceeded)} cases exceed 100% of Cramér bound")
    else:
        print(f"  • All gaps below Cramér bound (ln m)²")
    print()
    print("  The Fulcrum Bound Theorem provides a valid upper bound when conditions hold.")
    if gap_exceeds:
        print("  When conditions fail, gaps can exceed F_b - F_a.")
    print("=" * 75)
    print()


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Analyze Fulcrum Bound condition failures (v3.1 compatible)",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("input", nargs='?', help="Path to v3.1 harvest JSON file")
    parser.add_argument("--intervals", "-n", type=int, default=100000,
                        help="Number of intervals to analyze (default: 100000)")
    parser.add_argument("--max-streaks", type=int, default=None,
                        help="Maximum streaks to analyze (for faster runs)")
    parser.add_argument("--standalone", "-s", action="store_true",
                        help="Run standalone without JSON (generate primes directly)")
    parser.add_argument("--force", "-f", action="store_true",
                        help="Skip confirmation for large runs")
    parser.add_argument("--regenerate", action="store_true",
                        help="Regenerate streaks even if JSON has them")
    
    args = parser.parse_args()
    
    if args.input:
        # Load from JSON
        print(f"Loading {args.input}...")
        with open(args.input, 'r') as f:
            data = json.load(f)
        
        n_intervals = data['metadata']['n_intervals']
        print(f"JSON has {n_intervals:,} intervals")
        
        # Check if JSON has pre-computed streaks with boundary info
        # Check first 50 streaks in case first one is terminal (no boundary)
        json_streaks = data.get('streaks', [])
        has_boundary_info = any(('q' in s and 'r' in s) for s in json_streaks[:50])
        
        if json_streaks and has_boundary_info and not args.regenerate:
            # Use pre-computed streaks from JSON - NO prime generation needed!
            print(f"Using {len(json_streaks):,} pre-computed streaks from JSON")
            print("(No prime generation needed - boundary info included)")
            
            # Apply max-streaks limit if specified
            if args.max_streaks and len(json_streaks) > args.max_streaks:
                print(f"(Limiting analysis to {args.max_streaks} streaks)")
                json_streaks = json_streaks[:args.max_streaks]
            
            # Analyze directly from JSON streaks
            results = analyze_streaks_from_json(json_streaks)
            
            print()
            print_report(results, n_intervals)
            return 0
        
        elif json_streaks and not has_boundary_info:
            print(f"JSON has {len(json_streaks)} streaks but missing boundary info")
            print("Need to regenerate with primes...")
        
        elif args.regenerate and n_intervals > 10_000_000:
            print(f"\n⚠️  WARNING: --regenerate on {n_intervals:,} intervals requires")
            print(f"   sieving to ~{2.5 * n_intervals / 1e6:.0f}B which needs multi-GB RAM.")
            print(f"   Consider using the fast path (streaks already have boundary info).")
            if not args.force:
                response = input("Are you sure? [y/N]: ")
                if response.lower() != 'y':
                    print("Aborted. Remove --regenerate to use fast path.")
                    return 1
        
        # Fall through to prime generation if no usable streaks
        if n_intervals > 10_000_000 and not args.force:
            print(f"\nWARNING: Full streak analysis of {n_intervals:,} intervals requires")
            print(f"         generating primes and scanning all intervals.")
            print(f"         This may take several minutes and significant memory.")
            print(f"         Use --force to skip this prompt.")
            response = input("Continue? [y/N]: ")
            if response.lower() != 'y':
                print("Aborted. Use --force to skip confirmation.")
                return 1
    else:
        if not args.standalone:
            print("Usage: python fulcrum_analysis.py <harvest_json>")
            print("       python fulcrum_analysis.py --standalone -n 100000")
            return 1
        n_intervals = args.intervals
    
    # Generate primes with proper coverage guarantee
    print(f"\nGenerating primes for {n_intervals:,} intervals...")
    primes, coverage_meta = primes_for_intervals(n_intervals)
    
    print(f"Generated {coverage_meta['primes_generated']:,} primes")
    print(f"  p_n = {coverage_meta['p_n']:,}")
    print(f"  p_(n+1) = {coverage_meta['p_n_plus_1']:,}")
    print(f"  max_needed = {coverage_meta['max_needed']:,}")
    print(f"  largest_prime = {coverage_meta['largest_prime']:,}")
    print(f"  coverage_ok = {coverage_meta['coverage_ok']}")
    
    if not coverage_meta['coverage_ok']:
        print("ERROR: Coverage guarantee failed!")
        return 1
    
    # Find streaks
    print(f"\nScanning for empty interval streaks...")
    streaks = compute_streaks_from_primes(primes, n_intervals)
    print(f"Found {len(streaks):,} streaks")
    
    if args.max_streaks and len(streaks) > args.max_streaks:
        print(f"Limiting analysis to {args.max_streaks} streaks")
        streaks = streaks[:args.max_streaks]
    
    # Analyze streak bounds
    print(f"\nAnalyzing streak-bounding intervals...")
    results = analyze_streak_bounds(primes, streaks)
    
    # Print report
    print()
    print_report(results, n_intervals)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
