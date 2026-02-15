#!/usr/bin/env python3
"""
================================================================================
SAFE GAP CONJECTURE - ANALYSIS AND VERIFICATION (v3.1 Compatible)
================================================================================

This script tests and verifies the Safe Gap Conjecture against empirical data.

CONJECTURE:
    If g_i ≥ (1/π) · [ln(2)·ln(2i)·ln(p_i) + 2·ln(i)],
    then [2p_i, 2p_{i+1}) contains at least one prime.

The correction term (2/π)·ln(i) accounts for sample-size effects.

Usage:
    python safe_gap_analysis.py harvest_68M_v3_1.json
    python safe_gap_analysis.py harvest_68M_v3_1.json --verbose
    python safe_gap_analysis.py --predictions  # Show predictions at various scales

Author: Collaborative work (Human + Claude)
Date: January 2026
================================================================================
"""

import json
import math
import argparse
import sys
from datetime import datetime


# =============================================================================
# CONSTANTS
# =============================================================================

LN2_OVER_PI = math.log(2) / math.pi    # ≈ 0.2206
TWO_OVER_PI = 2 / math.pi              # ≈ 0.6366
ONE_OVER_PI = 1 / math.pi              # ≈ 0.3183


# =============================================================================
# THRESHOLD FUNCTIONS
# =============================================================================

def estimate_prime(i: int) -> float:
    """
    Estimate the i-th prime using the Prime Number Theorem.
    For large i: p_i ≈ i · (ln(i) + ln(ln(i)))
    """
    if i < 10:
        small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
        return small_primes[i - 1] if i > 0 else 2
    return i * (math.log(i) + math.log(math.log(i)))


def safe_gap_threshold(i: int, p_i: float = None) -> float:
    """
    Compute Safe Gap threshold T(i).
    
    T(i) = (1/π) · [ln(2)·ln(2i)·ln(p_i) + 2·ln(i)]
    
    If g ≥ T(i), the interval MUST contain a prime.
    """
    if i <= 1:
        return 0
    if p_i is None:
        p_i = estimate_prime(i)
    if p_i <= 2:
        return 0
    return ONE_OVER_PI * (math.log(2) * math.log(2 * i) * math.log(p_i) + 2 * math.log(i))


def original_threshold(i: int, p_i: float = None) -> float:
    """
    Original (uncorrected) threshold for comparison.
    T_orig(i) = (ln2/π) · ln(2i) · ln(p_i)
    """
    if i <= 1:
        return 0
    if p_i is None:
        p_i = estimate_prime(i)
    if p_i <= 2:
        return 0
    return LN2_OVER_PI * math.log(2 * i) * math.log(p_i)


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def analyze_from_json(data: dict, verbose: bool = False) -> dict:
    """
    Analyze Safe Gap Conjecture from v3.1 harvester JSON.
    
    v3.1 schema:
    - data['metadata']['n_intervals']
    - data['metadata']['largest_prime_in_list'] (largest prime in list, NOT p_i)
    
    v3.1 schema:
    - data['metadata']['n_intervals']
    - data['metadata']['p_n'] (exact n-th prime, for threshold formula)
    - data['metadata']['p_n_plus_1'] (defines last interval endpoint)
    - data['metadata']['sieve_bound'] (integer N sieved to)
    - data['metadata']['largest_prime_in_list'] (max prime ≤ sieve_bound)
    - data['by_gap'][gap_str]['count']
    - data['by_gap'][gap_str]['empty']
    - data['intervals']['empty_count']
    - data['by_gap'][gap_str]['count']
    - data['by_gap'][gap_str]['empty']
    - data['intervals']['empty_count']
    
    IMPORTANT: The conjecture uses p_i (the i-th prime), NOT the sieve limit.
    We estimate p_i using PNT, or use metadata field if available.
    """
    
    meta = data['metadata']
    n_intervals = meta['n_intervals']
    
    # largest_prime_in_list is the max prime in list, NOT p_i
    # We need p_n or p_{n+1} for the formula
    # Fallback chain for backward compatibility with older JSON
    sieve_limit = meta.get('largest_prime_in_list', 
                          meta.get('largest_prime_generated',  # legacy name
                          meta.get('sieve_limit', 0)))
    
    # Use p_i from metadata if available, otherwise estimate via PNT
    # The conjecture is about p_i where i = n_intervals
    if 'p_n' in meta:
        p_i_used = meta['p_n']
        p_i_source = 'metadata (p_n, exact)'
        p_i_is_proxy = False
    elif 'p_n_plus_1' in meta:
        # p_{n+1} is close enough, and actually what defines the last interval
        p_i_used = meta['p_n_plus_1']
        p_i_source = 'metadata (p_{n+1}, proxy for p_n)'
        p_i_is_proxy = True
    elif 'largest_p_used' in meta:
        p_i_used = meta['largest_p_used']
        p_i_source = 'metadata (largest_p_used, proxy)'
        p_i_is_proxy = True
    else:
        # Estimate using PNT - fallback for older JSON
        p_i_used = estimate_prime(n_intervals)
        p_i_source = 'PNT estimate (no p_n in metadata)'
        p_i_is_proxy = True
    
    by_gap = data['by_gap']
    total_empty = data['intervals']['empty_count']
    
    # Compute threshold using estimated/actual p_i (NOT sieve limit)
    threshold = safe_gap_threshold(n_intervals, p_i_used)
    threshold_orig = original_threshold(n_intervals, p_i_used)
    
    # Analyze gaps
    gaps_with_empty = []
    gaps_without_empty = []
    all_gaps = []
    
    for gap_str, gap_data in by_gap.items():
        gap = int(gap_str)
        count = gap_data['count']
        empty = gap_data['empty']
        
        all_gaps.append((gap, count, empty))
        
        if empty > 0:
            gaps_with_empty.append((gap, count, empty))
        else:
            gaps_without_empty.append((gap, count))
    
    # Sort
    all_gaps.sort(key=lambda x: x[0])
    gaps_with_empty.sort(key=lambda x: x[0])
    gaps_without_empty.sort(key=lambda x: x[0])
    
    # Key metrics
    max_gap_all = max(g[0] for g in all_gaps) if all_gaps else 0
    max_gap_with_empty = max(g[0] for g in gaps_with_empty) if gaps_with_empty else 0
    min_gap_without_empty = min(g[0] for g in gaps_without_empty) if gaps_without_empty else 0
    
    # Find safe gap metrics:
    # 1) first_zero_min10: first gap with empty==0 and count>=10
    # 2) stable_zero_min10: smallest gap where empty==0, count>=10, AND no later gap has empties
    first_zero_gap = None
    stable_zero_gap = None
    
    for gap, count, empty in sorted(all_gaps, key=lambda x: x[0]):
        if empty == 0 and count >= 10:
            if first_zero_gap is None:
                first_zero_gap = gap
            if stable_zero_gap is None:
                stable_zero_gap = gap
        elif empty > 0:
            stable_zero_gap = None  # Reset if we find empties again
    
    # Check conjecture (AGGREGATE consistency check, not full verification)
    # Full verification would require per-interval threshold checks
    aggregate_consistent = max_gap_with_empty < threshold
    margin = threshold - max_gap_with_empty
    
    # Cramér comparison
    cramer_bound = math.log(p_i_used) ** 2 if p_i_used > 2 else 0
    
    # Also compute threshold using p_{n+1} for comparison (since interval ends at 2p_{n+1})
    p_n_plus_1 = meta.get('p_n_plus_1', None)
    threshold_with_pn1 = safe_gap_threshold(n_intervals, p_n_plus_1) if p_n_plus_1 else None
    
    results = {
        'n_intervals': n_intervals,
        'p_i_used': p_i_used,
        'p_i_source': p_i_source,
        'p_i_is_proxy': p_i_is_proxy,
        'p_n_plus_1': p_n_plus_1,
        'sieve_bound': meta.get('sieve_bound', sieve_limit),
        'largest_prime': sieve_limit,
        'total_empty': total_empty,
        'threshold': threshold,
        'threshold_with_pn1': threshold_with_pn1,
        'threshold_orig': threshold_orig,
        'correction_term': TWO_OVER_PI * math.log(n_intervals),
        'max_gap_all': max_gap_all,
        'max_gap_with_empty': max_gap_with_empty,
        'min_gap_without_empty': min_gap_without_empty,
        'first_zero_gap': first_zero_gap,
        'stable_zero_gap': stable_zero_gap,
        'aggregate_consistent': aggregate_consistent,
        'margin': margin,
        'cramer_bound': cramer_bound,
        'gaps_with_empty': gaps_with_empty,
        'gaps_without_empty': gaps_without_empty,
        'all_gaps': all_gaps,
    }
    
    return results


def print_report(results: dict, verbose: bool = False) -> None:
    """Print analysis report."""
    
    print("=" * 75)
    print("SAFE GAP CONJECTURE VERIFICATION")
    print("=" * 75)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    print("-" * 75)
    print("THE CONJECTURE")
    print("-" * 75)
    print("For interval I_i = [2p_i, 2p_{i+1}) with gap g = p_{i+1} - p_i:")
    print()
    print("  If g ≥ T(i), the interval MUST contain at least one prime.")
    print()
    print("  T(i) = (1/π) · [ln(2)·ln(2i)·ln(p_i) + 2·ln(i)]")
    print()
    
    print("-" * 75)
    print("DATA SUMMARY")
    print("-" * 75)
    print(f"  Intervals analyzed:      {results['n_intervals']:,}")
    print(f"  p_n (exact):             {results['p_i_used']:,.0f}  ({results['p_i_source']})")
    if results['p_i_is_proxy']:
        if 'p_n_plus_1' in results['p_i_source']:
            print(f"    Note: Using p_(n+1) as close proxy for p_n")
        else:
            print(f"    Note: Using PNT estimate of p_n")
    print(f"  Sieve bound (N):         {results['sieve_bound']:,}")
    print(f"  Largest prime ≤N:        {results['largest_prime']:,}")
    print(f"  Total empty intervals:   {results['total_empty']:,} ({100*results['total_empty']/results['n_intervals']:.2f}%)")
    print(f"  Distinct gap sizes:      {len(results['all_gaps'])}")
    print()
    
    print("-" * 75)
    print("THRESHOLD ANALYSIS")
    print("-" * 75)
    print(f"  At i = {results['n_intervals']:,}:")
    print(f"    Original threshold (no correction):  {results['threshold_orig']:.2f}")
    print(f"    Corrected threshold T(n) with p_n:   {results['threshold']:.2f}")
    if results['threshold_with_pn1']:
        print(f"    Corrected threshold T(n) with p_(n+1): {results['threshold_with_pn1']:.2f}")
    print(f"    Correction term (2/π)·ln(n):         {results['correction_term']:.2f}")
    print()
    
    print("-" * 75)
    print("GAP ANALYSIS")
    print("-" * 75)
    print(f"  Maximum gap (all intervals):     {results['max_gap_all']}")
    print(f"  Maximum gap (empty intervals):   {results['max_gap_with_empty']}")
    print(f"  Minimum gap (non-empty only):    {results['min_gap_without_empty']}")
    print()
    print(f"  First zero gap (n≥10):           {results['first_zero_gap']}")
    print(f"  Stable zero gap (n≥10):          {results['stable_zero_gap']}")
    print(f"    (stable = no later gap has empties)")
    print()
    
    print("-" * 75)
    print("AGGREGATE CONSISTENCY CHECK")
    print("-" * 75)
    print(f"  Max empty gap:    {results['max_gap_with_empty']}")
    print(f"  Threshold T(i):   {results['threshold']:.2f}")
    print(f"  Margin:           {results['margin']:.2f}")
    print()
    print("  Note: This checks max_gap_among_empties < T(n), which is CONSISTENT WITH")
    print("        the conjecture but not a full per-interval verification.")
    print()
    
    if results['aggregate_consistent']:
        print(f"  ✓ AGGREGATE CHECK PASSES")
        print(f"    All empty intervals have gap < T({results['n_intervals']:,})")
    else:
        print(f"  ✗ AGGREGATE CHECK FAILS")
        print(f"    Max empty gap ({results['max_gap_with_empty']}) ≥ threshold ({results['threshold']:.2f})")
    print()
    
    print("-" * 75)
    print("TRANSITION ZONE (gaps near threshold)")
    print("-" * 75)
    print(f"  {'Gap':>6}  {'Intervals':>12}  {'Empty':>10}  {'Empty%':>10}  {'Status':>12}")
    print(f"  {'-'*6}  {'-'*12}  {'-'*10}  {'-'*10}  {'-'*12}")
    
    # Show gaps around the threshold
    threshold = results['threshold']
    for gap, count, empty in results['all_gaps']:
        if threshold - 20 <= gap <= threshold + 20:
            empty_pct = 100 * empty / count if count > 0 else 0
            status = "← threshold" if abs(gap - threshold) < 2 else ("ZERO" if empty == 0 else "")
            print(f"  {gap:>6}  {count:>12,}  {empty:>10,}  {empty_pct:>9.2f}%  {status:>12}")
    print()
    
    print("-" * 75)
    print("CRAMÉR COMPARISON")
    print("-" * 75)
    if results['cramer_bound'] > 0:
        print(f"  Cramér bound (ln p)²:         {results['cramer_bound']:.1f}")
        print(f"  Max gap (all):                {results['max_gap_all']} ({100*results['max_gap_all']/results['cramer_bound']:.1f}% of Cramér)")
        print(f"  Max gap (empty):              {results['max_gap_with_empty']} ({100*results['max_gap_with_empty']/results['cramer_bound']:.1f}% of Cramér)")
        print(f"  Our threshold T(i):           {results['threshold']:.1f} ({100*results['threshold']/results['cramer_bound']:.1f}% of Cramér)")
    print()
    
    if verbose:
        print("-" * 75)
        print("ALL GAPS WITH EMPTY INTERVALS")
        print("-" * 75)
        print(f"  {'Gap':>6}  {'Intervals':>12}  {'Empty':>10}  {'Empty%':>10}")
        print(f"  {'-'*6}  {'-'*12}  {'-'*10}  {'-'*10}")
        for gap, count, empty in results['gaps_with_empty']:
            empty_pct = 100 * empty / count if count > 0 else 0
            print(f"  {gap:>6}  {count:>12,}  {empty:>10,}  {empty_pct:>9.2f}%")
        print()
    
    print("=" * 75)
    print("SUMMARY")
    print("=" * 75)
    if results['aggregate_consistent']:
        print(f"  ✓ Aggregate consistency check PASSES at {results['n_intervals']:,} intervals")
        print(f"  ✓ Max empty gap: {results['max_gap_with_empty']}, Threshold T(n): {results['threshold']:.2f}")
        print(f"  ✓ Margin: {results['margin']:.2f}")
        print()
        print(f"  Note: Full verification requires per-interval threshold checks.")
        print(f"        This aggregate check is consistent with the conjecture.")
    else:
        print(f"  ✗ Aggregate check FAILS")
    print("=" * 75)
    print()


def print_predictions() -> None:
    """Print predictions at various scales."""
    
    print("=" * 75)
    print("SAFE GAP THRESHOLD PREDICTIONS")
    print("=" * 75)
    print()
    print("T(i) = (1/π) · [ln(2)·ln(2i)·ln(p_i) + 2·ln(i)]")
    print()
    
    scales = [
        10_000, 
        100_000, 
        1_000_000, 
        10_000_000, 
        68_000_000, 
        100_000_000, 
        1_000_000_000
    ]
    
    print(f"{'Scale (i)':>15}  {'Est. p_i':>15}  {'T_orig':>10}  {'T_corr':>10}  {'Correction':>12}")
    print("-" * 70)
    
    for i in scales:
        p_i = estimate_prime(i)
        t_orig = original_threshold(i, p_i)
        t_corr = safe_gap_threshold(i, p_i)
        correction = TWO_OVER_PI * math.log(i)
        print(f"{i:>15,}  {p_i:>15,.0f}  {t_orig:>10.1f}  {t_corr:>10.1f}  {correction:>12.1f}")
    
    print()
    print("As i grows, the correction term (2/π)·ln(i) becomes increasingly important.")
    print()


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Analyze and verify the Safe Gap Conjecture (v3.1 compatible)",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("input", nargs='?', help="Path to v3.1 harvest JSON file")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show detailed output")
    parser.add_argument("--predictions", "-p", action="store_true",
                        help="Show predictions at various scales (no data needed)")
    parser.add_argument("--output", "-o", help="Save report to file")
    
    args = parser.parse_args()
    
    if args.predictions:
        print_predictions()
        if not args.input:
            return 0
    
    if not args.input:
        print("Usage: python safe_gap_analysis.py <harvest_json> [--verbose]")
        print("       python safe_gap_analysis.py --predictions")
        return 1
    
    # Load and analyze
    print(f"Loading {args.input}...")
    with open(args.input, 'r') as f:
        data = json.load(f)
    
    # Check version
    version = data.get('metadata', {}).get('version', 'unknown')
    print(f"JSON version: {version}")
    
    results = analyze_from_json(data, verbose=args.verbose)
    if args.output:
        # Capture report to file
        import io
        import sys
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        print_report(results, verbose=args.verbose)
        report_text = buffer.getvalue()
        sys.stdout = old_stdout
        
        with open(args.output, 'w') as f:
            f.write(report_text)
        print(f"Report saved to {args.output}")
        
        # Also print to console
        print(report_text)
    else:
        print_report(results, verbose=args.verbose)
    
    return 0 if results['aggregate_consistent'] else 1


if __name__ == "__main__":
    sys.exit(main())
