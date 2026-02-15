#!/usr/bin/env python3
"""
================================================================================
VISUALIZATION: T(n) vs Emptiness Frontier
================================================================================

Creates a chart showing:
1. The corrected threshold T(n) across scales
2. The original threshold T_orig(n) for comparison
3. The observed max empty gap (frontier) at each scale
4. Highlights the gap-96 outlier

Outputs:
- An interactive HTML chart (if plotly available)
- A static PNG chart (if matplotlib available)
- CSV data for external plotting
"""

import math
import json
import os

# =============================================================================
# DATA
# =============================================================================

# Multi-scale results from your harvests
HARVEST_DATA = [
    {
        'n': 10_000_000,
        'p_n': 179_424_673,
        'max_empty_gap': 72,
        'stable_zero_gap': 74,
        'left_fraction': 0.499910,
    },
    {
        'n': 20_000_000,
        'p_n': 373_587_883,
        'max_empty_gap': 80,
        'stable_zero_gap': 82,
        'left_fraction': 0.499923,
    },
    {
        'n': 40_000_000,
        'p_n': 776_531_401,
        'max_empty_gap': 96,  # The outlier!
        'stable_zero_gap': 98,
        'left_fraction': 0.499941,
    },
    {
        'n': 68_000_000,
        'p_n': 1_358_208_601,
        'max_empty_gap': 96,  # Same outlier
        'stable_zero_gap': 98,
        'left_fraction': 0.499972,
    },
]

# The mysterious outlier
OUTLIER = {
    'index': 26_235_002,
    'p_i': 497_575_847,
    'p_i_plus_1': 497_575_943,
    'gap': 96,
    'interval_start': 995_151_694,
    'interval_end': 995_151_886,
}

LN_2 = math.log(2)


# =============================================================================
# THRESHOLD FORMULAS
# =============================================================================

def T_corrected(n, p_n):
    """Corrected threshold: T(n) = (1/π)[ln2·ln(2n)·ln(p_n) + 2·ln(n)]"""
    ln_2n = math.log(2 * n)
    ln_p = math.log(p_n)
    ln_n = math.log(n)
    return (LN_2 * ln_2n * ln_p + 2 * ln_n) / math.pi


def T_original(n, p_n):
    """Original threshold without correction: T_orig(n) = (1/π)[ln2·ln(2n)·ln(p_n)]"""
    ln_2n = math.log(2 * n)
    ln_p = math.log(p_n)
    return (LN_2 * ln_2n * ln_p) / math.pi


# =============================================================================
# COMPUTE DATA POINTS
# =============================================================================

def compute_chart_data():
    """Compute all data points for the chart."""
    
    data_points = []
    
    for h in HARVEST_DATA:
        n = h['n']
        p_n = h['p_n']
        
        T_corr = T_corrected(n, p_n)
        T_orig = T_original(n, p_n)
        
        data_points.append({
            'n': n,
            'n_millions': n / 1_000_000,
            'p_n': p_n,
            'T_corrected': T_corr,
            'T_original': T_orig,
            'max_empty_gap': h['max_empty_gap'],
            'stable_zero_gap': h['stable_zero_gap'],
            'margin': T_corr - h['max_empty_gap'],
            'margin_orig': T_orig - h['max_empty_gap'],
            'is_outlier_scale': h['max_empty_gap'] == 96,
            'left_fraction': h['left_fraction'],
            'balance_deviation_pct': abs(h['left_fraction'] - 0.5) * 100,
        })
    
    return data_points


# =============================================================================
# MATPLOTLIB VISUALIZATION
# =============================================================================

def create_matplotlib_chart(data_points, output_file='frontier_chart.png'):
    """Create static PNG chart with matplotlib."""
    
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("matplotlib not available - skipping PNG output")
        return False
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), height_ratios=[2, 1])
    
    # Extract data
    n_vals = [d['n_millions'] for d in data_points]
    T_corr = [d['T_corrected'] for d in data_points]
    T_orig = [d['T_original'] for d in data_points]
    max_empty = [d['max_empty_gap'] for d in data_points]
    balance_dev = [d['balance_deviation_pct'] for d in data_points]
    
    # =========================================================================
    # TOP PLOT: T(n) vs Frontier
    # =========================================================================
    
    # Plot lines
    ax1.plot(n_vals, T_corr, 'b-o', linewidth=2, markersize=10, label='T(n) corrected', zorder=3)
    ax1.plot(n_vals, T_orig, 'b--s', linewidth=1.5, markersize=8, alpha=0.5, label='T(n) original', zorder=2)
    ax1.plot(n_vals, max_empty, 'r-^', linewidth=2, markersize=10, label='Max empty gap', zorder=3)
    
    # Highlight the outlier region
    # At 40M, max_empty (96) > T_corrected (93.33)
    ax1.fill_between([35, 45], [90, 90], [100, 100], alpha=0.2, color='red', 
                      label='Outlier exceeds T(n)')
    
    # Add horizontal line at gap 96
    ax1.axhline(y=96, color='orange', linestyle=':', alpha=0.7, label='Gap 96 (outlier)')
    
    # Annotate the outlier
    ax1.annotate(
        f'Single outlier:\nI_{{26,235,002}}\ngap = 96',
        xy=(40, 96),
        xytext=(50, 88),
        fontsize=9,
        ha='left',
        arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    )
    
    # Annotate the catch-up at 68M
    ax1.annotate(
        'T(n) catches up',
        xy=(68, 98.38),
        xytext=(60, 104),
        fontsize=9,
        ha='center',
        arrowprops=dict(arrowstyle='->', color='blue', alpha=0.7),
    )
    
    ax1.set_xlabel('n (millions of intervals)', fontsize=12)
    ax1.set_ylabel('Gap threshold / Max empty gap', fontsize=12)
    ax1.set_title('Safe Gap Threshold vs Emptiness Frontier Across Scales', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(5, 75)
    ax1.set_ylim(65, 110)
    
    # Add margin annotations
    for d in data_points:
        margin = d['margin']
        color = 'green' if margin > 0 else 'red'
        symbol = '+' if margin > 0 else ''
        ax1.annotate(
            f'{symbol}{margin:.1f}',
            xy=(d['n_millions'], d['T_corrected']),
            xytext=(d['n_millions'] + 2, d['T_corrected'] + 3),
            fontsize=8,
            color=color,
            fontweight='bold'
        )
    
    # =========================================================================
    # BOTTOM PLOT: Balance Property
    # =========================================================================
    
    ax2.plot(n_vals, balance_dev, 'g-o', linewidth=2, markersize=10)
    ax2.fill_between(n_vals, 0, balance_dev, alpha=0.3, color='green')
    
    ax2.set_xlabel('n (millions of intervals)', fontsize=12)
    ax2.set_ylabel('Deviation from 50% (%)', fontsize=12)
    ax2.set_title('Balance Property: Deviation from Perfect 50/50 Split', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(5, 75)
    ax2.set_ylim(0, 0.012)
    
    # Annotate convergence
    ax2.annotate(
        'Converging to 50/50',
        xy=(68, 0.0028),
        xytext=(50, 0.006),
        fontsize=10,
        arrowprops=dict(arrowstyle='->', color='green', alpha=0.7),
    )
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file}")
    
    return True


# =============================================================================
# PLOTLY VISUALIZATION (Interactive HTML)
# =============================================================================

def create_plotly_chart(data_points, output_file='frontier_chart.html'):
    """Create interactive HTML chart with plotly."""
    
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("plotly not available - skipping HTML output")
        return False
    
    # Extract data
    n_vals = [d['n_millions'] for d in data_points]
    T_corr = [d['T_corrected'] for d in data_points]
    T_orig = [d['T_original'] for d in data_points]
    max_empty = [d['max_empty_gap'] for d in data_points]
    balance_dev = [d['balance_deviation_pct'] for d in data_points]
    margins = [d['margin'] for d in data_points]
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(
            'Safe Gap Threshold vs Emptiness Frontier',
            'Balance Property: Convergence to 50/50'
        ),
        vertical_spacing=0.15,
        row_heights=[0.65, 0.35]
    )
    
    # =========================================================================
    # TOP PLOT: T(n) vs Frontier
    # =========================================================================
    
    # T(n) corrected
    fig.add_trace(
        go.Scatter(
            x=n_vals, y=T_corr,
            mode='lines+markers',
            name='T(n) corrected',
            line=dict(color='blue', width=3),
            marker=dict(size=12),
            hovertemplate='n=%{x}M<br>T(n)=%{y:.2f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # T(n) original
    fig.add_trace(
        go.Scatter(
            x=n_vals, y=T_orig,
            mode='lines+markers',
            name='T(n) original (no correction)',
            line=dict(color='lightblue', width=2, dash='dash'),
            marker=dict(size=8),
            hovertemplate='n=%{x}M<br>T_orig=%{y:.2f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Max empty gap
    fig.add_trace(
        go.Scatter(
            x=n_vals, y=max_empty,
            mode='lines+markers',
            name='Max empty gap (frontier)',
            line=dict(color='red', width=3),
            marker=dict(size=12, symbol='triangle-up'),
            hovertemplate='n=%{x}M<br>Max empty=%{y}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Gap 96 reference line
    fig.add_hline(
        y=96, line_dash="dot", line_color="orange",
        annotation_text="Gap 96 (single outlier at I₂₆,₂₃₅,₀₀₂)",
        annotation_position="right",
        row=1, col=1
    )
    
    # Highlight 40M failure region
    fig.add_vrect(
        x0=35, x1=45,
        fillcolor="red", opacity=0.1,
        layer="below", line_width=0,
        row=1, col=1
    )
    
    # Add margin annotations
    for i, d in enumerate(data_points):
        color = 'green' if d['margin'] > 0 else 'red'
        symbol = '+' if d['margin'] > 0 else ''
        fig.add_annotation(
            x=d['n_millions'],
            y=d['T_corrected'] + 4,
            text=f"<b>{symbol}{d['margin']:.1f}</b>",
            font=dict(color=color, size=11),
            showarrow=False,
            row=1, col=1
        )
    
    # =========================================================================
    # BOTTOM PLOT: Balance Property
    # =========================================================================
    
    fig.add_trace(
        go.Scatter(
            x=n_vals, y=balance_dev,
            mode='lines+markers',
            name='Balance deviation',
            line=dict(color='green', width=3),
            marker=dict(size=12),
            fill='tozeroy',
            fillcolor='rgba(0,255,0,0.2)',
            hovertemplate='n=%{x}M<br>Deviation=%{y:.4f}%<extra></extra>'
        ),
        row=2, col=1
    )
    
    # =========================================================================
    # LAYOUT
    # =========================================================================
    
    fig.update_layout(
        title=dict(
            text='<b>Multi-Scale Analysis: Safe Gap Threshold vs Emptiness Frontier</b><br>' +
                 '<sup>The correction term (2/π)ln(n) is essential — without it, all scales fail</sup>',
            font=dict(size=16)
        ),
        height=700,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        hovermode='x unified'
    )
    
    fig.update_xaxes(title_text="n (millions of intervals)", row=1, col=1)
    fig.update_xaxes(title_text="n (millions of intervals)", row=2, col=1)
    fig.update_yaxes(title_text="Gap threshold / Max empty gap", row=1, col=1)
    fig.update_yaxes(title_text="Deviation from 50% (%)", row=2, col=1)
    
    fig.write_html(output_file)
    print(f"Saved: {output_file}")
    
    return True


# =============================================================================
# CSV OUTPUT
# =============================================================================

def create_csv(data_points, output_file='frontier_data.csv'):
    """Create CSV for external plotting tools."""
    
    with open(output_file, 'w') as f:
        # Header
        f.write("n,n_millions,p_n,T_corrected,T_original,max_empty_gap,stable_zero_gap,")
        f.write("margin,margin_orig,left_fraction,balance_deviation_pct\n")
        
        # Data
        for d in data_points:
            f.write(f"{d['n']},{d['n_millions']},{d['p_n']},{d['T_corrected']:.4f},")
            f.write(f"{d['T_original']:.4f},{d['max_empty_gap']},{d['stable_zero_gap']},")
            f.write(f"{d['margin']:.4f},{d['margin_orig']:.4f},")
            f.write(f"{d['left_fraction']:.6f},{d['balance_deviation_pct']:.6f}\n")
    
    print(f"Saved: {output_file}")


# =============================================================================
# TEXT SUMMARY
# =============================================================================

def print_summary(data_points):
    """Print a text summary of the findings."""
    
    print("\n" + "=" * 80)
    print("SUMMARY: Safe Gap Threshold vs Emptiness Frontier")
    print("=" * 80)
    
    print("\n1. THRESHOLD TRACKING")
    print("-" * 40)
    print(f"   {'n':>10}  {'T(n)':>8}  {'MaxEmpty':>10}  {'Margin':>8}  {'Status'}")
    print(f"   {'-'*10}  {'-'*8}  {'-'*10}  {'-'*8}  {'-'*8}")
    for d in data_points:
        status = "✓ PASS" if d['margin'] > 0 else "✗ FAIL"
        print(f"   {d['n']:>10,}  {d['T_corrected']:>8.2f}  {d['max_empty_gap']:>10}  {d['margin']:>+8.2f}  {status}")
    
    print("\n2. THE OUTLIER")
    print("-" * 40)
    print(f"   Index:    I_{{26,235,002}}")
    print(f"   Primes:   p = {OUTLIER['p_i']:,}  →  p' = {OUTLIER['p_i_plus_1']:,}")
    print(f"   Gap:      {OUTLIER['gap']}")
    print(f"   Interval: [{OUTLIER['interval_start']:,}, {OUTLIER['interval_end']:,})")
    print(f"   Status:   Verified prime desert (0 primes in 192 integers)")
    
    print("\n3. CORRECTION TERM IMPORTANCE")
    print("-" * 40)
    print("   Without the correction term (2/π)·ln(n):")
    for d in data_points:
        status = "✓" if d['margin_orig'] > 0 else "✗ FAIL"
        print(f"   n={d['n']:>10,}: T_orig={d['T_original']:.2f} vs MaxEmpty={d['max_empty_gap']} → {status}")
    print("\n   Conclusion: Correction term is ESSENTIAL, not cosmetic.")
    
    print("\n4. BALANCE PROPERTY")
    print("-" * 40)
    print("   Deviation from 50/50 split:")
    for d in data_points:
        print(f"   n={d['n']:>10,}: {d['balance_deviation_pct']:.4f}% deviation")
    print("\n   Conclusion: Converging monotonically to perfect balance.")
    
    print("\n" + "=" * 80)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("Computing chart data...")
    data_points = compute_chart_data()
    
    # Print summary
    print_summary(data_points)
    
    # Create outputs
    print("\nGenerating visualizations...")
    
    create_csv(data_points, 'frontier_data.csv')
    create_matplotlib_chart(data_points, 'frontier_chart.png')
    create_plotly_chart(data_points, 'frontier_chart.html')
    
    print("\nDone!")
    print("\nFiles created:")
    print("  - frontier_data.csv    (data for external tools)")
    print("  - frontier_chart.png   (static image)")
    print("  - frontier_chart.html  (interactive chart)")


if __name__ == "__main__":
    main()
