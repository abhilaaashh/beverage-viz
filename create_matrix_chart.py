#!/usr/bin/env python3
"""
2x2 Matrix Drift Visualization
Creates a publication-quality quadrant chart showing beverage subcategory
drift in Engagement and CAGR from 2025 to 2028.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects as pe
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# Color palette - clean white/light mode
# Axes: X = CAGR (horizontal), Y = Engagement (vertical)
COLORS = {
    'background': '#FFFFFF',           # White background
    'grid': '#E5E5E5',                  # Light gray grid
    'quadrant_stars': '#E8F5E9',        # Top-right: Stars (light green)
    'quadrant_mature': '#F3E5F5',       # Top-left: Mature (light purple)
    'quadrant_emerging': '#E3F2FD',     # Bottom-right: Emerging (light blue)
    'quadrant_declining': '#FFEBEE',    # Bottom-left: Declining (light red)
    'point_2025': '#90A4AE',            # 2025 points (blue-gray)
    'point_2028': '#1565C0',            # 2028 points (strong blue)
    'arrow_positive': '#00897B',        # Improving (teal)
    'arrow_negative': '#E53935',        # Declining (red)
    'arrow_neutral': '#FB8C00',         # Neutral (orange)
    'text_primary': '#212121',          # Primary text (dark gray)
    'text_secondary': '#616161',        # Secondary text (medium gray)
    'accent': '#1565C0',                # Accent color (blue)
}

FIGURE_SIZE = (48, 36)
DPI = 150

# ============================================================================
# DATA LOADING & PREPROCESSING
# ============================================================================

def load_and_preprocess_data(filepath: str) -> pd.DataFrame:
    """Load CSV and preprocess data for visualization."""
    # Read file and handle mixed delimiter format
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Parse header - split by multiple spaces
    header_raw = [col.strip() for col in lines[0].strip().split('  ') if col.strip()]
    
    # Check if first data row has an extra column (Category)
    first_data = [val.strip() for val in lines[1].strip().split('\t')]
    
    if len(first_data) == len(header_raw) + 1:
        # New format with Category column
        header = ['Category'] + header_raw
    else:
        header = header_raw
    
    # Parse data rows - split by tab
    data = []
    for line in lines[1:]:
        row = [val.strip() for val in line.strip().split('\t')]
        if len(row) == len(header):
            data.append(row)
    
    df = pd.DataFrame(data, columns=header)
    
    # Clean column names
    df.columns = df.columns.str.strip()
    
    # Parse engagement values (handle comma-formatted numbers)
    def parse_number(val):
        if isinstance(val, str):
            return float(val.replace(',', '').strip())
        return float(val)
    
    # Parse CAGR percentages
    def parse_percent(val):
        if isinstance(val, str):
            return float(val.replace('%', '').replace(',', '').strip())
        return float(val)
    
    df['Engagement_2025'] = df['Engagement 2025'].apply(parse_number)
    df['Engagement_2028'] = df['Engagement 2028'].apply(parse_number)
    df['CAGR_2025'] = df['CAGR 2025'].apply(parse_percent)
    df['CAGR_2028'] = df['CAGR 2028'].apply(parse_percent)
    df['Subcategory'] = df['Subcategory'].str.strip()
    
    # Calculate drift magnitude for arrow styling
    df['engagement_change'] = df['Engagement_2028'] - df['Engagement_2025']
    df['cagr_change'] = df['CAGR_2028'] - df['CAGR_2025']
    df['drift_magnitude'] = np.sqrt(
        (np.log10(df['Engagement_2028'] + 1) - np.log10(df['Engagement_2025'] + 1))**2 +
        (df['CAGR_2028']/100 - df['CAGR_2025']/100)**2
    )
    
    # Classify improvement (based on movement toward top-right quadrant)
    df['is_improving'] = (df['engagement_change'] > 0) | (df['cagr_change'] > 0)
    df['net_improvement'] = (
        np.sign(df['engagement_change']) * np.log10(np.abs(df['engagement_change']) + 1) +
        np.sign(df['cagr_change']) * df['cagr_change'] / 50
    )
    
    return df

# ============================================================================
# VISUALIZATION
# ============================================================================

def create_quadrant_chart(df: pd.DataFrame, output_prefix: str = 'quadrant_drift_chart'):
    """Create the main quadrant drift visualization."""
    
    # Set up the figure with light theme
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=FIGURE_SIZE, facecolor=COLORS['background'])
    ax.set_facecolor(COLORS['background'])
    
    # Linear scale for engagement (Y-axis)
    
    # Calculate quadrant boundaries (using median)
    # X-axis = CAGR, Y-axis = Engagement
    all_engagement = np.concatenate([df['Engagement_2025'].values, df['Engagement_2028'].values])
    all_cagr = np.concatenate([df['CAGR_2025'].values, df['CAGR_2028'].values])
    
    x_median = np.median(all_cagr)      # CAGR median for X
    y_median = np.median(all_engagement) # Engagement median for Y
    
    # Get axis limits: X = CAGR, Y = Engagement (linear)
    # Use tighter bounds to zoom into where data is concentrated
    x_min = min(all_cagr.min() - 20, -80)
    x_max = min(all_cagr.max() * 1.05, 550)  # Cap at 550% to reduce empty space
    y_min = 0
    y_max = min(all_engagement.max() * 1.05, 1100)  # Cap to reduce empty space
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    # Draw quadrant backgrounds with subtle shading
    draw_quadrant_backgrounds(ax, x_median, y_median, x_min, x_max, y_min, y_max)
    
    # Draw quadrant divider lines
    ax.axvline(x=x_median, color='#424242', linestyle='--', linewidth=2, alpha=0.7)
    ax.axhline(y=y_median, color='#424242', linestyle='--', linewidth=2, alpha=0.7)
    
    # Draw drift arrows and points
    draw_drift_arrows(ax, df)
    
    # Draw 2025 points (hollow circles) - X=CAGR, Y=Engagement
    ax.scatter(
        df['CAGR_2025'], df['Engagement_2025'],
        s=180, facecolors='none', edgecolors=COLORS['point_2025'],
        linewidths=3, alpha=0.7, zorder=5, label='2025 Position'
    )
    
    # Draw 2028 points (filled circles) - X=CAGR, Y=Engagement
    ax.scatter(
        df['CAGR_2028'], df['Engagement_2028'],
        s=200, c=COLORS['point_2028'], edgecolors='white',
        linewidths=2, alpha=0.9, zorder=6, label='2028 Position'
    )
    
    # Add labels for all trends
    add_all_labels(ax, df, x_median, y_median)
    
    # Add quadrant labels
    add_quadrant_labels(ax, x_median, y_median, x_min, x_max, y_min, y_max)
    
    # Styling
    style_chart(ax, fig)
    
    # Add legend
    add_legend(ax)
    
    # Add title and subtitle
    add_titles(fig)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.90, bottom=0.08, left=0.06, right=0.98)
    
    # Save outputs
    fig.savefig(f'{output_prefix}.png', dpi=DPI, facecolor=COLORS['background'], 
                edgecolor='none', bbox_inches='tight')
    fig.savefig(f'{output_prefix}.svg', format='svg', facecolor=COLORS['background'],
                edgecolor='none', bbox_inches='tight')
    
    print(f"Charts saved: {output_prefix}.png and {output_prefix}.svg")
    plt.close()


def draw_quadrant_backgrounds(ax, x_med, y_med, x_min, x_max, y_min, y_max):
    """Draw subtle colored backgrounds for each quadrant.
    
    X-axis = CAGR, Y-axis = Engagement (log scale)
    - Top-right: High Engagement, High CAGR = Stars
    - Top-left: High Engagement, Low CAGR = Mature
    - Bottom-right: Low Engagement, High CAGR = Emerging
    - Bottom-left: Low Engagement, Low CAGR = Declining
    """
    
    quadrants = [
        # (x_start, x_end, y_start, y_end, color)
        (x_min, x_med, y_med, y_max, COLORS['quadrant_mature']),      # Top-left: High Eng, Low CAGR
        (x_med, x_max, y_med, y_max, COLORS['quadrant_stars']),       # Top-right: High Eng, High CAGR
        (x_min, x_med, y_min, y_med, COLORS['quadrant_declining']),   # Bottom-left: Low Eng, Low CAGR
        (x_med, x_max, y_min, y_med, COLORS['quadrant_emerging']),    # Bottom-right: Low Eng, High CAGR
    ]
    
    for x0, x1, y0, y1, color in quadrants:
        rect = mpatches.Rectangle(
            (x0, y0), x1 - x0, y1 - y0,
            facecolor=color, alpha=0.6, zorder=0
        )
        ax.add_patch(rect)


def draw_drift_arrows(ax, df: pd.DataFrame):
    """Draw arrows showing drift from 2025 to 2028 positions.
    
    X = CAGR, Y = Engagement
    """
    
    for _, row in df.iterrows():
        # X = CAGR, Y = Engagement
        x1, y1 = row['CAGR_2025'], row['Engagement_2025']
        x2, y2 = row['CAGR_2028'], row['Engagement_2028']
        
        # Skip if positions are nearly identical
        if abs(x2 - x1) < 1 and abs(y2 - y1) < 1:
            continue
        
        # Determine arrow color based on net improvement
        net_imp = row['net_improvement']
        if net_imp > 0.5:
            color = COLORS['arrow_positive']
            alpha = min(0.8, 0.3 + abs(net_imp) * 0.1)
        elif net_imp < -0.5:
            color = COLORS['arrow_negative']
            alpha = min(0.8, 0.3 + abs(net_imp) * 0.1)
        else:
            color = COLORS['arrow_neutral']
            alpha = 0.4
        
        # Arrow styling based on magnitude
        linewidth = min(2.5, 0.8 + row['drift_magnitude'] * 0.5)
        
        # Create curved arrow
        arrow = FancyArrowPatch(
            (x1, y1), (x2, y2),
            connectionstyle="arc3,rad=0.1",
            arrowstyle='->,head_length=6,head_width=4',
            color=color,
            linewidth=linewidth,
            alpha=alpha,
            zorder=3,
            path_effects=[pe.withStroke(linewidth=linewidth + 1, foreground=COLORS['background'], alpha=0.5)]
        )
        ax.add_patch(arrow)


def add_all_labels(ax, df: pd.DataFrame, x_med, y_med):
    """Add labels for ALL data points - poster-style large format.
    
    X = CAGR, Y = Engagement
    """
    from adjustText import adjust_text
    
    texts = []
    
    for _, row in df.iterrows():
        # Place labels at 2025 (origin) position, not 2028
        x, y = row['CAGR_2025'], row['Engagement_2025']
        label = row['Subcategory']
        
        # Font size based on engagement for visual hierarchy
        if row['Engagement_2025'] > 400:
            font_size = 13
            font_weight = 'bold'
        elif row['Engagement_2025'] > 200:
            font_size = 11
            font_weight = 'semibold'
        else:
            font_size = 10
            font_weight = 'medium'
        
        text = ax.text(
            x, y, label,
            fontsize=font_size,
            color=COLORS['text_primary'],
            alpha=0.9,
            fontweight=font_weight,
            ha='left',
            va='center',
            path_effects=[pe.withStroke(linewidth=4, foreground=COLORS['background'])],
            zorder=10
        )
        texts.append(text)
    
    # Adjust text positions with strong spacing for all 81 labels
    adjust_text(
        texts,
        ax=ax,
        arrowprops=dict(arrowstyle='-', color='#9E9E9E', alpha=0.6, lw=1),
        expand_points=(4.0, 4.0),
        expand_text=(2.5, 2.5),
        force_points=(2.0, 2.0),
        force_text=(2.0, 2.0),
        lim=3000,
        only_move={'points': 'y', 'texts': 'xy'}
    )


def add_quadrant_labels(ax, x_med, y_med, x_min, x_max, y_min, y_max):
    """Add descriptive labels to each quadrant.
    
    X-axis = CAGR, Y-axis = Engagement (log scale)
    - Top-right: High Engagement, High CAGR = Stars
    - Top-left: High Engagement, Low CAGR = Mature
    - Bottom-right: Low Engagement, High CAGR = Emerging
    - Bottom-left: Low Engagement, Low CAGR = Declining
    """
    
    # Both axes linear - use arithmetic mean for positioning
    labels = [
        ((x_min + x_med) / 2, (y_med + y_max) / 2, 'MATURE\nMARKET', 'center'),       # Top-left
        ((x_med + x_max) / 2, (y_med + y_max) / 2, 'STARS', 'center'),                # Top-right
        ((x_min + x_med) / 2, (y_min + y_med) / 2, 'DECLINING', 'center'),            # Bottom-left
        ((x_med + x_max) / 2, (y_min + y_med) / 2, 'EMERGING\nOPPORTUNITIES', 'center'),  # Bottom-right
    ]
    
    for x, y, text, ha in labels:
        ax.text(
            x, y, text,
            fontsize=20,
            fontweight='bold',
            color=COLORS['text_secondary'],
            alpha=0.4,
            ha='center',
            va='center',
            zorder=1,
            path_effects=[pe.withStroke(linewidth=3, foreground=COLORS['background'])]
        )


def style_chart(ax, fig):
    """Apply premium styling to the chart."""
    
    # Axis labels - X = CAGR, Y = Engagement
    ax.set_xlabel('CAGR (%)', fontsize=18, color=COLORS['text_primary'], 
                  fontweight='medium', labelpad=20)
    ax.set_ylabel('Engagement', fontsize=18, color=COLORS['text_primary'], 
                  fontweight='medium', labelpad=20)
    
    # Tick styling
    ax.tick_params(axis='both', colors=COLORS['text_secondary'], labelsize=14)
    
    # Spine styling
    for spine in ax.spines.values():
        spine.set_color(COLORS['grid'])
        spine.set_linewidth(1.5)
    
    # Grid
    ax.grid(True, alpha=0.15, color=COLORS['text_secondary'], linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)


def add_legend(ax):
    """Add a styled legend."""
    
    from matplotlib.lines import Line2D
    
    legend_elements = [
        Line2D([0], [0], marker='o', color='none', markerfacecolor='none',
               markeredgecolor=COLORS['point_2025'], markersize=14, markeredgewidth=2.5,
               label='2025 Position'),
        Line2D([0], [0], marker='o', color='none', markerfacecolor=COLORS['point_2028'],
               markeredgecolor=COLORS['accent'], markersize=14, markeredgewidth=1.5,
               label='2028 Position'),
        Line2D([0], [0], color=COLORS['arrow_positive'], linewidth=3, label='Improving Trend'),
        Line2D([0], [0], color=COLORS['arrow_negative'], linewidth=3, label='Declining Trend'),
        Line2D([0], [0], color=COLORS['arrow_neutral'], linewidth=3, label='Stable/Mixed'),
    ]
    
    legend = ax.legend(
        handles=legend_elements,
        loc='upper left',
        frameon=True,
        facecolor='white',
        edgecolor='#BDBDBD',
        fontsize=14,
        framealpha=0.95
    )
    
    for text in legend.get_texts():
        text.set_color(COLORS['text_primary'])
    legend.get_frame().set_linewidth(1.5)


def add_titles(fig):
    """Add main title and subtitle."""
    
    fig.suptitle(
        'Beverage Category Drift Analysis: 2025 → 2028',
        fontsize=32,
        fontweight='bold',
        color=COLORS['text_primary'],
        y=0.96
    )
    
    fig.text(
        0.5, 0.93,
        'Tracking shifts in Engagement and Growth Rate across 81 beverage subcategories',
        ha='center',
        fontsize=16,
        color=COLORS['text_secondary'],
        style='italic'
    )


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    # Load data
    df = load_and_preprocess_data('data.csv')
    
    print(f"Loaded {len(df)} beverage subcategories")
    print(f"Engagement range: {df['Engagement_2025'].min():.2f} - {df['Engagement_2025'].max():.2f} (2025)")
    print(f"CAGR range: {df['CAGR_2025'].min():.1f}% - {df['CAGR_2025'].max():.1f}% (2025)")
    
    # Create visualization
    create_quadrant_chart(df, 'quadrant_drift_chart')
