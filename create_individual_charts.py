#!/usr/bin/env python3
"""
Individual 2x2 Matrix Visualizations for 2025 and 2028
Creates separate publication-quality quadrant charts for each year.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

COLORS = {
    'background': '#FFFFFF',
    'grid': '#E5E5E5',
    'quadrant_stars': '#C8E6C9',
    'quadrant_mature': '#E1BEE7',
    'quadrant_emerging': '#B3E5FC',
    'quadrant_declining': '#FFCDD2',
    'point_2025': '#546E7A',
    'point_2028': '#1565C0',
    'text_primary': '#212121',
    'text_secondary': '#424242',
    'accent': '#1565C0',
}

FIGURE_SIZE = (36, 28)
DPI = 150

# ============================================================================
# DATA LOADING & PREPROCESSING
# ============================================================================

def load_and_preprocess_data(filepath: str) -> pd.DataFrame:
    """Load CSV and preprocess data for visualization."""
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    data = []
    for line in lines[1:]:
        row = [val.strip() for val in line.strip().split('\t')]
        if len(row) >= 5:
            data.append(row[:5])
    
    df = pd.DataFrame(data, columns=['Subcategory', 'Engagement 2025', 'CAGR 2025', 'Engagement 2028', 'CAGR 2028'])
    
    def parse_number(val):
        if isinstance(val, str):
            return float(val.replace(',', '').strip())
        return float(val)
    
    def parse_percent(val):
        if isinstance(val, str):
            return float(val.replace('%', '').replace(',', '').strip())
        return float(val)
    
    df['Engagement_2025'] = df['Engagement 2025'].apply(parse_number)
    df['Engagement_2028'] = df['Engagement 2028'].apply(parse_number)
    df['CAGR_2025'] = df['CAGR 2025'].apply(parse_percent)
    df['CAGR_2028'] = df['CAGR 2028'].apply(parse_percent)
    df['Subcategory'] = df['Subcategory'].str.strip()
    
    return df

# ============================================================================
# VISUALIZATION
# ============================================================================

def create_single_year_chart(df: pd.DataFrame, year: str, output_prefix: str):
    """Create a quadrant chart for a single year."""
    
    eng_col = f'Engagement_{year}'
    cagr_col = f'CAGR_{year}'
    point_color = COLORS['point_2025'] if year == '2025' else COLORS['point_2028']
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=FIGURE_SIZE, facecolor=COLORS['background'])
    ax.set_facecolor(COLORS['background'])
    
    engagement = df[eng_col].values
    cagr = df[cagr_col].values
    
    x_median = np.median(cagr)
    y_median = np.median(engagement)
    
    x_min, x_max = min(cagr.min() * 1.2, -100), max(cagr.max() * 1.1, 200)
    y_min, y_max = 0, engagement.max() * 1.1
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    draw_quadrant_backgrounds(ax, x_median, y_median, x_min, x_max, y_min, y_max)
    
    ax.axvline(x=x_median, color='#424242', linestyle='--', linewidth=2, alpha=0.7)
    ax.axhline(y=y_median, color='#424242', linestyle='--', linewidth=2, alpha=0.7)
    
    ax.scatter(
        cagr, engagement,
        s=250, c=point_color, edgecolors='white',
        linewidths=2, alpha=0.85, zorder=6
    )
    
    add_labels(ax, df, cagr_col, eng_col)
    add_quadrant_labels(ax, x_median, y_median, x_min, x_max, y_min, y_max)
    style_chart(ax, fig)
    add_single_year_legend(ax, year, point_color)
    add_single_year_titles(fig, year)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.90, bottom=0.08, left=0.06, right=0.98)
    
    fig.savefig(f'{output_prefix}.png', dpi=DPI, facecolor=COLORS['background'], 
                edgecolor='none', bbox_inches='tight')
    fig.savefig(f'{output_prefix}.svg', format='svg', facecolor=COLORS['background'],
                edgecolor='none', bbox_inches='tight')
    
    print(f"Charts saved: {output_prefix}.png and {output_prefix}.svg")
    plt.close()


def draw_quadrant_backgrounds(ax, x_med, y_med, x_min, x_max, y_min, y_max):
    """Draw subtle colored backgrounds for each quadrant."""
    
    quadrants = [
        (x_min, x_med, y_med, y_max, COLORS['quadrant_mature']),
        (x_med, x_max, y_med, y_max, COLORS['quadrant_stars']),
        (x_min, x_med, y_min, y_med, COLORS['quadrant_declining']),
        (x_med, x_max, y_min, y_med, COLORS['quadrant_emerging']),
    ]
    
    for x0, x1, y0, y1, color in quadrants:
        rect = mpatches.Rectangle(
            (x0, y0), x1 - x0, y1 - y0,
            facecolor=color, alpha=0.6, zorder=0
        )
        ax.add_patch(rect)


def add_labels(ax, df: pd.DataFrame, cagr_col: str, eng_col: str):
    """Add labels for all data points."""
    from adjustText import adjust_text
    
    texts = []
    
    for _, row in df.iterrows():
        x, y = row[cagr_col], row[eng_col]
        label = row['Subcategory']
        
        # Smaller fonts for dense low-engagement area
        if row[eng_col] < 100:
            font_size = 12
            font_weight = 'medium'
        elif row[eng_col] < 200:
            font_size = 16
            font_weight = 'semibold'
        else:
            font_size = 22
            font_weight = 'bold'
        
        text = ax.text(
            x, y, label,
            fontsize=font_size,
            color=COLORS['text_primary'],
            alpha=0.95,
            fontweight=font_weight,
            ha='left',
            va='center',
            zorder=10
        )
        texts.append(text)
    
    adjust_text(
        texts,
        ax=ax,
        expand_points=(5.0, 5.0),
        expand_text=(3.0, 3.0),
        force_points=(3.0, 3.0),
        force_text=(2.5, 2.5),
        lim=3000,
        only_move={'points': 'y', 'texts': 'xy'}
    )


def add_quadrant_labels(ax, x_med, y_med, x_min, x_max, y_min, y_max):
    """Add descriptive labels to each quadrant."""
    
    labels = [
        ((x_min + x_med) / 2, (y_med + y_max) / 2, 'MATURE\nMARKET', '#8E24AA'),
        ((x_med + x_max) / 2, (y_med + y_max) / 2, 'STARS', '#43A047'),
        ((x_min + x_med) / 2, (y_min + y_med) / 2, 'DECLINING', '#E53935'),
        ((x_med + x_max) / 2, (y_min + y_med) / 2, 'EMERGING\nOPPORTUNITIES', '#1E88E5'),
    ]
    
    for x, y, text, color in labels:
        ax.text(
            x, y, text,
            fontsize=28,
            fontweight='bold',
            color=color,
            alpha=0.55,
            ha='center',
            va='center',
            zorder=1,
            path_effects=[pe.withStroke(linewidth=4, foreground=COLORS['background'])]
        )


def style_chart(ax, fig):
    """Apply premium styling to the chart."""
    
    ax.set_xlabel('CAGR (%)', fontsize=18, color=COLORS['text_primary'], 
                  fontweight='medium', labelpad=20)
    ax.set_ylabel('Engagement', fontsize=18, color=COLORS['text_primary'], 
                  fontweight='medium', labelpad=20)
    
    ax.tick_params(axis='both', colors=COLORS['text_secondary'], labelsize=14)
    
    for spine in ax.spines.values():
        spine.set_color(COLORS['grid'])
        spine.set_linewidth(1.5)
    
    ax.grid(True, alpha=0.15, color=COLORS['text_secondary'], linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)


def add_single_year_legend(ax, year: str, point_color: str):
    """Add a styled legend for single year chart."""
    from matplotlib.lines import Line2D
    
    legend_elements = [
        Line2D([0], [0], marker='o', color='none', markerfacecolor=point_color,
               markeredgecolor='white', markersize=14, markeredgewidth=1.5,
               label=f'{year} Position'),
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


def add_single_year_titles(fig, year: str):
    """Add main title and subtitle for single year."""
    
    fig.suptitle(
        f'Beverage Category Analysis: {year}',
        fontsize=32,
        fontweight='bold',
        color=COLORS['text_primary'],
        y=0.96
    )
    
    fig.text(
        0.5, 0.93,
        f'Engagement vs Growth Rate across 81 beverage subcategories ({year} data)',
        ha='center',
        fontsize=16,
        color=COLORS['text_secondary'],
        style='italic'
    )


def create_quadrant_zoom_chart(df: pd.DataFrame, year: str, quadrant: str, output_prefix: str):
    """Create a zoomed chart for a single quadrant."""
    
    eng_col = f'Engagement_{year}'
    cagr_col = f'CAGR_{year}'
    point_color = COLORS['point_2025'] if year == '2025' else COLORS['point_2028']
    
    engagement = df[eng_col].values
    cagr = df[cagr_col].values
    
    x_median = np.median(cagr)
    y_median = np.median(engagement)
    
    # Filter data for this quadrant
    if quadrant == 'stars':
        mask = (df[cagr_col] >= x_median) & (df[eng_col] >= y_median)
        quadrant_color = COLORS['quadrant_stars']
        label_color = '#43A047'
        title = 'STARS'
        subtitle = 'High Engagement, High Growth'
    elif quadrant == 'mature':
        mask = (df[cagr_col] < x_median) & (df[eng_col] >= y_median)
        quadrant_color = COLORS['quadrant_mature']
        label_color = '#8E24AA'
        title = 'MATURE MARKET'
        subtitle = 'High Engagement, Low Growth'
    elif quadrant == 'emerging':
        mask = (df[cagr_col] >= x_median) & (df[eng_col] < y_median)
        quadrant_color = COLORS['quadrant_emerging']
        label_color = '#1E88E5'
        title = 'EMERGING OPPORTUNITIES'
        subtitle = 'Low Engagement, High Growth'
    else:  # declining
        mask = (df[cagr_col] < x_median) & (df[eng_col] < y_median)
        quadrant_color = COLORS['quadrant_declining']
        label_color = '#E53935'
        title = 'DECLINING'
        subtitle = 'Low Engagement, Low Growth'
    
    df_quad = df[mask].copy()
    
    if len(df_quad) == 0:
        print(f"No data points in {quadrant} quadrant for {year}")
        return
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(28, 22), facecolor=COLORS['background'])
    ax.set_facecolor(quadrant_color)
    
    # Set axis limits based on quadrant data with padding
    quad_cagr = df_quad[cagr_col].values
    quad_eng = df_quad[eng_col].values
    
    x_pad = (quad_cagr.max() - quad_cagr.min()) * 0.15 + 10
    y_pad = (quad_eng.max() - quad_eng.min()) * 0.15 + 20
    
    ax.set_xlim(quad_cagr.min() - x_pad, quad_cagr.max() + x_pad)
    ax.set_ylim(max(0, quad_eng.min() - y_pad), quad_eng.max() + y_pad)
    
    # Draw points
    ax.scatter(
        quad_cagr, quad_eng,
        s=350, c=point_color, edgecolors='white',
        linewidths=2.5, alpha=0.85, zorder=6
    )
    
    # Add labels
    add_quadrant_zoom_labels(ax, df_quad, cagr_col, eng_col)
    
    # Styling
    style_chart(ax, fig)
    
    # Add legend
    add_single_year_legend(ax, year, point_color)
    
    # Add title
    fig.suptitle(
        f'{title} — {year}',
        fontsize=32,
        fontweight='bold',
        color=label_color,
        y=0.96
    )
    
    fig.text(
        0.5, 0.93,
        f'{subtitle} ({len(df_quad)} subcategories)',
        ha='center',
        fontsize=16,
        color=COLORS['text_secondary'],
        style='italic'
    )
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.90, bottom=0.08, left=0.06, right=0.98)
    
    fig.savefig(f'{output_prefix}.png', dpi=DPI, facecolor=COLORS['background'], 
                edgecolor='none', bbox_inches='tight')
    fig.savefig(f'{output_prefix}.svg', format='svg', facecolor=COLORS['background'],
                edgecolor='none', bbox_inches='tight')
    
    print(f"  {title}: {len(df_quad)} items -> {output_prefix}.png")
    plt.close()


def add_quadrant_zoom_labels(ax, df: pd.DataFrame, cagr_col: str, eng_col: str):
    """Add labels for zoomed quadrant chart."""
    from adjustText import adjust_text
    
    texts = []
    
    for _, row in df.iterrows():
        x, y = row[cagr_col], row[eng_col]
        label = row['Subcategory']
        
        text = ax.text(
            x, y, label,
            fontsize=22,
            color=COLORS['text_primary'],
            alpha=0.95,
            fontweight='bold',
            ha='left',
            va='center',
            zorder=10
        )
        texts.append(text)
    
    adjust_text(
        texts,
        ax=ax,
        expand_points=(5.0, 5.0),
        expand_text=(3.0, 3.0),
        force_points=(3.0, 3.0),
        force_text=(2.5, 2.5),
        lim=3000,
        only_move={'points': 'y', 'texts': 'xy'}
    )


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    df = load_and_preprocess_data('data.csv')
    
    print(f"Loaded {len(df)} beverage subcategories")
    
    print("\n--- Creating 2025 Chart ---")
    print(f"Engagement range: {df['Engagement_2025'].min():.2f} - {df['Engagement_2025'].max():.2f}")
    print(f"CAGR range: {df['CAGR_2025'].min():.1f}% - {df['CAGR_2025'].max():.1f}%")
    create_single_year_chart(df, '2025', 'quadrant_chart_2025')
    
    print("\n--- Creating 2028 Chart ---")
    print(f"Engagement range: {df['Engagement_2028'].min():.2f} - {df['Engagement_2028'].max():.2f}")
    print(f"CAGR range: {df['CAGR_2028'].min():.1f}% - {df['CAGR_2028'].max():.1f}%")
    create_single_year_chart(df, '2028', 'quadrant_chart_2028')
    
    # Create zoomed quadrant charts
    quadrants = ['stars', 'mature', 'emerging', 'declining']
    
    print("\n--- Creating 2025 Quadrant Zoom Charts ---")
    for quad in quadrants:
        create_quadrant_zoom_chart(df, '2025', quad, f'quadrant_2025_{quad}')
    
    print("\n--- Creating 2028 Quadrant Zoom Charts ---")
    for quad in quadrants:
        create_quadrant_zoom_chart(df, '2028', quad, f'quadrant_2028_{quad}')
    
    print("\nAll charts generated successfully!")
