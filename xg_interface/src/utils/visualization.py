"""
Visualization utilities for the xG prediction application.
All visualizations use vertical pitch orientation via mplsoccer VerticalPitch.
"""

import matplotlib.pyplot as plt
from mplsoccer import Pitch, VerticalPitch
import pandas as pd
import numpy as np
import io
import base64
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import gaussian_kde

# Import alternative visualization approaches
try:
    from .visualization_seaborn import create_seaborn_shot_map, create_bokeh_shot_map
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False


def create_shot_map(df: pd.DataFrame, title: str = "Shot Map with xG") -> tuple:
    """
    Create a professional vertical shot map visualization showing shots with high contrast design.
    
    Args:
        df: DataFrame containing shot data with 'start_x', 'start_y', and 'xG' columns
        title: Title for the plot
        
    Returns:
        Tuple of (figure, axis) objects
    """
    # DEBUG: Print data info
    print(f"DEBUG - Shot Map Data Info:")
    print(f"- Number of shots: {len(df)}")
    print(f"- Columns: {df.columns.tolist()}")
    if len(df) > 0:
        print(f"- X range: {df['start_x'].min():.2f} to {df['start_x'].max():.2f}")
        print(f"- Y range: {df['start_y'].min():.2f} to {df['start_y'].max():.2f}")
        print(f"- xG range: {df['xG'].min():.3f} to {df['xG'].max():.3f}")
        print(f"- Sample data:")
        print(df[['start_x', 'start_y', 'xG']].head())
    
    # Validate coordinate ranges for StatsBomb pitch (0-120 x 0-80)
    valid_shots = df[
        (df['start_x'] >= 0) & (df['start_x'] <= 120) & 
        (df['start_y'] >= 0) & (df['start_y'] <= 80)
    ].copy()
    
    if len(valid_shots) < len(df):
        print(f"WARNING: Filtered out {len(df) - len(valid_shots)} shots with invalid coordinates")
    
    if len(valid_shots) == 0:
        print("ERROR: No valid shots to display!")
        # Create empty plot
        pitch = VerticalPitch(pitch_type='statsbomb', pitch_color='black', line_color='white', linewidth=4)
        fig, ax = pitch.draw(figsize=(12, 18))
        ax.text(60, 40, "No valid shots to display", ha='center', va='center', 
                fontsize=20, color='white', fontweight='bold')
        return fig, ax
    
    # Create the vertical pitch with maximum contrast
    pitch = VerticalPitch(pitch_type='statsbomb', pitch_color='black', line_color='white', 
                          linewidth=4)
    fig, ax = pitch.draw(figsize=(12, 18), constrained_layout=True, tight_layout=False)
    
    # Set black background for maximum contrast
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')

    # DRAMATICALLY BRIGHT APPROACH: Single large marker per shot
    for i, (idx, row) in enumerate(valid_shots.iterrows()):
        x, y, xg = row['start_x'], row['start_y'], row['xG']
        
        # Determine base color based on xG with NEON BRIGHT colors
        if xg >= 0.7:
            color = '#FF0040'  # Neon Red - Very high xG
        elif xg >= 0.5:
            color = '#FF8000'  # Neon Orange - High xG  
        elif xg >= 0.3:
            color = '#FFFF00'  # Neon Yellow - Medium xG
        elif xg >= 0.1:
            color = '#00FFFF'  # Neon Cyan - Low-Medium xG
        else:
            color = '#4080FF'  # Neon Blue - Very low xG
        
        # Calculate size based on xG (reduced for full-pitch clarity)
        size = (xg * 700) + 120  # smaller, readable on full pitch
        
        # Single ultra-bright marker with thick outline
        ax.scatter(x, y, 
                  s=size, 
                  c=color, 
                  alpha=1.0,
                  edgecolors='white', 
                  linewidths=8, 
                  zorder=20)

    # Create high-contrast legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF0040', 
                  markeredgecolor='white', markeredgewidth=4, markersize=18, 
                  linestyle='None', label='Very High xG (≥0.7)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF8000', 
                  markeredgecolor='white', markeredgewidth=4, markersize=15, 
                  linestyle='None', label='High xG (0.5-0.7)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#FFFF00', 
                  markeredgecolor='white', markeredgewidth=4, markersize=12, 
                  linestyle='None', label='Medium xG (0.3-0.5)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#00FFFF', 
                  markeredgecolor='white', markeredgewidth=4, markersize=10, 
                  linestyle='None', label='Low xG (0.1-0.3)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#4080FF', 
                  markeredgecolor='white', markeredgewidth=4, markersize=8, 
                  linestyle='None', label='Very Low xG (<0.1)')
    ]
    
    legend = ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1), 
                      fontsize=12, facecolor='black', edgecolor='white', labelcolor='white')
    legend.get_frame().set_linewidth(2)

    # High-contrast title
    ax.set_title(title, fontsize=24, color='white', fontweight='bold', pad=40,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="black", edgecolor="white", linewidth=2))
    
    return fig, ax


def create_half_pitch_shot_map(df: pd.DataFrame, title: str = "Half Pitch Shot Map") -> tuple:
    """
    Create a professional vertical half-pitch shot map with ultra-bright visualization.
    
    Args:
        df: DataFrame containing shot data with 'start_x', 'start_y', and 'xG' columns
        title: Title for the plot
        
    Returns:
        Tuple of (figure, axis) objects
    """
    # DEBUG: Print data info
    print(f"DEBUG - Half Pitch Data Info:")
    print(f"- Number of shots: {len(df)}")
    if len(df) > 0:
        print(f"- X range: {df['start_x'].min():.2f} to {df['start_x'].max():.2f}")
        print(f"- Y range: {df['start_y'].min():.2f} to {df['start_y'].max():.2f}")
    
    # Filter for attacking half (x >= 60) and valid coordinates
    valid_shots = df[
        (df['start_x'] >= 60) & (df['start_x'] <= 120) & 
        (df['start_y'] >= 0) & (df['start_y'] <= 80)
    ].copy()
    
    print(f"DEBUG - Valid shots in attacking half: {len(valid_shots)}")
    
    if len(valid_shots) == 0:
        print("ERROR: No valid shots in attacking half!")
        # Create empty plot
        pitch = VerticalPitch(pitch_type='statsbomb', pitch_color='black', line_color='white', 
                              linewidth=4, half=True)
        fig, ax = pitch.draw(figsize=(14, 14))
        ax.text(40, 90, "No shots in attacking half", ha='center', va='center', 
                fontsize=20, color='white', fontweight='bold')
        return fig, ax
    
    # Create vertical half pitch with maximum contrast
    pitch = VerticalPitch(pitch_type='statsbomb', pitch_color='black', line_color='white', 
                          linewidth=4, half=True)
    fig, ax = pitch.draw(figsize=(14, 14), constrained_layout=True, tight_layout=False)
    
    # Set black background for maximum contrast
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')

    # ULTRA-BRIGHT single marker approach
    for i, (idx, row) in enumerate(valid_shots.iterrows()):
        x, y, xg = row['start_x'], row['start_y'], row['xG']
        
        # Determine base color based on xG - using ultra-bright neon colors
        if xg >= 0.7:
            color = '#FF0040'  # Neon Red - Very high xG
        elif xg >= 0.5:
            color = '#FF8000'  # Neon Orange - High xG  
        elif xg >= 0.3:
            color = '#FFFF00'  # Neon Yellow - Medium xG
        elif xg >= 0.1:
            color = '#00FFFF'  # Neon Cyan - Low-Medium xG
        else:
            color = '#4080FF'  # Neon Blue - Very low xG
        
        # Calculate MASSIVE size for half pitch (even bigger)
        size = (xg * 10000) + 3000  # Enormous markers
        
        # Single ultra-bright marker with ultra-thick white outline
        ax.scatter(x, y, 
                  s=size, 
                  c=color, 
                  alpha=1.0,
                  edgecolors='white', 
                  linewidths=10, 
                  zorder=20)

    # Create high-contrast legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF0040', 
                  markeredgecolor='white', markeredgewidth=4, markersize=18, 
                  linestyle='None', label='Very High xG (≥0.7)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF8000', 
                  markeredgecolor='white', markeredgewidth=4, markersize=15, 
                  linestyle='None', label='High xG (0.5-0.7)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#FFFF00', 
                  markeredgecolor='white', markeredgewidth=4, markersize=12, 
                  linestyle='None', label='Medium xG (0.3-0.5)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#00FFFF', 
                  markeredgecolor='white', markeredgewidth=4, markersize=10, 
                  linestyle='None', label='Low xG (0.1-0.3)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#4080FF', 
                  markeredgecolor='white', markeredgewidth=4, markersize=8, 
                  linestyle='None', label='Very Low xG (<0.1)')
    ]
    
    legend = ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1), 
                      fontsize=12, facecolor='black', edgecolor='white', labelcolor='white')
    legend.get_frame().set_linewidth(3)

    # High-contrast title
    ax.set_title(title, fontsize=24, color='white', fontweight='bold', pad=40,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="black", edgecolor="white", linewidth=3))
    
    # Add subtitle with data info
    total_xg = valid_shots['xG'].sum()
    ax.text(90, 55, f'Final Third Focus | {len(valid_shots)} shots | Total xG: {total_xg:.2f}', 
            ha='center', va='center', fontsize=14, color='white', fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="black", edgecolor="white", alpha=0.8))
    
    return fig, ax


def create_single_shot_visualization(x: float, y: float, xg_value: float, 
                                   title: str = "Shot Location Preview") -> tuple:
    """
    Create a revolutionary single shot visualization with dramatic glow effects.
    
    Args:
        x: X coordinate of the shot
        y: Y coordinate of the shot
        xg_value: xG value of the shot (optional, for annotation)
        title: Title for the plot
        
    Returns:
        Tuple of (figure, axis) objects
    """
    # Create professional vertical pitch styling with black background
    pitch = VerticalPitch(pitch_type='statsbomb', pitch_color='black', line_color='white', 
                          linewidth=4)
    fig, ax = pitch.draw(figsize=(10, 16), constrained_layout=True, tight_layout=False)
    
    # Set black background for maximum contrast
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    
    # Determine color based on xG value using dramatic neon colors
    if xg_value >= 0.7:
        base_color = '#FF0000'  # Pure Red - Very high xG
        glow_color = '#FF6666'  # Red glow
        ring_color = '#FFAAAA'  # Light red ring
    elif xg_value >= 0.5:
        base_color = '#FF6600'  # Orange Red - High xG
        glow_color = '#FF9966'  # Orange glow
        ring_color = '#FFCC99'  # Light orange ring
    elif xg_value >= 0.3:
        base_color = '#FFAA00'  # Orange - Medium xG
        glow_color = '#FFCC66'  # Orange glow
        ring_color = '#FFDD99'  # Light orange ring
    elif xg_value >= 0.1:
        base_color = '#00FFFF'  # Cyan - Low-Medium xG
        glow_color = '#66FFFF'  # Cyan glow
        ring_color = '#99FFFF'  # Light cyan ring
    else:
        base_color = '#0066FF'  # Blue - Very low xG
        glow_color = '#6699FF'  # Blue glow
        ring_color = '#99CCFF'  # Light blue ring
    
    # Calculate dramatic size for single shot
    base_size = xg_value * 5000 + 1000
    
    # MASSIVE MULTI-LAYER GLOW EFFECT for single shot
    # Outer rings for dramatic effect
    for i in range(5, 0, -1):
        alpha = 0.1 + (i * 0.05)
        size_mult = 2.0 + (i * 0.5)
        ax.scatter(x, y, s=base_size * size_mult, c=ring_color, alpha=alpha, 
                  edgecolors='none', zorder=5+i)
    
    # Main glow layers
    ax.scatter(x, y, s=base_size * 2.5, c=glow_color, alpha=0.4, 
              edgecolors='none', zorder=12)
    ax.scatter(x, y, s=base_size * 1.8, c=glow_color, alpha=0.6, 
              edgecolors='none', zorder=13)
    ax.scatter(x, y, s=base_size * 1.3, c=glow_color, alpha=0.8, 
              edgecolors='none', zorder=14)
    
    # Main marker with ultra-thick white outline
    ax.scatter(x, y, s=base_size, c=base_color, alpha=1.0,
              edgecolors='white', linewidths=10, zorder=20)
    
    # Bright white core for extra drama
    ax.scatter(x, y, s=base_size * 0.25, c='white', alpha=1.0,
              edgecolors='none', zorder=21)
    
    # Add dramatic annotation with xG value
    ax.annotate(f"xG: {xg_value:.3f}", 
                xy=(x, y), 
                xytext=(15, -30), textcoords='offset points', 
                fontsize=20, color='white', ha='center', va='center', fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.8", facecolor=base_color, edgecolor="white", 
                         alpha=0.9, linewidth=4),
                zorder=25,
                arrowprops=dict(arrowstyle='->', color='white', lw=3))

    # Dramatic title styling
    ax.set_title(title, fontsize=24, color='white', fontweight='bold', pad=40,
                bbox=dict(boxstyle="round,pad=0.8", facecolor="black", edgecolor="white", linewidth=4))
    
    # Add coordinate info with dramatic styling
    ax.text(x, -15, f'Position: ({x:.1f}, {y:.1f})', 
            ha='center', va='center', fontsize=16, color='white', fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="black", edgecolor="white", alpha=0.8))
    
    return fig, ax


def create_preview_pitch(x: float, y: float) -> tuple:
    """
    Create a simple preview pitch showing shot location.
    
    Args:
        x: X coordinate of the shot
        y: Y coordinate of the shot
        
    Returns:
        Tuple of (figure, axis) objects
    """
    pitch = VerticalPitch(pitch_type='statsbomb', pitch_color='#f4f4f4', line_color='grey')
    fig, ax = pitch.draw(figsize=(6, 9))
    pitch.scatter(x, y, ax=ax, s=150, color='red', edgecolors='black', zorder=2)
    ax.set_title("Shot Location Preview")
    
    return fig, ax


def save_figure_to_bytes(fig, format='png', dpi=300):
    """
    Save matplotlib figure to bytes for download.
    
    Args:
        fig: Matplotlib figure object
        format: File format ('png', 'pdf', 'svg')
        dpi: Resolution for raster formats
        
    Returns:
        Bytes object of the saved figure
    """
    buf = io.BytesIO()
    fig.savefig(buf, format=format, dpi=dpi, bbox_inches='tight', 
                facecolor='#22312b', edgecolor='none')
    buf.seek(0)
    return buf.getvalue()


def save_plotly_figure_to_bytes(fig, format_type: str = 'png', dpi: int = 300) -> bytes:
    """
    Save Plotly figure to bytes for download.
    
    Args:
        fig: Plotly Figure object
        format_type: File format ('png', 'jpg', 'pdf', 'svg')
        dpi: Resolution for raster formats
        
    Returns:
        Bytes data of the saved figure
    """
    import plotly.io as pio
    
    if format_type.lower() in ['png', 'jpg', 'jpeg']:
        img_bytes = pio.to_image(fig, format=format_type, width=1200, height=1800, scale=2)
    elif format_type.lower() == 'pdf':
        img_bytes = pio.to_image(fig, format='pdf', width=1200, height=1800)
    elif format_type.lower() == 'svg':
        img_bytes = pio.to_image(fig, format='svg', width=1200, height=1800)
    else:
        # Default to PNG
        img_bytes = pio.to_image(fig, format='png', width=1200, height=1800, scale=2)
    
    return img_bytes


def create_download_filename(prefix, extension):
    """
    Create a timestamped filename for downloads.
    
    Args:
        prefix: Filename prefix
        extension: File extension (without dot)
        
    Returns:
        Formatted filename with timestamp
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}.{extension}"


def prepare_csv_download(df: pd.DataFrame) -> str:
    """
    Prepare DataFrame for CSV download.
    
    Args:
        df: DataFrame to convert
        
    Returns:
        CSV string
    """
    return df.to_csv(index=False)


def create_custom_shots_visualization(df: pd.DataFrame, title: str = "Custom Shots Analysis") -> tuple:
    """
    Create a professional vertical visualization for multiple custom shots.
    
    Args:
        df: DataFrame containing custom shots with 'start_x', 'start_y', and 'xg_value' columns
        title: Title for the plot
        
    Returns:
        Tuple of (figure, axis) objects
    """
    if df.empty:
        # Return professional empty plot if no data
        pitch = VerticalPitch(pitch_type='statsbomb', pitch_color='#0d1117', line_color='#30363d', 
                      linewidth=2)
        fig, ax = pitch.draw(figsize=(10, 16))
        fig.patch.set_facecolor('#0d1117')
        ax.set_facecolor('#0d1117')
        ax.set_title("No custom shots to display", fontsize=22, color='#8b949e', fontweight='bold')
        ax.text(60, 40, 'Create some custom shots to see the shot map visualization', 
                ha='center', va='center', fontsize=14, color='#8b949e', style='italic')
        return fig, ax
    
    # Create professional vertical pitch styling
    pitch = VerticalPitch(pitch_type='statsbomb', pitch_color='#0d1117', line_color='#30363d', 
                  linewidth=2)
    fig, ax = pitch.draw(figsize=(10, 16), constrained_layout=True, tight_layout=False)
    
    # Set dark background for professional look
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')

    # Plot the shots with professional styling based on xG value - mplsoccer handles vertical orientation
    size = df['xg_value'] * 900 + 150  # Size based on xG value, larger for visibility
    
    sc = pitch.scatter(df.start_x, df.start_y,
                       s=size,
                       c=df['xg_value'],
                       cmap='hot',  # Hot colormap for maximum contrast against green
                       ax=ax,
                       edgecolors='white',  # White edges for maximum contrast
                       linewidths=2.5,
                       alpha=1.0,
                       vmin=0, vmax=1,
                       zorder=10)  # High z-order to ensure shots are on top

    # Professional colorbar
    cbar = fig.colorbar(sc, ax=ax, shrink=0.8, aspect=20, pad=0.02)
    cbar.set_label('Expected Goals (xG) Value', color='white', fontsize=14, labelpad=20)
    cbar.ax.yaxis.set_tick_params(color='white', labelsize=12)
    cbar.ax.yaxis.set_ticklabels([f'{x:.2f}' for x in cbar.get_ticks()], color='white')
    cbar.outline.set_edgecolor('white')
    cbar.outline.set_linewidth(1)

    # Add shot labels if there are few shots (professional styling) using original coordinates
    if len(df) <= 8:  # Reduced number for clarity
        for _, shot in df.iterrows():
            pitch.annotate(
                f"{shot.get('shot_name', 'Shot')}\n{shot['xg_value']:.3f}",
                xy=(shot['start_x'], shot['start_y']),
                ax=ax,
                xytext=(8, 8),
                textcoords='offset points',
                fontsize=10,
                color='white',
                ha='left',
                va='bottom',
                fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.4", fc="black", ec="white", alpha=0.8, linewidth=1),
                zorder=15  # Higher z-order than shot markers
            )

    # Professional title styling
    ax.set_title(title, fontsize=22, color='white', fontweight='bold', pad=30)
    
    return fig, ax


def create_shot_heat_map(df: pd.DataFrame, title: str = "Shot Heat Map") -> tuple:
    """
    Create a professional smooth heat map visualization showing shot density.
    
    Args:
        df: DataFrame containing shot data with 'start_x', 'start_y', and 'xG' columns
        title: Title for the plot
        
    Returns:
        Tuple of (figure, axis) objects
    """
    # Create the pitch with professional styling
    pitch = VerticalPitch(pitch_type='statsbomb', pitch_color='#0d1117', line_color='#30363d', linewidth=2)
    fig, ax = pitch.draw(figsize=(10, 16), constrained_layout=True, tight_layout=False)
    
    # Set dark background for professional look
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')

    # Create smooth continuous heat map using kde
    from scipy.stats import gaussian_kde
    from matplotlib.colors import LinearSegmentedColormap
    import numpy as np
    
    # Create custom red-to-white colormap for better visibility
    colors = ['#000000', '#1a0000', '#330000', '#4d0000', '#660000', '#800000', 
              '#990000', '#b30000', '#cc0000', '#e60000', '#ff0000', '#ff1a1a', 
              '#ff3333', '#ff4d4d', '#ff6666', '#ff8080', '#ff9999', '#ffb3b3', '#ffcccc', '#ffffff']
    custom_cmap = LinearSegmentedColormap.from_list('smooth_red', colors, N=256)
    
    # Create higher resolution grid for smoother heat map (adjusted for vertical pitch)
    x_grid = np.linspace(0, 80, 134)  # Width of vertical pitch
    y_grid = np.linspace(0, 120, 200)  # Height of vertical pitch
    X, Y = np.meshgrid(x_grid, y_grid)
    
    # Transform coordinates for vertical pitch display
    # For VerticalPitch: horizontal x becomes vertical y, horizontal y becomes vertical x
    df_transformed = df.copy()
    df_transformed['start_x_vertical'] = df['start_y']  # horizontal y becomes vertical x
    df_transformed['start_y_vertical'] = df['start_x']  # horizontal x becomes vertical y (NO inversion)
    
    # Create positions array with transformed coordinates
    positions = np.column_stack([df_transformed.start_x_vertical.values, df_transformed.start_y_vertical.values])
    
    if len(positions) > 1:
        # Create KDE with optimal bandwidth for smooth coverage
        kde = gaussian_kde(positions.T)
        kde.set_bandwidth(bw_method=0.4)  # Larger bandwidth for smoother, more connected areas
        
        # Evaluate KDE on grid
        Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
        
        # Normalize Z to a good range for visualization
        Z_normalized = (Z - Z.min()) / (Z.max() - Z.min())
        
        # Apply power transformation for better visual contrast
        Z_enhanced = np.power(Z_normalized, 0.7)
        
        # Create smooth heat map with many levels for smoothness
        levels = np.linspace(0.05, 1.0, 50)  # Start from 0.05 to avoid showing noise
        im = ax.contourf(X, Y, Z_enhanced, levels=levels, cmap=custom_cmap, alpha=0.85, extend='max')
        
        # Add very subtle contour lines for definition
        contour_levels = np.linspace(0.2, 1.0, 8)
        ax.contour(X, Y, Z_enhanced, levels=contour_levels, colors='white', alpha=0.15, linewidths=0.5)
        
    else:
        # Fallback for single point - create a smooth gradient around it (use transformed coordinates)
        x_center, y_center = df_transformed.start_x_vertical.iloc[0], df_transformed.start_y_vertical.iloc[0]
        
        # Create circular gradient around the point
        distance = np.sqrt((X - x_center)**2 + (Y - y_center)**2)
        Z_single = np.exp(-distance**2 / (2 * 8**2))  # Gaussian with sigma=8
        
        im = ax.contourf(X, Y, Z_single, levels=30, cmap=custom_cmap, alpha=0.85, extend='max')

    # Professional colorbar styling
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, aspect=20, pad=0.02)
    cbar.set_label('Shot Density', color='white', fontsize=14, labelpad=20)
    cbar.ax.yaxis.set_tick_params(color='white', labelsize=12)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')
    cbar.outline.set_edgecolor('white')
    cbar.outline.set_linewidth(1)

    # Professional title styling
    ax.set_title(title, fontsize=22, color='white', fontweight='bold', pad=30)
    
    return fig, ax


def create_half_pitch_heat_map(df: pd.DataFrame, title: str = "Half Pitch Heat Map") -> tuple:
    """
    Create a professional smooth half-pitch heat map visualization.
    
    Args:
        df: DataFrame containing shot data with 'start_x', 'start_y', and 'xG' columns
        title: Title for the plot
        
    Returns:
        Tuple of (figure, axis) objects
    """
    # Create half pitch with professional styling
    pitch = VerticalPitch(pitch_type='statsbomb', pitch_color='#0d1117', line_color='#30363d', 
                  linewidth=2, half=True)
    fig, ax = pitch.draw(figsize=(10, 8), constrained_layout=True, tight_layout=False)
    
    # Set dark background for professional look
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')

    # Create smooth continuous heat map using kde
    from scipy.stats import gaussian_kde
    from matplotlib.colors import LinearSegmentedColormap
    import numpy as np
    
    # Create custom red-to-white colormap for better visibility
    colors = ['#000000', '#1a0000', '#330000', '#4d0000', '#660000', '#800000', 
              '#990000', '#b30000', '#cc0000', '#e60000', '#ff0000', '#ff1a1a', 
              '#ff3333', '#ff4d4d', '#ff6666', '#ff8080', '#ff9999', '#ffb3b3', '#ffcccc', '#ffffff']
    custom_cmap = LinearSegmentedColormap.from_list('smooth_red', colors, N=256)
    
    # Create grid for smooth heat map (half pitch - adjusted for vertical)
    x_grid = np.linspace(0, 80, 134)  # Width of vertical pitch
    y_grid = np.linspace(60, 120, 120)  # Upper half of vertical pitch (attacking half)
    X, Y = np.meshgrid(x_grid, y_grid)
    
    # Transform coordinates for vertical pitch display
    # For VerticalPitch: horizontal x becomes vertical y, horizontal y becomes vertical x
    df_transformed = df.copy()
    df_transformed['start_x_vertical'] = df['start_y']  # horizontal y becomes vertical x
    df_transformed['start_y_vertical'] = df['start_x']  # horizontal x becomes vertical y (NO inversion)
    
    # Create positions array with transformed coordinates
    positions = np.column_stack([df_transformed.start_x_vertical.values, df_transformed.start_y_vertical.values])
    
    if len(positions) > 1:
        # Create KDE with optimal bandwidth for smooth coverage
        kde = gaussian_kde(positions.T)
        kde.set_bandwidth(bw_method=0.35)  # Slightly smaller for half pitch
        
        # Evaluate KDE on grid
        Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
        
        # Normalize Z to a good range for visualization
        Z_normalized = (Z - Z.min()) / (Z.max() - Z.min())
        
        # Apply power transformation for better visual contrast
        Z_enhanced = np.power(Z_normalized, 0.7)
        
        # Create smooth heat map with many levels for smoothness
        levels = np.linspace(0.05, 1.0, 40)  # Start from 0.05 to avoid showing noise
        im = ax.contourf(X, Y, Z_enhanced, levels=levels, cmap=custom_cmap, alpha=0.85, extend='max')
        
        # Add very subtle contour lines for definition
        contour_levels = np.linspace(0.2, 1.0, 6)
        ax.contour(X, Y, Z_enhanced, levels=contour_levels, colors='white', alpha=0.15, linewidths=0.5)
        
    else:
        # Fallback for single point - create a smooth gradient around it (use transformed coordinates)
        x_center, y_center = df_transformed.start_x_vertical.iloc[0], df_transformed.start_y_vertical.iloc[0]
        
        # Create circular gradient around the point
        distance = np.sqrt((X - x_center)**2 + (Y - y_center)**2)
        Z_single = np.exp(-distance**2 / (2 * 6**2))  # Gaussian with sigma=6 for half pitch
        
        im = ax.contourf(X, Y, Z_single, levels=25, cmap=custom_cmap, alpha=0.85, extend='max')

    # Professional colorbar styling
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, aspect=15, pad=0.02)
    cbar.set_label('Shot Density', color='white', fontsize=14, labelpad=20)
    cbar.ax.yaxis.set_tick_params(color='white', labelsize=12)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')
    cbar.outline.set_edgecolor('white')
    cbar.outline.set_linewidth(1)

    # Professional title styling
    ax.set_title(title, fontsize=22, color='white', fontweight='bold', pad=30)
    
    return fig, ax


def create_custom_shots_heat_map(df: pd.DataFrame, title: str = "Custom Shots Heat Map") -> tuple:
    """
    Create a professional smooth heat map visualization for multiple custom shots.
    
    Args:
        df: DataFrame containing custom shots with 'start_x', 'start_y', and 'xg_value' columns
        title: Title for the plot
        
    Returns:
        Tuple of (figure, axis) objects
    """
    if df.empty:
        # Return professional empty plot if no data with proper aspect ratio
        pitch = VerticalPitch(pitch_type='statsbomb', pitch_color='#0d1117', line_color='#30363d', linewidth=2)
        fig, ax = pitch.draw(figsize=(10, 16))  # Proper aspect ratio for vertical pitch
        fig.patch.set_facecolor('#0d1117')
        ax.set_facecolor('#0d1117')
        ax.set_title("No custom shots to display", fontsize=22, color='#8b949e', fontweight='bold')
        ax.text(60, 40, 'Create some custom shots to see the heat map visualization', 
                ha='center', va='center', fontsize=14, color='#8b949e', style='italic')
        return fig, ax
    
    # Create professional pitch styling with proper aspect ratio
    pitch = VerticalPitch(pitch_type='statsbomb', pitch_color='#0d1117', line_color='#30363d', linewidth=2)
    fig, ax = pitch.draw(figsize=(10, 16))  # Proper aspect ratio for vertical pitch
    
    # Set dark background for professional look
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')

    # Create smooth continuous heat map using kde
    from scipy.stats import gaussian_kde
    from matplotlib.colors import LinearSegmentedColormap
    import numpy as np
    
    # Create custom red-to-white colormap for better visibility
    colors = ['#000000', '#1a0000', '#330000', '#4d0000', '#660000', '#800000', 
              '#990000', '#b30000', '#cc0000', '#e60000', '#ff0000', '#ff1a1a', 
              '#ff3333', '#ff4d4d', '#ff6666', '#ff8080', '#ff9999', '#ffb3b3', '#ffcccc', '#ffffff']
    custom_cmap = LinearSegmentedColormap.from_list('smooth_red', colors, N=256)
    
    # Create higher resolution grid for smoother heat map (adjusted for vertical pitch)
    x_grid = np.linspace(0, 80, 160)   # Width of vertical pitch
    y_grid = np.linspace(0, 120, 240)  # Height of vertical pitch
    X, Y = np.meshgrid(x_grid, y_grid)
    
    # Transform coordinates for vertical pitch display
    # For VerticalPitch: horizontal x becomes vertical y, horizontal y becomes vertical x
    df_transformed = df.copy()
    df_transformed['start_x_vertical'] = df['start_y']  # horizontal y becomes vertical x
    df_transformed['start_y_vertical'] = df['start_x']  # horizontal x becomes vertical y (NO inversion)
    
    # Create positions array with transformed coordinates
    positions = np.column_stack([df_transformed.start_x_vertical.values, df_transformed.start_y_vertical.values])
    
    if len(positions) > 1:
        try:
            # Check if we have enough unique points for KDE
            unique_positions = np.unique(positions, axis=0)
            
            if len(unique_positions) > 1:
                # Check for sufficient variance in the data
                x_var = np.var(positions[:, 0])
                y_var = np.var(positions[:, 1])
                
                if x_var > 1e-6 and y_var > 1e-6:  # Sufficient variance threshold
                    # Create KDE with adaptive bandwidth based on data distribution
                    kde = gaussian_kde(positions.T)
                    
                    # Calculate adaptive bandwidth to prevent overly stretched heat maps
                    n_points = len(positions)
                    if n_points > 10:
                        bw_factor = 0.3  # Smaller bandwidth for many points
                    elif n_points > 5:
                        bw_factor = 0.4  # Medium bandwidth
                    else:
                        bw_factor = 0.5  # Larger bandwidth for few points
                    
                    kde.set_bandwidth(bw_method=bw_factor)
                    
                    # Evaluate KDE on grid
                    Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
                    
                    # Apply square root scaling to prevent extreme stretching
                    Z_sqrt = np.sqrt(Z)
                    Z_normalized = (Z_sqrt - Z_sqrt.min()) / (Z_sqrt.max() - Z_sqrt.min())
                    
                    # Apply mild power transformation
                    Z_enhanced = np.power(Z_normalized, 0.6)
                    
                    # Create heat map with circular-friendly levels
                    levels = np.linspace(0.05, 1.0, 45)
                    im = ax.contourf(X, Y, Z_enhanced, levels=levels, cmap=custom_cmap, alpha=0.85, extend='max')
                    
                    # Add subtle contour lines
                    contour_levels = np.linspace(0.15, 0.9, 5)
                    ax.contour(X, Y, Z_enhanced, levels=contour_levels, colors='white', alpha=0.25, linewidths=0.6)
                    
                else:
                    # Fallback: Points too concentrated, use manual kernel method
                    raise np.linalg.LinAlgError("Insufficient variance for KDE")
            else:
                # Fallback: Not enough unique points, use manual kernel method
                raise np.linalg.LinAlgError("Insufficient unique points for KDE")
                
        except (np.linalg.LinAlgError, ValueError):
            # Fallback method: Create manual heat map using individual point kernels
            Z_manual = np.zeros_like(X)
            
            # Adaptive sigma based on number of points
            n_points = len(positions)
            if n_points > 5:
                sigma = 6  # Smaller kernel for many points
            else:
                sigma = 8  # Larger kernel for few points
            
            for i, (x_pos, y_pos) in enumerate(positions):
                # Create circular Gaussian kernel around each point
                distance = np.sqrt((X - x_pos)**2 + (Y - y_pos)**2)
                kernel = np.exp(-distance**2 / (2 * sigma**2))
                Z_manual += kernel
            
            # Apply square root scaling to prevent stretching
            Z_manual_sqrt = np.sqrt(Z_manual)
            if Z_manual_sqrt.max() > 0:
                Z_manual_normalized = Z_manual_sqrt / Z_manual_sqrt.max()
            else:
                Z_manual_normalized = Z_manual_sqrt
            
            # Create circular heat map with appropriate levels
            levels = np.linspace(0.05, 1.0, 35)
            im = ax.contourf(X, Y, Z_manual_normalized, levels=levels, cmap=custom_cmap, alpha=0.85, extend='max')
            
    else:
        # Single point fallback - create a circular gradient (use transformed coordinates)
        x_center, y_center = df_transformed.start_x_vertical.iloc[0], df_transformed.start_y_vertical.iloc[0]
        
        # Create circular gradient around the point
        distance = np.sqrt((X - x_center)**2 + (Y - y_center)**2)
        Z_single = np.exp(-distance**2 / (2 * 8**2))  # Circular sigma=8
        
        # Create circular heat map
        levels = np.linspace(0.05, 1.0, 30)
        im = ax.contourf(X, Y, Z_single, levels=levels, cmap=custom_cmap, alpha=0.85, extend='max')

    # Professional colorbar styling
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, aspect=20, pad=0.02)
    cbar.set_label('Shot Density', color='white', fontsize=14, labelpad=20)
    cbar.ax.yaxis.set_tick_params(color='white', labelsize=12)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')
    cbar.outline.set_edgecolor('white')
    cbar.outline.set_linewidth(1)

    # Professional title styling
    ax.set_title(title, fontsize=22, color='white', fontweight='bold', pad=30)
    
    return fig, ax


def get_visualization_options(lang: str = 'en') -> dict:
    """Return mapping of localized visualization option labels to internal types.

    Pages call this to populate selectboxes. Ensure keys match translation keys
    defined in language.py (shot_map_option, heat_map_option).
    """
    from .language import get_translation  # local import to avoid circular
    return {
        get_translation('shot_map_option', lang): 'shot_map',
        get_translation('heat_map_option', lang): 'heat_map',
    }


def _select_shot_map_function(custom_shots: bool, half_pitch: bool):
    """Internal helper choosing appropriate shot map function.
    Preference order: Seaborn alternative (manual pitch) if available, else mplsoccer.
    """
    if custom_shots:
        # For custom shots we keep the distinct styling function already present.
        return create_custom_shots_visualization if not half_pitch else create_custom_shots_visualization
    # Dataset shots
    if SEABORN_AVAILABLE:
        return create_seaborn_shot_map if not half_pitch else create_seaborn_shot_map
    return create_shot_map if not half_pitch else create_half_pitch_shot_map


def _select_heat_map_function(custom_shots: bool, half_pitch: bool):
    if custom_shots:
        return create_custom_shots_heat_map if not half_pitch else create_custom_shots_heat_map  # half version not separate for custom; reuse full
    return create_shot_heat_map if not half_pitch else create_half_pitch_heat_map


def create_visualization_by_type(
    df: pd.DataFrame,
    viz_type: str,
    title: str,
    half_pitch: bool = False,
    interactive: bool = True,
    custom_shots: bool = False,
):
    """Factory wrapper used by pages to build the requested visualization.

    Returns (figure, axis) for matplotlib based visualizations OR (plotly_fig, None)
    for interactive Plotly versions. Currently alternative Seaborn/manual approach
    returns matplotlib figures for reliability & marker visibility.
    """
    # Defensive copy
    local_df = df.copy() if df is not None else pd.DataFrame()

    # Normalize expected column naming for custom shots (xg_value vs xG)
    if custom_shots and 'xG' not in local_df.columns and 'xg_value' in local_df.columns:
        local_df = local_df.rename(columns={'xg_value': 'xG'})

    # Prefer Plotly for interactive shot maps (hover, bright markers)
    if viz_type == 'shot_map' and interactive:
        fig, _ = create_plotly_shot_map(local_df, title=title, half_pitch=half_pitch, custom_shots=custom_shots)
        return fig, None

    if viz_type == 'shot_map':
        func = _select_shot_map_function(custom_shots, half_pitch)
        # Seaborn alternative uses same signature (df, title, half_pitch=bool)
        if func.__name__ == 'create_seaborn_shot_map':
            return func(local_df, title=title, half_pitch=half_pitch)
        # Custom shots visualization already adapts
        if func.__name__ == 'create_custom_shots_visualization':
            return func(local_df, title=title)
        # Default mplsoccer versions
        return func(local_df, title=title)
    elif viz_type == 'heat_map':
        func = _select_heat_map_function(custom_shots, half_pitch)
        return func(local_df, title=title)
    else:
        func = _select_shot_map_function(custom_shots, half_pitch)
        if func.__name__ == 'create_seaborn_shot_map':
            return func(local_df, title=title, half_pitch=half_pitch)
        if func.__name__ == 'create_custom_shots_visualization':
            return func(local_df, title=title)
        return func(local_df, title=title)


# === NEW: Plotly interactive shot map for hover support ===

def create_plotly_shot_map(df: pd.DataFrame, title: str = "Shot Map", half_pitch: bool = False, custom_shots: bool = False):
    """Create vertical pitch shot map using Plotly with hover tooltips.
    Improvements: debug output, robust filtering, fill NaNs, guaranteed marker rendering.
    Returns (fig, None) so caller treats it as interactive figure.
    """
    # Defensive copy
    if df is None or df.empty:
        fig = go.Figure()
        fig.update_layout(title=title + " (no data)", plot_bgcolor='black', paper_bgcolor='black')
        return fig, None

    # Ensure numeric columns
    work = df.copy()
    # Normalize xG column
    if 'xG' not in work.columns and 'xg_value' in work.columns:
        work = work.rename(columns={'xg_value': 'xG'})
    if 'xG' not in work.columns:
        # Create placeholder xG so we can still plot locations
        work['xG'] = 0.05

    # Fill NaNs
    work['xG'] = pd.to_numeric(work['xG'], errors='coerce').fillna(0.05)
    work['start_x'] = pd.to_numeric(work['start_x'], errors='coerce')
    work['start_y'] = pd.to_numeric(work['start_y'], errors='coerce')

    # Filter valid coordinates
    if half_pitch:
        valid = work[(work['start_x'] >= 60) & (work['start_x'] <= 120) & (work['start_y'] >= 0) & (work['start_y'] <= 80)].copy()
    else:
        valid = work[(work['start_x'].between(0, 120)) & (work['start_y'].between(0, 80))].copy()

    print(f"DEBUG Plotly Shot Map - total rows: {len(work)}, valid rows: {len(valid)}, half_pitch={half_pitch}")

    if valid.empty:
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.update_layout(title=title + " (no valid shots)", plot_bgcolor='black', paper_bgcolor='black',
                          xaxis=dict(visible=False), yaxis=dict(visible=False))
        return fig, None

    # Ensure xG column name
    if 'xG' not in valid.columns and 'xg_value' in valid.columns:
        valid = valid.rename(columns={'xg_value': 'xG'})

    # Mapping for colors
    def color_for_xg(xg):
        if xg >= 0.7: return '#FF0040'
        if xg >= 0.5: return '#FF8000'
        if xg >= 0.3: return '#FFFF00'
        if xg >= 0.1: return '#00FFFF'
        return '#4080FF'

    valid['color'] = valid['xG'].apply(color_for_xg)
    valid['size'] = (valid['xG'] * 35) + 14  # Reduced, moderate sizes

    # Transform to vertical pitch
    valid['display_x'] = valid['start_y']
    valid['display_y'] = valid['start_x']

    # Choose label column for hover: player_name -> shot_name -> generic
    label_col = None
    for cand in ['player_name', 'shot_name', 'player', 'shooter', 'name']:
        if cand in valid.columns:
            label_col = cand
            break
    if label_col is None:
        valid[label_col := 'temp_label'] = 'Player'

    import plotly.graph_objects as go
    fig = go.Figure()

    # Pitch base rectangle (vertical orientation) with shapes layered below traces
    if half_pitch:
        fig.add_shape(type="rect", x0=0, y0=60, x1=80, y1=120, line=dict(color="white", width=3), fillcolor="rgba(0,0,0,0)", layer='below')
        # Penalty area & 6-yard (top end)
        fig.add_shape(type="rect", x0=22, y0=102, x1=58, y1=120, line=dict(color="white", width=2), fillcolor="rgba(0,0,0,0)", layer='below')
        fig.add_shape(type="rect", x0=30, y0=114, x1=50, y1=120, line=dict(color="white", width=2), fillcolor="rgba(0,0,0,0)", layer='below')
        # Goal
        fig.add_shape(type="rect", x0=36, y0=120, x1=44, y1=122, line=dict(color="white", width=2), fillcolor='white', layer='below')
        y_range = [60, 122]
    else:
        fig.add_shape(type="rect", x0=0, y0=0, x1=80, y1=120, line=dict(color="white", width=3), fillcolor="rgba(0,0,0,0)", layer='below')
        # Center line & circle
        fig.add_shape(type="line", x0=0, y0=60, x1=80, y1=60, line=dict(color="white", width=2), layer='below')
        fig.add_shape(type="circle", x0=30, y0=50, x1=50, y1=70, line=dict(color="white", width=2), layer='below')
        # Penalty areas (top & bottom)
        for y0 in (0, 102):
            fig.add_shape(type="rect", x0=22, y0=y0, x1=58, y1=y0+18, line=dict(color="white", width=2), fillcolor="rgba(0,0,0,0)", layer='below')
        for y0 in (0, 114):
            fig.add_shape(type="rect", x0=30, y0=y0, x1=50, y1=y0+6, line=dict(color="white", width=2), fillcolor="rgba(0,0,0,0)", layer='below')
        # Goals
        fig.add_shape(type="rect", x0=36, y0=-2, x1=44, y1=0, line=dict(color="white", width=2), fillcolor='white', layer='below')
        fig.add_shape(type="rect", x0=36, y0=120, x1=44, y1=122, line=dict(color="white", width=2), fillcolor='white', layer='below')
        y_range = [-2, 122]

    customdata = np.stack([
        valid[label_col].fillna('Unknown'),
        valid['xG'],
        valid['start_x'],
        valid['start_y']
    ], axis=1)

    # Build legend by creating separate traces per xG bin
    bins = [
        ("Very High xG (≥0.7)", valid['xG'] >= 0.7, '#FF0040'),
        ("High xG (0.5–0.7)", (valid['xG'] >= 0.5) & (valid['xG'] < 0.7), '#FF8000'),
        ("Medium xG (0.3–0.5)", (valid['xG'] >= 0.3) & (valid['xG'] < 0.5), '#FFFF00'),
        ("Low xG (0.1–0.3)", (valid['xG'] >= 0.1) & (valid['xG'] < 0.3), '#00FFFF'),
        ("Very Low xG (<0.1)", valid['xG'] < 0.1, '#4080FF'),
    ]

    hovertemplate = (
        "<b>%{customdata[0]}</b><br>"
        "xG: %{customdata[1]:.3f}<br>"
        "x: %{customdata[2]:.1f}, y: %{customdata[3]:.1f}<extra></extra>"
    )

    for label, mask, color in bins:
        sub = valid[mask]
        if sub.empty:
            continue
        sub_custom = np.stack([
            sub[label_col].fillna('Unknown'),
            sub['xG'],
            sub['start_x'],
            sub['start_y']
        ], axis=1)
        # Smaller markers for full pitch, keep larger for half-pitch
        size_series = (sub['xG'] * (35 if half_pitch else 22)) + (14 if half_pitch else 9)
        fig.add_trace(go.Scatter(
            x=sub['display_x'],
            y=sub['display_y'],
            mode='markers',
            marker=dict(size=size_series.tolist(), color=color, line=dict(width=1.5, color='white')),
            customdata=sub_custom,
            hovertemplate=hovertemplate,
            name=label
        ))

    fig.update_layout(
        title=title,
        plot_bgcolor='black',
        paper_bgcolor='black',
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='bottom', y=1.02,
            xanchor='left', x=0,
            font=dict(color='white'),
            bgcolor='rgba(0,0,0,0.4)',
            bordercolor='white', borderwidth=1
        ),
        margin=dict(l=10, r=10, t=90, b=10),
        xaxis=dict(range=[-5, 85], showgrid=False, zeroline=False, visible=False),
        yaxis=dict(range=y_range, showgrid=False, zeroline=False, visible=False, scaleanchor='x', scaleratio=1),
        font=dict(color='white')
    )

    return fig, None

# End of file
