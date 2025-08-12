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


def create_shot_map(df: pd.DataFrame, title: str = "Shot Map with xG") -> tuple:
    """
    Create a professional vertical shot map visualization showing shots colored by xG value.
    
    Args:
        df: DataFrame containing shot data with 'start_x', 'start_y', and 'xG' columns
        title: Title for the plot
        
    Returns:
        Tuple of (figure, axis) objects
    """
    # Create the vertical pitch with professional styling
    pitch = VerticalPitch(pitch_type='statsbomb', pitch_color='#0d1117', line_color='#30363d', 
                          linewidth=2)
    fig, ax = pitch.draw(figsize=(10, 16), constrained_layout=True, tight_layout=False)
    
    # Set dark background for professional look
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')

    # Plot the shots with professional styling - VerticalPitch handles coordinate transformation
    # Normalize xG for size to make visualization clearer
    size = df['xG'] * 800 + 100  # Ensure good visibility
    sc = pitch.scatter(df.start_x, df.start_y,
                       s=size,
                       c=df['xG'],
                       cmap='plasma',  # Professional colormap
                       ax=ax,
                       edgecolors='white',
                       linewidths=1.5,
                       alpha=0.85,
                       vmin=0, vmax=1)

    # Professional colorbar
    cbar = fig.colorbar(sc, ax=ax, shrink=0.8, aspect=20, pad=0.02)
    cbar.set_label('Expected Goals (xG) Value', color='white', fontsize=14, labelpad=20)
    cbar.ax.yaxis.set_tick_params(color='white', labelsize=12)
    cbar.ax.yaxis.set_ticklabels([f'{x:.2f}' for x in cbar.get_ticks()], color='white')
    cbar.outline.set_edgecolor('white')
    cbar.outline.set_linewidth(1)

    # Professional title styling
    ax.set_title(title, fontsize=22, color='white', fontweight='bold', pad=30)
    
    return fig, ax


def create_half_pitch_shot_map(df: pd.DataFrame, title: str = "Half Pitch Shot Map") -> tuple:
    """
    Create a professional vertical half-pitch shot map visualization with goal at the top.
    
    Args:
        df: DataFrame containing shot data with 'start_x', 'start_y', and 'xG' columns
        title: Title for the plot
        
    Returns:
        Tuple of (figure, axis) objects
    """
    # Create vertical half pitch with professional styling
    pitch = VerticalPitch(pitch_type='statsbomb', pitch_color='#0d1117', line_color='#30363d', 
                          linewidth=2, half=True)
    fig, ax = pitch.draw(figsize=(12, 12), constrained_layout=True, tight_layout=False)
    
    # Set dark background for professional look
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')

    # Plot the shots with professional styling - VerticalPitch handles coordinate transformation
    # Normalize xG for size to make visualization clearer
    size = df['xG'] * 800 + 100  # Ensure good visibility
    sc = pitch.scatter(df.start_x, df.start_y,
                       s=size,
                       c=df['xG'],
                       cmap='viridis',  # Professional colormap
                       ax=ax,
                       edgecolors='white',
                       linewidths=1.5,
                       alpha=0.85,
                       vmin=0, vmax=1)

    # Professional colorbar styling
    cbar = fig.colorbar(sc, ax=ax, shrink=0.8, aspect=15, pad=0.02)
    cbar.set_label('Expected Goals (xG) Value', color='white', fontsize=14, labelpad=20)
    cbar.ax.yaxis.set_tick_params(color='white', labelsize=12)
    cbar.ax.yaxis.set_ticklabels([f'{x:.2f}' for x in cbar.get_ticks()], color='white')
    cbar.outline.set_edgecolor('white')
    cbar.outline.set_linewidth(1)

    # Professional title styling
    ax.set_title(title, fontsize=22, color='white', fontweight='bold', pad=30)
    
    # Add subtitle with data info
    total_xg = df['xG'].sum()
    ax.text(90, -5, f'Final Third Focus | {len(df)} shots | Total xG: {total_xg:.2f}', 
            ha='center', va='center', fontsize=12, color='#8b949e', style='italic')
    
    return fig, ax


def create_single_shot_visualization(x: float, y: float, xg_value: float, 
                                   title: str = "Shot Location Preview") -> tuple:
    """
    Create a professional vertical visualization for a single shot location.
    
    Args:
        x: X coordinate of the shot
        y: Y coordinate of the shot
        xg_value: xG value of the shot (optional, for annotation)
        title: Title for the plot
        
    Returns:
        Tuple of (figure, axis) objects
    """
    # Create professional vertical pitch styling
    pitch = VerticalPitch(pitch_type='statsbomb', pitch_color='#0d1117', line_color='#30363d', 
                          linewidth=2)
    fig, ax = pitch.draw(figsize=(9, 14), constrained_layout=True, tight_layout=False)
    
    # Set dark background for professional look
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')
    
    size = xg_value * 1200 + 200  # Ensure even low xG is visible, larger for single shot
    
    # Plot using original coordinates - VerticalPitch handles coordinate transformation
    sc = pitch.scatter(x, y,
                       s=size,
                       c=xg_value,
                       cmap='plasma', 
                       vmin=0, vmax=1,
                       ax=ax,
                       edgecolors='white',
                       linewidths=3,
                       alpha=0.9)
    
    # Add professional annotation with xG value using original coordinates
    pitch.annotate(f"xG: {xg_value:.3f}", 
                   xy=(x, y), 
                   ax=ax, xytext=(10, -20), textcoords='offset points', 
                   fontsize=14, color='white', ha='center', va='center', fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.5", fc="black", ec="white", alpha=0.8, linewidth=2))

    # Professional title styling
    ax.set_title(title, fontsize=20, color='white', fontweight='bold', pad=25)
    
    # Add coordinate info
    ax.text(x, -8, f'Position: ({x}, {y})', 
            ha='center', va='center', fontsize=12, color='#8b949e', style='italic')
    
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
                       cmap='magma',  # Professional colormap for custom shots
                       ax=ax,
                       edgecolors='white',
                       linewidths=2,
                       alpha=0.85,
                       vmin=0, vmax=1)

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
                bbox=dict(boxstyle="round,pad=0.4", fc="black", ec="white", alpha=0.8, linewidth=1)
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


def get_visualization_options(lang="en"):
    """
    Get visualization type options for selectbox.
    
    Args:
        lang: Language code ('en' or 'id')
        
    Returns:
        Dictionary with display names as keys and function names as values
    """
    if lang == "id":
        return {
            "Shot Map": "shot_map",
            "Heat Map": "heat_map"
        }
    else:
        return {
            "Shot Map": "shot_map", 
            "Heat Map": "heat_map"
        }


def create_visualization_by_type(df: pd.DataFrame, viz_type: str, title: str, 
                                half_pitch: bool = False, custom_shots: bool = False) -> tuple:
    """
    Create visualization based on selected type.
    
    Args:
        df: DataFrame containing shot data
        viz_type: Type of visualization ('shot_map' or 'heat_map')
        title: Title for the plot
        half_pitch: Whether to use half pitch
        custom_shots: Whether this is for custom shots
        
    Returns:
        Tuple of (figure, axis) objects
    """
    if custom_shots:
        if viz_type == "heat_map":
            return create_custom_shots_heat_map(df, title)
        else:
            return create_custom_shots_visualization(df, title)
    else:
        if half_pitch:
            if viz_type == "heat_map":
                return create_half_pitch_heat_map(df, title)
            else:
                return create_half_pitch_shot_map(df, title)
        else:
            if viz_type == "heat_map":
                return create_shot_heat_map(df, title)
            else:
                return create_shot_map(df, title)
