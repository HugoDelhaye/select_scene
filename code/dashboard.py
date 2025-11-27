"""
Interactive dashboard for exploring scene selection metrics.

This module provides functions to create an interactive Plotly dashboard
for visualizing learning metrics across different subjects and scenes.
"""

import os
import glob
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.colors import sample_colorscale

import ipywidgets as widgets
from IPython.display import display, clear_output


# ========================================
# DATA CONFIGURATION
# ========================================

def get_phase_configurations():
    """
    Get phase configurations for all subject types.

    Returns:
        dict: Dictionary containing phase orders, colors, and mappings
    """
    # Human subjects configuration
    order_hum = ['sub-01', 'sub-02', 'sub-03', 'sub-05', 'sub-06']
    order_2phases = ["Early discovery", "Late discovery", "Early practice", "Late practice"]
    colors_2phases = dict(zip(order_2phases, sample_colorscale("Viridis", [0.00, 0.33, 0.66, 1.00])))
    position_2phase = dict(discovery="bottom left", practice="bottom right")
    hum_2phases = dict(zip(order_hum, [order_2phases] * len(order_hum)))

    # PPO and imitation model configuration
    order_ppos = ['im_sub-01', 'im_sub-02', 'im_sub-03', 'im_sub-05', 'im_sub-06', 'ppo']
    order_4phases = ["Early discovery", "Late discovery", "Early practice", "Late practice"]

    ppos_phases = [
        ['sub-01_epoch=0-step=500', 'sub-01_epoch=0-step=2000', 'sub-01_epoch=0-step=3500',
         'sub-01_epoch=0-step=5000', 'sub-01_epoch=0-step=6500'],
        ['sub-02_epoch=0-step=500', 'sub-02_epoch=0-step=3000', 'sub-02_epoch=0-step=5500',
         'sub-02_epoch=0-step=8000', 'sub-02_epoch=0-step=10000'],
        ['sub-03_epoch=0-step=500', 'sub-03_epoch=0-step=4000', 'sub-03_epoch=0-step=7500',
         'sub-03_epoch=1-step=11408', 'sub-03_epoch=1-step=14908'],
        ['sub-05_epoch=0-step=500', 'sub-05_epoch=0-step=1500', 'sub-05_epoch=0-step=3000',
         'sub-05_epoch=0-step=4000', 'sub-05_epoch=0-step=5000'],
        ['sub-06_epoch=0-step=500', 'sub-06_epoch=0-step=2000', 'sub-06_epoch=0-step=4000',
         'sub-06_epoch=0-step=5500', 'sub-06_epoch=0-step=7000'],
        ['ep-20', 'ep-2000', 'ep-4000', 'ep-6000', 'ep-8000']
    ]

    phases_everyone = dict(zip(order_hum + order_ppos, [order_4phases] * len(order_hum) + ppos_phases))
    order_all_phases = order_4phases + [phase for subset in ppos_phases for phase in subset]

    color_4phases = sample_colorscale("Viridis", [0.00, 0.33, 0.66, 1.00])
    color_ckpt = sample_colorscale("magma", [0.2, 0.4, 0.6, 0.8, 1.0])
    colors_phase_ckpt = dict(zip(order_all_phases, color_4phases + color_ckpt * 6))

    return {
        'order_hum': order_hum,
        'order_2phases': order_2phases,
        'colors_2phases': colors_2phases,
        'position_2phase': position_2phase,
        'hum_2phases': hum_2phases,
        'order_ppos': order_ppos,
        'order_4phases': order_4phases,
        'ppos_phases': ppos_phases,
        'phases_everyone': phases_everyone,
        'order_all_phases': order_all_phases,
        'color_4phases': color_4phases,
        'color_ckpt': color_ckpt,
        'colors_phase_ckpt': colors_phase_ckpt
    }


def get_phase_to_index_mappings(config):
    """
    Create mappings from phase names to their index positions.

    Args:
        config: Configuration dictionary from get_phase_configurations()

    Returns:
        dict: Dictionary with phase-to-index mappings
    """
    phase_to_human = {p: i % 4 for i, p in enumerate(config['order_4phases'])}
    phase_to_checkpoint = {
        p: i % 5 for i, p in enumerate([phase for subset in config['ppos_phases'] for phase in subset])
    }
    phase_to_all = {**phase_to_human, **phase_to_checkpoint}

    return {
        'phase_to_human': phase_to_human,
        'phase_to_checkpoint': phase_to_checkpoint,
        'phase_to_all': phase_to_all
    }


# ========================================
# DATA FILTERING HELPERS
# ========================================

def scenes_for_level(df, level):
    """
    Get all available scenes for a given level.

    Args:
        df: DataFrame with scene data
        level: Level identifier (e.g., 'w1l1')

    Returns:
        list: Sorted list of scene numbers
    """
    return sorted(df.loc[df["level_full_name"] == level, "scene"].dropna().unique().tolist())


def subjects_for(df, level, scene):
    """
    Get all subjects who played a specific level-scene combination.

    Args:
        df: DataFrame with scene data
        level: Level identifier
        scene: Scene number

    Returns:
        list: Sorted list of subject identifiers
    """
    m = (df["level_full_name"] == level) & (df["scene"] == scene)
    return sorted(df.loc[m, "subject"].dropna().unique().tolist())


def get_subset(df, level, scene):
    """
    Extract subset of data for a specific level and scene.

    Args:
        df: DataFrame with scene data
        level: Level identifier (e.g., 'w1l1')
        scene: Scene number

    Returns:
        pd.DataFrame: Filtered dataframe for the scene
    """
    wls = level[1] + '-' + level[3] + '-' + str(scene)
    m = (df["scene_full_name"] == wls)
    return df.loc[m].copy()


# ========================================
# IMAGE LOADING FUNCTIONS
# ========================================

def load_scene_image(level, scene, root_dir=None):
    """
    Load background image for a scene.

    Args:
        level: Level identifier (e.g., 'w1l1')
        scene: Scene number
        root_dir: Root directory path. If None, uses script's parent directory

    Returns:
        tuple: (numpy array of image, path to image file) or (None, path) if not found
    """
    if root_dir is None:
        # Get project root directory (parent of code/ directory)
        root_dir = Path(__file__).parent.parent

    scene_fullname = f"{level}s{scene}"
    path = os.path.join(root_dir, 'sourcedata', 'mario_backgrounds', 'scene_backgrounds',
                        f"{scene_fullname}.png")

    if os.path.exists(path):
        im = Image.open(path).convert("RGB")
        return np.array(im), path

    return None, path


def get_traces(level, scene, root_dir=None):
    """
    Load trajectory trace images for all subjects in a scene.

    Args:
        level: Level identifier (e.g., 'w1l1')
        scene: Scene number
        root_dir: Root directory path. If None, uses script's parent directory

    Returns:
        tuple: (list of image arrays, list of image paths)
    """
    if root_dir is None:
        # Get project root directory (parent of code/ directory)
        root_dir = Path(__file__).parent.parent

    paths_traces = glob.glob(
        os.path.join(str(root_dir), 'sourcedata', 'traces', 'sub-*', 'scenes',
                     f'*scene-{level}s{scene}_traces.png')
    )
    paths_traces = sorted(paths_traces)

    imgs_list = []
    for path in paths_traces:
        img = Image.open(path).convert("RGB")
        imgs_list.append(np.array(img))

    return imgs_list, paths_traces


def concat_traces(img_scene, imgs_traces):
    """
    Concatenate scene background with trajectory trace images.

    Args:
        img_scene: Scene background image as numpy array
        imgs_traces: List of trace images as numpy arrays

    Returns:
        numpy array: Combined image
    """
    img_tot = img_scene

    for img in imgs_traces:
        # Find white margin boundaries
        idx_marge = np.where(img.mean(axis=2) == 255)
        slide_img = (
            slice(idx_marge[0].min(), idx_marge[0].max() + 1),
            slice(idx_marge[1].max() + 1, img.shape[1])
        )
        img = img[slide_img]

        # Add small separator and concatenate
        img_tot = np.concatenate([img_tot, np.zeros([img.shape[0], 5, 3]), img], axis=1)

    return img_tot


# ========================================
# PLOT CREATION FUNCTIONS
# ========================================

def add_scene_background(fig, level, scene):
    """
    Add scene background image to dashboard.

    Args:
        fig: Plotly figure object
        level: Level identifier
        scene: Scene number

    Returns:
        Plotly figure object
    """
    img_arr, img_path = load_scene_image(level, scene)

    if img_arr is not None:
        fig.add_trace(go.Image(z=img_arr, hoverinfo="skip"), row=1, col=1)
        fig.update_xaxes(visible=False, fixedrange=True, row=1, col=1)
        fig.update_yaxes(visible=False, fixedrange=True, row=1, col=1)
    else:
        # No image found - show message
        fig.add_annotation(
            row=1, col=1, x=0.5, y=0.5, xref="x domain", yref="y domain",
            text=f"No image found:<br>{img_path}", showarrow=False,
            font=dict(size=14, color="gray")
        )
        fig.update_xaxes(visible=False, fixedrange=True, row=1, col=1)
        fig.update_yaxes(visible=False, fixedrange=True, row=1, col=1)

    return fig


def add_traces_overview(fig, level, scene, start_col=1):
    """
    Add individual subject trace images to dashboard in separate subplots.

    Args:
        fig: Plotly figure object
        level: Level identifier
        scene: Scene number
        start_col: Starting column for traces (default: 1)

    Returns:
        Plotly figure object
    """
    traces_arr, traces_path = get_traces(level, int(scene))

    if not traces_arr:
        # No traces found
        fig.add_annotation(
            row=2, col=start_col, x=0.5, y=0.5, xref="x domain", yref="y domain",
            text="No trace images found", showarrow=False,
            font=dict(size=14, color="gray")
        )
        fig.update_xaxes(visible=False, fixedrange=True, row=2, col=start_col)
        fig.update_yaxes(visible=False, fixedrange=True, row=2, col=start_col)
        return fig

    # Extract subject names from paths
    subject_names = []
    for path in traces_path:
        # Extract subject name from path like '.../sub-01/scenes/...'
        parts = path.split(os.sep)
        for part in parts:
            if part.startswith('sub-'):
                subject_names.append(part)
                break

    # Add each trace image to its own subplot
    for i, (trace_img, subject) in enumerate(zip(traces_arr, subject_names)):
        col_idx = start_col + i

        # Get image dimensions
        img_height, img_width = trace_img.shape[:2]

        # Add the trace image
        fig.add_trace(
            go.Image(z=trace_img, hoverinfo="skip"),
            row=2,
            col=col_idx
        )

        # Configure axes to preserve aspect ratio and pixel-perfect rendering
        # Set axis ranges to match image pixel dimensions
        fig.update_xaxes(
            visible=True,
            fixedrange=True,
            showticklabels=False,
            range=[0, img_width],
            scaleanchor=f"y{i+2}",
            scaleratio=1,
            row=2,
            col=col_idx
        )
        fig.update_yaxes(
            visible=False,
            fixedrange=True,
            range=[img_height, 0],  # Reversed for image display
            row=2,
            col=col_idx
        )

        # Add subject name as annotation below the image
        # Calculate correct xaxis/yaxis number based on subplot position
        # Row 2, columns 1-5 correspond to axes 2-6
        axis_num = i + 2
        fig.add_annotation(
            x=0.5, y=-0.08,
            xref=f"x{axis_num} domain",
            yref=f"y{axis_num} domain",
            text=subject,
            showarrow=False,
            font=dict(size=10),
            xanchor='center'
        )

    return fig


def add_mad_variance_plot(fig, df_wls, config, mappings):
    """
    Add MAD variance bar plot to dashboard.

    Args:
        fig: Plotly figure object
        df_wls: Filtered dataframe for current scene
        config: Phase configuration dictionary
        mappings: Phase-to-index mappings

    Returns:
        Plotly figure object
    """
    phase_to_all = mappings['phase_to_all']
    phases_everyone = config['phases_everyone']
    colors_phase_ckpt = config['colors_phase_ckpt']

    for sub in phases_everyone.keys():
        for idx, i in enumerate(phases_everyone[sub]):
            mask_mad = (df_wls['subject'] == sub) & (df_wls['learning_phase'] == i)
            df_phase = df_wls[mask_mad]

            if df_phase.empty:
                df_phase = pd.DataFrame({
                    "subject": [sub],
                    "learning_phase": [i],
                    "MAD_mean": [0.0]
                })

            fig.add_trace(
                go.Bar(
                    x=df_phase["subject"],
                    y=df_phase["MAD_mean"],
                    name=i,
                    marker_color=colors_phase_ckpt[i],
                    offsetgroup=f"chk{phase_to_all[i]}",
                    text=df_phase["MAD_mean"].round(2).astype(str),
                    textposition="inside",
                    cliponaxis=False,
                    hovertemplate="Value: %{y:.2f}<extra></extra>",
                    showlegend=False
                ),
                row=3, col=1
            )

    fig.update_xaxes(visible=False, fixedrange=True, row=3, col=1)
    fig.update_yaxes(title_text="Mean MAD", row=3, col=1,
                     range=[-0.1, df_wls["MAD_mean"].max() * 1.4])

    # Add delta labels
    for sub in df_wls['subject'].sort_values().unique():
        df_sub = df_wls[df_wls['subject'] == sub]
        fig.add_trace(
            go.Scatter(
                x=df_sub['subject'],
                y=[df_wls["MAD_mean"].max()],
                mode='text',
                text=(df_sub['delta_MAD_tot']).round(2).astype(str) + " d",
                textposition="top center",
                textfont=dict(size=10, color='rgb(68, 1, 84)', family="Arial Black"),
                name=f"Delta Total MAD for {sub}",
                showlegend=False
            ),
            row=3, col=1
        )

    return fig


def add_clip_count_plot(fig, df_wls, config, col=3):
    """
    Add clip count bar plot for human subjects.

    Args:
        fig: Plotly figure object
        df_wls: Filtered dataframe for current scene
        config: Phase configuration dictionary

    Returns:
        Plotly figure object
    """
    hum_2phases = config['hum_2phases']
    colors_2phases = config['colors_2phases']

    show_legend = False
    legende_is_first_time = True

    for sub in hum_2phases.keys():
        sub_phases = df_wls[df_wls["subject"] == sub]["learning_phase"].unique()

        for phase in hum_2phases[sub]:
            mask = (df_wls['subject'] == sub) & (df_wls['learning_phase'] == phase)
            df_phase = df_wls[mask]

            if df_phase.empty:
                df_phase = pd.DataFrame({
                    "subject": [sub],
                    "phase": [phase],
                    "count": [0],
                    "cleared": [0.0]
                })

            if set(hum_2phases).issubset(set(sub_phases)):
                if legende_is_first_time:
                    show_legend = True
                    legende_is_first_time = False

            fig.add_trace(
                go.Bar(
                    x=df_phase["subject"],
                    y=df_phase["count"],
                    name=phase,
                    marker_color=colors_2phases[phase],
                    textfont=dict(size=13, color='rgb(68, 1, 84)', family="Arial Black"),
                    text=df_phase["count"],
                    textposition="outside",
                    offsetgroup=phase,
                    cliponaxis=False,
                    hovertemplate="Phase: %{name}<br>Value: %{y}<extra></extra>",
                    showlegend=show_legend
                ),
                row=3, col=col
            )
        show_legend = False

    fig.update_layout(barmode="group")
    max_count = df_wls[df_wls['subject'] != 'ppo']["count"].max()
    fig.update_yaxes(title_text="N tries", row=3, col=col, range=[-3, max_count * 1.2])

    return fig


def add_clearance_plot(fig, df_wls, config, mappings, col=1):
    """
    Add clearance rate bar plot to dashboard.

    Args:
        fig: Plotly figure object
        df_wls: Filtered dataframe for current scene
        config: Phase configuration dictionary
        mappings: Phase-to-index mappings

    Returns:
        Plotly figure object
    """
    phase_to_all = mappings['phase_to_all']
    phases_everyone = config['phases_everyone']
    colors_phase_ckpt = config['colors_phase_ckpt']
    order_4phases = config['order_4phases']
    color_4phases = config['color_4phases']
    color_ckpt = config['color_ckpt']

    for sub in phases_everyone.keys():
        for idx, i in enumerate(phases_everyone[sub]):
            mask_clr = (df_wls['subject'] == sub) & (df_wls['learning_phase'] == i)
            df_phase = df_wls[mask_clr]

            if df_phase.empty:
                df_phase = pd.DataFrame({
                    "subject": [sub],
                    "learning_phase": [i],
                    "cleared": [0.0]
                })

            chk = phase_to_all[i]
            fig.add_trace(
                go.Bar(
                    x=df_phase["subject"],
                    y=(df_phase["cleared"] * 100).round(0),
                    name=i,
                    marker_color=colors_phase_ckpt[i],
                    offsetgroup=f"chk{chk}",
                    text=(df_phase["cleared"] * 100).round(0).astype(str) + "%",
                    textposition="inside",
                    cliponaxis=False,
                    hovertemplate="Value: %{y:.2f}<extra></extra>",
                    showlegend=False
                ),
                row=4, col=col
            )

    fig.update_yaxes(title_text="Clearance Rate (%)", row=4, col=col, range=[-10, 120])
    fig.update_xaxes(title_text="Learning Phase", row=4, col=col)

    # Add simple legend
    simple_names = order_4phases + ["Checkpoint 1", "Checkpoint 2", "Checkpoint 3",
                                    "Checkpoint 4", "Checkpoint 5"]
    for label, color in zip(simple_names, color_4phases + color_ckpt):
        fig.add_trace(
            go.Scatter(
                x=[None], y=[None],
                mode="markers",
                marker=dict(size=12, color=color),
                name=label,
                showlegend=True,
            ),
            row=4, col=col
        )

    # Add delta labels
    for sub in df_wls['subject'].sort_values().unique():
        df_sub = df_wls[df_wls['subject'] == sub]
        fig.add_trace(
            go.Scatter(
                x=df_sub['subject'],
                y=[100],
                mode='text',
                text=(df_sub['delta_clr_tot'] * 100).round(2).astype(str) + " d",
                textposition="top center",
                textfont=dict(size=10, color='rgb(68, 1, 84)', family="Arial Black"),
                name=f"Delta Total Clearance for {sub}",
                showlegend=False
            ),
            row=4, col=col
        )

    return fig


def add_speed_plot(fig, df_wls, config, mappings, col=3):
    """
    Add average speed bar plot to dashboard.

    Args:
        fig: Plotly figure object
        df_wls: Filtered dataframe for current scene
        config: Phase configuration dictionary
        mappings: Phase-to-index mappings

    Returns:
        Plotly figure object
    """
    phase_to_all = mappings['phase_to_all']
    phases_everyone = config['phases_everyone']
    colors_phase_ckpt = config['colors_phase_ckpt']
    order_4phases = config['order_4phases']
    color_4phases = config['color_4phases']
    color_ckpt = config['color_ckpt']

    for sub in phases_everyone.keys():
        for idx, i in enumerate(phases_everyone[sub]):
            mask_spd = (df_wls['subject'] == sub) & (df_wls['learning_phase'] == i)
            df_phase = df_wls[mask_spd]

            if df_phase.empty:
                df_phase = pd.DataFrame({
                    "subject": [sub],
                    "learning_phase": [i],
                    "speed": [0.0]
                })

            chk = phase_to_all[i]
            fig.add_trace(
                go.Bar(
                    x=df_phase["subject"],
                    y=df_phase["speed"],
                    name=i,
                    marker_color=colors_phase_ckpt[i],
                    offsetgroup=f"chk{chk}",
                    text=df_phase["speed"].round(0).astype(str),
                    textposition="inside",
                    cliponaxis=False,
                    hovertemplate="Value: %{y:.2f}<extra></extra>",
                    showlegend=False
                ),
                row=4, col=col
            )

    max_y = max(df_wls["speed"].max(), abs(df_wls["speed"].min()))
    fig.update_yaxes(
        title_text="Average Speed<br>(horizontal pixel<br>traveled per sec.)",
        row=4, col=col,
        range=[-0.1, max_y * 1.4]
    )

    # Add simple legend (hidden)
    simple_names = order_4phases + ["Checkpoint 1", "Checkpoint 2", "Checkpoint 3",
                                    "Checkpoint 4", "Checkpoint 5"]
    for label, color in zip(simple_names, color_4phases + color_ckpt):
        fig.add_trace(
            go.Scatter(
                x=[None], y=[None],
                mode="markers",
                marker=dict(size=12, color=color),
                name=label,
                showlegend=False,
            ),
            row=4, col=col
        )

    # Add delta labels
    for sub in df_wls['subject'].sort_values().unique():
        df_sub = df_wls[df_wls['subject'] == sub]
        fig.add_trace(
            go.Scatter(
                x=df_sub['subject'],
                y=[df_wls["speed"].max()],
                mode='text',
                text=(df_sub['delta_spd_tot']).round(2).astype(str) + " d",
                textposition="top center",
                textfont=dict(size=10, color='rgb(68, 1, 84)', family="Arial Black"),
                name=f"Delta Total Speed for {sub}",
                showlegend=False
            ),
            row=4, col=col
        )

    return fig


# ========================================
# MAIN DASHBOARD FUNCTION
# ========================================

def make_dashboard(df, level, scene):
    """
    Create complete interactive dashboard for a scene.

    Args:
        df: Full metrics dataframe
        level: Level identifier (e.g., 'w1l1')
        scene: Scene number

    Returns:
        Plotly figure object
    """
    # Get data and configurations
    df_wls = get_subset(df, level, scene)
    config = get_phase_configurations()
    mappings = get_phase_to_index_mappings(config)

    # Create subplot structure with 4 rows
    # Row 1: Scene background (full width)
    # Row 2: 5 individual trace subplots (larger for better resolution)
    # Row 3-4: Metrics (left plot spans 2 cols, right plot spans 3 cols)
    fig = make_subplots(
        rows=4, cols=5,
        specs=[[{"type": "xy", 'colspan': 5}, None, None, None, None],
               [{"type": "xy"}, {"type": "xy"}, {"type": "xy"}, {"type": "xy"}, {"type": "xy"}],
               [{"type": "xy", 'colspan': 2}, None, {"type": "xy", 'colspan': 3}, None, None],
               [{"type": "xy", 'colspan': 2}, None, {"type": "xy", 'colspan': 3}, None, None]],
        column_widths=[0.2, 0.2, 0.2, 0.2, 0.2],
        row_heights=[0.20, 0.35, 0.225, 0.225],  # More space for traces
        horizontal_spacing=0.08,  # Increased spacing to prevent label overlap
        vertical_spacing=0.15,  # Increased vertical spacing for titles
        subplot_titles=(
            "Scene Background",
            "", "", "", "", "",  # 5 trace subplots (no individual titles)
            "Trajectory Variability (Mean MAD of Y-position)",
            "Number of Clips per Learning Phase",
            "Success Rate (% Cleared) per Learning Phase",
            "Average Horizontal Speed (pixels/sec)"
        )
    )

    # Add "Traces Overview" as a centered annotation above the trace row
    fig.add_annotation(
        text="Traces Overview",
        xref="paper", yref="paper",
        x=0.5, y=0.785,  # Adjusted for new row heights
        xanchor='center', yanchor='bottom',
        showarrow=False,
        font=dict(size=12)
    )

    # Add all plot components
    fig = add_scene_background(fig, level, scene)
    fig = add_traces_overview(fig, level, scene)
    fig = add_mad_variance_plot(fig, df_wls, config, mappings)  # Row 3, Col 1
    fig = add_clip_count_plot(fig, df_wls, config, col=3)  # Row 3, Col 3
    fig = add_clearance_plot(fig, df_wls, config, mappings, col=1)  # Row 4, Col 1
    fig = add_speed_plot(fig, df_wls, config, mappings, col=3)  # Row 4, Col 3

    # Configure global layout
    scene_fullname = f"{level}s{scene}"
    fig.update_layout(
        height=1000,  # Increased height for better trace resolution
        autosize=True,
        margin=dict(l=20, r=20, t=70, b=70),
        title_text=f"Level: {level} | Scene: {scene} ({scene_fullname})",
        legend=dict(orientation="h", y=-0.10, x=0.0),
        uniformtext_minsize=10,
        uniformtext_mode='hide',
        bargap=0.3,
    )

    # Adjust subplot title sizes and position
    # Make subplot titles visible and bold
    for ann in fig.layout.annotations:
        # Check if this is a main subplot title (not "Traces Overview" or subject labels)
        if hasattr(ann, 'text') and ann.text:
            # Subject labels (sub-01, sub-02, etc.) - keep smaller
            if ann.text.startswith("sub-"):
                ann.font.size = 10
                ann.yshift = 0
            # Traces Overview - keep as is
            elif ann.text == "Traces Overview":
                pass
            # Main subplot titles - make prominent
            else:
                ann.font.size = 16  # Larger font
                ann.font.color = "black"
                ann.yshift = 25  # Move titles much higher above the plots
                ann.yanchor = "bottom"  # Ensure proper anchoring
                # Make titles bold
                if not ann.text.startswith("<b>"):
                    ann.text = f"<b>{ann.text}</b>"

    return fig


# ========================================
# WIDGET CREATION AND MANAGEMENT
# ========================================

def build_scene_slider(df, level):
    """
    Build scene selection slider with available scenes for a level.

    Args:
        df: Metrics dataframe
        level: Level identifier

    Returns:
        ipywidgets SelectionSlider
    """
    opts = scenes_for_level(df, level)
    if not opts:
        return widgets.SelectionSlider(
            options=[],
            value=None,
            description="Scene:",
            layout=widgets.Layout(width="500px")
        )

    return widgets.SelectionSlider(
        options=[(str(s), s) for s in opts],
        value=opts[0],
        description="Scene:",
        layout=widgets.Layout(width="500px"),
        continuous_update=False
    )


def create_widgets(df):
    """
    Create all widgets for the dashboard.

    Args:
        df: Metrics dataframe

    Returns:
        dict: Dictionary containing all widgets
    """
    levels = sorted(df["level_full_name"].dropna().unique())

    level_dd = widgets.Dropdown(
        options=levels,
        value=levels[0],
        description="Level:",
        layout=widgets.Layout(width="240px")
    )

    scene_slider = build_scene_slider(df, level_dd.value)

    out = widgets.Output(layout=widgets.Layout(border="0px"))

    return {
        'level_dd': level_dd,
        'scene_slider': scene_slider,
        'out': out
    }


def setup_callbacks(df, widgets_dict):
    """
    Set up callbacks for widget interactions.

    Args:
        df: Metrics dataframe
        widgets_dict: Dictionary containing all widgets

    Returns:
        dict: Dictionary with callback functions
    """
    level_dd = widgets_dict['level_dd']
    scene_slider = widgets_dict['scene_slider']
    out = widgets_dict['out']

    def draw():
        """Render the dashboard with current widget values."""
        with out:
            clear_output(wait=True)
            if (level_dd.value is None) or (scene_slider.value is None):
                print("No data available for the current selection.")
                return
            fig = make_dashboard(df, level_dd.value, scene_slider.value)
            fig.show(config={"displaylogo": False, "modeBarButtonsToRemove": ["select", "lasso2d"]})

    def on_scene_change(change):
        """Handle scene slider changes."""
        draw()

    def on_level_change(change):
        """Handle level dropdown changes and rebuild scene slider."""
        # Rebuild scene slider with new level's scenes
        new_slider = build_scene_slider(df, level_dd.value)

        # Update the controls widget
        controls = widgets_dict.get('controls')
        if controls:
            controls.children = (level_dd, new_slider)

        # Rebind handler
        new_slider.observe(on_scene_change, names="value")
        scene_slider.unobserve(on_scene_change, names="value")

        # Update reference
        widgets_dict['scene_slider'] = new_slider

        draw()

    # Wire up observers
    level_dd.observe(on_level_change, names="value")
    scene_slider.observe(on_scene_change, names="value")

    return {
        'draw': draw,
        'on_scene_change': on_scene_change,
        'on_level_change': on_level_change
    }


def run_dashboard(df):
    """
    Main function to run the complete dashboard.

    Args:
        df: Metrics dataframe
    """
    # Create widgets
    widgets_dict = create_widgets(df)

    # Create controls container
    controls = widgets.HBox([widgets_dict['level_dd'], widgets_dict['scene_slider']])
    widgets_dict['controls'] = controls

    # Set up callbacks
    callbacks = setup_callbacks(df, widgets_dict)

    # Display widgets and initial dashboard
    display(controls, widgets_dict['out'])
    callbacks['draw']()
