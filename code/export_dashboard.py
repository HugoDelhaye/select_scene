"""
Export lightweight standalone HTML dashboard.

Instead of embedding all rendered figures, this embeds the data and configuration,
then generates figures client-side using Plotly.js. Much smaller file size.
"""

import os
import json
import base64
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
from io import BytesIO
from tqdm import tqdm
from dashboard import (
    get_phase_configurations,
    get_phase_to_index_mappings,
    scenes_for_level,
    get_subset,
    load_scene_image,
    get_traces
)


def image_to_base64(img_array):
    """
    Convert numpy image array to base64 string.

    Args:
        img_array: Numpy array representing image

    Returns:
        str: Base64-encoded PNG image
    """
    img = Image.fromarray(img_array.astype('uint8'), 'RGB')
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"


def prepare_data_for_export(df):
    """
    Prepare minimal data needed for client-side rendering.

    Args:
        df: Metrics dataframe

    Returns:
        dict: Data structure for embedding in HTML
    """
    print("Preparing data for export...")

    # Convert dataframe to JSON-serializable format
    # Only include necessary columns
    columns_needed = [
        'level_full_name', 'scene', 'scene_full_name', 'subject', 'learning_phase',
        'MAD_mean', 'count', 'cleared', 'speed',
        'delta_MAD_tot', 'delta_clr_tot', 'delta_spd_tot'
    ]

    df_export = df[columns_needed].copy()

    # Convert to dict format (records)
    data_records = df_export.to_dict('records')

    # Get level-scene mapping
    levels = sorted(df["level_full_name"].dropna().unique().tolist())
    scene_index = {}
    for level in levels:
        scenes = scenes_for_level(df, level)
        scene_index[level] = scenes

    # Get phase configurations
    config = get_phase_configurations()

    # Serialize config (convert numpy arrays and other non-JSON types)
    config_serializable = {
        'order_hum': config['order_hum'],
        'order_2phases': config['order_2phases'],
        'colors_2phases': config['colors_2phases'],
        'position_2phase': config['position_2phase'],
        'hum_2phases': config['hum_2phases'],
        'order_ppos': config['order_ppos'],
        'order_4phases': config['order_4phases'],
        'ppos_phases': config['ppos_phases'],
        'phases_everyone': config['phases_everyone'],
        'order_all_phases': config['order_all_phases'],
        'color_4phases': config['color_4phases'],
        'color_ckpt': config['color_ckpt'],
        'colors_phase_ckpt': config['colors_phase_ckpt']
    }

    # Load and encode all images
    print("\nLoading and encoding images...")
    images = {}

    for level in tqdm(levels, desc="Levels"):
        scenes = scene_index[level]
        for scene in tqdm(scenes, desc=f"  {level} scenes", leave=False):
            scene_key = f"{level}|{scene}"

            # Load scene background
            img_arr, _ = load_scene_image(level, scene)
            if img_arr is not None:
                images[f"{scene_key}|background"] = image_to_base64(img_arr)

            # Load trace images
            traces_arr, traces_paths = get_traces(level, int(scene))
            for i, trace_img in enumerate(traces_arr):
                # Extract subject name from path
                path = traces_paths[i]
                parts = path.split(os.sep)
                subject = None
                for part in parts:
                    if part.startswith('sub-'):
                        subject = part
                        break

                if subject:
                    images[f"{scene_key}|trace|{subject}"] = image_to_base64(trace_img)

    print(f"Encoded {len(images)} images")

    return {
        'data': data_records,
        'scene_index': scene_index,
        'config': config_serializable,
        'images': images
    }


def create_lightweight_html(df, output_path='dashboard.html'):
    """
    Create a lightweight standalone HTML file.

    Args:
        df: Metrics dataframe
        output_path: Path to output HTML file

    Returns:
        Path to created HTML file
    """
    print("Generating lightweight standalone dashboard...")

    # Prepare data
    export_data = prepare_data_for_export(df)

    # Get levels for dropdown
    levels = sorted(df["level_full_name"].dropna().unique().tolist())

    # Create HTML
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Scene Exploration Dashboard</title>

    <!-- Plotly.js -->
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>

    <!-- Bootstrap CSS -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">

    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            background-color: #f8f9fa;
            padding: 20px;
        }}

        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            text-align: center;
        }}

        .controls-card {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }}

        .dashboard-container {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}

        .control-group {{
            margin-bottom: 15px;
        }}

        label {{
            font-weight: 600;
            margin-bottom: 5px;
            display: block;
        }}

        select, input[type="range"] {{
            width: 100%;
        }}

        .scene-info {{
            background: #e3f2fd;
            padding: 10px 15px;
            border-radius: 5px;
            margin-top: 10px;
            font-size: 14px;
        }}

        #loading {{
            display: none;
            text-align: center;
            padding: 20px;
            font-size: 18px;
            color: #667eea;
        }}

        .slider-container {{
            display: flex;
            align-items: center;
            gap: 15px;
        }}

        .slider-value {{
            min-width: 50px;
            text-align: center;
            font-weight: 600;
            color: #667eea;
        }}
    </style>
</head>
<body>
    <div class="container-fluid">
        <div class="header">
            <h1>🎮 Scene Exploration Dashboard</h1>
            <p class="mb-0">Explore learning metrics across different scenes, levels, and subjects</p>
        </div>

        <div class="controls-card">
            <div class="row">
                <div class="col-md-4">
                    <div class="control-group">
                        <label for="level-select">Level:</label>
                        <select id="level-select" class="form-select">
                            {_generate_level_options(levels)}
                        </select>
                    </div>
                </div>

                <div class="col-md-8">
                    <div class="control-group">
                        <label for="scene-slider">Scene:</label>
                        <div class="slider-container">
                            <input type="range" id="scene-slider" class="form-range" min="0" max="10" value="0" step="1">
                            <div class="slider-value" id="scene-value">0</div>
                        </div>
                    </div>
                </div>
            </div>

            <div class="scene-info" id="scene-info">
                Select a level and scene to view metrics
            </div>
        </div>

        <!-- Level Overview with clickable thumbnails -->
        <div class="controls-card">
            <h5 style="text-align: center; margin-bottom: 15px;">Level Overview</h5>
            <div id="level-overview" style="display: flex; justify-content: center; gap: 10px; flex-wrap: wrap;">
                Loading...
            </div>
        </div>

        <div id="loading">
            <div class="spinner-border text-primary" role="status">
                <span class="visually-hidden">Loading...</span>
            </div>
            <p>Generating dashboard...</p>
        </div>

        <div class="dashboard-container">
            <!-- Images section -->
            <div id="images-section" style="margin-bottom: 30px;">
                <h5 style="text-align: center; margin-bottom: 15px;">Scene Background</h5>
                <div id="scene-background" style="text-align: center; margin-bottom: 20px;"></div>

                <h5 style="text-align: center; margin-bottom: 15px;">Traces Overview</h5>
                <div id="traces-overview" style="display: flex; justify-content: center; gap: 10px; flex-wrap: wrap;"></div>
            </div>

            <!-- Metrics plots -->
            <div id="dashboard"></div>
        </div>
    </div>

    <script>
        // Embedded data and configuration
        const exportData = {json.dumps(export_data, default=str)};
        const allData = exportData.data;
        const sceneIndex = exportData.scene_index;
        const config = exportData.config;
        const images = exportData.images;

        // Create lookup maps for faster access
        const dataBySceneLevel = {{}};
        allData.forEach(row => {{
            const key = `${{row.level_full_name}}|${{row.scene}}`;
            if (!dataBySceneLevel[key]) {{
                dataBySceneLevel[key] = [];
            }}
            dataBySceneLevel[key].push(row);
        }});

        // DOM elements
        const levelSelect = document.getElementById('level-select');
        const sceneSlider = document.getElementById('scene-slider');
        const sceneValue = document.getElementById('scene-value');
        const sceneInfo = document.getElementById('scene-info');
        const dashboardDiv = document.getElementById('dashboard');
        const loadingDiv = document.getElementById('loading');

        // Current state
        let currentLevel = levelSelect.value;
        let currentScene = 0;

        // Phase-to-index mappings (similar to Python function)
        const phaseToAll = {{}};
        config.order_4phases.forEach((p, i) => phaseToAll[p] = i % 4);
        const allCheckpoints = config.ppos_phases.flat();
        allCheckpoints.forEach((p, i) => phaseToAll[p] = i % 5);

        // Initialize
        function init() {{
            updateSceneSlider();
            updateDashboard();
        }}

        // Update scene slider based on selected level
        function updateSceneSlider() {{
            currentLevel = levelSelect.value;
            const scenes = sceneIndex[currentLevel];

            if (scenes && scenes.length > 0) {{
                sceneSlider.min = Math.min(...scenes);
                sceneSlider.max = Math.max(...scenes);
                sceneSlider.value = scenes[0];
                currentScene = scenes[0];
                sceneValue.textContent = currentScene;
            }} else {{
                sceneSlider.min = 0;
                sceneSlider.max = 0;
                sceneSlider.value = 0;
                currentScene = 0;
                sceneValue.textContent = 'N/A';
            }}
        }}

        // Update level overview with clickable thumbnails
        function updateLevelOverview() {{
            const overviewDiv = document.getElementById('level-overview');
            const scenes = sceneIndex[currentLevel];

            if (!scenes || scenes.length === 0) {{
                overviewDiv.innerHTML = '<p class="text-muted">No scenes available</p>';
                return;
            }}

            overviewDiv.innerHTML = '';

            scenes.forEach(scene => {{
                const sceneKey = `${{currentLevel}}|${{scene}}`;
                const bgKey = `${{sceneKey}}|background`;
                const isSelected = (scene === currentScene);

                // Create thumbnail button
                const button = document.createElement('button');
                button.style.cssText = `
                    display: inline-block;
                    text-align: center;
                    cursor: pointer;
                    margin: 5px;
                    padding: 5px;
                    border: ${{isSelected ? '3px solid #dc3545' : '2px solid transparent'}};
                    background: transparent;
                    box-shadow: ${{isSelected ? '0 4px 8px rgba(0,0,0,0.3)' : 'none'}};
                    transform: ${{isSelected ? 'scale(1.05)' : 'scale(1)'}};
                    transition: all 0.3s ease;
                `;

                // Add image
                if (images[bgKey]) {{
                    const img = document.createElement('img');
                    img.src = images[bgKey];
                    img.style.cssText = `
                        width: ${{isSelected ? '180px' : '150px'}};
                        height: auto;
                        max-height: ${{isSelected ? '144px' : '120px'}};
                        object-fit: contain;
                        display: block;
                    `;
                    button.appendChild(img);
                }} else {{
                    const placeholder = document.createElement('div');
                    placeholder.textContent = `Scene ${{scene}}`;
                    placeholder.style.cssText = `
                        width: ${{isSelected ? '180px' : '150px'}};
                        height: ${{isSelected ? '144px' : '120px'}};
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        background-color: #e0e0e0;
                    `;
                    button.appendChild(placeholder);
                }}

                // Add label
                const label = document.createElement('div');
                label.textContent = `Scene ${{scene}}${{isSelected ? ' ✓' : ''}}`;
                label.style.cssText = `
                    font-size: ${{isSelected ? '14px' : '12px'}};
                    margin-top: 5px;
                    font-weight: ${{isSelected ? 'bold' : 'normal'}};
                    color: ${{isSelected ? '#dc3545' : 'inherit'}};
                `;
                button.appendChild(label);

                // Add click handler
                button.onclick = () => {{
                    currentScene = scene;
                    sceneSlider.value = scene;
                    sceneValue.textContent = scene;
                    updateDashboard();
                }};

                overviewDiv.appendChild(button);
            }});
        }}

        // Get subset of data for current level/scene
        function getSubset() {{
            const key = `${{currentLevel}}|${{currentScene}}`;
            return dataBySceneLevel[key] || [];
        }}

        // Create dashboard figure
        function createDashboard(df_wls) {{
            if (df_wls.length === 0) {{
                return null;
            }}

            const traces = [];

            // Add MAD plot
            addMADPlot(traces, df_wls);

            // Add clip count plot
            addClipCountPlot(traces, df_wls);

            // Add clearance plot
            addClearancePlot(traces, df_wls);

            // Add speed plot
            addSpeedPlot(traces, df_wls);

            // Create layout
            const layout = {{
                height: 800,
                title: `Level: ${{currentLevel}} | Scene: ${{currentScene}}`,
                showlegend: true,
                legend: {{orientation: "h", y: -0.15, x: 0.0}},
                grid: {{rows: 2, columns: 2, pattern: 'independent'}},
                annotations: []
            }};

            return {{data: traces, layout: layout}};
        }}

        // Add MAD variance plot
        function addMADPlot(traces, df_wls) {{
            config.phases_everyone && Object.keys(config.phases_everyone).forEach(sub => {{
                config.phases_everyone[sub].forEach(phase => {{
                    const rows = df_wls.filter(r => r.subject === sub && r.learning_phase === phase);
                    if (rows.length === 0) return;

                    const row = rows[0];
                    traces.push({{
                        type: 'bar',
                        x: [row.subject],
                        y: [row.MAD_mean || 0],
                        name: phase,
                        marker: {{color: config.colors_phase_ckpt[phase]}},
                        offsetgroup: `chk${{phaseToAll[phase]}}`,
                        text: [(row.MAD_mean || 0).toFixed(2)],
                        textposition: 'inside',
                        showlegend: false,
                        xaxis: 'x',
                        yaxis: 'y'
                    }});
                }});
            }});
        }}

        // Add clip count plot
        function addClipCountPlot(traces, df_wls) {{
            let showLegend = false;
            let legendFirst = true;

            config.order_hum && config.order_hum.forEach(sub => {{
                config.hum_2phases[sub].forEach(phase => {{
                    const rows = df_wls.filter(r => r.subject === sub && r.learning_phase === phase);
                    const count = rows.length > 0 ? rows[0].count : 0;

                    if (legendFirst && rows.length > 0) {{
                        showLegend = true;
                        legendFirst = false;
                    }}

                    traces.push({{
                        type: 'bar',
                        x: [sub],
                        y: [count],
                        name: phase,
                        marker: {{color: config.colors_2phases[phase]}},
                        offsetgroup: phase,
                        text: [count],
                        textposition: 'outside',
                        showlegend: showLegend,
                        xaxis: 'x2',
                        yaxis: 'y2'
                    }});
                    showLegend = false;
                }});
            }});
        }}

        // Add clearance plot
        function addClearancePlot(traces, df_wls) {{
            config.phases_everyone && Object.keys(config.phases_everyone).forEach(sub => {{
                config.phases_everyone[sub].forEach(phase => {{
                    const rows = df_wls.filter(r => r.subject === sub && r.learning_phase === phase);
                    if (rows.length === 0) return;

                    const row = rows[0];
                    const clearedPct = (row.cleared || 0) * 100;
                    traces.push({{
                        type: 'bar',
                        x: [row.subject],
                        y: [clearedPct],
                        name: phase,
                        marker: {{color: config.colors_phase_ckpt[phase]}},
                        offsetgroup: `chk${{phaseToAll[phase]}}`,
                        text: [clearedPct.toFixed(0) + '%'],
                        textposition: 'inside',
                        showlegend: false,
                        xaxis: 'x3',
                        yaxis: 'y3'
                    }});
                }});
            }});
        }}

        // Add speed plot
        function addSpeedPlot(traces, df_wls) {{
            config.phases_everyone && Object.keys(config.phases_everyone).forEach(sub => {{
                config.phases_everyone[sub].forEach(phase => {{
                    const rows = df_wls.filter(r => r.subject === sub && r.learning_phase === phase);
                    if (rows.length === 0) return;

                    const row = rows[0];
                    traces.push({{
                        type: 'bar',
                        x: [row.subject],
                        y: [row.speed || 0],
                        name: phase,
                        marker: {{color: config.colors_phase_ckpt[phase]}},
                        offsetgroup: `chk${{phaseToAll[phase]}}`,
                        text: [(row.speed || 0).toFixed(0)],
                        textposition: 'inside',
                        showlegend: false,
                        xaxis: 'x4',
                        yaxis: 'y4'
                    }});
                }});
            }});
        }}

        // Update images
        function updateImages() {{
            const sceneKey = `${{currentLevel}}|${{currentScene}}`;
            const sceneBackgroundDiv = document.getElementById('scene-background');
            const tracesOverviewDiv = document.getElementById('traces-overview');

            // Update scene background
            const bgKey = `${{sceneKey}}|background`;
            if (images[bgKey]) {{
                sceneBackgroundDiv.innerHTML = `<img src="${{images[bgKey]}}" style="max-width: 100%; height: auto; border: 1px solid #ddd;">`;
            }} else {{
                sceneBackgroundDiv.innerHTML = '<p class="text-muted">No background image available</p>';
            }}

            // Update traces
            tracesOverviewDiv.innerHTML = '';
            const traceSubjects = ['sub-01', 'sub-02', 'sub-03', 'sub-05', 'sub-06'];
            traceSubjects.forEach(subject => {{
                const traceKey = `${{sceneKey}}|trace|${{subject}}`;
                if (images[traceKey]) {{
                    const div = document.createElement('div');
                    div.style.textAlign = 'center';
                    div.innerHTML = `
                        <img src="${{images[traceKey]}}" style="max-width: 200px; height: auto; border: 1px solid #ddd;">
                        <div style="font-size: 12px; margin-top: 5px;">${{subject}}</div>
                    `;
                    tracesOverviewDiv.appendChild(div);
                }}
            }});
        }}

        // Update dashboard with current level and scene
        function updateDashboard() {{
            const scenes = sceneIndex[currentLevel];

            if (!scenes || scenes.length === 0) {{
                dashboardDiv.innerHTML = '<div class="alert alert-warning">No scenes available for this level.</div>';
                sceneInfo.textContent = 'No data available';
                return;
            }}

            // Snap to nearest valid scene
            let nearestScene = scenes.reduce((prev, curr) =>
                Math.abs(curr - currentScene) < Math.abs(prev - currentScene) ? curr : prev
            );

            currentScene = nearestScene;
            sceneSlider.value = currentScene;
            sceneValue.textContent = currentScene;

            // Show loading
            loadingDiv.style.display = 'block';
            dashboardDiv.style.opacity = '0.3';

            // Update level overview to highlight selected scene
            updateLevelOverview();

            // Update images
            updateImages();

            // Get data subset
            const df_wls = getSubset();

            if (df_wls.length === 0) {{
                dashboardDiv.innerHTML = '<div class="alert alert-warning">No data available for this scene.</div>';
                sceneInfo.textContent = 'No data available';
                loadingDiv.style.display = 'none';
                dashboardDiv.style.opacity = '1';
                return;
            }}

            // Create figure
            setTimeout(() => {{
                const figure = createDashboard(df_wls);

                if (!figure) {{
                    dashboardDiv.innerHTML = '<div class="alert alert-warning">Could not generate dashboard.</div>';
                    loadingDiv.style.display = 'none';
                    dashboardDiv.style.opacity = '1';
                    return;
                }}

                Plotly.newPlot('dashboard', figure.data, figure.layout, {{
                    responsive: true,
                    displaylogo: false,
                    modeBarButtonsToRemove: ['select', 'lasso2d']
                }});

                // Update info
                const sceneFullName = currentLevel + 's' + currentScene;
                sceneInfo.innerHTML = `<strong>Current scene:</strong> ${{sceneFullName}} | <strong>Level:</strong> ${{currentLevel}} | <strong>Scene:</strong> ${{currentScene}}`;

                // Hide loading
                loadingDiv.style.display = 'none';
                dashboardDiv.style.opacity = '1';
            }}, 100);
        }}

        // Event listeners
        levelSelect.addEventListener('change', () => {{
            updateSceneSlider();
            updateLevelOverview();
            updateDashboard();
        }});

        sceneSlider.addEventListener('input', (e) => {{
            currentScene = parseInt(e.target.value);
            sceneValue.textContent = currentScene;
        }});

        sceneSlider.addEventListener('change', () => {{
            updateDashboard();
        }});

        // Initialize on page load
        window.addEventListener('load', init);
    </script>
</body>
</html>
"""

    # Write HTML file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\n✓ Lightweight standalone dashboard created!")
    print(f"  File: {output_path}")
    print(f"  Size: {file_size_mb:.1f} MB")
    print(f"  Levels: {len(export_data['scene_index'])}")
    print(f"  Total scenes: {sum(len(scenes) for scenes in export_data['scene_index'].values())}")
    print(f"\nOpen the file in your browser to use the dashboard.")

    return str(output_path)


def _generate_level_options(levels):
    """Generate HTML option elements for level select."""
    options = []
    for level in levels:
        options.append(f'<option value="{level}">{level}</option>')
    return '\n                            '.join(options)


def main():
    """Main function for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Export lightweight standalone HTML dashboard'
    )
    parser.add_argument(
        '--data',
        default='sourcedata/df_metrics.parquet',
        help='Path to metrics parquet file (default: sourcedata/df_metrics.parquet)'
    )
    parser.add_argument(
        '--output',
        default='dashboard.html',
        help='Output HTML file path (default: dashboard.html)'
    )

    args = parser.parse_args()

    # Load data
    print(f"Loading data from {args.data}...")
    df = pd.read_parquet(args.data)

    # Create lightweight HTML
    create_lightweight_html(df, args.output)


if __name__ == '__main__':
    main()
