"""
Dash web application for interactive scene exploration.

This module provides a Dash-based web interface for visualizing learning metrics.
Can be run as a local server or exported to standalone HTML.
"""

import os
import pandas as pd
from pathlib import Path

import dash
from dash import dcc, html, Input, Output, State, ALL, MATCH
import dash_bootstrap_components as dbc

# Import plotting functions from dashboard module
from dashboard import (
    make_dashboard,
    scenes_for_level,
    get_phase_configurations
)


def load_data(data_path=None):
    """
    Load metrics data.

    Args:
        data_path: Path to parquet file. If None, uses default location.

    Returns:
        pd.DataFrame: Metrics dataframe
    """
    if data_path is None:
        root = Path(__file__).parent.parent
        data_path = os.path.join(root, 'sourcedata', 'df_metrics.parquet')

    df = pd.read_parquet(data_path)
    return df


def create_app(df=None, external_stylesheets=None):
    """
    Create Dash application.

    Args:
        df: Metrics dataframe. If None, loads from default location.
        external_stylesheets: List of external CSS stylesheets

    Returns:
        dash.Dash: Configured Dash application
    """
    if df is None:
        df = load_data()

    if external_stylesheets is None:
        external_stylesheets = [dbc.themes.BOOTSTRAP]

    # Initialize Dash app
    app = dash.Dash(
        __name__,
        external_stylesheets=external_stylesheets,
        suppress_callback_exceptions=True
    )

    # Get available levels
    levels = sorted(df["level_full_name"].dropna().unique())

    # Create app layout
    app.layout = dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H1("Scene Exploration Dashboard", className="text-center mb-4 mt-4"),
                html.P(
                    "Explore learning metrics across different scenes, levels, and subjects.",
                    className="text-center text-muted mb-4"
                )
            ])
        ]),

        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.Label("Select Level:", className="fw-bold"),
                                dcc.Dropdown(
                                    id='level-dropdown',
                                    options=[{'label': level, 'value': level} for level in levels],
                                    value=levels[0],
                                    clearable=False,
                                    className="mb-3"
                                )
                            ], width=4),

                            dbc.Col([
                                html.Label("Select Scene:", className="fw-bold"),
                                dcc.Slider(
                                    id='scene-slider',
                                    min=0,
                                    max=10,
                                    value=0,
                                    marks={0: '0'},
                                    step=None,
                                    tooltip={"placement": "bottom", "always_visible": True},
                                    className="mb-3"
                                )
                            ], width=8),

                        ])
                    ])
                ], className="mb-4")
            ])
        ]),

        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Level Overview", className="mb-3"),
                        html.Div(id='level-overview', className="text-center")
                    ])
                ], className="mb-4")
            ])
        ]),

        dbc.Row([
            dbc.Col([
                dcc.Loading(
                    id="loading",
                    type="default",
                    children=html.Div(id='dashboard-content')
                )
            ])
        ]),

        html.Footer([
            html.Hr(),
            html.P(
                "Dashboard generated from scene selection analysis pipeline",
                className="text-center text-muted small"
            )
        ], className="mt-5 mb-3")

    ], fluid=True)

    # Callback to update scene slider properties based on level selection
    @app.callback(
        Output('scene-slider', 'min'),
        Output('scene-slider', 'max'),
        Output('scene-slider', 'value'),
        Output('scene-slider', 'marks'),
        Input('level-dropdown', 'value')
    )
    def update_scene_slider(selected_level):
        """Update scene slider properties when level changes."""
        if selected_level is None:
            return 0, 10, 0, {0: '0'}

        scenes = scenes_for_level(df, selected_level)
        if not scenes:
            return 0, 10, 0, {0: '0'}

        # Create marks for each scene
        marks = {scene: str(scene) for scene in scenes}

        return min(scenes), max(scenes), scenes[0], marks

    # Callback to create level overview with scene thumbnails (initial load)
    @app.callback(
        Output('level-overview', 'children'),
        Input('level-dropdown', 'value')
    )
    def update_level_overview_initial(selected_level):
        """Generate level overview with scene thumbnails on level change."""
        if selected_level is None:
            return html.Div("Select a level to see overview", className="text-muted")

        scenes = scenes_for_level(df, selected_level)
        if not scenes:
            return html.Div("No scenes available", className="text-muted")

        # Import image loading function
        from dashboard import load_scene_image
        import base64
        from io import BytesIO
        from PIL import Image as PILImage

        # Create thumbnail for each scene
        thumbnails = []
        for scene in scenes:
            img_arr, img_path = load_scene_image(selected_level, scene)

            if img_arr is not None:
                # Convert to base64 for display
                img = PILImage.fromarray(img_arr.astype('uint8'), 'RGB')
                # Resize to thumbnail width, maintain aspect ratio
                img.thumbnail((200, 500))  # Wide limit, tall limit to preserve full image
                buffer = BytesIO()
                img.save(buffer, format='PNG')
                img_str = base64.b64encode(buffer.getvalue()).decode()
                img_src = f"data:image/png;base64,{img_str}"
            else:
                # Placeholder image
                img_src = ""

            # Clickable thumbnail wrapped in a button
            thumbnail = html.Button([
                html.Div([
                    html.Img(
                        src=img_src,
                        style={
                            'width': '150px',
                            'height': 'auto',  # Auto height to preserve aspect ratio
                            'maxHeight': '120px',
                            'objectFit': 'contain',  # Contain instead of cover
                            'display': 'block'
                        }
                    ) if img_src else html.Div(
                        f"Scene {scene}",
                        style={
                            'width': '150px',
                            'height': '120px',
                            'display': 'flex',
                            'alignItems': 'center',
                            'justifyContent': 'center',
                            'backgroundColor': '#e0e0e0'
                        }
                    ),
                    html.Div(f"Scene {scene}", style={'fontSize': '12px', 'marginTop': '5px'})
                ])
            ],
            id={'type': 'scene-thumb-btn', 'scene': scene},
            style={
                'display': 'inline-block',
                'textAlign': 'center',
                'cursor': 'pointer',
                'margin': '5px',
                'padding': '5px',
                'border': '2px solid transparent',
                'background': 'transparent',
                'transition': 'all 0.3s ease'
            })

            thumbnails.append(thumbnail)

        return html.Div(thumbnails, style={'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'center'})

    # Callback to handle thumbnail clicks
    @app.callback(
        Output('scene-slider', 'value', allow_duplicate=True),
        Input({'type': 'scene-thumb-btn', 'scene': ALL}, 'n_clicks'),
        State({'type': 'scene-thumb-btn', 'scene': ALL}, 'id'),
        prevent_initial_call=True
    )
    def thumbnail_clicked(n_clicks_list, ids_list):
        """Update slider when a thumbnail is clicked."""
        from dash import callback_context

        if not callback_context.triggered:
            raise dash.exceptions.PreventUpdate

        # Find which button was clicked
        trigger_id = callback_context.triggered[0]['prop_id']

        # Extract scene from the trigger
        import json
        if trigger_id != '.':
            # Parse the ID to get the scene number
            id_dict = json.loads(trigger_id.split('.')[0])
            scene = id_dict['scene']
            return scene

        raise dash.exceptions.PreventUpdate

    # Callback to update dashboard and level overview
    @app.callback(
        Output('dashboard-content', 'children'),
        Output('level-overview', 'children', allow_duplicate=True),
        Input('level-dropdown', 'value'),
        Input('scene-slider', 'value'),
        prevent_initial_call='initial_duplicate'
    )
    def update_dashboard(selected_level, selected_scene):
        """Update dashboard when level or scene changes."""
        if selected_level is None:
            return html.Div("Please select a level.", className="text-muted"), html.Div()

        scenes = scenes_for_level(df, selected_level)
        if not scenes:
            return html.Div("No scenes available.", className="text-muted"), html.Div()

        # Use first scene if none selected
        if selected_scene is None:
            selected_scene = scenes[0]

        # Create dashboard figure
        fig = make_dashboard(df, selected_level, selected_scene)
        scene_fullname = f"{selected_level}s{selected_scene}"

        dashboard_content = dcc.Graph(
            figure=fig,
            config={
                'displaylogo': False,
                'modeBarButtonsToRemove': ['select', 'lasso2d'],
                'toImageButtonOptions': {
                    'format': 'png',
                    'filename': f'scene_{scene_fullname}',
                    'height': 1500,
                    'width': 1920,
                    'scale': 2
                }
            },
            style={'height': '1050px'}
        )

        # Update level overview with highlighted scene
        from dashboard import load_scene_image
        import base64
        from io import BytesIO
        from PIL import Image as PILImage

        thumbnails = []
        for scene in scenes:
            img_arr, img_path = load_scene_image(selected_level, scene)
            is_selected = (scene == selected_scene)

            if img_arr is not None:
                img = PILImage.fromarray(img_arr.astype('uint8'), 'RGB')
                # Resize to thumbnail width, maintain aspect ratio
                img.thumbnail((220, 500) if is_selected else (200, 500))
                buffer = BytesIO()
                img.save(buffer, format='PNG')
                img_str = base64.b64encode(buffer.getvalue()).decode()
                img_src = f"data:image/png;base64,{img_str}"
            else:
                img_src = ""

            # Style for selected vs non-selected
            img_style = {
                'width': '180px' if is_selected else '150px',
                'height': 'auto',
                'maxHeight': '144px' if is_selected else '120px',
                'objectFit': 'contain',
                'display': 'block'
            }

            button_style = {
                'display': 'inline-block',
                'textAlign': 'center',
                'cursor': 'pointer',
                'margin': '5px',
                'padding': '5px',
                'border': '3px solid #dc3545' if is_selected else '2px solid transparent',
                'background': 'transparent',
                'boxShadow': '0 4px 8px rgba(0,0,0,0.3)' if is_selected else 'none',
                'transform': 'scale(1.05)' if is_selected else 'scale(1)',
                'transition': 'all 0.3s ease'
            }

            thumbnail = html.Button([
                html.Div([
                    html.Img(
                        src=img_src,
                        style=img_style
                    ) if img_src else html.Div(
                        f"Scene {scene}",
                        style={
                            'width': '180px' if is_selected else '150px',
                            'height': '144px' if is_selected else '120px',
                            'display': 'flex',
                            'alignItems': 'center',
                            'justifyContent': 'center',
                            'backgroundColor': '#e0e0e0'
                        }
                    ),
                    html.Div(
                        f"Scene {scene}" + (" ✓" if is_selected else ""),
                        style={
                            'fontSize': '14px' if is_selected else '12px',
                            'marginTop': '5px',
                            'fontWeight': 'bold' if is_selected else 'normal',
                            'color': '#dc3545' if is_selected else 'inherit'
                        }
                    )
                ])
            ],
            id={'type': 'scene-thumb-btn', 'scene': scene},
            style=button_style)

            thumbnails.append(thumbnail)

        level_overview = html.Div(thumbnails, style={'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'center'})

        return dashboard_content, level_overview

    return app


def run_server(df=None, host='127.0.0.1', port=8050, debug=True):
    """
    Run Dash application server.

    Args:
        df: Metrics dataframe. If None, loads from default location.
        host: Host address
        port: Port number
        debug: Enable debug mode
    """
    app = create_app(df)

    print(f"\n{'='*60}")
    print(f"Starting Scene Exploration Dashboard...")
    print(f"{'='*60}")
    print(f"\nOpen your browser and navigate to:")
    print(f"  → http://{host}:{port}")
    print(f"\nPress Ctrl+C to stop the server\n")

    app.run(host=host, port=port, debug=debug)


if __name__ == '__main__':
    run_server()
