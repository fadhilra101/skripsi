"""
Custom shot simulation page for the xG prediction application.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from ..utils.data_processing import preprocess_shot_data
from ..utils.visualization import create_single_shot_visualization, save_figure_to_bytes, create_download_filename, create_custom_shots_visualization, get_visualization_options, create_visualization_by_type
from ..utils.language import get_translation, get_language_options
from ..utils.custom_shot_manager import (
    initialize_custom_shots_session, add_custom_shot, get_custom_shots_dataframe,
    get_custom_shots_count, prepare_custom_shots_for_download, get_custom_shots_summary,
    clear_all_custom_shots, remove_custom_shot, validate_shot_name
)
from ..models.model_manager import predict_xg


def create_interactive_pitch_simple(current_x=108, current_y=40):
    """
    Create a simple interactive pitch using Plotly with clear coordinate display.
    
    Args:
        current_x: Current x coordinate
        current_y: Current y coordinate
        
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    # Pitch outline - main field
    fig.add_shape(
        type="rect",
        x0=0, y0=0, x1=120, y1=80,
        line=dict(color="white", width=3),
        fillcolor="rgba(34, 100, 34, 0.8)"  # Green field
    )
    
    # Center circle
    fig.add_shape(
        type="circle",
        x0=50, y0=30, x1=70, y1=50,
        line=dict(color="white", width=2),
        fillcolor="rgba(0,0,0,0)"
    )
    
    # Center line
    fig.add_shape(
        type="line",
        x0=60, y0=0, x1=60, y1=80,
        line=dict(color="white", width=2)
    )
    
    # Center spot
    fig.add_shape(
        type="circle",
        x0=59, y0=39, x1=61, y1=41,
        line=dict(color="white", width=1),
        fillcolor="white"
    )
    
    # Left penalty area
    fig.add_shape(
        type="rect",
        x0=0, y0=22, x1=18, y1=58,
        line=dict(color="white", width=2),
        fillcolor="rgba(0,0,0,0)"
    )
    
    # Right penalty area  
    fig.add_shape(
        type="rect",
        x0=102, y0=22, x1=120, y1=58,
        line=dict(color="white", width=2),
        fillcolor="rgba(0,0,0,0)"
    )
    
    # Left 6-yard box
    fig.add_shape(
        type="rect",
        x0=0, y0=30, x1=6, y1=50,
        line=dict(color="white", width=2),
        fillcolor="rgba(0,0,0,0)"
    )
    
    # Right 6-yard box
    fig.add_shape(
        type="rect",
        x0=114, y0=30, x1=120, y1=50,
        line=dict(color="white", width=2),
        fillcolor="rgba(0,0,0,0)"
    )
    
    # Penalty spots
    fig.add_shape(
        type="circle",
        x0=11, y0=39, x1=13, y1=41,
        line=dict(color="white", width=1),
        fillcolor="white"
    )
    
    fig.add_shape(
        type="circle",
        x0=107, y0=39, x1=109, y1=41,
        line=dict(color="white", width=1),
        fillcolor="white"
    )
    
    # Goals
    fig.add_shape(
        type="rect",
        x0=-2, y0=36, x1=0, y1=44,
        line=dict(color="white", width=3),
        fillcolor="rgba(255,255,255,0.3)"
    )
    
    fig.add_shape(
        type="rect",
        x0=120, y0=36, x1=122, y1=44,
        line=dict(color="white", width=3),
        fillcolor="rgba(255,255,255,0.3)"
    )
    
    # Corner arcs
    # Add corner arcs for visual completeness
    fig.add_shape(
        type="circle",
        x0=-1, y0=-1, x1=1, y1=1,
        line=dict(color="white", width=1)
    )
    fig.add_shape(
        type="circle",
        x0=119, y0=-1, x1=121, y1=1,
        line=dict(color="white", width=1)
    )
    fig.add_shape(
        type="circle",
        x0=-1, y0=79, x1=1, y1=81,
        line=dict(color="white", width=1)
    )
    fig.add_shape(
        type="circle",
        x0=119, y0=79, x1=121, y1=81,
        line=dict(color="white", width=1)
    )
    
    # Add current shot location - LARGE RED DOT
    fig.add_trace(go.Scatter(
        x=[current_x],
        y=[current_y],
        mode='markers+text',
        marker=dict(
            size=20,  # Large marker
            color='red',
            symbol='circle',
            line=dict(width=3, color='white')
        ),
        text=[f'⚽ ({current_x}, {current_y})'],
        textposition="top center",
        textfont=dict(size=14, color='white'),
        name='Shot Location',
        hovertemplate=f'<b>Shot Location</b><br>X: {current_x}<br>Y: {current_y}<extra></extra>'
    ))
    
    # Add grid lines for better coordinate reference (subtle)
    for x in range(0, 121, 20):
        fig.add_shape(
            type="line",
            x0=x, y0=0, x1=x, y1=80,
            line=dict(color="rgba(255,255,255,0.1)", width=1)
        )
    
    for y in range(0, 81, 20):
        fig.add_shape(
            type="line",
            x0=0, y0=y, x1=120, y1=y,
            line=dict(color="rgba(255,255,255,0.1)", width=1)
        )
    
    # Add coordinate labels for reference
    fig.add_annotation(
        x=5, y=5,
        text="(0,0)<br>Bottom Left",
        showarrow=False,
        font=dict(color="white", size=10),
        bgcolor="rgba(0,0,0,0.5)"
    )
    
    fig.add_annotation(
        x=5, y=75,
        text="(0,80)<br>Top Left", 
        showarrow=False,
        font=dict(color="white", size=10),
        bgcolor="rgba(0,0,0,0.5)"
    )
    
    fig.add_annotation(
        x=115, y=75,
        text="(120,80)<br>Top Right",
        showarrow=False,
        font=dict(color="white", size=10),
        bgcolor="rgba(0,0,0,0.5)"
    )
    
    fig.add_annotation(
        x=115, y=5,
        text="(120,0)<br>Bottom Right",
        showarrow=False,
        font=dict(color="white", size=10),
        bgcolor="rgba(0,0,0,0.5)"
    )
    
    # Update layout
    fig.update_layout(
        plot_bgcolor='rgba(34, 100, 34, 1)',  # Green background
        paper_bgcolor='rgba(34, 100, 34, 1)',
        showlegend=False,
        width=800,
        height=500,
        margin=dict(l=10, r=10, t=50, b=10),
        title=dict(
            text=f"Shot Location: X={current_x} (0=Left Goal, 120=Right Goal), Y={current_y} (0=Bottom, 80=Top)",
            font=dict(color="white", size=14, family="Arial Black"),
            x=0.5
        ),
        xaxis=dict(
            range=[-5, 125],
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(255,255,255,0.2)',
            showticklabels=True,
            tickfont=dict(color='white', size=10),
            title=dict(text="X Coordinate", font=dict(color='white')),
            zeroline=False,
            fixedrange=True
        ),
        yaxis=dict(
            range=[85, -5],  # Reversed range to match StatsBomb coordinates
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(255,255,255,0.2)',
            showticklabels=True,
            tickfont=dict(color='white', size=10),
            title=dict(text="Y Coordinate", font=dict(color='white')),
            scaleanchor="x",
            scaleratio=1,
            zeroline=False,
            fixedrange=True
        )
    )
    
    return fig


def render_custom_shot_page(model, lang="en"):
    """
    Render the custom shot simulation page.
    
    Args:
        model: Trained model pipeline
        lang: Language code ('en' or 'id')
    """
    # Initialize custom shots session
    initialize_custom_shots_session()
    
    st.header(get_translation("simulate_header", lang))
    st.markdown(get_translation("simulate_desc", lang))

    # Get localized options
    options = get_language_options(lang)

    # Initialize shot coordinates in session state
    if 'shot_x' not in st.session_state:
        st.session_state.shot_x = 108
    if 'shot_y' not in st.session_state:
        st.session_state.shot_y = 40

    col1, col2 = st.columns((1, 1))

    with col1:
        st.subheader(get_translation("shot_characteristics", lang))
        
        # Play pattern selection
        play_pattern_label = st.selectbox(
            get_translation("play_pattern", lang), 
            options=list(options["play_pattern"].keys())
        )
        
        # Position selection
        position_label = st.selectbox(
            get_translation("player_position_select", lang), 
            options=list(options["position"].keys())
        )
        
        # Shot technique selection
        shot_tech_label = st.selectbox(
            get_translation("shot_technique_select", lang), 
            options=list(options["shot_technique"].keys())
        )
        
        # Body part selection
        body_part_label = st.selectbox(
            get_translation("body_part", lang), 
            options=list(options["shot_body_part"].keys())
        )
        
        # Period selection (moved from Additional Features)
        period_options = {
            get_translation("first_half", lang): 1,
            get_translation("second_half", lang): 2
        }
        period_label = st.selectbox(
            get_translation("period", lang), 
            options=list(period_options.keys())
        )
        period = period_options[period_label]

    with col2:
        # Shot type selection
        shot_type_label = st.selectbox(
            get_translation("shot_type", lang), 
            options=list(options["shot_type"].keys())
        )
        
        # Event before shot selection
        type_before_label = st.selectbox(
            get_translation("event_before", lang), 
            options=list(options["type_before"].keys())
        )
        
        st.subheader(get_translation("shot_context_section", lang))
        # Context checkboxes
        open_goal = st.checkbox(get_translation("open_goal", lang))
        one_on_one = st.checkbox(get_translation("one_on_one", lang))
        aerial_won = st.checkbox(get_translation("aerial_won", lang))
        under_pressure = st.checkbox(get_translation("under_pressure", lang))
        
        # Additional shot characteristics (moved from Additional Features)
        shot_first_time = st.checkbox(get_translation("shot_first_time", lang), value=False)
        shot_key_pass = st.checkbox(get_translation("shot_key_pass", lang), value=False)

        st.subheader(get_translation("time_of_shot", lang))
        minute = st.slider(get_translation("minute", lang), 0, 120, 45)
        second = st.slider(get_translation("second", lang), 0, 59, 30)

    # Shot location section - moved below the other inputs
    st.subheader(get_translation("shot_location_xy", lang))
    
    # Instructions for interactive pitch
    st.markdown(get_translation("click_instruction", lang))
    
    # Create coordinate selection using slider-based approach
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        # Display pitch visualization with current coordinates
        fig = create_interactive_pitch_simple(st.session_state.shot_x, st.session_state.shot_y)
        st.plotly_chart(fig, use_container_width=True, key="pitch_display")
    
    with col_right:
        st.markdown("### " + get_translation('current_coordinates', lang))
        st.markdown(f"**X:** {st.session_state.shot_x}")
        st.markdown(f"**Y:** {st.session_state.shot_y}")
        
        st.markdown("### Manual Input")
        # Use sliders for easier coordinate selection
        new_x = st.slider(
            get_translation("start_x", lang) + " (0-120)", 
            min_value=0, max_value=120, 
            value=st.session_state.shot_x, 
            step=1,
            key="slider_x"
        )
        new_y = st.slider(
            get_translation("start_y", lang) + " (0-80)", 
            min_value=0, max_value=80, 
            value=st.session_state.shot_y, 
            step=1,
            key="slider_y",
            help="0 = Bottom of pitch, 80 = Top of pitch (StatsBomb coordinates)"
        )
        
        # Auto-update when sliders change
        if new_x != st.session_state.shot_x or new_y != st.session_state.shot_y:
            st.session_state.shot_x = new_x
            st.session_state.shot_y = new_y
            st.rerun()
    
    # Create a grid-based click system
    st.markdown("---")
    st.markdown("### " + ("Klik area untuk memilih lokasi:" if lang == "id" else "Click area to select location:"))
    
    # Create a visual grid system - coordinates adjusted for StatsBomb system
    pitch_areas = [
        # Format: (name_id, name_en, x, y, description)
        ("area_penalty", "Penalty Area", 108, 40, "Area Penalti"),
        ("area_six_yard", "Six-yard Box", 116, 40, "Kotak 6 Yard"),
        ("area_left_wing", "Left Wing", 100, 15, "Sayap Kiri"),  # Swapped Y coordinates
        ("area_right_wing", "Right Wing", 100, 65, "Sayap Kanan"),  # Swapped Y coordinates
        ("area_outside_box", "Outside Box", 85, 40, "Luar Kotak Penalti"),
        ("area_far_post", "Far Post", 115, 30, "Dekat Tiang Jauh"),  # Swapped Y coordinates
        ("area_near_post", "Near Post", 115, 50, "Dekat Tiang Dekat"),  # Swapped Y coordinates
        ("area_center", "Center", 60, 40, "Tengah Lapangan"),
        ("area_left_flank", "Left Flank", 90, 10, "Sisi Kiri"),  # Swapped Y coordinates
        ("area_right_flank", "Right Flank", 90, 70, "Sisi Kanan")  # Swapped Y coordinates
    ]
    
    # Create grid layout for area buttons
    cols = st.columns(5)
    for i, (area_id, name_en, x, y, name_id) in enumerate(pitch_areas):
        with cols[i % 5]:
            area_name = name_id if lang == "id" else name_en
            if st.button(f"📍 {area_name}", key=area_id, help=f"X: {x}, Y: {y}"):
                st.session_state.shot_x = x
                st.session_state.shot_y = y
                st.rerun()

    # Use session state coordinates for calculation
    start_x = st.session_state.shot_x
    start_y = st.session_state.shot_y

    # Prepare shot data outside of button click to ensure it's available for all operations
    shot_data = {
        'minute': minute,
        'second': second,
        'period': period,
        'play_pattern': options["play_pattern"][play_pattern_label],
        'position': options["position"][position_label],
        'shot_technique': options["shot_technique"][shot_tech_label],
        'shot_body_part': options["shot_body_part"][body_part_label],
        'shot_type': options["shot_type"][shot_type_label],
        'shot_open_goal': int(open_goal),
        'shot_one_on_one': int(one_on_one),
        'shot_aerial_won': int(aerial_won),
        'shot_first_time': int(shot_first_time),
        'shot_key_pass': int(shot_key_pass),
        'under_pressure': int(under_pressure),
        'start_x': start_x,
        'start_y': start_y,
        'type_before': options["type_before"][type_before_label],
    }

    # Shot name input (optional) - placed before the calculate button
    st.markdown("---")
    st.subheader(get_translation("enter_shot_name", lang) + " (Optional)")
    col_name, col_help = st.columns([3, 1])
    with col_name:
        # Handle resetting shot name after successful addition
        default_name = f"Shot {get_custom_shots_count() + 1}"
        if hasattr(st.session_state, 'reset_shot_name') and st.session_state.reset_shot_name:
            # Reset the flag and use new default name
            st.session_state.reset_shot_name = False
            if 'shot_name_input' in st.session_state:
                del st.session_state.shot_name_input
        
        shot_name = st.text_input(
            get_translation("enter_shot_name", lang),
            value=default_name,
            key="shot_name_input",
            help="Akan menggunakan nama default jika dikosongkan" if lang == "id" else "Will use default name if left empty"
        )
    # with col_help:
    #     st.write("")  # Spacing
    #     if st.button("�️ " + ("Gunakan Default" if lang == "id" else "Use Default"), help="Clear input to use default name"):
    #         if 'shot_name_input' in st.session_state:
    #             del st.session_state.shot_name_input
    #         st.rerun()
    with col_help:
        st.write("")  # Spacing
        st.info("💡 " + ("Semua tembakan akan otomatis disimpan ke koleksi" if lang == "id" else "All shots will be automatically saved to collection"))

    if st.button(get_translation("calculate_xg", lang), use_container_width=True):
        # Store shot data in session state for persistence
        st.session_state.current_shot_data = shot_data
        st.session_state.custom_shot_data = shot_data  # For work preservation check
        
        # Convert to DataFrame and preprocess
        shot_df = pd.DataFrame([shot_data])
        shot_df = preprocess_shot_data(shot_df)
        
        # Predict with safe handling and timing
        try:
            predictions, prediction_time = predict_xg(model, shot_df)
            
            if predictions is None:
                # Model went missing during prediction
                st.session_state.model_missing_mid_work = True
                st.rerun()
                return
            
            if predictions is not None:
                xg_value = predictions[0]
                st.session_state.current_xg_value = xg_value
                st.session_state.current_shot_df = shot_df

                # Display results with model performance timing
                col_xg, col_perf = st.columns([2, 1])
                with col_xg:
                    st.metric(
                        label=get_translation("predicted_xg", lang), 
                        value=f"{xg_value:.3f}"
                    )
                with col_perf:
                    st.metric(
                        label=f"⚡ {get_translation('model_performance', lang)}", 
                        value=f"{prediction_time:.4f}s"
                    )

                # Visualize the single shot
                fig, ax = create_single_shot_visualization(
                    shot_df.start_x.iloc[0], 
                    shot_df.start_y.iloc[0], 
                    xg_value,
                    get_translation("simulated_shot_xg", lang)
                )
                st.session_state.current_fig = fig
                st.pyplot(fig)
                
                # Automatically add to collection - use default name if empty
                final_shot_name = shot_name.strip() if shot_name and shot_name.strip() else f"Shot {get_custom_shots_count() + 1}"
                
                if validate_shot_name(final_shot_name):
                    add_custom_shot(shot_data, xg_value, final_shot_name)
                    st.success(f"✅ {get_translation('shot_added_success', lang)} - {final_shot_name}")
                    # Set flag to reset input name on next run
                    st.session_state.reset_shot_name = True
                else:
                    # If default name conflicts, find next available number
                    counter = get_custom_shots_count() + 1
                    while not validate_shot_name(f"Shot {counter}"):
                        counter += 1
                    final_shot_name = f"Shot {counter}"
                    add_custom_shot(shot_data, xg_value, final_shot_name)
                    st.success(f"✅ {get_translation('shot_added_success', lang)} - {final_shot_name}")
                    # Set flag to reset input name on next run
                    st.session_state.reset_shot_name = True
                
        except Exception as e:
            st.error(f"{get_translation('prediction_error_custom', lang)} {e}")
    
    # Show download option for current visualization if available
    if hasattr(st.session_state, 'current_fig') and st.session_state.current_fig is not None:
        st.markdown("---")
        st.subheader("📥 " + ("Unduh Visualisasi" if lang == "id" else "Download Visualization"))
        img_data = save_figure_to_bytes(st.session_state.current_fig, 'png', 300)
        st.download_button(
            label=f"📥 {get_translation('download_viz_desc', lang).split(' sebagai')[0] if lang == 'id' else 'Download Visualization'}",
            data=img_data,
            file_name=create_download_filename("custom_shot_xg", "png"),
            mime="image/png",
            help=get_translation("download_viz_desc", lang),
            use_container_width=True
        )
    
    # Custom Shots Collection Section
    st.markdown("---")
    st.header(get_translation("custom_shots_collection", lang))
    
    # Force refresh session state
    initialize_custom_shots_session()
    shots_count = get_custom_shots_count()
    
    if shots_count == 0:
        st.info(get_translation("no_custom_shots", lang))
    else:
        # Display summary statistics
        summary = get_custom_shots_summary()
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(get_translation("shots_count", lang), summary['total_shots'])
        with col2:
            st.metric(get_translation("average_xg", lang), f"{summary['avg_xg']:.3f}")
        with col3:
            st.metric(get_translation("total_expected_goals", lang), f"{summary['total_expected_goals']:.3f}")
        with col4:
            st.metric("Min/Max xG", f"{summary['min_xg']:.3f}/{summary['max_xg']:.3f}")
        
        # Display custom shots table
        df_custom = get_custom_shots_dataframe()
        
        # Show table with shot management
        st.subheader("Shot Details")
        for _, shot in df_custom.iterrows():
            with st.expander(f"🎯 {shot['shot_name']} - xG: {shot['xg_value']:.3f}"):
                col_info, col_remove = st.columns([4, 1])
                
                with col_info:
                    st.write(f"**Created:** {shot['created_at']}")
                    st.write(f"**Location:** X={shot['start_x']}, Y={shot['start_y']}")
                    st.write(f"**Time:** {shot['minute']}:{shot['second']:02d} - Period {shot['period']}")
                    st.write(f"**xG Value:** {shot['xg_value']:.3f}")
                
                with col_remove:
                    if st.button(f"🗑️ {get_translation('remove_shot', lang)}", key=f"remove_{shot['shot_id']}"):
                        remove_custom_shot(shot['shot_id'])
                        st.rerun()
        
        # Visualization and download options
        st.subheader(get_translation("custom_shots_visualization", lang))
        
        # Visualization type selector for custom shots
        viz_options = get_visualization_options(lang)
        col_viz_selector, col_viz_spacer = st.columns([1, 2])
        
        with col_viz_selector:
            custom_viz_type = st.selectbox(
                get_translation("visualization_type", lang),
                options=list(viz_options.keys()),
                index=0,  # Default to shot map
                key="custom_shots_viz_type"
            )
        
        selected_custom_viz_type = viz_options[custom_viz_type]
        
        # Create visualization for custom shots
        fig_custom, ax_custom = create_visualization_by_type(
            df_custom, 
            selected_custom_viz_type,
            get_translation("custom_shots_visualization", lang),
            half_pitch=False,
            custom_shots=True
        )
        st.pyplot(fig_custom)
        
        # Download options
        col_download, col_action = st.columns(2)
        
        with col_download:
            if shots_count >= 3:  # Only show CSV download if 3 or more shots
                csv_data = prepare_custom_shots_for_download()
                st.download_button(
                    label=f"📊 {get_translation('download_custom_shots', lang)} (CSV)",
                    data=csv_data,
                    file_name=create_download_filename("custom_shots_collection", "csv"),
                    mime="text/csv",
                    use_container_width=True
                )
            else:
                st.info(f"Need at least 3 shots for CSV download (current: {shots_count})")
            
            # Download visualization
            img_custom_data = save_figure_to_bytes(fig_custom, 'png', 300)
            st.download_button(
                label=f"📥 Download Visualization",
                data=img_custom_data,
                file_name=create_download_filename("custom_shots_visualization", "png"),
                mime="image/png",
                use_container_width=True
            )
        
        with col_action:
            st.write("")  # Spacing
            st.write("")  # Spacing
            if st.button(f"🗑️ {get_translation('clear_all_shots', lang)}", use_container_width=True):
                clear_all_custom_shots()
                st.success("All custom shots cleared!")
                st.rerun()
