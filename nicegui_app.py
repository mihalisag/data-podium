import os
import json
import re
from nicegui import ui
from dotenv import load_dotenv

import general_utils
from general_utils import *

from openai import OpenAI
from nicegui import ui, run

import fastf1

# # Enable dark mode
# dark = ui.dark_mode()
# dark.enable()

ui.page_title("🏎️ F1 Assistant")


with ui.row():
    ui.markdown("# 🏎️ F1 Assistant")

    # ui.button('L/D', on_click=dark.toggle)


# Helper function to style the Plotly figure
def style_plotly_figure(fig, dark_mode):
    """Update Plotly figure styles based on the theme."""
    if dark_mode:
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",  # Transparent background
            paper_bgcolor="rgba(0,0,0,0)",  # Transparent background
            font=dict(size=12, family="Arial, sans-serif", color="white"),  # White text for dark mode
        )
    else:
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",  # Transparent background
            paper_bgcolor="rgba(0,0,0,0)",  # Transparent background
            font=dict(size=12, family="Arial, sans-serif", color="black"),  # Black text for light mode
        )



# Description to function - need to improve
desc_to_function = {
    "Show the race schedule": "get_schedule_until_now",
    "Retrieve drivers' reaction times to reach a specific speed at the race start.": "get_reaction_time",
    "Show the fastest lap time": "get_fastest_lap_time_print",
    "Display the podium finishes for all races in the season": "get_season_podiums",
    "Shows the final race results": "get_race_results",
    "Output the winner": "get_winner",
    "Display drivers' positions throughout the race": "get_positions_during_race",
    "Compare telemetry data of drivers for specific laps": "compare_telemetry",
    "Plot the count of fastest laps for each driver in the season": "fastest_driver_freq_plot",
    "Compare qualifying performance between drivers in the season": "compare_quali_season",
    "Plot lap times of specified drivers": "laptime_plot",
    "Visualize qualifying results": "get_qualifying_results",
    "Show the lap time distribution": "laptime_distribution_plot",
    "Display the pit stop details": "get_pit_stops",
    "Show basic statistics": "race_statistics"
}


functions = functions_registry

# # Convert functions to JSON-like text
# functions_text = json.dumps(functions, indent=2)

# Automatically create the function dispatcher
function_dispatcher = {
    name: func
    for name, func in globals()['general_utils'].__dict__.items()
    if name in functions_registry
}

func_param_gen = lambda f : dict(inspect.signature(f).parameters.items()).keys()

# Initialize data
grand_prix_by_year = {}
YEARS = range(2018, 2025)

for year in YEARS:
    year_event_names = list(get_schedule_until_now(year)['EventName'])
    grand_prix_by_year[year] = year_event_names


# Assuming `desc_to_function`, `function_dispatcher`, and `func_param_gen` are defined elsewhere
dynamic_ui_placeholder = None  # Placeholder for dynamic UI elements


# Initialize selected values with default dropdown values
selected_values = {
    'drivers_list': None,
    'metrics': None,
    'laps': None,
    'speed': None,
    'year': list(YEARS)[-1],  # Default to the latest year
    'event': grand_prix_by_year[list(YEARS)[-1]][0],  # Default to the first Grand Prix of the latest year
}

def update_selected_value(key, value):
    """Update the selected values dictionary."""
    selected_values[key] = value
    print(f"Updated {key} to {value}")


# Global placeholder for rendering results
result_placeholder = None

async def render_result(result):
    """Dynamically render different types of results in NiceGUI."""
    global result_placeholder

    # Clear the previous results
    if result_placeholder is not None:
        result_placeholder.clear()

    with ui.row().classes('w-full') as result_placeholder:  # Use a row with full width
        if isinstance(result, str):
            ui.label(result).style("white-space: pre-wrap;")
        elif isinstance(result, pd.DataFrame):
            df_serializable = result.copy()
            for col in df_serializable.select_dtypes(include=['datetime', 'datetimetz']):
                df_serializable[col] = df_serializable[col].dt.strftime('%Y-%m-%d %H:%M:%S')
            ui.table(
                columns=[{'field': col, 'title': col} for col in df_serializable.columns],
                rows=df_serializable.to_dict('records'),
            )
        elif isinstance(result, plt.Figure):
            ui.pyplot(result)
        elif isinstance(result, go.Figure):

            # Streamlit-like styling
            result.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",  # Transparent background
                paper_bgcolor="rgba(0,0,0,0)",  # Transparent background
                font=dict(size=16, family="Arial, sans-serif", color="grey"),  # Grey font for dark mode
                # margin=dict(l=10, r=10, t=30, b=10),
            )

            ui.plotly(result).style('width: 100%')
        elif isinstance(result, list):
            if len(result) > 0:
                first_item = result[0]
                if isinstance(first_item, str):
                    for idx, item in enumerate(result):
                        ui.label(f"{item}\n").style("white-space: pre-wrap;")
                elif isinstance(first_item, pd.DataFrame):
                    for idx, item in enumerate(result):
                        df_serializable = item.copy()
                        for col in df_serializable.select_dtypes(include=['datetime', 'datetimetz']): # can make date simpler to prevent this
                            df_serializable[col] = df_serializable[col].dt.strftime('%Y-%m-%d %H:%M:%S')
                        ui.table(
                            columns=[{'field': col, 'title': col} for col in df_serializable.columns],
                            rows=df_serializable.to_dict('records'),
                        )
                elif isinstance(first_item, plt.Figure):
                    for idx, item in enumerate(result):
                        ui.pyplot(item)
                elif isinstance(first_item, go.Figure):
                    for idx, item in enumerate(result):
                        ui.plotly(item).style('width: 100%')
                else:
                    ui.label("List contains unsupported item types.").style("color: orange;")
                    for idx, item in enumerate(result):
                        ui.label(f"Item {idx + 1}:")
                        ui.label(str(item)).style("white-space: pre-wrap;")
            else:
                ui.label("The list is empty.").style("color: blue;")
        else:
            ui.label("Unexpected output type.").style("color: red;")
            ui.label(str(result)).style("white-space: pre-wrap;")

async def execute_function():
    """
    Dynamically execute the selected function with appropriate parameters and render the output.
    """
    global result_placeholder  # Use the global placeholder

    function_name = desc_to_function[function.value]  # Get the selected function name
    print(f"Executing {function_name} with parameters: {selected_values}")

    # Ensure all required parameters are provided
    required_params = list(func_param_gen(function_dispatcher[function_name]))
    function_args = {key: selected_values[key] for key in required_params if key in selected_values}
    print(f"Function Arguments: {function_args}")

    try:
        spinner = ui.spinner(size='lg')  # Display spinner
        # Run the function execution in a separate thread or process
        result = await run.cpu_bound(function_dispatcher[function_name], **function_args)
        spinner.delete()

        # Dynamically render the result
        await render_result(result)

    except Exception as e:
        print(f"Error while executing {function_name}: {e}")

        # Display error in the UI
        if result_placeholder is not None:
            result_placeholder.clear()
        with ui.column() as result_placeholder:
            ui.label(f"Error: {e}").style("color: red;")


# Function select handler
async def function_select(event):
    global dynamic_ui_placeholder
    function = event
    function_name = desc_to_function[function.value]
    function_object = function_dispatcher[function_name]
    function_parameters = list(func_param_gen(function_object))
    print(f"Function: {function_name}")
    print(f"Parameters: {function_parameters}")

    # Clear previous dynamic UI if it exists
    if dynamic_ui_placeholder is not None:
        dynamic_ui_placeholder.clear()
        dynamic_ui_placeholder = None

    # Create new dynamic UI based on function parameters
    with ui.column() as dynamic_ui_placeholder:
        with ui.row():  # .style('flex-wrap: wrap; gap: 20px;'):

            # Handle drivers_list parameter
            if 'drivers_list' in function_parameters:
                spinner = ui.spinner(size='lg')  # Display spinner
                drivers, driver_colors = await run.io_bound(
                    get_drivers, selected_gp_dropdown.value, selected_year_dropdown.value
                )
                spinner.delete()  # Remove spinner when done
                selected_drivers = ui.select(
                    label='Select driver(s):',
                    options=drivers,
                    multiple=True,
                    on_change=lambda e: update_selected_value('drivers_list', e.value)
                ).props('use-chips').style('width: 300px;')

            # Handle metrics parameter
            if 'metrics' in function_parameters:
                metrics = ['speed', 'throttle', 'brake']
                selected_metrics = ui.select(
                    label='Select metric(s):',
                    options=metrics,
                    multiple=True,
                    on_change=lambda e: update_selected_value('metrics', e.value)
                ).props('use-chips').style('width: 300px;')

            # Handle laps parameter
            if 'laps' in function_parameters:
                spinner = ui.spinner(size='lg')  # Display spinner
                total_laps = await run.io_bound(
                    get_total_laps, selected_gp_dropdown.value, selected_year_dropdown.value
                )
                spinner.delete()  # Remove spinner when done
                lap_options = ['fastest'] + [i for i in range(1, total_laps + 1)]

                # Create the lap selector
                laps_selector = ui.select(
                    label='Select laps:',
                    options=lap_options,
                    multiple=True,
                    on_change=lambda e: update_selected_value('laps', e.value)
                ).props('use-chips').style('width: 300px;')

            # Handle speed parameter
            if 'speed' in function_parameters:
                ui.label('Speed (km/h): ')
                speed_slider = ui.slider(
                    min=50,
                    max=250,
                    step=50,
                    on_change=lambda e: update_selected_value('speed', e.value)
                ).style('width: 300px;')
                ui.label


# Function to update the Grand Prix list based on the selected year
def update_grand_prix_list(event):
    year = event.value  # Get the selected year value]
    update_selected_value('year', year)
    grand_prix_list = [*grand_prix_by_year.get(year, []), 'Season']  # Get the Grand Prix options based on the selected year
    selected_gp_dropdown.options = grand_prix_list  # Update the Grand Prix dropdown options
    selected_gp_dropdown.value = grand_prix_list[0]  # Optionally reset the Grand Prix value to the first item
    selected_gp_dropdown.update()  # Re-render the Grand Prix dropdown to reflect the updated options

# Create the UI layout with the year selection and Grand Prix selection
with ui.row():
    # Year dropdown, triggered on change to update Grand Prix options
    selected_year_dropdown = ui.select(
        label="Select a year:",
        options=list(YEARS),
        value=selected_values['year'],  # Default value
        on_change=update_grand_prix_list  # Call update_grand_prix_list when the year changes
    ).style('width: 300px;')

    # Initialize the Grand Prix list based on the default year
    grand_prix_list = [*grand_prix_by_year.get(selected_values['year'], []), 'Season']
    selected_gp_dropdown = ui.select(
        label="Select Grand Prix:",
        options=grand_prix_list,
        value=selected_values['event'],  # Default value
        on_change=lambda e: update_selected_value('event', e.value)
    ).style('width: 400px;')

    function = ui.select(
    label="Select a function:",
    options=list(desc_to_function.keys()),
    on_change=function_select,
    # with_input=True,
    ).style('width: 400px;')
    
# Always show the "Execute" button in the same row
ui.button("Execute", on_click=execute_function)

ui.run(host='0.0.0.0', port=8080)
# ui.run()



# # Button callback to display selected values
# def show_selected_values(event, values_list=[]):
#     for item in values_list:
#         ui.label(f"Item: {item}")
#     # ui.label(f"Selected Year: {selected_year}, Selected Grand Prix: {selected_grand_prix}")
#     # ui.notify(f"Selected Year: {selected_year}, Selected Grand Prix: {selected_grand_prix}")


# # Add the fixed "Execute" button and function select dropdown
# with ui.row():  # Create a row for the "Select function" and "Execute" button
#     with ui.column():  # Column for function select dropdown
#         function = ui.select(
#             label="Select a function:",
#             options=list(desc_to_function.keys()),
#             on_change=function_select,
#             with_input=True,
#         ).style('width: 400px;')

# # Always show the "Execute" button in the same row
# ui.button("Execute", on_click=execute_function)

# # Add an empty row for the results, so they appear in the next line
# with ui.row():
#     with ui.column() as dynamic_ui_placeholder: 
#         pass  # This is the place where the results will appear after execution

# # Results and dynamic content container, placed below the Execute button
# with ui.column() as result_placeholder:
#     pass  # Dynamic content will be added here after executing the function


# session = fastf1.get_session(selected_year_dropdown.value, selected_gp_dropdown.value, 'R')  
# session.load()

# driver_colors = fastf1.plotting.get_driver_color_mapping(session=session)

# def render_chip(driver):
#     # Function to render a chip with the appropriate color
#     return f'<div style="background-color: {driver_colors[driver]}; color: white; padding: 5px; border-radius: 15px;">{driver}</div>'

# # Corrected use of `options` instead of `valueoptions`
# selected_drivers = ui.select(
#     options=[{'value': d, 'text': render_chip(d)} for d in drivers],
#     multiple=True
# ).props('use-chips')

