import os
import json

import general_utils
from general_utils import *

from openai import OpenAI
from nicegui import ui, run # , native

import time
import threading

import secrets

import gc


# # Multiprocessing freeze
# import multiprocessing
# multiprocessing.freeze_support()


# # Generate a fixed secret key to ensure storage works correctly
# STORAGE_SECRET = secrets.token_hex(32)
# global session

# @app.on_disconnect
# def clear_variables():
#     print("Clearing app storage on disconnect...")
#     # app.storage.client.clear()  # Clears client-specific storage (per session)
#     app.storage.tab.clear()     # Clears tab-specific storage
#     # app.storage.user.clear()    # Clears user-specific storage
#     # app.storage.general.clear() # Clears shared storage
#     # app.storage.browser.clear() # Clears browser-stored session data
#     print("All app variables cleared!")


@ui.page('/other_page', dark=True)
def other_page():
    ui.link('Main', main_page)
    
    # "About" page
    with open("about.md", "r", encoding="utf-8") as file:
        markdown_content = file.read()

    ui.markdown(markdown_content)

# Create a hidden input to store the computed enabled state for the button.
# Its value will be True if all three selects have a valid value.
computed_enabled = ui.input(value=False).props('hidden')

@ui.page('/')
def main_page():

    computed_enabled.value = False
    
    ui.link('About', other_page).style("position: absolute; top: 8px; right: 20px;").classes('ml-2')

    ui.html('<div style="height: 10px;"></div>')  # This adds a 10px spacer

    ui.page_title("Data Podium")
    ui.colors(primary='#FF1821')#, secondary='#53B689', accent='#111B1E', positive='#53B689')

    # global result_placeholder, dynamic_ui_placeholder  # Declare as global variables

    # Initialize placeholders
    result_placeholder = None
    dynamic_ui_placeholder = None

        
    # Helper function to style the Plotly figure
    def style_plotly_figure(fig, dark_mode):

        color = dark_mode*'black' + (1-dark_mode)*'white'

        """Update Plotly figure styles based on the theme."""
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",  # Transparent background
            paper_bgcolor="rgba(0,0,0,0)",  # Transparent background
            font=dict(size=12, family="Arial, sans-serif", color=color),  # White text for dark mode
            legend=dict(font=dict(size=8))
        )
       

    # Description to function - need to improve AND MAKE SMALLER
    desc_to_function = {
        "Race results": "get_race_results",
        "Race position changes": "get_positions_during_race",
        "Race overview": "race_statistics",
        "Race lap times": "laptime_plot",
        "Race start reaction times": "get_reaction_time",
        "Telemetry comparison": "compare_telemetry",
        "Lap time distribution": "laptime_distribution_plot",
        "Qualifying results": "get_qualifying_results",
        "Fastest lap time": "get_fastest_lap_time_print",
        "Season schedule": "get_schedule_until_now",
        "Season podium finishes": "get_season_podiums",
        "Season fastest laps": "fastest_driver_freq_plot",
        "Season qualifying performance": "compare_quali_season",
        "Tyre strategies": "plot_tyre_strategies",
        "Pit stop information": "get_pit_stops",
        # "Output the winner": "get_winner",
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

    # Initialize selected values with default dropdown values
    selected_values = {
        'drivers_list': None,
        'metrics': None,
        'laps': None,
        'speed': None,
        'year': list(YEARS)[-1],  # Default to the latest year
        'event': grand_prix_by_year[list(YEARS)[-1]][0],  # Default to the first Grand Prix of the latest year
        'session': fastf1.get_session(list(YEARS)[-1], grand_prix_by_year[list(YEARS)[-1]][0], 'R'), # Default session
    }

    # # Load default session
    # selected_values['session'].load(weather=False, messages=False)
   
    def update_selected_value(key, value):
        """Update the selected values dictionary."""
        selected_values[key] = value
        print(f"Updated {key} to {value}")


    # Global placeholder for rendering results
    result_placeholder = None

    async def render_result(result):
        """Dynamically render different types of results in NiceGUI."""
        nonlocal result_placeholder  # Use the instance-specific placeholder

        def render_dataframe(df):
            """Helper to render a pandas DataFrame as a table."""
            df_serializable = df.copy()

            # print(df_serializable.dtypes)

            for col in df_serializable.select_dtypes(include=['datetime', 'datetimetz']):
                df_serializable[col] = df_serializable[col].dt.strftime('%Y-%m-%d %H:%M:%S')
            
            # workaround to handle case of season schedule function - can't fix it, shows TypeError
            if 'EventDate' not in df.columns:
                ui.table.from_pandas(df.astype(str)).style("width: 100%; display: flex; justify-content: center; align-items: center;")

            else:
                ui.table(
                    columns=[{'field': col, 'title': col} for col in df_serializable.columns],
                    rows=df_serializable.to_dict('records'),
                ).style("width: 100%; display: flex; justify-content: center; align-items: center;")

        # @ui.refreshable
        def render_item(item):
            """Helper to render a single item based on its type."""
            if isinstance(item, str):
                ui.label(item).style("white-space: pre-wrap;")
            elif isinstance(item, pd.DataFrame):
                render_dataframe(item)
            elif isinstance(item, plt.Figure):
                ui.pyplot(item).style('width: 100%; height: 100%;')
            elif isinstance(item, go.Figure):
                v_grid = True

                # # Remove grid vertical grid line
                # if item.data[0].type == 'bar':
                #     v_grid = False

                item.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)",  # Transparent background
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="white"),
                    xaxis=dict(showgrid=v_grid, gridcolor="rgba(128,128,128,0.4)", gridwidth=0.75, griddash="dash"),
                    yaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,0.4)", gridwidth=0.75, griddash="dash"),
                )
                ui.plotly(item).style("width: 100%; align-items: center;")
            else:
                ui.label("Unsupported item type.").style("color: orange;")
                ui.label(str(item)).style("white-space: pre-wrap;")

        # Normalize result to a list
        result_list = result if isinstance(result, list) else [result]

        # Clear the previous results
        if result_placeholder is not None:
            result_placeholder.clear()

        with ui.row().classes('w-full') as result_placeholder:  # Use a row with full width
            if not result_list:
                ui.label("The list is empty.").style("color: blue;")
            else:
                for item in result_list:
                    render_item(item)

        # # Print the local variables inside main_page
        # print("Local variables in main_page:")
        # for var, _ in locals().items():
        #     print(f"{var}")

    async def execute_function():
        """
        Execute the selected function and display the result in the second card.
        """

        # # this to find optimal position in code
        # session = update_session()
        # selected_values['session'] = session

        start = time.time()
        # print(f"START TIME: {start}")

        function_name = desc_to_function[function_select.value]  # Get the selected function name
        print(f"Executing {function_name} with parameters: {selected_values}")

        # Ensure all required parameters are provided
        required_params = list(func_param_gen(function_dispatcher[function_name]))
        function_args = {key: selected_values[key] for key in required_params if key in selected_values}
        print(f"Function Arguments: {function_args}")

        try:
            spinner = ui.spinner(size='lg').style("position: absolute; top: 8px; right: 8px;").classes('ml-2')
            # Run the function execution in a separate thread or process
            result = await run.cpu_bound(function_dispatcher[function_name], **function_args)
            spinner.delete()
            print("Execution complete. Result stored.")

            # Automatically display the result in the second card
            if result_placeholder is not None:
                result_placeholder.clear()
                with result_placeholder:
                    await render_result(result)

        except Exception as e:
            print(f"Error while executing {function_name}: {e}")
            # Clear the result placeholder and display the error
            if result_placeholder is not None:
                result_placeholder.clear()
                with result_placeholder:
                    ui.label(f"Error: {e}").style("color: red;")

            # Notify the user of the error
            ui.notify(f"Execution failed: {e}", color="red")
        

        duration = time.time() - start
        print(f"DURATION: {duration}s")
        log_time(function_name, duration)


    def update_button_status():
        """Set computed_enabled.value True if year, event, and function are all selected."""
        computed_enabled.value = bool(
            computed_enabled.value and
            selected_year_dropdown.value and
            selected_gp_dropdown.value and
            function_select.value
        )
        
        # # If driver selection is needed, ensure it's selected
        # if function_select.value in ["Race lap times", "Season qualifying performance"]:
        #     computed_enabled.value = computed_enabled.value and bool(selected_drivers.value)

        # if function_select.value == "Telemetry comparison":
        #     computed_enabled.value = computed_enabled.value and bool(selected_drivers.value and laps_selector.value)

        # print("Event spinner = ", event_spinner.visible)
        # computed_enabled.value = not event_spinner.visible

        print("Button enabled status updated to:", computed_enabled.value)


    # Function select handler
    async def function_select_handler(event):
        nonlocal dynamic_ui_placeholder
        function_select = event
        function_name = desc_to_function[function_select.value]
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
                    spinner = ui.spinner(size='lg').style("position: absolute; top: 8px; right: 8px;").classes('ml-2')  # Display spinner
                    drivers, driver_colors = await run.io_bound(
                        get_drivers, selected_values['session'] # did change here with session 
                    )
                    spinner.delete()  # Remove spinner when done
                    
                    global selected_drivers
                    selected_drivers = ui.select(
                        label='Select driver(s):',
                        options=drivers,
                        multiple=True,
                        on_change=lambda e: (update_selected_value('drivers_list', e.value), update_button_status())
                        # clearable=True # clear selections button
                    ).props('use-chips').style('width: 250px;')

                # Handle metrics parameter
                if 'metrics' in function_parameters:
                    update_selected_value('metrics', 'All')

                # Handle laps parameter
                if 'laps' in function_parameters:
                    spinner = ui.spinner(size='lg').style("position: absolute; top: 8px; right: 8px;").classes('ml-2')  # Display spinner
                    total_laps = await run.io_bound(
                        get_total_laps, selected_values['session'] # did change here with session 
                    )
                    spinner.delete()  # Remove spinner when done
                    lap_options = ['fastest'] + [i for i in range(1, total_laps + 1)]

                    # Create the lap selector
                    global laps_selector
                    laps_selector = ui.select(
                        label='Select lap:',
                        options=lap_options,
                        # multiple=True,
                        on_change=lambda e: (update_selected_value('laps', e.value), update_button_status())
                    ).props('use-chips').style('width: 150px;')

                # Handle speed parameter
                if 'speed' in function_parameters:
                    with ui.row():
                        ui.label('Speed (km/h): ')
                        speed_slider = ui.slider(
                            value=100, # needs to be same value as the default from the function (?)
                            min=50,
                            max=300,
                            step=10,
                            on_change=lambda e: update_selected_value('speed', e.value)
                        ).style('width: 300px;')
                        ui.label().bind_text_from(speed_slider, 'value')


        update_button_status()

        
    # Really difficult, need to understand how it works
    # Function to simulate loading the session and show the spinner
    def update_session_with_spinner(event=None):
        # Show the spinner
        event_spinner.visible = True
        
        # Call the blocking update session function in a separate thread to avoid UI freeze
        def load_session():
            year = selected_year_dropdown.value
            event = selected_gp_dropdown.value
            
            try:
                computed_enabled.value = False
                session = fastf1.get_session(year, event, 'R')
                session.load(weather=False, messages=False)  # This is blocking
                print(f"Session loaded for {year} {event}")
                computed_enabled.value = True
            except Exception as e:
                print(f"Error loading session: {e}")
                session = None  # Reset session on failure
            
            selected_values['session'] = session

            # After session is loaded, hide the spinner
            event_spinner.visible = False

        # Run the loading process in a separate thread to avoid blocking UI
        threading.Thread(target=load_session).start()

        update_button_status()


     # Function to update the Grand Prix list based on the selected year
    def update_grand_prix_list(event):
        year = event.value  # Get the selected year value]
        update_selected_value('year', year)

        grand_prix_list = grand_prix_by_year.get(year, [])  # Get the Grand Prix options based on the selected year
        selected_gp_dropdown.options = grand_prix_list  # Update the Grand Prix dropdown options
        selected_gp_dropdown.value = grand_prix_list[0]  # Optionally reset the Grand Prix value to the first item
        selected_gp_dropdown.update()  # Re-render the Grand Prix dropdown to reflect the updated options

        update_button_status()


    # Create the UI layout
    with ui.card().classes("w-full p-4 shadow-lg"):
        # Enable dark mode
        dark = ui.dark_mode()
        dark.enable()
        
        # # Create a button and position it at the top-right corner
        # ui.button(icon='brightness_auto', on_click=dark.toggle).style("position: absolute; top: 12.5px; right: 12.5px; font-size: 18px; padding: 5px 8px;")    

        # Application title
        ui.markdown("# 🏎️ Data Podium")

        # Create a spinner widget (initially hidden)
        event_spinner = ui.spinner(size='lg').style("position: absolute; top: 18px; right: 8px;").classes('ml-2')
        event_spinner.visible = False
        
        with ui.row():

            # Year dropdown, triggered on change to update Grand Prix options
            selected_year_dropdown = ui.select(
                label="Select year:",
                options=list(YEARS),
                value=selected_values['year'],
                on_change=lambda e: update_grand_prix_list(e),
            ).style('width: 150px;')


            # Initialize the Grand Prix list based on the default year
            grand_prix_list = grand_prix_by_year.get(selected_values['year'], [])
            
            # Modify the dropdown to trigger the session load with spinner
            selected_gp_dropdown = ui.select(
                label="Select event:",
                options=grand_prix_list,
                # value=selected_values['event'],  # Default value
                on_change=lambda e: (update_selected_value('event', e.value),
                                    update_session_with_spinner())
            ).style('width: 225px;')
        

            # Rename the function select to avoid confusion with Python's keyword:
            function_select = ui.select(
                label="Select a function:",
                options=list(desc_to_function.keys()),  # using your existing dict keys
                value=None,  # No default selection
                on_change=function_select_handler,
            ).style('width: 250px;')

        
            # Now create the button and bind its enabled state to computed_enabled.value:
            execute_button = ui.button("Show results", on_click=execute_function).style("position: absolute; bottom: 12.5px; right: 12.5px; margin-top: 12px;").classes('ml-2')
            execute_button.bind_enabled_from(computed_enabled, 'value')

    
    result_placeholder = ui.column().style("width: 100%;") # Placeholder for the rendered result

        
ui.run(host='127.0.0.1', port=8080, favicon="🏎️") #, storage_secret=STORAGE_SECRET)