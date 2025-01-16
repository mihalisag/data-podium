import pandas as pd
import numpy as np

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

import plotly.graph_objects as go

import fastf1
import fastf1.plotting
from fastf1.core import Laps
from fastf1.ergast import Ergast

from timple.timedelta import strftimedelta

import difflib # For matching similar strings

from datetime import datetime

from collections import Counter

import inspect
from typing import Callable, Dict, Any, List

from wiki_utils import *

# # Only show important warnings
# fastf1.set_log_level('WARNING')

# # Enable fastf1 cache
# fastf1.Cache.enable_cache('.cache/fastf1')  # Create a cache folder for faster loading

# A dictionary to store registered functions
functions_registry: Dict[str, Dict[str, Any]] = {}

def register_function(func: Callable) -> Callable:
    """
    Decorator to register a function into the `functions_registry`.
    Parses the docstring to extract description and parameter info.
    """
    docstring = inspect.getdoc(func) or ""
    lines = docstring.split("\n")
    description = lines[0].strip() if lines else "No description available."

    # Parse parameters from the docstring
    params = {}
    parsing_params = False
    for line in lines[1:]:
        line = line.strip()
        if line.lower().startswith("parameters:"):
            parsing_params = True
            continue
        if parsing_params:
            if not line or line.startswith("Returns:"):
                break
            # Parse each parameter line (e.g., "name (type): description")
            if "(" in line and ")" in line and ":" in line:
                param_name = line.split("(")[0].strip()
                param_type = line.split("(")[1].split(")")[0].strip()
                param_desc = line.split(":")[1].strip()
                params[param_name] = {"type": param_type, "description": param_desc}

    # Register function in the global registry
    functions_registry[func.__name__] = {
        "description": description,
        "params": params
    }
    return func


PREDEF_METRICS = ['Speed', 'Throttle', 'Brake', 'nGear', 'RPM']


def lighten_color(color, amount=0.35):
    """
    Lightens a given color by blending it with white.
    
    Parameters:
    - color: str (HEX code or color name)
    - amount: float (0.0 for no change, 1.0 for full white)
    """
    try:
        # Convert color to RGB
        rgb = mcolors.to_rgb(color)
        # Blend with white
        lightened_rgb = [(1 - amount) * c + amount for c in rgb]
        # Convert back to HEX
        return mcolors.to_hex(lightened_rgb)
    except ValueError:
        print(f"Invalid color value: {color}")
        return color


# Function to find the most similar word
def get_most_similar_word(input_word, possible_words):
    # Find the closest match using difflib
    best_match = difflib.get_close_matches(input_word, possible_words, n=1, cutoff=0.6)
    return best_match[0] if best_match else None


def format_timedelta(td: pd.Timedelta) -> str:
    # Convert Timedelta to total seconds
    total_seconds = td.total_seconds()
    
    # Get minutes and remaining seconds
    minutes = int(total_seconds // 60)
    seconds = total_seconds % 60
    
    # Format the output in mm:ss.ssss
    formatted_time = f"{minutes:02}:{seconds:06.4f}"
    
    return formatted_time



## Telemetry functions

@register_function
def get_schedule_until_now(year: int=2024):
    """
    Filters and returns the schedule of races up to the current date for a given season.

    Parameters:
        year (int): The season's year.
    """

    # Retrieve the event schedule for the specified season
    schedule = fastf1.get_event_schedule(year)

    # Filter sessions until current date
    schedule = schedule[schedule['EventDate'] <= datetime.today()]

    # Do not need the preseason results
    schedule = schedule[~schedule['EventName'].str.startswith('Pre-Season')]

    cols = ['RoundNumber', 'EventName', 'Country', 'Location', 'EventDate']

    schedule = schedule[cols]

    return schedule


@register_function
def get_reaction_time(event:str, speed: int, year: int=2024):
    """
    Retrieves the reaction time of drivers to reach a specific speed at the start of the race.

    Parameters:
        event (str): The specific Grand Prix or event (e.g., 'Monaco Grand Prix').
        speed (int): The target speed (in km/h) to reach at the race start.
        year (int): The Grand Prix's year.

    """

    if speed == None: speed = 100

    session = fastf1.get_session(year, event, 'R')  
    session.load()

    drivers_list = session.laps['Driver'].unique()

    driver_reaction_dict = dict()

    for driver in drivers_list:

        driver_reaction_df = session.laps.pick_drivers(driver).pick_laps(1).get_telemetry()
        reaction_time = driver_reaction_df[driver_reaction_df['Speed'] > speed-1].iloc[0]['Time']

        # Convert to seconds and milliseconds
        seconds = int(reaction_time.total_seconds())  # Whole seconds
        milliseconds = int(reaction_time.microseconds / 1000)  # Convert microseconds to milliseconds

        # Combine into desired format
        raw_reaction_time = float(f"{seconds}.{milliseconds:03d}")

        # Save driver reaction time in dictionary
        driver_reaction_dict[driver] = raw_reaction_time

    reaction_df = pd.DataFrame(driver_reaction_dict.items(), columns=['Driver', 'ReactionTime'])
    reaction_df = reaction_df.sort_values('ReactionTime')
    reaction_df = reaction_df.head() # add more than just five

    sorted_drivers = reaction_df['Driver']
    sorted_reaction_times = reaction_df['ReactionTime']

    # Get driver colors of the session
    driver_colors = get_driver_colors(session=session, year=year)
    bar_colors = [driver_colors[driver] for driver in sorted_drivers]

    # Create a Plotly bar chart
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=sorted_drivers,
        y=sorted_reaction_times,
        marker=dict(color=bar_colors),
        # text=sorted_reaction_times,  # Display reaction times on hover
        # textposition='outside',  # Show text above bars
        # textfont=dict(size=12)  # Font size for the text
    ))

    # Update layout
    fig.update_layout(
        title=f'Reaction Time of Drivers to {speed} km/h',
        xaxis_title='Drivers',
        yaxis_title='Reaction Time (s)',
        yaxis=dict(
            range=[
                min(sorted_reaction_times) - (max(sorted_reaction_times) - min(sorted_reaction_times)) * 0.1,
                max(sorted_reaction_times) + (max(sorted_reaction_times) - min(sorted_reaction_times)) * 0.1
            ]
        ),  # Add margin to y-axis limits
        height=500,
        width=800
    )

    return fig

    

# @register_function
def get_fastest_lap_time_result(event: str, year: int=2024):
    """
    Finds the fastest lap time for a specific Grand Prix or event and returns relevant details.

    Parameters:
        event (str): The specific Grand Prix or event (e.g., 'Monaco Grand Prix').
        year (int): The Grand Prix's year.
    """
    session = fastf1.get_session(year, event, 'R')  
    session.load()

    fastest_lap = session.laps.pick_fastest()

    # Convert to full name
    driver = fastf1.plotting.get_driver_name(fastest_lap.Driver, session)
    lap_num = int(fastest_lap.LapNumber)
    lap_time = format_timedelta(fastest_lap.LapTime)
    
    return driver, lap_num, lap_time
    

@register_function
def get_fastest_lap_time_print(event: str, year: int=2024):
    """
    Finds and prints the fastest lap time for a specific Grand Prix or event.

    Parameters:
        event (str): The specific Grand Prix or event (e.g., 'Monaco Grand Prix').
    """

    driver, lap_num, lap_time = get_fastest_lap_time_result(event, year)
    sentence = f"{driver} had the fastest lap time of {lap_time} at lap {lap_num}."

    return sentence


def get_driver_colors(session=None, year: int=2024):
    """
        Returns the driver colors for a specific session or the season
    """

    if session != None:
        # Get driver colors of the session
        driver_colors = fastf1.plotting.get_driver_color_mapping(session=session)
    else:
        events = list(get_schedule_until_now(year)['EventName'])
        driver_colors = dict()

        for event in events:
            session = fastf1.get_session(year, event, 'R')
            session.load()

            # Get driver colors of the session
            temp = fastf1.plotting.get_driver_color_mapping(session=session)
            driver_colors.update(temp)

    return driver_colors


# FIX: the driver colors change by year, check plotting color function
@register_function
def get_season_podiums(year: int = 2024):
    """
    Retrieves the podium finishes for all races in a season and visualizes them using Plotly.

    Parameters:
        year (int): The season's year.
    """

    # Load and preprocess results data
    results_df = (
        pd.read_csv(f'data/gps_{year}_season_results.csv')
        .rename(columns={'Abbreviation': 'Driver'})
    )

    results_df = results_df[['Driver', 'ClassifiedPosition', 'Status']]
    results_df = results_df.dropna()  # Handle potential missing data
    
    season_podiums_df = results_df.copy()

    # Filter numeric positions only
    season_podiums_df = season_podiums_df[season_podiums_df['ClassifiedPosition'].apply(lambda x: x.isnumeric())]
    
    # Convert to integer
    season_podiums_df['ClassifiedPosition'] = season_podiums_df['ClassifiedPosition'].astype(int)

    # Filter podium positions
    season_podiums_df = season_podiums_df[season_podiums_df['ClassifiedPosition'] <= 3]

    # Count podiums per driver and convert to DataFrame
    season_podiums_df = pd.DataFrame(season_podiums_df['Driver'].value_counts()).reset_index()
    season_podiums_df.columns = ['Driver', 'Podium Count']

    # Get driver colors of the session
    driver_colors = get_driver_colors(year=year)
    bar_colors = [driver_colors.get(driver, 'gray') for driver in season_podiums_df["Driver"]]

    # Create a Plotly bar chart
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=season_podiums_df["Driver"],
        y=season_podiums_df["Podium Count"],
        marker=dict(color=bar_colors),
        # text=season_podiums_df["Podium Count"],  # Show the count on hover
        # textposition='outside',  # Position text above bars
        # textfont=dict(size=12)  # Font size of the text
    ))

    # Update layout
    fig.update_layout(
        title=f"Podiums for Each Driver | {year} championship",
        xaxis_title="Driver",
        yaxis_title="Podium Count",
        xaxis=dict(tickangle=45),
        yaxis=dict(tickmode="linear", dtick=1),  # Increment y-axis by 1
        height=500,
        width=800
    )

    return fig

# Might need to update with calling session.results or something
@register_function
def get_race_results(event: str, year: int) -> pd.DataFrame:
    """
    Retrieves the race results or final positions for a specific Grand Prix.

    Parameters:
        event (str): Name of the Grand Prix.
        year (int): The Grand Prix's year.
    """
    # Retrieve the correct event name from fastf1
    event_name = fastf1.get_event(year, event)['EventName']

    # Load and preprocess results data
    results_df = (
        pd.read_csv(f'data/gps_{year}_season_results.csv')
        .rename(columns={'Abbreviation': 'Driver'})
        .loc[lambda df: df['EventName'] == event_name]
        .drop(columns=['EventName'])
    )

    results_df = results_df[['Driver', 'ClassifiedPosition', 'Status']]

    return results_df

# Need to make this faster, use results from another function
@register_function
def get_winner(event: str, year: int) -> str:
    """
    Find the winner of a specific Grand Prix or race.
    
    Parameters:
        event (str): The specific Grand Prix or event (e.g., 'Monaco Grand Prix').
        year (int): The Grand Prix's year.
    """

    # Retrieve the correct event name from fastf1
    event_name = fastf1.get_event(year, event)['EventName']

    # Get race results using the modular function
    race_results_df = get_race_results(event_name, year)

    # Identify the winner (driver with ClassifiedPosition == '1')
    winner = race_results_df.loc[race_results_df['ClassifiedPosition'] == '1', 'Driver'].iloc[0]

    # Get full name
    session = fastf1.get_session(year, event, 'R')  
    session.load()

    winner = fastf1.plotting.get_driver_name(winner, session)

    return f"{winner} won the {year} {event_name}"


# New implementation (inspired by fastf1 existing implementation) - no subset of drivers (could filter out with plotly?)
@register_function
def get_positions_during_race(event: str, year: int=2024):
    """
    Show positions of drivers throughout a race using Plotly.

    Parameters:
        event (str): The specific Grand Prix.
        year (int): The Grand Prix's year.
    """

    session = fastf1.get_session(year, event, 'R')
    session.load(telemetry=False, weather=False)

    # Create a Plotly figure
    fig = go.Figure()

    for drv in session.drivers:
        drv_laps = session.laps.pick_drivers(drv)
        abb = drv_laps['Driver'].iloc[0]

        style = fastf1.plotting.get_driver_style(identifier=abb, style=['color', 'linestyle'], session=session)
        color = style['color']
        linestyle = style['linestyle'].replace('ed', '') # dashed to dash because of plotly :)

        fig.add_trace(go.Scatter(x=drv_laps['LapNumber'], 
                                 y=drv_laps['Position'], 
                                 mode='lines',
                                 name=abb, 
                                 line=dict(color=color, dash=linestyle)))


    # Finalize the plot by inverting the y-axis and customizing labels and title
    fig.update_layout(title=f'Position vs Lap | {year} {event}',
                      xaxis_title='Lap',
                      yaxis_title='Position',
                      xaxis=dict(showgrid=False),
                      yaxis=dict(autorange='reversed', tickvals=[1, 5, 10, 15, 20], showgrid=False, zeroline=False), # bug with white line at y=0 fixed
                      font=dict(size=12, family="Arial, sans-serif"),
                      legend_title='Abbreviation',
                    #   title_y=0.98,
                        legend=dict(
                            font=dict(size=10),
                            orientation='h',  # Horizontal legend
                            yanchor='top',  # Align the legend to the top of its defined space
                            y=-0.2,  # Position the legend below the plot
                            xanchor='center',  # Center the legend horizontally
                            x=0.5,  # Place the legend in the middle horizontally
                        )
                    )
    return fig



def get_drs_zones(car_data):
    """
        Locates DRS for a specific track
    """

    # Extract distances for specific DRS zones
    drs_distances = car_data[car_data['DRS'].isin([10, 12, 14])]['Distance']

    if len(drs_distances) >= 2:

        # Calculate differences between consecutive DRS distances
        distance_differences = drs_distances.diff()

        # Compute the mean and standard deviation of the differences
        mean_difference = np.mean(distance_differences)
        std_dev_difference = np.std(distance_differences)

        # Calculate Z-scores to identify outliers in distance differences
        z_scores = [(diff - mean_difference) / std_dev_difference for diff in distance_differences]

        # Filter Z-scores to identify positive outliers
        positive_outlier_z_scores = [score for score in z_scores if score > 0]

        # Find the indices of the positive outlier Z-scores
        outlier_indices = [z_scores.index(score) for score in positive_outlier_z_scores]

        # Define the start and end indices for DRS zones
        drs_zone_boundaries = [int(drs_distances.index[0]), *outlier_indices, int(drs_distances.index[-1])]

        # Pair consecutive indices to define DRS zones
        drs_zone_indices = [
            [drs_zone_boundaries[i], drs_zone_boundaries[i + 1]]
            for i in range(len(drs_zone_boundaries) - 1)
        ]

        # Extract the DRS zones as lists of distances
        drs_zones = [
            list(drs_distances.iloc[start:end]) for start, end in drs_zone_indices
        ]

        # workaround
        drs_zones = [[zone[0], zone[-1]] for zone in drs_zones if len(zone) >= 2]

        # Merge zones based on proximity
        merged_zones = []
        proximity_threshold = 200 # might need to adjust it better
        for zone in drs_zones:
            if not merged_zones or zone[0] - merged_zones[-1][-1] > proximity_threshold:
                merged_zones.append(zone)
            else:
                merged_zones[-1][-1] = zone[-1]  # Extend the last merged zone
    else:
        return []

    return merged_zones

# Add fasatest lap time as subtitle in plot for each driver
@register_function
def compare_telemetry(event: str, drivers_list: list, metrics: list, laps: list, year: int = 2024):
    """
    Compares/plots telemetry data (e.g., 'speed', 'throttle', 'brake', 'gear') for given drivers, metrics, and laps.

    Parameters:
        event (str): The specific Grand Prix or event (e.g., 'Monaco Grand Prix').
        drivers_list (list): A list of driver abbreviations to compare.
        metrics (list): A list of metrics to compare (e.g., 'speed', 'gas').
        laps (list): A list of laps to analyze (single lap or multiple laps).
        year (int): The Grand Prix's year. Default is 2024.
    """

    figures = []

    # Ensure inputs are lists for consistency
    if isinstance(metrics, str):
        metrics = [metrics]
    if isinstance(laps, (int, str)):
        laps = [laps]

    # Load the session data
    session = fastf1.get_session(year, event, 'R')
    session.load()

    # Get driver colors
    driver_colors = get_driver_colors(session=session)

    # Ensure driver names are abbreviations
    drivers = [fastf1.plotting.get_driver_abbreviation(driver, session) for driver in drivers_list]

    # Validate available metrics
    possible_metrics = [
        'Date', 'SessionTime', 'DriverAhead', 'DistanceToDriverAhead', 'Time',
        'RPM', 'Speed', 'nGear', 'Throttle', 'Brake', 'DRS', 'Source',
        'Distance', 'RelativeDistance', 'Status', 'X', 'Y', 'Z'
    ]

    # Bool to check if selected 'All' in metrics
    all_selection = False

    # Handle case of plotting all metrics
    if metrics[0] == 'All':
        all_selection = True
        metrics =  PREDEF_METRICS
    else:
        metrics = [get_most_similar_word(metric, possible_metrics) for metric in metrics]

    fastest_bool = False
    if laps[0] in ['fastest', 'quickest', 'best']: fastest_bool = True

    def create_plot(telemetry_data, car_data, metric, driver_colors, session, figures, fastest_bool, lap=None):
        """
            Subfunction to generate the plots
        """
        # Create a Plotly figure
        fig = go.Figure()

        print('Metric now: ', metric)

        # Plot telemetry for each driver
        for driver_abbr, telemetry_driver in telemetry_data.items():
            fig.add_trace(
                go.Scatter(
                    x=telemetry_driver['Distance'],
                    y=telemetry_driver[metric],
                    mode='lines',
                    name=driver_abbr,
                    line=dict(color=driver_colors[driver_abbr])
                )
            )

        # Plot corner information
        circuit_info = session.get_circuit_info()
        random_driver_data = list(car_data.values())[0]

        # Positions of corner info in plot
        metric_min = float(random_driver_data[metric].min())
        metric_max = float(random_driver_data[metric].max())

        # Handle gear plot bugs
        if metric == 'nGear': 
            metric_min = (metric_max - metric_min) / metric_max + 1
            metric_max += 0.25

        # Update corner line coordinates to span full y-range of the plot
        corner_line_coors = [metric_min, metric_max]
        corner_num_coor = metric_min - 0.05 * (metric_max - metric_min)  # Slightly below the start of the line

        for _, corner in circuit_info.corners.iterrows():
            # Add vertical line for the corner
            fig.add_shape(
                type='line',
                x0=corner['Distance'], x1=corner['Distance'],
                y0=corner_line_coors[0], y1=corner_line_coors[1],
                line=dict(color='grey', dash='dot')
            )
            # Add corner number annotation just below the start of the line
            fig.add_annotation(
                x=corner['Distance'], y=corner_num_coor,
                text=f"{corner['Number']}{corner['Letter']}",
                showarrow=False, font=dict(size=10),
                align='center'
            )

        # Highlight DRS zones
        if metric == 'Speed':
            for driver_abbr, car_data_driver in car_data.items():
                drs_zones = get_drs_zones(car_data_driver)
                for start, end in drs_zones:
                    fig.add_shape(
                        type='rect',
                        x0=start, x1=end,
                        y0=0.7 * metric_min, y1=1.05 * metric_max,
                        fillcolor=driver_colors[driver_abbr],
                        opacity=0.15,
                        line_width=0
                    )

        if fastest_bool:
            title_substring = '| '
            for abbr in drivers:
                lap_num, laptime = session.laps.pick_drivers(abbr).pick_fastest()[['LapNumber', 'LapTime']] 
                title_substring += f'{abbr}: {strftimedelta(laptime, "%m:%s.%ms")} at lap {int(lap_num)} | '
            title = f'{metric} Graph | {year} {event}'
            title = f"{title}<br><span style='font-size:16px;'>{title_substring}</span>"
        else:
            title = f'{metric} Graph | Lap {lap} | {year} {event}'

        # Add labels and legend
        fig.update_layout(
             title={
                'text': title,
                'x': 0.5,  # Centers the title
                'xanchor': 'center'
            },
            xaxis_title='Distance (m)',
            yaxis_title=metric,
            legend_title='Drivers',
            xaxis=dict(zeroline=False),
            yaxis=dict(zeroline=False),            
        )

        # Store the figure
        figures.append(fig)

    # Plot for each metric
    # Modified main loop
    for metric in metrics:
        
        if fastest_bool:
            drivers_data = {}
            telemetry_data = {}
            car_data = {}

            for abbr in drivers:
                drivers_data[abbr] = session.laps.pick_drivers(abbr).pick_fastest()
                telemetry_data[abbr] = drivers_data[abbr].get_telemetry()
                car_data[abbr] = drivers_data[abbr].get_car_data().add_distance() # need to refactor this, check documentation, simpler way

            # Call the plot function here
            create_plot(telemetry_data, car_data, metric, driver_colors, session, figures, fastest_bool=True)

        else:
            for lap in laps:
                drivers_data = {abbr: session.laps.pick_drivers(abbr).pick_laps(lap) for abbr in drivers}
                telemetry_data = {abbr: drivers_data[abbr].get_telemetry() for abbr in drivers}
                car_data = {abbr: drivers_data[abbr].get_car_data().add_distance() for abbr in drivers}

                # Call the plot function here for non-fastest laps if needed
                create_plot(telemetry_data, car_data, metric, driver_colors, session, figures, fastest_bool=False, lap=lap)

    # Create unified telemetry plot
    if all_selection:
        first_title = figures[0].layout.title.text.split('|')

        if fastest_bool:
            first_title = '|'.join(first_title[2:4]).lstrip(' ').rstrip(' ')
        else:
            first_title = first_title[1].lstrip(' ').rstrip(' ')

        figures = [fig.update_layout(title=None) for fig in figures]
        figures[0].update_layout(title=first_title)

    return figures



# Improve naming
# FIX: driver colors
@register_function
def fastest_driver_freq_plot(year: int=2024):
    """
    Plots the count of fastest laps for every driver who achieved at least one fastest lap in a given season.

    Parameters:
        year (int): The season's year. Defaults to 2024.
    """

    schedule = get_schedule_until_now(2024)
    events = list(schedule['EventName'])

    drivers_list = []

    for event in events:
        driver, _, _ = get_fastest_lap_time_result(event)
        drivers_list.append(driver)

    # Create a frequency dictionary
    fastest_driver_freq = dict(Counter(drivers_list))

    fastest_driver_freq = pd.DataFrame(fastest_driver_freq.items(), columns=['Driver', 'Frequency'])
    fastest_driver_freq = fastest_driver_freq.sort_values(by='Frequency', ascending=False)

    # Extract keys (drivers) and values (frequencies)
    drivers = list(fastest_driver_freq['Driver'])
    counts = list(fastest_driver_freq['Frequency'])

    # Get driver colors of the season
    driver_colors = get_driver_colors(year=year)

    # Generate the list of colors for the bars based on the driver_colors mapping
    bar_colors = [driver_colors[driver] for driver in drivers]

    # Create a Plotly bar chart
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=drivers, 
        y=counts, 
        marker=dict(color=bar_colors),
        name='Fastest Lap Frequency'
    ))

    # Customize layout
    fig.update_layout(
        title='Frequency of Fastest Laps',
        xaxis_title='Drivers',
        yaxis_title='Frequency',
        showlegend=False
    )

    return fig


# NEED TO IMPLEMENT DRIVER -> ABBR function like other ones
# Also better color
# Extend this by:
#  - Having specific color for each driver (need to think for global solution, that is other plots)
#  - Making it for multiple teams (pairs of drivers) (only for the second plot might be problematic)
@register_function
def compare_quali_season(drivers_list: list, year: int=2024):
    """
    Compares the qualifying performance of pairs of drivers across a season.

    Parameters:
        drivers_list (list): A list of driver names or abbreviations to compare.
        year (int): The season's year. Defaults to 2024.
    """
    # It is random and might not contain driver -> need to fix
    session = fastf1.get_session(year, 10, 'R')

    # Make sure the driver names are abbreviations
    drivers_list = [fastf1.plotting.get_driver_abbreviation(driver, session) for driver in drivers_list]
    drivers_list.sort()

    # Load and preprocess the data
    quali_df = pd.read_csv(f'data/gps_{year}_season_quali.csv')
    quali_df = quali_df[['Abbreviation', 'Position', 'EventName']]
    quali_df = quali_df.rename(columns={'Abbreviation': 'Driver'})
    quali_df = quali_df[quali_df['Driver'].isin(drivers_list)]
    quali_df = quali_df[~quali_df['EventName'].str.startswith('Pre-Season')]

    # Plot based on driver colors - solution for same team drivers comparison
    bar_colors = {}
    for driver in drivers_list:
        style = fastf1.plotting.get_driver_style(identifier=driver, style=['color', 'linestyle'], session=session)
        color = style['color']
        linestyle = style['linestyle'].replace('ed', '') # dashed to dash because of plotly :)

        if linestyle == 'dash': color = lighten_color(color)
        
        bar_colors[driver] = color

    # Plot 1: Line plot for qualifying positions
    fig1 = go.Figure()

    for driver, data in quali_df.groupby('Driver'):
        short_event = data['EventName'].apply(lambda x: x.replace('Grand Prix', ''))
        fig1.add_trace(go.Scatter(
            x=short_event, 
            y=data['Position'],
            mode='markers+lines',
            name=driver,
            marker=dict(color=bar_colors[driver]),
            line=dict(color=bar_colors[driver])
        ))

    fig1.update_layout(
        title=f'Qualifying Position vs Event of Different Drivers for {year}',
        xaxis_title='Grand Prix',
        yaxis_title='Position',
        yaxis=dict(
            tickvals=[1, 5, 10, 15, 20],
            ticktext=['1', '5', '10', '15', '20'],
            autorange='reversed',
            zeroline=False,
        ),
        xaxis=dict(
            tickangle=90,
            showgrid=True,
        ),
        showlegend=True,
        height=600
    )


    # Pivot the DataFrame for easier comparison by EventName
    pivoted = quali_df.pivot(index='EventName', columns='Driver', values='Position')
    pivoted['BetterDriver'] = pivoted.idxmin(axis=1)  # Find the driver with the minimum position

    # Create a DataFrame with counts of "BetterDriver"
    outquali_df = pd.DataFrame(pivoted['BetterDriver'].value_counts())
    outquali_df = outquali_df.reset_index()
    outquali_df.columns = ['BetterDriver', 'Count']

    # Extract drivers and counts
    drivers, counts = outquali_df['BetterDriver'], outquali_df['Count']
    driver_counts_map = dict(zip(drivers, counts))

    # Plot 2: Horizontal bar chart for driver performance counts
    fig2 = go.Figure()

    fig2.add_trace(go.Bar(
        y=['Qualifying Performance'],
        x=[-driver_counts_map[drivers[0]]],
        name=f'{drivers[0]} Count',
        orientation='h',
        marker=dict(color=bar_colors[drivers[0]]),
        hoverinfo='skip'  # Disable hover information for this bar
    ))

    fig2.add_trace(go.Bar(
        y=['Qualifying Performance'],
        x=[driver_counts_map[drivers[1]]],
        name=f'{drivers[1]} Count',
        orientation='h',
        marker=dict(color=bar_colors[drivers[1]]),
        hoverinfo='skip'  # Disable hover information for this bar
    ))

    # Add the count values inside the bars
    fig2.add_trace(go.Scatter(
        x=[-counts[0] / 2],
        y=['Qualifying Performance'],
        text=[str(counts[0])],
        mode='text',
        textfont=dict(color='black'),
        showlegend=False,
        hoverinfo='skip'  # Disable hover information for this bar
    ))

    fig2.add_trace(go.Scatter(
        x=[counts[1] / 2],
        y=['Qualifying Performance'],
        text=[str(counts[1])],
        mode='text',
        textfont=dict(color='black'),
        showlegend=False,
        hoverinfo='skip'  # Disable hover information for this bar
    ))

    fig2.update_layout(
        title=f'{year} Qualifying Performance: Outqualified Counts',
        xaxis=dict(
            title='Count',
            zeroline=False,  # Disable the default zero line
            showgrid=False,
            showticklabels=False
        ),
        yaxis=dict(showticklabels=False),
        barmode='overlay',
        showlegend=True,
        height=250,
        shapes=[
            dict(
                type='line',
                x0=0,
                x1=0,
                y0=-0.5,
                y1=0.5,
                line=dict(color='white', width=2),
                layer='above'  # Ensure this line appears above the bars
            )
        ]
    )   
    return [fig1, fig2]

# change color of second driver if in same team (see season qualifying performance function)
# show laptimes in correct format
@register_function
def laptime_plot(event: str, drivers_list: list = [], year: int = 2024):
    """
    Generates a line plot of the lap times of specified drivers in a specific Grand Prix using Plotly.

    Parameters:
        event (str): The specific Grand Prix or event (e.g., 'Monaco Grand Prix').
        drivers_list (list): The names of the drivers to compare.
        year (int): The Grand Prix's year.
    """

    # Load a specific race session
    session = fastf1.get_session(year, event, 'R')
    session.load()

    # Get driver colors of the session
    driver_colors = get_driver_colors(session=session, year=year)

    if not drivers_list:
        drivers_list = session.drivers
        drivers_list = [session.get_driver(driver)['Abbreviation'] for driver in session.drivers]
    else:
        drivers_list = [fastf1.plotting.get_driver_abbreviation(driver, session) for driver in drivers_list]

    # Retrieve the correct event name from fastf1
    event_name = fastf1.get_event(year, event)['EventName']

    # Create a Plotly figure
    fig = go.Figure()

    # Loop over each driver abbreviation
    for driver in drivers_list:
        driver_laps = session.laps.pick_drivers(driver).pick_quicklaps().reset_index()

        fig.add_trace(go.Scatter(
            x=driver_laps['LapNumber'],
            y=driver_laps['LapTime'].dt.total_seconds(),  # Convert timedelta to seconds for plotting
            mode='lines+markers',
            name=driver,
            line=dict(color=driver_colors[driver], width=2),
            marker=dict(symbol='circle')
        ))

    # Customize layout
    fig.update_layout(
        title=f"Laptimes in the {year} {event_name}",
        xaxis_title="Lap Number",
        yaxis_title="Lap Time (s)",
        yaxis=dict(autorange='reversed', zeroline=False),  # Invert y-axis to show faster laps at the top
        xaxis=dict(zeroline=False),
        legend_title="Driver",
    )

    return fig


# Might need to also check the classified grid starting position if needed?
# Inspired by fastf1
@register_function
def get_qualifying_results(event: str, year: int = 2024):
    """
    Retrieves and visualizes the qualifying results for a specific Grand Prix using Plotly.

    Parameters:
        event (str): Name of the Grand Prix.
        year (int): Year of the Grand Prix.
    """
    session = fastf1.get_session(year, event, 'Q')
    session.load()

    list_fastest_laps = []
    for drv in session.drivers:
        drv_fastest_lap = session.laps.pick_drivers(drv).pick_fastest()
        list_fastest_laps.append(drv_fastest_lap)
    
    fastest_laps = Laps(list_fastest_laps).sort_values(by='LapTime').reset_index(drop=True)

    pole_lap = fastest_laps.pick_fastest()
    fastest_laps['LapTimeDelta'] = fastest_laps['LapTime'] - pole_lap['LapTime']

    team_colors = [
        fastf1.plotting.get_team_color(lap['Team'], session=session) 
        for _, lap in fastest_laps.iterlaps()
    ]

    # Create the horizontal bar chart
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=fastest_laps['LapTimeDelta'].dt.total_seconds(),  # Convert timedelta to seconds
        y=fastest_laps['Driver'],
        orientation='h',
        marker=dict(color=team_colors, line=dict(color='grey', width=1)),
        customdata=fastest_laps['LapTimeDelta'].dt.total_seconds(),  # Store seconds in customdata
        hovertemplate='+%{customdata:.3f}s',  # Add 's' to the hover text
        name=''
    ))

    # Customize layout
    pole_time_str = strftimedelta(pole_lap['LapTime'], '%m:%s.%ms')
    fig.update_layout(
        title={
            'text': f"{session.event['EventName']} {session.event.year} Qualifying<br>"
                    f"Fastest Lap: {pole_time_str} ({pole_lap['Driver']})",
            'x': 0.5,
        },
        xaxis=dict(
            title="Lap Time Delta (s)",
            showgrid=True,
            gridcolor='lightgrey',
            zeroline=True,
            zerolinecolor='black',
            zerolinewidth=1,
        ),
        yaxis=dict(
            title="Driver",
            autorange='reversed'  # Fastest on top
        ),
        # plot_bgcolor='white',
        height=600
    )

    return fig

# Inspired by the fastf1 example function
# Has a bug with compound colors
@register_function
def laptime_distribution_plot(event: str, year: int=2024):
    """
    Generates a violin plot of the lap time distributions for the top 10 point finishers in a specified Grand Prix using Plotly.
    
    Parameters:
        event (str): The specific Grand Prix or event (e.g., 'Monaco Grand Prix').
        year (int): The Grand Prix's year.
    """
    # Load the race session
    session = fastf1.get_session(year, event, 'R')  # 'R' indicates the race
    session.load()

    # Retrieve the correct event name from fastf1
    event_name = fastf1.get_event(year, event)['EventName']

    # Get all the laps for the point finishers only
    point_finishers = session.drivers[:10]
    driver_laps = session.laps.pick_drivers(point_finishers).pick_quicklaps().reset_index()

    # Get finishing order abbreviations
    finishing_order = [session.get_driver(i)["Abbreviation"] for i in point_finishers]

    # Convert timedelta to float (seconds) for proper plotting
    driver_laps["LapTime(s)"] = driver_laps["LapTime"].dt.total_seconds()

    # Drop rows with NaN in 'Compound' to avoid errors
    driver_laps = driver_laps.dropna(subset=['Compound'])

    # Create driver color mapping
    driver_colors = fastf1.plotting.get_driver_color_mapping(session=session)

    # Create compound color mapping
    compound_colors = fastf1.plotting.get_compound_mapping(session=session)

    # Create the figure
    fig = go.Figure()

    # Add violin traces for each driver
    for driver in finishing_order:
        driver_data = driver_laps[driver_laps["Driver"] == driver]
        
        # Use the compound color for the entire driver
        driver_color = driver_data["Compound"].map(compound_colors).mode()[0]  # Use most common compound color
        
        fig.add_trace(go.Violin(
            x=driver_data["Driver"],
            y=driver_data["LapTime(s)"],
            name=driver,
            line_color=driver_colors.get(driver, 'blue'),
            box_visible=True,
            meanline_visible=True,
            points="all",
            pointpos=0,
            marker=dict(
                size=4,
                color=driver_color  # Use the computed color for the driver
            )
        ))

    # Customize layout
    fig.update_layout(
        title=f"{year} {event_name} Lap Time Distributions",
        xaxis_title="Driver",
        yaxis_title="Lap Time (s)",
        yaxis=dict(showgrid=True, zeroline=True, zerolinecolor='black'),
        # plot_bgcolor="white",
        height=600,
        legend_title="Driver",
    )

    return fig


    
# temporary before I make another one to make it look nicer
@register_function
def get_pit_stops(event: str, year: int=2024):
    """
        Returns the pit stops of a specific race.

        Parameters:
            event (str): The specific Grand Prix or event (e.g., 'Monaco Grand Prix').
            year (int): The season's year. Defaults to 2024.

    """
    
    session = fastf1.get_session(year, event, 'R')
    session.load()

    # Create an instance of the Ergast class
    ergast = Ergast()

    round_num = session.event.RoundNumber

    pit_df = ergast.get_pit_stops(season=year, round=round_num).content[0]

    driver_info_df = ergast.get_driver_info(season=year, round=round_num)
    pit_df = ergast.get_pit_stops(season=year, round=round_num).content[0]

    # Create driver abbreviations column
    pit_df['driverAbbr'] = [fastf1.plotting.get_driver_abbreviation(driver, session) for driver in pit_df['driverId']]

    # Map abbreviations to number
    abbr_to_num = dict(zip(driver_info_df['driverCode'], driver_info_df['driverNumber']))

    # DataFrame column
    pit_df['driverNum'] = pit_df['driverAbbr'].map(abbr_to_num)

    # Move to first column
    pit_df.insert(0, 'driverNum', pit_df.pop('driverNum'))

    # Drop columns (not needed for now)
    pit_df = pit_df.drop(columns=['driverAbbr', 'driverId', 'duration', 'time'])

    # Group by driverNum and aggregate stop and lap columns as lists
    stops_by_driver = pit_df.groupby('driverNum').agg({'lap': list}).reset_index()

    # check this to understand more
    # stops_by_driver = stops_by_driver.drop(columns=['stop'])

    return stops_by_driver



def get_wikipedia_text(event: str, year: int=2024):
    """
        Returns wikipedia race report
    """

    session = fastf1.get_session(year, event, 'R')
    session.load()

    # race_names = race_names_gen(year=year)

    # Retrieve the correct event name from fastf1
    full_event_name = f"{year} {fastf1.get_event(2024, event)['EventName']}"

    long_text = get_race_report_text(full_event_name)
    short_text = extract_short_paragraph(long_text, max_length=3000)
    print(f"Race: {full_event_name}")
    print(short_text)

    return short_text


def get_drivers(event: str, year: int=2024):
    """
        Get drivers and their colors for specific event
    """

    session = fastf1.get_session(year, event, 'R')
    session.load()

    # drivers = session.drivers
    # drivers = [fastf1.plotting.get_driver_abbreviation(driver, session) for driver in drivers]

    drivers = fastf1.plotting.list_driver_abbreviations(session=session)
    driver_colors = fastf1.plotting.get_driver_color_mapping(session=session)

    return drivers, driver_colors


# New implementation - still needs adjustment and maybe summary from chatgpt
# Could also print
# - average lap time and best lap time - driver
# - average number of pit stops
# - ranking of drivers
# - qualifying ranking of drivers
# - which "round" it is of the schedule
# - have an option for whole season
@register_function
def race_statistics(event: str, year: int=2024):
    """
    Fetches and displays basic statistics about a Formula 1 race.
    
    Parameters:
        event (str): The specific Grand Prix or event (e.g., 'Monaco Grand Prix').
        year (int): The year of the race.
    """

    try:
        # Load the session
        session = fastf1.get_session(year, event, 'R')
        session.load()
        
        short_text = get_wikipedia_text(event, year)

        return short_text
    
    except Exception as e:
        return f"An error occurred: {e}"
    
# Return parameters of function
func_param_gen = lambda f : dict(inspect.signature(f).parameters.items()).keys()

def get_total_laps(event:str, year: int=2024):
    """
        Returns total laps number of a race
    """

    # Load the session
    session = fastf1.get_session(year, event, 'R')
    session.load()

    return session.total_laps

# Could also print
# - average lap time and best lap time - driver
# - average number of pit stops
# - ranking of drivers
# - qualifying ranking of drivers
# - which "round" it is of the schedule
# - have an option for whole season
# @register_function
# def race_statistics(event: str, year: int=2024):
#     """
#     Fetches and displays basic statistics about a Formula 1 race.
    
#     Parameters:
#         event (str): The specific Grand Prix or event (e.g., 'Monaco Grand Prix').
#         year (int): The year of the race.
#     """

#     try:
#         # Load the session
#         session = fastf1.get_session(year, event, 'R')
#         session.load()
        
#         # General information
#         race_name = f"Race Name: {session.event.EventName}"
#         round_num = f"Round: {session.event.RoundNumber}"
#         track = f"Track: {session.event.Location}"
#         country = f"Country: {session.event.Country}"
#         date = f"Date: {session.event.Session5DateUtc}"
#         total_laps = f"Laps: {session.total_laps}"
#         weather = f"Weather: {session.weather_data['TrackTemp'].mean():.1f}°C (track)"
#         rainfall = (session.weather_data['Rainfall'] == True).any()*"There was rainfall during the race"
#         # winner = get_winner(event)
#         # fastest_lap = get_fastest_lap_time_print(event)

#         statistics_list = [race_name, round_num, track, country, date, total_laps, weather, rainfall] #, winner, fastest_lap]
#         # statistics_str = '\n'.join(statistics_list)

#         return statistics_list
    
#     except Exception as e:
#         return f"An error occurred: {e}"