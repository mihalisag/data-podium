import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import fastf1
import fastf1.plotting
from fastf1.core import Laps

from timple.timedelta import strftimedelta

import difflib # For matching similar strings

from datetime import datetime

from collections import Counter

import inspect
from typing import Callable, Dict, Any, List

# Only show important warnings
fastf1.set_log_level('WARNING')

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


# Driver colors codes
DRIVER_COLORS = {
 'VER': '#0600ef',
 'PER': '#0600ef',
 'GAS': '#ff87bc',
 'OCO': '#ff87bc',
 'ALO': '#00665f',
 'STR': '#00665f',
 'LEC': '#e8002d',
 'SAI': '#e8002d',
 'SAR': '#00a0dd',
 'ALB': '#00a0dd',
 'MAG': '#b6babd',
 'HUL': '#b6babd',
 'TSU': '#364aa9',
 'RIC': '#364aa9',
 'ZHO': '#00e700',
 'BOT': '#00e700',
 'NOR': '#ff8000',
 'PIA': '#ff8000',
 'HAM': '#27f4d2',
 'RUS': '#27f4d2',
 'BEA': '#b6babd',
 'COL': '#00a0dd',
 'LAW': '#364aa9'}

# Team color codes
TEAM_COLORS = {
 'Red Bull Racing': '#0600ef',
 'Ferrari': '#e8002d',
 'Mercedes': '#27f4d2',
 'McLaren': '#ff8000',
 'Aston Martin': '#00665f',
 'Kick Sauber': '#00e700',
 'Haas F1 Team': '#b6babd',
 'RB': '#364aa9',
 'Williams': '#00a0dd',
 'Alpine': '#ff87bc'}


def lighten_color(color, amount=0.5):
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
    schedule = schedule[schedule['EventName'] != 'Pre-Season Testing']

    return schedule


@register_function
def get_reaction_time(event:str, speed: int=100):
    """
    Retrieves the reaction time of drivers to reach a specific speed at the start of the race.

    Parameters:
        event (str): The specific Grand Prix or event (e.g., 'Monaco Grand Prix').
        speed (int): The target speed (in km/h) to reach at the race start.
    """

    year = 2024

    session = fastf1.get_session(year, event, 'R')  # 'R' indicates the race; can also use 'Q', 'FP1', 'FP2', 'FP3'
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
    reaction_df = reaction_df.head()

    sorted_drivers = reaction_df['Driver']
    sorted_reaction_times = reaction_df['ReactionTime']

    # Create a matplotlib figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Generate the list of colors for the bars based on the DRIVER_COLORS mapping
    bar_colors = [DRIVER_COLORS[driver] for driver in sorted_drivers]

    # Plot the bar chart with the custom colors
    ax.bar(sorted_drivers, sorted_reaction_times, color=bar_colors)
    # Add labels and title
    ax.set_xlabel('Drivers')
    ax.set_ylabel('Reaction time')

    # Set the y-axis limits to focus on the range of reaction times
    min_reaction = min(sorted_reaction_times)
    max_reaction = max(sorted_reaction_times)

    # Add a small margin around the reaction times
    y_margin = (max_reaction - min_reaction) * 0.7  # 10% of the range

    # Apply the y-axis limits
    ax.set_ylim(min_reaction - y_margin, max_reaction + y_margin)

    ax.set_title(f'Reaction time of drivers to {speed} km/h')

    return fig
    

@register_function
def get_fastest_lap_time_result(event: str):
    """
    Finds the fastest lap time for a specific Grand Prix or event and returns relevant details.

    Parameters:
        event (str): The specific Grand Prix or event (e.g., 'Monaco Grand Prix').
    """
    year = 2024
    session = fastf1.get_session(year, event, 'R')  # 'R' indicates the race; can also use 'Q', 'FP1', 'FP2', 'FP3'
    session.load()

    fastest_lap = session.laps.pick_fastest()

    driver = fastest_lap.Driver
    lap_num = int(fastest_lap.LapNumber)
    lap_time = format_timedelta(fastest_lap.LapTime)
    
    return driver, lap_num, lap_time
    

@register_function
def get_fastest_lap_time_print(event: str):
    """
    Finds and prints the fastest lap time for a specific Grand Prix or event.

    Parameters:
        event (str): The specific Grand Prix or event (e.g., 'Monaco Grand Prix').
    """

    driver, lap_num, lap_time = get_fastest_lap_time_result(event)
    sentence = f'Driver {driver} had the fastest lap time of {lap_time} at lap {lap_num}.'

    return sentence


@register_function
def get_season_podiums():
    """
    Retrieves the podium finishes for all races in a season.

    Parameters:
        None
    """

    # Load and preprocess results data
    results_df = (
        pd.read_csv('data/gps_2024_season_results.csv')
        .rename(columns={'Abbreviation': 'Driver'})
    )

    results_df = results_df[['Driver', 'ClassifiedPosition', 'Status']]
    results_df = results_df.dropna() # problem with Las Vegas Grand Prix?
    
    season_podiums_df = results_df.copy()

    # Take only numberic positions
    season_podiums_df = season_podiums_df[season_podiums_df['ClassifiedPosition'].apply(lambda x: x.isnumeric())]
    
    # Convert to integer
    season_podiums_df['ClassifiedPosition'] = season_podiums_df['ClassifiedPosition'].astype(int)

    # Take only podiums
    season_podiums_df = season_podiums_df[season_podiums_df['ClassifiedPosition']<= 3]

    # Take value counts of podiums and convert to DataFrame
    season_podiums_df = pd.DataFrame(season_podiums_df['Driver'].value_counts()).reset_index()

    bar_colors = [DRIVER_COLORS[driver] for driver in season_podiums_df["Driver"]]

    # Create a matplotlib figure and axis
    fig, ax = plt.subplots(figsize=(8, 5))  # Optional: Set figure size

    # Plot the bar chart on the ax object
    ax.bar(season_podiums_df["Driver"], season_podiums_df["count"], color=bar_colors)

    # Set labels and title using ax
    ax.set_xlabel("Driver", fontsize=12)
    ax.set_ylabel("Podium count", fontsize=12)
    ax.set_title("Podiums for each driver", fontsize=14)

    # Set y-axis ticks to increment by one (from 0 to max count)
    ax.set_yticks(range(0, max(season_podiums_df["count"]) + 1, 1))  # Start at 0, go to max count, step by 1

    return fig


@register_function
def get_race_results(event: str, year: int=2024) -> pd.DataFrame:
    """
    Retrieves the race results for a specific Grand Prix.

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

@register_function
def get_winner(event: str) -> str:
    """
    Find the winner of a specific Grand Prix or race.
    
    Parameters:
        event (str): The specific Grand Prix or event (e.g., 'Monaco Grand Prix').
    """
    # Enable fastf1 cache
    fastf1.Cache.enable_cache('.cache/fastf1')  # Create a cache folder for faster loading

    # Retrieve the correct event name from fastf1
    event_name = fastf1.get_event(2024, event)['EventName']

    # Get race results using the modular function
    race_results_df = get_race_results(event_name)

    # Identify the winner (driver with ClassifiedPosition == '1')
    winner = race_results_df.loc[race_results_df['ClassifiedPosition'] == '1', 'Driver'].iloc[0]

    return f"Driver {winner} won the {event_name}"


# # Old implementation
# @register_function
# def get_positions_during_race(event: str, drivers_abbrs: list=[]):
#     """
#     Show positions of drivers throughout a race.
    
#     Parameters:
#         event (str): The specific Grand Prix or event.
#         drivers_abbrs (list): The names of the drivers.
#     """
#     pos_df = pd.read_csv("data/gps_2024_season_laps.csv")
#     pos_df = pos_df[['Driver', 'LapNumber', 'Stint', 'Position', 'EventName']]
#     pos_df = pos_df[pos_df['EventName'] == event]

#     if drivers_abbrs:
#         # Filter DataFrame for specific drivers
#         pos_df = pos_df[pos_df['Driver'].isin(drivers_abbrs)]

#     # Create a matplotlib figure
#     fig, ax = plt.subplots(figsize=(10, 6))

#     # Group by 'Driver' and plot each driver's data
#     for driver, data in pos_df.groupby('Driver'):
#         ax.plot(data['LapNumber'], data['Position'], label=driver, color=DRIVER_COLORS[driver])

#     # Adding labels and legend
#     ax.set_xlabel('Lap')
#     ax.set_ylabel('Position')
#     ax.set_title(f'Position vs Lap for Different Drivers | {event}')
#     ax.legend(title='Driver')
#     ax.invert_yaxis()  # Optional: Reverse y-axis so 1st position is at the top
    
#     return fig

# New implementation (inspired by fastf1 existing implementation) - no subset of drivers (could filter out with plotly?)
@register_function
def get_positions_during_race(event: str, year: int=2024):
    """
    Show positions of drivers throughout a race.

    Parameters:
        event (str): The specific Grand Prix or event.
        year (int): The Grand Prix's year.
    """

    # Enable fastf1 cache
    fastf1.Cache.enable_cache('.cache/fastf1')  # Create a cache folder for faster loading

    session = fastf1.get_session(year, event, 'R')
    session.load(telemetry=False, weather=False)

    # Create a matplotlib figure
    fig, ax = plt.subplots(figsize=(10, 6))

    for drv in session.drivers:
        drv_laps = session.laps.pick_drivers(drv)
        abb = drv_laps['Driver'].iloc[0]

        style = fastf1.plotting.get_driver_style(identifier=abb,
                                                style=['color', 'linestyle'],
                                                session=session)

        ax.plot(drv_laps['LapNumber'], drv_laps['Position'],
                label=abb, **style)
    
    # Finalize the plot by setting y-limits that invert the y-axis so that position
    # one is at the top, set custom tick positions and axis labels.
    ax.set_ylim([20.5, 0.5])
    ax.set_yticks([1, 5, 10, 15, 20])
    ax.set_xlabel('Lap')
    ax.set_ylabel('Position')
    ax.set_title(f'Position vs Lap for Different Drivers | {event}')
    ax.legend(title='Driver')

    return fig


def get_drs_zones(car_data):
    '''
        Locates DRS for a specific track
    '''

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
    else:
        return []

    return drs_zones


# Expand for multiple laps (of same driver) ?
@register_function
def compare_metric(event: str, session_type: str, drivers_abbrs: list, metric: str, lap:int, year: int=2024):
    """
    Compares the telemetry of a specific metric (e.g., 'speed', 'gas', 'throttle', 'gear') from a list of drivers.
    
    Parameters:
        event (str): The specific Grand Prix or event (e.g., 'Monaco Grand Prix').
        session_type (str): The type of session (e.g., 'race', 'qualifying').
        drivers_abbrs (list): The names of the drivers to compare.
        metric (str): The metric to compare (e.g., 'speed', 'gas').
        lap (int): The specific lap of the session.
        year (int): The Grand Prix's year.
    """

    # Enable fastf1 cache
    fastf1.Cache.enable_cache('.cache/fastf1')  # Create a cache folder for faster loading

    # Specific plottting style
    fastf1.plotting.setup_mpl(mpl_timedelta_support=True, misc_mpl_mods=False, color_scheme='fastf1')

    session = fastf1.get_session(year, event, session_type)  # 'R' indicates the race; can also use 'Q', 'FP1', 'FP2', 'FP3'
    session.load()

    # Create dictionaries for laps and telemetry data.
    drivers = {abbr: session.laps.pick_drivers(abbr).pick_laps(lap) for abbr in drivers_abbrs}
    telemetry_drivers = {abbr: drivers[abbr].get_telemetry() for abbr in drivers_abbrs}

    # Find relevant names based on existing telemetry DataFrame columns
    possible_metrics = ['Date', 'SessionTime', 'DriverAhead', 'DistanceToDriverAhead', 'Time',
                        'RPM', 'Speed', 'nGear', 'Throttle', 'Brake', 'DRS', 'Source',
                        'Distance', 'RelativeDistance', 'Status', 'X', 'Y', 'Z'] # easier than accessing random driver and then telemetry -> maybe can improve
   
    metric = get_most_similar_word(metric, possible_metrics)

    # Create a matplotlib figure
    fig, ax = plt.subplots(figsize=(16, 6))

    # Main metric graph
    for abbr, telemetry_driver in telemetry_drivers.items():
        ax.plot(telemetry_driver['Distance'], telemetry_driver[metric], label=abbr, color=DRIVER_COLORS[abbr])

    car_data_drivers = {abbr: session.laps.pick_drivers(abbr).pick_laps(lap).get_car_data().add_distance() for abbr in drivers_abbrs}
    
    # random driver needed for the plots
    random_data = list(car_data_drivers.values())[0]

    circuit_info = session.get_circuit_info()

    # Draw vertical dotted lines at each corner that range from slightly below the
    # minimum speed to slightly above the maximum speed.
    v_min = random_data['Speed'].min()
    v_max = random_data['Speed'].max()
    ax.vlines(x=circuit_info.corners['Distance'], ymin=v_min-20, ymax=v_max+20,
            linestyles='dotted', colors='grey')

    # Plot the corner number just below each vertical line.
    # For corners that are very close together, the text may overlap. A more
    # complicated approach would be necessary to reliably prevent this.
    for _, corner in circuit_info.corners.iterrows():
        txt = f"{corner['Number']}{corner['Letter']}"
        ax.text(corner['Distance'], v_min-30, txt,
                va='center_baseline', ha='center', size='small')


    # Find DRS zones for each driver and plot box with their color
    for abbr in drivers_abbrs:

        # Load car data for specific driver
        car_data = car_data_drivers[abbr]

        # Locate DRS zones
        drs_zones = get_drs_zones(car_data)

        # Highlight specific distance ranges with a green transparent box
        for drs_zone in drs_zones:
            start, end = drs_zone
            ax.axvspan(start, end, color=DRIVER_COLORS[abbr], alpha=0.15)

    # Adding labels and legend
    ax.set_xlabel('Distance (m)')
    ax.set_ylabel(f'{metric} Input')
    ax.set_title(f'{metric} Comparison Between Drivers {drivers_abbrs} | Lap {lap} | {event}')
    ax.legend()
    
    return fig


# Improve naming
@register_function
def fastest_driver_freq_plot(year: int=2024):
    """
    Plots the count of fastest laps for every driver who achieved at least one fastest lap in a given season.

    Parameters:
        year (int): The season's year. Defaults to 2024.
    """

    # Enable fastf1 cache
    fastf1.Cache.enable_cache('.cache/fastf1')  # Create a cache folder for faster loading

    schedule = get_schedule_until_now(2024)
    events = list(schedule['EventName'])

    driver_list = []

    for event in events:
        driver, _, _ = get_fastest_lap_time_result(event)

        driver_list.append(driver)

    # Create a frequency dictionary
    fastest_driver_freq = dict(Counter(driver_list))

    fastest_driver_freq = pd.DataFrame(fastest_driver_freq.items(), columns=['Driver', 'Frequency'])
    fastest_driver_freq = fastest_driver_freq.sort_values(by='Frequency', ascending=False)

    # Extract keys (drivers) and values (frequencies)
    drivers = list(fastest_driver_freq['Driver'])
    counts = list(fastest_driver_freq['Frequency'])

    # Create a matplotlib figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Generate the list of colors for the bars based on the DRIVER_COLORS mapping
    bar_colors = [DRIVER_COLORS[driver] for driver in drivers]

    # Plot the bar chart with the custom colors
    ax.bar(drivers, counts, color=bar_colors)
    
    # Add labels and title
    ax.set_xlabel('Drivers')
    ax.set_ylabel('Frequency')
    ax.set_title('Frequency of Fastest Laps')
    
    return fig



# Extend this by:
#  - Having specific color for each driver (need to think for global solution, that is other plots)
#  - Making it for multiple teams (pairs of drivers) (only for the second plot might be problematic)
@register_function
def compare_quali_season(drivers_list: list):
    """
    Compares the qualifying performance of pairs of drivers across a season.

    Parameters:
        drivers_list (list): A list of driver names or abbreviations to compare.
    """

    # Enable fastf1 cache
    fastf1.Cache.enable_cache('.cache/fastf1')  # Create a cache folder for faster loading

    # Load and preprocess the data
    quali_df = pd.read_csv('data/gps_2024_season_quali.csv')
    quali_df = quali_df[['Abbreviation', 'Position', 'EventName']]
    quali_df = quali_df.rename(columns={'Abbreviation': 'Driver'})
    quali_df = quali_df[quali_df['Driver'].isin(drivers_list)]

    # Plot with specific DRIVER_COLORS
    bar_colors = {driver: DRIVER_COLORS[driver] for driver in drivers_list}
    bar_colors[drivers_list[1]] = lighten_color(bar_colors[drivers_list[1]], amount=0.2)

    # Plot 1: Line plot for qualifying positions
    fig1, ax1 = plt.subplots(figsize=(10, 6))

    # Adding markers to the plot
    for driver, data in quali_df.groupby('Driver'):
        short_event = data['EventName'].apply(lambda x: x.replace('Grand Prix', ''))
        ax1.plot(short_event, data['Position'], label=driver, marker='o', color=bar_colors[driver])

    # Adding labels and legend
    ax1.set_xlabel('Grand Prix')
    ax1.set_ylabel('Position')
    ax1.set_title('Qualifying Position vs Event for Different Drivers')
    ax1.legend(title='Driver')
    ax1.invert_yaxis()  # Reverse y-axis so 1st position is at the top
    ax1.set_xticks(range(len(short_event)))
    ax1.set_xticklabels(short_event, rotation=90, ha='right')  # Adjust rotation
    ax1.set_ylim([20.5, 0.5])
    ax1.set_yticks([1, 5, 10, 15, 20])
    ax1.grid()

    # Automatically adjust layout for the first plot
    fig1.tight_layout()

    # Pivot the DataFrame for easier comparison by EventName
    pivoted = quali_df.pivot(index='EventName', columns='Driver', values='Position')
    pivoted['BetterDriver'] = pivoted.idxmin(axis=1)  # Find the driver with the minimum position

    # Create a DataFrame with counts of "BetterDriver"
    outquali_df = pd.DataFrame(pivoted['BetterDriver'].value_counts())
    outquali_df = outquali_df.reset_index()
    outquali_df.columns = ['BetterDriver', 'Count']

    # Extract drivers and counts
    drivers, counts = outquali_df['BetterDriver'], outquali_df['Count']

    # Plot 2: Horizontal bar chart for driver performance counts
    fig2, ax2 = plt.subplots(figsize=(10, 2))

    # Plot left and right bars with ax.barh
    ax2.barh(0, -counts[0], color=bar_colors[drivers_list[0]] , label=f'{drivers[0]} Count', align='center')
    ax2.barh(0, counts[1], color=bar_colors[drivers_list[1]] , label=f'{drivers[1]} Count', align='center')

    # Add a vertical line at center
    ax2.axvline(0, color='white', linewidth=1)

    # Add labels and title
    ax2.set_xlabel('Count')
    ax2.set_title('Driver Performance: Left and Right Counts')

    # Add legend
    ax2.legend()

    # Remove y-axis ticks (since there's only one row)
    ax2.set_yticks([])
    ax2.set_yticklabels([])

    # Remove x-axis ticks
    ax2.set_xticks([])

    # Adjust the x-axis limits for better spacing
    ax2.set_xlim(-max(counts)-2, max(counts)+2)

    # Add count values inside the bars
    ax2.text(-counts[0]/2, 0, str(counts[0]), color='black', ha='center', va='center')
    ax2.text(counts[1]/2, 0, str(counts[1]), color='black', ha='center', va='center')

    # Automatically adjust layout for the second plot
    fig2.tight_layout()
    
    return [fig1, fig2]

@register_function
def laptime_plot(event:str, drivers_abbrs: list, year: int=2024):
    """
    Generates a line plot of the lap times of specified drivers in a specific Grand Prix.
    
    Parameters:
        event (str): The specific Grand Prix or event (e.g., 'Monaco Grand Prix').
        drivers_abbrs (list): The names of the drivers to compare.
        year (int): The Grand Prix's year.
    """
    # Enable fastf1 cache
    fastf1.Cache.enable_cache('.cache/fastf1')  # Create a cache folder for faster loading

    # Enable Matplotlib patches for plotting timedelta values and load
    # FastF1's dark color scheme
    fastf1.plotting.setup_mpl(mpl_timedelta_support=True, misc_mpl_mods=False,
                          color_scheme=None)


    # Load a specific race session
    session = fastf1.get_session(year, event, 'R')  # 'R' indicates the race; can also use 'Q', 'FP1', 'FP2', 'FP3'
    session.load()

    # Retrieve the correct event name from fastf1
    event_name = session.session_info['Meeting']['Name']

    fig, ax = plt.subplots(figsize=(12, 8))

    # Loop over each driver abbreviation
    for driver in drivers_abbrs:
        driver_laps = session.laps.pick_drivers(driver).pick_quicklaps().reset_index()

        sns.lineplot(
        data=driver_laps,
        x="LapNumber",
        y="LapTime",
        ax=ax,
        label=driver,  # Label each driver's data for the legend
        color=DRIVER_COLORS[driver],
        linewidth=2,  # Adjust the thickness of the lines
        marker='o'
        )


        # sns.scatterplot(
        #     data=driver_laps,
        #     x="LapNumber",
        #     y="LapTime",
        #     ax=ax,
        #     # hue="Compound",
        #     # palette=fastf1.plotting.get_compound_mapping(session=session),
        #     s=80,
        #     linewidth=0,
        #     legend='auto',
        #     label=driver,  # Label each driver's data for the legend
        #     color=DRIVER_COLORS[driver]
        # )


    ax.set_xlabel("Lap Number")
    ax.set_ylabel("Lap Time")
    ax.invert_yaxis()

    ax.set_title(f"Laptimes in the {year} {event_name}")
    ax.grid(which='major', axis='both')
    sns.despine(left=True, bottom=True)

    ax.legend(title="Driver")
    fig.tight_layout()

    return fig

# Might need to also check the classified grid starting position if needed?
@register_function
def get_qualifying_results(event: str):
    """
    Retrieves the qualifying results for a specific Grand Prix.

    Parameters:
        event (str): Name of the Grand Prix.
    """

    # Enable fastf1 cache
    fastf1.Cache.enable_cache('.cache/fastf1')  # Create a cache folder for faster loading

    # Enable Matplotlib patches for plotting timedelta values
    fastf1.plotting.setup_mpl(mpl_timedelta_support=True, misc_mpl_mods=False,
                            color_scheme=None)


    session = fastf1.get_session(2024, event, 'Q')
    session.load()

    list_fastest_laps = list()

    for drv in session.drivers:
        drvs_fastest_lap = session.laps.pick_drivers(drv).pick_fastest()
        list_fastest_laps.append(drvs_fastest_lap)
    
    fastest_laps = Laps(list_fastest_laps) \
        .sort_values(by='LapTime') \
        .reset_index(drop=True)

    pole_lap = fastest_laps.pick_fastest()
    fastest_laps['LapTimeDelta'] = fastest_laps['LapTime'] - pole_lap['LapTime']

    team_colors = list()
    for index, lap in fastest_laps.iterlaps():
        color = fastf1.plotting.get_team_color(lap['Team'], session=session)
        team_colors.append(color)


    fig, ax = plt.subplots()
    ax.barh(fastest_laps.index, fastest_laps['LapTimeDelta'],
            color=team_colors, edgecolor='grey')
    ax.set_yticks(fastest_laps.index)
    ax.set_yticklabels(fastest_laps['Driver'])

    # show fastest at the top
    ax.invert_yaxis()

    # draw vertical lines behind the bars
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, which='major', linestyle='--', color='black', zorder=-1000)


    lap_time_string = strftimedelta(pole_lap['LapTime'], '%m:%s.%ms')

    ax.set_title(f"{session.event['EventName']} {session.event.year} Qualifying\n"
                f"Fastest Lap: {lap_time_string} ({pole_lap['Driver']})")

    return fig


@register_function
def laptime_distribution_plot(event: str, year: int=2024):
    """
    Generates a violin plot of the lap time distributions for the top 10 point finishers in a specified Grand Prix.
    
    Parameters:
        event (str): The specific Grand Prix or event (e.g., 'Monaco Grand Prix').
        year (int): The Grand Prix's year.
    """
    # Enable fastf1 cache
    fastf1.Cache.enable_cache('.cache/fastf1')  # Create a cache folder for faster loading

    # Enable Matplotlib patches for plotting timedelta values and load FastF1's dark color scheme
    fastf1.plotting.setup_mpl(mpl_timedelta_support=True, misc_mpl_mods=False, color_scheme=None)

    # Load the race session
    session = fastf1.get_session(year, event, 'R')  # 'R' indicates the race
    session.load()

    # Retrieve the correct event name from fastf1
    event_name = session.session_info['Meeting']['Name'] # maybe prettier way?

    # Get all the laps for the point finishers only
    point_finishers = session.drivers[:10]
    driver_laps = session.laps.pick_drivers(point_finishers).pick_quicklaps().reset_index()

    # Get finishing order abbreviations
    finishing_order = [session.get_driver(i)["Abbreviation"] for i in point_finishers]

    # Convert timedelta to float (seconds) for proper plotting
    driver_laps["LapTime(s)"] = driver_laps["LapTime"].dt.total_seconds()

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 5))

    # Violin plot
    sns.violinplot(
        data=driver_laps,
        x="Driver",
        y="LapTime(s)",
        hue="Driver",
        inner=None,
        density_norm="area",
        order=finishing_order,
        palette=fastf1.plotting.get_driver_color_mapping(session=session)
    )

    # Swarm plot
    sns.swarmplot(
        data=driver_laps,
        x="Driver",
        y="LapTime(s)",
        order=finishing_order,
        hue="Compound",
        palette=fastf1.plotting.get_compound_mapping(session=session),
        hue_order=["SOFT", "MEDIUM", "HARD"],
        linewidth=0,
        size=4
    )

    ax.set_xlabel("Driver")
    ax.set_ylabel("Lap Time (s)")
    ax.set_title(f"{year} {event_name} Lap Time Distributions")
    sns.despine(left=True, bottom=True)

    fig.tight_layout()

    return fig


@register_function
def race_statistics(event: str, year: int=2024):
    """
    Fetches and displays basic statistics about a Formula 1 race.
    
    Parameters:
        event (str): The specific Grand Prix or event (e.g., 'Monaco Grand Prix').
        year (int): The year of the race.

    Returns:
        None
    """
    # Enable fastf1 cache
    fastf1.Cache.enable_cache('.cache/fastf1')  # Create a cache folder for faster loading

    try:
        # Load the session
        session = fastf1.get_session(year, event, 'R')
        session.load()
        
        # General information
        print(f"Race Name: {session.event.EventName}")
        print(f"Track: {session.event.Location}")
        print(f"Country: {session.event.Country}")
        print(f"Date: {session.event.Date}")
        
        # Track information
        print(f"Laps: {session.event.NumberOfLaps}")
        print(f"Track Length: {session.event.TrackLength}")
        
        # Weather information
        print(f"Weather: {session.weather_data['trackTemp'].iloc[0]:.1f}°C (track)")
        print(f"Air Temp: {session.weather_data['airTemp'].iloc[0]:.1f}°C")
        print(f"Humidity: {session.weather_data['humidity'].iloc[0]:.1f}%")
        print(f"Wind Speed: {session.weather_data['windSpeed'].iloc[0]:.1f} km/h")
    
    except Exception as e:
        print(f"An error occurred: {e}")
