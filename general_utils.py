import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import fastf1
import difflib # For matching similar strings
from datetime import datetime
from collections import Counter


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
    
    Returns:
    - str: Lightened color as HEX code
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


def get_schedule_until_now(year: int=2024):
    '''
        Filters schedule of races until today
    '''

    # Retrieve the event schedule for the specified season
    schedule = fastf1.get_event_schedule(year)

    # Filter sessions until current date
    schedule = schedule[schedule['EventDate'] <= datetime.today()]

    # Do not need the preseason results
    schedule = schedule[schedule['EventName'] != 'Pre-Season Testing']

    return schedule



def get_reaction_time(event:str, drivers_list: list, speed: float):
    '''
        Get reaction time of drivers to reach a specific speed in the race start
    '''

    year = 2024

    session = fastf1.get_session(year, event, 'R')  # 'R' indicates the race; can also use 'Q', 'FP1', 'FP2', 'FP3'
    session.load()

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

    sorted_drivers = reaction_df['Driver']
    sorted_reaction_times = reaction_df['ReactionTime']

    # Create a matplotlib figure
    fig, ax = plt.subplots(figsize=(5, 3))

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
    y_margin = (max_reaction - min_reaction) * 0.5  # 10% of the range

    # Apply the y-axis limits
    ax.set_ylim(min_reaction - y_margin, max_reaction + y_margin)

    ax.set_title('Reaction time of drivers')

    plt.show()

    # return driver_reaction_dict
    


def get_fastest_lap_time_result(event: str):
    '''
        Find the fastest lap of a Grand Prix (expand for qualifying ?? or not)
        and returns variables
    '''
    year = 2024
    session = fastf1.get_session(year, event, 'R')  # 'R' indicates the race; can also use 'Q', 'FP1', 'FP2', 'FP3'
    session.load()

    fastest_lap = session.laps.pick_fastest()

    driver = fastest_lap.Driver
    lap_num = int(fastest_lap.LapNumber)
    lap_time = format_timedelta(fastest_lap.LapTime)
    
    return driver, lap_num, lap_time
    

def get_fastest_lap_time_print(event: str):
    '''
        Find the fastest lap of a Grand Prix (expand for qualifying ?? or not)
        and prints result
    '''

    driver, lap_num, lap_time = get_fastest_lap_time_result(event)
    sentence = f'Driver {driver} had the fastest lap time of {lap_time} at lap {lap_num}.'

    return sentence


def get_season_podiums():
    '''
        Returns podiums of the season
    '''

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

    # Create bar chart
    plt.figure(figsize=(8, 5))  # Optional: Set figure size
    plt.bar(season_podiums_df["Driver"], season_podiums_df["count"], color=bar_colors)

    # Add labels and title
    plt.xlabel("Driver", fontsize=12)
    plt.ylabel("Podium count", fontsize=12)
    plt.title("Podiums for each driver", fontsize=14)

    # Set y-axis ticks to increment one by one
    plt.yticks(range(0, max(season_podiums_df["count"]) + 1, 1))  # Start at 0, go to max count, step by 1

    # Show the plot
    plt.show()



def get_race_results(event: str) -> pd.DataFrame:
    """
    Retrieves the race results for a specific Grand Prix.

    Parameters:
        event_name (str): Name of the Grand Prix.

    Returns:
        pd.DataFrame: DataFrame containing race results for the specified event.
    """
    # Retrieve the correct event name from fastf1
    event_name = fastf1.get_event(2024, event)['EventName']

    # Load and preprocess results data
    results_df = (
        pd.read_csv('data/gps_2024_season_results.csv')
        .rename(columns={'Abbreviation': 'Driver'})
        .loc[lambda df: df['EventName'] == event_name]
        .drop(columns=['EventName'])
    )

    results_df = results_df[['Driver', 'ClassifiedPosition', 'Status']]

    return results_df


def get_winner(event: str) -> str:
    '''
    Finds the winner of a specific Grand Prix.

    Parameters:
        event (str): Name of the Grand Prix.

    Returns:
        str: Sentence indicating the winner of the specified Grand Prix.
    '''

    # Retrieve the correct event name from fastf1
    event_name = fastf1.get_event(2024, event)['EventName']

    # Get race results using the modular function
    race_results_df = get_race_results(event_name)

    # Identify the winner (driver with ClassifiedPosition == '1')
    winner = race_results_df.loc[race_results_df['ClassifiedPosition'] == '1', 'Driver'].iloc[0]

    return f"Driver {winner} won the {event_name}"


# Old implementation
def get_positions_during_race(event: str, drivers_abbrs: list=[]):
    '''
        Returns a matplotlib figure of the positions of drivers
        in a specific Grand Prix. Can filter drivers (optional)
    '''
    pos_df = pd.read_csv("data/gps_2024_season_laps.csv")
    pos_df = pos_df[['Driver', 'LapNumber', 'Stint', 'Position', 'EventName']]
    pos_df = pos_df[pos_df['EventName'] == event]

    if drivers_abbrs:
        # Filter DataFrame for specific drivers
        pos_df = pos_df[pos_df['Driver'].isin(drivers_abbrs)]

    # Create a matplotlib figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Group by 'Driver' and plot each driver's data
    for driver, data in pos_df.groupby('Driver'):
        ax.plot(data['LapNumber'], data['Position'], label=driver, color=DRIVER_COLORS[driver])

    # Adding labels and legend
    ax.set_xlabel('Lap')
    ax.set_ylabel('Position')
    ax.set_title(f'Position vs Lap for Different Drivers | {event}')
    ax.legend(title='Driver')
    ax.invert_yaxis()  # Optional: Reverse y-axis so 1st position is at the top
    
    return fig


# def get_positions_during_race(event: str):
#     '''
#         Returns a matplotlib figure of the positions of drivers
#         in a specific Grand Prix. Inspired by fastf1 example.
#     '''

#     year = 2024
#     session = fastf1.get_session(year, event, 'R')
#     session.load(telemetry=False, weather=False)

#     # Create a matplotlib figure
#     fig, ax = plt.subplots(figsize=(10, 6))

#     for drv in session.drivers:
#         drv_laps = session.laps.pick_driver(drv)
#         abb = drv_laps['Driver'].iloc[0]

#         style = fastf1.plotting.get_driver_style(identifier=abb,
#                                                 style=['color', 'linestyle'],
#                                                 session=session)

#         ax.plot(drv_laps['LapNumber'], drv_laps['Position'],
#                 label=abb, **style)
    
#     # Finalize the plot by setting y-limits that invert the y-axis so that position
#     # one is at the top, set custom tick positions and axis labels.
#     ax.set_ylim([20.5, 0.5])
#     ax.set_yticks([1, 5, 10, 15, 20])
#     ax.set_xlabel('Lap')
#     ax.set_ylabel('Position')
#     ax.set_title(f'Position vs Lap for Different Drivers | {event}')
#     ax.legend(title='Driver')

#     return fig


# Expand for multiple laps (of same driver) ?
def compare_metric(year: int, event: str, session_type: str, drivers_abbrs: list, metric: str, lap:int):
    '''
    Compares the telemetry of a specific metric for a specific lap from a list of drivers' abbreviations.

    Parameters:
    year:
    event:
    session_type:
    drivers_abbrs (list): A list of driver abbreviations to compare.
    metric (str): The telemetry metric to compare-plot (e.g., 'Speed', 'Throttle').
    lap (int): The specific lap number to use for comparison.
    
    Returns:
    None. Displays a plot of the telemetry attribute comparison.
    '''
    year = 2024

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
    fig, ax = plt.subplots(figsize=(12, 6))

    for abbr, telemetry_driver in telemetry_drivers.items():
        ax.plot(telemetry_driver['Distance'], telemetry_driver[metric], label=abbr)

    # Adding labels and legend
    ax.set_xlabel('Distance (m)')
    ax.set_ylabel(f'{metric} Input')
    ax.set_title(f'{metric} Comparison Between Drivers {drivers_abbrs} | Lap {lap} | {event}')
    ax.legend()
    
    return fig


# Improve naming
def fastest_driver_freq_plot(year: int=2024):
    '''
        Plots fastest lap count for every driver (at least one)
    '''

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
def compare_quali_season(drivers_list: list):
    '''
        Compares qualifying performance of pair of drivers across season
    '''

    # Load and preprocess the data
    quali_df = pd.read_csv('data/gps_2024_season_quali.csv')
    quali_df = quali_df[['Abbreviation', 'Position', 'EventName']]
    quali_df = quali_df.rename(columns={'Abbreviation': 'Driver'})
    quali_df = quali_df[quali_df['Driver'].isin(drivers_list)]

    # Plot with specific DRIVER_COLORS
    bar_colors = {driver: DRIVER_COLORS[driver] for driver in drivers_list}
    bar_colors[drivers_list[1]] = lighten_color(bar_colors[drivers_list[1]], amount=0.4)

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
    plt.show()

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
    plt.show()


