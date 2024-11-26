import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import fastf1
import difflib # For matching similar strings
from datetime import datetime
from collections import Counter

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


def get_winner(event: str):
    '''
        Finds the winner of a specific Grand Prix
    '''

    # Stick to fastf1-saved event format (maybe cannot use difflib due to exceptions like Italy - Emilia Romagna Grand Prix (??))
    event = fastf1.get_event(2024, event)
    event = event['EventName']

    results_df = pd.read_csv('data/gps_2024_season_results.csv')
    results_df = results_df[['Abbreviation', 'ClassifiedPosition', 'GridPosition', 'Status', 'EventName']]
    results_df = results_df[results_df['EventName'] != 'Pre-Season Testing']
    results_df = results_df.rename(columns={'Abbreviation': 'Driver'})
    results_df = results_df.drop(columns=['GridPosition'])

    race_results_df = results_df.copy()

    race_results_df = race_results_df[race_results_df['EventName'] == event]
    race_results_df = race_results_df.drop(columns=['EventName'])

    winner_df = race_results_df[race_results_df['ClassifiedPosition'] == '1']
    winner = winner_df['Driver'].values[0]

    winner_sentence = f"Driver {winner} won the {event}"

    return winner_sentence


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
        ax.plot(data['LapNumber'], data['Position'], label=driver)

    # Adding labels and legend
    ax.set_xlabel('Lap')
    ax.set_ylabel('Position')
    ax.set_title(f'Position vs Lap for Different Drivers | {event}')
    ax.legend(title='Driver')
    ax.invert_yaxis()  # Optional: Reverse y-axis so 1st position is at the top
    
    return fig



def compare_metric(year: int, event: str, session_type: str, drivers_abbrs: list, metric: str, lap:int):
    '''
    Compares the telemetry of a specific metric from a list of drivers' abbreviations.

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

    # Create the bar chart
    ax.bar(drivers, counts, color='skyblue')

    # Add labels and title
    ax.set_xlabel('Drivers')
    ax.set_ylabel('Frequency')
    ax.set_title('Frequency of Fastest Laps')
    
    return fig
