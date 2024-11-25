import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import fastf1

def get_winner(event: str):
    '''
        Finds the winner of a specific Grand Prix
    '''

    results_df = pd.read_csv('data/gps_2024_season_results.csv')
    results_df = results_df[['Abbreviation', 'ClassifiedPosition', 'GridPosition', 'Status', 'EventName']]
    results_df = results_df[results_df['EventName'] != 'Pre-Season Testing']
    results_df = results_df.rename(columns={'Abbreviation': 'Driver'})
    results_df = results_df.drop(columns=['GridPosition'])

    race_results_df = results_df.copy()

    race_results_df = race_results_df[race_results_df['EventName'] == event]
    race_results_df = race_results_df.drop(columns=['EventName'])

    winner_df = race_results_df[race_results_df['ClassifiedPosition'] == '1']

    winner_sentence = f"Driver {winner_df['Driver'].values[0]} won the {event}"

    return winner_sentence


def positions_during_race(event: str):
    '''
        Returns a matplotlib figure of the positions of drivers
        in a specific Grand Prix.
    '''
    pos_df = pd.read_csv("data/gps_2024_season_laps.csv")
    pos_df = pos_df[['Driver', 'LapNumber', 'Stint', 'Position', 'EventName']]
    pos_df = pos_df[pos_df['EventName'] == event]

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
    Compares the telemetry of a specific attribute from a list of drivers' abbreviations.

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

    session = fastf1.get_session(year, event, session_type)  # 'R' indicates the race; can also use 'Q', 'FP1', 'FP2', 'FP3'
    session.load()

    # Create dictionaries for laps and telemetry data.
    drivers = {abbr: session.laps.pick_drivers(abbr).pick_laps(lap) for abbr in drivers_abbrs}
    telemetry_drivers = {abbr: drivers[abbr].get_telemetry() for abbr in drivers_abbrs}

    # Create a matplotlib figure
    fig, ax = plt.subplots(figsize=(12, 6))

    for abbr, telemetry_driver in telemetry_drivers.items():
        ax.plot(telemetry_driver['Distance'], telemetry_driver[metric], label=abbr)

    # Adding labels and legend
    ax.set_xlabel('Distance (m)')
    ax.set_ylabel(f'{metric} Input')
    ax.set_title(f'{metric} Comparison Between Drivers {drivers_abbrs} | Lap {lap}')
    ax.legend()
    
    return fig