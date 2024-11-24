import pandas as pd


def winner_func(event):
    '''
        Finds the winner of a specific Grand Prix
    '''

    results_df = pd.read_csv('data/gps_2024_season_results.csv')

    results_df = results_df[['Abbreviation', 'ClassifiedPosition', 'GridPosition', 'Status', 'EventName']]
    results_df = results_df[results_df['EventName'] != 'Pre-Season Testing']
    results_df = results_df.rename(columns={'Abbreviation': 'Driver'})

    races_df = results_df.copy()

    races_df = races_df.drop(columns=['GridPosition'])

    race_results_df = races_df.copy()

    # event = 'Bahrain Grand Prix'

    race_results_df = race_results_df[race_results_df['EventName'] == event]
    race_results_df = race_results_df.drop(columns=['EventName'])

    winner_df = race_results_df[race_results_df['ClassifiedPosition'] == '1']

    winner_sentence = f"Driver {winner_df['Driver'].values[0]} won the {event}"

    # print(winner_sentence)

    return winner_sentence