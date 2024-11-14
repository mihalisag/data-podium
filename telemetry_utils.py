import matplotlib.pyplot as plt

def telemetry_attr_comparison(session, drivers_abbrs: list, attr: str, lap:int):
    '''
    Compares the telemetry of a specific attribute from a list of drivers' abbreviations.

    Parameters:
    session: The session object containing lap and telemetry data.
    drivers_abbrs (list): A list of driver abbreviations to compare.
    attr (str): The telemetry attribute to plot (e.g., 'Speed', 'Throttle').
    lap (int): The specific lap number to use for comparison.
    
    Returns:
    None. Displays a plot of the telemetry attribute comparison.
    '''

    # Create dictionaries for laps and telemetry data.
    drivers = {abbr: session.laps.pick_drivers(abbr).pick_laps(lap) for abbr in drivers_abbrs}
    telemetry_drivers = {abbr: drivers[abbr].get_telemetry() for abbr in drivers_abbrs}

    # Plotting each driver's telemetry for the specified attribute.
    plt.figure(figsize=(12, 6))
    for abbr, telemetry_driver in telemetry_drivers.items():
        plt.plot(telemetry_driver['Distance'], telemetry_driver[attr], label=abbr)

    # Labeling and final touches.
    plt.xlabel('Distance (m)')
    plt.ylabel(f'{attr} Input')
    plt.title(f'{attr} Comparison Between Drivers {drivers_abbrs} | Lap {lap}')
    plt.legend()
    # plt.grid(True)
    plt.tight_layout()
    plt.show()