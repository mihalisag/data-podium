# 🏎️ Data Podium

Data Podium is a data-driven tool designed to analyse Formula 1 race data. It is built using Python with NiceGUI as the UI frontend and fetches data from the FastF1 project. This application enables users to explore Formula 1 data effortlessly, compare drivers and uncover exciting insights.


![Main Screen](/images/main_screen.png)


## Features

- **Driver-Level Analysis**: Compare drivers based on telemetry, lap times, positions, and other metrics.
- **Session Analysis**: Dive into qualifying or race statistics.
- **Event Analysis**: Explore rankings, fastest laps, and other race-specific details.
- **Race Report**: Fetches the race report found in Wikipedia
- **Season Comparisons**: Compare multiple drivers across the season.
- *Soon:* Identify and analyse battles in the track.


## Technologies Used

- **NiceGUI**: For creating the interactive web application.
- **Pandas**: Data manipulation and processing.
- **Plotly**: Visualization of graphs and statistics.
- **FastF1**: The Formula 1 data source



<!-- USAGE -->
## Usage

1. Clone the repository:
    ```sh
    git clone https://github.com/mihalisag/data-podium.git
    ```
2. Create a virtual environment and activate it
    ```sh
    python3 -m venv data-podium-env && source data-podium-env/bin/activate
    ```
3. Install Python dependencies (need Python 3.11+):
    ```sh
    cd data-podium && pip install -r requirements.txt
    ```
4. Run the application:
    ```sh
    python main.py
    ```

<!-- ROADMAP -->
## Roadmap

- Identify and analyse battles in the track.
- Show a race overview with information about race, a report and a track map.
- More advanced driver performance metrics.

<!-- LICENSE -->
## License

Distributed under the Unlicense License. See `LICENSE.txt` for more information.


## Acknowledgments

- Formula 1 for the inspiration.
- [FastF1](https://github.com/theOehrly/Fast-F1) for the data source.


