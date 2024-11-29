# 🏎️ F1 Assistant Application

The F1 Assistant Application is a powerful data-driven tool designed to analyze Formula 1 race data. Built using Python, Streamlit, and a Large Language Model (LLM), this application enables users to explore F1 data effortlessly, compare drivers, teams, and events, and uncover exciting insights.

---

## 🚀 Features

- **Driver-Level Analysis**: Compare drivers based on speed, lap times, positions, and other metrics.
- **Team-Level Analysis**: Evaluate team performance and pit stop strategies.
- **Session Analysis**: Dive into qualifying, race, or practice session statistics.
- **Event Analysis**: Explore rankings, overtakes, fastest laps, and other race-specific details.
- **Season Comparisons**: Compare multiple drivers, teams, or events across the season.
- **Generalized Function Support**: Add new queries easily by extending modular function definitions.
  
---


## 🛠️ Technologies Used

- **Streamlit**: For creating the interactive web application.
- **Python**: Core programming language for data analysis and function calling.
- **Large Language Model (LLM)**: For natural language understanding and generating responses.
- **Pandas**: Data manipulation and processing.
- **Matplotlib/Plotly**: Visualization of graphs and statistics.

---

## 🔍 How It Works

1. **Natural Language Input**: Users type questions about F1 races (e.g., "Who won the Monaco Grand Prix?").
2. **LLM Integration**: The model parses the query and maps it to a relevant function (e.g., `get_event_winner`).
3. **Function Calling**: The appropriate function retrieves data and processes the result.
4. **Output Visualization**: Results are displayed as text, graphs, or tables in an interactive interface.

---

## 🧰 Installation Guide

### Prerequisites
- Python 3.9+
- Streamlit

### Steps
1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-username/F1-Assistant.git
   cd F1-Assistant
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the App**
   ```bash
   streamlit run app.py
   ```

4. **Access the App**
   Open your browser and go to `http://localhost:8501`.

---

## 🧪 Example Questions

Here are some sample queries the app can handle:
1. "Who won the Monaco Grand Prix?"
2. "Compare the fastest lap of Lewis Hamilton and Max Verstappen in Silverstone."
3. "Which team had the best pit stop strategy in Monza?"
4. "How consistent was Charles Leclerc's lap time in Bahrain?"

---

## 🔄 Extensibility

### Adding New Queries
To add new types of analysis:
1. **Define a New Function**
   - Add the new function in the relevant file under `functions/`.
   - Ensure it takes in appropriate arguments like `driver`, `team`, `event`, or `session`.

2. **Update the Function Mapping**
   - Add the new function name and description in the `functions` dictionary in `app.py`.

3. **Test the Query**
   - Run the app and test your new query to ensure it works as expected.

---

## 📊 Example Outputs

### Driver Comparison Example
// ![Driver Comparison Example](path-to-your-image/driver-comparison.png)

### Race Positions Visualization
// ![Race Positions Visualization](path-to-your-image/race-positions.png)

---

## 📜 License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## 🤝 Contributing

Contributions are welcome! Follow these steps:
1. Fork the repository.
2. Create a feature branch:
   ```bash
   git checkout -b feature-name
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add a meaningful message"
   ```
4. Push to your fork:
   ```bash
   git push origin feature-name
   ```
5. Submit a pull request.

---

## 🛡️ Future Roadmap

- Add support for real-time F1 data feeds.
- Expand to cover historical data for all seasons.
- Implement advanced AI-driven insights, like predicting race outcomes.
- Support voice-based queries.

---

## 🌟 Acknowledgments

- Formula 1 for the inspiration.
- ChatGPT for natural language understanding.
- FastF1.

---

## 📬 Contact

For feedback or queries, reach out to [your_email@example.com] or open an issue in the repository.