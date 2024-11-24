import streamlit as st
from gpt4all import GPT4All
import json
import re

from winner import winner_func

# Path to the downloaded model
model_path = "/Users/mike/Library/Application Support/nomic.ai/GPT4All/airoboros-m-7b-3.1.2.Q4_0.gguf" # macos
# model_path = "/home/michail/.local/share/nomic.ai/GPT4All/airoboros-m-7b-3.1.2.Q4_0.gguf" # linux

# Load the GPT4All model and create a persistent session
gpt_model = GPT4All(model_name=model_path)


# Function definitions
functions = {
    "grand_prix_statistics": {
        "description": "This tool finds statistics of a grand prix",
        "params": {
            "action": "The operation we want to perform on the data, such as 'winner', 'ranking', 'find_fastest_lap_time', etc",
            "filters": {
                "event": "The specific Grand Prix or event (e.g., 'Monaco Grand Prix', 'Silverstone 2023')",
                "driver": "The name of the racing driver (e.g., 'Max Verstappen', 'Lewis Hamilton')",
                "session": "The type of session (e.g., 'race', 'qualifying', 'practice')"
            }
        }
    }
}

# Streamlit app setup
st.title("F1 Assistant")
st.write("Ask questions, and get answers powered by GPT4All.")

# Template for priming
prime_template = """
As an AI assistant, analyze the user's input and select the most suitable function and parameters from the list of available functions below. Extract relevant information for each parameter and structure your response in JSON format.

Input: {user_input}

Available functions: 
{functions_text}
"""

# Convert functions to JSON-like text
functions_text = json.dumps(functions, indent=2)

# User input
user_input = st.text_area("Enter your question here:")

# if st.button("Get Answer"):
#     if user_input:
#         # Generate the prime dynamically
#         complete_prime = prime_template.format(user_input=user_input, functions_text=functions_text)
        
#         # Generate response
#         response = gpt_model.generate(complete_prime, max_tokens=512)
        
#         try:
#             # Parse response as JSON
#             response_data = json.loads(response)
#             st.write("### Parsed Response:")
#             st.json(response_data)
#         except json.JSONDecodeError:
#             st.write("### Raw Response:")
#             st.write(response)
#     else:
#         st.write("Please enter a question.")


if st.button("Get Answer"):
    if user_input:
        # Generate the prompt dynamically
        complete_prime = prime_template.format(user_input=user_input, functions_text=json.dumps(functions, indent=2))
        
        # Generate response
        response = gpt_model.generate(complete_prime, max_tokens=512)

        # Extract JSON substring using regex - we only want the JSON part
        match = re.search(r"{.*}", response, re.DOTALL)
        if match:
            response = match.group(0)  # Get the JSON part
        
        # print(response)

        try:
            # Parse response as JSON
            response_data = json.loads(response)
            st.write("### Parsed Response:")
            st.json(response_data)

            # Extract multiple filters for processing
            filters = response_data["params"]["filters"]
            action = response_data["params"]["action"]
            event = filters.get("event", "Unknown Event")
            driver = filters.get("driver", "Unknown Driver")
            session = filters.get("session", "Unknown Session")

            st.write(f"Event: {event}")
            st.write(f"Driver: {driver}")
            st.write(f"Session: {session}")

            if action == 'winner':
                winner_sentence = winner_func(event)
                st.write(winner_sentence)


        except json.JSONDecodeError:
            st.write("### Raw Response:")
            st.write(response)
