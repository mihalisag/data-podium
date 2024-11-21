import streamlit as st
from gpt4all import GPT4All
import json

# Path to the downloaded model
model_path = "/Users/mike/Library/Application Support/nomic.ai/GPT4All/airoboros-m-7b-3.1.2.Q4_0.gguf" # macos

# Load the GPT4All model and create a persistent session
gpt_model = GPT4All(model_name=model_path)

# Function definitions
functions = {
    "grand_prix_statistics": {
        "description": "This tool finds statistics of a grand prix",
        "params": {
            "action": "The operation we want to perform on the data, such as 'winner', 'ranking', 'find_fastest_lap_time', etc",
            "filters": {
                "keyword": "The winner of a grand prix"
            }
        }
    },
    # Add more functions as needed
}

# Streamlit app setup
st.title("F1 Assistant")
st.write("Ask questions, and get answers powered by GPT4All.")

# Template for priming
prime_template = """
As an AI assistant, please select the most suitable function and parameters from the list of available functions below, based on the user's input. Provide your response in JSON format.

Input: {user_input}

Available functions: 
{functions_text}
"""

# Convert functions to JSON-like text
functions_text = json.dumps(functions, indent=2)

# User input
user_input = st.text_area("Enter your question here:")

if st.button("Get Answer"):
    if user_input:
        # Generate the prime dynamically
        complete_prime = prime_template.format(user_input=user_input, functions_text=functions_text)
        
        # Generate response
        response = gpt_model.generate(complete_prime, max_tokens=512)
        
        try:
            # Parse response as JSON
            response_data = json.loads(response)
            st.write("### Parsed Response:")
            st.json(response_data)
        except json.JSONDecodeError:
            st.write("### Raw Response:")
            st.write(response)
    else:
        st.write("Please enter a question.")
