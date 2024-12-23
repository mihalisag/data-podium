import os
import json
import re
import streamlit as st
from dotenv import load_dotenv

import general_utils
from general_utils import *

from openai import OpenAI

st.title("🏎️ F1 Assistant")
st.write("Ask questions, and get answers powered by gpt-4o-mini.")

grand_prix_by_year = {}
YEARS = range(2018, 2025)

for year in YEARS:
    year_event_names = list(get_schedule_until_now(year)['EventName'])
    grand_prix_by_year[year] = year_event_names

# Year selection
selected_year = st.selectbox("Select a Year:", YEARS)

# Grand Prix selection based on selected year
grand_prix_list = [*grand_prix_by_year[selected_year], 'Season']
selected_gp = st.selectbox(f"Select a Grand Prix (or whole season):", grand_prix_list)

# st.write(f"You selected: {selected_gp} from {selected_year}")

# # -- OPENAI --
# Load environment variables from the .env file
# load_dotenv()

# # Initialize OpenAI client with the API key from environment variables
# client = OpenAI(
#     api_key=os.getenv("OPENAI_API_KEY"),
# )

# MODEL = "gpt-4o-mini"


# -- GROQ --
load_dotenv('groq.env')

client =OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY")
)

MODEL = "llama3-8b-8192"

# Set a default model
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = MODEL

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if 'memory' not in st.session_state:
    st.session_state['memory'] = {
        "race_name": selected_gp,
        "year": selected_year,
        # "driver_names": []
    }

# function_examples = {
#     "get_winner": "Who won the 2023 Monaco Grand Prix?",
#     "compare_metric": "Compare throttle input of Verstappen and Hamilton in the Spanish Grand Prix, laps 15, 16, 17.",
#     "get_fastest_lap_time_result": "What was the fastest lap in the 2023 Silverstone GP?",
# }

# for func, example in function_examples.items():
#     st.write(f"- {example}")

# # old (working with openai)
# prime_template = """
# As an AI assistant, analyze the user's input and select the most suitable function and parameters from the list of available functions below. Match the user's intent to the corresponding function and fill in its parameters using information extracted from the input. Structure your response as JSON.

# Input: {user_input}

# Available functions:
# {functions_text}
# """

# prime_template = """
# As an AI assistant, analyze the user's input and select the most suitable function and parameters from the list of available functions below. Match the user's intent to the corresponding function and fill in its parameters using information extracted from the input. Please structure your response **only** as JSON. Do not explain your decision, just output the JSON.

# Input: {user_input}

# Available functions:
# {functions_text}
# """

prime_template = """
As an AI assistant, analyze the user's input and select the most suitable function and parameters from the list of available functions below. Match the user's intent to the corresponding function and fill in its parameters using information extracted from the input and the provided memory. Please structure your response **only** as JSON. Do not explain your decision, just output the JSON.

Memory:
{memory}

Input: {user_input}

Available functions:
{functions_text}
"""


functions = functions_registry

# Convert functions to JSON-like text
functions_text = json.dumps(functions, indent=2)

# Automatically create the function dispatcher
function_dispatcher = {
    name: func
    for name, func in globals()['general_utils'].__dict__.items()
    if name in functions_registry
}


if st.button('Overview'):
    statistics = race_statistics(selected_gp, selected_year)
    st.write(statistics)

    # # Basic race statistics
    # for idx, item in enumerate(statistics):
    #     if len(item) >= 1:
    #         st.write(f"{item}\n")


user_input = st.text_area("Enter your question here:")

    
if st.button("Get Answer"):
    if user_input:
        # st.write("Debug - Memory:", st.session_state['memory'])
        # st.write("Debug - User Input:", user_input)
        # st.write("Debug - Functions Text:", functions_text)

        # st.write(prime_template)

        # Generate the prompt
        complete_prime = prime_template.format(
            memory=json.dumps(st.session_state['memory'], indent=2),
            user_input=user_input,
            functions_text=functions_text
        )

        # st.write(f"Complete prime: {complete_prime}")

        # Call OpenAI API
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": complete_prime}
            ],
            max_tokens=256,
            temperature=0
        )
        
        # Extract response content
        response_content = response.choices[0].message.content
        
        # Extract JSON response from model
        match = re.search(r"{.*}", response_content, re.DOTALL)
        if match:
            response_json = match.group(0)
            try:
                # Parse the JSON string into a dictionary
                response_data = json.loads(response_json)

                # st.write(f"Response data: {response_data}")
                
                # # Update memory with model-provided memory
                # st.session_state['memory'] = response_data.get("memory", st.session_state['memory'])

                # Extract function and parameters
                function_name = response_data["function"]
                params = response_data["params"]

                if 'event' in params:
                    params['event'] = selected_gp

                if 'year' in params:
                    params['year'] = selected_year

                # st.write(f"Parameters: {params}")

                 # Call the selected function dynamically
                if function_name in function_dispatcher:
                    result = function_dispatcher[function_name](**params)  # Pass params as kwargs
                    
                    # Handle different result types dynamically
                    if isinstance(result, str):  # If the result is text
                        st.write(result)
                    elif isinstance(result, pd.DataFrame):  # If the result is a pandas DataFrame
                        df_height = (len(result) + 1) * 35 + 3 # workaround to avoid scrolling
                        st.dataframe(result, hide_index=True, height=df_height)  # Display the DataFrame
                    elif isinstance(result, plt.Figure):  # If the result is a Matplotlib figure
                        st.pyplot(result)  # Render the Matplotlib figure
                    elif isinstance(result, list):  # If the result is a list
                        if len(result) > 0:  # Ensure the list is not empty
                            first_item = result[0]
                            if isinstance(first_item, str):  # List of strings
                                for idx, item in enumerate(result):
                                    if len(item) >= 1:
                                        st.write(f"{item}\n")
                            elif isinstance(first_item, pd.DataFrame):  # List of DataFrames
                                for idx, item in enumerate(result):
                                    st.write(f"DataFrame {idx + 1}:")
                                    df_height = (len(item) + 1) * 35 + 3 # workaround to avoid scrolling
                                    st.dataframe(item, hide_index=True, height=df_height)  # Display the DataFrame
                            elif isinstance(first_item, plt.Figure):  # List of Matplotlib figures
                                for idx, item in enumerate(result):
                                    st.write(f"Figure {idx + 1}:")
                                    st.pyplot(item)
                            else:
                                st.warning("List contains unsupported item types.")
                                for idx, item in enumerate(result):
                                    st.write(f"Item {idx + 1}:")
                                    st.write(item)  # Fallback for unsupported list items
                        else:
                            st.info("The list is empty.")
                    else:
                        st.warning("Unexpected output type.")
                        st.write(result)  # Fallback to display the raw result

                else:
                    st.error(f"Function '{function_name}' not implemented.")
            
            except json.JSONDecodeError:
                st.error("Failed to parse response.")
                st.write("### Raw Response:")
                st.write(response)
