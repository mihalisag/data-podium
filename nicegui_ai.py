import os
import json
import re

import general_utils
from general_utils import *

from openai import OpenAI
from nicegui import ui, run # , native

from dotenv import load_dotenv

# # Multiprocessing freeze
# import multiprocessing
# multiprocessing.freeze_support()

def initialize_client(platform="groq"):
    """
        Initialize the OpenAI client based on the specified platform.
    """
    if platform.lower() == "openai":
        load_dotenv('openai.env')
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        model = "gpt-4o-mini"
    elif platform.lower() == "groq":
        load_dotenv('groq.env')
        client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=os.getenv("GROQ_API_KEY"))
        model = "llama3-70b-8192"
    else:
        raise ValueError(f"Unsupported platform: {platform}")

    return client, model


platform = "groq"  # Change to "openai" for OpenAI platform
client, model = initialize_client(platform)
print(f"Initialized client for platform: {platform}, using model: {model}")



prime_template = """
As an AI assistant, analyze the user's input and select the most suitable function and parameters from the list of available functions below. Match the user's intent to the corresponding function and fill in its parameters using information extracted from the input and the provided memory. Please structure your response **only** as JSON. Do not explain your decision, just output the JSON.

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


def render_dataframe(df):
    """Helper to render a pandas DataFrame as a table."""
    df_serializable = df.copy()

    # print(df_serializable.dtypes)

    for col in df_serializable.select_dtypes(include=['datetime', 'datetimetz']):
        df_serializable[col] = df_serializable[col].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # workaround to handle case of season schedule function - can't fix it, shows TypeError
    if 'EventDate' not in df.columns:
        ui.table.from_pandas(df.astype(str)).style("width: 100%; display: flex; justify-content: center; align-items: center;")

    else:
        ui.table(
            columns=[{'field': col, 'title': col} for col in df_serializable.columns],
            rows=df_serializable.to_dict('records'),
        ).style("width: 100%; display: flex; justify-content: center; align-items: center;")


def get_response(user_input):

    # Generate the prompt
    complete_prime = prime_template.format(
        user_input=user_input,
        functions_text=functions_text
    )


    # Call OpenAI API
    response = client.chat.completions.create(
        model=model,
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


            # Extract function and parameters
            function_name = response_data["function"]
            params = response_data["params"]

            # session = fastf1.get_session(params["year"], params["event"], 'R')  
            # session.load(weather=False, messages=False)

            ui.label(f"Parameters: {params}")

            # params = dict()
            # params["session"] = session

            ui.label(f"Function: {function_name}")

            # Call the selected function dynamically
            if function_name in function_dispatcher:
                result = function_dispatcher[function_name](**params)  # Pass params as kwargs

                if isinstance(result, str):  # If the result is text
                    ui.label(result)
                elif isinstance(result, pd.DataFrame):
                    render_dataframe(result)

        except json.JSONDecodeError:
            ui.error("Failed to parse response.")
            ui.write("### Raw Response:")
            ui.write(response)



@ui.page('/')
def main_page():
    
    ui.page_title("🏎️ Data Podium")
    ui.colors(primary='#FF1821')#, secondary='#53B689', accent='#111B1E', positive='#53B689')

    # global result_placeholder, dynamic_ui_placeholder  # Declare as global variables

    user_input = ui.input(label="Enter your question here:").style('width: 300px;')
  
    ui.button("Get Answer", on_click=lambda: get_response(user_input.value))
            
    result_placeholder = ui.column().style("width: 100%;") # Placeholder for the rendered result

ui.run(host='127.0.0.1', port=8080)
