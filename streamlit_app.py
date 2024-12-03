import os
from dotenv import load_dotenv
import json
import re
import streamlit as st

import general_utils
from general_utils import *

from gpt4all import GPT4All
from openai import OpenAI


# Load environment variables from the .env file
load_dotenv()

# Initialize OpenAI client with the API key from environment variables
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
)

# # Path to the downloaded model
# # model_path = "/Users/mike/Library/Application Support/nomic.ai/GPT4All/airoboros-m-7b-3.1.2.Q4_0.gguf" # macos
# model_path = "/home/michail/.local/share/nomic.ai/GPT4All/airoboros-m-7b-3.1.2.Q4_0.gguf" # linux

# # Load the GPT4All model and create a persistent session
# gpt_model = GPT4All(model_name=model_path)


functions = functions_registry

# Streamlit app setup
st.title("F1 Assistant")
st.write("Ask questions, and get answers powered by gpt-4o-mini.")

prime_template = """
As an AI assistant, analyze the user's input and select the most suitable function and parameters from the list of available functions below. Match the user's intent to the corresponding function and fill in its parameters using information extracted from the input. Structure your response as JSON.

Input: {user_input}

Available functions:
{functions_text}
"""


# Convert functions to JSON-like text
functions_text = json.dumps(functions, indent=2)

# User input
user_input = st.text_area("Enter your question here:")


# Automatically create the function dispatcher
function_dispatcher = {
    name: func
    for name, func in globals()['general_utils'].__dict__.items()
    if name in functions_registry
}


if st.button("Get Answer"):
    if user_input:
        # Generate the prompt dynamically
        complete_prime = prime_template.format(user_input=user_input, functions_text=functions_text)

        # Call OpenAI API
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Use the desired model
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": complete_prime}
            ],
            max_tokens=512,
            temperature=0  # Use deterministic responses for function parsing
        )
        
        # Extract response content
        response_content = response.choices[0].message.content
        
        # Extract JSON substring using regex - we only want the JSON part
        match = re.search(r"{.*}", response_content, re.DOTALL)
        if match:
            response_json = match.group(0)  # Get the JSON part

            try:
                # Parse the JSON string into a dictionary
                response_data = json.loads(response_json)  # Convert to dict

                # # OPTIONAL
                # # Display the JSON result in Streamlit
                # st.json(response_data)
                # st.write('works')
                        
                # Extract function name and parameters
                function_name = response_data["function"]
                params = response_data["params"]

                # Call the selected function dynamically
                if function_name in function_dispatcher:
                    result = function_dispatcher[function_name](**params)  # Pass params as kwargs
                    
                    # Handle different result types dynamically
                    if isinstance(result, str):  # If the result is text
                        st.success(f"Answer: {result}")
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
                                    st.success(f"Answer {idx + 1}: {item}")
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

        else:
            st.write("Could not extract valid JSON from the response.")