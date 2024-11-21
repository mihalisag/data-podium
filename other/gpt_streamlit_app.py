import streamlit as st
import openai

# Set up OpenAI API Key
openai.api_key = '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'

# Streamlit app setup
st.title("F1 Assistant")
st.write("Ask questions, and get answers powered by OpenAI's GPT-4 API.")

# Load primes
with open('first_prime.txt', 'r') as file:
    first_prime = file.read()

with open('second_prime.txt', 'r') as file:
    second_prime = file.read()

# User input
user_input = st.text_area("Enter your question here:")

# Define a function schema for GPT function calling (optional)
function_definitions = [
    {
        "name": "get_fastest_lap",
        "description": "Retrieve the driver with the fastest lap in a specific Grand Prix.",
        "parameters": {
            "type": "object",
            "properties": {
                "grand_prix_name": {
                    "type": "string",
                    "description": "The name of the Grand Prix to search for."
                }
            },
            "required": ["grand_prix_name"]
        }
    }
]

if st.button("Get Answer"):
    if user_input:
        # Combine primes with user input
        full_prompt = first_prime + user_input + second_prime

        # ChatGPT API call
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": first_prime + second_prime},
                {"role": "user", "content": user_input},
            ],
            functions=function_definitions,
            function_call="auto",  # Enable automatic function calling
        )

        # Display the response or handle function call
        if "choices" in response and response["choices"][0]["message"].get("function_call"):
            function_call = response["choices"][0]["message"]["function_call"]
            function_name = function_call["name"]
            arguments = function_call["arguments"]

            # Example of handling a function call output
            st.write(f"Function called: {function_name}")
            st.write(f"With arguments: {arguments}")
        else:
            answer = response["choices"][0]["message"]["content"]
            st.write("### Answer:")
            st.write(answer)
    else:
        st.write("Please enter a question.")
