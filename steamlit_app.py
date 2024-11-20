import streamlit as st
from gpt4all import GPT4All

# Path to the downloaded model
model_path = "/home/michail/.local/share/nomic.ai/GPT4All/airoboros-m-7b-3.1.2.Q4_0.gguf"

# Load the GPT4All model
gpt_model = GPT4All(model_name=model_path)

# Streamlit app setup
st.title("F1 Assistant")
st.write("Ask questions, and get answers powered by GPT4All.")

# User input
user_input = st.text_area("Enter your question here:")

if st.button("Get Answer"):
    if user_input:
        with gpt_model.chat_session():
            response = gpt_model.generate(user_input, max_tokens=512)
            st.write("### Answer:")
            st.write(response)
    else:
        st.write("Please enter a question.")