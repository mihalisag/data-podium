import openai

api_key = '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'

# Set the API key
openai.api_key = api_key


# Test the key with a simple request (e.g., asking for a model list)
response = openai.Model.list()

# If the request is successful, print the list of models
print("API Key is working!")
print("Models available:", response['data'])

