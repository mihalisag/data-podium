import requests

API_URL = "https://api-inference.huggingface.co/models/openai-community/gpt2"
headers = {f"Authorization": "Bearer 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"}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()
	

if __name__ == "__main__":
	
    output = query({
	"inputs": "Tyre wear is",
    })

    print(output)