import requests
import json

# Define the URL of the Flask server
url = "http://127.0.0.1:5000/api/openai"

# Define the data to send in the POST request
data = {
    "prompt": "Tell me a joke.",
    "max_tokens": 50,
    "temperature": 0.5
}

# Convert the data to JSON format
json_data = json.dumps(data)

# Send the POST request to the server
response = requests.post(url, headers={"Content-Type": "application/json"}, data=json_data)

# Check the response status and print the response data
if response.status_code == 200:
    print("Response from the server:")
    print(response.json())
else:
    print(f"Failed to get a valid response. Status code: {response.status_code}")
    print(response.text)
