import requests
import sys

# Define the endpoint URL
url = "http://127.0.0.1:5000/v1/internal/logits"
testing_key = 'Password12344321'
AUTH = testing_key

# Define the headers
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {AUTH}"
}

QUESTION = "what do the dead know?"
TEXT = """
The dead know nothing.
"""

# Define the JSON payload
data = {
    "prompt": f"""[INST]determine whether the bible text is applicable for answering the provided question[/INST]
[QUESTION]{QUESTION}[/QUESTION]
[TEXT]{TEXT}[/TEXT]
Answer (Must be 'yes' or 'no' without quotes):""",
    "custom_token_bans": ','.join(str(i) for i in range(256*256) if i not in [5081, 708]),
    "top_logits": 2,
    "add_bos_token": True,
    "use_samplers": True
}

try:
    # Send POST request
    response = requests.post(url, headers=headers, json=data, verify=False)

    # Check if request was successful
    if response.status_code == 200:
        print("Response.")
    else:
        print(f"Request failed with status code: {response.status_code}")
        print(response.content)

except requests.exceptions.RequestException as e:
    print(f"Request failed: {e}")
    sys.exit(1)

# Extract probabilities for "yes" and "no"
response_json = response.json()

yes_raw = response_json['▁yes']
no_raw = response_json['▁no']
prob_yes = yes_raw
prob_no = no_raw

# Normalize the probabilities
total_prob = prob_yes + prob_no
prob_yes /= total_prob
prob_no /= total_prob

# Example usage
print(f"Question: {QUESTION}")
print(f"Probability of Yes: {prob_yes:.4f}")
print(f"Probability of No: {prob_no:.4f}")