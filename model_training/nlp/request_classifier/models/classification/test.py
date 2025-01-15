import requests
response = requests.get("https://huggingface.co")
print(response.status_code)