import requests
import json

def send_message_to_server(message, session_history):
    api_url = "http://192.168.1.34:1234/v1/chat/completions"  # Replace with the server's IP address if necessary

    payload = {
        "model": "local-model",  # Adjust the model name if needed
        "messages": session_history + [{"role": "user", "content": message}]
    }

    headers = {
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(api_url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as err:
        print(f"HTTP Error: {err}")
        print(response.text)  # Provides more insight into what went wrong
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Start an interactive chat session
session_history = []
while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        break

    response = send_message_to_server(user_input, session_history)
    if response and response.get("choices"):
        assistant_message = response["choices"][0]["message"]["content"]
        print("Assistant:", assistant_message)
        session_history.append({"role": "user", "content": user_input})
        session_history.append({"role": "assistant", "content": assistant_message})

print("Chat session ended.")
