import requests
import os
import json

lastNotificationTime = time.time() - 20

def HaPostSensorValueWithRateLimiter(value, base_url):
    global lastNotificationTime
    if (time.time() - lastNotificationTime) > 20:
        lastNotificationTime = time.time()
        HaPostSensorValue(value, base_url)

def HaPostSensorValue(value, base_url):
    """
    Sends a POST request with a specific value and attributes to a given URL.

    :param value: The state value to send in the JSON payload.
    :param base_url: The base URL for the POST request.
    :param auth_token: Optional. Authorization token. If not provided, it's fetched from the environment.
    """
    # Fetch the auth token from environment if not provided
        auth_token = os.getenv("HASSIO_TOKEN")

    # Prepare the headers
    headers = {
        "Authorization": f"Bearer {auth_token}",
        "Content-Type": "application/json"
    }

    # Prepare the JSON payload
    data = {
        "state": value,
        "attributes": {
            "friendly_name": "Cat AI detect result"
        }
    }

    # Construct the full URL
    full_url = f"{base_url}/api/states/sensor.catAIdetect"

    try:
        # Make the POST request
        response = requests.post(full_url, headers=headers, json=data)
        response.raise_for_status()  # Raises an exception for 4XX/5XX errors
        return response.json()  # Return the JSON response if successful
    except requests.RequestException as e:
        # Handle any errors that occur during the request
        return {"error": str(e)}

# Example usage
# Replace 'YOUR_BASE_URL' with your actual base URL
# print(post_cat_ai_detect_result("Micka", "http://hassio.lan:8124"))