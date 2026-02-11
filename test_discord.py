import os
from dotenv import load_dotenv
import requests
import json

# 1. Load Environment Variables
load_dotenv()
url = os.getenv("discord_webhook_url")

print(f"--- Discord Connection Test ---")
print(f"1. Loaded URL from .env: {url}")

if not url or "discord.com" not in url:
    print("❌ Error: URL looks invalid or empty. Check your .env file.")
    exit()

# 2. Try to Send a Message
payload = {
    "content": "🔔 **Ping!** This is a test message from your Build-A-Bot."
}

try:
    print("2. Sending request to Discord...")
    response = requests.post(
        url,
        data=json.dumps(payload),
        headers={"Content-Type": "application/json"}
    )

    if response.status_code == 204:
        print("✅ Success! Check your Discord channel.")
    else:
        print(f"❌ Failed. Status Code: {response.status_code}")
        print(f"Response: {response.text}")

except Exception as e:
    print(f"❌ Crash: {e}")
