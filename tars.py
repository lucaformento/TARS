import os
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=200,
    system="You are TARS, the robot from Interstellar. Dry humor, brutally honest, loyal. Keep replies short.",
    messages=[
        {"role": "user", "content": "TARS, are you online?"}
    ]
)

print(response.content[0].text)
