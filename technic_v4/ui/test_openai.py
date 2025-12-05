from openai import OpenAI
from config_secrets import OPENAI_API_KEY

# Strip any stray whitespace/newlines just in case
clean_key = OPENAI_API_KEY.strip()

print("Using key prefix:", clean_key[:10], "...", clean_key[-6:])

client = OpenAI(api_key=clean_key)

resp = client.chat.completions.create(
    model="gpt-4.1-mini",
    messages=[{"role": "user", "content": "Reply with 'OK' if this works."}],
)

print("Response from OpenAI:", resp.choices[0].message.content)