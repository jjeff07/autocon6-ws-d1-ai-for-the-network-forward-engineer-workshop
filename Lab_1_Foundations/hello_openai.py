"""hello_openai.py — first call to OpenAI's gpt-6-luna via Python."""
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()  # reads .env, sets OPENAI_API_KEY in environment
client = OpenAI()

response = client.chat.completions.create(
    model="gpt-6-luna",
    messages=[
        {"role": "user", "content": "Explain OSPF in one sentence."}
    ],
)

print(response.choices[0].message.content)

usage = response.usage
print(f"\nTokens: prompt={usage.prompt_tokens}, "
      f"completion={usage.completion_tokens} "
      f"(of which reasoning={usage.completion_tokens_details.reasoning_tokens})")
