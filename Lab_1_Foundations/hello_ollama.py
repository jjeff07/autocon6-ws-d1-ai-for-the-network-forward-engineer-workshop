"""hello_ollama.py — first call to the local LLM via Python."""
from ollama import Client

client = Client(host="http://localhost:11434")

response = client.chat(
    model="qwen3:4b-q4_K_M",
    messages=[
        {"role": "user", "content": "Explain OSPF in one sentence."}
    ],
    think=False,   # the same dial as --think=false on the CLI
)

print(response["message"]["content"])
