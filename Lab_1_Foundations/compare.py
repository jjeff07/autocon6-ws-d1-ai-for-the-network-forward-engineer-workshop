"""
compare.py — Send the same prompt to both Ollama and OpenAI,
print responses side-by-side. Demonstrates LangChain's provider abstraction.
"""
import time
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

load_dotenv()

# Model strings live on one line each so they're easy to find and swap.
LOCAL_MODEL = "qwen3:4b-q4_K_M"
CLOUD_MODEL = "gpt-6-luna"

PROMPT = "In one sentence, explain why BGP uses TCP instead of UDP."


# def ask(llm):
#     """Send the prompt, time the call, return (response_text, elapsed_seconds)."""
#     start = time.time()
#     response = llm.invoke([HumanMessage(content=PROMPT)])
#     elapsed = time.time() - start
#     return response.content, elapsed

def ask(llm):
    """Stream the response chunk-by-chunk, return (full_text, elapsed_seconds)."""
    start = time.time()
    chunks = []
    for chunk in llm.stream([HumanMessage(content=PROMPT)]):
        print(chunk.content, end="", flush=True)   # print each chunk as it arrives
        chunks.append(chunk.content)
    print()                                         # newline once the stream ends
    elapsed = time.time() - start
    return "".join(chunks), elapsed



# Same interface, different providers. Both have their "reasoning" dial turned
# off here so we compare the models, not how long they spend thinking.
# (Lab 1 Step 6 turns the dial back on.)
ollama = ChatOllama(model=LOCAL_MODEL, base_url="http://localhost:11434", reasoning=False)
openai = ChatOpenAI(model=CLOUD_MODEL, reasoning_effort="none")

# print(f"Prompt: {PROMPT}\n")
# 
# for llm, label in [(ollama, f"Ollama / {LOCAL_MODEL}"),
#                    (openai, f"OpenAI / {CLOUD_MODEL}")]:
#     text, elapsed = ask(llm)
#     print(f"=== {label} ({elapsed:.2f}s) ===")
#     print(text)
#     print()

import statistics

N = 5

print(f"Prompt: {PROMPT}\n")

for llm, label in [(ollama, f"Ollama / {LOCAL_MODEL}"),
                   (openai, f"OpenAI / {CLOUD_MODEL}")]:
    times = []
    for i in range(N):
        text, elapsed = ask(llm)
        times.append(elapsed)
        print(f"  run {i+1}: {elapsed:.2f}s")
    median = statistics.median(times)
    print(f"=== {label} — median of {N} runs: {median:.2f}s ===")
    print(text)   # last response — eyeball quality
    print()
