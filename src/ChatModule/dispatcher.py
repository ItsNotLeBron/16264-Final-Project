import os
import json
import time
import re
from dotenv import load_dotenv

load_dotenv()

from datetime import datetime, timedelta
import openai
from openai import OpenAI

# Initialize OpenAI client
openai.api_key = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI()

from MemoryModule.memory_module import MemoryModule
from InferenceModule.inference_module import InferenceModule

# Define system instruction to guide function calling
SYSTEM_MESSAGE = {
    "role": "system",
    "content": (
        "You are an AI assistant that helps track personal objects using specialized APIs. "
        "When the user asks about object location or history, use the provided function definitions to call the appropriate function. "
        "After calling the function, generate a clear natural-language response based on the function result."
    )
}

# Function definitions for ChatGPT function-calling
tools = [
    {
        "name": "get_history",
        "type": "function",
        "description": "List all recorded sightings for an object",
        "parameters": {
            "type": "object",
            "properties": {
                "label": { "type": "string", "description": "The object label (e.g., 'my_laptop')" },
                "since": { "type": "string", "format": "date-time", "description": "ISO timestamp to start from (inclusive)" },
                "until": { "type": "string", "format": "date-time", "description": "ISO timestamp to end at (inclusive)" }
            },
            "required": ["label"]
        }
    },
    {
        "name": "last_seen",
        "type": "function",
        "description": "Get the most recent sighting of an object",
        "parameters": {
            "type": "object",
            "properties": {
                "label": { "type": "string", "description": "The object label" }
            },
            "required": ["label"]
        }
    },
    {
        "name": "predict_location",
        "type": "function",
        "description": "Estimate where the object is likely to be at a given time",
        "parameters": {
            "type": "object",
            "properties": {
                "label": { "type": "string", "description": "The object label" },
                "at_time": { "type": "string", "format": "date-time", "description": "ISO timestamp for prediction (default now)" }
            },
            "required": ["label"]
        }
    },
    {
        "name": "explain_prediction",
        "type": "function",
        "description": "Explain how the location was predicted",
        "parameters": {
            "type": "object",
            "properties": {
                "label": { "type": "string" },
                "at_time": { "type": "string", "format": "date-time" }
            },
            "required": ["label"]
        }
    }
]


# Initialize modules
mem = MemoryModule()
inf = InferenceModule(mem)


def dispatch_function(name: str, args: dict):
    # Parse ISO date-time strings into datetime objects
    for key in ("at_time", "since", "until"):
        if key in args and args[key]:
            t = args[key]
            # Remove trailing Z for UTC
            if t.endswith('Z'):
                t = t[:-1]
            # Parse into datetime (handles fractional seconds)
            args[key] = datetime.fromisoformat(t)

    if name == "get_history":
        return inf.get_history(**args)
    if name == "last_seen":
        return inf.last_seen(**args)
    if name == "predict_location":
        return inf.predict_location(**args)
    if name == "explain_prediction":
        return inf.explain_prediction(**args)
    raise ValueError(f"Unknown function: {name}")


def safe_chat_call(**kwargs):
    """
    Call OpenAI chat completion with retries on rate limits.
    """
    max_retries = 3
    backoff = 1
    for attempt in range(max_retries):
        try:
            return client.responses.create(**kwargs)
        except openai.RateLimitError:
            if attempt < max_retries - 1:
                time.sleep(backoff)
                backoff *= 2
                continue
            else:
                raise
        except openai.OpenAIError:
            raise


def extract_label(prompt: str) -> str:
    """
    Heuristic to extract the object label from user prompt, e.g. 'my laptop'.
    """
    m = re.search(r"my (\w+)", prompt.lower())
    return m.group(1) if m else prompt.strip().split()[-1].rstrip('?.')


def chat_with_bot(user_prompt: str) -> str:
    """
    Send prompt to ChatGPT, handle function calls, fallback heuristics if needed.
    """
    try:
        # Initial request with system message and function definitions
        resp = safe_chat_call(
            model="gpt-4.1",
            input=user_prompt,
            tools=tools,
        )
    except openai.RateLimitError:
        return "Sorry, rate limit error. Please try again later."
    except openai.OpenAIError as e:
        return f"An API error occurred: {e}"

    # Debug: prettify raw response
    # print(json.dumps(resp.model_dump(), indent=2, default=str))

    # Check for function call in resp.output
    if resp.output and len(resp.output) > 0:
        call = resp.output[0]
        name = call.name
        args = json.loads(call.arguments)
        print(f"Function call: {name} with args: {args}")
        try:
            result = dispatch_function(name, args)
        except Exception as e:
            return f"Error executing function '{name}': {e}"
        return json.dumps(result, default=str)

    # Fallback: echo direct text if available
    return getattr(resp, 'text', '') or ''
