import os
import requests
from dotenv import load_dotenv

load_dotenv()

GROQ_KEY = os.getenv("GROQ_API_KEY")


def get_llm_response(prompt: str, model_name: str = "llama3-70b-8192") -> str:
    """
    Query Groq Llama models only.
    Always returns a string. Never returns None.
    """
    if not GROQ_KEY:
        return "⚠️ Groq API key missing."

    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {GROQ_KEY}"
        }

        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 200,
            "temperature": 0.3
        }

        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            json=payload,
            headers=headers,
            timeout=20
        )

        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"].strip()

        return f"⚠️ Groq API error: {response.status_code}, {response.text}"

    except Exception as e:
        return f"⚠️ Groq request failed: {e}"
