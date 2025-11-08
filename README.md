##LLM Automation Testing

A lightweight automated testing framework for evaluating LLM behavior, consistency, safety, and reliability.

This project helps QA engineers systematically audit LLM responses using real test cases, paraphrasing tests, keyword validation, latency checks, and safety/hallucination detection — all fully automated with PyTest. Currently, the framework uses Groq LLaMA3 as the primary LLM for testing.

✅ Features
1. Automated LLM Response Testing

Validates expected keywords in responses

Measures latency for each request

Grades response quality

Detects hallucinations using ground truth

Warns on prompt injection behavior

2. Paraphrase Robustness Testing

Ensures LLM remains consistent when prompts are rephrased or modified

3. Multi-Model Support (Planned)

Groq LLaMA3-70B (currently in use)

OpenAI models can be added if needed

⚡ Getting Started

Clone the repo:

git clone <your-repo-url>
cd llm-automation-testing


Install dependencies:

pip install -r requirements.txt


Add your Groq API key to a .env file:

GROQ_API_KEY=your_api_key_here


Run tests:

pytest --maxfail=1 --disable-warnings -q
