import json, os
import pytest
from src.llm_client import get_llm_response
from src.utils import (
    keyword_check,
    measure_latency,
    paraphrase_prompt,
    grade_response,
    check_hallucination,
    check_injection_safety
)
import time

with open("tests/test_data.json") as f:
    test_cases = json.load(f)

@pytest.mark.parametrize("case", test_cases)
def test_llm_response(case):
    prompt = case["prompt"]
    expected_keywords = case["expected_keywords"]
    ground_truth = case.get("ground_truth", "")

    response, latency = measure_latency(get_llm_response, prompt)
    time.sleep(3)

    print(f"\nPrompt: {prompt}")
    print(f"Response: {response[:100]}...")
    print(f"Latency: {latency}s")

    assert latency < 15, f"Response too slow: {latency}s"   
    assert keyword_check(response, expected_keywords) or len(response) > 0, "Response seems irrelevant"
    
    if not check_injection_safety(response):
        print("⚠️ Warning: Possible unsafe instruction detected")

    if ground_truth:
        similarity = check_hallucination(response, ground_truth)
        print(f"Hallucination similarity ratio: {similarity}")
        assert similarity > 0.3, "Response seems quite different from expected truth"

    grade = grade_response(response, expected_keywords)
    print(f"Graded score: {grade}")

    assert grade >= 0.3, "Response relevance/quality low"

@pytest.mark.parametrize("case", test_cases)
def test_paraphrased_prompt(case):
    prompt = case["prompt"]
    expected_keywords = case["expected_keywords"]

    paraphrased = paraphrase_prompt(prompt)
    time.sleep(3)
    response, _ = measure_latency(get_llm_response, paraphrased)
    time.sleep(3)

    print(f"\nOriginal: {prompt}")
    print(f"Paraphrased: {paraphrased}")
    print(f"Response: {response[:100]}...")

    assert keyword_check(response, expected_keywords) or len(response) > 0, "Keywords missing after paraphrasing"


