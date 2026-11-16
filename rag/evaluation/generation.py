import json

from django.conf import settings
from openai import OpenAI


client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=settings.OPENROUTER_API_KEY,
)


def evaluate_correctness(question, expected_answer, generated_answer):

    prompt = f"""
    You are an evaluator for a RAG application.

    Evaluate whether the generated answer is factually correct
    compared with the expected answer.

    Question:
    {question}

    Expected Answer:
    {expected_answer}

    Generated Answer:
    {generated_answer}

    Different wording is acceptable if the meaning is equivalent.

    Return only valid JSON in this format:
    {{
        "score": 1 or 0,
        "reason": "Brief explanation"
    }}

    Score 1 if correct, otherwise 0.
    """

    response = client.chat.completions.create(
        model=settings.OPENROUTER_MODEL,
        messages=[
            {"role": "user", "content": prompt}
        ],
        response_format={"type": "json_object"},
    )

    return json.loads(response.choices[0].message.content)

def evaluate_faithfulness(question, context, generated_answer):

    prompt = f"""
    You are an evaluator for a RAG application.

    Evaluate whether the generated answer is supported by
    the information provided in the context.

    Question:
    {question}

    Context:
    {context}

    Generated Answer:
    {generated_answer}

    Instructions:
    - Check every factual claim in the generated answer.
    - Every factual claim must be supported by the context.
    - If the answer contains unsupported or contradictory claims,
      assign score 0.
    - If all factual claims are supported, assign score 1.

    Return only valid JSON:
    {{
        "score": 1 or 0,
        "reason": "Brief explanation"
    }}
    """

    response = client.chat.completions.create(
        model=settings.OPENROUTER_MODEL,
        messages=[
            {"role": "user", "content": prompt}
        ],
        response_format={"type": "json_object"},
    )

    return json.loads(response.choices[0].message.content)

def evaluate_relevance(question, generated_answer):
    prompt = f"""
    You are an evaluator for a RAG application.

    Evaluate whether the generated answer directly addresses
    the user's question.

    Question:
    {question}

    Generated Answer:
    {generated_answer}

    Instructions:
    - Score 1 if the answer directly addresses the question.
    - Score 0 if the answer is irrelevant or does not answer the question.
    - Do not evaluate factual correctness or faithfulness.

    Return only valid JSON:
    {{
        "score": 1 or 0,
        "reason": "Brief explanation"
    }}
    """

    response = client.chat.completions.create(
        model=settings.OPENROUTER_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )

    return json.loads(response.choices[0].message.content)