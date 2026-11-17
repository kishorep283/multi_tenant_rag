from django.conf import settings
from openai import OpenAI


client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=settings.OPENROUTER_API_KEY,
    timeout=30.0,
)


from openai import APIError, APIConnectionError, APITimeoutError
import json

def generate_answer(question: str, context: str) -> str:
    prompt = f"""
    You are an HR policy assistant.

    Answer the user's question using only the information provided in the context.

    If the answer cannot be found in the context, say:
    "I could not find this information in the provided documents."

    Do not make up information.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    try:
        response = client.chat.completions.create(
            model=settings.OPENROUTER_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )

        answer = response.choices[0].message.content

        if not answer or not answer.strip():
            raise ValueError("LLM returned an empty response.")

        return answer.strip()

    except APITimeoutError as e:
        raise RuntimeError("LLM request timed out.") from e

    except APIConnectionError as e:
        raise RuntimeError("Unable to connect to the LLM provider.") from e

    except APIError as e:
        raise RuntimeError("LLM provider returned an API error.") from e

def reciprocal_rank_fusion(
    vector_results,
    keyword_results,
    k=60,
):
    results = {}

    for rank, result in enumerate(vector_results, start=1):
        chunk_id = result["id"]

        if chunk_id not in results:
            results[chunk_id] = {
                "id": chunk_id,
                "parent_id": result["parent_id"],
                "source_filename": result["source_filename"],
                "content": result["content"],
                "metadata": result["metadata"],
                "rrf_score": 0,
            }

        results[chunk_id]["rrf_score"] += 1 / (k + rank)

    for rank, result in enumerate(keyword_results, start=1):
        chunk_id = result["id"]

        if chunk_id not in results:
            results[chunk_id] = {
                "id": chunk_id,
                "source_filename": result["source_filename"],
                "content": result["content"],
                "metadata": result["metadata"],
                "rrf_score": 0,
            }

        results[chunk_id]["rrf_score"] += 1 / (k + rank)

    ranked_results = sorted(
        results.values(),
        key=lambda item: item["rrf_score"],
        reverse=True,
    )

    return ranked_results


def generate_queries(question: str) -> list[str]:
    """
    Generate alternative search queries using an LLM.
    Returns the original question along with three alternatives.
    """
    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    prompt = f"""
    You are a search query generation assistant.

    Generate exactly 3 alternative search queries for the
    user's question.

    Requirements:
    - Preserve the original meaning and intent.
    - Use different wording and search perspectives.
    - Do not answer the question.
    - Return ONLY a valid JSON object.
    - The object must contain one key named "queries".
    - "queries" must contain exactly 3 strings.
    - Do not include Markdown or explanations.

    Original question: {question}

    Example output:
    {{
        "queries": [
            "What is the annual casual leave entitlement?",
            "How many casual leave days are allocated per employee?",
            "What does the HR policy specify about casual leave?"
        ]
    }}
    """

    try:
        response = client.chat.completions.create(
            model=settings.OPENROUTER_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content

        if not content or not content.strip():
            raise ValueError("LLM returned an empty query response.")

        parsed_response = json.loads(content)

        # Expecting: {"queries": ["query 1", "query 2", "query 3"]}
        alternative_queries = parsed_response.get("queries")

        if (
            not isinstance(alternative_queries, list)
            or len(alternative_queries) != 3
            or not all(
                isinstance(query, str) and query.strip()
                for query in alternative_queries
            )
        ):
            raise ValueError("LLM returned an invalid query structure.")

        return [
            question.strip(),
            *[query.strip() for query in alternative_queries],
        ]

    except (APITimeoutError, APIConnectionError, APIError) as exc:
        raise RuntimeError("Query generation service is unavailable.") from exc
    except (json.JSONDecodeError, ValueError, AttributeError) as exc:
        raise RuntimeError("Failed to generate valid search queries.") from exc


def reciprocal_rank_fusion_multiple(rankings, k=60):
    """
    Combine multiple ranked result lists using RRF.
    Each ranking contributes independently to the final score.
    """
    results = {}

    for ranking in rankings:
        for rank, result in enumerate(ranking, start=1):
            chunk_id = result["id"]

            if chunk_id not in results:
                results[chunk_id] = {
                    **result,
                    "rrf_score": 0,
                }
            else:
                # Preserve fields available in other rankings
                for key, value in result.items():
                    if key not in results[chunk_id]:
                        results[chunk_id][key] = value

            results[chunk_id]["rrf_score"] += 1 / (k + rank)

    return sorted(
        results.values(),
        key=lambda item: item["rrf_score"],
        reverse=True,
    )
