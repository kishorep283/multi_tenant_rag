from django.conf import settings
from openai import OpenAI


client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=settings.OPENROUTER_API_KEY,
    timeout=30.0,
)


from openai import APIError, APIConnectionError, APITimeoutError


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