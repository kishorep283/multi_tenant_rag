import json

from rag.services.llm import client
from django.conf import settings


def compress_documents(question: str, parent_documents: list[dict]) -> list[dict]:
    """
    Extract query-relevant information from parent documents.
    """

    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    if not parent_documents:
        return []

    compressed_documents = []

    for document in parent_documents:

        prompt = f"""
        You are a document compression assistant.

        Your task is to extract only the information relevant
        to answering the user's question.

        Rules:
        - Preserve the original meaning and factual details.
        - Do not add new information.
        - Do not answer the user's question.
        - Return only relevant sentences or passages.
        - If nothing is relevant, return an empty string.

        User question:
        {question}

        Document content:
        {document["content"]}

        Return ONLY a valid JSON object in this format:
        {{
            "compressed_content": "Relevant extracted text"
        }}
        """

        try:
            response = client.chat.completions.create(
                model=settings.OPENROUTER_MODEL,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )

            content = response.choices[0].message.content

            if not content or not content.strip():
                raise ValueError("LLM returned empty compression response.")

            parsed_response = json.loads(content)
            compressed_content = parsed_response.get("compressed_content")

            if not isinstance(compressed_content, str):
                raise ValueError("Invalid compression response structure.")

            if compressed_content.strip():
                compressed_documents.append({
                    **document,
                    "content": compressed_content.strip(),
                })

        except Exception as exc:
            raise RuntimeError("Document compression failed.") from exc

    return compressed_documents