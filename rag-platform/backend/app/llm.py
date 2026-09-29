import logging

from ollama import Client

from app.config import settings
from app.errors import OllamaUnavailableError


logger = logging.getLogger(__name__)
client = Client(
    host=settings.ollama_base_url,
    timeout=settings.ollama_timeout_seconds,
)
MODEL_NAME = settings.ollama_model


def generate_answer(
    question: str,
    context: str
) -> str:

    system_prompt = """
You are a question-answering assistant.

Answer the user's question using ONLY the provided context.

Rules:
- Do not invent information.
- If the answer cannot be found in the context, say that you do not know based on the provided documents.
- Keep the answer concise and clear.
"""

    user_prompt = f"""
CONTEXT:

{context}

QUESTION:

{question}
"""

    try:
        response = client.chat(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
        )
    except Exception as exc:
        logger.exception("Ollama answer generation failed")
        raise OllamaUnavailableError() from exc

    return response.message.content


def generate_answer_stream(
    question: str,
    context: str
):
    system_prompt = """
You are a question-answering assistant.

Answer the user's question using ONLY the provided context.

Rules:
- Do not invent information.
- If the answer cannot be found in the context, say that you do not know based on the provided documents.
- Keep the answer concise and clear.
"""

    user_prompt = f"""
CONTEXT:

{context}

QUESTION:

{question}
"""

    try:
        stream = client.chat(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            stream=True,
        )

        for chunk in stream:
            content = chunk.message.content
            if content:
                yield content
    except Exception as exc:
        logger.exception("Ollama streaming generation failed")
        raise OllamaUnavailableError() from exc


def ollama_available() -> bool:
    health_client = Client(
        host=settings.ollama_base_url,
        timeout=settings.health_timeout_seconds,
    )
    try:
        health_client.list()
        return True
    except Exception:
        logger.warning("Ollama health check failed", exc_info=True)
        return False
    finally:
        try:
            health_client.close()
        except Exception:
            pass


def close_ollama_client() -> None:
    try:
        client.close()
    except Exception:
        logger.warning("Failed to close Ollama client cleanly", exc_info=True)
