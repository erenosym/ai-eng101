from ollama import chat


MODEL_NAME = "qwen2.5:7b"


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

    response = chat(
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

    stream = chat(
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